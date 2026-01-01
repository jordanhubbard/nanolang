#include "nanolang.h"
#include "version.h"
#include "interpreter_ffi.h"
#include "module_builder.h"

#ifdef __APPLE__
#include <mach-o/loader.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* Global argc/argv for runtime access by transpiled programs */
int g_argc = 0;
char **g_argv = NULL;

/* Compilation options */
typedef struct {
    bool verbose;
    bool keep_c;
    bool save_asm;            /* -S flag: save generated C to .genC file */
    char **include_paths;      /* -I flags */
    int include_count;
    char **library_paths;     /* -L flags */
    int library_path_count;
    char **libraries;         /* -l flags */
    int library_count;
} CompilerOptions;

static bool deterministic_outputs_enabled(void) {
    const char *v = getenv("NANO_DETERMINISTIC");
    return v && (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "yes") == 0);
}

#ifdef __APPLE__
static int determinize_macho_uuid_and_signature(const char *path) {
    /* On modern macOS, Mach-O binaries are ad-hoc signed and include a randomized LC_UUID.
     * For deterministic bootstrap verification we:
     *  1) overwrite the LC_UUID bytes with a fixed value
     *  2) re-sign with a fixed identifier so the LC_CODE_SIGNATURE blob is deterministic
     */
    int fd = open(path, O_RDWR);
    if (fd < 0) return -1;

    struct mach_header_64 hdr;
    ssize_t n = pread(fd, &hdr, sizeof(hdr), 0);
    if (n != (ssize_t)sizeof(hdr) || hdr.magic != MH_MAGIC_64) {
        close(fd);
        return -1;
    }

    off_t off = (off_t)sizeof(hdr);
    for (uint32_t i = 0; i < hdr.ncmds; i++) {
        struct load_command lc;
        if (pread(fd, &lc, sizeof(lc), off) != (ssize_t)sizeof(lc) || lc.cmdsize < sizeof(lc)) {
            close(fd);
            return -1;
        }

        if (lc.cmd == LC_UUID) {
            struct uuid_command uc;
            if (lc.cmdsize < sizeof(uc) || pread(fd, &uc, sizeof(uc), off) != (ssize_t)sizeof(uc)) {
                close(fd);
                return -1;
            }

            static const uint8_t fixed_uuid[16] = {
                0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
            };
            memcpy(uc.uuid, fixed_uuid, sizeof(fixed_uuid));

            if (pwrite(fd, &uc, sizeof(uc), off) != (ssize_t)sizeof(uc)) {
                close(fd);
                return -1;
            }
            break;
        }

        off += (off_t)lc.cmdsize;
    }

    close(fd);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "codesign -s - --force -i nanolang.deterministic '%s' >/dev/null 2>&1", path);
    return system(cmd);
}
#endif

/* Compile nanolang source to executable */
static int compile_file(const char *input_file, const char *output_file, CompilerOptions *opts) {
    /* Read source file */
    FILE *file = fopen(input_file, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", input_file);
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *source = malloc(size + 1);
    fread(source, 1, size, file);
    source[size] = '\0';
    fclose(file);

    if (opts->verbose) printf("Compiling %s...\n", input_file);

    /* Phase 1: Lexing */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "Lexing failed\n");
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Lexing complete (%d tokens)\n", token_count);

    /* Phase 2: Parsing */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "Parsing failed\n");
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Parsing complete\n");

    /* Phase 3: Create environment and process imports */
    clear_module_cache();  /* Clear cache from any previous compilation */
    Environment *env = create_environment();
    ModuleList *modules = create_module_list();
    if (!process_imports(program, env, modules, input_file)) {
        fprintf(stderr, "Module loading failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        return 1;
    }
    if (opts->verbose && modules->count > 0) {
        printf("✓ Loaded %d module(s)\n", modules->count);
    }

    /* Compile modules early so extern C functions are available for shadow tests (via FFI). */
    char module_objs[2048] = "";
    char module_compile_flags[2048] = "";

    /* Phase 4: Type Checking */
    if (!type_check(program, env)) {
        fprintf(stderr, "Type checking failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Type checking complete\n");

    /* Phase 4.5: Build imported modules (object + shared libs) */
    if (modules->count > 0) {
        if (!compile_modules(modules, env, module_objs, sizeof(module_objs),
                             module_compile_flags, sizeof(module_compile_flags),
                             opts->verbose)) {
            fprintf(stderr, "Error: Failed to compile modules\n");
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free_module_list(modules);
            free(source);
            return 1;
        }
    }

    /* Phase 4.6: Initialize FFI and load module shared libraries for shadow tests */
    (void)ffi_init(opts->verbose);
    for (int i = 0; i < modules->count; i++) {
        const char *module_path = modules->module_paths[i];

        char *module_dir = strdup(module_path);
        char *last_slash = strrchr(module_dir, '/');
        if (last_slash) {
            *last_slash = '\0';
        } else {
            free(module_dir);
            module_dir = strdup(".");
        }

        ModuleBuildMetadata *meta = module_load_metadata(module_dir);

        char mod_name[256];
        if (meta && meta->name) {
            snprintf(mod_name, sizeof(mod_name), "%s", meta->name);
        } else {
            const char *base_name = last_slash ? last_slash + 1 : module_path;
            snprintf(mod_name, sizeof(mod_name), "%s", base_name);
            char *dot = strrchr(mod_name, '.');
            if (dot) *dot = '\0';
        }

        (void)ffi_load_module(mod_name, module_path, env, opts->verbose);

        if (meta) module_metadata_free(meta);
        free(module_dir);
    }

    /* Phase 5: Shadow-Test Execution */
    if (!run_shadow_tests(program, env)) {
        fprintf(stderr, "Shadow tests failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        ffi_cleanup();
        return 1;
    }
    if (opts->verbose) printf("✓ Shadow tests passed\n");

    /* Phase 6: C Transpilation */
    if (opts->verbose) printf("Transpiling to C...\n");
    char *c_code = transpile_to_c(program, env, input_file);
    if (!c_code) {
        fprintf(stderr, "Transpilation failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        return 1;
    }
    
    /* Calculate C code size */
    size_t c_code_size = strlen(c_code);
    int c_code_lines = 1;
    for (size_t i = 0; i < c_code_size; i++) {
        if (c_code[i] == '\n') c_code_lines++;
    }
    
    if (opts->verbose) {
        printf("✓ Transpilation complete (%d lines of C, %zu bytes)\n", c_code_lines, c_code_size);
    }
    
    /* Save generated C to .genC file if -S flag is set */
    if (opts->save_asm) {
        char gen_c_file[512];
        snprintf(gen_c_file, sizeof(gen_c_file), "%s.genC", input_file);
        FILE *gen_c = fopen(gen_c_file, "w");
        if (gen_c) {
            fprintf(gen_c, "%s", c_code);
            fclose(gen_c);
            if (opts->verbose) {
                printf("✓ Saved generated C to: %s\n", gen_c_file);
            }
        } else {
            fprintf(stderr, "Warning: Could not save generated C to %s\n", gen_c_file);
        }
    }

    /* Write C code to temporary file in /tmp (or keep in output dir if --keep-c) */
    char temp_c_file[512];
    if (opts->keep_c) {
        /* Keep in output directory if --keep-c is set */
        snprintf(temp_c_file, sizeof(temp_c_file), "%s.c", output_file);
    } else {
        /* Use /tmp for temporary files */
        snprintf(temp_c_file, sizeof(temp_c_file), "/tmp/nanoc_%d_%s.c", 
                 (int)getpid(), strrchr(output_file, '/') ? strrchr(output_file, '/') + 1 : output_file);
    }

    FILE *c_file = fopen(temp_c_file, "w");
    if (!c_file) {
        fprintf(stderr, "Error: Could not create C file '%s'\n", temp_c_file);
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        return 1;
    }

    fprintf(c_file, "%s", c_code);
    fclose(c_file);
    if (opts->verbose) printf("✓ Generated C code: %s\n", temp_c_file);

    char compile_cmd[16384];  /* Increased to handle long command lines with many modules */
    
    /* Build include flags */
    char include_flags[2048] = "-Isrc";
    for (int i = 0; i < opts->include_count; i++) {
        char temp[512];
        snprintf(temp, sizeof(temp), " -I%s", opts->include_paths[i]);
        strncat(include_flags, temp, sizeof(include_flags) - strlen(include_flags) - 1);
    }
    
    /* Add module compile flags (include paths from pkg-config) */
    if (module_compile_flags[0] != '\0') {
        strncat(include_flags, " ", sizeof(include_flags) - strlen(include_flags) - 1);
        strncat(include_flags, module_compile_flags, sizeof(include_flags) - strlen(include_flags) - 1);
    }
    
    /* Build library path flags */
    char lib_path_flags[2048] = "";
    for (int i = 0; i < opts->library_path_count; i++) {
        char temp[512];
        snprintf(temp, sizeof(temp), " -L%s", opts->library_paths[i]);
        strncat(lib_path_flags, temp, sizeof(lib_path_flags) - strlen(lib_path_flags) - 1);
    }
    
    /* Build library flags */
    char lib_flags[2048] = "-lm";
    for (int i = 0; i < opts->library_count; i++) {
        char temp[512];
        snprintf(temp, sizeof(temp), " -l%s", opts->libraries[i]);
        strncat(lib_flags, temp, sizeof(lib_flags) - strlen(lib_flags) - 1);
    }
    
    /* Detect and generate generic list types from the C code AND compiler_schema.h */
    char generated_lists[1024] = "";
    char detected_types[64][64]; /* Increased to handle more types */
    int detected_count = 0;
    
    /* First, scan compiler_schema.h if it exists */
    FILE *schema_h = fopen("src/generated/compiler_schema.h", "r");
    if (schema_h) {
        fseek(schema_h, 0, SEEK_END);
        long size = ftell(schema_h);
        fseek(schema_h, 0, SEEK_SET);
        char *schema_content = malloc(size + 1);
        if (schema_content) {
            fread(schema_content, 1, size, schema_h);
            schema_content[size] = '\0';
            
            const char *ptr = schema_content;
            while ((ptr = strstr(ptr, "List_")) != NULL) {
                ptr += 5;
                const char *end = ptr;
                while ((*end >= 'A' && *end <= 'Z') || (*end >= 'a' && *end <= 'z') || (*end >= '0' && *end <= '9') || *end == '_') {
                    end++;
                }
                if (*end == '*' || *end == ' ' || *end == '\n' || *end == ';') {
                    int len = end - ptr;
                    char type_name[64];
                    strncpy(type_name, ptr, len);
                    type_name[len] = '\0';
                    
                    if (strcmp(type_name, "int") != 0 && strcmp(type_name, "string") != 0 && strcmp(type_name, "token") != 0 && strcmp(type_name, "Generic") != 0) {
                        bool found = false;
                        for (int i = 0; i < detected_count; i++) {
                            if (strcmp(detected_types[i], type_name) == 0) {
                                found = true;
                                break;
                            }
                        }
                        if (!found && detected_count < 64) {
                            strcpy(detected_types[detected_count++], type_name);
                        }
                    }
                }
            }
            free(schema_content);
        }
        fclose(schema_h);
    }

    const char *scan_ptr = c_code;
    /* Scan for List_TypeName* patterns in generated code too */
    while ((scan_ptr = strstr(scan_ptr, "List_")) != NULL) {
        scan_ptr += 5; /* Skip "List_" */
        const char *end_ptr = scan_ptr;
        
        /* Extract type name (alphanumeric + underscore) */
        while ((*end_ptr >= 'A' && *end_ptr <= 'Z') || 
               (*end_ptr >= 'a' && *end_ptr <= 'z') || 
               (*end_ptr >= '0' && *end_ptr <= '9') || 
               *end_ptr == '_') {
            end_ptr++;
        }
        
        /* Check if followed by * or space (valid list type) */
        if (*end_ptr == '*' || *end_ptr == ' ' || *end_ptr == '\n') {
            int len = end_ptr - scan_ptr;
            char type_name[64];
            strncpy(type_name, scan_ptr, len);
            type_name[len] = '\0';
            
            /* Skip built-in types */
            if (strcmp(type_name, "int") == 0 || 
                strcmp(type_name, "string") == 0 || 
                strcmp(type_name, "token") == 0) {
                continue;
            }
            
            /* Check if already detected */
            bool already_detected = false;
            for (int i = 0; i < detected_count; i++) {
                if (strcmp(detected_types[i], type_name) == 0) {
                    already_detected = true;
                    break;
                }
            }
            
            if (!already_detected && detected_count < 32) {
                strncpy(detected_types[detected_count], type_name, 63);
                detected_types[detected_count][63] = '\0';
                detected_count++;
                
                /* Generate list runtime files */
                const char *c_type = type_name;
                if (strcmp(type_name, "LexerToken") == 0) c_type = "Token";
                else if (strcmp(type_name, "NSType") == 0) c_type = "NSType";
                else if (strncmp(type_name, "AST", 3) == 0 || strncmp(type_name, "Compiler", 8) == 0) {
                    /* For schema types, use the typedef name.
                     * We'll ensure compiler_schema.h is included. */
                    c_type = type_name;
                }
                
                char gen_cmd[512];
                snprintf(gen_cmd, sizeof(gen_cmd), 
                        "./scripts/generate_list.sh %s /tmp %s > /dev/null 2>&1", 
                        type_name, c_type);
                if (opts->verbose) {
                    printf("Generating List<%s> runtime...\n", type_name);
                }
                int gen_result = system(gen_cmd);
                if (gen_result != 0 && opts->verbose) {
                    fprintf(stderr, "Warning: Failed to generate list_%s runtime\n", type_name);
                }
                
                /* Create wrapper that includes struct definition */
                char wrapper_file[512];
                snprintf(wrapper_file, sizeof(wrapper_file), "/tmp/list_%s_wrapper.c", type_name);
                FILE *wrapper = fopen(wrapper_file, "w");
                if (wrapper) {
                    /* Extract struct definition from generated C code */
                    const char *struct_search = c_code;
                    char struct_pattern[128];
                    snprintf(struct_pattern, sizeof(struct_pattern), "typedef struct nl_%s {", type_name);
                    const char *struct_start = strstr(struct_search, struct_pattern);
                    const char *struct_start_original = struct_start;
                    
                    /* Try to find guards if they exist */
                    char guard_pattern[128];
                    snprintf(guard_pattern, sizeof(guard_pattern), "#ifndef DEFINED_nl_%s", type_name);
                    const char *guard_start = strstr(struct_search, guard_pattern);
                    if (guard_start && guard_start < struct_start) {
                        struct_start = guard_start;
                    }
                    
                    if (struct_start) {
                        /* Find the end of the struct (closing brace + semicolon) */
                        const char *struct_end = struct_start;
                        int brace_count = 0;
                        bool found_open_brace = false;
                        while (*struct_end) {
                            if (*struct_end == '{') {
                                found_open_brace = true;
                                brace_count++;
                            } else if (*struct_end == '}' && found_open_brace) {
                                brace_count--;
                                if (brace_count == 0) {
                                    /* Found the closing brace, look for semicolon */
                                    struct_end++;
                                    while (*struct_end && *struct_end != ';') struct_end++;
                                    if (*struct_end == ';') struct_end++;
                                    
                                    /* Look for #endif if we started with a guard */
                                    if (guard_start && guard_start < struct_start_original) {
                                        const char *endif_search = strstr(struct_end, "#endif");
                                        if (endif_search) {
                                            struct_end = endif_search + 6;
                                        }
                                    }
                                    break;
                                }
                            }
                            struct_end++;
                        }
                        
                        /* Write the wrapper */
                        fprintf(wrapper, "#include <stdint.h>\n");
                        fprintf(wrapper, "#include <stdbool.h>\n");
                        fprintf(wrapper, "#include <stdlib.h>\n");
                        fprintf(wrapper, "#include <stdio.h>\n");
                        fprintf(wrapper, "#include <string.h>\n\n");
                        /* Needed for DynArray/nl_string_t/Token used by extracted structs */
                        fprintf(wrapper, "#include \"nanolang.h\"\n");
                        fprintf(wrapper, "#include \"generated/compiler_schema.h\"\n\n");
                        fprintf(wrapper, "/* Struct definition extracted from main file */\n");
                        fprintf(wrapper, "%.*s\n", (int)(struct_end - struct_start), struct_start);
                        /* Set guard macro to prevent typedef redefinition in list header */
                        char type_upper[128];
                        strncpy(type_upper, type_name, sizeof(type_upper) - 1);
                        type_upper[sizeof(type_upper) - 1] = '\0';
                        for (char *p = type_upper; *p; p++) {
                            *p = (char)toupper((unsigned char)*p);
                        }
                        fprintf(wrapper, "\n/* Guard macro set - typedef already defined above */\n");
                        fprintf(wrapper, "#define NL_%s_DEFINED\n\n", type_upper);
                        fprintf(wrapper, "/* Include list implementation */\n");
                        fprintf(wrapper, "#include \"/tmp/list_%s.c\"\n", type_name);
                    } else {
                        /* Fallback: just include the list file.
                         * We include schema headers in case it's a schema type. */
                        fprintf(wrapper, "#include <stdint.h>\n");
                        fprintf(wrapper, "#include <stdbool.h>\n");
                        fprintf(wrapper, "#include <stdlib.h>\n");
                        fprintf(wrapper, "#include <stdio.h>\n");
                        fprintf(wrapper, "#include <string.h>\n\n");
                        fprintf(wrapper, "#include \"nanolang.h\"\n");
                        fprintf(wrapper, "#include \"generated/compiler_schema.h\"\n\n");
                        fprintf(wrapper, "#include \"/tmp/list_%s.c\"\n", type_name);
                    }
                    fclose(wrapper);
                }
                
                /* Add wrapper to compile list */
                char list_file[256];
                snprintf(list_file, sizeof(list_file), " /tmp/list_%s_wrapper.c", type_name);
                strncat(generated_lists, list_file, sizeof(generated_lists) - strlen(generated_lists) - 1);
            }
        }
    }
    
    if (opts->verbose && detected_count > 0) {
        printf("Detected %d generic list type(s): ", detected_count);
        for (int i = 0; i < detected_count; i++) {
            printf("%s%s", detected_types[i], i < detected_count - 1 ? ", " : "");
        }
        printf("\n");
    }
    
    /* Build runtime files string */
    /* Note: sdl_helpers.c is NOT included here - it's provided by the sdl_helpers module */
    char runtime_files[4096] = "src/runtime/list_int.c src/runtime/list_string.c src/runtime/list_LexerToken.c src/runtime/list_token.c src/runtime/list_CompilerDiagnostic.c src/runtime/list_CompilerSourceLocation.c src/runtime/list_ASTNumber.c src/runtime/list_ASTFloat.c src/runtime/list_ASTString.c src/runtime/list_ASTBool.c src/runtime/list_ASTIdentifier.c src/runtime/list_ASTBinaryOp.c src/runtime/list_ASTCall.c src/runtime/list_ASTArrayLiteral.c src/runtime/list_ASTLet.c src/runtime/list_ASTSet.c src/runtime/list_ASTStmtRef.c src/runtime/list_ASTIf.c src/runtime/list_ASTWhile.c src/runtime/list_ASTFor.c src/runtime/list_ASTReturn.c src/runtime/list_ASTBlock.c src/runtime/list_ASTUnsafeBlock.c src/runtime/list_ASTPrint.c src/runtime/list_ASTAssert.c src/runtime/list_ASTFunction.c src/runtime/list_ASTShadow.c src/runtime/list_ASTStruct.c src/runtime/list_ASTStructLiteral.c src/runtime/list_ASTFieldAccess.c src/runtime/list_ASTEnum.c src/runtime/list_ASTUnion.c src/runtime/list_ASTUnionConstruct.c src/runtime/list_ASTMatch.c src/runtime/list_ASTImport.c src/runtime/list_ASTOpaqueType.c src/runtime/list_ASTTupleLiteral.c src/runtime/list_ASTTupleIndex.c src/runtime/token_helpers.c src/runtime/gc.c src/runtime/dyn_array.c src/runtime/gc_struct.c src/runtime/nl_string.c src/runtime/cli.c";
    strncat(runtime_files, generated_lists, sizeof(runtime_files) - strlen(runtime_files) - 1);
    
    /* Add /tmp to include path for generated list headers */
    char include_flags_with_tmp[2560];
    snprintf(include_flags_with_tmp, sizeof(include_flags_with_tmp), "%s -I/tmp", include_flags);
    
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = "cc";

    int cmd_len = snprintf(compile_cmd, sizeof(compile_cmd),
            "%s -std=c99 %s -o %s %s %s %s %s %s",
            cc, include_flags_with_tmp, output_file, temp_c_file, module_objs, runtime_files, lib_path_flags, lib_flags);
    
    if (cmd_len >= (int)sizeof(compile_cmd)) {
        fprintf(stderr, "Error: Compile command too long (%d bytes, max %zu)\n", cmd_len, sizeof(compile_cmd));
        fprintf(stderr, "Try reducing the number of modules or shortening paths.\n");
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        if (!opts->keep_c) {
            remove(temp_c_file);
        }
        return 1;
    }

    if (opts->verbose) printf("Compiling C code: %s\n", compile_cmd);
    int result = system(compile_cmd);

    if (result == 0) {
        if (deterministic_outputs_enabled()) {
#ifdef __APPLE__
            (void)determinize_macho_uuid_and_signature(output_file);
#endif
        }
        if (opts->verbose) printf("✓ Compilation successful: %s\n", output_file);
    } else {
        fprintf(stderr, "C compilation failed\n");
        /* Cleanup */
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        /* Remove temporary C file unless --keep-c */
        if (!opts->keep_c) {
            remove(temp_c_file);
        }
        return 1;  /* Return error if C compilation failed */
    }

    /* Remove temporary C file unless --keep-c (cleanup on both success and failure) */
    if (!opts->keep_c) {
        remove(temp_c_file);
    }

    /* Cleanup */
    free(c_code);
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free_module_list(modules);
    free(source);

    return 0;  /* Success */
}

/* Main entry point */
int main(int argc, char *argv[]) {
    /* Store argc/argv for runtime access */
    g_argc = argc;
    g_argv = argv;
    
    /* Handle --version */
    if (argc >= 2 && (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0)) {
        printf("nanoc %s\n", NANOLANG_VERSION);
        printf("nanolang compiler\n");
        printf("Built: %s %s\n", NANOLANG_BUILD_DATE, NANOLANG_BUILD_TIME);
        return 0;
    }

    /* Handle --help */
    if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("nanoc - Compiler for the nanolang programming language\n\n");
        printf("Usage: %s <input.nano> [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("  -o <file>      Specify output file (default: /tmp/nanoc_a.out)\n");
        printf("  --verbose      Show detailed compilation steps and commands\n");
        printf("  --keep-c       Keep generated C file (saves to output dir instead of /tmp)\n");
        printf("  -S             Save generated C to <input>.genC (for inspection)\n");
        printf("  -I <path>      Add include path for C compilation\n");
        printf("  -L <path>      Add library path for C linking\n");
        printf("  -l <lib>       Link against library (e.g., -lSDL2)\n");
        printf("  --version, -v  Show version information\n");
        printf("  --help, -h     Show this help message\n");
        printf("\nExamples:\n");
        printf("  %s hello.nano -o hello\n", argv[0]);
        printf("  %s program.nano --verbose -S          # Show steps and save C code\n", argv[0]);
        printf("  %s example.nano -o example --verbose\n", argv[0]);
        printf("  %s sdl_app.nano -o app -I/opt/homebrew/include/SDL2 -L/opt/homebrew/lib -lSDL2\n\n", argv[0]);
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.nano> [OPTIONS]\n", argv[0]);
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = "/tmp/nanoc_a.out";  /* Default to /tmp to avoid polluting project dir */
    CompilerOptions opts = {
        .verbose = false,
        .keep_c = false,
        .save_asm = false,
        .include_paths = NULL,
        .include_count = 0,
        .library_paths = NULL,
        .library_path_count = 0,
        .libraries = NULL,
        .library_count = 0
    };
    
    /* Allocate arrays for flags */
    char **include_paths = malloc(sizeof(char*) * 32);
    char **library_paths = malloc(sizeof(char*) * 32);
    char **libraries = malloc(sizeof(char*) * 32);
    int include_count = 0;
    int library_path_count = 0;
    int library_count = 0;

    /* Parse command-line options */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "--keep-c") == 0) {
            opts.keep_c = true;
        } else if (strcmp(argv[i], "-S") == 0) {
            opts.save_asm = true;
        } else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            if (include_count < 32) {
                include_paths[include_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-I", 2) == 0) {
            /* Handle -I/path form */
            if (include_count < 32) {
                include_paths[include_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "-L") == 0 && i + 1 < argc) {
            if (library_path_count < 32) {
                library_paths[library_path_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-L", 2) == 0) {
            /* Handle -L/path form */
            if (library_path_count < 32) {
                library_paths[library_path_count++] = argv[i] + 2;
            }
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            if (library_count < 32) {
                libraries[library_count++] = argv[i + 1];
            }
            i++;
        } else if (strncmp(argv[i], "-l", 2) == 0) {
            /* Handle -llibname form */
            if (library_count < 32) {
                libraries[library_count++] = argv[i] + 2;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            free(include_paths);
            free(library_paths);
            free(libraries);
            return 1;
        }
    }
    
    /* Set parsed flags in options */
    opts.include_paths = include_paths;
    opts.include_count = include_count;
    opts.library_paths = library_paths;
    opts.library_path_count = library_path_count;
    opts.libraries = libraries;
    opts.library_count = library_count;
    
    /* Check for NANO_VERBOSE_BUILD environment variable */
    if (getenv("NANO_VERBOSE_BUILD")) {
        opts.verbose = true;
    }
    
    int result = compile_file(input_file, output_file, &opts);
    
    /* Cleanup */
    free(include_paths);
    free(library_paths);
    free(libraries);
    
    return result;
}