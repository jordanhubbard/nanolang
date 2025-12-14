#include "nanolang.h"
#include "version.h"

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

    /* Phase 5: Shadow-Test Execution */
    if (!run_shadow_tests(program, env)) {
        fprintf(stderr, "Shadow tests failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Shadow tests passed\n");

    /* Phase 6: C Transpilation */
    if (opts->verbose) printf("Transpiling to C...\n");
    char *c_code = transpile_to_c(program, env);
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

    /* Compile modules to object files */
    char compile_cmd[16384];  /* Increased to handle long command lines with many modules */
    char module_objs[2048] = "";
    char module_compile_flags[2048] = "";
    
    if (modules->count > 0) {
        if (!compile_modules(modules, env, module_objs, sizeof(module_objs), module_compile_flags, sizeof(module_compile_flags), opts->verbose)) {
            fprintf(stderr, "Error: Failed to compile modules\n");
            free(c_code);
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free_module_list(modules);
            free(source);
            return 1;
        }
    }
    
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
    
    /* Detect and generate generic list types from the C code */
    char generated_lists[1024] = "";
    const char *scan_ptr = c_code;
    char detected_types[32][64]; /* Track up to 32 unique list types */
    int detected_count = 0;
    
    /* Scan for List_TypeName* patterns */
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
                char gen_cmd[512];
                snprintf(gen_cmd, sizeof(gen_cmd), 
                        "./scripts/generate_list.sh %s /tmp > /dev/null 2>&1", 
                        type_name);
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
                        fprintf(wrapper, "/* Struct definition extracted from main file */\n");
                        fprintf(wrapper, "%.*s\n\n", (int)(struct_end - struct_start), struct_start);
                        fprintf(wrapper, "/* Include list implementation */\n");
                        fprintf(wrapper, "#include \"/tmp/list_%s.c\"\n", type_name);
                    } else {
                        /* Fallback: just include the list file */
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
    char runtime_files[1536] = "src/runtime/list_int.c src/runtime/list_string.c src/runtime/list_token.c src/runtime/token_helpers.c src/runtime/gc.c src/runtime/dyn_array.c src/runtime/gc_struct.c src/runtime/nl_string.c src/runtime/cli.c";
    strncat(runtime_files, generated_lists, sizeof(runtime_files) - strlen(runtime_files) - 1);
    
    /* Add /tmp to include path for generated list headers */
    char include_flags_with_tmp[2560];
    snprintf(include_flags_with_tmp, sizeof(include_flags_with_tmp), "%s -I/tmp", include_flags);
    
    int cmd_len = snprintf(compile_cmd, sizeof(compile_cmd), 
            "gcc -std=c99 %s -o %s %s %s %s %s %s", 
            include_flags_with_tmp, output_file, temp_c_file, module_objs, runtime_files, lib_path_flags, lib_flags);
    
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