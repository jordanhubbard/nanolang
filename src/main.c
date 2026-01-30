#include "nanolang.h"
#include "colors.h"
#include "version.h"
#include "module_builder.h"
#include "interpreter_ffi.h"
#include "reflection.h"
#include "runtime/list_CompilerDiagnostic.h"
#include <unistd.h>  /* For getpid() on all POSIX systems */

#ifdef __APPLE__
#include <mach-o/loader.h>
#include <fcntl.h>
#endif

/* Global argc/argv for runtime access by transpiled programs */
int g_argc = 0;
char **g_argv = NULL;

/* Compilation options */
typedef struct {
    bool verbose;
    bool keep_c;
    bool show_intermediate_code;
    bool save_asm;            /* -S flag: save generated C to .genC file */
    bool json_errors;         /* Output errors in JSON format for tooling */
    bool profile_gprof;       /* -pg flag: enable gprof profiling support */
    const char *llm_diags_json_path; /* --llm-diags-json <path> (agent-only): write diagnostics as JSON */
    const char *llm_shadow_json_path; /* --llm-shadow-json <path> (agent-only): write shadow failure summary as JSON */
    const char *reflect_output_path;  /* --reflect <path>: emit module API as JSON */
    char **include_paths;      /* -I flags */
    int include_count;
    char **library_paths;     /* -L flags */
    int library_path_count;
    char **libraries;         /* -l flags */
    int library_count;
    /* Phase 3: Module safety warnings */
    bool warn_unsafe_imports;  /* Warn when importing unsafe modules */
    bool warn_unsafe_calls;    /* Warn when calling functions from unsafe modules */
    bool warn_ffi;             /* Warn on any FFI call */
    bool forbid_unsafe;        /* Error (not warn) on unsafe modules */
} CompilerOptions;

static void json_escape(FILE *out, const char *s) {
    if (!s) return;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '\\': fputs("\\\\", out); break;
            case '"': fputs("\\\"", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", (unsigned int)c);
                else fputc((int)c, out);
        }
    }
}

static const char *phase_name(int phase) {
    switch (phase) {
        case CompilerPhase_PHASE_LEXER: return "lexer";
        case CompilerPhase_PHASE_PARSER: return "parser";
        case CompilerPhase_PHASE_TYPECHECK: return "typecheck";
        case CompilerPhase_PHASE_TRANSPILER: return "transpiler";
        case CompilerPhase_PHASE_RUNTIME: return "runtime";
        default: return "unknown";
    }
}

static const char *severity_name(int severity) {
    switch (severity) {
        case DiagnosticSeverity_DIAG_INFO: return "info";
        case DiagnosticSeverity_DIAG_WARNING: return "warning";
        case DiagnosticSeverity_DIAG_ERROR: return "error";
        default: return "unknown";
    }
}

static void llm_emit_diags_json(
    const char *path,
    const char *input_file,
    const char *output_file,
    int exit_code,
    List_CompilerDiagnostic *diags
) {
    if (!path || path[0] == '\0') return;

    FILE *f = fopen(path, "w");
    if (!f) return; /* best-effort */

    fprintf(f, "{");
    fprintf(f, "\"tool\":\"nanoc_c\",");
    fprintf(f, "\"success\":%s,", exit_code == 0 ? "true" : "false");
    fprintf(f, "\"exit_code\":%d,", exit_code);
    fprintf(f, "\"input_file\":\""); json_escape(f, input_file); fprintf(f, "\",");
    fprintf(f, "\"output_file\":\""); json_escape(f, output_file); fprintf(f, "\",");
    fprintf(f, "\"diagnostics\":[");

    int n = diags ? nl_list_CompilerDiagnostic_length(diags) : 0;
    for (int i = 0; i < n; i++) {
        CompilerDiagnostic d = nl_list_CompilerDiagnostic_get(diags, i);
        if (i > 0) fprintf(f, ",");
        fprintf(f, "{");
        fprintf(f, "\"code\":\""); json_escape(f, d.code); fprintf(f, "\",");
        fprintf(f, "\"message\":\""); json_escape(f, d.message); fprintf(f, "\",");
        fprintf(f, "\"phase\":%d,", d.phase);
        fprintf(f, "\"phase_name\":\"%s\",", phase_name(d.phase));
        fprintf(f, "\"severity\":%d,", d.severity);
        fprintf(f, "\"severity_name\":\"%s\",", severity_name(d.severity));
        fprintf(f, "\"location\":{");
        fprintf(f, "\"file\":\""); json_escape(f, d.location.file); fprintf(f, "\",");
        fprintf(f, "\"line\":%d,", d.location.line);
        fprintf(f, "\"column\":%d", d.location.column);
        fprintf(f, "}");
        fprintf(f, "}");
    }

    fprintf(f, "]}");
    fclose(f);
}

static void diags_push_simple(List_CompilerDiagnostic *diags, int phase, int severity, const char *code, const char *message) {
    if (!diags) return;
    CompilerDiagnostic d;
    d.phase = phase;
    d.severity = severity;
    d.code = code ? code : "C0000";
    d.message = message ? message : "";
    d.location.file = "";
    d.location.line = 0;
    d.location.column = 0;
    nl_list_CompilerDiagnostic_push(diags, d);
}

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
    List_CompilerDiagnostic *diags = nl_list_CompilerDiagnostic_new();

    /* Read source file */
    FILE *file = fopen(input_file, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", input_file);
        diags_push_simple(diags, CompilerPhase_PHASE_LEXER, DiagnosticSeverity_DIAG_ERROR, "CIO01", "Could not open input file");
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
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
        diags_push_simple(diags, CompilerPhase_PHASE_LEXER, DiagnosticSeverity_DIAG_ERROR, "CLEX01", "Lexing failed");
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose) printf("✓ Lexing complete (%d tokens)\n", token_count);

    /* Phase 2: Parsing */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "Parsing failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_PARSER, DiagnosticSeverity_DIAG_ERROR, "CPARSE01", "Parsing failed");
        free_tokens(tokens, token_count);
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose) printf("✓ Parsing complete\n");

    /* Phase 3: Create environment and process imports */
    clear_module_cache();  /* Clear cache from any previous compilation */
    Environment *env = create_environment();
    
    /* Set warning flags from compiler options */
    env->warn_unsafe_imports = opts->warn_unsafe_imports;
    env->warn_unsafe_calls = opts->warn_unsafe_calls;
    env->warn_ffi = opts->warn_ffi;
    env->forbid_unsafe = opts->forbid_unsafe;
    
    ModuleList *modules = create_module_list();
    if (!process_imports(program, env, modules, input_file)) {
        fprintf(stderr, "Module loading failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_PARSER, DiagnosticSeverity_DIAG_ERROR, "CIMPORT01", "Module loading failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose && modules->count > 0) {
        printf("✓ Loaded %d module(s)\n", modules->count);
    }

    /* Compile modules early so extern C functions are available for shadow tests (via FFI). */
    char module_objs[2048] = "";
    char module_compile_flags[2048] = "";

    /* Phase 4: Type Checking */
    typecheck_set_current_file(input_file);
    /* Use type_check_module if reflection is requested (modules don't need main) */
    bool typecheck_success = opts->reflect_output_path ? 
        type_check_module(program, env) : 
        type_check(program, env);
    
    if (!typecheck_success) {
        fprintf(stderr, "Type checking failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_TYPECHECK, DiagnosticSeverity_DIAG_ERROR, "CTYPE01", "Type checking failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;
    }
    if (opts->verbose) printf("✓ Type checking complete\n");

    /* Phase 4.4: Module Reflection (if requested) */
    if (opts->reflect_output_path) {
        /* Extract module name from input file */
        const char *module_name = strrchr(input_file, '/');
        module_name = module_name ? module_name + 1 : input_file;
        /* Remove .nano extension if present */
        char *name_copy = strdup(module_name);
        char *dot = strrchr(name_copy, '.');
        if (dot && strcmp(dot, ".nano") == 0) {
            *dot = '\0';
        }
        
        if (opts->verbose) printf("→ Emitting module reflection to %s\n", opts->reflect_output_path);
        
        if (!emit_module_reflection(opts->reflect_output_path, program, env, name_copy)) {
            fprintf(stderr, "Error: Failed to emit module reflection\n");
            free(name_copy);
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free_module_list(modules);
            free(source);
            llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
            nl_list_CompilerDiagnostic_free(diags);
            return 1;
        }
        
        if (opts->verbose) printf("✓ Module reflection complete\n");
        free(name_copy);
        
        /* Clean up and exit - no need to compile when reflecting */
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        nl_list_CompilerDiagnostic_free(diags);
        return 0;
    }

    /* Phase 4.5: Build imported modules (object + shared libs) */
    if (modules->count > 0) {
        if (!compile_modules(modules, env, module_objs, sizeof(module_objs),
                             module_compile_flags, sizeof(module_compile_flags),
                             opts->verbose)) {
            fprintf(stderr, "Error: Failed to compile modules\n");
            diags_push_simple(diags, CompilerPhase_PHASE_PARSER, DiagnosticSeverity_DIAG_ERROR, "CMOD01", "Failed to compile imported modules");
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free_module_list(modules);
            free(source);
            llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
            nl_list_CompilerDiagnostic_free(diags);
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

    /* Phase 5: Shadow-Test Execution (Compile-Time Function Execution) */
    if (opts->llm_shadow_json_path && opts->llm_shadow_json_path[0] != '\0') {
        setenv("NANO_LLM_SHADOW_JSON", opts->llm_shadow_json_path, 1);
    } else {
        unsetenv("NANO_LLM_SHADOW_JSON");
    }
    if (!run_shadow_tests(program, env)) {
        fprintf(stderr, "Shadow tests failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_RUNTIME, DiagnosticSeverity_DIAG_ERROR, "CSHADOW01", "Shadow tests failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        ffi_cleanup();
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        unsetenv("NANO_LLM_SHADOW_JSON");
        return 1;
    }
    if (opts->verbose) printf("✓ Shadow tests passed\n");
    unsetenv("NANO_LLM_SHADOW_JSON");

    /* Phase 5.5: Ensure module ASTs are in cache for declaration generation */
    /* Module compilation uses isolated caches that get cleared, so we need to
     * re-load modules into the main cache before transpilation so that
     * generate_module_function_declarations() can find them. */
    if (modules->count > 0) {
        if (opts->verbose) printf("Ensuring module ASTs are cached for declaration generation...\n");
        for (int i = 0; i < modules->count; i++) {
            const char *module_path = modules->module_paths[i];
            if (module_path) {
                /* Load module into cache (won't re-parse if already loaded) */
                ASTNode *module_ast = load_module(module_path, env);
                if (!module_ast) {
                    fprintf(stderr, "Warning: Failed to load module '%s' for declaration generation\n", module_path);
                }
            }
        }
        if (opts->verbose) printf("✓ Module ASTs cached\n");
    }

    /* Phase 6: C Transpilation */
    if (opts->verbose) printf("Transpiling to C...\n");
    char *c_code = transpile_to_c(program, env, input_file);
    if (!c_code) {
        fprintf(stderr, "Transpilation failed\n");
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CTRANS01", "Transpilation failed");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        ffi_cleanup();
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
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

    if (opts->show_intermediate_code) {
        fwrite(c_code, 1, c_code_size, stdout);
        fflush(stdout);
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
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CC01", "Could not create temporary C file");
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
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
    
    /* Add module directories to include path (for FFI headers) */
    /* This enables standalone tools to import modules like "modules/std/fs.nano" */
    /* and have the C compiler find the corresponding "fs.h" header */
    if (modules && modules->count > 0) {
        /* Track unique directories to avoid duplicates */
        char **unique_dirs = malloc(sizeof(char*) * modules->count);
        int unique_count = 0;
        
        for (int i = 0; i < modules->count; i++) {
            const char *module_path = modules->module_paths[i];
            if (!module_path) continue;
            
            /* Extract directory from module path */
            char dir_path[512];
            strncpy(dir_path, module_path, sizeof(dir_path) - 1);
            dir_path[sizeof(dir_path) - 1] = '\0';
            
            /* Find last slash to get directory */
            char *last_slash = strrchr(dir_path, '/');
            if (last_slash) {
                *last_slash = '\0';  /* Trim filename to get directory */
                
                /* Check if this directory is already in the list */
                bool already_added = false;
                for (int j = 0; j < unique_count; j++) {
                    if (strcmp(unique_dirs[j], dir_path) == 0) {
                        already_added = true;
                        break;
                    }
                }
                
                if (!already_added) {
                    unique_dirs[unique_count] = strdup(dir_path);
                    unique_count++;
                    
                    /* Add -I flag for this directory */
                    char temp[1024];
                    snprintf(temp, sizeof(temp), " -I%s", dir_path);
                    strncat(include_flags, temp, sizeof(include_flags) - strlen(include_flags) - 1);
                    
                    if (opts->verbose) {
                        printf("Adding module include path: %s\n", dir_path);
                    }
                }
            }
        }
        
        /* Free unique_dirs */
        for (int i = 0; i < unique_count; i++) {
            free(unique_dirs[i]);
        }
        free(unique_dirs);
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
    char runtime_files[4096] = "src/runtime/list_int.c src/runtime/list_string.c src/runtime/list_LexerToken.c src/runtime/list_token.c src/runtime/list_CompilerDiagnostic.c src/runtime/list_CompilerSourceLocation.c src/runtime/list_ASTNumber.c src/runtime/list_ASTFloat.c src/runtime/list_ASTString.c src/runtime/list_ASTBool.c src/runtime/list_ASTIdentifier.c src/runtime/list_ASTBinaryOp.c src/runtime/list_ASTCall.c src/runtime/list_ASTModuleQualifiedCall.c src/runtime/list_ASTArrayLiteral.c src/runtime/list_ASTLet.c src/runtime/list_ASTSet.c src/runtime/list_ASTStmtRef.c src/runtime/list_ASTIf.c src/runtime/list_ASTWhile.c src/runtime/list_ASTFor.c src/runtime/list_ASTReturn.c src/runtime/list_ASTBlock.c src/runtime/list_ASTUnsafeBlock.c src/runtime/list_ASTPrint.c src/runtime/list_ASTAssert.c src/runtime/list_ASTFunction.c src/runtime/list_ASTShadow.c src/runtime/list_ASTStruct.c src/runtime/list_ASTStructLiteral.c src/runtime/list_ASTFieldAccess.c src/runtime/list_ASTEnum.c src/runtime/list_ASTUnion.c src/runtime/list_ASTUnionConstruct.c src/runtime/list_ASTMatch.c src/runtime/list_ASTImport.c src/runtime/list_ASTOpaqueType.c src/runtime/list_ASTTupleLiteral.c src/runtime/list_ASTTupleIndex.c src/runtime/token_helpers.c src/runtime/gc.c src/runtime/dyn_array.c src/runtime/gc_struct.c src/runtime/nl_string.c src/runtime/cli.c src/runtime/regex.c";
    strncat(runtime_files, generated_lists, sizeof(runtime_files) - strlen(runtime_files) - 1);
    
    /* Add /tmp to include path for generated list headers */
    char include_flags_with_tmp[2560];
    snprintf(include_flags_with_tmp, sizeof(include_flags_with_tmp), "%s -I/tmp", include_flags);
    
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = "cc";

    const char *export_dynamic_flag = "";
#ifdef __linux__
    export_dynamic_flag = "-rdynamic";
#elif defined(__FreeBSD__)
    export_dynamic_flag = "-Wl,-E";
#endif

    /* Profiling flags for gprof support (-pg option) */
    const char *profile_flags = "";
    if (opts->profile_gprof) {
        profile_flags = "-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls";
        if (opts->verbose) {
            printf("Profiling enabled: adding %s\n", profile_flags);
        }
    }

    int cmd_len = snprintf(compile_cmd, sizeof(compile_cmd),
            "%s -std=c99 -Wall -Wextra -Werror -Wno-error=unused-function -Wno-error=unused-parameter -Wno-error=unused-variable -Wno-error=unused-but-set-variable -Wno-error=logical-not-parentheses -Wno-error=duplicate-decl-specifier %s %s %s -o %s %s %s %s %s %s",
            cc, profile_flags, include_flags_with_tmp, export_dynamic_flag, output_file, temp_c_file, module_objs, runtime_files, lib_path_flags, lib_flags);
    
    if (cmd_len >= (int)sizeof(compile_cmd)) {
        fprintf(stderr, "Error: Compile command too long (%d bytes, max %zu)\n", cmd_len, sizeof(compile_cmd));
        fprintf(stderr, "Try reducing the number of modules or shortening paths.\n");
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CCC02", "C compile command too long");
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        if (!opts->keep_c) {
            remove(temp_c_file);
        }
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
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
        diags_push_simple(diags, CompilerPhase_PHASE_TRANSPILER, DiagnosticSeverity_DIAG_ERROR, "CCC01", "C compilation failed");
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
        llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 1, diags);
        nl_list_CompilerDiagnostic_free(diags);
        return 1;  /* Return error if C compilation failed */
    }

    /* Remove temporary C file unless --keep-c (cleanup on both success and failure) */
    if (!opts->keep_c) {
        remove(temp_c_file);
    }

    llm_emit_diags_json(opts->llm_diags_json_path, input_file, output_file, 0, diags);
    nl_list_CompilerDiagnostic_free(diags);

    /* Cleanup */
    free(c_code);
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free_module_list(modules);
    free(source);
    ffi_cleanup();

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
        printf("  -fshow-intermediate-code  Print generated C to stdout\n");
        printf("  -S             Save generated C to <input>.genC (for inspection)\n");
        printf("  --json-errors  Output errors in JSON format for tool integration\n");
        printf("  --reflect <path>  Emit module API as JSON (for documentation generation)\n");
        printf("  -I <path>      Add include path for C compilation\n");
        printf("  -L <path>      Add library path for C linking\n");
        printf("  -l <lib>       Link against library (e.g., -lSDL2)\n");
        printf("  -pg            Enable gprof profiling (adds -g -fno-omit-frame-pointer)\n");
        printf("  --version, -v  Show version information\n");
        printf("  --help, -h     Show this help message\n");
        printf("\nSafety Options:\n");
        printf("  --warn-unsafe-imports  Warn when importing unsafe modules\n");
        printf("  --warn-unsafe-calls    Warn when calling functions from unsafe modules\n");
        printf("  --warn-ffi             Warn on any FFI (extern function) call\n");
        printf("  --forbid-unsafe        Error (not warn) on unsafe module imports\n");
        printf("\nAgent Options:\n");
        printf("  --llm-diags-json <p>   Write machine-readable diagnostics JSON (agent-only)\n");
        printf("  --llm-shadow-json <p>  Write machine-readable shadow failure summary JSON (agent-only)\n");
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
        .show_intermediate_code = false,
        .save_asm = false,
        .json_errors = false,
        .profile_gprof = false,
        .llm_diags_json_path = NULL,
        .llm_shadow_json_path = NULL,
        .reflect_output_path = NULL,
        .include_paths = NULL,
        .include_count = 0,
        .library_paths = NULL,
        .library_path_count = 0,
        .libraries = NULL,
        .library_count = 0,
        .warn_unsafe_imports = false,
        .warn_unsafe_calls = false,
        .warn_ffi = false,
        .forbid_unsafe = false
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
        } else if (strcmp(argv[i], "-fshow-intermediate-code") == 0) {
            opts.show_intermediate_code = true;
        } else if (strcmp(argv[i], "-S") == 0) {
            opts.save_asm = true;
        } else if (strcmp(argv[i], "--json-errors") == 0) {
            opts.json_errors = true;
        } else if (strcmp(argv[i], "-pg") == 0) {
            opts.profile_gprof = true;
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
        } else if (strcmp(argv[i], "--warn-unsafe-imports") == 0) {
            opts.warn_unsafe_imports = true;
        } else if (strcmp(argv[i], "--warn-unsafe-calls") == 0) {
            opts.warn_unsafe_calls = true;
        } else if (strcmp(argv[i], "--warn-ffi") == 0) {
            opts.warn_ffi = true;
        } else if (strcmp(argv[i], "--forbid-unsafe") == 0) {
            opts.forbid_unsafe = true;
        } else if (strcmp(argv[i], "--llm-diags-json") == 0 && i + 1 < argc) {
            opts.llm_diags_json_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--llm-shadow-json") == 0 && i + 1 < argc) {
            opts.llm_shadow_json_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--reflect") == 0 && i + 1 < argc) {
            opts.reflect_output_path = argv[i + 1];
            i++;
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