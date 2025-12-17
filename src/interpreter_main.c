#include "nanolang.h"
#include "version.h"

/* Global argc/argv for runtime access */
int g_argc = 0;
char **g_argv = NULL;

#include "tracing.h"
#include "runtime/gc.h"
#include "interpreter_ffi.h"

/* Interpreter options */
typedef struct {
    bool verbose;
    const char *call_function;
    char **call_args;
    int call_arg_count;
} InterpreterOptions;

/* Run nanolang source with interpreter */
static int interpret_file(const char *input_file, InterpreterOptions *opts, int argc, char **argv) {
    /* Initialize GC */
    gc_init();
    
    /* Initialize tracing */
    tracing_init();
#ifdef TRACING_ENABLED
    tracing_configure(argc, argv);
#else
    (void)argc;  /* Unused without tracing */
    (void)argv;  /* Unused without tracing */
#endif
    
    /* Read source file */
    FILE *file = fopen(input_file, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", input_file);
        gc_shutdown();
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *source = malloc(size + 1);
    fread(source, 1, size, file);
    source[size] = '\0';
    fclose(file);

    if (opts->verbose) printf("Interpreting %s...\n", input_file);

    /* Phase 1: Lexing */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "Lexing failed\n");
        free(source);
        gc_shutdown();
        return 1;
    }
    if (opts->verbose) printf("✓ Lexing complete (%d tokens)\n", token_count);

    /* Phase 2: Parsing */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "Parsing failed\n");
        free_tokens(tokens, token_count);
        free(source);
        gc_shutdown();
        return 1;
    }
    if (opts->verbose) printf("✓ Parsing complete\n");

    /* Phase 3: Create environment and process imports */
    Environment *env = create_environment();
    ModuleList *modules = create_module_list();
    if (!process_imports(program, env, modules, input_file)) {
        fprintf(stderr, "Module loading failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        gc_shutdown();
        return 1;
    }
    if (opts->verbose && modules->count > 0) {
        printf("✓ Loaded %d module(s)\n", modules->count);
    }

    /* Phase 3.5: Initialize FFI and load module shared libraries */
    if (!ffi_init(opts->verbose)) {
        fprintf(stderr, "FFI initialization failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        gc_shutdown();
        return 1;
    }
    
    /* Load each module's shared library for FFI access */
    for (int i = 0; i < modules->count; i++) {
        const char *module_name = modules->module_paths[i];
        /* Extract module name from path (last component without .nano) */
        const char *last_slash = strrchr(module_name, '/');
        const char *base_name = last_slash ? last_slash + 1 : module_name;
        char mod_name[256];
        snprintf(mod_name, sizeof(mod_name), "%s", base_name);
        char *dot = strrchr(mod_name, '.');
        if (dot) *dot = '\0';
        
        /* Try to load module shared library (may not exist for pure nanolang modules) */
        ffi_load_module(mod_name, module_name, env, opts->verbose);
    }

    /* Phase 4: Type Checking */
    if (!type_check(program, env)) {
        fprintf(stderr, "Type checking failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        gc_shutdown();
        return 1;
    }
    if (opts->verbose) printf("✓ Type checking complete\n");

    /* Phase 5: Interpret */
    if (!run_program(program, env)) {
        fprintf(stderr, "Interpretation failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        gc_shutdown();
        return 1;
    }
    if (opts->verbose) printf("✓ Interpretation complete\n");

    /* Phase 6: Shadow Test Execution (parity with compiler) */
    if (!run_shadow_tests(program, env)) {
        fprintf(stderr, "Shadow tests failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        free(source);
        gc_shutdown();
        return 1;
    }
    if (opts->verbose) printf("✓ Shadow tests passed\n");

    /* Phase 7: Call main() or specified function */
    int exit_code = 0;
    const char *func_to_call = opts->call_function ? opts->call_function : "main";
    bool should_call_function = opts->call_function != NULL;
    
    /* If no explicit --call, check if main() exists */
    if (!opts->call_function) {
        Function *main_func = env_get_function(env, "main");
        should_call_function = (main_func != NULL);
    }
    
    if (should_call_function) {
        Function *func = env_get_function(env, func_to_call);
        if (!func) {
            if (opts->call_function) {
                /* Only error if explicitly requested via --call */
                fprintf(stderr, "Error: Function '%s' not found\n", func_to_call);
                free_ast(program);
                free_tokens(tokens, token_count);
                free_environment(env);
                free(source);
                return 1;
            }
            /* If main() doesn't exist and wasn't explicitly requested, just finish */
            free_module_list(modules);
            modules = NULL;  /* Prevent double-free */
            goto cleanup;
        }

        if (opts->verbose) printf("Calling function '%s'...\n", func_to_call);

        /* Cleanup modules before function call */
        free_module_list(modules);
        modules = NULL;  /* Prevent double-free */

        /* Parse and convert call_args to proper argument values */
        Value *args = NULL;
        int arg_count = 0;
        
        if (opts->call_arg_count > 0) {
            /* Check argument count matches function signature */
            if (opts->call_arg_count != func->param_count) {
                fprintf(stderr, "Error: Function '%s' expects %d arguments, got %d\n",
                        func_to_call, func->param_count, opts->call_arg_count);
                free_ast(program);
                free_tokens(tokens, token_count);
                free_environment(env);
                free(source);
                tracing_cleanup();
                gc_shutdown();
                return 1;
            }
            
            /* Allocate argument array */
            args = malloc(sizeof(Value) * opts->call_arg_count);
            arg_count = opts->call_arg_count;
            
            /* Convert string arguments to Value types based on parameter types */
            for (int i = 0; i < opts->call_arg_count; i++) {
                Type param_type = func->params[i].type;
                const char *arg_str = opts->call_args[i];
                
                switch (param_type) {
                    case TYPE_INT: {
                        long long val = strtoll(arg_str, NULL, 10);
                        args[i] = create_int(val);
                        break;
                    }
                    case TYPE_FLOAT: {
                        double val = strtod(arg_str, NULL);
                        args[i] = create_float(val);
                        break;
                    }
                    case TYPE_BOOL: {
                        bool val = (strcmp(arg_str, "true") == 0 || 
                                   strcmp(arg_str, "1") == 0 ||
                                   strcmp(arg_str, "True") == 0);
                        args[i] = create_bool(val);
                        break;
                    }
                    case TYPE_STRING: {
                        args[i] = create_string(strdup(arg_str));
                        break;
                    }
                    case TYPE_ARRAY:
                        /* Arrays from command line not supported yet */
                        fprintf(stderr, "Error: Array arguments from command line not supported\n");
                        free(args);
                        free_ast(program);
                        free_tokens(tokens, token_count);
                        free_environment(env);
                        free(source);
                        tracing_cleanup();
                        gc_shutdown();
                        return 1;
                    default:
                        fprintf(stderr, "Error: Unsupported parameter type for argument %d\n", i + 1);
                        free(args);
                        free_ast(program);
                        free_tokens(tokens, token_count);
                        free_environment(env);
                        free(source);
                        tracing_cleanup();
                        gc_shutdown();
                        return 1;
                }
            }
        }

        /* Call the function */
        Value result = call_function(func_to_call, args, arg_count, env);
        
        /* Free argument strings if allocated */
        if (args) {
            for (int i = 0; i < arg_count; i++) {
                if (func->params[i].type == TYPE_STRING && args[i].type == VAL_STRING) {
                    free(args[i].as.string_val);
                }
            }
            free(args);
        }

        /* If the function returns an int, use it as exit code */
        if (result.type == VAL_INT) {
            exit_code = (int)result.as.int_val;
        }
    }

cleanup:
    /* Cleanup */
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    if (modules) {
        free_module_list(modules);
    }
    free(source);
    ffi_cleanup();
    tracing_cleanup();
    gc_shutdown();

    return exit_code;
}

/* Main entry point */
int main(int argc, char *argv[]) {
    /* Store argc/argv for runtime access */
    g_argc = argc;
    g_argv = argv;
    
    /* Handle --version */
    if (argc >= 2 && (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0)) {
        printf("nano %s\n", NANOLANG_VERSION);
        printf("nanolang interpreter\n");
        printf("Built: %s %s\n", NANOLANG_BUILD_DATE, NANOLANG_BUILD_TIME);
        return 0;
    }

    /* Handle --help */
    if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("nano - Interpreter for the nanolang programming language\n\n");
        printf("Usage: %s <input.nano> [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("  --call <func>  Call specified function after loading program\n");
        printf("  --verbose      Show detailed interpretation steps\n");
        printf("  --version, -v  Show version information\n");
        printf("  --help, -h     Show this help message\n");
        printf("\nTracing Options:\n");
        printf("  --trace, --trace-all          Enable tracing for all functions and variables\n");
        printf("  --trace-function=<name>       Trace specific function (can be used multiple times)\n");
        printf("  --trace-var=<name>           Trace specific variable (can be used multiple times)\n");
        printf("  --trace-scope=<func>         Trace everything inside specific function (can be used multiple times)\n");
        printf("  --trace-regex=<pattern>      Trace anything matching regex pattern (can be used multiple times)\n");
        printf("\nExamples:\n");
        printf("  %s hello.nano --call main\n", argv[0]);
        printf("  %s program.nano --verbose\n", argv[0]);
        printf("  %s program.nano --trace-all\n", argv[0]);
        printf("  %s program.nano --trace-function=calculate_sum\n", argv[0]);
        printf("  %s program.nano --trace-scope=main\n", argv[0]);
        printf("  %s program.nano --trace-regex='^test.*'\n\n", argv[0]);
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.nano> [OPTIONS]\n", argv[0]);
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    InterpreterOptions opts = {
        .verbose = false,
        .call_function = NULL,
        .call_args = NULL,
        .call_arg_count = 0
    };

    /* Parse command-line options */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "--call") == 0 && i + 1 < argc) {
            opts.call_function = argv[i + 1];
            i++;
            /* Parse additional arguments after function name */
            int arg_start = i + 1;
            int arg_end = argc;
            /* Find where arguments end (next option or end of argv) */
            for (int j = arg_start; j < argc; j++) {
                if (argv[j][0] == '-' && strncmp(argv[j], "--trace", 7) != 0) {
                    arg_end = j;
                    break;
                }
            }
            opts.call_arg_count = arg_end - arg_start;
            if (opts.call_arg_count > 0) {
                opts.call_args = &argv[arg_start];
            }
            i = arg_end - 1;  /* Skip processed arguments */
        } else if (strcmp(argv[i], "-o") == 0) {
            fprintf(stderr, "Error: Interpreter does not support -o flag (use nanoc compiler instead)\n");
            return 1;
        } else if (strncmp(argv[i], "--trace", 7) == 0) {
            /* Tracing options are handled by tracing_configure */
            continue;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            return 1;
        }
    }

    return interpret_file(input_file, &opts, argc, argv);
}
