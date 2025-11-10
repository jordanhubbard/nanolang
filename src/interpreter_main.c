#include "nanolang.h"
#include "version.h"

/* Interpreter options */
typedef struct {
    bool verbose;
    const char *call_function;
    char **call_args;
    int call_arg_count;
} InterpreterOptions;

/* Run nanolang source with interpreter */
static int interpret_file(const char *input_file, InterpreterOptions *opts) {
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

    if (opts->verbose) printf("Interpreting %s...\n", input_file);

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

    /* Phase 3: Type Checking */
    Environment *env = create_environment();
    if (!type_check(program, env)) {
        fprintf(stderr, "Type checking failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Type checking complete\n");

    /* Phase 4: Interpret */
    if (!run_program(program, env)) {
        fprintf(stderr, "Interpretation failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Interpretation complete\n");

    /* Phase 5: Call specified function if requested */
    int exit_code = 0;
    if (opts->call_function) {
        Function *func = env_get_function(env, opts->call_function);
        if (!func) {
            fprintf(stderr, "Error: Function '%s' not found\n", opts->call_function);
            free_ast(program);
            free_tokens(tokens, token_count);
            free_environment(env);
            free(source);
            return 1;
        }

        if (opts->verbose) printf("Calling function '%s'...\n", opts->call_function);

        /* For now, we only support calling functions with no arguments */
        /* TODO: Parse and convert call_args to proper argument values */
        if (opts->call_arg_count > 0) {
            fprintf(stderr, "Warning: Function arguments not yet implemented, calling with no args\n");
        }

        /* Call the function */
        Value result = call_function(opts->call_function, NULL, 0, env);

        /* If the function returns an int, use it as exit code */
        if (result.type == VAL_INT) {
            exit_code = (int)result.as.int_val;
        }
    }

    /* Cleanup */
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free(source);

    return exit_code;
}

/* Main entry point */
int main(int argc, char *argv[]) {
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
        printf("\nExamples:\n");
        printf("  %s hello.nano --call main\n", argv[0]);
        printf("  %s program.nano --verbose\n\n", argv[0]);
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
            /* TODO: Parse additional arguments after function name */
        } else if (strcmp(argv[i], "-o") == 0) {
            fprintf(stderr, "Error: Interpreter does not support -o flag (use nanoc compiler instead)\n");
            return 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            return 1;
        }
    }

    return interpret_file(input_file, &opts);
}
