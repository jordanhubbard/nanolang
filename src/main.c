#include "nanolang.h"
#include "version.h"

/* Compilation options */
typedef struct {
    bool verbose;
    bool keep_c;
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

    /* Phase 4: Shadow-Test Execution */
    if (!run_shadow_tests(program, env)) {
        fprintf(stderr, "Shadow tests failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Shadow tests passed\n");

    /* Phase 5: C Transpilation */
    char *c_code = transpile_to_c(program, env);
    if (!c_code) {
        fprintf(stderr, "Transpilation failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free(source);
        return 1;
    }
    if (opts->verbose) printf("✓ Transpilation complete\n");

    /* Write C code to temporary file */
    char temp_c_file[256];
    snprintf(temp_c_file, sizeof(temp_c_file), "%s.c", output_file);

    FILE *c_file = fopen(temp_c_file, "w");
    if (!c_file) {
        fprintf(stderr, "Error: Could not create C file '%s'\n", temp_c_file);
        free(c_code);
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free(source);
        return 1;
    }

    fprintf(c_file, "%s", c_code);
    fclose(c_file);
    if (opts->verbose) printf("✓ Generated C code: %s\n", temp_c_file);

    /* Compile C code with gcc (include runtime) */
    char compile_cmd[1024];
    if (opts->verbose) {
        snprintf(compile_cmd, sizeof(compile_cmd), 
                "gcc -std=c99 -Isrc -o %s %s src/runtime/list_int.c src/runtime/list_string.c -lm", 
                output_file, temp_c_file);
    } else {
        snprintf(compile_cmd, sizeof(compile_cmd), 
                "gcc -std=c99 -Isrc -o %s %s src/runtime/list_int.c src/runtime/list_string.c -lm 2>/dev/null", 
                output_file, temp_c_file);
    }

    if (opts->verbose) printf("Compiling C code: %s\n", compile_cmd);
    int result = system(compile_cmd);

    if (result == 0) {
        if (opts->verbose) printf("✓ Compilation successful: %s\n", output_file);
        /* Remove temporary C file unless --keep-c */
        if (!opts->keep_c) {
            remove(temp_c_file);
        }
    } else {
        fprintf(stderr, "C compilation failed\n");
    }

    /* Cleanup */
    free(c_code);
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free(source);

    return result;
}

/* Main entry point */
int main(int argc, char *argv[]) {
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
        printf("  -o <file>      Specify output file (default: a.out)\n");
        printf("  --verbose      Show detailed compilation steps\n");
        printf("  --keep-c       Keep generated C file\n");
        printf("  --version, -v  Show version information\n");
        printf("  --help, -h     Show this help message\n");
        printf("\nExamples:\n");
        printf("  %s hello.nano -o hello\n", argv[0]);
        printf("  %s program.nano --verbose --keep-c\n", argv[0]);
        printf("  %s example.nano -o example --verbose\n\n", argv[0]);
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.nano> [OPTIONS]\n", argv[0]);
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = "a.out";
    CompilerOptions opts = {
        .verbose = false,
        .keep_c = false
    };

    /* Parse command-line options */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "--keep-c") == 0) {
            opts.keep_c = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            return 1;
        }
    }

    return compile_file(input_file, output_file, &opts);
}