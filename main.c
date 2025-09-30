#include "nanolang.h"

/* Compile nanolang source to executable */
static int compile_file(const char *input_file, const char *output_file) {
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

    printf("Compiling %s...\n", input_file);

    /* Phase 1: Lexing */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "Lexing failed\n");
        free(source);
        return 1;
    }
    printf("✓ Lexing complete (%d tokens)\n", token_count);

    /* Phase 2: Parsing */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "Parsing failed\n");
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }
    printf("✓ Parsing complete\n");

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
    printf("✓ Type checking complete\n");

    /* Phase 4: Shadow-Test Execution */
    if (!run_shadow_tests(program, env)) {
        fprintf(stderr, "Shadow tests failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free(source);
        return 1;
    }
    printf("✓ Shadow tests passed\n");

    /* Phase 5: C Transpilation */
    char *c_code = transpile_to_c(program);
    if (!c_code) {
        fprintf(stderr, "Transpilation failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free(source);
        return 1;
    }
    printf("✓ Transpilation complete\n");

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
    printf("✓ Generated C code: %s\n", temp_c_file);

    /* Compile C code with gcc */
    char compile_cmd[512];
    snprintf(compile_cmd, sizeof(compile_cmd), "gcc -std=c99 -o %s %s", output_file, temp_c_file);

    printf("Compiling C code: %s\n", compile_cmd);
    int result = system(compile_cmd);

    if (result == 0) {
        printf("✓ Compilation successful: %s\n", output_file);
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
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.nano> [-o output]\n", argv[0]);
        fprintf(stderr, "\nCompiler for the nanolang programming language\n");
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  -o <file>    Specify output file (default: a.out)\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s hello.nano -o hello\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = "a.out";

    /* Parse command-line options */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[i + 1];
            i++;
        }
    }

    return compile_file(input_file, output_file);
}