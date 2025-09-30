#include "nanolang.h"

/* Execute source code */
static void execute(const char *source, Environment *env) {
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    
    if (token_count <= 1) {
        free_tokens(tokens, token_count);
        return;
    }
    
    int pos = 0;
    while (pos < token_count && tokens[pos].type != TOKEN_EOF) {
        ASTNode *ast = parse(tokens, token_count, &pos);
        if (ast) {
            eval(ast, env);
            free_ast(ast);
        }
    }
    
    free_tokens(tokens, token_count);
}

/* REPL - Read-Eval-Print Loop */
void repl(void) {
    Environment *env = create_environment();
    char line[1024];
    
    printf("NanoLang v0.1.0 Interactive Interpreter\n");
    printf("Type 'exit' to quit\n\n");
    
    while (1) {
        printf("nano> ");
        fflush(stdout);
        
        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }
        
        /* Remove trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        
        /* Check for exit command */
        if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) {
            break;
        }
        
        /* Skip empty lines */
        if (strlen(line) == 0) {
            continue;
        }
        
        execute(line, env);
    }
    
    free_environment(env);
    printf("\nGoodbye!\n");
}

/* Main entry point */
int main(int argc, char *argv[]) {
    if (argc > 1) {
        /* Execute file */
        FILE *file = fopen(argv[1], "r");
        if (!file) {
            fprintf(stderr, "Error: Could not open file '%s'\n", argv[1]);
            return 1;
        }
        
        /* Read entire file */
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        
        char *source = malloc(size + 1);
        fread(source, 1, size, file);
        source[size] = '\0';
        fclose(file);
        
        /* Execute */
        Environment *env = create_environment();
        execute(source, env);
        free_environment(env);
        free(source);
    } else {
        /* Start REPL */
        repl();
    }
    
    return 0;
}
