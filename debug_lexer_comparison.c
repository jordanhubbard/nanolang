/* Debug tool to compare C lexer vs nanolang lexer tokens */
#include <stdio.h>
#include <stdlib.h>
#include "src/nanolang.h"

extern Token *tokenize(const char *source, int *count);
extern Token *tokenize_nano(const char *source, int *count);

void print_tokens(const char *label, Token *tokens, int count) {
    printf("\n=== %s (%d tokens) ===\n", label, count);
    for (int i = 0; i < count && i < 10; i++) {
        printf("Token %d: type=%d (%s), value='%s', line=%d, col=%d\n",
               i, tokens[i].type, 
               tokens[i].type < 50 ? "valid" : "INVALID",
               tokens[i].value ? tokens[i].value : "(null)",
               tokens[i].line, tokens[i].column);
    }
}

int main() {
    const char *source = "fn test() -> int {\n    return 42\n}\n";
    
    printf("Source code:\n%s\n", source);
    
    /* Test C lexer */
    int c_count = 0;
    Token *c_tokens = tokenize(source, &c_count);
    print_tokens("C Lexer", c_tokens, c_count);
    
    /* Test nanolang lexer */
    int nano_count = 0;
    Token *nano_tokens = tokenize_nano(source, &nano_count);
    print_tokens("Nanolang Lexer", nano_tokens, nano_count);
    
    /* Compare first few tokens */
    printf("\n=== Comparison ===\n");
    int min_count = c_count < nano_count ? c_count : nano_count;
    for (int i = 0; i < min_count && i < 10; i++) {
        if (c_tokens[i].type != nano_tokens[i].type) {
            printf("Token %d MISMATCH: C type=%d, Nano type=%d\n",
                   i, c_tokens[i].type, nano_tokens[i].type);
        }
    }
    
    return 0;
}

