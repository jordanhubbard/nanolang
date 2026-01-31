/* Stage 1.5: Bridge between nanolang lexer and C compiler
 * This file provides a C function that:
 * 1. Calls the nanolang tokenize() function (compiled from lexer_main.nano)
 * 2. Converts the returned list_token to a Token* array
 * 3. Returns it in the format expected by the C parser
 */

#include "nanolang.h"
#include "runtime/list_token.h"

/* External reference to nanolang tokenize function
 * This will be linked from the compiled lexer_main.nano.o object file
 * Signature: fn tokenize(source: string) -> list_token
 * Note: The nanolang main() becomes nl_main() and won't conflict with C main()
 */
extern List_token* nl_tokenize(const char* source);

/* Convert list_token to Token* array for C parser */
Token *tokenize_nano(const char *source, int *token_count) {
    /* Call the nanolang lexer */
    List_token *token_list = nl_tokenize(source);
    
    if (!token_list || token_list->length == 0) {
        *token_count = 0;
        return NULL;
    }
    
    /* Allocate Token array */
    int count = token_list->length;
    Token *tokens = malloc(sizeof(Token) * count);
    
    if (!tokens) {
        list_token_free(token_list);
        *token_count = 0;
        return NULL;
    }
    
    /* Copy tokens from list to array */
    for (int i = 0; i < count; i++) {
        Token *src = list_token_get(token_list, i);
        if (!src) {
            /* Error: token is NULL */
            for (int j = 0; j < i; j++) {
                free(tokens[j].value);
            }
            free(tokens);
            list_token_free(token_list);
            *token_count = 0;
            return NULL;
        }
        
        /* Deep copy token */
        tokens[i].token_type = src->token_type;
        tokens[i].value = src->value ? strdup(src->value) : NULL;
        tokens[i].line = src->line;
        tokens[i].column = src->column;
    }
    
    *token_count = count;
    
    /* Free the list (but not the tokens, they're copied) */
    list_token_free(token_list);
    
    return tokens;
}

/* For Stage 1.5: Use nanolang lexer, fallback to C lexer if needed */
Token *tokenize_hybrid(const char *source, int *token_count, bool use_nano_lexer) {
    if (use_nano_lexer) {
        return tokenize_nano(source, token_count);
    } else {
        /* Use C lexer (from lexer.c) */
        extern Token *tokenize(const char *source, int *token_count);
        return tokenize(source, token_count);
    }
}

