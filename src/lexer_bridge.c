/* Stage 1.5: Bridge between nanolang lexer and C compiler
 * This file provides a C function that:
 * 1. Calls the nanolang tokenize() function (compiled from lexer_main.nano)
 * 2. Converts the returned list_Token to a Token* array
 * 3. Returns it in the format expected by the C parser
 */

#include "nanolang.h"
#include "runtime/list_LexerToken.h"

/* External reference to nanolang tokenize function
 * This will be linked from the compiled lexer_main.nano.o object file
 * Signature: fn tokenize(source: string) -> list_Token
 * Note: The nanolang main() becomes nl_main() and won't conflict with C main()
 */
extern List_LexerToken* nl_tokenize(const char* source);

/* Convert list_Token to Token* array for C parser */
Token *tokenize_nano(const char *source, int *token_count) {
    /* Call the nanolang lexer */
    List_LexerToken *token_list = nl_tokenize(source);
    
    if (!token_list || token_list->length == 0) {
        if (token_list) nl_list_LexerToken_free(token_list);
        *token_count = 0;
        return NULL;
    }
    
    /* Allocate Token array */
    int count = token_list->length;
    Token *tokens = malloc(sizeof(Token) * count);
    
    if (!tokens) {
        nl_list_LexerToken_free(token_list);
        *token_count = 0;
        return NULL;
    }
    
    /* Copy tokens from list to array */
    for (int i = 0; i < count; i++) {
        Token src = nl_list_LexerToken_get(token_list, i);
        tokens[i] = src; /* I preserve every field, including the original count. */
        tokens[i].value = src.value ? strdup(src.value) : NULL;
        if (src.value && !tokens[i].value) {
            for (int j = 0; j < i; j++) free((void *)tokens[j].value);
            free(tokens);
            nl_list_LexerToken_free(token_list);
            *token_count = 0;
            return NULL;
        }
    }
    
    *token_count = count;
    
    /* Free the list (but not the tokens, they're copied) */
    nl_list_LexerToken_free(token_list);
    
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

