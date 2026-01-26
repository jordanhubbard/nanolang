/*
 * Fuzzer target for nanolang lexer
 *
 * This fuzzer tests the lexer with arbitrary input to find crashes,
 * hangs, or memory errors.
 *
 * Build with libFuzzer:
 *   clang -g -O1 -fsanitize=fuzzer,address -I../../src \
 *     fuzz_lexer.c ../../src/lexer.c -o fuzz_lexer
 *
 * Build with AFL++:
 *   afl-clang-fast -g -O1 -fsanitize=address -I../../src \
 *     fuzz_lexer.c ../../src/lexer.c -o fuzz_lexer
 *   afl-fuzz -i corpus_lexer -o findings_lexer ./fuzz_lexer
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

/* Minimal nanolang definitions needed for lexer */
typedef enum {
    TOK_EOF,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_COLON,
    TOK_COMMA,
    TOK_ARROW,
    TOK_PIPE,
    TOK_IDENTIFIER,
    TOK_INT,
    TOK_FLOAT,
    TOK_STRING,
    TOK_FN,
    TOK_LET,
    TOK_SET,
    TOK_RETURN,
    TOK_IF,
    TOK_WHILE,
    TOK_BREAK,
    TOK_CONTINUE,
    TOK_STRUCT,
    TOK_ENUM,
    TOK_UNION,
    TOK_IMPORT,
    TOK_EXPORT,
    TOK_PUB,
    TOK_SHADOW,
    TOK_ASSERT,
    TOK_MATCH,
    TOK_EXTERN,
    TOK_UNSAFE,
    TOK_ELLIPSIS,
    TOK_DOT,
    TOK_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char *value;
    int line;
    int column;
} Token;

/* Lexer function from src/lexer.c */
extern Token *tokenize(const char *source, int *token_count);
extern void free_tokens(Token *tokens, int count);

#ifdef __AFL_FUZZ_TESTCASE_LEN
/* AFL++ persistent mode */
__AFL_FUZZ_INIT();

int main(int argc, char **argv) {
    #ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
    #endif

    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(10000)) {
        int len = __AFL_FUZZ_TESTCASE_LEN;

        /* Ensure null termination */
        char *input = (char *)malloc(len + 1);
        if (!input) continue;
        memcpy(input, buf, len);
        input[len] = '\0';

        /* Fuzz the lexer */
        int token_count = 0;
        Token *tokens = tokenize(input, &token_count);
        if (tokens) {
            free_tokens(tokens, token_count);
        }

        free(input);
    }

    return 0;
}

#else
/* libFuzzer mode */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > 100000) {
        return 0;  /* Skip empty or huge inputs */
    }

    /* Ensure null termination */
    char *input = (char *)malloc(size + 1);
    if (!input) {
        return 0;
    }
    memcpy(input, data, size);
    input[size] = '\0';

    /* Fuzz the lexer */
    int token_count = 0;
    Token *tokens = tokenize(input, &token_count);
    if (tokens) {
        free_tokens(tokens, token_count);
    }

    free(input);
    return 0;
}
#endif
