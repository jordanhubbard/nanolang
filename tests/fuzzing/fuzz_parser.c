/*
 * Fuzzer target for nanolang parser
 *
 * This fuzzer tests the parser with arbitrary input to find crashes,
 * hangs, or memory errors in the parsing stage.
 *
 * Build with libFuzzer:
 *   clang -g -O1 -fsanitize=fuzzer,address -I../../src \
 *     fuzz_parser.c ../../src/lexer.c ../../src/parser.c -o fuzz_parser
 *
 * Build with AFL++:
 *   afl-clang-fast -g -O1 -fsanitize=address -I../../src \
 *     fuzz_parser.c ../../src/lexer.c ../../src/parser.c -o fuzz_parser
 *   afl-fuzz -i corpus_parser -o findings_parser ./fuzz_parser
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

/* Forward declarations */
typedef struct TokenList TokenList;
typedef struct ASTNode ASTNode;
typedef struct ParseError ParseError;

/* External functions from lexer and parser */
extern TokenList *tokenize(const char *source, const char *filename);
extern void free_tokens(TokenList *list);
extern ASTNode *parse(TokenList *tokens, ParseError **error);
extern void free_ast(ASTNode *node);
extern void free_parse_error(ParseError *error);

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

        /* Fuzz lexer + parser */
        TokenList *tokens = tokenize(input, "fuzz_input");
        if (tokens) {
            ParseError *error = NULL;
            ASTNode *ast = parse(tokens, &error);

            if (ast) {
                free_ast(ast);
            }
            if (error) {
                free_parse_error(error);
            }

            free_tokens(tokens);
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

    /* Fuzz lexer + parser */
    TokenList *tokens = tokenize(input, "fuzz_input");
    if (tokens) {
        ParseError *error = NULL;
        ASTNode *ast = parse(tokens, &error);

        if (ast) {
            free_ast(ast);
        }
        if (error) {
            free_parse_error(error);
        }

        free_tokens(tokens);
    }

    free(input);
    return 0;
}
#endif
