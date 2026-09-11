/* I exercise the production lexer/parser API without executing input.
 * Build with make fuzz-parser-build (libFuzzer). */
#include "nanolang.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 100000) return 0;
    char *input = malloc(size + 1);
    if (!input) return 0;
    if (size) memcpy(input, data, size);
    input[size] = '\0';
    /* My source API ignores bytes after an embedded NUL. */
    int count = 0;
    Token *tokens = tokenize(input, &count);
    if (tokens) {
        ASTNode *program = parse_program(tokens, count);
        if (program) free_ast(program);
        free_tokens(tokens, count);
    }
    free(input);
    return 0;
}

#ifdef __AFL_FUZZ_TESTCASE_LEN
__AFL_FUZZ_INIT();
int main(void) {
#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif
    unsigned char *data = __AFL_FUZZ_TESTCASE_BUF;
    while (__AFL_LOOP(10000)) {
        size_t size = __AFL_FUZZ_TESTCASE_LEN;
        LLVMFuzzerTestOneInput(data, size);
    }
    return 0;
}
#endif
