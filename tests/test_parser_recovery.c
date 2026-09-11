#include "nanolang.h"
#include <assert.h>

int g_argc = 0;
char **g_argv = NULL;

int main(void) {
    const char *source = "fn probe(line: string) -> int {\n"
        "let byte: int = 0\nif (== byte 34) { return 1 }\n"
        "return 0\n}\nfn main() -> int { return 0 }\n";
    int count = 0;
    Token *tokens = tokenize(source, &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    assert(program == NULL);
    free_tokens(tokens, count);

    const char *bad[] = {
        "fn main() -> int { return (+ 1 int) }",
        "fn main() -> int { return (f 1 int) }",
        "fn main() -> int { return (+ (+ 1 2) int) }",
        "fn main() -> bool { return (not int) }",
        "fn main() -> int { return (+ 1 2"
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        tokens = tokenize(bad[i], &count);
        assert(tokens);
        assert(parse_program(tokens, count) == NULL);
        free_tokens(tokens, count);
    }
    const char *good[] = {
        "fn main() -> int { return (+ 1 2) }",
        "fn main() -> int { return (f 1 2 3 4 5 6 7 8 9) }",
        "fn main() -> int { return (+ 1 2 3 4 5 6 7 8 9) }",
        "fn main() -> int { return (f) }"
    };
    for (size_t i = 0; i < sizeof(good) / sizeof(good[0]); i++) {
        tokens = tokenize(good[i], &count);
        assert(tokens);
        program = parse_program(tokens, count);
        assert(program);
        free_ast(program);
        free_tokens(tokens, count);
    }
    return 0;
}
