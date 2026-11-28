#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "../src/nanolang.h"
#include "../src/runtime/gc.h"
#include "../src/runtime/dyn_array.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;

static void check(const char *label, const char *input, const char *delimiter,
                  const char **expected, size_t count) {
    const char *source =
        "fn detached(source: string, delimiter: string) -> array<string> { "
        "return (str_split source delimiter) }\n"
        "shadow detached { let values: array<string> = (detached \"a,b\" \",\") "
        "assert (== (array_length values) 2) assert (== (at values 0) \"a\") }\n"
        "fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n";
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count); assert(tokens);
    ASTNode *program = parse_program(tokens, token_count); assert(program);
    clear_module_cache();
    Environment *env = create_environment(); assert(env);
    assert(type_check(program, env)); assert(run_program(program, env));
    Value args[] = {create_string(input), create_string(delimiter)};
    Value value = call_function("detached", args, 2, env);
    assert(value.type == VAL_DYN_ARRAY);
    DynArray *array = value.as.dyn_array_val;
    assert(dyn_array_length(array) == (int64_t)count);
    /* I retire the caller's inputs and frame before reading every result byte. */
    gc_release(args[0].as.string_val); gc_release(args[1].as.string_val);
    free_environment(env);
    for (int round = 0; round < 64; ++round) {
        char *churn = malloc(strlen(input) + 1); assert(churn);
        memset(churn, 'X', strlen(input)); churn[strlen(input)] = '\0'; free(churn);
        for (size_t i = 0; i < count; ++i) {
            const char *actual = dyn_array_get_string(array, (int64_t)i);
            assert(actual && strcmp(actual, expected[i]) == 0);
        }
    }
    free_ast(program); free_tokens(tokens, token_count); clear_module_cache();
    gc_shutdown();
    printf("I retained %s across input and frame retirement.\n", label);
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *empty[] = {""}; check("empty", "", "\n", empty, 1);
    const char *plain[] = {"hello"}; check("plain tail", "hello", "\n", plain, 1);
    const char *boundaries[] = {"", "a", "", "b", ""};
    check("newline boundaries", "\na\n\nb\n", "\n", boundaries, 5);
    const char *multi[] = {"", "a", "", "b", ""};
    check("multi-byte delimiter", "--a----b--", "--", multi, 5);
    const char *crlf[] = {"a\r", "b\r", ""}; check("CRLF", "a\r\nb\r\n", "\n", crlf, 3);
    const char *utf8[] = {"café", "茶", ""}; check("UTF8 segments", "café\n茶\n", "\n", utf8, 3);
    const char *bytes[] = {"c", "a", "f", "\xc3", "\xa9", "\xe8", "\x8c", "\xb6"};
    check("empty-delimiter bytes", "café茶", "", bytes, 8);
    check("empty input and delimiter", "", "", NULL, 0);
    const char *overlap[] = {"", "", "a"}; check("nonoverlapping matches", "aaaaa", "aa", overlap, 3);
    return 0;
}
