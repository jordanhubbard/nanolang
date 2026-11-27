#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "../src/nanolang.h"
#include "../src/runtime/gc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;

static const char *observed_name;
static size_t observed_lookups;

/* I observe only evaluator lookups; the actual environment still resolves them. */
Function *nano_test_env_get_function(Environment *env, const char *name) {
    if (observed_name && !strcmp(name, observed_name)) ++observed_lookups;
    return env_get_function(env, name);
}

static void check(const char *label, const char *name, const char *declarations,
                  const char *statement, long long expected, long long side_effects) {
    char source[16384];
    int length = snprintf(source, sizeof source,
        "let mut actual: int = 0\nlet mut seen: int = 0\n%s\n"
        "fn main() -> int { %s return 0 }\n"
        "shadow main { assert (== (main) 0) }\n", declarations, statement);
    assert(length > 0 && (size_t)length < sizeof source);
    int count = 0;
    Token *tokens = tokenize(source, &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    assert(program);
    clear_module_cache();
    Environment *env = create_environment();
    assert(env);
    env_set_current_file(env, "<builtin-resolution>");
    typecheck_set_current_file("<builtin-resolution>");
    assert(type_check(program, env));
    observed_name = name;
    observed_lookups = 0;
    assert(run_program(program, env));
    observed_lookups = 0;
    Value result = call_function("main", NULL, 0, env);
    assert(result.type == VAL_INT && result.as.int_val == 0);
    size_t calls = observed_lookups;
    observed_name = NULL;
    Symbol *actual = env_get_var(env, "actual");
    Symbol *seen = env_get_var(env, "seen");
    if (!actual || actual->value.type != VAL_INT || actual->value.as.int_val != expected)
        fprintf(stderr, "I found a different value for %s: type=%d value=%lld expected=%lld\n",
                label, actual ? (int)actual->value.type : -1,
                actual && actual->value.type == VAL_INT ? actual->value.as.int_val : 0, expected);
    assert(actual && actual->value.type == VAL_INT && actual->value.as.int_val == expected);
    assert(seen && seen->value.type == VAL_INT && seen->value.as.int_val == side_effects);
    printf("{\"case\":\"%s\",\"value\":%lld,\"side_effects\":%lld,\"evaluator_lookups\":%zu}\n",
           label, expected, side_effects, calls);
    free_environment(env);
    free_ast(program);
    free_tokens(tokens, count);
    clear_module_cache();
    gc_shutdown();
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    check("string-length", "str_length", "", "set actual (str_length \"abcd\")", 4, 0);
    check("byte-access", "char_at", "", "set actual (char_at \"abc\" 1)", 98, 0);
    check("absolute-value", "abs", "", "set actual (abs (- 0 9))", 9, 0);
    check("array-length", "array_length", "let items: array<int> = [1,2,3]",
          "set actual (array_length items)", 3, 0);
    check("integer-text", "int_to_string", "", "set actual (str_length (int_to_string 1234))", 4, 0);
    check("empty-print", "print", "", "(print \"\") set actual 1", 1, 0);
    check("ordinary-function", "ordinary",
          "fn ordinary(x: int) -> int { return (+ x 10) }\n"
          "shadow ordinary { assert (== (ordinary 3) 13) }",
          "set actual (ordinary 9)", 19, 0);
    check("declared-builder", "nl_sb_length",
          "fn nl_sb_length(x: int) -> int { return (+ x 7) }\n"
          "shadow nl_sb_length { assert (== (nl_sb_length 1) 8) }",
          "set actual (nl_sb_length 5)", 12, 0);
    check("declared-push", "array_push",
          "fn array_push(x: int, y: int) -> int { return (+ x y) }\n"
          "shadow array_push { assert (== (array_push 2 3) 5) }",
          "set actual (array_push 9 4)", 13, 0);
    check("bstring-prefixed-function", "bstr_custom",
          "fn bstr_custom(x: int, y: int) -> int { return (+ x y) }\n"
          "shadow bstr_custom { assert (== (bstr_custom 1 2) 3) }",
          "set actual (bstr_custom 6 8)", 14, 0);
    check("ordinary-builtin-suffix", "str_length_custom",
          "fn str_length_custom(x: string) -> int { return 42 }\n"
          "shadow str_length_custom { assert (== (str_length_custom \"abc\") 42) }",
          "set actual (str_length_custom \"abc\")", 42, 0);
    check("bstring-provider", "bstr_length", "",
          "let bytes: bstring = (bstr_new \"abc\") set actual (bstr_length bytes) (bstr_free bytes)", 3, 0);
    check("argument-order-once", "abs",
          "fn mark() -> int { set seen (+ seen 1) return (- 0 seen) }\n"
          "shadow mark { assert (< (mark) 0) }",
          "set actual (+ (abs (mark)) (abs (mark)))", 3, 2);
    return 0;
}
