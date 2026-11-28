#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
/* I retire expression arrays without taking my C caller's array ownership. */
#include "../../src/nanolang.h"
#include "../../src/runtime/gc.h"
#include "../../src/runtime/list_int.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;

typedef struct {
    Token *tokens;
    int token_count;
    ASTNode *program;
    Environment *env;
} Fixture;

static Fixture load(const char *source) {
    Fixture f = {0};
    clear_module_cache();
    f.tokens = tokenize(source, &f.token_count); assert(f.tokens);
    f.program = parse_program(f.tokens, f.token_count); assert(f.program);
    f.env = create_environment(); assert(f.env);
    env_set_current_file(f.env, "<ephemeral-array-ownership>");
    typecheck_set_current_file("<ephemeral-array-ownership>");
    assert(type_check(f.program, f.env));
    return f;
}

static void release(Fixture f) {
    free_environment(f.env);
    free_ast(f.program);
    free_tokens(f.tokens, f.token_count);
    clear_module_cache();
    gc_shutdown();
}

static unsigned check(const char *label, const char *source, size_t expected) {
    size_t before = env_static_array_reclaim_count();
    Fixture f = load(source);
    /* I evaluate globals once, through the actual shadow entry point. */
    assert(run_shadow_tests(f.program, f.env, false));
    release(f);
    size_t actual = env_static_array_reclaim_count() - before;
    printf("{\"case\":\"%s\",\"expected_reclaims\":%zu,\"actual_reclaims\":%zu}\n",
           label, expected, actual);
    return actual != expected;
}

static void caller_boundaries(void) {
    Fixture f = load(
        "fn consume(xs: array<int>) -> int { return (at xs 0) }\n"
        "shadow consume { assert (== (consume [9]) 9) }\n"
        "fn make() -> array<int> { return [7] }\n"
        "shadow make { let xs: array<int> = (make) assert (== (at xs 0) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n");
    assert(run_program(f.program, f.env));
    Value input = create_array(VAL_INT, 1, 1);
    ((long long *)input.as.array_val->data)[0] = 9;
    Value result = call_function("consume", &input, 1, f.env);
    assert(result.type == VAL_INT && result.as.int_val == 9);
    Value returned = call_function("make", NULL, 0, f.env);
    assert(returned.type == VAL_ARRAY && returned.as.array_val != input.as.array_val);
    size_t before = env_static_array_reclaim_count();
    release(f);
    assert(env_static_array_reclaim_count() == before);
    assert(input.as.array_val->length == 1 && ((long long *)input.as.array_val->data)[0] == 9);
    assert(returned.as.array_val->length == 1 && ((long long *)returned.as.array_val->data)[0] == 7);
    free(input.as.array_val->data); free(input.as.array_val);
    free(returned.as.array_val->data); free(returned.as.array_val);
    puts("I preserve caller-supplied and C-caller-returned array storage.");
}

static void release_caller_nested(Value outer) {
    assert(outer.type == VAL_ARRAY && outer.as.array_val->element_type == VAL_ARRAY);
    assert(outer.as.array_val->length == 1);
    Value child = ((Value *)outer.as.array_val->data)[0];
    assert(child.type == VAL_ARRAY && child.as.array_val->element_type == VAL_INT);
    assert(child.as.array_val->length == 1 && ((long long *)child.as.array_val->data)[0] == 7);
    free(child.as.array_val->data); free(child.as.array_val);
    free(outer.as.array_val->data); free(outer.as.array_val);
}

static void caller_argument_escapes(void) {
    Fixture f = load(
        "fn install(xs: array<array<int>>) -> int { (array_push xs [7]) return 7 }\n"
        "shadow install { let xs: array<array<int>> = [] assert (== (install xs) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n");
    assert(run_program(f.program, f.env));
    Value outer = create_array(VAL_ARRAY, 0, 0);
    Value result = call_function("install", &outer, 1, f.env);
    assert(result.type == VAL_INT && result.as.int_val == 7);
    size_t before = env_static_array_reclaim_count();
    release(f);
    assert(env_static_array_reclaim_count() == before);
    release_caller_nested(outer);
    puts("I preserve a fresh nested array stored in my C caller's argument after a scalar return.");

    f = load(
        "struct Row { cells: array<array<int>> }\n"
        "fn install(xs: List<Row>) -> int { (array_push (list_Row_get xs 0).cells [7]) return 7 }\n"
        "shadow install { let xs: List<Row> = (list_Row_new) let cells: array<array<int>> = [] "
        "(list_Row_push xs Row { cells: cells }) assert (== (install xs) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n");
    assert(run_program(f.program, f.env));
    outer = create_array(VAL_ARRAY, 0, 0);
    char *fields[] = {"cells"};
    Value row = create_struct("Row", fields, &outer, 1);
    List_int *rows = list_int_new(); assert(rows);
    list_int_push(rows, (int64_t)(intptr_t)row.as.struct_val);
    Value argument = create_int((long long)(intptr_t)rows);
    result = call_function("install", &argument, 1, f.env);
    assert(result.type == VAL_INT && result.as.int_val == 7);
    before = env_static_array_reclaim_count();
    release(f);
    assert(env_static_array_reclaim_count() == before);
    assert(list_int_get(rows, 0) == (int64_t)(intptr_t)row.as.struct_val);
    assert(row.as.struct_val->field_count == 1);
    release_caller_nested(row.as.struct_val->field_values[0]);
    release_struct_carrier_names(row.as.struct_val);
    free(row.as.struct_val->field_values); free(row.as.struct_val);
    list_int_free(rows);
    puts("I preserve fresh array escape through my C caller's typed List<Row> record field.");
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    caller_boundaries();
    caller_argument_escapes();
    unsigned failed = 0;
    failed += check("scalar-argument",
        "let stable: array<int> = [0]\n"
        "fn consume(xs: array<int>) -> int { return (array_length xs) }\n"
        "shadow consume { assert (== (consume stable) 1) }\n"
        "fn main() -> int { return (consume [1, 2]) }\n"
        "shadow main { assert (== (main) 2) }\n", 2);
    failed += check("string-argument",
        "let stable: array<string> = [\"keep\"]\n"
        "fn consume(xs: array<string>) -> int { return (str_length (at xs 0)) }\n"
        "shadow consume { assert (== (consume stable) 4) }\n"
        "fn main() -> int { return (consume [\"shadow\"]) }\n"
        "shadow main { assert (== (main) 6) }\n", 2);
    failed += check("nested-argument",
        "let stable: array<array<int>> = [[0]]\n"
        "fn consume(xs: array<array<int>>) -> int { return (at (at xs 0) 0) }\n"
        "shadow consume { assert (== (consume stable) 0) }\n"
        "fn main() -> int { return (consume [[1], [2]]) }\n"
        "shadow main { assert (== (main) 1) }\n", 5);
    failed += check("empty-local-argument",
        "let stable: array<int> = [0]\n"
        "fn consume(xs: array<int>) -> int { return (array_length xs) }\n"
        "shadow consume { assert (== (consume stable) 1) }\n"
        "fn main() -> int { let empty: array<int> = [] return (consume empty) }\n"
        "shadow main { assert (== (main) 0) }\n", 2);
    failed += check("repeated-arguments",
        "let stable: array<int> = [0]\n"
        "fn consume(xs: array<int>) -> int { return (at xs 0) }\n"
        "shadow consume { assert (== (consume stable) 0) }\n"
        "fn main() -> int { let mut i: int = 0 let mut total: int = 0 while (< i 64) { set total (+ total (consume [i])) set i (+ i 1) } return total }\n"
        "shadow main { assert (== (main) 2016) }\n", 65);
    failed += check("early-return",
        "let stable: array<int> = [0]\n"
        "fn consume(xs: array<int>) -> int { if (> (array_length xs) 0) { return (at xs 0) } return 0 }\n"
        "shadow consume { assert (== (consume stable) 0) }\n"
        "fn main() -> int { return (consume [7]) }\n"
        "shadow main { assert (== (main) 7) }\n", 2);
    failed += check("returned-local-alias",
        "let stable: array<int> = [0]\n"
        "fn same(xs: array<int>) -> array<int> { return xs }\n"
        "shadow same { assert (== (at (same stable) 0) 0) }\n"
        "fn main() -> int { let first: array<int> = (same [5]) let second: array<int> = first return (+ (at first 0) (at second 0)) }\n"
        "shadow main { assert (== (main) 10) }\n", 2);
    failed += check("global-escape-and-replacement",
        "let mut escaped: array<int> = [0]\n"
        "fn install(xs: array<int>) -> int { set escaped xs return (at xs 0) }\n"
        "shadow install { assert (== (install escaped) 0) }\n"
        "fn main() -> int { let value: int = (install [7]) return (+ value (at escaped 0)) }\n"
        "shadow main { assert (== (main) 14) }\n", 2);
    failed += check("shadow-only-argument",
        "fn consume(xs: array<int>) -> int { return (at xs 0) }\n"
        "shadow consume { assert (== (consume [9]) 9) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n", 1);
    failed += check("global-shared-positive",
        "let stable: array<int> = [9]\n"
        "let alias: array<int> = stable\n"
        "fn main() -> int { return (+ (at stable 0) (at alias 0)) }\n"
        "shadow main { assert (== (main) 18) }\n", 1);
    printf("I found %u temporary-array ownership failures.\n", failed);
    return failed ? 1 : 0;
}
