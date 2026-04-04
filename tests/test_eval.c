/**
 * test_eval.c — unit tests for the nanolang tree-walking interpreter (eval.c)
 *
 * Exercises run_program(), call_function(), run_shadow_tests(), and
 * the full interpreter pipeline (lex → parse → typecheck → eval) on
 * a variety of nano programs without importing external modules.
 *
 * The goal is to cover eval.c code paths that are not exercised by the
 * standard test suite (which uses the C transpiler, not the interpreter).
 */

#include "../src/nanolang.h"
#include "../src/builtins_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %lld, expected %lld)\n", \
        #a, #b, __LINE__, (long long)(a), (long long)(b)); exit(1); }
#define ASSERT_NOT_NULL(p) \
    if ((p) == NULL) { printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); }

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* Suppress stderr for expected-error paths */
static FILE *s_orig_stderr = NULL;
static void suppress_stderr(void) {
    fflush(stderr);
    s_orig_stderr = stderr;
    stderr = fopen("/dev/null", "w");
}
static void restore_stderr(void) {
    if (stderr && stderr != s_orig_stderr) fclose(stderr);
    stderr = s_orig_stderr;
    s_orig_stderr = NULL;
}

/* ============================================================================
 * Helper: run a nano program through the full interpreter pipeline
 * Returns the environment (caller must free) or NULL on failure.
 * ============================================================================ */
typedef struct {
    Environment *env;
    ASTNode     *program;
    Token       *tokens;
    int          token_count;
} RunCtx;

static bool run_ctx_init(RunCtx *ctx, const char *src) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->tokens = tokenize(src, &ctx->token_count);
    if (!ctx->tokens) return false;

    ctx->program = parse_program(ctx->tokens, ctx->token_count);
    if (!ctx->program) return false;

    clear_module_cache();
    ctx->env = create_environment();

    typecheck_set_current_file("<test>");
    suppress_stderr();
    bool ok = type_check(ctx->program, ctx->env);
    restore_stderr();
    if (!ok) return false;

    suppress_stderr();
    ok = run_program(ctx->program, ctx->env);
    restore_stderr();
    return ok;
}

static void run_ctx_free(RunCtx *ctx) {
    if (ctx->env)     free_environment(ctx->env);
    if (ctx->program) free_ast(ctx->program);
    if (ctx->tokens)  free_tokens(ctx->tokens, ctx->token_count);
    clear_module_cache();
    memset(ctx, 0, sizeof(*ctx));
}

/* ============================================================================
 * Basic evaluation tests
 * ============================================================================ */

void test_eval_integer_arithmetic(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn add(x: int, y: int) -> int { return (+ x y) }\n"
        "fn main() -> int { return (add 3 4) }\n"
    );
    ASSERT(ok);

    Value args[2] = { {.type = VAL_INT, .as.int_val = 10},
                      {.type = VAL_INT, .as.int_val = 20} };
    Value result = call_function("add", args, 2, ctx.env);
    ASSERT_EQ(result.as.int_val, 30);

    run_ctx_free(&ctx);
}

void test_eval_subtraction(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn sub(x: int, y: int) -> int { return (- x y) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);
    Value args[2] = { {.type = VAL_INT, .as.int_val = 100},
                      {.type = VAL_INT, .as.int_val = 37} };
    Value r = call_function("sub", args, 2, ctx.env);
    ASSERT_EQ(r.as.int_val, 63);
    run_ctx_free(&ctx);
}

void test_eval_multiplication(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn mul(x: int, y: int) -> int { return (* x y) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);
    Value args[2] = { {.type = VAL_INT, .as.int_val = 6},
                      {.type = VAL_INT, .as.int_val = 7} };
    Value r = call_function("mul", args, 2, ctx.env);
    ASSERT_EQ(r.as.int_val, 42);
    run_ctx_free(&ctx);
}

void test_eval_division(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn div(x: int, y: int) -> int { return (/ x y) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);
    Value args[2] = { {.type = VAL_INT, .as.int_val = 100},
                      {.type = VAL_INT, .as.int_val = 4} };
    Value r = call_function("div", args, 2, ctx.env);
    ASSERT_EQ(r.as.int_val, 25);
    run_ctx_free(&ctx);
}

void test_eval_modulo(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn mod_fn(x: int, y: int) -> int { return (% x y) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);
    Value args[2] = { {.type = VAL_INT, .as.int_val = 17},
                      {.type = VAL_INT, .as.int_val = 5} };
    Value r = call_function("mod_fn", args, 2, ctx.env);
    ASSERT_EQ(r.as.int_val, 2);
    run_ctx_free(&ctx);
}

void test_eval_boolean_ops(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn land(a: bool, b: bool) -> bool { return (and a b) }\n"
        "fn lor(a: bool, b: bool)  -> bool { return (or a b) }\n"
        "fn lnot(a: bool) -> bool { return (not a) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value t = {.type = VAL_BOOL, .as.bool_val = true};
    Value f = {.type = VAL_BOOL, .as.bool_val = false};

    Value args_tt[2] = {t, t};
    Value r = call_function("land", args_tt, 2, ctx.env);
    ASSERT(r.as.bool_val == true);

    Value args_tf[2] = {t, f};
    r = call_function("land", args_tf, 2, ctx.env);
    ASSERT(r.as.bool_val == false);

    r = call_function("lor", args_tf, 2, ctx.env);
    ASSERT(r.as.bool_val == true);

    Value args_f[1] = {f};
    r = call_function("lnot", args_f, 1, ctx.env);
    ASSERT(r.as.bool_val == true);

    run_ctx_free(&ctx);
}

void test_eval_comparison_ops(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn lt(a: int, b: int) -> bool { return (< a b) }\n"
        "fn lte(a: int, b: int) -> bool { return (<= a b) }\n"
        "fn gt(a: int, b: int) -> bool { return (> a b) }\n"
        "fn gte(a: int, b: int) -> bool { return (>= a b) }\n"
        "fn eq(a: int, b: int) -> bool { return (== a b) }\n"
        "fn ne(a: int, b: int) -> bool { return (!= a b) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value three = {.type = VAL_INT, .as.int_val = 3};
    Value five  = {.type = VAL_INT, .as.int_val = 5};
    Value args[2] = {three, five};

    Value r = call_function("lt", args, 2, ctx.env);
    ASSERT(r.as.bool_val == true);

    r = call_function("gt", args, 2, ctx.env);
    ASSERT(r.as.bool_val == false);

    Value args_eq[2] = {three, three};
    r = call_function("eq", args_eq, 2, ctx.env);
    ASSERT(r.as.bool_val == true);

    r = call_function("ne", args, 2, ctx.env);
    ASSERT(r.as.bool_val == true);

    run_ctx_free(&ctx);
}

void test_eval_if_else(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn abs_val(x: int) -> int {\n"
        "    if (< x 0) { return (* x -1) } else { return x }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value neg5 = {.type = VAL_INT, .as.int_val = -5};
    Value pos3 = {.type = VAL_INT, .as.int_val = 3};

    Value r = call_function("abs_val", &neg5, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 5);

    r = call_function("abs_val", &pos3, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 3);

    run_ctx_free(&ctx);
}

void test_eval_let_bindings(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn compute(x: int) -> int {\n"
        "    let a: int = (* x 2)\n"
        "    let b: int = (+ a 1)\n"
        "    return (+ a b)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value five = {.type = VAL_INT, .as.int_val = 5};
    Value r = call_function("compute", &five, 1, ctx.env);
    /* a = 10, b = 11, return 10+11 = 21 */
    ASSERT_EQ(r.as.int_val, 21);

    run_ctx_free(&ctx);
}

void test_eval_recursion(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn factorial(n: int) -> int {\n"
        "    if (<= n 1) { return 1 } else { return (* n (factorial (- n 1))) }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value six = {.type = VAL_INT, .as.int_val = 6};
    Value r = call_function("factorial", &six, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 720);

    run_ctx_free(&ctx);
}

void test_eval_while_loop(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn sum_to(n: int) -> int {\n"
        "    let mut total: int = 0\n"
        "    let mut i: int = 1\n"
        "    while (<= i n) {\n"
        "        set total (+ total i)\n"
        "        set i (+ i 1)\n"
        "    }\n"
        "    return total\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value ten = {.type = VAL_INT, .as.int_val = 10};
    Value r = call_function("sum_to", &ten, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 55);

    run_ctx_free(&ctx);
}

void test_eval_for_loop(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn sum_range(n: int) -> int {\n"
        "    let mut total: int = 0\n"
        "    for i in (range 0 n) {\n"
        "        set total (+ total i)\n"
        "    }\n"
        "    return total\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value five = {.type = VAL_INT, .as.int_val = 5};
    Value r = call_function("sum_range", &five, 1, ctx.env);
    /* range 0..5 = 0+1+2+3+4 = 10 */
    ASSERT_EQ(r.as.int_val, 10);

    run_ctx_free(&ctx);
}

void test_eval_string_ops(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn greet(name: string) -> string {\n"
        "    return (str_concat \"Hello, \" name)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value name = {.type = VAL_STRING, .as.string_val = "world"};
    Value r = call_function("greet", &name, 1, ctx.env);
    ASSERT(r.type == VAL_STRING);
    ASSERT(strcmp(r.as.string_val, "Hello, world") == 0);

    run_ctx_free(&ctx);
}

void test_eval_string_length(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn slen(s: string) -> int { return (str_length s) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value s = {.type = VAL_STRING, .as.string_val = "nanolang"};
    Value r = call_function("slen", &s, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 8);

    run_ctx_free(&ctx);
}

void test_eval_nested_calls(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn double(x: int) -> int { return (* x 2) }\n"
        "fn quad(x: int)   -> int { return (double (double x)) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value three = {.type = VAL_INT, .as.int_val = 3};
    Value r = call_function("quad", &three, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 12);

    run_ctx_free(&ctx);
}

void test_eval_multiple_return_paths(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn classify(n: int) -> string {\n"
        "    if (< n 0) { return \"negative\" } else {\n"
        "        if (== n 0) { return \"zero\" } else {\n"
        "            return \"positive\"\n"
        "        }\n"
        "    }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value neg = {.type = VAL_INT, .as.int_val = -5};
    Value zer = {.type = VAL_INT, .as.int_val = 0};
    Value pos = {.type = VAL_INT, .as.int_val = 7};

    Value r = call_function("classify", &neg, 1, ctx.env);
    ASSERT(strcmp(r.as.string_val, "negative") == 0);

    r = call_function("classify", &zer, 1, ctx.env);
    ASSERT(strcmp(r.as.string_val, "zero") == 0);

    r = call_function("classify", &pos, 1, ctx.env);
    ASSERT(strcmp(r.as.string_val, "positive") == 0);

    run_ctx_free(&ctx);
}

void test_eval_shadow_tests(void) {
    RunCtx ctx;
    /* run_shadow_tests exercises the shadow test runner in eval.c */
    bool ok = run_ctx_init(&ctx,
        "fn square(x: int) -> int { return (* x x) }\n"
        "fn main() -> int { return (square 5) }\n"
        "shadow square {\n"
        "    assert (== (square 3) 9)\n"
        "    assert (== (square 0) 0)\n"
        "    assert (== (square -4) 16)\n"
        "}\n"
    );
    ASSERT(ok);

    suppress_stderr();
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    restore_stderr();
    ASSERT(shadows_ok);

    run_ctx_free(&ctx);
}

void test_eval_float_arithmetic(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn fadd(x: float, y: float) -> float { return (+ x y) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value a = {.type = VAL_FLOAT, .as.float_val = 1.5};
    Value b = {.type = VAL_FLOAT, .as.float_val = 2.5};
    Value args[2] = {a, b};
    Value r = call_function("fadd", args, 2, ctx.env);
    ASSERT(r.type == VAL_FLOAT);
    ASSERT(r.as.float_val > 3.9 && r.as.float_val < 4.1);

    run_ctx_free(&ctx);
}

void test_eval_string_comparison(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn same(a: string, b: string) -> bool { return (== a b) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value hello1 = {.type = VAL_STRING, .as.string_val = "hello"};
    Value hello2 = {.type = VAL_STRING, .as.string_val = "hello"};
    Value world  = {.type = VAL_STRING, .as.string_val = "world"};

    Value args_eq[2] = {hello1, hello2};
    Value r = call_function("same", args_eq, 2, ctx.env);
    ASSERT(r.as.bool_val == true);

    Value args_ne[2] = {hello1, world};
    r = call_function("same", args_ne, 2, ctx.env);
    ASSERT(r.as.bool_val == false);

    run_ctx_free(&ctx);
}

void test_eval_int_to_string(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn i2s(x: int) -> string { return (int_to_string x) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value n = {.type = VAL_INT, .as.int_val = 42};
    Value r = call_function("i2s", &n, 1, ctx.env);
    ASSERT(r.type == VAL_STRING);
    ASSERT(strcmp(r.as.string_val, "42") == 0);

    run_ctx_free(&ctx);
}

void test_eval_string_to_int(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn s2i(s: string) -> int { return (string_to_int s) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value s = {.type = VAL_STRING, .as.string_val = "123"};
    Value r = call_function("s2i", &s, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 123);

    run_ctx_free(&ctx);
}

void test_eval_min_max(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn get_min(a: int, b: int) -> int { return (min a b) }\n"
        "fn get_max(a: int, b: int) -> int { return (max a b) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value three = {.type = VAL_INT, .as.int_val = 3};
    Value seven = {.type = VAL_INT, .as.int_val = 7};
    Value args[2] = {three, seven};

    Value r = call_function("get_min", args, 2, ctx.env);
    ASSERT_EQ(r.as.int_val, 3);

    r = call_function("get_max", args, 2, ctx.env);
    ASSERT_EQ(r.as.int_val, 7);

    run_ctx_free(&ctx);
}

void test_eval_abs(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn get_abs(x: int) -> int { return (abs x) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value neg7 = {.type = VAL_INT, .as.int_val = -7};
    Value r = call_function("get_abs", &neg7, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 7);

    run_ctx_free(&ctx);
}

void test_eval_program_with_top_level_let(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "let PI: float = 3.14159\n"
        "fn area(r: float) -> float { return (* PI (* r r)) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value r = {.type = VAL_FLOAT, .as.float_val = 2.0};
    Value result = call_function("area", &r, 1, ctx.env);
    ASSERT(result.type == VAL_FLOAT);
    ASSERT(result.as.float_val > 12.5 && result.as.float_val < 12.6);

    run_ctx_free(&ctx);
}

void test_eval_negative_zero(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn negate(x: int) -> int { return (* x -1) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    Value zero = {.type = VAL_INT, .as.int_val = 0};
    Value r = call_function("negate", &zero, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 0);

    run_ctx_free(&ctx);
}

/* ============================================================================
 * Additional tests for broader coverage
 * ============================================================================ */

void test_eval_struct_creation_and_access(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "struct Point {\n"
        "    x: int,\n"
        "    y: int\n"
        "}\n"
        "fn make_point(x: int, y: int) -> Point {\n"
        "    return Point { x: x, y: y }\n"
        "}\n"
        "fn get_x(p: Point) -> int { return p.x }\n"
        "fn get_y(p: Point) -> int { return p.y }\n"
        "fn main() -> int { return 0 }\n"
        "shadow make_point {\n"
        "    assert (== ((make_point 3 4).x) 3)\n"
        "    assert (== ((make_point 3 4).y) 4)\n"
        "}\n"
        "shadow get_x { assert (== (get_x (make_point 7 8)) 7) }\n"
        "shadow get_y { assert (== (get_y (make_point 7 8)) 8) }\n"
    );
    ASSERT(ok);

    /* make_point should set fields correctly */
    Value three = {.type = VAL_INT, .as.int_val = 3};
    Value four = {.type = VAL_INT, .as.int_val = 4};
    Value args[2] = {three, four};
    Value pt = call_function("make_point", args, 2, ctx.env);
    ASSERT(pt.type == VAL_STRUCT);

    run_ctx_free(&ctx);
}

void test_eval_struct_pythagorean(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "struct Point {\n"
        "    x: int,\n"
        "    y: int\n"
        "}\n"
        "fn make_point(x: int, y: int) -> Point {\n"
        "    return Point { x: x, y: y }\n"
        "}\n"
        "fn distance_sq(p: Point) -> int {\n"
        "    return (+ (* p.x p.x) (* p.y p.y))\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow make_point { assert (== ((make_point 3 4).x) 3) }\n"
        "shadow distance_sq { assert (== (distance_sq (make_point 3 4)) 25) }\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_match_expression(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn classify(x: int) -> string {\n"
        "    match x {\n"
        "        0 -> \"zero\",\n"
        "        1 -> \"one\",\n"
        "        _ -> \"other\"\n"
        "    }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow classify {\n"
        "    assert (== (classify 0) \"zero\")\n"
        "    assert (== (classify 1) \"one\")\n"
        "    assert (== (classify 99) \"other\")\n"
        "}\n"
    );
    ASSERT(ok);

    Value zero = {.type = VAL_INT, .as.int_val = 0};
    Value r = call_function("classify", &zero, 1, ctx.env);
    ASSERT(r.type == VAL_STRING);

    run_ctx_free(&ctx);
}

void test_eval_list_iteration(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn sum_list(lst: List<int>) -> int {\n"
        "    let mut total: int = 0\n"
        "    for x in lst {\n"
        "        set total (+ total x)\n"
        "    }\n"
        "    return total\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow sum_list { assert (== (sum_list [1, 2, 3, 4, 5]) 15) }\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_string_builtins(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn first_char(s: string) -> string { return (str_substring s 0 1) }\n"
        "fn char_code(s: string) -> int { return (char_at s 0) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow first_char { assert (== (first_char \"hello\") \"h\") }\n"
        "shadow char_code { assert (== (char_code \"A\") 65) }\n"
    );
    ASSERT(ok);

    Value hello = {.type = VAL_STRING, .as.string_val = "hello"};
    Value r = call_function("first_char", &hello, 1, ctx.env);
    ASSERT(r.type == VAL_STRING);

    run_ctx_free(&ctx);
}

void test_eval_higher_order_returns(void) {
    /* Test functions that call other functions indirectly */
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn apply_twice(x: int) -> int {\n"
        "    let r1: int = (* x 2)\n"
        "    let r2: int = (* r1 2)\n"
        "    return r2\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow apply_twice {\n"
        "    assert (== (apply_twice 3) 12)\n"
        "    assert (== (apply_twice 5) 20)\n"
        "}\n"
    );
    ASSERT(ok);

    Value v = {.type = VAL_INT, .as.int_val = 3};
    Value r = call_function("apply_twice", &v, 1, ctx.env);
    ASSERT_EQ(r.as.int_val, 12);

    run_ctx_free(&ctx);
}

void test_eval_mutual_recursion(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn is_even(n: int) -> bool {\n"
        "    if (== n 0) { return true } else { return (is_odd (- n 1)) }\n"
        "}\n"
        "fn is_odd(n: int) -> bool {\n"
        "    if (== n 0) { return false } else { return (is_even (- n 1)) }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow is_even {\n"
        "    assert (is_even 4)\n"
        "    assert (not (is_even 3))\n"
        "}\n"
        "shadow is_odd {\n"
        "    assert (is_odd 3)\n"
        "    assert (not (is_odd 4))\n"
        "}\n"
    );
    ASSERT(ok);

    Value four = {.type = VAL_INT, .as.int_val = 4};
    Value r = call_function("is_even", &four, 1, ctx.env);
    ASSERT(r.type == VAL_BOOL);
    ASSERT(r.as.bool_val == true);

    run_ctx_free(&ctx);
}

void test_eval_nested_match(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn sign(x: int) -> string {\n"
        "    if (> x 0) {\n"
        "        return \"positive\"\n"
        "    } else {\n"
        "        if (< x 0) {\n"
        "            return \"negative\"\n"
        "        } else {\n"
        "            return \"zero\"\n"
        "        }\n"
        "    }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow sign {\n"
        "    assert (== (sign 5) \"positive\")\n"
        "    assert (== (sign -3) \"negative\")\n"
        "    assert (== (sign 0) \"zero\")\n"
        "}\n"
    );
    ASSERT(ok);

    Value five = {.type = VAL_INT, .as.int_val = 5};
    Value r = call_function("sign", &five, 1, ctx.env);
    ASSERT(r.type == VAL_STRING);

    run_ctx_free(&ctx);
}

void test_eval_string_contains(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn has_prefix(s: string, prefix: string) -> bool {\n"
        "    return (== (str_substring s 0 (str_length prefix)) prefix)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow has_prefix {\n"
        "    assert (has_prefix \"hello world\" \"hello\")\n"
        "    assert (not (has_prefix \"hello world\" \"world\"))\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_multiple_lets(void) {
    /* Multiple let bindings, some mutable, some immutable */
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn compute(n: int) -> int {\n"
        "    let a: int = (* n 2)\n"
        "    let b: int = (* a 3)\n"
        "    let c: int = (+ a b)\n"
        "    let mut acc: int = 0\n"
        "    set acc (+ acc c)\n"
        "    set acc (+ acc 1)\n"
        "    return acc\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow compute {\n"
        "    assert (== (compute 2) 17)\n"
        "}\n"
    );
    ASSERT(ok);

    Value two = {.type = VAL_INT, .as.int_val = 2};
    Value r = call_function("compute", &two, 1, ctx.env);
    /* n=2: a=4, b=12, c=16, acc=0+16+1=17 */
    ASSERT_EQ(r.as.int_val, 17);

    run_ctx_free(&ctx);
}

void test_builtins_registry_lookup(void) {
    /* Test the builtins registry directly */
    ASSERT(builtin_is_known("print"));
    ASSERT(builtin_is_known("str_length"));
    ASSERT(builtin_is_known("map_new"));
    ASSERT(!builtin_is_known("nonexistent_function_xyz"));

    const BuiltinEntry *e = builtin_find("str_length");
    ASSERT(e != NULL);

    const char *c_name = builtin_c_name("str_length");
    ASSERT(c_name != NULL);
}

void test_eval_hashmap_operations(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn test_map() -> int {\n"
        "    let mut m: HashMap<string, int> = (map_new)\n"
        "    (map_put m \"a\" 10)\n"
        "    (map_put m \"b\" 20)\n"
        "    let va: int = (map_get m \"a\")\n"
        "    let vb: int = (map_get m \"b\")\n"
        "    let sz: int = (map_size m)\n"
        "    return (+ va vb)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow test_map { assert (== (test_map) 30) }\n"
    );
    ASSERT(ok);

    Value r = call_function("test_map", NULL, 0, ctx.env);
    ASSERT_EQ(r.as.int_val, 30);

    run_ctx_free(&ctx);
}

void test_eval_hashmap_has_and_remove(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn test_has() -> bool {\n"
        "    let mut m: HashMap<string, int> = (map_new)\n"
        "    (map_put m \"x\" 99)\n"
        "    let before: bool = (map_has m \"x\")\n"
        "    (map_remove m \"x\")\n"
        "    let after: bool = (map_has m \"x\")\n"
        "    return (and before (not after))\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow test_has { assert (test_has) }\n"
    );
    ASSERT(ok);

    Value r = call_function("test_has", NULL, 0, ctx.env);
    ASSERT(r.type == VAL_BOOL);
    ASSERT(r.as.bool_val == true);

    run_ctx_free(&ctx);
}

void test_eval_union_types(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "union Status {\n"
        "    Ok {},\n"
        "    Error {}\n"
        "}\n"
        "fn test_ok() -> int {\n"
        "    let s = Status.Ok {}\n"
        "    return match s {\n"
        "        Ok(x) => 1,\n"
        "        Error(e) => 0\n"
        "    }\n"
        "}\n"
        "fn test_err() -> int {\n"
        "    let s = Status.Error {}\n"
        "    return match s {\n"
        "        Ok(x) => 0,\n"
        "        Error(e) => 1\n"
        "    }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow test_ok { assert (== (test_ok) 1) }\n"
        "shadow test_err { assert (== (test_err) 1) }\n"
    );
    ASSERT(ok);

    Value r = call_function("test_ok", NULL, 0, ctx.env);
    ASSERT_EQ(r.as.int_val, 1);

    run_ctx_free(&ctx);
}

void test_eval_tuple_types(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn make_pair(a: int, b: int) -> (int, int) {\n"
        "    return (a, b)\n"
        "}\n"
        "fn first(p: (int, int)) -> int { return p.0 }\n"
        "fn second(p: (int, int)) -> int { return p.1 }\n"
        "fn main() -> int { return 0 }\n"
        "shadow make_pair {\n"
        "    assert (== ((make_pair 3 4).0) 3)\n"
        "    assert (== ((make_pair 3 4).1) 4)\n"
        "}\n"
        "shadow first { assert (== (first (make_pair 5 6)) 5) }\n"
        "shadow second { assert (== (second (make_pair 5 6)) 6) }\n"
    );
    ASSERT(ok);

    Value three = {.type = VAL_INT, .as.int_val = 3};
    Value four = {.type = VAL_INT, .as.int_val = 4};
    Value args[2] = {three, four};
    Value pair = call_function("make_pair", args, 2, ctx.env);
    ASSERT(pair.type == VAL_TUPLE);

    run_ctx_free(&ctx);
}

void test_eval_not_operator(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn negate_bool(b: bool) -> bool { return (not b) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow negate_bool {\n"
        "    assert (negate_bool false)\n"
        "    assert (not (negate_bool true))\n"
        "}\n"
    );
    ASSERT(ok);

    Value t = {.type = VAL_BOOL, .as.bool_val = true};
    Value r = call_function("negate_bool", &t, 1, ctx.env);
    ASSERT(r.type == VAL_BOOL);
    ASSERT(r.as.bool_val == false);

    run_ctx_free(&ctx);
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== Interpreter (eval.c) Tests ===\n");
    TEST(eval_integer_arithmetic);
    TEST(eval_subtraction);
    TEST(eval_multiplication);
    TEST(eval_division);
    TEST(eval_modulo);
    TEST(eval_boolean_ops);
    TEST(eval_comparison_ops);
    TEST(eval_if_else);
    TEST(eval_let_bindings);
    TEST(eval_recursion);
    TEST(eval_while_loop);
    TEST(eval_for_loop);
    TEST(eval_string_ops);
    TEST(eval_string_length);
    TEST(eval_nested_calls);
    TEST(eval_multiple_return_paths);
    TEST(eval_shadow_tests);
    TEST(eval_float_arithmetic);
    TEST(eval_string_comparison);
    TEST(eval_int_to_string);
    TEST(eval_string_to_int);
    TEST(eval_min_max);
    TEST(eval_abs);
    TEST(eval_program_with_top_level_let);
    TEST(eval_negative_zero);
    TEST(eval_struct_creation_and_access);
    TEST(eval_struct_pythagorean);
    TEST(eval_match_expression);
    TEST(eval_list_iteration);
    TEST(eval_string_builtins);
    TEST(eval_higher_order_returns);
    TEST(eval_mutual_recursion);
    TEST(eval_nested_match);
    TEST(eval_string_contains);
    TEST(eval_multiple_lets);
    TEST(eval_not_operator);
    TEST(builtins_registry_lookup);
    TEST(eval_hashmap_operations);
    TEST(eval_hashmap_has_and_remove);
    TEST(eval_union_types);
    TEST(eval_tuple_types);

    printf("\n✓ All eval tests passed!\n");
    return 0;
}
