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

    printf("\n✓ All eval tests passed!\n");
    return 0;
}
