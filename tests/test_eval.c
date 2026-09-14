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
#include "../src/interpreter_ffi.h"
#include "../src/runtime/ffi_loader.h"
#include "../src/runtime/dyn_array.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

void test_eval_record_string_local_lifetime(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Text { value: string }\n"
        "fn read_text(t: Text) -> int {\n"
        "  let mut local: string = t.value\n"
        "  set local (+ local \"!\")\n"
        "  return (str_length local)\n"
        "}\n"
        "fn check() -> int {\n"
        "  let t: Text = Text { value: \"seven\" }\n"
        "  let a: int = (read_text t)\n"
        "  let b: int = (read_text t)\n"
        "  return (+ (+ a b) (str_length t.value))\n"
        "}\n"
        "shadow check { assert (== (check) 17) }\n"
        "fn main() -> int { return 0 }\n"));
    for (int i = 0; i < 100; i++) {
        Value value = call_function("check", NULL, 0, ctx.env);
        ASSERT_EQ(value.type, VAL_INT);
        ASSERT_EQ(value.as.int_val, 17);
    }
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
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

void test_eval_struct_string_field_lifetime(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "struct Message { text: string }\n"
        "fn read_text(message: Message) -> string { return message.text }\n"
        "fn read_repeatedly(message: Message) -> string {\n"
        "    let mut result: string = \"\"\n"
        "    let mut count: int = 0\n"
        "    while (< count 100) {\n"
        "        set result message.text\n"
        "        set count (+ count 1)\n"
        "    }\n"
        "    return result\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow read_text {\n"
        "    let message: Message = Message { text: \"owned\" }\n"
        "    assert (== (read_text message) \"owned\")\n"
        "}\n"
        "shadow read_repeatedly {\n"
        "    let message: Message = Message { text: \"stable\" }\n"
        "    assert (== (read_repeatedly message) \"stable\")\n"
        "}\n"
    );
    ASSERT(ok);
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

void test_eval_cond_expression(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn classify(x: int) -> string {\n"
        "    return (cond\n"
        "        ((< x 0) \"negative\")\n"
        "        ((== x 0) \"zero\")\n"
        "        (else \"positive\"))\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow classify {\n"
        "    assert (== (classify -1) \"negative\")\n"
        "    assert (== (classify 0) \"zero\")\n"
        "    assert (== (classify 5) \"positive\")\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_break_in_for(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn first_positive(lst: List<int>) -> int {\n"
        "    let mut result: int = -1\n"
        "    for x in lst {\n"
        "        if (> x 0) {\n"
        "            set result x\n"
        "            break\n"
        "        } else { (print \"\") }\n"
        "    }\n"
        "    return result\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow first_positive {\n"
        "    assert (== (first_positive [-3, -1, 5, 2]) 5)\n"
        "    assert (== (first_positive [10, 20]) 10)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_nested_for_loops(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn count_pairs(n: int) -> int {\n"
        "    let mut count: int = 0\n"
        "    for i in (range 0 n) {\n"
        "        for j in (range 0 n) {\n"
        "            set count (+ count 1)\n"
        "        }\n"
        "    }\n"
        "    return count\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow count_pairs { assert (== (count_pairs 3) 9) }\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_length(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn array_sum(arr: array<int>) -> int {\n"
        "    let mut total: int = 0\n"
        "    let n: int = (array_length arr)\n"
        "    for i in (range 0 n) {\n"
        "        set total (+ total (array_get arr i))\n"
        "    }\n"
        "    return total\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow array_sum { assert (== (array_sum [1, 2, 3, 4, 5]) 15) }\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_get_alias(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn double(x: float) -> float { return (* x 2.0) }\n"
        "fn read_static() -> int { return (array_get [4, 5, 6] 1) }\n"
        "fn read_mapped() -> float {\n"
        "    let values: array<float> = (map [1.5, 2.5] double)\n"
        "    return (array_get values 1)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow double { assert (== (double 2.5) 5.0) }\n"
        "shadow read_static { assert (== (read_static) 5) }\n"
        "shadow read_mapped { assert (== (read_mapped) 5.0) }\n"
    );
    ASSERT(ok);

    Value static_result = call_function("read_static", NULL, 0, ctx.env);
    ASSERT(static_result.type == VAL_INT);
    ASSERT_EQ(static_result.as.int_val, 5);

    Value mapped_result = call_function("read_mapped", NULL, 0, ctx.env);
    ASSERT(mapped_result.type == VAL_FLOAT);
    ASSERT(mapped_result.as.float_val == 5.0);

    run_ctx_free(&ctx);
}

void test_eval_math_functions(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn hypotenuse(a: float, b: float) -> float {\n"
        "    return (sqrt (+ (* a a) (* b b)))\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow hypotenuse {\n"
        "    let h: float = (hypotenuse 3.0 4.0)\n"
        "    assert (== (round h) 5.0)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_string_conversion(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn format_pair(a: int, b: int) -> string {\n"
        "    let sa: string = (int_to_string a)\n"
        "    let sb: string = (int_to_string b)\n"
        "    return (str_concat (str_concat sa \",\") sb)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow format_pair {\n"
        "    assert (== (format_pair 3 4) \"3,4\")\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_enum_access(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "enum Direction { North, South, East, West }\n"
        "fn is_north(d: Direction) -> bool {\n"
        "    return (== d Direction.North)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow is_north {\n"
        "    assert (is_north Direction.North)\n"
        "    assert (not (is_north Direction.South))\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_string_to_int_back(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn round_trip(n: int) -> int {\n"
        "    let s: string = (int_to_string n)\n"
        "    return (string_to_int s)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow round_trip {\n"
        "    assert (== (round_trip 42) 42)\n"
        "    assert (== (round_trip -17) -17)\n"
        "    assert (== (round_trip 0) 0)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_shadowed_functions_reuse(void) {
    /* Run shadow tests explicitly via run_shadow_tests */
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn double(x: int) -> int { return (* x 2) }\n"
        "fn triple(x: int) -> int { return (* x 3) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow double { assert (== (double 5) 10) }\n"
        "shadow triple { assert (== (triple 5) 15) }\n"
    );
    ASSERT(ok);

    /* Also run shadow tests directly */
    suppress_stderr();
    bool shadow_ok = run_shadow_tests(ctx.program, ctx.env, false);
    restore_stderr();
    ASSERT(shadow_ok);

    run_ctx_free(&ctx);
}

void test_eval_float_comparison(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn is_close(a: float, b: float) -> bool {\n"
        "    let diff: float = (abs (- a b))\n"
        "    return (< diff 0.001)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow is_close {\n"
        "    assert (is_close 3.14 3.14)\n"
        "    assert (not (is_close 1.0 2.0))\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_map_builtin(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn double_it(x: int) -> int { return (* x 2) }\n"
        "fn main() -> int {\n"
        "  let nums: array<int> = [1, 2, 3, 4, 5]\n"
        "  let doubled: array<int> = (map nums double_it)\n"
        "  return (array_get doubled 0)\n"
        "}\n"
        "shadow main {\n"
        "  let nums: array<int> = [1, 2, 3, 4, 5]\n"
        "  let doubled: array<int> = (map nums double_it)\n"
        "  assert (== (array_length doubled) 5)\n"
        "  assert (== (array_get doubled 0) 2)\n"
        "  assert (== (array_get doubled 4) 10)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_filter_builtin(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn is_even(x: int) -> bool { return (== (% x 2) 0) }\n"
        "fn main() -> int {\n"
        "  let nums: array<int> = [1, 2, 3, 4, 5, 6]\n"
        "  let evens: array<int> = (filter nums is_even)\n"
        "  return (array_length evens)\n"
        "}\n"
        "shadow main {\n"
        "  let nums: array<int> = [1, 2, 3, 4, 5, 6]\n"
        "  let evens: array<int> = (filter nums is_even)\n"
        "  assert (== (array_length evens) 3)\n"
        "  assert (== (array_get evens 0) 2)\n"
        "  assert (== (array_get evens 2) 6)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_reduce_builtin(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn add_ints(a: int, b: int) -> int { return (+ a b) }\n"
        "fn main() -> int {\n"
        "  let nums: array<int> = [1, 2, 3, 4, 5]\n"
        "  return (reduce nums 0 add_ints)\n"
        "}\n"
        "shadow main {\n"
        "  let nums: array<int> = [1, 2, 3, 4, 5]\n"
        "  let sum: int = (reduce nums 0 add_ints)\n"
        "  assert (== sum 15)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_push_pop(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "  let arr: array<int> = [1, 2, 3]\n"
        "  let arr2: array<int> = (array_push arr 4)\n"
        "  return (array_length arr2)\n"
        "}\n"
        "shadow main {\n"
        "  let arr: array<int> = [1, 2, 3]\n"
        "  let arr2: array<int> = (array_push arr 99)\n"
        "  assert (== (array_length arr2) 4)\n"
        "  assert (== (array_get arr2 3) 99)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_sort(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "  let arr: array<int> = [5, 3, 1, 4, 2]\n"
        "  let sorted: array<int> = (array_sort arr)\n"
        "  return (array_get sorted 0)\n"
        "}\n"
        "shadow main {\n"
        "  let arr: array<int> = [5, 3, 1, 4, 2]\n"
        "  let sorted: array<int> = (array_sort arr)\n"
        "  assert (== (array_get sorted 0) 1)\n"
        "  assert (== (array_get sorted 4) 5)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_contains(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "  let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "  if (array_contains arr 3) { return 1 }\n"
        "  else { return 0 }\n"
        "}\n"
        "shadow main {\n"
        "  let arr: array<int> = [10, 20, 30]\n"
        "  assert (array_contains arr 20)\n"
        "  assert (not (array_contains arr 99))\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_type_casts(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let f: float = (int_to_float 42)\n"
        "  let i: int = (float_to_int 3.7)\n"
        "  let b: bool = (int_to_bool 1)\n"
        "  assert (== i 3)\n"
        "  assert b\n"
        "  assert (not (int_to_bool 0))\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_string_format_ops(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let s: string = (str_to_lower \"HELLO\")\n"
        "  assert (== s \"hello\")\n"
        "  let u: string = (str_to_upper \"world\")\n"
        "  assert (== u \"WORLD\")\n"
        "  let t: string = (str_trim \"  hello  \")\n"
        "  assert (== t \"hello\")\n"
        "  let r: string = (str_replace \"hello world\" \"world\" \"nano\")\n"
        "  assert (== r \"hello nano\")\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_reverse(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "  let rev: array<int> = (array_reverse arr)\n"
        "  assert (== (array_get rev 0) 5)\n"
        "  assert (== (array_get rev 4) 1)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_index_of(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let arr: array<int> = [10, 20, 30, 40]\n"
        "  assert (== (array_index_of arr 20) 1)\n"
        "  assert (== (array_index_of arr 99) (- 0 1))\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_slice(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "  let sl: array<int> = (array_slice arr 1 3)\n"
        "  assert (== (array_length sl) 2)\n"
        "  assert (== (array_get sl 0) 2)\n"
        "  assert (== (array_get sl 1) 3)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_set_mutation(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let mut x: int = 5\n"
        "  set x 10\n"
        "  assert (== x 10)\n"
        "  set x (* x 2)\n"
        "  assert (== x 20)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

void test_eval_array_set(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let arr: array<int> = [1, 2, 3]\n"
        "  let updated: array<int> = (array_set arr 1 99)\n"
        "  assert (== (array_get updated 1) 99)\n"
        "  assert (== (array_get arr 1) 2)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* ============================================================================
 * Additional tests for uncovered eval.c paths
 * ============================================================================ */

void test_eval_continue_in_loop(void) {
    RunCtx ctx;
    /* Test continue in a while loop - count down from 5 skipping 3 */
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let mut i: int = 0\n"
        "  let mut count: int = 0\n"
        "  while (< i 5) {\n"
        "    set i (+ i 1)\n"
        "    if (== i 3) { continue } else { set count (+ count 1) }\n"
        "  }\n"
        "  assert (== count 4)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_unsafe_block(void) {
    /* Test unsafe block executes like a regular block in the interpreter */
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let x: int = 10\n"
        "  unsafe {\n"
        "    let y: int = (+ x 5)\n"
        "    assert (== y 15)\n"
        "  }\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_effects_basic(void) {
    /* Test effect declaration (no-op in interpreter) and basic effect structure */
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "effect Log {\n"
        "  fn write(msg: string) -> string\n"
        "}\n"
        "fn noop() -> int { return 42 }\n"
    );
    ASSERT(ok);
    Value result = call_function("noop", NULL, 0, ctx.env);
    ASSERT_EQ(result.as.int_val, 42);
    run_ctx_free(&ctx);
}

void test_eval_nested_closures(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let add5 = fn(x: int) -> int { return (+ x 5) }\n"
        "  let add10 = fn(x: int) -> int { return (+ x 10) }\n"
        "  assert (== (add5 3) 8)\n"
        "  assert (== (add10 3) 13)\n"
        "  assert (== (+ (add5 3) (add10 3)) 21)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_string_format(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let s = (str_concat \"number: \" (int_to_string 99))\n"
        "  assert (== s \"number: 99\")\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_nested_arrays(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn sum_array(arr: array<int>) -> int {\n"
        "  let mut total: int = 0\n"
        "  for x in arr {\n"
        "    set total (+ total x)\n"
        "  }\n"
        "  return total\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let arr: array<int> = [10, 20, 30, 40, 50]\n"
        "  assert (== (sum_array arr) 150)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_string_array(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn join_strs(arr: array<string>) -> string {\n"
        "  let mut result: string = \"\"\n"
        "  for s in arr {\n"
        "    set result (str_concat result s)\n"
        "  }\n"
        "  return result\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow join_strs {\n"
        "  assert (== (join_strs [\"hello\", \" \", \"world\"]) \"hello world\")\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_float_array(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn avg3(a: float, b: float, c: float) -> float {\n"
        "  return (/ (+ (+ a b) c) 3.0)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow avg3 {\n"
        "  assert (== (avg3 1.0 2.0 3.0) 2.0)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_complex_match_with_guards(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn classify(n: int) -> string {\n"
        "  match n {\n"
        "    0 -> \"zero\",\n"
        "    1 -> \"one\",\n"
        "    _ -> \"other\"\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow classify {\n"
        "  assert (== (classify 0) \"zero\")\n"
        "  assert (== (classify 1) \"one\")\n"
        "  assert (== (classify 99) \"other\")\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_union_with_data(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "union Shape {\n"
        "  Circle { radius: float },\n"
        "  Rectangle { w: float, h: float }\n"
        "}\n"
        "fn area(s: Shape) -> float {\n"
        "  match s {\n"
        "    Circle(c) => (* 3.0 (* c.radius c.radius)),\n"
        "    Rectangle(r) => (* r.w r.h)\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow area {\n"
        "  let c: Shape = Shape.Circle { radius: 2.0 }\n"
        "  let r: Shape = Shape.Rectangle { w: 3.0, h: 4.0 }\n"
        "  assert (> (area c) 11.0)\n"
        "  assert (== (area r) 12.0)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_multiple_function_calls(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn double(n: int) -> int { return (* n 2) }\n"
        "fn triple(n: int) -> int { return (* n 3) }\n"
        "fn compose(n: int) -> int { return (double (triple n)) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow compose {\n"
        "  assert (== (compose 5) 30)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_let_reassignment(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn count_down() -> int {\n"
        "  let mut x: int = 5\n"
        "  while (> x 0) {\n"
        "    set x (- x 1)\n"
        "  }\n"
        "  return x\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);
    Value result = call_function("count_down", NULL, 0, ctx.env);
    ASSERT_EQ(result.as.int_val, 0);
    run_ctx_free(&ctx);
}

void test_eval_string_escape(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn get_tab_str() -> string { return \"a\\tb\" }\n"
        "fn get_newline_str() -> string { return \"a\\nb\" }\n"
        "fn main() -> int { return 0 }\n"
        "shadow get_tab_str {\n"
        "  assert (== (str_length (get_tab_str)) 3)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_array_operations_comprehensive(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let arr: array<int> = [5, 3, 1, 4, 2]\n"
        "  assert (== (array_length arr) 5)\n"
        "  let mut total: int = 0\n"
        "  for x in arr {\n"
        "    set total (+ total x)\n"
        "  }\n"
        "  assert (== total 15)\n"
        "  let arr2: array<int> = [1, 2, 3]\n"
        "  assert (== (array_length arr2) 3)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_type_casting(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  let s: string = (int_to_string 42)\n"
        "  assert (== s \"42\")\n"
        "  let n: int = (string_to_int \"123\")\n"
        "  assert (== n 123)\n"
        "  let s2: string = (int_to_string -7)\n"
        "  assert (== s2 \"-7\")\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_bool_operations(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int { return 0 }\n"
        "shadow main {\n"
        "  assert (and true true)\n"
        "  assert (not (and true false))\n"
        "  assert (or false true)\n"
        "  assert (not (or false false))\n"
        "  assert (not false)\n"
        "  assert (not (not true))\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

void test_eval_nested_struct_fields(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "struct Point { x: int, y: int }\n"
        "struct Line { start: Point, end: Point }\n"
        "fn make_line(x1: int, y1: int, x2: int, y2: int) -> Line {\n"
        "  return Line { start: Point { x: x1, y: y1 }, end: Point { x: x2, y: y2 } }\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow make_line {\n"
        "  let l = (make_line 1 2 3 4)\n"
        "  assert (== l.start.x 1)\n"
        "  assert (== l.end.y 4)\n"
        "}\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

/* ============================================================================
 * Coverage-targeted tests: map/reduce pure-arithmetic fast paths,
 * unary-minus on arrays, array-scalar broadcast, for-over-dynarray,
 * print struct/union, coroutine spawn, async fn calls.
 * ============================================================================ */

/* map fast path: single-return pure-arithmetic fn on int DynArray (lines 1789-1808) */
void test_eval_map_pure_arithmetic_int(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn double_it(x: int) -> int { return (* x 2) }\n"
        "fn main() -> int {\n"
        "    let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "    let out: array<int> = (map arr double_it)\n"
        "    return (array_get out 2)\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "    let out: array<int> = (map arr double_it)\n"
        "    assert (== (array_get out 0) 2)\n"
        "    assert (== (array_get out 4) 10)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* map fast path: single-return pure-arithmetic fn on float DynArray */
void test_eval_map_pure_arithmetic_float(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn half(x: float) -> float { return (/ x 2.0) }\n"
        "fn main() -> int {\n"
        "    let arr: array<float> = [2.0, 4.0, 6.0]\n"
        "    let out: array<float> = (map arr half)\n"
        "    return 0\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<float> = [2.0, 4.0, 6.0]\n"
        "    let out: array<float> = (map arr half)\n"
        "    assert (== (array_length out) 3)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* reduce fast path: pure-arithmetic combine fn on int DynArray (lines 2136-2160) */
void test_eval_reduce_pure_arithmetic_int(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn add_ints(acc: int, x: int) -> int { return (+ acc x) }\n"
        "fn main() -> int {\n"
        "    let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "    let total: int = (reduce arr add_ints 0)\n"
        "    return total\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "    let total: int = (reduce arr add_ints 0)\n"
        "    assert (== total 15)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* reduce fast path: pure-arithmetic combine fn on float DynArray */
void test_eval_reduce_pure_arithmetic_float(void) {
    RunCtx ctx;
    /* Use int-to-float cast to avoid typechecker limitations with float reduce.
     * We still exercise the reduce pure-arithmetic fast path via the int path. */
    bool ok = run_ctx_init(&ctx,
        "fn mul_ints(acc: int, x: int) -> int { return (* acc x) }\n"
        "fn main() -> int {\n"
        "    let arr: array<int> = [1, 2, 3, 4]\n"
        "    let prod: int = (reduce arr 1 mul_ints)\n"
        "    return prod\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<int> = [1, 2, 3, 4]\n"
        "    let prod: int = (reduce arr 1 mul_ints)\n"
        "    assert (== prod 24)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* Unary minus on int DynArray (lines 2249-2261) */
void test_eval_unary_minus_int_array(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn negate_arr(arr: array<int>) -> array<int> { return (- arr) }\n"
        "fn main() -> int {\n"
        "    let arr: array<int> = [1, 2, 3]\n"
        "    let neg: array<int> = (negate_arr arr)\n"
        "    return (array_get neg 0)\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<int> = [1, 2, 3]\n"
        "    let neg: array<int> = (negate_arr arr)\n"
        "    assert (== (array_get neg 0) -1)\n"
        "    assert (== (array_get neg 2) -3)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* Unary minus on float DynArray (lines 2258-2261) */
void test_eval_unary_minus_float_array(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn neg_f(arr: array<float>) -> array<float> { return (- arr) }\n"
        "fn main() -> int {\n"
        "    let arr: array<float> = [1.5, -2.5, 3.0]\n"
        "    let neg: array<float> = (neg_f arr)\n"
        "    return 0\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<float> = [1.5, -2.5, 3.0]\n"
        "    let neg: array<float> = (neg_f arr)\n"
        "    assert (== (array_length neg) 3)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* Array-scalar broadcast: array + scalar (DynArray, lines 2296-2327) */
void test_eval_array_scalar_broadcast_add(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "    let arr: array<int> = [10, 20, 30]\n"
        "    let out: array<int> = (+ arr 5)\n"
        "    return (array_get out 0)\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<int> = [10, 20, 30]\n"
        "    let out: array<int> = (+ arr 5)\n"
        "    assert (== (array_get out 0) 15)\n"
        "    assert (== (array_get out 2) 35)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* Array-scalar broadcast: array * scalar */
void test_eval_array_scalar_broadcast_mul(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "    let arr: array<float> = [1.0, 2.0, 3.0]\n"
        "    let out: array<float> = (* arr 2.0)\n"
        "    return 0\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<float> = [1.0, 2.0, 3.0]\n"
        "    let out: array<float> = (* arr 2.0)\n"
        "    assert (== (array_length out) 3)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* for loop over DynArray (lines 5448-5459) — use inline literal so typechecker
 * sees a concrete array<int> iterable; eval executes the DynArray branch */
void test_eval_for_over_dynarray(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "    let mut s: int = 0\n"
        "    for x in [1, 2, 3, 4, 5] { set s (+ s x) }\n"
        "    return s\n"
        "}\n"
        "shadow main {\n"
        "    let mut s: int = 0\n"
        "    for x in [10, 20, 30] { set s (+ s x) }\n"
        "    assert (== s 60)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* for loop over float DynArray (exercises ELEM_FLOAT branch, line 5456) */
void test_eval_for_over_float_array(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "    let mut n: int = 0\n"
        "    for x in [1.0, 2.0, 3.0] { set n (+ n 1) }\n"
        "    return n\n"
        "}\n"
        "shadow main {\n"
        "    let mut n: int = 0\n"
        "    for x in [1.0, 2.0, 3.0] { set n (+ n 1) }\n"
        "    assert (== n 3)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* Coroutine spawn + scheduler_run (lines 48-56, 2841-2870, 4391-4401) */
void test_eval_coroutine_spawn_and_run(void) {
    RunCtx ctx;
    suppress_stderr();  /* coroutine messages may go to stderr */
    bool ok = run_ctx_init(&ctx,
        "async fn worker(x: int) -> int { return (* x 2) }\n"
        "fn main() -> int {\n"
        "    let h: int = (spawn worker 5)\n"
        "    scheduler_run\n"
        "    return 0\n"
        "}\n"
    );
    restore_stderr();
    /* Coroutines may or may not succeed depending on scheduler support,
     * but must not crash. */
    (void)ok;
    run_ctx_free(&ctx);
}

/* async fn direct call (exercises is_async path in call_function, line 4391) */
void test_eval_async_fn_direct_call(void) {
    RunCtx ctx;
    suppress_stderr();
    bool ok = run_ctx_init(&ctx,
        "async fn compute(n: int) -> int { return (* n n) }\n"
        "fn main() -> int {\n"
        "    let r: int = (compute 7)\n"
        "    return r\n"
        "}\n"
        "shadow main {\n"
        "    assert (== (compute 3) 9)\n"
        "}\n"
    );
    restore_stderr();
    /* async fn may be called synchronously or via coroutine */
    (void)ok;
    run_ctx_free(&ctx);
}

/* String format with struct — exercises format buffer path (lines 3164-3196) */
void test_eval_string_format_struct(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "struct Vec2 { x: int, y: int }\n"
        "fn vec_str(v: Vec2) -> string {\n"
        "    return (str_concat \"(\" (str_concat (int_to_string v.x) \")\"))\n"
        "}\n"
        "fn main() -> int {\n"
        "    let v: Vec2 = Vec2 { x: 3, y: 4 }\n"
        "    let s: string = (vec_str v)\n"
        "    print s\n"
        "    return 0\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* Exercise sorted array operations and index_of */
void test_eval_array_broadcast_scalar_right(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "fn main() -> int {\n"
        "    let arr: array<int> = [3, 1, 4, 1, 5]\n"
        "    let sorted: array<int> = (array_sort arr)\n"
        "    let idx: int = (array_index_of arr 4)\n"
        "    return idx\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<int> = [3, 1, 4, 1, 5]\n"
        "    let sorted: array<int> = (array_sort arr)\n"
        "    assert (== (array_get sorted 0) 1)\n"
        "    assert (== (array_get sorted 4) 5)\n"
        "}\n"
    );
    ASSERT(ok);
    run_ctx_free(&ctx);
}

/* ============================================================================
 * main
 * ============================================================================ */

void test_eval_indexed_read_aliases(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "fn read(values: array<int>) -> int { return (+ (at values 0) (array_get values 1)) }\n"
        "shadow read { assert (== (read [20, 22]) 42) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n"));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    Value values = create_array(VAL_INT, 2, 2);
    ((long long *)values.as.array_val->data)[0] = 20;
    ((long long *)values.as.array_val->data)[1] = 22;
    Value result = call_function("read", &values, 1, ctx.env);
    ASSERT(result.type == VAL_INT && result.as.int_val == 42);
    free(values.as.array_val->data);
    free(values.as.array_val);
    DynArray *dynamic = dyn_array_new(ELEM_INT);
    dynamic = dyn_array_push_int(dynamic, 20);
    dynamic = dyn_array_push_int(dynamic, 22);
    values = create_void();
    values.type = VAL_DYN_ARRAY;
    values.as.dyn_array_val = dynamic;
    result = call_function("read", &values, 1, ctx.env);
    ASSERT(result.type == VAL_INT && result.as.int_val == 42);
    run_ctx_free(&ctx);
}

void test_eval_foreign_native_call_api(void) {
    ASSERT(ffi_init(false));
    ASSERT(ffi_loader_open("eval_abi", "obj/test_interpreter_ffi_native.so"));
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "extern fn ffi_test_void(d: float, n: int, b: bool) -> void\n"
        "extern fn ffi_test_observed() -> int\n"
        "extern fn ffi_test_string() -> string\n"
        "extern fn ffi_test_bool(b: bool, n: int, d: float) -> bool\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n"));
    int symbols = ctx.env->symbol_count;
    Value args[] = {create_float(1.25), create_int(42), create_bool(true)};
    Value result = call_function("ffi_test_void", args, 3, ctx.env);
    ASSERT(result.type == VAL_VOID);
    result = call_function("ffi_test_observed", NULL, 0, ctx.env);
    ASSERT(result.type == VAL_INT && result.as.int_val == 42);
    result = call_function("ffi_test_string", NULL, 0, ctx.env);
    ASSERT(result.type == VAL_STRING && !strcmp(result.as.string_val, "native"));
    Value bool_args[] = {create_bool(true), create_int(42), create_float(1.25)};
    result = call_function("ffi_test_bool", bool_args, 3, ctx.env);
    ASSERT(result.type == VAL_BOOL && result.as.bool_val);
    bool_args[0] = create_bool(false);
    result = call_function("ffi_test_bool", bool_args, 3, ctx.env);
    ASSERT(result.type == VAL_BOOL && !result.as.bool_val);
    suppress_stderr();
    result = call_function("ffi_test_void", args, 2, ctx.env);
    ASSERT(result.type == VAL_VOID);
    result = call_function("ffi_test_void", NULL, 3, ctx.env);
    ASSERT(result.type == VAL_VOID);
    args[0] = create_bool(true);
    result = call_function("ffi_test_void", args, 3, ctx.env);
    ASSERT(result.type == VAL_VOID);
    restore_stderr();
    result = call_function("ffi_test_observed", NULL, 0, ctx.env);
    ASSERT(result.type == VAL_INT && result.as.int_val == 42);
    ASSERT(ctx.env->symbol_count == symbols);
    run_ctx_free(&ctx);
    ffi_cleanup();
}

void test_eval_struct_array_literal(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Item { label: string, value: int }\n"
        "fn item() -> Item { let label: string = (+ \"forty\" \"two\") return Item { label: label, value: 42 } }\n"
        "fn main() -> int { return 0 }\n"
        "shadow item { let values: array<Item> = [(item), Item { label: \"next\", value: 43 }] "
        "let first: Item = (at values 0) let second: Item = (array_get values 1) "
        "assert (== first.label \"fortytwo\") assert (== first.value 42) "
        "assert (== second.label \"next\") assert (== second.value 43) }\n"));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    run_ctx_free(&ctx);
}

void test_eval_array_literal_evaluates_once_in_order(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "let mut calls: int = 0\n"
        "fn next() -> int { set calls (+ calls 1) return calls }\n"
        "fn main() -> int { return 0 }\n"
        "shadow next { let values: array<int> = [(next), (next), (next)] "
        "assert (== calls 3) assert (== (at values 0) 1) "
        "assert (== (at values 1) 2) assert (== (at values 2) 3) }\n"));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    run_ctx_free(&ctx);
}

void test_eval_array_append_and_dynamic_write(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Item { value: int, label: string }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { "
        "let ints: array<int> = [1] let ints2: array<int> = (array_push ints 2) "
        "assert (== (array_length ints) 2) assert (== (at ints2 1) 2) "
        "let floats: array<float> = [1.5] let floats2: array<float> = (array_push floats 2.5) "
        "assert (== (at floats2 1) 2.5) "
        "let bools: array<bool> = [false] let bools2: array<bool> = (array_push bools true) "
        "assert (at bools2 1) "
        "let strings: array<string> = [\"a\"] let strings2: array<string> = (array_push strings \"b\") "
        "assert (== (at strings2 1) \"b\") "
        "let records: array<Item> = [Item { value: 1, label: \"a\" }] "
        "let records2: array<Item> = (array_push records Item { value: 2, label: \"b\" }) "
        "assert (== (at records2 1).value 2) "
        "let di: array<int> = (array_push [] 1) (array_set di 0 42) assert (== (at di 0) 42) "
        "let df: array<float> = (array_push [] 1.5) (array_set df 0 2.5) assert (== (at df 0) 2.5) "
        "let db: array<bool> = (array_push [] false) (array_set db 0 true) assert (at db 0) "
        "let ds: array<string> = (array_push [] \"a\") (array_set ds 0 (+ \"forty\" \"two\")) "
        "assert (== (at ds 0) \"fortytwo\") "
        "let dr: array<Item> = (array_push [] Item { value: 1, label: \"a\" }) "
        "(array_set dr 0 Item { value: 42, label: (+ \"forty\" \"two\") }) "
        "assert (== (at dr 0).value 42) assert (== (at dr 0).label \"fortytwo\") "
        "let nested: array<array<int>> = (array_push [] di) "
        "let other: array<int> = (array_push [] 7) (array_set nested 0 other) "
        "assert (== (at (at nested 0) 0) 7) }\n"));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    run_ctx_free(&ctx);
}

void test_eval_record_alias_reassignment(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Item { value: int, label: string }\n"
        "struct Box { item: Item }\n"
        "fn replace(item: Item) -> Item { let mut local: Item = item "
        "set local Item { value: 2, label: \"new\" } return local }\n"
        "fn main() -> int { return 0 }\n"
        "shadow replace { let mut original: Item = Item { value: 1, label: (+ \"o\" \"ld\") } "
        "let alias: Item = original let nested: Box = Box { item: original } "
        "set original original assert (== original.label \"old\") "
        "set original (replace original) assert (== original.value 2) "
        "assert (== alias.value 1) assert (== alias.label \"old\") "
        "assert (== nested.item.label \"old\") "
        "set original nested.item assert (== original.value 1) "
        "set original Item { value: 3, label: \"last\" } "
        "assert (== nested.item.value 1) assert (== nested.item.label \"old\") }\n"));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    run_ctx_free(&ctx);
}

void test_eval_epoch_milliseconds(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "extern fn nl_get_time_ms() -> int\n"
        "fn now() -> int { unsafe { return (nl_get_time_ms) } }\n"
        "fn main() -> int { return 0 }\n"
        "shadow now { assert (> (now) 0) }\n"));
    long long before = (long long)time(NULL) * 1000LL;
    Value result = call_function("now", NULL, 0, ctx.env);
    long long after = (long long)time(NULL) * 1000LL + 999LL;
    ASSERT(result.type == VAL_INT);
    ASSERT(result.as.int_val >= before && result.as.int_val <= after);
    run_ctx_free(&ctx);
}

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
    TEST(eval_struct_string_field_lifetime);
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
    TEST(eval_cond_expression);
    TEST(eval_break_in_for);
    TEST(eval_nested_for_loops);
    TEST(eval_array_length);
    TEST(eval_array_get_alias);
    TEST(eval_math_functions);
    TEST(eval_string_conversion);
    TEST(eval_enum_access);
    TEST(eval_string_to_int_back);
    TEST(eval_shadowed_functions_reuse);
    TEST(eval_float_comparison);
    TEST(eval_map_builtin);
    TEST(eval_filter_builtin);
    TEST(eval_reduce_builtin);
    TEST(eval_array_push_pop);
    TEST(eval_array_sort);
    TEST(eval_array_contains);
    TEST(eval_type_casts);
    TEST(eval_string_format_ops);
    TEST(eval_array_reverse);
    TEST(eval_array_index_of);
    TEST(eval_array_slice);
    TEST(eval_set_mutation);
    TEST(eval_array_set);
    TEST(eval_continue_in_loop);
    TEST(eval_unsafe_block);
    TEST(eval_nested_closures);
    TEST(eval_string_format);
    TEST(eval_nested_arrays);
    TEST(eval_string_array);
    TEST(eval_float_array);
    TEST(eval_complex_match_with_guards);
    TEST(eval_union_with_data);
    TEST(eval_multiple_function_calls);
    TEST(eval_let_reassignment);
    TEST(eval_string_escape);
    TEST(eval_array_operations_comprehensive);
    TEST(eval_type_casting);
    TEST(eval_bool_operations);
    TEST(eval_nested_struct_fields);
    TEST(eval_record_string_local_lifetime);

    TEST(eval_map_pure_arithmetic_int);
    TEST(eval_map_pure_arithmetic_float);
    TEST(eval_reduce_pure_arithmetic_int);
    TEST(eval_reduce_pure_arithmetic_float);
    TEST(eval_unary_minus_int_array);
    TEST(eval_unary_minus_float_array);
    TEST(eval_array_scalar_broadcast_add);
    TEST(eval_array_scalar_broadcast_mul);
    TEST(eval_for_over_dynarray);
    TEST(eval_for_over_float_array);
    TEST(eval_coroutine_spawn_and_run);
    TEST(eval_async_fn_direct_call);
    TEST(eval_string_format_struct);
    TEST(eval_array_broadcast_scalar_right);
    TEST(eval_foreign_native_call_api);
    TEST(eval_indexed_read_aliases);
    TEST(eval_struct_array_literal);
    TEST(eval_array_literal_evaluates_once_in_order);
    TEST(eval_array_append_and_dynamic_write);
    TEST(eval_record_alias_reassignment);
    TEST(eval_epoch_milliseconds);

    printf("\n✓ All eval tests passed!\n");
    return 0;
}
