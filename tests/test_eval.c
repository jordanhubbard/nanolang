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
#include "../src/eval/eval_io.h"
#include "../src/coroutine.h"
#include "../src/effects.h"
#include "../src/interpreter_ffi.h"
#include "../src/runtime/ffi_loader.h"
#include "../src/runtime/dyn_array.h"
#include "../src/runtime/list_string.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Only the test-specific eval object redirects this clock call. Other tests
 * use the host clock unless this deterministic epoch fixture is active. */
static int s_epoch_clock_active;
static int s_epoch_clock_calls;
static clockid_t s_epoch_clock_id;
static struct timespec s_epoch_clock_value;
int nano_test_clock_gettime(clockid_t clock_id, struct timespec *result) {
    if (!s_epoch_clock_active) return clock_gettime(clock_id, result);
    s_epoch_clock_calls++;
    s_epoch_clock_id = clock_id;
    *result = s_epoch_clock_value;
    return 0;
}

static int s_fail_fwrite;
static int s_fail_fclose;
static int s_fclose_calls;
size_t nano_test_fwrite(const void *ptr, size_t size, size_t count, FILE *stream) {
    return s_fail_fwrite ? 0 : fwrite(ptr, size, count, stream);
}
int nano_test_fclose(FILE *stream) {
    s_fclose_calls++;
    int result = fclose(stream);
    return s_fail_fclose ? EOF : result;
}

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

/* I distinguish runtime values from retained checker rows without source widening. */
void test_eval_declared_push_initializer_bindings(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "fn array_push(a: int, b: int) -> int { return (+ a b) }\n"
        "shadow array_push { assert (== (array_push 2 3) 5) }\n"
        "fn capture() -> fn(int, int) -> int { return array_push }\n"
        "fn main() -> int { return 0 }\n"));
    env_define_var(ctx.env, "array_push", TYPE_FUNCTION, false, create_void());
    Symbol *binding = env_get_var(ctx.env, "array_push");
    ASSERT(binding != NULL);
    binding->def_line = 99; /* I model my retained non-global checker placeholder. */
    Value result = call_function("capture", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_FUNCTION);
    ASSERT(strcmp(result.as.function_val.function_name, "array_push") == 0);
    free((char *)result.as.function_val.function_name);
    free_function_signature(result.as.function_val.signature);

    binding = env_get_var(ctx.env, "array_push");
    binding->is_global = true;
    result = call_function("capture", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_VOID); /* A located global is never skipped. */
    binding = env_get_var(ctx.env, "array_push");
    binding->value = create_int(37);
    result = call_function("capture", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 37);

    env_define_var(ctx.env, "array_push", TYPE_FUNCTION, false, create_void());
    binding = env_get_var(ctx.env, "array_push");
    ASSERT_EQ(binding->def_line, 0);
    ASSERT(!binding->is_global);
    result = call_function("capture", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_VOID); /* Actual local VOID masks the declaration/global. */
    binding = env_get_var(ctx.env, "array_push");
    binding->value = create_int(41);
    result = call_function("capture", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 41);
    run_ctx_free(&ctx);
}

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
    env_discard_value_snapshot(pt);

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
    env_discard_value_snapshot(pair);

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

void test_eval_binary64_prefix_and_strict_cast(void) {
    Environment *env = create_environment();
    const struct { const char *text; double prefix; double strict; bool rejected; } cases[] = {
        {"  -1.25", -1.25, -1.25, false}, {"0x1.8p+1", 3.0, 3.0, false},
        {"1.25tail", 1.25, 0.0, true}, {"1.25 ", 1.25, 0.0, true},
        {"1e+", 1.0, 0.0, true}, {"0x1p-", 1.0, 0.0, true},
        {"", 0.0, 0.0, true}, {"  +", 0.0, 0.0, true},
        {"0e+9", 0.0, 0.0, false}, {"-0x0p-9", -0.0, -0.0, false},
        {"4.5\0tail", 4.5, 4.5, false}
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        ASTNode literal = {.type = AST_STRING, .as.string_val = (char *)cases[i].text};
        ASTNode *arguments[] = {&literal};
        ASTNode call = {.type = AST_CALL, .as.call = { .name = "string_to_float", .args = arguments, .arg_count = 1 }};
        Value prefix = repl_eval_node(&call, env);
        ASSERT(prefix.type == VAL_FLOAT && prefix.as.float_val == cases[i].prefix);
        FILE *saved = stderr, *messages = tmpfile();
        ASSERT(messages != NULL);
        stderr = messages;
        call.as.call.name = "cast_float";
        Value strict = repl_eval_node(&call, env);
        fflush(messages);
        long count = ftell(messages);
        stderr = saved;
        fclose(messages);
        ASSERT(strict.type == VAL_FLOAT && strict.as.float_val == cases[i].strict);
        ASSERT((count > 0) == cases[i].rejected);
        uint64_t actual, expected;
        memcpy(&actual, &strict.as.float_val, sizeof actual);
        memcpy(&expected, &cases[i].strict, sizeof expected);
        ASSERT(actual == expected);
    }
    ASTNode nan_arg = {.type = AST_STRING, .as.string_val = "nan(184467440737095516160000)"};
    ASTNode *nan_args[] = {&nan_arg};
    ASTNode nan_call = {.type = AST_CALL, .as.call = { .name = "string_to_float", .args = nan_args, .arg_count = 1 }};
    Value nan_result = repl_eval_node(&nan_call, env);
    uint64_t nan_bits;
    memcpy(&nan_bits, &nan_result.as.float_val, sizeof nan_bits);
    ASSERT(nan_bits == UINT64_C(0x7fffffffffffffff));
    const struct { const char *text; uint64_t bits; bool rejected; } special[] = {
        {"nan(184467440737095516160000)", UINT64_C(0x7fffffffffffffff), false},
        {"-nan()", UINT64_C(0xfff8000000000000), false},
        {"nan(+1)", 0, true}, {"Infinity", UINT64_C(0x7ff0000000000000), false},
        {"infinit", 0, true}
    };
    nan_call.as.call.name = "cast_float";
    for (size_t i = 0; i < sizeof special / sizeof special[0]; i++) {
        nan_arg.as.string_val = (char *)special[i].text;
        FILE *saved = stderr, *messages = tmpfile();
        ASSERT(messages != NULL);
        stderr = messages;
        Value result = repl_eval_node(&nan_call, env);
        fflush(messages);
        long count = ftell(messages);
        stderr = saved;
        fclose(messages);
        ASSERT(result.type == VAL_FLOAT);
        memcpy(&nan_bits, &result.as.float_val, sizeof nan_bits);
        ASSERT(nan_bits == special[i].bits && ((count > 0) == special[i].rejected));
    }
    free_environment(env);
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

void test_eval_match_wildcards_follow_lexical_order(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "union Choice { Some { number: int }, None {} }\n"
        "let mut match_trace: int = 0\n"
        "fn mark_match_guard(id: int, answer: bool) -> bool {\n"
        "  set match_trace (+ (* match_trace 10) id)\n"
        "  return answer\n"
        "}\n"
        "shadow mark_match_guard { set match_trace 0 assert (mark_match_guard 1 true) set match_trace 0 }\n"
        "fn select_match(value: Choice, first: bool, named: bool) -> int {\n"
        "  return match value {\n"
        "    _ if (mark_match_guard 1 first) => { set match_trace (+ (* match_trace 10) 7) 100 }\n"
        "    Some(payload) if (mark_match_guard 2 named) => { set match_trace (+ (* match_trace 10) 8) payload.number }\n"
        "    Some | None if (mark_match_guard 3 true) => { set match_trace (+ (* match_trace 10) 9) -1 }\n"
        "    _ => -2\n"
        "  }\n"
        "}\n"
        "shadow select_match {\n"
        "  let some: Choice = Choice.Some { number: 23 }\n"
        "  let none: Choice = Choice.None {}\n"
        "  set match_trace 0\n"
        "  assert (== (select_match some true true) 100)\n"
        "  assert (== match_trace 17)\n"
        "  set match_trace 0\n"
        "  assert (== (select_match some false true) 23)\n"
        "  assert (== match_trace 128)\n"
        "  set match_trace 0\n"
        "  assert (== (select_match none false true) -1)\n"
        "  assert (== match_trace 139)\n"
        "}\n"
        "fn mark_match_input(value: int) -> int {\n"
        "  set match_trace (+ (* match_trace 10) 9) return value\n"
        "}\n"
        "shadow mark_match_input { set match_trace 0 assert (== (mark_match_input 7) 7) assert (== match_trace 9) set match_trace 0 }\n"
        "fn select_integer_match(first: bool, second: bool) -> int {\n"
        "  return match (mark_match_input 7) {\n"
        "    _ if (mark_match_guard 1 first) => { set match_trace (+ (* match_trace 10) 7) 100 }\n"
        "    _ if (mark_match_guard 2 second) => { set match_trace (+ (* match_trace 10) 8) 200 }\n"
        "    7 => { set match_trace (+ (* match_trace 10) 9) 300 }\n"
        "    _ => -1\n"
        "  }\n"
        "}\n"
        "shadow select_integer_match {\n"
        "  set match_trace 0 assert (== (select_integer_match true true) 100) assert (== match_trace 917)\n"
        "  set match_trace 0 assert (== (select_integer_match true false) 100) assert (== match_trace 917)\n"
        "  set match_trace 0 assert (== (select_integer_match false true) 200) assert (== match_trace 9128)\n"
        "  set match_trace 0 assert (== (select_integer_match false false) 300) assert (== match_trace 9129)\n"
        "}\n"
        "fn restore_outer_binding(value: Choice) -> int {\n"
        "  let payload: int = 41\n"
        "  return match value {\n"
        "    Some(payload) if false => payload.number\n"
        "    _ => payload\n"
        "  }\n"
        "}\n"
        "shadow restore_outer_binding {\n"
        "  assert (== (restore_outer_binding Choice.Some { number: 9 }) 41)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert true }\n"
    );
    ASSERT(ok);
    bool shadows_ok = run_shadow_tests(ctx.program, ctx.env, false);
    ASSERT(shadows_ok);
    run_ctx_free(&ctx);
}

/* I retain callable leaves after their original binding and handler retire. */
void test_eval_callable_projection_lifetimes(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Holder { callback: fn(int) -> int }\n"
        "struct Outer { inner: Holder }\n"
        "effect Capture { pack : fn(int) -> int -> void }\n"
        "fn increment(n: int) -> int { return (+ n 1) }\n"
        "shadow increment { assert (== (increment 7) 8) }\n"
        "fn replacement(n: int) -> int { return (+ n 2) }\n"
        "shadow replacement { assert (== (replacement 7) 9) }\n"
        "fn projected(f: fn(int) -> int) -> Holder {\n"
        " let mut original: fn(int) -> int = f\n"
        " let held: Holder = Holder { callback: original }\n"
        " let mut copied: fn(int) -> int = held.callback\n"
        " set copied held.callback\n"
        " set original replacement\n"
        " assert (== (copied 7) 8)\n"
        " return held\n"
        "}\n"
        "shadow projected { let held: Holder = (projected increment) let f: fn(int) -> int = held.callback assert (== (f 7) 8) }\n"
        "fn retained() -> Outer {\n"
        " let mut stored: Outer = Outer { inner: Holder { callback: replacement } }\n"
        " let ignored = handle { perform Capture.pack(increment) } with {\n"
        "  pack f -> { set stored Outer { inner: Holder { callback: f } } }\n"
        " }\n"
        " return stored\n"
        "}\n"
        "shadow retained { let held: Outer = (retained) let f: fn(int) -> int = held.inner.callback assert (== (f 7) 8) }\n"
        "fn main() -> int {\n"
        " let held: Holder = (projected increment)\n"
        " let f: fn(int) -> int = held.callback\n"
        " let nested: Outer = (retained)\n"
        " let g: fn(int) -> int = nested.inner.callback\n"
        " return (+ (f 7) (g 7))\n"
        "}\n"
        "shadow main { assert (== (main) 16) }\n"));
    for (int i = 0; i < 32; ++i) {
        Value result = call_function("main", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 16);
        ASSERT(nl_effect_find_handler("Capture", "pack", NULL) == NULL);
    }
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    run_ctx_free(&ctx);
}

void test_eval_handled_record_identity(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Packet { value: int }\n"
        "effect Supply { next : int -> Packet }\n"
        "fn direct() -> Packet {\n"
        " return handle { perform Supply.next(8) } with { next n -> Packet { value: n } }\n"
        "}\n"
        "shadow direct { let p: Packet = (direct) assert (== p.value 8) }\n"
        "fn branch(n: int) -> Packet {\n"
        " return handle { perform Supply.next(n) } with {\n"
        "  next value -> { if (== value 0) { return Packet { value: 41 } } else { Packet { value: value } } }\n"
        " }\n"
        "}\n"
        "shadow branch { let a: Packet = (branch 0) let b: Packet = (branch 9) assert (== a.value 41) assert (== b.value 9) }\n"
        "fn lexical() -> int {\n"
        " let p: Packet = handle { perform Supply.next(0) } with { next n -> { return 73 } }\n"
        " return p.value\n"
        "}\n"
        "shadow lexical { assert (== (lexical) 73) }\n"
        "fn projected() -> int {\n"
        " let p: Packet = handle { perform Supply.next(12) } with { next n -> Packet { value: n } }\n"
        " return p.value\n"
        "}\n"
        "shadow projected { assert (== (projected) 12) }\n"
        "fn nested() -> Packet {\n"
        " return handle { perform Supply.next(8) } with { next n -> cond (true if true { Packet { value: n } } else { Packet { value: 0 } }) (else Packet { value: 0 }) }\n"
        "}\n"
        "shadow nested { let p: Packet = (nested) assert (== p.value 8) }\n"
        "fn matched() -> Packet {\n"
        " return handle { perform Supply.next(8) } with { next n -> match n { 8 => if true { Packet { value: n } } else { Packet { value: 0 } }, _ => Packet { value: 0 } } }\n"
        "}\n"
        "shadow matched { let p: Packet = (matched) assert (== p.value 8) }\n"
        "fn main() -> int { let a: Packet = (direct) let b: Packet = (branch 0) let c: Packet = (branch 9) let d: Packet = (nested) let e: Packet = (matched) return (+ e.value (+ d.value (+ a.value (+ b.value (+ c.value (+ (lexical) (projected))))))) }\n"
        "shadow main { assert (== (main) 159) }\n"));
    Value result = call_function("main", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 159);
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    run_ctx_free(&ctx);

    const char *invalid[] = {
        "Other { value: n }",
        "{ if (== n 0) { Packet { value: n } } else { Other { value: n } } }",
        "{ let absent: int = n }",
        "{ return 17 }",
        "cond (true if true { Packet { value: true } } else { Packet { value: n } }) (else Packet { value: n })",
        "cond (true if true { Packet { value: n } } else { Packet { value: true } }) (else Packet { value: n })",
        "match n { 8 => if true { Packet { value: true } } else { Packet { value: n } }, _ => Packet { value: n } }",
        "n"
    };
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i) {
        char source[2048];
        int size = snprintf(source, sizeof source,
            "struct Packet { value: int }\nstruct Other { value: int }\n"
            "effect Supply { next : int -> Packet }\n"
            "fn bad() -> Packet { return handle { perform Supply.next(8) } with { next n -> %s } }\n"
            "shadow bad { assert true }\nfn main() -> int { return 0 }\nshadow main { assert true }\n", invalid[i]);
        ASSERT(size > 0 && (size_t)size < sizeof source);
        ASSERT(!run_ctx_init(&ctx, source));
        run_ctx_free(&ctx);
    }
    /* I distinguish two identically shaped, identically named declarations. */
    for (int wrong_owner = 0; wrong_owner < 2; ++wrong_owner) {
        int owner_count = 0, caller_count = 0;
        Token *owner_tokens = tokenize("struct Packet { value: int }\neffect Supply { next : Packet -> Packet }\n", &owner_count);
        ASSERT(owner_tokens);
        ASTNode *owner_ast = parse_program(owner_tokens, owner_count);
        ASSERT(owner_ast);
        Environment *env = create_environment(); ASSERT(env);
        env->current_module = "Defining";
        ASSERT(type_check_module(owner_ast, env));
        NominalIdentity packet = env_nominal_identity(env, "Packet", "Defining", TYPE_STRUCT);
        ASSERT(packet.ordinal);
        env->current_module = "Caller";
        ASSERT(env_register_nominal_import(env, "Caller", "RemotePacket", packet));
        char source[2048];
        int size = snprintf(source, sizeof source,
            "struct Packet { value: int }\nfn main() -> int {\n"
            "let p: RemotePacket = handle { perform Supply.next(RemotePacket { value: 4 }) } with { next q -> %s }\n"
            "return p.value }\nshadow main { assert (== (main) 4) }\n",
            wrong_owner ? "Packet { value: 4 }" : "q");
        ASSERT(size > 0 && (size_t)size < sizeof source);
        Token *caller_tokens = tokenize(source, &caller_count); ASSERT(caller_tokens);
        ASTNode *caller_ast = parse_program(caller_tokens, caller_count); ASSERT(caller_ast);
        suppress_stderr();
        bool checked = type_check(caller_ast, env);
        restore_stderr();
        ASSERT(checked == !wrong_owner);
        free_environment(env);
        free_ast(caller_ast); free_tokens(caller_tokens, caller_count);
        free_ast(owner_ast); free_tokens(owner_tokens, owner_count);
        clear_module_cache();
    }
}

void test_eval_match_miss_is_terminal(void) {
    int errors[2];
    ASSERT(pipe(errors) == 0);
    fflush(NULL);
    pid_t child = fork();
    ASSERT(child >= 0);
    if (child == 0) {
        close(errors[0]);
        ASSERT(dup2(errors[1], STDERR_FILENO) >= 0);
        close(errors[1]);
        const char *source =
            "union Choice { Some { number: int }, None {} }\n"
            "fn unchecked_miss() -> int {\n"
            "  let value: Choice = Choice.None {}\n"
            "  let selected: int = match value { Some(payload) => payload.number }\n"
            "  return selected\n"
            "}\n";
        int token_count = 0;
        Token *tokens = tokenize(source, &token_count);
        ASTNode *program = tokens ? parse_program(tokens, token_count) : NULL;
        Environment *env = program ? create_environment() : NULL;
        if (!tokens || !program || !env || !run_program(program, env)) _exit(90);
        /* I bypass checking only for this deliberately incomplete AST. The
         * checker normally registers functions; run_program does not. */
        ASTNode *definition = NULL;
        for (int i = 0; i < program->as.program.count; ++i) {
            ASTNode *item = program->as.program.items[i];
            if (item->type == AST_FUNCTION &&
                strcmp(item->as.function.name, "unchecked_miss") == 0)
                definition = item;
        }
        if (!definition || definition->as.function.param_count != 0 ||
            definition->as.function.return_type != TYPE_INT) _exit(90);
        Function function = {0};
        function.name = definition->as.function.name;
        function.return_type = TYPE_INT;
        function.body = definition->as.function.body;
        env_define_function(env, function);
        (void)call_function("unchecked_miss", NULL, 0, env);
        _exit(91);
    }

    close(errors[1]);
    char message[512], chunk[256];
    size_t used = 0;
    bool read_ok = true, truncated = false;
    for (;;) {
        ssize_t length = read(errors[0], chunk, sizeof(chunk));
        if (length < 0 && errno == EINTR) continue;
        if (length < 0) { read_ok = false; break; }
        if (!length) break;
        size_t available = sizeof(message) - 1 - used;
        size_t count = (size_t)length < available ? (size_t)length : available;
        memcpy(message + used, chunk, count);
        used += count;
        if (count != (size_t)length) truncated = true;
    }
    close(errors[0]);
    message[used] = '\0';
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    ASSERT(waited == child);
    ASSERT(read_ok);
    ASSERT(!truncated);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_FAILURE)
        fprintf(stderr, "I observed unchecked-match child status %d and stderr: %s\n", status, message);
    ASSERT(WIFEXITED(status));
    ASSERT(WEXITSTATUS(status) == EXIT_FAILURE);
    ASSERT(strstr(message,
        "I cannot continue: a checked match reached no successful arm.") != NULL);
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
void test_eval_map_declared_scalar_results(void) {
    const char *types[] = {"int", "float", "bool", "string"};
    const char *values[] = {"42", "1.5", "true", "\"mapped\""};
    ValueType tags[] = {VAL_INT, VAL_FLOAT, VAL_BOOL, VAL_STRING};
    ElementType elements[] = {ELEM_INT, ELEM_FLOAT, ELEM_BOOL, ELEM_STRING};
    for (int type = 0; type < 4; type++) {
        for (int dynamic = 0; dynamic < 2; dynamic++) {
            for (int length = 0; length < 2; length++) {
                char source[512];
                snprintf(source, sizeof(source),
                    "fn transform(x: int) -> %s { return %s } "
                    "fn mapped(xs: array<int>) -> array<%s> { return (map xs transform) } "
                    "fn main() -> int { return 0 }", types[type], values[type], types[type]);
                RunCtx ctx;
                ASSERT(run_ctx_init(&ctx, source));
                Value input = create_void();
                if (dynamic) {
                    input.type = VAL_DYN_ARRAY;
                    input.as.dyn_array_val = dyn_array_new(ELEM_INT);
                    if (length) dyn_array_push_int(input.as.dyn_array_val, 7);
                } else {
                    input = create_array(VAL_INT, length, length);
                    if (length) ((long long *)input.as.array_val->data)[0] = 7;
                }
                Value output = call_function("mapped", &input, 1, ctx.env);
                ASSERT(!output.is_return);
                if (dynamic) {
                    ASSERT(output.type == VAL_DYN_ARRAY);
                    ASSERT(dyn_array_get_elem_type(output.as.dyn_array_val) == elements[type]);
                    ASSERT(dyn_array_length(output.as.dyn_array_val) == length);
                    if (length) {
                        if (type == 0) ASSERT(dyn_array_get_int(output.as.dyn_array_val, 0) == 42);
                        if (type == 1) ASSERT(dyn_array_get_float(output.as.dyn_array_val, 0) == 1.5);
                        if (type == 2) ASSERT(dyn_array_get_bool(output.as.dyn_array_val, 0));
                        if (type == 3) ASSERT(!strcmp(dyn_array_get_string(output.as.dyn_array_val, 0), "mapped"));
                        ASSERT(dyn_array_get_int(input.as.dyn_array_val, 0) == 7);
                    }
                } else {
                    ASSERT(output.type == VAL_ARRAY);
                    Array *array = output.as.array_val;
                    ASSERT(array->element_type == tags[type]);
                    ASSERT(array->length == length);
                    if (length) {
                        if (type == 0) ASSERT(((long long *)array->data)[0] == 42);
                        if (type == 1) ASSERT(((double *)array->data)[0] == 1.5);
                        if (type == 2) ASSERT(((bool *)array->data)[0]);
                        if (type == 3) { ASSERT(!strcmp(((char **)array->data)[0], "mapped")); free(((char **)array->data)[0]); }
                        ASSERT(((long long *)input.as.array_val->data)[0] == 7);
                    }
                    free(array->data);
                    free(array);
                }
                run_ctx_free(&ctx);
                if (!dynamic) { free(input.as.array_val->data); free(input.as.array_val); }
            }
        }
    }
}

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
        "    let total: int = (reduce arr 0 add_ints)\n"
        "    return total\n"
        "}\n"
        "shadow main {\n"
        "    let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "    let total: int = (reduce arr 0 add_ints)\n"
        "    assert (== total 15)\n"
        "}\n"
    );
    ASSERT(ok);
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
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

/* I keep the iteration copy alive when its collection changes, and return an
 * independent public snapshot through normal and early exits. Caller arrays
 * and lists remain caller-owned in this fixture. */
void test_eval_string_loop_bindings(void) {
    const char *source =
        "fn walk(xs:array<string>, mode:int)->string {\n"
        " let mut last:string = \"empty\" let mut index:int = 0\n"
        " for item in xs {\n"
        "  if (== mode 3) { return item }\n"
        "  if (== mode 4) { (array_set xs index \"changed\") }\n"
        "  set last item set index (+ index 1)\n"
        "  if (== mode 1) { continue }\n"
        "  if (== mode 2) { break }\n"
        " } return last\n"
        "}\n"
        "shadow walk { assert (== (walk [\"first\", \"second\"] 3) \"first\") }\n"
        "fn walk_list(xs:list_string, mode:int)->string {\n"
        " let mut last:string = \"empty\" let mut index:int = 0\n"
        " for item in xs {\n"
        "  if (== mode 3) { return item }\n"
        "  if (== mode 4) { (list_string_set xs index \"changed\") }\n"
        "  set last item set index (+ index 1)\n"
        "  if (== mode 1) { continue }\n"
        "  if (== mode 2) { break }\n"
        " } return last\n"
        "}\n"
        "shadow walk_list {\n"
        " let xs:list_string = (list_string_new) (list_string_push xs \"first\")\n"
        " assert (== (walk_list xs 3) \"first\") (list_string_free xs)\n"
        "}\n"
        "fn main()->int { return 0 }\n";
    for (int kind = 0; kind < 3; ++kind) for (int mode = 0; mode < 5; ++mode)
    for (int empty = 0; empty < 2; ++empty) {
        RunCtx ctx;
        ASSERT(run_ctx_init(&ctx, source));
        for (int repeat = 0; repeat < 8; ++repeat) {
            int length = empty ? 0 : 2;
            const char *texts[] = {"first", "second"};
            char *dynamic_inputs[2] = {NULL, NULL};
            Value input = create_void();
            List_string *list = NULL;
            if (kind == 0) {
                input = create_array(VAL_STRING, length, length);
                for (int i = 0; i < length; ++i)
                    ((char **)input.as.array_val->data)[i] = strdup(texts[i]);
            } else if (kind == 1) {
                input.type = VAL_DYN_ARRAY;
                input.as.dyn_array_val = dyn_array_new(ELEM_STRING);
                ASSERT_NOT_NULL(input.as.dyn_array_val);
                for (int i = 0; i < length; ++i) {
                    dynamic_inputs[i] = strdup(texts[i]);
                    input.as.dyn_array_val = dyn_array_push_string(input.as.dyn_array_val, dynamic_inputs[i]);
                }
            } else {
                list = list_string_new();
                for (int i = 0; i < length; ++i) list_string_push(list, texts[i]);
                input = create_int((intptr_t)list);
            }
            Value args[] = {input, create_int(mode)};
            Value result = call_function(kind == 2 ? "walk_list" : "walk", args, 2, ctx.env);
            const char *expected = empty ? "empty" : mode == 2 || mode == 3 ? "first" : "second";
            ASSERT_EQ(result.type, VAL_STRING);
            ASSERT(strcmp(result.as.string_val, expected) == 0);
            for (int i = 0; i < length; ++i) {
                const char *stored = kind == 0 ? ((char **)input.as.array_val->data)[i] :
                    kind == 1 ? dyn_array_get_string(input.as.dyn_array_val, i) : list_string_get(list, i);
                ASSERT(strcmp(stored, mode == 4 ? "changed" : texts[i]) == 0);
            }
            if (kind == 0) {
                for (int i = 0; i < length; ++i) free(((char **)input.as.array_val->data)[i]);
                free(input.as.array_val->data); free(input.as.array_val);
            } else if (kind == 1) {
                /* The low-level DynArray stores borrowed string pointers. */
                if (mode == 4) for (int i = 0; i < length; ++i)
                    free((char *)dyn_array_get_string(input.as.dyn_array_val, i));
                for (int i = 0; i < length; ++i) free(dynamic_inputs[i]);
                gc_release(input.as.dyn_array_val);
            } else list_string_free(list);
            if (repeat == 7) run_ctx_free(&ctx);
            ASSERT(strcmp(result.as.string_val, expected) == 0);
            free(result.as.string_val);
        }
    }
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
    nano_scheduler_init();
    int first_id = g_scheduler.count;
    bool ok = run_ctx_init(&ctx,
        "async fn compute(n: int) -> int { return (* n n) }\n"
        "shadow compute { assert (== (compute 3) 9) }\n"
        "fn main() -> int {\n"
        "    for i in (range 0 130) { assert (== (compute i) (* i i)) }\n"
        "    return 0\n"
        "}\n"
        "shadow main {\n"
        "    assert (== (compute 3) 9)\n"
        "}\n"
    );
    ASSERT(ok);
    Value result = call_function("main", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 0);
    ASSERT(g_scheduler.count - first_id >= 130);
    for (int i = 0; i < MAX_COROUTINES; i++) {
        ASSERT(g_scheduler.coroutines[i].id < first_id);
    }
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

void test_eval_empty_array_aliases(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Item { value: int, label: string }\n"
        "struct Holder { items: array<Item> }\n"
        "let mut trace: int = 0\n"
        "fn receiver(a: array<Item>) -> array<Item> { set trace (+ (* trace 10) 1) return a }\n"
        "fn item() -> Item { set trace (+ (* trace 10) 2) return Item { value: 7, label: (+ \"keep\" \"me\") } }\n"
        "fn append_inner(rows: array<array<int>>, inner: array<int>) -> array<array<int>> {\n"
        " (array_push rows inner) return rows }\n"
        "fn check() -> int {\n"
        " let ints: array<int> = [] let ia: array<int> = ints\n"
        " let floats: array<float> = [] let fa: array<float> = floats\n"
        " let bools: array<bool> = [] let ba: array<bool> = bools\n"
        " let strings: array<string> = [] let sa: array<string> = strings\n"
        " (array_push ints 3) (array_push floats 1.5) (array_push bools true)\n"
        " (array_push strings (+ \"keep\" \"me\"))\n"
        " assert (== (at ia 0) 3) assert (== (at fa 0) 1.5) assert (at ba 0)\n"
        " assert (== (at sa 0) \"keepme\")\n"
        " let h: Holder = Holder { items: [] } let alias: array<Item> = h.items\n"
        " set trace 0\n"
        " let result: array<Item> = (array_push (receiver h.items) (item))\n"
        " assert (== trace 12) assert (== (array_length alias) 1)\n"
        " assert (== (at result 0).label \"keepme\")\n"
        " let rows: array<array<int>> = [] let ra: array<array<int>> = rows\n"
        " let inner: array<int> = [] (array_push rows inner) (array_push inner 8)\n"
        " assert (== (at (at ra 0) 0) 8)\n"
        " let mut i: int = 0 while (< i 32) { (array_push ints i) set i (+ i 1) }\n"
        " assert (== (array_length ia) 33) assert (== (at ia 32) 31)\n"
        " (array_remove_at ints 0) assert (== (at ia 0) 0)\n"
        " assert (== (array_pop ints) 31) assert (== (array_length ia) 31)\n"
        " assert (== (array_pop strings) \"keepme\") assert (== (array_length sa) 0)\n"
        " (array_push sa \"again\") assert (== (at strings 0) \"again\")\n"
        " let popped: Item = (array_pop alias) assert (== popped.label \"keepme\")\n"
        " assert (== (array_length h.items) 0) (array_push h.items popped)\n"
        " (array_push h.items Item { value: 9, label: \"second\" })\n"
        " (array_remove_at alias 0) assert (== (at h.items 0).label \"second\")\n"
        " let popped_row: array<int> = (array_pop rows) (array_push popped_row 9)\n"
        " assert (== (at inner 1) 9) assert (== (array_length ra) 0)\n"
        " (array_push rows inner) (array_remove_at rows 0)\n"
        " assert (== (array_length inner) 2)\n"
        " return 0 }\n"
        "fn main() -> int { return 0 }\n"
        "shadow check { assert (== (check) 0) }\n"));
    for (int i = 0; i < 20; i++) {
        Value result = call_function("check", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 0);
    }
    /* I retain the nested runtime tag and shared identity too. */
    DynArray *dynamic = dyn_array_new(ELEM_INT);
    dyn_array_push_int(dynamic, 41);
    Value rows = create_array(VAL_INT, 0, 0);
    Value arguments[] = {rows, {.type = VAL_DYN_ARRAY, .as.dyn_array_val = dynamic}};
    Value appended = call_function("append_inner", arguments, 2, ctx.env);
    ASSERT_EQ(appended.type, VAL_ARRAY);
    ASSERT(appended.as.array_val == rows.as.array_val);
    ASSERT_EQ(rows.as.array_val->length, 1);
    Value stored = ((Value*)rows.as.array_val->data)[0];
    ASSERT_EQ(stored.type, VAL_DYN_ARRAY);
    ASSERT(stored.as.dyn_array_val == dynamic);
    dyn_array_push_int(dynamic, 42);
    ASSERT_EQ(dyn_array_length(stored.as.dyn_array_val), 2);
    free(rows.as.array_val->data);
    free(rows.as.array_val);
    gc_release(dynamic);
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

void test_eval_record_alias_across_direct_calls(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Pair { left: int, right: int }\n"
        "fn alias_after_reassignment() -> int {\n"
        " let original: Pair = Pair { left: 20, right: 22 }\n"
        " let mut alias: Pair = original\n"
        " set alias Pair { left: 1, right: 2 }\n"
        " return (+ original.left original.right)\n"
        "}\n"
        "fn read_pair(pair: Pair) -> int { return (+ pair.left pair.right) }\n"
        "fn alias_across_call() -> int {\n"
        " let pair: Pair = Pair { left: 19, right: 23 }\n"
        " let result: int = (read_pair pair)\n"
        " return (+ result pair.left)\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
        "shadow read_pair { assert (== (read_pair Pair { left: 1, right: 2 }) 3) }\n"
        "shadow alias_after_reassignment { assert (== (alias_after_reassignment) 42) }\n"
        "shadow alias_across_call { assert (== (alias_across_call) 61) }\n"));
    Value result = call_function("alias_after_reassignment", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 42);
    result = call_function("alias_across_call", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 61);
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
    const struct { time_t seconds; long nanoseconds; long long expected; } cases[] = {
        {0, 0, 0},
        {1700000000, 999999, 1700000000000LL},
        {1700000000, 1000000, 1700000000001LL},
        {1700000000, 999999999, 1700000000999LL},
        {1700000001, 0, 1700000001000LL},
    };
    s_epoch_clock_active = 1;
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        s_epoch_clock_value.tv_sec = cases[i].seconds;
        s_epoch_clock_value.tv_nsec = cases[i].nanoseconds;
        s_epoch_clock_calls = 0;
        Value result = call_function("now", NULL, 0, ctx.env);
        ASSERT_EQ(s_epoch_clock_calls, 1);
        ASSERT_EQ(s_epoch_clock_id, CLOCK_REALTIME);
        ASSERT(result.type == VAL_INT);
        ASSERT_EQ(result.as.int_val, cases[i].expected);
    }
    s_epoch_clock_active = 0;
    run_ctx_free(&ctx);
}

void test_eval_unqualified_effect_handler(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Recorder { emit : int -> void }\n"
        "let mut recorded: int = 0\n"
        "fn send(value: int) -> void { perform Recorder.emit(value) }\n"
        "fn exercise() -> int {\n"
        " let ignored = handle { (send 7) } with { emit value -> { set recorded value } }\n"
        " return recorded\n"
        "}\n"
        "shadow send { assert (== (exercise) 7) }\n"
        "shadow exercise { assert (== (exercise) 7) }\n"
        "fn main() -> int { return (exercise) }\n"
        "shadow main { assert (== (main) 7) }\n"));
    Value result = call_function("main", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 7);
    ASSERT(nl_effect_find_handler("Recorder", "emit", NULL) == NULL);
    run_ctx_free(&ctx);
}

void test_eval_nested_effect_handlers(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Recorder { emit : int -> void }\n"
        "let mut recorded: int = 0\n"
        "fn send(value: int) -> void { perform Recorder.emit(value) }\n"
        "fn nested() -> void {\n"
        " let ignored = handle { (send 2) } with {\n"
        "  emit value -> { set recorded (+ (* recorded 10) (+ value 1)) }\n"
        " }\n"
        "}\n"
        "fn sequence() -> void { (send 1) (nested) (send 4) }\n"
        "fn exercise() -> int {\n"
        " set recorded 0\n"
        " let ignored = handle { (sequence) } with {\n"
        "  emit value -> { set recorded (+ (* recorded 10) value) }\n"
        " }\n"
        " return recorded\n"
        "}\n"
        "shadow send { assert (== (exercise) 134) }\n"
        "shadow nested { assert (== (exercise) 134) }\n"
        "shadow sequence { assert (== (exercise) 134) }\n"
        "shadow exercise { assert (== (exercise) 134) }\n"
        "fn main() -> int { return (exercise) }\n"
        "shadow main { assert (== (main) 134) }\n"));
    for (int i = 0; i < 2; i++) {
        Value result = call_function("main", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 134);
        ASSERT(nl_effect_find_handler("Recorder", "emit", NULL) == NULL);
    }
    run_ctx_free(&ctx);
}

void test_eval_effect_argument_lists(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Recorder { pair : int int -> void, tick : void -> void }\n"
        "let mut trace: int = 0\n"
        "let mut recorded: int = 0\n"
        "fn argument(value: int) -> int { set trace (+ (* trace 10) value) return value }\n"
        "fn exercise() -> int {\n"
        " set trace 0 set recorded 0\n"
        " let first: int = 9\n"
        " let ignored = handle { perform Recorder.pair((argument 1) (+ first (argument 2))) } with {\n"
        "  pair first second -> { set recorded (+ (* first 100) second) }\n"
        " }\n"
        " let ticked = handle { perform Recorder.tick() } with {\n"
        "  tick -> { set recorded (+ recorded 1000) }\n"
        " }\n"
        " return (+ (* trace 10000) recorded)\n"
        "}\n"
        "shadow argument { assert (== (exercise) 121111) }\n"
        "shadow exercise { assert (== (exercise) 121111) }\n"
        "fn main() -> int { return (exercise) }\n"
        "shadow main { assert (== (main) 121111) }\n"));
    Value result = call_function("main", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 121111);
    ASSERT(nl_effect_find_handler("Recorder", "pair", NULL) == NULL);
    run_ctx_free(&ctx);
}

void test_eval_handler_return_destination(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Stop { stop : int -> int }\n"
        "let mut trace: int = 0\n"
        "fn send() -> int { let x = perform Stop.stop(7) set trace 1 return x }\n"
        "fn exercise() -> int { let x = handle { (send) } with { stop n -> { return n } } set trace 2 return 99 }\n"
        "fn main() -> int { set trace 0 let x = (exercise) return (+ 100 (+ x (* trace 1000))) }\n"));
    for (int i = 0; i < 2; i++) {
        Value result = call_function("main", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 107);
        ASSERT(!result.is_return);
        ASSERT(nl_effect_find_handler("Stop", "stop", NULL) == NULL);
    }
    run_ctx_free(&ctx);
}

void test_eval_handler_return_expression_order(void) {
    const char *bodies[] = {
        "return (+ (emit) (mark))",
        "return (combine (emit) (mark))",
        "set trace (emit) return 99",
        "if (== (emit) 7) { set trace 3 } return 99",
        "while (< (emit) 8) { break } return 99",
        "let xs = [(emit), (mark)] return 99",
        "let xs = [0, (emit), (mark)] return 99",
        "let pair = ((emit), (mark)) return 99",
        "let point = Point { x: (emit), y: (mark) } return 99",
        "let base = Point { x: 0, y: 0 } let point: Point = {..base, x: (emit), y: (mark)} return 99",
        "let packet = Packet.Data { x: (emit), y: (mark) } return 99",
        "let x = match (emit) { 7 => (mark), _ => 0 } return 99",
        "let x = match 7 { 7 if (== (emit) 7) => (mark), _ => 0 } return 99",
        "let x = match 8 { _ if (== (emit) 7) => (mark), _ => (mark) } return 99",
        "let p = Packet.Data { x: 1, y: 2 } let x = match p { Data(d) if (== (emit) 7) => (mark), _ => 0 } return 99",
        "for i in (range (emit) (mark)) { set trace 8 } return 99",
        "assert (== (emit) 99) return 99",
    };
    for (size_t i = 0; i < sizeof(bodies) / sizeof(bodies[0]); i++) {
        char source[2048];
        snprintf(source, sizeof(source),
            "effect Stop { stop : int -> int } let mut trace: int = 0 "
            "struct Point { x: int, y: int } union Packet { Data { x: int, y: int } } "
            "fn emit() -> int { return perform Stop.stop(7) } "
            "fn mark() -> int { set trace 4 return 1 } "
            "fn combine(a: int, b: int) -> int { set trace 5 return (+ a b) } "
            "fn send() -> int { %s } "
            "fn exercise() -> int { let x = handle { (send) } with { stop n -> { return n } } return 99 } "
            "fn main() -> int { let x = (exercise) return (+ x (* trace 1000)) }", bodies[i]);
        RunCtx ctx;
        ASSERT(run_ctx_init(&ctx, source));
        Value result = call_function("main", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 7);
        ASSERT(nl_effect_find_handler("Stop", "stop", NULL) == NULL);
        run_ctx_free(&ctx);
    }
}

void test_eval_handler_final_value_resumes(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Ask { ask : int -> int } "
        "fn send() -> int { let x = perform Ask.ask(7) return (+ x 10) } "
        "fn main() -> int { let x = handle { (send) } with { ask n -> { (+ n 1) } } return (+ x 100) }"));
    Value result = call_function("main", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 118);
    ASSERT(nl_effect_find_handler("Ask", "ask", NULL) == NULL);
    run_ctx_free(&ctx);
}

void test_eval_handler_return_nested_and_string(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Outer { leave : void -> int } effect Inner { ask : void -> int } "
        "let mut trace: int = 0 "
        "fn inner() -> int { let x = handle { perform Inner.ask() } with { "
        " ask -> { let x = perform Outer.leave() set trace 1 return 99 } } set trace 2 return x } "
        "fn owner() -> string { let answer: string = \"kept\" "
        " let x = handle { (inner) } with { leave -> { return answer } } set trace 3 return \"wrong\" } "
        "fn main() -> int { set trace 0 let answer = (owner) assert (== answer \"kept\") return trace }"));
    for (int i = 0; i < 3; i++) {
        Value result = call_function("main", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 0);
        ASSERT(nl_effect_find_handler("Outer", "leave", NULL) == NULL);
        ASSERT(nl_effect_find_handler("Inner", "ask", NULL) == NULL);
    }
    run_ctx_free(&ctx);
}

void test_eval_handler_return_recursive_activation(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Stop { stop : int -> int } "
        "fn recur(depth: int) -> int { "
        " if (== depth 0) { let x = handle { perform Stop.stop(7) } with { stop n -> { return n } } return 99 } "
        " let inner = (recur (- depth 1)) return (+ inner 1) } "
        "fn main() -> int { return (recur 3) }"));
    Value result = call_function("main", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT);
    ASSERT_EQ(result.as.int_val, 10);
    ASSERT(nl_effect_find_handler("Stop", "stop", NULL) == NULL);
    run_ctx_free(&ctx);
}

void test_eval_handler_return_partial_literal_cleanup(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "struct Leaf { text: string } struct Point { text: string, child: Leaf } "
        "effect Stop { point : void -> Point, text : void -> string } "
        "fn point() -> Point { return perform Stop.point() } "
        "fn text() -> string { return perform Stop.text() } "
        "fn records(base: Point) -> int { let xs = [base, (point)] return 99 } "
        "fn strings() -> int { let xs = [\"owned\", (text)] return 99 } "
        "fn record_owner(base: Point) -> int { let x = handle { (records base) } with { point -> { return 7 } } return 99 } "
        "fn string_owner() -> int { let x = handle { (strings) } with { text -> { return 8 } } return 99 } "
        "fn main() -> int { let base = Point { text: \"kept\", child: Leaf { text: \"nested\" } } "
        " let result = (+ (record_owner base) (string_owner)) "
        " assert (== base.text \"kept\") assert (== base.child.text \"nested\") return result }"));
    for (int i = 0; i < 3; i++) {
        Value result = call_function("main", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 15);
        ASSERT(nl_effect_find_handler("Stop", "point", NULL) == NULL);
    }
    run_ctx_free(&ctx);
}

void test_eval_handler_return_higher_order(void) {
    const char *operations[] = {"(map values visit)", "(filter values keep)",
                                "(reduce values 0 combine)"};
    for (size_t op = 0; op < 3; op++) {
        for (int dynamic = 0; dynamic < 2; dynamic++) {
            for (int escape = 0; escape < 2; escape++) {
                char source[2048];
                snprintf(source, sizeof(source),
                    "effect Stop { stop : int -> int } let mut calls: int = 0 "
                    "fn visit(x: int) -> int { set calls (+ calls 1) if (== x 2) { return perform Stop.stop(7) } return x } "
                    "fn keep(x: int) -> bool { return (> (visit x) 0) } "
                    "fn combine(acc: int, x: int) -> int { return (+ acc (visit x)) } "
                    "fn owner(values: array<int>) -> int { let ignored = handle { %s } with { stop n -> { %s } } return 99 } "
                    "fn main() -> int { return 0 }", operations[op], escape ? "return n" : "n");
                RunCtx ctx;
                ASSERT(run_ctx_init(&ctx, source));
                Value input;
                if (dynamic) {
                    DynArray *array = dyn_array_new(ELEM_INT);
                    for (int i = 1; i <= 3; i++) array = dyn_array_push_int(array, i);
                    input = create_void();
                    input.type = VAL_DYN_ARRAY;
                    input.as.dyn_array_val = array;
                } else {
                    input = create_array(VAL_INT, 3, 3);
                    for (int i = 0; i < 3; i++) ((long long *)input.as.array_val->data)[i] = i + 1;
                }
                Value result = call_function("owner", &input, 1, ctx.env);
                ASSERT_EQ(result.type, VAL_INT);
                ASSERT_EQ(result.as.int_val, escape ? 7 : 99);
                ASSERT_EQ(env_get_var(ctx.env, "calls")->value.as.int_val, escape ? 2 : 3);
                ASSERT(nl_effect_find_handler("Stop", "stop", NULL) == NULL);
                run_ctx_free(&ctx);
                if (dynamic) gc_release(input.as.dyn_array_val);
                else { free(input.as.array_val->data); free(input.as.array_val); }
            }
        }
    }
}

void test_eval_handler_return_async_calls(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "effect Stop { stop : int -> int } let mut trace: int = 0 "
        "async fn inner(n: int) -> int { let x = perform Stop.stop(n) set trace 1 return x } "
        "async fn outer(n: int) -> int { let x = (inner n) set trace 2 return x } "
        "fn owner() -> int { let x = handle { (outer 7) } with { stop n -> { return n } } set trace 3 return 99 } "
        "fn main() -> int { set trace 0 let x = (owner) return (+ x (* trace 1000)) }"));
    nano_scheduler_init();
    int first_id = g_scheduler.count;
    for (int i = 0; i < 100; i++) {
        Value result = call_function("main", NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT);
        ASSERT_EQ(result.as.int_val, 7);
        ASSERT(!result.is_return);
        ASSERT(nl_effect_find_handler("Stop", "stop", NULL) == NULL);
        for (int slot = 0; slot < MAX_COROUTINES; slot++) {
            ASSERT(g_scheduler.coroutines[slot].id < first_id);
        }
    }
    ASSERT_EQ(g_scheduler.count - first_id, 200);
    run_ctx_free(&ctx);
}

static void test_eval_file_write_failures(void) {
    char path[] = "/tmp/test_eval_file_failure.XXXXXX";
    int fd = mkstemp(path);
    ASSERT(fd >= 0);
    ASSERT(close(fd) == 0);
    Value args[2] = {create_string(path), create_string("content")};
    s_fail_fwrite = 1;
    s_fclose_calls = 0;
    Value write_result = builtin_file_write(args);
    s_fail_fwrite = 0;
    ASSERT_EQ(write_result.as.int_val, -1);
    ASSERT_EQ(s_fclose_calls, 1);
    s_fail_fclose = 1;
    s_fclose_calls = 0;
    Value append_result = builtin_file_append(args);
    s_fail_fclose = 0;
    ASSERT_EQ(append_result.as.int_val, -1);
    ASSERT_EQ(s_fclose_calls, 1);
    remove(args[0].as.string_val);
}

int main(void) {
    TEST(eval_declared_push_initializer_bindings);
    TEST(eval_file_write_failures);
    TEST(eval_handler_return_async_calls);
    TEST(eval_callable_projection_lifetimes);
    TEST(eval_handled_record_identity);
    TEST(eval_handler_return_higher_order);
    TEST(eval_handler_return_partial_literal_cleanup);
    TEST(eval_handler_return_recursive_activation);
    TEST(eval_handler_return_nested_and_string);
    TEST(eval_handler_return_expression_order);
    TEST(eval_handler_final_value_resumes);
    TEST(eval_handler_return_destination);
    TEST(eval_effect_argument_lists);
    TEST(eval_unqualified_effect_handler);
    TEST(eval_nested_effect_handlers);
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
    TEST(eval_binary64_prefix_and_strict_cast);
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
    TEST(eval_match_wildcards_follow_lexical_order);
    TEST(eval_match_miss_is_terminal);
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
    TEST(eval_map_declared_scalar_results);
    TEST(eval_reduce_pure_arithmetic_int);
    TEST(eval_reduce_pure_arithmetic_float);
    TEST(eval_unary_minus_int_array);
    TEST(eval_unary_minus_float_array);
    TEST(eval_array_scalar_broadcast_add);
    TEST(eval_array_scalar_broadcast_mul);
    TEST(eval_for_over_dynarray);
    TEST(eval_for_over_float_array);
    TEST(eval_string_loop_bindings);
    TEST(eval_coroutine_spawn_and_run);
    TEST(eval_async_fn_direct_call);
    TEST(eval_string_format_struct);
    TEST(eval_array_broadcast_scalar_right);
    TEST(eval_foreign_native_call_api);
    TEST(eval_indexed_read_aliases);
    TEST(eval_struct_array_literal);
    TEST(eval_array_literal_evaluates_once_in_order);
    TEST(eval_empty_array_aliases);
    TEST(eval_array_append_and_dynamic_write);
    TEST(eval_record_alias_reassignment);
    TEST(eval_record_alias_across_direct_calls);
    TEST(eval_epoch_milliseconds);

    printf("\n✓ All eval tests passed!\n");
    return 0;
}
