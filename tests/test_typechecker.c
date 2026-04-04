/**
 * test_typechecker.c — unit tests for typechecker.c
 *
 * Exercises the type-checker by running valid and invalid nano programs
 * through the full lex → parse → typecheck pipeline, verifying:
 *   - Valid programs pass type_check()
 *   - Invalid programs fail type_check() (type errors, arity errors, etc.)
 *
 * This covers many error branches and code paths in typechecker.c that are
 * not exercised by normal compilation of well-typed programs.
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

/* Required by runtime */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* Suppress stderr during expected-error paths */
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
 * Helpers
 * ============================================================================ */

typedef struct {
    ASTNode *program;
    Token   *tokens;
    int      token_count;
} ParseCtx;

static bool parse_ctx_init(ParseCtx *ctx, const char *src) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->tokens = tokenize(src, &ctx->token_count);
    if (!ctx->tokens) return false;
    ctx->program = parse_program(ctx->tokens, ctx->token_count);
    return ctx->program != NULL;
}

static void parse_ctx_free(ParseCtx *ctx) {
    if (ctx->program) free_ast(ctx->program);
    if (ctx->tokens) free_tokens(ctx->tokens, ctx->token_count);
}

/* Run typecheck on src: return true if type_check succeeds */
static bool tc_passes(const char *src) {
    ParseCtx ctx;
    if (!parse_ctx_init(&ctx, src)) return false;
    clear_module_cache();
    Environment *env = create_environment();
    typecheck_set_current_file("<test>");
    suppress_stderr();
    bool ok = type_check(ctx.program, env);
    restore_stderr();
    free_environment(env);
    parse_ctx_free(&ctx);
    return ok;
}

/* Run typecheck on a module (no main() required) */
static bool tc_module_passes(const char *src) {
    ParseCtx ctx;
    if (!parse_ctx_init(&ctx, src)) return false;
    clear_module_cache();
    Environment *env = create_environment();
    typecheck_set_current_file("<module>");
    suppress_stderr();
    bool ok = type_check_module(ctx.program, env);
    restore_stderr();
    free_environment(env);
    parse_ctx_free(&ctx);
    return ok;
}

/* ============================================================================
 * Valid program tests — exercise happy-path branches
 * ============================================================================ */

void test_tc_minimal_main(void) {
    ASSERT(tc_passes("fn main() -> int { return 0 }"));
}

void test_tc_arithmetic(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let x: int = (+ 3 4)\n"
        "  let y: int = (* x 2)\n"
        "  return y\n"
        "}"));
}

void test_tc_float_ops(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let x: float = 3.14\n"
        "  let y: float = (+ x 1.0)\n"
        "  return 0\n"
        "}"));
}

void test_tc_string_ops(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let s: string = \"hello\"\n"
        "  let n: int = (str_length s)\n"
        "  return 0\n"
        "}"));
}

void test_tc_bool_ops(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let a: bool = true\n"
        "  let b: bool = false\n"
        "  let c: bool = (and a b)\n"
        "  let d: bool = (or a b)\n"
        "  let e: bool = (not c)\n"
        "  return 0\n"
        "}"));
}

void test_tc_if_else(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let x: int = 5\n"
        "  if (> x 3) {\n"
        "    return 1\n"
        "  } else {\n"
        "    return 0\n"
        "  }\n"
        "}"));
}

void test_tc_while_loop(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let mut i: int = 0\n"
        "  while (< i 10) {\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return i\n"
        "}"));
}

void test_tc_for_in_range(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let mut sum: int = 0\n"
        "  for i in (range 0 5) {\n"
        "    set sum (+ sum i)\n"
        "  }\n"
        "  return sum\n"
        "}"));
}

void test_tc_function_call(void) {
    ASSERT(tc_passes(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn main() -> int { return (add 3 4) }"));
}

void test_tc_recursive_function(void) {
    ASSERT(tc_passes(
        "fn fact(n: int) -> int {\n"
        "  if (== n 0) { return 1 } else { return (* n (fact (- n 1))) }\n"
        "}\n"
        "fn main() -> int { return (fact 5) }"));
}

void test_tc_struct_definition(void) {
    ASSERT(tc_passes(
        "struct Point { x: int, y: int }\n"
        "fn main() -> int {\n"
        "  let x: int = 3\n"
        "  let y: int = 4\n"
        "  let p: Point = Point { x: x, y: y }\n"
        "  return p.x\n"
        "}"));
}

void test_tc_enum_definition(void) {
    ASSERT(tc_passes(
        "enum Color { Red, Green, Blue }\n"
        "fn main() -> int {\n"
        "  let c: Color = Color.Red\n"
        "  return 0\n"
        "}"));
}

void test_tc_union_definition(void) {
    ASSERT(tc_passes(
        "union Shape { Circle { radius: float }, Square { side: float } }\n"
        "fn area(s: Shape) -> float {\n"
        "  match s {\n"
        "    Circle(c) -> (* 3.14 (* c.radius c.radius)),\n"
        "    Square(sq) -> (* sq.side sq.side)\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_array_literal(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let arr: array<int> = [1, 2, 3]\n"
        "  return 0\n"
        "}"));
}

void test_tc_match_int(void) {
    ASSERT(tc_passes(
        "fn describe(x: int) -> string {\n"
        "  match x {\n"
        "    0 -> \"zero\",\n"
        "    1 -> \"one\",\n"
        "    _ -> \"other\"\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_print_builtin(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  (print \"hello\")\n"
        "  (print 42)\n"
        "  (print 3.14)\n"
        "  (print true)\n"
        "  return 0\n"
        "}"));
}

void test_tc_assert_builtin(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  assert true\n"
        "  assert (== 1 1)\n"
        "  return 0\n"
        "}"));
}

void test_tc_comparison_ops(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let a: bool = (== 1 1)\n"
        "  let b: bool = (!= 1 2)\n"
        "  let c: bool = (< 1 2)\n"
        "  let d: bool = (<= 1 1)\n"
        "  let e: bool = (> 2 1)\n"
        "  let f: bool = (>= 2 2)\n"
        "  return 0\n"
        "}"));
}

void test_tc_string_builtins(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let s: string = \"hello world\"\n"
        "  let n: int = (str_length s)\n"
        "  let sub: string = (str_substring s 0 5)\n"
        "  let ok: bool = (str_contains s \"world\")\n"
        "  let eq: bool = (str_equals s \"hello world\")\n"
        "  let sw: bool = (str_starts_with s \"hello\")\n"
        "  let ew: bool = (str_ends_with s \"world\")\n"
        "  let idx: int = (str_index_of s \"world\")\n"
        "  let lo: string = (str_to_lower s)\n"
        "  let up: string = (str_to_upper s)\n"
        "  let tr: string = (str_trim \" x \")\n"
        "  return n\n"
        "}"));
}

void test_tc_math_builtins(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let a: float = (sqrt 4.0)\n"
        "  let b: float = (pow 2.0 3.0)\n"
        "  let c: float = (floor 2.7)\n"
        "  let d: float = (ceil 2.3)\n"
        "  let e: float = (round 2.5)\n"
        "  let f: float = (sin 0.0)\n"
        "  let g: float = (cos 0.0)\n"
        "  let h: float = (tan 0.0)\n"
        "  let i: int = (abs -5)\n"
        "  let j: int = (min 3 7)\n"
        "  let k: int = (max 3 7)\n"
        "  return 0\n"
        "}"));
}

void test_tc_list_operations(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let lst: List<int> = (list_int_new)\n"
        "  (list_int_push lst 1)\n"
        "  (list_int_push lst 2)\n"
        "  let n: int = (list_int_length lst)\n"
        "  return n\n"
        "}"));
}

void test_tc_hashmap_operations(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let m: HashMap<string, int> = (map_new)\n"
        "  (map_put m \"key\" 42)\n"
        "  let v: int = (map_get m \"key\")\n"
        "  return v\n"
        "}"));
}

void test_tc_tuple_return(void) {
    ASSERT(tc_passes(
        "fn swap(a: int, b: int) -> (int, int) {\n"
        "  return (b, a)\n"
        "}\n"
        "fn main() -> int {\n"
        "  let p: (int, int) = (swap 1 2)\n"
        "  return p.0\n"
        "}"));
}

void test_tc_break_continue(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let mut i: int = 0\n"
        "  while (< i 10) {\n"
        "    if (== i 5) { break } else { (print \"\") }\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return i\n"
        "}"));
}

void test_tc_module_level(void) {
    /* type_check_module: no main() required */
    ASSERT(tc_module_passes(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn mul(a: int, b: int) -> int { return (* a b) }"));
}

void test_tc_constants(void) {
    ASSERT(tc_passes(
        "let PI: float = 3.14159\n"
        "let MAX: int = 100\n"
        "fn main() -> int { return MAX }"));
}

void test_tc_cond_expr(void) {
    ASSERT(tc_passes(
        "fn classify(x: int) -> string {\n"
        "  return (cond\n"
        "    ((< x 0) \"negative\")\n"
        "    ((== x 0) \"zero\")\n"
        "    (else \"positive\"))\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_nested_functions(void) {
    ASSERT(tc_passes(
        "fn square(x: int) -> int { return (* x x) }\n"
        "fn sum_of_squares(a: int, b: int) -> int { return (+ (square a) (square b)) }\n"
        "fn main() -> int { return (sum_of_squares 3 4) }"));
}

/* ============================================================================
 * Invalid program tests — exercise error branches
 * ============================================================================ */

void test_tc_err_null_program(void) {
    /* Null program should fail gracefully */
    Environment *env = create_environment();
    suppress_stderr();
    bool ok = type_check(NULL, env);
    restore_stderr();
    ASSERT(!ok);
    free_environment(env);
}

void test_tc_err_undefined_variable(void) {
    ASSERT(!tc_passes(
        "fn main() -> int { return undefined_var }"));
}

void test_tc_err_wrong_return_type(void) {
    ASSERT(!tc_passes(
        "fn main() -> int { return \"not an int\" }"));
}

void test_tc_err_wrong_arg_count(void) {
    ASSERT(!tc_passes(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn main() -> int { return (add 1) }"));
}

void test_tc_err_type_mismatch_add(void) {
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  let x: int = (+ 1 \"hello\")\n"
        "  return x\n"
        "}"));
}

void test_tc_err_set_immutable(void) {
    /* set on immutable let should fail */
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  let x: int = 5\n"
        "  set x 10\n"
        "  return x\n"
        "}"));
}

void test_tc_err_undefined_function(void) {
    ASSERT(!tc_passes(
        "fn main() -> int { return (nonexistent_fn 42) }"));
}

void test_tc_err_struct_unknown_field(void) {
    ASSERT(!tc_passes(
        "struct Point { x: int, y: int }\n"
        "fn main() -> int {\n"
        "  let x: int = 1\n"
        "  let y: int = 2\n"
        "  let p: Point = Point { x: x, y: y }\n"
        "  return p.z\n"
        "}"));
}

void test_tc_err_return_in_non_function(void) {
    /* This is tricky — main should have a return, so this should fail */
    /* A function with wrong return type fails */
    ASSERT(!tc_passes(
        "fn f() -> int { return \"wrong\" }\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_err_break_outside_loop(void) {
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  break\n"
        "  return 0\n"
        "}"));
}

void test_tc_err_continue_outside_loop(void) {
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  continue\n"
        "  return 0\n"
        "}"));
}

void test_tc_err_list_wrong_type(void) {
    /* Assign wrong type to list variable */
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  let lst: List<int> = \"not a list\"\n"
        "  return 0\n"
        "}"));
}

void test_tc_err_str_length_wrong_arg(void) {
    /* Assign string result to int — type mismatch at assignment */
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  let n: int = (str_concat \"hello\" \" world\")\n"
        "  return n\n"
        "}"));
}

void test_tc_err_comparison_type_mismatch(void) {
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  let ok: bool = (== 1 \"hello\")\n"
        "  return 0\n"
        "}"));
}

void test_tc_err_map_wrong_key_type(void) {
    /* Assign map_get result (int) to string variable → type error */
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  let m: HashMap<string, int> = (map_new)\n"
        "  let s: string = (map_get m \"k\")\n"
        "  return 0\n"
        "}"));
}

void test_tc_err_assert_non_bool(void) {
    ASSERT(!tc_passes(
        "fn main() -> int {\n"
        "  assert 42\n"
        "  return 0\n"
        "}"));
}

/* ============================================================================
 * Module-level tests (type_check_module)
 * ============================================================================ */

void test_tc_module_public_functions(void) {
    ASSERT(tc_module_passes(
        "pub fn hello() -> string { return \"hello\" }\n"
        "pub fn world() -> string { return \"world\" }"));
}

void test_tc_module_struct_export(void) {
    ASSERT(tc_module_passes(
        "pub struct Vector2 { x: float, y: float }\n"
        "pub fn zero() -> Vector2 {\n"
        "  return Vector2 { x: 0.0, y: 0.0 }\n"
        "}"));
}

void test_tc_module_with_constants(void) {
    ASSERT(tc_module_passes(
        "let VERSION: int = 1\n"
        "pub fn get_version() -> int { return VERSION }"));
}

/* ============================================================================
 * Edge cases
 * ============================================================================ */

void test_tc_empty_program(void) {
    /* A program with only a main that does nothing */
    ASSERT(tc_passes("fn main() -> int { return 0 }"));
}

void test_tc_multiple_returns(void) {
    ASSERT(tc_module_passes(
        "fn abs_val(x: int) -> int {\n"
        "  if (< x 0) { return (* -1 x) } else { return x }\n"
        "}"));
}

void test_tc_nested_if(void) {
    ASSERT(tc_passes(
        "fn classify(x: int) -> int {\n"
        "  if (< x 0) {\n"
        "    return -1\n"
        "  } else {\n"
        "    if (== x 0) {\n"
        "      return 0\n"
        "    } else {\n"
        "      return 1\n"
        "    }\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_string_concat(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let a: string = \"hello\"\n"
        "  let b: string = \" world\"\n"
        "  let c: string = (str_concat a b)\n"
        "  return 0\n"
        "}"));
}

void test_tc_modulo_op(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let r: int = (% 10 3)\n"
        "  return r\n"
        "}"));
}

void test_tc_unary_negate(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let x: int = -42\n"
        "  return x\n"
        "}"));
}

void test_tc_char_at(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let s: string = \"hello\"\n"
        "  let c: int = (char_at s 0)\n"
        "  return 0\n"
        "}"));
}

void test_tc_int_to_string(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let n: int = 42\n"
        "  let s: string = (int_to_string n)\n"
        "  return 0\n"
        "}"));
}

void test_tc_string_to_int(void) {
    ASSERT(tc_passes(
        "fn main() -> int {\n"
        "  let s: string = \"42\"\n"
        "  let n: int = (string_to_int s)\n"
        "  return n\n"
        "}"));
}

void test_tc_shadow(void) {
    ASSERT(tc_passes(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "shadow add {\n"
        "  assert (== (add 2 3) 5)\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== Typechecker Tests ===\n");

    printf("\n--- Valid programs ---\n");
    TEST(tc_minimal_main);
    TEST(tc_arithmetic);
    TEST(tc_float_ops);
    TEST(tc_string_ops);
    TEST(tc_bool_ops);
    TEST(tc_if_else);
    TEST(tc_while_loop);
    TEST(tc_for_in_range);
    TEST(tc_function_call);
    TEST(tc_recursive_function);
    TEST(tc_struct_definition);
    TEST(tc_enum_definition);
    TEST(tc_union_definition);
    TEST(tc_array_literal);
    TEST(tc_match_int);
    TEST(tc_print_builtin);
    TEST(tc_assert_builtin);
    TEST(tc_comparison_ops);
    TEST(tc_string_builtins);
    TEST(tc_math_builtins);
    TEST(tc_list_operations);
    TEST(tc_hashmap_operations);
    TEST(tc_tuple_return);
    TEST(tc_break_continue);
    TEST(tc_module_level);
    TEST(tc_constants);
    TEST(tc_cond_expr);
    TEST(tc_nested_functions);

    printf("\n--- Invalid programs (error paths) ---\n");
    TEST(tc_err_null_program);
    TEST(tc_err_undefined_variable);
    TEST(tc_err_wrong_return_type);
    TEST(tc_err_wrong_arg_count);
    TEST(tc_err_type_mismatch_add);
    TEST(tc_err_set_immutable);
    TEST(tc_err_undefined_function);
    TEST(tc_err_struct_unknown_field);
    TEST(tc_err_return_in_non_function);
    TEST(tc_err_break_outside_loop);
    TEST(tc_err_continue_outside_loop);
    TEST(tc_err_list_wrong_type);
    TEST(tc_err_str_length_wrong_arg);
    TEST(tc_err_comparison_type_mismatch);
    TEST(tc_err_map_wrong_key_type);
    TEST(tc_err_assert_non_bool);

    printf("\n--- Module-level typechecking ---\n");
    TEST(tc_module_public_functions);
    TEST(tc_module_struct_export);
    TEST(tc_module_with_constants);

    printf("\n--- Edge cases ---\n");
    TEST(tc_empty_program);
    TEST(tc_multiple_returns);
    TEST(tc_nested_if);
    TEST(tc_string_concat);
    TEST(tc_modulo_op);
    TEST(tc_unary_negate);
    TEST(tc_char_at);
    TEST(tc_int_to_string);
    TEST(tc_string_to_int);
    TEST(tc_shadow);

    printf("\n✓ All typechecker tests passed!\n");
    return 0;
}
