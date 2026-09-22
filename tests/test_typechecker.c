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

void test_tc_format_template(void) {
    ASSERT(tc_passes("fn main() -> int { let s = (format \"%s %d\" \"ok\" 42) return 0 }"));
    ASSERT(tc_passes("fn main() -> int { let template = \"plain\" let s = (format template) return 0 }"));
    ASSERT(!tc_passes("fn main() -> int { let s = (format 42) return 0 }"));
    ASSERT(!tc_passes("fn main() -> int { let s = (format true 42) return 0 }"));
    ASSERT(!tc_passes("fn main() -> int { let template = 3.5 let s = (format template) return 0 }"));
    ASSERT(!tc_passes("fn template() -> int { return 42 } fn main() -> int { let s = (format (template)) return 0 }"));
    ASSERT(!tc_passes("fn main() -> int { let s = (format [1, 2]) return 0 }"));
    ASSERT(!tc_passes("fn main() -> int { let s = (format) return 0 }"));
}

void test_tc_array_index_contract(void) {
    const char *names[] = {"at", "array_get"};
    const char *bad_indices[] = {"\"hello\"", "true", "1.5", "[0]", "missing"};
    char source[512];
    for (size_t n = 0; n < 2; n++) {
        snprintf(source, sizeof(source), "fn main() -> int { return (%s [1, 2] 0) }", names[n]);
        ASSERT(tc_passes(source));
        snprintf(source, sizeof(source), "fn main() -> int { let index: u8 = 0 let s: string = (%s [\"ok\"] index) return 0 }", names[n]);
        ASSERT(tc_passes(source));
        snprintf(source, sizeof(source), "fn main() -> int { let inner: array<int> = [42] let rows: array<array<int>> = [inner] let cube: array<array<array<int>>> = [rows] return (%s (%s (%s cube 0) 0) 0) }", names[n], names[n], names[n]);
        ASSERT(tc_passes(source));
        for (size_t i = 0; i < sizeof(bad_indices) / sizeof(bad_indices[0]); i++) {
            snprintf(source, sizeof(source), "fn main() -> int { return (%s [1, 2] %s) }", names[n], bad_indices[i]);
            ASSERT(!tc_passes(source));
        }
        snprintf(source, sizeof(source), "fn main() -> int { return (%s 42 0) }", names[n]);
        ASSERT(!tc_passes(source));
        snprintf(source, sizeof(source), "fn main() -> int { return (%s [1, 2]) }", names[n]);
        ASSERT(!tc_passes(source));
        snprintf(source, sizeof(source), "fn main() -> int { return (%s [1, 2] 0 1) }", names[n]);
        ASSERT(!tc_passes(source));
    }
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

void test_tc_returned_function_signature(void) {
    ASSERT(tc_passes(
        "fn plus_one(x: float) -> float { return (+ x 1.0) }\n"
        "fn choose() -> fn(float) -> float { return plus_one }\n"
        "fn main() -> int { let x: float = ((choose) 2.0) return 0 }"));
}

void test_tc_function_variable_alias_signature(void) {
    ASSERT(tc_passes(
        "fn plus_one(x: float) -> float { return (+ x 1.0) }\n"
        "fn main() -> int { let f: fn(float) -> float = plus_one "
        "let g: fn(float) -> float = f let x: float = (g 2.0) return 0 }"));
    ASSERT(!tc_passes(
        "fn plus_one(x: float) -> float { return (+ x 1.0) }\n"
        "fn main() -> int { let f: fn(float) -> float = plus_one "
        "let g: fn(int) -> int = f return 0 }"));
}

void test_tc_map_result_signature(void) {
    ASSERT(tc_passes("fn f(x: int) -> float { return 1.0 } "
        "fn choose() -> fn(int) -> float { return f } "
        "fn main() -> int { let y: float = (at (map [1] (choose)) 0) return 0 }"));
    ASSERT(!tc_passes("fn f(x: int) -> int { return x } "
        "fn main() -> int { let y = (map 1 f) return 0 }"));
    ASSERT(!tc_passes("fn main() -> int { let y = (map [1] 2) return 0 }"));
    ASSERT(!tc_passes("fn f(x: int) -> void { return } "
        "fn main() -> int { let y = (map [1] f) return 0 }"));
    ASSERT(tc_passes("fn f(x: int) -> float { return 1.0 } "
        "fn main() -> int { let values = (map [1] f) "
        "let y: float = (array_get values 0) return 0 }"));
    ASSERT(!tc_passes("fn f(x: int) -> float { return 1.0 } "
        "fn main() -> int { let values: array<int> = (map [1] f) return 0 }"));
    ASSERT(tc_passes("fn f(x: int) -> float { return 1.0 } "
        "fn main() -> int { let y: float = (array_get (map [1] f) 0) return 0 }"));
    ASSERT(tc_passes("fn f(x: int) -> float { return 1.0 } "
        "fn main() -> int { let g: fn(int) -> float = f "
        "let y: float = (at (map [1] g) 0) return 0 }"));
    ASSERT(!tc_passes("fn f(x: int) -> float { return 1.0 } "
        "fn main() -> int { let y: int = (array_get (map [1] f) 0) return 0 }"));
    ASSERT(!tc_passes("fn f(x: float) -> float { return x } "
        "fn main() -> int { let y = (map [1] f) return 0 }"));
    ASSERT(!tc_passes("fn f(x: int, y: int) -> int { return x } "
        "fn main() -> int { let y = (map [1] f) return 0 }"));
    ASSERT(!tc_passes("fn f(x: int) -> int { return x } "
        "fn main() -> int { let y = (map [1] f 3) return 0 }"));
}

void test_tc_err_returned_function_argument_type(void) {
    ASSERT(!tc_passes(
        "fn plus_one(x: float) -> float { return (+ x 1.0) }\n"
        "fn choose() -> fn(float) -> float { return plus_one }\n"
        "fn main() -> int { let x: float = ((choose) 2) return 0 }"));
}

void test_tc_err_returned_function_arity(void) {
    ASSERT(!tc_passes(
        "fn plus_one(x: float) -> float { return (+ x 1.0) }\n"
        "fn choose() -> fn(float) -> float { return plus_one }\n"
        "fn main() -> int { let x: float = ((choose) 2.0 3.0) return 0 }"));
}

/* ============================================================================
 * Pure fn tests — purity enforcement via check_purity()
 * ============================================================================ */

void test_tc_pure_fn_simple(void) {
    /* Pure fn with only immutable lets and arithmetic — should pass */
    ASSERT(tc_passes(
        "pure fn square(x: int) -> int { return (* x x) }\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_fn_calls_pure(void) {
    /* Pure fn calling another pure fn — should pass */
    ASSERT(tc_passes(
        "pure fn double(x: int) -> int { return (* x 2) }\n"
        "pure fn quad(x: int) -> int { return (double (double x)) }\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_fn_recursive(void) {
    /* Pure recursive fn — should pass */
    ASSERT(tc_passes(
        "pure fn fib(n: int) -> int {\n"
        "  if (<= n 1) { return n } else {\n"
        "    return (+ (fib (- n 1)) (fib (- n 2)))\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_extern_fn(void) {
    /* Pure extern fn — should pass typechecking as a module */
    ASSERT(tc_module_passes(
        "pub pure extern fn fabs(x: float) -> float\n"));
}

void test_tc_pure_fn_violation_set(void) {
    /* set inside pure fn — should fail */
    ASSERT(!tc_passes(
        "pure fn bad(x: int) -> int {\n"
        "  let mut v: int = x\n"
        "  set v (+ v 1)\n"
        "  return v\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_fn_violation_let_mut(void) {
    /* let mut inside pure fn — should fail */
    ASSERT(!tc_passes(
        "pure fn bad(x: int) -> int {\n"
        "  let mut acc: int = 0\n"
        "  return acc\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_fn_violation_while(void) {
    /* while loop inside pure fn — should fail */
    ASSERT(!tc_passes(
        "pure fn bad(n: int) -> int {\n"
        "  while (> n 0) { return 0 }\n"
        "  return n\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_fn_violation_for(void) {
    /* for loop inside pure fn — should fail */
    ASSERT(!tc_passes(
        "pure fn bad(n: int) -> int {\n"
        "  for i in (range 0 n) { return 0 }\n"
        "  return n\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_fn_violation_print(void) {
    /* print inside pure fn — should fail */
    ASSERT(!tc_passes(
        "pure fn bad(x: int) -> int {\n"
        "  print x\n"
        "  return x\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_pure_fn_violation_calls_impure(void) {
    /* pure fn calling an impure fn — should fail */
    ASSERT(!tc_passes(
        "fn impure(x: int) -> int {\n"
        "  let mut v: int = x\n"
        "  set v (+ v 1)\n"
        "  return v\n"
        "}\n"
        "pure fn bad(x: int) -> int { return (impure x) }\n"
        "fn main() -> int { return 0 }"));
}

void test_tc_impure_fn_not_affected(void) {
    /* Regular fn with let mut / set / while — should still pass */
    ASSERT(tc_passes(
        "fn mutable_ok(n: int) -> int {\n"
        "  let mut acc: int = 0\n"
        "  let mut i: int = 0\n"
        "  while (< i n) {\n"
        "    set acc (+ acc i)\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return acc\n"
        "}\n"
        "fn main() -> int { return 0 }"));
}

/* ============================================================================
 * main
 * ============================================================================ */

void test_tc_nested_return_context(void) {
    const char *valid =
        "fn outer() -> int {\n"
        "  fn inner() -> float { return 2.5 }\n"
        "  assert (== (inner) 2.5)\n"
        "  return 7\n"
        "}\n"
        "shadow outer { assert (== (outer) 7) }\n"
        "fn main() -> int { return (outer) }\n"
        "shadow main { assert (== (main) 7) }\n";
    const char *invalid =
        "fn outer() -> int {\n"
        "  fn inner() -> bool { return 7 }\n"
        "  return 0\n"
        "}\n"
        "shadow outer { assert (== (outer) 0) }\n"
        "fn main() -> int { return (outer) }\n"
        "shadow main { assert (== (main) 0) }\n";
    ASSERT(tc_passes(valid));
    ASSERT(tc_module_passes(valid));
    ASSERT(!tc_passes(invalid));
    ASSERT(!tc_module_passes(invalid));
    ASSERT(!tc_passes(
        "fn main() -> int { while true {\n"
        "  fn inner() -> void { break }\n"
        "  break\n"
        "} return 0 }\n"));
}

void test_tc_handler_effect_inference(void) {
    const char *a = "effect Alpha { common : int -> int }\n";
    const char *b = "effect Beta { common : int -> int, unique : int -> int }\n";
    const char *valid = "fn main() -> int { return handle { 99 } with { common x -> { x } unique x -> { x } } }";
    char source[1024];
    snprintf(source, sizeof(source), "%s%s%s", a, b, valid);
    ASSERT(tc_passes(source));
    snprintf(source, sizeof(source), "%s%s%s", b, a, valid);
    ASSERT(tc_passes(source));
    snprintf(source, sizeof(source), "%s%sfn main() -> int { return handle { 99 } with { common x -> { x } } }", a, b);
    ASSERT(!tc_passes(source));
    snprintf(source, sizeof(source), "%sfn main() -> int { return handle { 99 } with { common x -> { x } common y -> { y } } }", a);
    ASSERT(!tc_passes(source));
    snprintf(source, sizeof(source), "%sfn main() -> int { return handle { 99 } with { common -> { 0 } } }", a);
    ASSERT(!tc_passes(source));
    snprintf(source, sizeof(source), "%sfn main() -> int { return handle { 99 } with { common x y -> { x } } }", a);
    ASSERT(!tc_passes(source));
}

void test_tc_perform_signatures(void) {
    ASSERT(tc_passes("effect Tick { now : void -> void } fn main() -> int { perform Tick.now return 0 }"));
    ASSERT(tc_passes("effect Pair { emit : int string -> void } fn main() -> int { perform Pair.emit(1 \"ok\") return 0 }"));
    ASSERT(!tc_passes("effect Pair { emit : int string -> void } fn main() -> int { perform Pair.emit(1 2) return 0 }"));
    ASSERT(!tc_passes("effect Pair { emit : int string -> void } fn main() -> int { perform Pair.emit(1) return 0 }"));
    ASSERT(!tc_passes("effect Pair { emit : int string -> void } fn main() -> int { perform Pair.emit(1 \"ok\" 3) return 0 }"));
    ASSERT(!tc_passes("fn main() -> int { perform Missing.emit(1) return 0 }"));
    ASSERT(!tc_passes("effect Recorder { emit : int -> void } fn main() -> int { perform Recorder.missing(1) return 0 }"));
    ASSERT(!tc_passes("effect Recorder { emit : int -> void } fn main() -> int { perform Recorder.emit(\"wrong\") return 0 }"));
    ASSERT(!tc_passes("effect Recorder { emit : int -> void } fn main() -> int { let value = perform Recorder.emit(true) return 0 }"));
    ASSERT(!tc_passes("effect Recorder { emit : int -> void } fn main() -> int { perform Recorder.emit() return 0 }"));
    ASSERT(!tc_passes("effect Clock { now : void -> int } fn main() -> int { perform Clock.now(1) return 0 }"));
    ASSERT(tc_passes("fn main() -> int { return perform Clock.now() } effect Clock { now : void -> int }"));
    ASSERT(tc_passes("effect Echo { value : int -> int } fn main() -> int { return perform Echo.value(7) }"));
    ASSERT(!tc_passes("effect Echo { value : int -> string } fn main() -> int { return perform Echo.value(7) }"));
}

void test_tc_handler_parameter_metadata(void) {
    ASSERT(tc_passes("struct Point { x: int } effect Visit { point : Point -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { point p -> { let x: int = p.x } } return 0 }"));
    ASSERT(tc_passes("effect Visit { values : array<string> -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { values xs -> { let x: string = (at xs 0) } } return 0 }"));
    ASSERT(!tc_passes("effect Visit { values : array<string> -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { values xs -> { let x: int = (at xs 0) } } return 0 }"));
    ASSERT(!tc_passes("struct Point { x: string } effect Visit { point : Point -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { point p -> { let x: int = p.x } } return 0 }"));
    ASSERT(tc_passes("struct Point { x: int } struct Other { x: string } effect Visit { point : Point -> void } "
        "fn main() -> int { let p = Other { x: \"outer\" } let ignored = handle { 0 } with { point p -> { let x: int = p.x } } let outside: string = p.x return 0 }"));
    ASSERT(tc_passes("effect Visit { rows : array<array<string>> -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { rows xs -> { let x: string = (at (at xs 0) 0) } } return 0 }"));
    ASSERT(!tc_passes("effect Visit { rows : array<array<string>> -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { rows xs -> { let x: int = (at (at xs 0) 0) } } return 0 }"));
    ASSERT(tc_passes("effect Visit { callback : fn(int) -> string -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { callback f -> { let x: string = (f 7) } } return 0 }"));
    ASSERT(!tc_passes("effect Visit { callback : fn(int) -> string -> void } "
        "fn main() -> int { let ignored = handle { 0 } with { callback f -> { let x: string = (f true) } } return 0 }"));
}

/* I check identities without executing callbacks outside their runtime profile. */
void test_tc_reduce_exact_identities(void) {
    const char *kinds[] = {"int", "float", "bool", "string", "array<int>", "Point", "Box<int>", "Choice"};
    for (size_t i = 0; i < sizeof kinds / sizeof kinds[0]; ++i) {
        char source[2048];
        snprintf(source, sizeof source,
            "struct Point { value:int } union Box<T> { Value { value:T } } enum Choice { One, Two } "
            "fn fold(a:%s,b:%s)->%s{return a} "
            "fn apply(xs:array<%s>,initial:%s)->%s{return (reduce xs initial fold)}",
            kinds[i], kinds[i], kinds[i], kinds[i], kinds[i], kinds[i]);
        printf(" [%s]", kinds[i]); fflush(stdout);
        ASSERT(tc_module_passes(source));
    }
    ASSERT(tc_module_passes("enum Choice { One, Two } "
        "fn fold(a:Choice,b:Choice)->Choice{return a} "
        "fn apply(initial:Choice)->Choice{return (reduce [Choice.One,Choice.Two] initial fold)}"));
    ASSERT(tc_module_passes("fn fold(a:int,b:int)->int{return (+ a b)} "
        "fn main()->int{let xs:array<int> = [] return (reduce xs 9 fold)}"));
    ASSERT(tc_module_passes("fn fold(a:int,b:int)->int{return a} "
        "fn choose()->fn(int,int)->int{return fold} "
        "fn main()->int{let local:fn(int,int)->int=fold "
        "let x:int=(reduce [1] 0 local) return (reduce [1] x (choose))}"));
    ASSERT(tc_module_passes("fn fold(a:float,b:float)->float{return a} "
        "fn apply(xs:array<int>,f:fn(int,int)->int)->int{"
        "let fold:fn(int,int)->int=f return (reduce xs 0 fold)}"));
    ASSERT(tc_module_passes("fn fold(a:array<int>,b:array<int>)->array<int>{return a} "
        "fn main()->int{let x:array<int> = (reduce [[1],[2]] [0] fold) return 0}"));
}

/* I preserve lexical callable authority before projecting builtin reduce's initializer. */
void test_tc_reduce_record_result_identity(void) {
    const char *prefix="struct Point { value:int } struct Other { value:int } "
        "fn fold(a:Point,b:Point)->Point{return a} ";
    char source[2048];
    snprintf(source,sizeof source,"%sfn apply(xs:array<Point>,initial:Point)->Other{return (reduce xs initial fold)}",prefix);
    ASSERT(!tc_module_passes(source));
    snprintf(source,sizeof source,"%sfn apply(xs:array<Point>,initial:Point,reduce:fn(array<Point>,Point,int)->Other)->Other{return (reduce xs initial 7)}",prefix);
    ASSERT(tc_module_passes(source));
    snprintf(source,sizeof source,"%sfn apply(xs:array<Point>,initial:Point,reduce:fn(array<Point>,Point,int)->Other)->Point{return (reduce xs initial 7)}",prefix);
    ASSERT(!tc_module_passes(source));
}

void test_tc_reduce_global_callback_identity(void) {
    const char *source = "fn selected(a:float,b:float)->float{return (+ a b)} "
        "fn difference(a:float,b:float)->float{return (- a b)} "
        "let mut selected:fn(float,float)->float=difference "
        "fn initial()->float{set selected difference return 10.0} "
        "fn apply()->float{return (reduce [3.0] (initial) selected)} "
        "fn main()->int{let result:float=(apply) return 0}";
    ASSERT(tc_passes(source));
    ASSERT(tc_module_passes(source));
    ASSERT(!tc_module_passes("fn selected(a:int,b:int)->int{return a} "
        "fn floating(a:float,b:float)->float{return a} "
        "let selected:fn(float,float)->float=floating "
        "fn apply()->int{return (reduce [1] 0 selected)}"));
}

void test_tc_reduce_exact_refusals(void) {
    const char *cases[] = {
        "fn main()->int{return (reduce [1] 0)}",
        "fn fold(a:int,b:int)->int{return a} fn main()->int{return (reduce [1] 0 fold 4)}",
        "fn fold(a:int,b:int)->int{return a} fn main()->int{return (reduce 1 0 fold)}",
        "fn main()->int{return (reduce [1] 0 2)}",
        "fn fold(a:int,b:int)->int{return a} fn main()->int{let x=(reduce [1] [] fold) return 0}",
        "fn fold(a:int)->int{return a} fn main()->int{return (reduce [1] 0 fold)}",
        "fn fold(a:int,b:float)->int{return a} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}",
        "fn fold(a:float,b:float)->int{return 0} fn main()->int{let x=(reduce [1.0] 0.0 fold) return 0}",
        "fn fold(a:int,b:float)->int{return a} fn main()->int{return (reduce [1] 0 fold)}",
        "fn fold(a:int,b:int)->void{} fn main()->int{let x=(reduce [1] 0 fold) return 0}",
        "fn fold(a:int,b:int)->int{return a} fn main()->int{let fold:int=4 return (reduce [1] 0 fold)}",
        "fn fold(a:int,b:int)->int{return a} fn apply(f:fn(float,float)->float)->int{"
        "let fold:fn(float,float)->float=f let x=(reduce [1] 0 fold) return 0}",
        "enum Choice { One, Two } fn fold(a:int,b:int)->int{return a} "
        "fn apply(xs:array<Choice>)->int{return (reduce xs 0 fold)}",
        "struct A { x:int } struct B { x:int } fn fold(a:A,b:A)->A{return a} "
        "fn apply(xs:array<B>,initial:A)->A{return (reduce xs initial fold)}",
        "union Box<T> { Value { value:T } } fn fold(a:Box<int>,b:Box<int>)->Box<int>{return a} "
        "fn apply(xs:array<Box<bool>>,initial:Box<int>)->Box<int>{return (reduce xs initial fold)}",
        "fn fold(a:array<int>,b:array<int>)->array<int>{return a} "
        "fn main()->int{let x=(reduce [[true]] [0] fold) return 0}",
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        printf(" [%zu]", i); fflush(stdout);
        ASSERT(!tc_module_passes(cases[i]));
    }
}

/* I check the complete flat arithmetic family without executing refused routes. */
static void test_array_arithmetic_result_views(void) {
    const char *types[] = {"int", "float", "string"};
    const char *ops[] = {"+", "-", "*", "/", "%"};
    char source[1024];
    for (int type = 0; type < 3; ++type) {
        int count = type == 0 ? 5 : type == 1 ? 4 : 1;
        for (int op = 0; op < count; ++op) for (int route = 0; route < 3; ++route) {
            snprintf(source, sizeof source,
                "fn result(a:array<%s>, b:%s%s%s)->array<%s> { return (%s %s) } "
                "fn main()->int{return 0}", types[type], route == 0 ? "array<" : "",
                types[type], route == 0 ? ">" : "", types[type], ops[op],
                route == 2 ? "b a" : "a b");
            ASSERT(tc_passes(source));
        }
    }
    ASSERT(tc_passes("fn result(a:array<int>)->array<int>{return (- a)} fn main()->int{return 0}"));
    ASSERT(tc_passes("fn result(a:array<float>)->array<float>{return (- a)} fn main()->int{return 0}"));
    ASSERT(tc_passes("fn result(a:array<int>, b:u8)->array<int>{return (+ a b)} fn main()->int{return 0}"));
    ASSERT(tc_passes("fn result(a:array<int>, b:u8)->array<int>{return (- b a)} fn main()->int{return 0}"));
    ASSERT(tc_passes("enum Choice { One, Two } fn result(a:array<Choice>)->array<int>{return (- a)} fn main()->int{return 0}"));
    ASSERT(tc_passes("enum Choice { One, Two } fn result(a:array<Choice>, b:array<int>)->array<int>{return (+ a b)} fn main()->int{return 0}"));
    ASSERT(tc_passes("enum Choice { One, Two } fn result(a:array<int>, b:Choice)->array<int>{return (+ a b)} fn main()->int{return 0}"));
    ASSERT(tc_passes("fn result(a:array<int>)->array<int>{return (+ (- a) (* a 2))} fn main()->int{return 0}"));
    const char *refusals[] = {
        "fn result(a:array<int>)->array<float>{return (- a)}",
        "enum Choice { One, Two } fn result(a:array<Choice>)->array<Choice>{return (- a)}",
        "fn result(a:array<int>, b:float)->array<int>{return (+ a b)}",
        "fn result(a:array<float>, b:array<int>)->array<float>{return (+ a b)}",
        "fn result(a:array<float>)->array<float>{return (% a 2.0)}",
        "fn result(a:array<string>)->array<string>{return (- a)}",
        "fn result(a:array<string>)->array<string>{return (- a a)}",
        "fn result(a:array<bool>)->array<bool>{return (+ a a)}",
        "struct Item { value:int } fn result(a:array<Item>)->array<Item>{return (+ a a)}",
        "fn result(a:array<u8>)->array<int>{return (+ a a)}",
        "fn result(a:array<array<int>>)->array<array<int>>{return (+ a a)}",
        "fn result(a:array<array<int>>)->array<array<int>>{return (- a)}",
        "fn result(a:array<int>)->array<int>{return (+ a)}",
        "fn result(a:array<int>)->array<int>{return (+ a a a)}"
    };
    for (size_t i = 0; i < sizeof refusals / sizeof refusals[0]; ++i) {
        snprintf(source, sizeof source, "%s fn main()->int{return 0}", refusals[i]);
        ASSERT(!tc_passes(source));
    }
}

int main(void) {
    TEST(array_arithmetic_result_views);
    TEST(tc_handler_parameter_metadata);
    TEST(tc_perform_signatures);
    TEST(tc_handler_effect_inference);
    TEST(tc_nested_return_context);
    printf("=== Typechecker Tests ===\n");

    printf("\n--- Valid programs ---\n");
    TEST(tc_minimal_main);
    TEST(tc_format_template);
    TEST(tc_array_index_contract);
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
    TEST(tc_returned_function_signature);
    TEST(tc_function_variable_alias_signature);
    TEST(tc_map_result_signature);
    TEST(tc_reduce_exact_identities);
    TEST(tc_reduce_exact_refusals);
    TEST(tc_reduce_record_result_identity);
    TEST(tc_reduce_global_callback_identity);
    TEST(tc_err_returned_function_argument_type);
    TEST(tc_err_returned_function_arity);

    printf("\n--- Pure fn: purity enforcement ---\n");
    TEST(tc_pure_fn_simple);
    TEST(tc_pure_fn_calls_pure);
    TEST(tc_pure_fn_recursive);
    TEST(tc_pure_extern_fn);
    TEST(tc_pure_fn_violation_set);
    TEST(tc_pure_fn_violation_let_mut);
    TEST(tc_pure_fn_violation_while);
    TEST(tc_pure_fn_violation_for);
    TEST(tc_pure_fn_violation_print);
    TEST(tc_pure_fn_violation_calls_impure);
    TEST(tc_impure_fn_not_affected);

    printf("\n✓ All typechecker tests passed!\n");
    return 0;
}
