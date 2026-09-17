/**
 * test_parser.c — unit tests for parser.c
 *
 * Exercises the nanolang parser by parsing a wide variety of nano programs
 * and fragments, covering:
 *   - All major AST node types
 *   - Effects (handle/raise/resume)
 *   - Opaque types
 *   - Match guards
 *   - parse_repl_input
 *   - Error recovery (programs that fail to parse)
 *   - free_ast on complex ASTs
 */

#include "../src/nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_NOT_NULL(p) \
    if ((p) == NULL) { printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_NULL(p) \
    if ((p) != NULL) { printf("\n    FAILED: expected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %d != %d at line %d\n", (int)(a), (int)(b), __LINE__); exit(1); }

/* Required by runtime */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* Suppress stderr during parse error tests */
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

/* Helper: parse a nano program and return AST (NULL on failure) */
static ASTNode *parse_ok(const char *src) {
    int count = 0;
    Token *tokens = tokenize(src, &count);
    if (!tokens) return NULL;
    ASTNode *prog = parse_program(tokens, count);
    free_tokens(tokens, count);
    return prog;
}

/* Helper: parse with repl mode */
static ASTNode *parse_repl(const char *src) {
    int count = 0;
    Token *tokens = tokenize(src, &count);
    if (!tokens) return NULL;
    ASTNode *prog = parse_repl_input(tokens, count);
    free_tokens(tokens, count);
    return prog;
}

/* ============================================================================
 * Basic parsing tests
 * ============================================================================ */

/* I preserve each nested annotation rather than flattening its first name. */
void test_nested_generic_annotation_metadata(void) {
    ASTNode *program = parse_ok("fn identity(value: Box<Result<int,string>>) -> Box<Result<int,string>> { return value }");
    ASSERT_NOT_NULL(program);
    ASTNode *function = program->as.program.items[0];
    TypeInfo *parameter = function->as.function.params[0].type_info;
    TypeInfo *result = function->as.function.return_type_info;
    TypeInfo *roots[] = {parameter, result};
    for (int i = 0; i < 2; ++i) {
        TypeInfo *outer = roots[i];
        ASSERT_NOT_NULL(outer);
        ASSERT(strcmp(outer->generic_name, "Box") == 0);
        ASSERT_EQ(outer->type_param_count, 1);
        TypeInfo *inner = outer->type_params[0];
        ASSERT_EQ(inner->base_type, TYPE_UNION);
        ASSERT(strcmp(inner->generic_name, "Result") == 0);
        ASSERT_EQ(inner->type_param_count, 2);
        ASSERT_EQ(inner->type_params[0]->base_type, TYPE_INT);
        ASSERT_EQ(inner->type_params[1]->base_type, TYPE_STRING);
    }
    free_ast(program);

    program = parse_ok("fn inspect(value: Box<array<Result<int,types.Item>>>) -> int { return 0 }");
    ASSERT_NOT_NULL(program);
    parameter = program->as.program.items[0]->as.function.params[0].type_info;
    TypeInfo *array = parameter->type_params[0];
    ASSERT_EQ(array->base_type, TYPE_ARRAY);
    ASSERT_NOT_NULL(array->element_type);
    ASSERT(strcmp(array->element_type->generic_name, "Result") == 0);
    ASSERT(strcmp(array->element_type->type_params[1]->generic_name, "types.Item") == 0);
    free_ast(program);
}

void test_nested_generic_annotation_errors(void) {
    const char *cases[] = {
        "fn broken(value: Box<Result<int,string>) -> int { return 0 }",
        "fn broken(value: Box<Result<int,string> int>) -> int { return 0 }",
        "fn broken(value: Box<array<Result<int,string>>) -> int { return 0 }"
    };
    suppress_stderr();
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ASTNode *program = parse_ok(cases[i]);
        ASSERT_NULL(program);
    }
    restore_stderr();
}

void test_parse_minimal(void) {
    ASTNode *prog = parse_ok("fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASSERT_EQ(prog->type, AST_PROGRAM);
    free_ast(prog);
}

void test_parse_empty_program(void) {
    ASTNode *prog = parse_ok("");
    /* Empty program might parse OK (empty program node) or return NULL */
    if (prog) free_ast(prog);
}

void test_parse_arithmetic(void) {
    ASTNode *prog = parse_ok(
        "fn compute() -> int {\n"
        "  let a: int = (+ 1 (* 2 3))\n"
        "  let b: int = (- a (/ 6 2))\n"
        "  let c: int = (% b 3)\n"
        "  return c\n"
        "}\n"
        "fn main() -> int { return (compute) }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_all_comparisons(void) {
    ASTNode *prog = parse_ok(
        "fn f(x: int, y: int) -> bool {\n"
        "  let a: bool = (== x y)\n"
        "  let b: bool = (!= x y)\n"
        "  let c: bool = (< x y)\n"
        "  let d: bool = (<= x y)\n"
        "  let e: bool = (> x y)\n"
        "  let f: bool = (>= x y)\n"
        "  return a\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_bool_ops(void) {
    ASTNode *prog = parse_ok(
        "fn f(a: bool, b: bool) -> bool {\n"
        "  let c: bool = (and a b)\n"
        "  let d: bool = (or a b)\n"
        "  let e: bool = (not c)\n"
        "  return d\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_if_else_chain(void) {
    ASTNode *prog = parse_ok(
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
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_while_loop(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  let mut i: int = 0\n"
        "  while (< i 10) {\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return i\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_for_range(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  let mut sum: int = 0\n"
        "  for i in (range 0 10) {\n"
        "    set sum (+ sum i)\n"
        "  }\n"
        "  return sum\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_struct_def(void) {
    ASTNode *prog = parse_ok(
        "struct Point { x: int, y: int }\n"
        "struct Rect { origin: Point, width: int, height: int }\n"
        "fn area(r: Rect) -> int { return (* r.width r.height) }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_enum_def(void) {
    ASTNode *prog = parse_ok(
        "enum Color { Red, Green, Blue }\n"
        "enum Day { Mon, Tue, Wed, Thu, Fri, Sat, Sun }\n"
        "fn is_weekend(d: Day) -> bool {\n"
        "  return (or (== d Day.Sat) (== d Day.Sun))\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_union_def(void) {
    ASTNode *prog = parse_ok(
        "union Result<T> { Ok { value: T }, Err { message: string } }\n"
        "union Shape { Circle { radius: float }, Square { side: float }, Triangle { base: float, height: float } }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_match_expr(void) {
    ASTNode *prog = parse_ok(
        "fn describe(x: int) -> string {\n"
        "  match x {\n"
        "    0 -> \"zero\",\n"
        "    1 -> \"one\",\n"
        "    2 -> \"two\",\n"
        "    _ -> \"many\"\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_match_with_guards(void) {
    ASTNode *prog = parse_ok(
        "fn classify(x: int) -> string {\n"
        "  match x {\n"
        "    0 -> \"zero\",\n"
        "    _ if (> x 0) -> \"positive\",\n"
        "    _ -> \"negative\"\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_union_match(void) {
    ASTNode *prog = parse_ok(
        "union Shape { Circle { r: float }, Square { s: float } }\n"
        "fn area(sh: Shape) -> float {\n"
        "  match sh {\n"
        "    Circle(c) -> (* 3.14 (* c.r c.r)),\n"
        "    Square(sq) -> (* sq.s sq.s)\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_cond_expr(void) {
    ASTNode *prog = parse_ok(
        "fn f(x: int) -> string {\n"
        "  return (cond\n"
        "    ((< x 0) \"neg\")\n"
        "    ((== x 0) \"zero\")\n"
        "    (else \"pos\"))\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_array_literal(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "  return (array_get arr 0)\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_tuple_literal(void) {
    ASTNode *prog = parse_ok(
        "fn swap(a: int, b: int) -> (int, int) {\n"
        "  return (b, a)\n"
        "}\n"
        "fn main() -> int {\n"
        "  let p: (int, int) = (swap 1 2)\n"
        "  return p.0\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_struct_literal(void) {
    ASTNode *prog = parse_ok(
        "struct Vec2 { x: float, y: float }\n"
        "fn origin() -> Vec2 { return Vec2 { x: 0.0, y: 0.0 } }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_let_mut(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  let mut x: int = 0\n"
        "  let y: int = 5\n"
        "  set x y\n"
        "  return x\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_constants(void) {
    ASTNode *prog = parse_ok(
        "let PI: float = 3.14159\n"
        "let MAX_SIZE: int = 1024\n"
        "let GREETING: string = \"hello\"\n"
        "fn main() -> int { return MAX_SIZE }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_pub_functions(void) {
    ASTNode *prog = parse_ok(
        "pub fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "pub fn sub(a: int, b: int) -> int { return (- a b) }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_extern_fn(void) {
    ASTNode *prog = parse_ok(
        "extern fn printf(fmt: string) -> int\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_pure_fn(void) {
    ASTNode *prog = parse_ok(
        "pure fn square(x: int) -> int { return (* x x) }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASSERT_EQ(prog->as.program.count, 2);
    ASTNode *fn = prog->as.program.items[0];
    ASSERT_EQ(fn->type, AST_FUNCTION);
    ASSERT(fn->as.function.is_pure == true);
    ASSERT(fn->as.function.is_pub  == false);
    free_ast(prog);
}

void test_parse_pub_pure_fn(void) {
    ASTNode *prog = parse_ok(
        "pub pure fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASTNode *fn = prog->as.program.items[0];
    ASSERT_EQ(fn->type, AST_FUNCTION);
    ASSERT(fn->as.function.is_pure == true);
    ASSERT(fn->as.function.is_pub  == true);
    free_ast(prog);
}

void test_parse_pure_extern_fn(void) {
    ASTNode *prog = parse_ok(
        "pure extern fn fabs(x: float) -> float\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASTNode *fn = prog->as.program.items[0];
    ASSERT_EQ(fn->type, AST_FUNCTION);
    ASSERT(fn->as.function.is_pure   == true);
    ASSERT(fn->as.function.is_extern == true);
    free_ast(prog);
}

void test_parse_pub_pure_extern_fn(void) {
    ASTNode *prog = parse_ok(
        "pub pure extern fn sin(x: float) -> float\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASTNode *fn = prog->as.program.items[0];
    ASSERT_EQ(fn->type, AST_FUNCTION);
    ASSERT(fn->as.function.is_pure   == true);
    ASSERT(fn->as.function.is_pub    == true);
    ASSERT(fn->as.function.is_extern == true);
    free_ast(prog);
}

void test_parse_pure_fn_not_set_on_regular(void) {
    ASTNode *prog = parse_ok(
        "fn regular(x: int) -> int { return x }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASTNode *fn = prog->as.program.items[0];
    ASSERT_EQ(fn->type, AST_FUNCTION);
    ASSERT(fn->as.function.is_pure == false);
    free_ast(prog);
}

void test_parse_shadow(void) {
    ASTNode *prog = parse_ok(
        "fn double(x: int) -> int { return (* x 2) }\n"
        "shadow double {\n"
        "    assert (== (double 5) 10)\n"
        "    assert (== (double 0) 0)\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_break_continue(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  let mut i: int = 0\n"
        "  while (< i 100) {\n"
        "    if (== i 10) {\n"
        "      break\n"
        "    } else {\n"
        "      set i (+ i 1)\n"
        "      continue\n"
        "    }\n"
        "  }\n"
        "  return i\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_print_println(void) {
    ASTNode *prog = parse_ok(
        "fn main() -> int {\n"
        "  (print \"hello\")\n"
        "  (println \"world\")\n"
        "  (print 42)\n"
        "  (println 3.14)\n"
        "  return 0\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_assert(void) {
    ASTNode *prog = parse_ok(
        "fn main() -> int {\n"
        "  assert true\n"
        "  assert (== 1 1)\n"
        "  assert (not false)\n"
        "  return 0\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_float_literals(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> float {\n"
        "  let a: float = 3.14\n"
        "  let b: float = 2.718\n"
        "  let c: float = 1.0e10\n"
        "  return (+ a b)\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_string_escape(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> string {\n"
        "  let s1: string = \"hello\\nworld\"\n"
        "  let s2: string = \"tab\\there\"\n"
        "  let s3: string = \"quote\\\"here\"\n"
        "  return s1\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_generic_list(void) {
    ASTNode *prog = parse_ok(
        "fn sum(lst: List<int>) -> int {\n"
        "  let mut total: int = 0\n"
        "  for x in lst {\n"
        "    set total (+ total x)\n"
        "  }\n"
        "  return total\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_hashmap(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  let m: HashMap<string, int> = (map_new)\n"
        "  (map_put m \"key\" 42)\n"
        "  return (map_get m \"key\")\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_opaque_type(void) {
    ASTNode *prog = parse_ok(
        "opaque type Buffer\n"
        "extern fn buffer_new() -> Buffer\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_effect_decl(void) {
    ASTNode *prog = parse_ok(
        "effect IO {\n"
        "  print_line : string -> void,\n"
        "  read_line : string -> string\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_handle_expr(void) {
    ASTNode *prog = parse_ok(
        "effect Console {\n"
        "  print : string -> void\n"
        "}\n"
        "fn greet(name: string) -> void {\n"
        "  perform Console.print(name)\n"
        "}\n"
        "fn main() -> int {\n"
        "  (greet \"world\")\n"
        "  return 0\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_unary_operators(void) {
    ASTNode *prog = parse_ok(
        "fn f(x: int) -> int {\n"
        "  let neg: int = (- 0 x)\n"
        "  let abs_val: int = (abs x)\n"
        "  return neg\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_nested_function_calls(void) {
    ASTNode *prog = parse_ok(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn mul(a: int, b: int) -> int { return (* a b) }\n"
        "fn f(x: int) -> int {\n"
        "  return (add (mul x 2) (add x 1))\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_multiline_string(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> string {\n"
        "  let s: string = \"line1\\nline2\\nline3\"\n"
        "  return s\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_unsafe_block(void) {
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  unsafe {\n"
        "    return 42\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

/* ============================================================================
 * Import parsing tests
 * ============================================================================ */

void test_parse_selective_import_alias(void) {
    ASTNode *prog = parse_ok(
        "from \"modules/std/collections/stringbuilder.nano\" import sb_join as join_strings");
    ASSERT_NOT_NULL(prog);
    ASSERT_EQ(prog->type, AST_PROGRAM);
    ASSERT_EQ(prog->as.program.count, 1);

    ASTNode *import = prog->as.program.items[0];
    ASSERT_NOT_NULL(import);
    ASSERT_EQ(import->type, AST_IMPORT);
    ASSERT(strcmp(import->as.import_stmt.module_path,
                  "modules/std/collections/stringbuilder.nano") == 0);
    ASSERT_NULL(import->as.import_stmt.module_alias);
    ASSERT(import->as.import_stmt.is_unsafe == false);
    ASSERT(import->as.import_stmt.is_pub_use == false);
    ASSERT(import->as.import_stmt.is_selective == true);
    ASSERT(import->as.import_stmt.is_wildcard == false);
    ASSERT_EQ(import->as.import_stmt.import_symbol_count, 1);
    ASSERT_NOT_NULL(import->as.import_stmt.import_symbols);
    ASSERT(strcmp(import->as.import_stmt.import_symbols[0], "sb_join") == 0);
    ASSERT_NOT_NULL(import->as.import_stmt.import_aliases);
    ASSERT_NOT_NULL(import->as.import_stmt.import_aliases[0]);
    ASSERT(strcmp(import->as.import_stmt.import_aliases[0], "join_strings") == 0);

    free_ast(prog);
}

/* Regression: pub use parses as a public re-export import with alias metadata */
void test_parse_pub_use_reexport(void) {
    ASTNode *prog = parse_ok("pub use \"std/math/extended.nano\" as Math");
    ASSERT_NOT_NULL(prog);
    ASSERT_EQ(prog->type, AST_PROGRAM);
    ASSERT_EQ(prog->as.program.count, 1);

    ASTNode *import = prog->as.program.items[0];
    ASSERT_NOT_NULL(import);
    ASSERT_EQ(import->type, AST_IMPORT);

    ASSERT_NOT_NULL(import->as.import_stmt.module_path);
    ASSERT(strcmp(import->as.import_stmt.module_path,
                  "std/math/extended.nano") == 0);
    ASSERT_NOT_NULL(import->as.import_stmt.module_alias);
    ASSERT(strcmp(import->as.import_stmt.module_alias, "Math") == 0);

    ASSERT(import->as.import_stmt.is_pub_use == true);
    ASSERT(import->as.import_stmt.is_unsafe == false);
    ASSERT(import->as.import_stmt.is_selective == false);
    ASSERT(import->as.import_stmt.is_wildcard == false);
    ASSERT_EQ(import->as.import_stmt.import_symbol_count, 0);
    ASSERT_NULL(import->as.import_stmt.import_symbols);
    ASSERT_NULL(import->as.import_stmt.import_aliases);

    free_ast(prog);
}

/* Regression: normal module import sets path and clears selective/wildcard flags */
void test_parse_normal_module_import(void) {
    ASTNode *prog = parse_ok("module \"modules/std/math.nano\"");
    ASSERT_NOT_NULL(prog);
    ASSERT_EQ(prog->type, AST_PROGRAM);
    ASSERT_EQ(prog->as.program.count, 1);

    ASTNode *import = prog->as.program.items[0];
    ASSERT_NOT_NULL(import);
    ASSERT_EQ(import->type, AST_IMPORT);

    /* module_path must reflect the declared path */
    ASSERT_NOT_NULL(import->as.import_stmt.module_path);
    ASSERT(strcmp(import->as.import_stmt.module_path, "modules/std/math.nano") == 0);

    /* A plain module import carries no alias */
    ASSERT_NULL(import->as.import_stmt.module_alias);

    /* Must NOT be public, unsafe, selective, or wildcard */
    ASSERT(import->as.import_stmt.is_pub_use == false);
    ASSERT(import->as.import_stmt.is_unsafe == false);
    ASSERT(import->as.import_stmt.is_selective == false);
    ASSERT(import->as.import_stmt.is_wildcard  == false);

    /* No symbol list for a non-selective import */
    ASSERT_EQ(import->as.import_stmt.import_symbol_count, 0);
    ASSERT_NULL(import->as.import_stmt.import_symbols);
    ASSERT_NULL(import->as.import_stmt.import_aliases);

    free_ast(prog);
}

/* Regression: unsafe module import preserves path, alias, and unsafe flag */
void test_parse_unsafe_module_import(void) {
    ASTNode *prog = parse_ok("unsafe module \"modules/libc/libc.nano\" as C");
    ASSERT_NOT_NULL(prog);
    ASSERT_EQ(prog->type, AST_PROGRAM);
    ASSERT_EQ(prog->as.program.count, 1);

    ASTNode *import = prog->as.program.items[0];
    ASSERT_NOT_NULL(import);
    ASSERT_EQ(import->type, AST_IMPORT);

    ASSERT_NOT_NULL(import->as.import_stmt.module_path);
    ASSERT(strcmp(import->as.import_stmt.module_path,
                  "modules/libc/libc.nano") == 0);
    ASSERT_NOT_NULL(import->as.import_stmt.module_alias);
    ASSERT(strcmp(import->as.import_stmt.module_alias, "C") == 0);

    ASSERT(import->as.import_stmt.is_unsafe == true);
    ASSERT(import->as.import_stmt.is_pub_use == false);
    ASSERT(import->as.import_stmt.is_selective == false);
    ASSERT(import->as.import_stmt.is_wildcard == false);
    ASSERT_EQ(import->as.import_stmt.import_symbol_count, 0);
    ASSERT_NULL(import->as.import_stmt.import_symbols);
    ASSERT_NULL(import->as.import_stmt.import_aliases);

    free_ast(prog);
}

/* Regression: wildcard import sets is_wildcard without populating symbols or aliases */
void test_parse_wildcard_import(void) {
    ASTNode *prog = parse_ok("from \"modules/std/collections.nano\" import *");
    ASSERT_NOT_NULL(prog);
    ASSERT_EQ(prog->type, AST_PROGRAM);
    ASSERT_EQ(prog->as.program.count, 1);

    ASTNode *import = prog->as.program.items[0];
    ASSERT_NOT_NULL(import);
    ASSERT_EQ(import->type, AST_IMPORT);

    ASSERT_NOT_NULL(import->as.import_stmt.module_path);
    ASSERT(strcmp(import->as.import_stmt.module_path,
                  "modules/std/collections.nano") == 0);

    /* from...import * carries no module alias */
    ASSERT_NULL(import->as.import_stmt.module_alias);

    /* Wildcard implies selective (from…import form) */
    ASSERT(import->as.import_stmt.is_pub_use == false);
    ASSERT(import->as.import_stmt.is_unsafe == false);
    ASSERT(import->as.import_stmt.is_selective == true);
    ASSERT(import->as.import_stmt.is_wildcard  == true);

    /* No named symbols or aliases for a wildcard import */
    ASSERT_EQ(import->as.import_stmt.import_symbol_count, 0);
    ASSERT_NULL(import->as.import_stmt.import_symbols);
    ASSERT_NULL(import->as.import_stmt.import_aliases);

    free_ast(prog);
}

/* ============================================================================
 * parse_repl_input tests
 * ============================================================================ */

void test_parse_repl_simple(void) {
    ASTNode *prog = parse_repl("let x: int = 42");
    /* REPL mode might return NULL or an AST for partial input */
    if (prog) free_ast(prog);
}

void test_parse_repl_expression(void) {
    ASTNode *prog = parse_repl("(+ 1 2)");
    if (prog) free_ast(prog);
}

void test_parse_repl_function(void) {
    ASTNode *prog = parse_repl("fn double(x: int) -> int { return (* x 2) }");
    if (prog) free_ast(prog);
}

void test_parse_repl_empty(void) {
    ASTNode *prog = parse_repl("");
    if (prog) free_ast(prog);
}

/* ============================================================================
 * free_ast edge case tests — exercise all AST node free paths
 * ============================================================================ */

void test_free_ast_null(void) {
    free_ast(NULL);  /* should not crash */
}

void test_free_ast_complex(void) {
    ASTNode *prog = parse_ok(
        "struct Point { x: float, y: float }\n"
        "enum Dir { N, S, E, W }\n"
        "union Val { Int { n: int }, Str { s: string } }\n"
        "fn classify(v: Val) -> string {\n"
        "  match v {\n"
        "    Int(i) -> (int_to_string i.n),\n"
        "    Str(s) -> s.s\n"
        "  }\n"
        "}\n"
        "fn main() -> int {\n"
        "  let p: Point = Point { x: 1.0, y: 2.0 }\n"
        "  let arr: array<int> = [1, 2, 3]\n"
        "  let t: (int, string) = (42, \"hello\")\n"
        "  return 0\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);  /* Should not crash or leak */
}

/* ============================================================================
 * Error case tests — programs that fail to parse
 * ============================================================================ */

void test_parse_ascii_identifiers_only(void) {
    ASTNode *ok = parse_ok("fn cafe() -> int { return 0 }");
    ASSERT_NOT_NULL(ok);
    free_ast(ok);

    int count = 0;
    suppress_stderr();
    Token *tokens = tokenize("fn caf\xc3\xa9() -> int { return 0 }", &count);
    restore_stderr();
    ASSERT_NULL(tokens);
}

void test_parse_err_unclosed_paren(void) {
    /* Use EOF-terminated input so error recovery exits cleanly */
    suppress_stderr();
    ASTNode *prog = parse_ok("fn f() -> int { return (+ 1 2");
    restore_stderr();
    /* Should either fail or produce partial AST; just shouldn't crash */
    if (prog) free_ast(prog);
}

void test_parse_err_missing_return_type(void) {
    suppress_stderr();
    ASTNode *prog = parse_ok("fn f() { return 42 }");
    restore_stderr();
    if (prog) free_ast(prog);
}

void test_parse_err_invalid_token(void) {
    suppress_stderr();
    ASTNode *prog = parse_ok("fn main() -> int { return @ }");
    restore_stderr();
    if (prog) free_ast(prog);
}

void test_parse_err_selective_import_dangling_as(void) {
    suppress_stderr();
    ASTNode *prog = parse_ok(
        "from \"modules/std/collections/stringbuilder.nano\" import sb_join as");
    restore_stderr();
    ASSERT_NULL(prog);
}

void test_parse_many_functions(void) {
    /* Parse a program with many functions to exercise the program array growth */
    ASTNode *prog = parse_ok(
        "fn f1(x: int) -> int { return (* x 1) }\n"
        "fn f2(x: int) -> int { return (* x 2) }\n"
        "fn f3(x: int) -> int { return (* x 3) }\n"
        "fn f4(x: int) -> int { return (* x 4) }\n"
        "fn f5(x: int) -> int { return (* x 5) }\n"
        "fn f6(x: int) -> int { return (* x 6) }\n"
        "fn f7(x: int) -> int { return (* x 7) }\n"
        "fn f8(x: int) -> int { return (* x 8) }\n"
        "fn f9(x: int) -> int { return (* x 9) }\n"
        "fn f10(x: int) -> int { return (* x 10) }\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_deep_nesting(void) {
    ASTNode *prog = parse_ok(
        "fn f(x: int) -> int {\n"
        "  if (> x 0) {\n"
        "    if (> x 10) {\n"
        "      if (> x 100) {\n"
        "        return 3\n"
        "      } else {\n"
        "        return 2\n"
        "      }\n"
        "    } else {\n"
        "      return 1\n"
        "    }\n"
        "  } else {\n"
        "    return 0\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

void test_parse_infix_operators(void) {
    /* Test infix +, -, *, /, % syntax (alternative to prefix) */
    ASTNode *prog = parse_ok(
        "fn f(x: int, y: int) -> int {\n"
        "  return (x + y)\n"
        "}");
    if (prog) {
        free_ast(prog);
    }
}

void test_parse_type_annotations(void) {
    /* Test various type annotation forms */
    ASTNode *prog = parse_ok(
        "fn f() -> int {\n"
        "  let a: int = 1\n"
        "  let b: float = 1.0\n"
        "  let c: string = \"x\"\n"
        "  let d: bool = true\n"
        "  let e: array<int> = [1, 2]\n"
        "  let f: List<string> = (list_string_new)\n"
        "  return 0\n"
        "}");
    ASSERT_NOT_NULL(prog);
    free_ast(prog);
}

/* ============================================================================
 * main
 * ============================================================================ */

static void test_owned_generic_function_annotations(void) {
    ASTNode *program = parse_ok(
        "union Box<T> { Some { value: T } } "
        "union Result<T,E> { Ok { value: T }, Err { error: E } } "
        "fn apply(callback: fn(Box<array<int>>, HashMap<string,int>) -> Result<Box<int>,string>) -> int { return 0 }");
    ASSERT_NOT_NULL(program);
    ASTNode *function = program->as.program.items[2];
    FunctionSignature *sig = function->as.function.params[0].fn_sig;
    ASSERT_NOT_NULL(sig);
    ASSERT_EQ(sig->param_count, 2);
    ASSERT_NOT_NULL(sig->param_type_info);
    ASSERT(strcmp(sig->param_type_info[0]->generic_name, "Box") == 0);
    ASSERT_EQ(sig->param_type_info[0]->type_params[0]->base_type, TYPE_ARRAY);
    ASSERT_EQ(sig->param_type_info[0]->type_params[0]->element_type->base_type, TYPE_INT);
    ASSERT(strcmp(sig->param_type_info[1]->generic_name, "HashMap") == 0);
    ASSERT_EQ(sig->param_type_info[1]->type_params[0]->base_type, TYPE_STRING);
    ASSERT_NOT_NULL(sig->return_type_info);
    ASSERT(strcmp(sig->return_type_info->generic_name, "Result") == 0);
    ASSERT(strcmp(sig->return_type_info->type_params[0]->generic_name, "Box") == 0);
    TypeInfo wrapper = {0};
    wrapper.base_type = TYPE_FUNCTION;
    wrapper.fn_sig = sig;
    TypeInfo *copy = copy_payload_type_info(&wrapper);
    ASSERT_NOT_NULL(copy);
    ASSERT(copy->fn_sig != sig);
    ASSERT(copy->fn_sig->param_type_info[0] != sig->param_type_info[0]);
    ASSERT(copy->fn_sig->return_type_info != sig->return_type_info);
    FunctionSignature *direct = copy_function_signature(sig);
    ASSERT(function_signatures_equal(sig, direct));
    ASSERT(type_infos_equal(sig->return_type_info, direct->return_type_info));
    direct->param_type_info[0]->type_params[0]->element_type->base_type = TYPE_BOOL;
    ASSERT(!function_signatures_equal(sig, direct));
    direct->param_type_info[0]->type_params[0]->element_type->base_type = TYPE_INT;
    direct->return_type_info->type_params[0]->type_params[0]->base_type = TYPE_STRING;
    ASSERT(!function_signatures_equal(sig, direct));
    free_function_signature(direct);
    Parameter parameters[2] = {0};
    for (int i = 0; i < 2; ++i) {
        parameters[i].type = sig->param_types[i];
        parameters[i].struct_type_name = sig->param_struct_names[i];
        parameters[i].type_info = sig->param_type_info[i];
    }
    Function declaration = {.param_count = 2, .params = parameters,
        .return_type = sig->return_type, .return_struct_type_name = sig->return_struct_name,
        .return_type_info = sig->return_type_info};
    direct = function_signature_from_function(&declaration);
    ASSERT(function_signatures_equal(sig, direct));
    free_ast(program);
    ASSERT(function_signatures_equal(copy->fn_sig, direct));
    free_function_signature(direct);
    ASSERT_EQ(copy->fn_sig->return_type_info->type_params[0]->type_params[0]->base_type, TYPE_INT);
    ASSERT_EQ(copy->fn_sig->param_type_info[1]->type_params[1]->base_type, TYPE_INT);
    free_payload_type_info(copy);

    program = parse_ok("fn choose() -> fn() -> fn() -> Box<int> { return choose }");
    ASSERT_NOT_NULL(program);
    sig = program->as.program.items[0]->as.function.return_fn_sig;
    ASSERT_NOT_NULL(sig->return_fn_sig);
    ASSERT(strcmp(sig->return_fn_sig->return_type_info->generic_name, "Box") == 0);
    wrapper.fn_sig = sig;
    copy = copy_payload_type_info(&wrapper);
    free_ast(program);
    ASSERT_EQ(copy->fn_sig->return_fn_sig->return_type_info->type_params[0]->base_type, TYPE_INT);
    free_payload_type_info(copy);
}

static void test_borrow_annotation_retention(void) {
    ASTNode *program = parse_ok("extern fn observe(a: &owned.Handle, b: &mut Box<array<int>>, callback: &fn(array<int>)->string) -> int");
    ASSERT_NOT_NULL(program);
    ASTNode *function = program->as.program.items[0];
    ASSERT_EQ(function->as.function.param_count, 3);
    TypeInfo *shared = function->as.function.params[0].type_info;
    TypeInfo *exclusive = function->as.function.params[1].type_info;
    ASSERT_NOT_NULL(shared);
    ASSERT_EQ(shared->base_type, TYPE_BORROW_SHARED);
    ASSERT_EQ(shared->element_type->base_type, TYPE_STRUCT);
    ASSERT(strcmp(shared->element_type->generic_name, "owned.Handle") == 0);
    ASSERT_EQ(exclusive->base_type, TYPE_BORROW_MUT);
    ASSERT(strcmp(exclusive->element_type->generic_name, "Box") == 0);
    ASSERT_EQ(exclusive->element_type->type_params[0]->base_type, TYPE_ARRAY);
    ASSERT_EQ(exclusive->element_type->type_params[0]->element_type->base_type, TYPE_INT);
    TypeInfo *callback = function->as.function.params[2].type_info;
    ASSERT_EQ(callback->element_type->base_type, TYPE_FUNCTION);
    ASSERT_NOT_NULL(callback->element_type->fn_sig);
    ASSERT_EQ(callback->element_type->fn_sig->param_type_info[0]->base_type, TYPE_ARRAY);
    ASSERT_EQ(callback->element_type->fn_sig->return_type, TYPE_STRING);
    TypeInfo *copy = copy_payload_type_info(exclusive);
    ASSERT(type_infos_equal(copy, exclusive));
    copy->base_type = TYPE_BORROW_SHARED;
    ASSERT(!type_infos_equal(copy, exclusive));
    copy->base_type = TYPE_BORROW_MUT;
    free_ast(program);
    ASSERT_EQ(copy->element_type->type_params[0]->element_type->base_type, TYPE_INT);
    free_payload_type_info(copy);
    ASSERT_NULL(parse_ok("extern fn invalid(a: &&int) -> int"));
    ASSERT_NULL(parse_ok("extern fn invalid() -> &int"));
    ASSERT_NULL(parse_ok("struct Invalid { value: &int }"));
}

static void test_borrow_postcondition_copy(void) {
    ASTNode *program = parse_ok("fn inspect(view: &Handle) -> int ensures (valid &view) { return 0 }");
    ASSERT_NOT_NULL(program);
    ASTNode *body = program->as.program.items[0]->as.function.body;
    ASTNode *returned = body->as.block.statements[0];
    ASSERT_EQ(returned->type, AST_BLOCK);
    ASTNode *condition = returned->as.block.statements[1]->as.assert.condition;
    ASSERT_EQ(condition->type, AST_CALL);
    ASTNode *borrow = condition->as.call.args[0];
    ASSERT_EQ(borrow->type, AST_CALL);
    ASSERT_EQ(borrow->as.call.borrow_mode, 1);
    ASSERT_EQ(borrow->as.call.arg_count, 1);
    ASSERT(strcmp(borrow->as.call.args[0]->as.identifier, "view") == 0);
    free_ast(program);
}

static void test_exclusive_field_place_retention(void) {
    ASTNode *program = parse_ok("fn update(view: &mut Handle) -> int { set view.fd 8 return view.fd }");
    ASSERT_NOT_NULL(program);
    ASTNode *function = program->as.program.items[0];
    ASSERT_EQ(function->as.function.params[0].type, TYPE_BORROW_MUT);
    ASTNode *assignment = function->as.function.body->as.block.statements[0];
    ASSERT_EQ(assignment->type, AST_SET);
    ASSERT(strcmp(assignment->as.set.name, "view") == 0);
    ASSERT(strcmp(assignment->as.set.field_name, "fd") == 0);
    ASSERT_EQ(assignment->as.set.value->as.number, 8);
    free_ast(program);
}

int main(void) {
    printf("=== Parser Tests ===\n");

    printf("\n--- Valid programs ---\n");
    TEST(parse_minimal);
    TEST(borrow_annotation_retention);
    TEST(borrow_postcondition_copy);
    TEST(exclusive_field_place_retention);
    TEST(nested_generic_annotation_metadata);
    TEST(nested_generic_annotation_errors);
    TEST(parse_empty_program);
    TEST(parse_arithmetic);
    TEST(parse_all_comparisons);
    TEST(parse_bool_ops);
    TEST(parse_if_else_chain);
    TEST(parse_while_loop);
    TEST(parse_for_range);
    TEST(parse_struct_def);
    TEST(parse_enum_def);
    TEST(parse_union_def);
    TEST(parse_match_expr);
    TEST(parse_match_with_guards);
    TEST(parse_union_match);
    TEST(parse_cond_expr);
    TEST(parse_array_literal);
    TEST(parse_tuple_literal);
    TEST(parse_struct_literal);
    TEST(parse_let_mut);
    TEST(parse_constants);
    TEST(parse_pub_functions);
    TEST(parse_extern_fn);
    TEST(parse_pure_fn);
    TEST(parse_pub_pure_fn);
    TEST(parse_pure_extern_fn);
    TEST(parse_pub_pure_extern_fn);
    TEST(parse_pure_fn_not_set_on_regular);
    TEST(parse_shadow);
    TEST(parse_break_continue);
    TEST(parse_print_println);
    TEST(parse_assert);
    TEST(parse_float_literals);
    TEST(parse_string_escape);
    TEST(parse_generic_list);
    TEST(parse_hashmap);
    TEST(parse_opaque_type);
    TEST(parse_effect_decl);
    TEST(parse_handle_expr);
    TEST(parse_unary_operators);
    TEST(parse_nested_function_calls);
    TEST(parse_multiline_string);
    TEST(parse_unsafe_block);

    printf("\n--- Import parsing ---\n");
    TEST(parse_selective_import_alias);
    TEST(parse_pub_use_reexport);
    TEST(parse_normal_module_import);
    TEST(parse_unsafe_module_import);
    TEST(parse_wildcard_import);

    printf("\n--- parse_repl_input tests ---\n");
    TEST(parse_repl_simple);
    TEST(parse_repl_expression);
    TEST(parse_repl_function);
    TEST(parse_repl_empty);

    printf("\n--- free_ast edge cases ---\n");
    TEST(free_ast_null);
    TEST(free_ast_complex);

    printf("\n--- Error cases ---\n");
    TEST(parse_ascii_identifiers_only);
    TEST(parse_err_unclosed_paren);
    TEST(parse_err_missing_return_type);
    TEST(parse_err_invalid_token);
    TEST(parse_err_selective_import_dangling_as);
    TEST(parse_many_functions);
    TEST(parse_deep_nesting);
    TEST(parse_infix_operators);
    TEST(parse_type_annotations);
    TEST(owned_generic_function_annotations);

    printf("\n✓ All parser tests passed!\n");
    return 0;
}
