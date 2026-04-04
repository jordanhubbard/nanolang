/**
 * test_transpiler.c - Unit tests for transpiler components
 * 
 * Tests StringBuilder, registries, and error handling.
 */

#include "../src/nanolang.h"
#include "../src/stdlib_runtime.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %d, expected %d)\n", #a, #b, __LINE__, (int)(a), (int)(b)); exit(1); }
#define ASSERT_STR_EQ(a, b) if (strcmp((a), (b)) != 0) { printf("\n    FAILED: %s == %s at line %d\n    got: \"%s\"\n    expected: \"%s\"\n", #a, #b, __LINE__, (a), (b)); exit(1); }

/* Required by runtime */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* ============================================================================
 * StringBuilder Tests
 * ============================================================================ */

void test_stringbuilder_create() {
    StringBuilder *sb = sb_create();
    ASSERT(sb != NULL);
    ASSERT(sb->buffer != NULL);
    ASSERT(sb->length == 0);
    ASSERT(sb->capacity >= 1024);
    ASSERT(sb->buffer[0] == '\0');
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_append_simple() {
    StringBuilder *sb = sb_create();
    
    sb_append(sb, "Hello");
    ASSERT_EQ(sb->length, 5);
    ASSERT_STR_EQ(sb->buffer, "Hello");
    
    sb_append(sb, " World");
    ASSERT_EQ(sb->length, 11);
    ASSERT_STR_EQ(sb->buffer, "Hello World");
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_append_empty() {
    StringBuilder *sb = sb_create();
    
    sb_append(sb, "");
    ASSERT_EQ(sb->length, 0);
    ASSERT_STR_EQ(sb->buffer, "");
    
    sb_append(sb, "Test");
    ASSERT_EQ(sb->length, 4);
    ASSERT_STR_EQ(sb->buffer, "Test");
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_append_multiple() {
    StringBuilder *sb = sb_create();
    
    for (int i = 0; i < 10; i++) {
        sb_append(sb, "X");
    }
    
    ASSERT_EQ(sb->length, 10);
    ASSERT_STR_EQ(sb->buffer, "XXXXXXXXXX");
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_large_append() {
    StringBuilder *sb = sb_create();
    
    // Create a string larger than initial capacity (1024)
    char large_str[2000];
    for (int i = 0; i < 1999; i++) {
        large_str[i] = 'A';
    }
    large_str[1999] = '\0';
    
    sb_append(sb, large_str);
    ASSERT_EQ(sb->length, 1999);
    ASSERT(sb->capacity >= 1999);
    ASSERT_STR_EQ(sb->buffer, large_str);
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_capacity_growth() {
    StringBuilder *sb = sb_create();
    int initial_capacity = sb->capacity;
    
    // Append enough to trigger reallocation
    char buffer[2048];
    for (int i = 0; i < 2047; i++) {
        buffer[i] = 'B';
    }
    buffer[2047] = '\0';
    
    sb_append(sb, buffer);
    
    // Capacity should have grown
    ASSERT(sb->capacity > initial_capacity);
    ASSERT(sb->capacity >= 2047);
    ASSERT_EQ(sb->length, 2047);
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_null_termination() {
    StringBuilder *sb = sb_create();
    
    sb_append(sb, "Test");
    ASSERT(sb->buffer[sb->length] == '\0');
    
    sb_append(sb, "123");
    ASSERT(sb->buffer[sb->length] == '\0');
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Stdlib Runtime Generation Tests
 * ============================================================================ */

void test_generate_string_operations() {
    StringBuilder *sb = sb_create();
    
    generate_string_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "char_at") != NULL);
    ASSERT(strstr(sb->buffer, "string_from_char") != NULL);
    ASSERT(strstr(sb->buffer, "is_digit") != NULL);
    ASSERT(strstr(sb->buffer, "is_alpha") != NULL);
    ASSERT(strstr(sb->buffer, "int_to_string") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_file_operations() {
    StringBuilder *sb = sb_create();
    
    generate_file_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_os_file_read") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_write") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_exists") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_read_bytes") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_dir_operations() {
    StringBuilder *sb = sb_create();
    
    generate_dir_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_os_dir_create") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_dir_remove") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_dir_list") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_getcwd") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_path_operations() {
    StringBuilder *sb = sb_create();
    
    generate_path_operations(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_os_path_isfile") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_isdir") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_join") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_basename") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_dirname") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_generate_math_utility_builtins() {
    StringBuilder *sb = sb_create();
    
    generate_math_utility_builtins(sb);
    
    ASSERT(sb->length > 0);
    ASSERT(strstr(sb->buffer, "nl_abs") != NULL);
    ASSERT(strstr(sb->buffer, "nl_min") != NULL);
    ASSERT(strstr(sb->buffer, "nl_max") != NULL);
    ASSERT(strstr(sb->buffer, "nl_cast_int") != NULL);
    ASSERT(strstr(sb->buffer, "nl_println") != NULL);
    ASSERT(strstr(sb->buffer, "nl_array_length") != NULL);
    
    free(sb->buffer);
    free(sb);
}

void test_stdlib_runtime_complete() {
    StringBuilder *sb = sb_create();
    
    generate_stdlib_runtime(sb);
    
    // Should include all components
    ASSERT(sb->length > 5000);  // Substantial amount of code
    ASSERT(strstr(sb->buffer, "OS Standard Library") != NULL);
    ASSERT(strstr(sb->buffer, "Advanced String Operations") != NULL);
    ASSERT(strstr(sb->buffer, "Math and Utility Built-in Functions") != NULL);
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Memory Safety Tests
 * ============================================================================ */

void test_stringbuilder_no_buffer_overflow() {
    StringBuilder *sb = sb_create();
    
    // Append many small strings to test growth
    for (int i = 0; i < 1000; i++) {
        sb_append(sb, "test");
    }
    
    ASSERT_EQ(sb->length, 4000);
    ASSERT(sb->capacity >= 4000);
    
    // Verify no corruption
    for (int i = 0; i < 4000; i += 4) {
        ASSERT(sb->buffer[i] == 't');
        ASSERT(sb->buffer[i+1] == 'e');
        ASSERT(sb->buffer[i+2] == 's');
        ASSERT(sb->buffer[i+3] == 't');
    }
    
    free(sb->buffer);
    free(sb);
}

void test_stringbuilder_empty_appends() {
    StringBuilder *sb = sb_create();
    
    // Many empty appends shouldn't break anything
    for (int i = 0; i < 100; i++) {
        sb_append(sb, "");
    }
    
    ASSERT_EQ(sb->length, 0);
    ASSERT_STR_EQ(sb->buffer, "");
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Integration Tests
 * ============================================================================ */

void test_generate_all_stdlib_functions() {
    StringBuilder *sb = sb_create();
    
    // Generate all stdlib components
    generate_math_utility_builtins(sb);
    generate_string_operations(sb);
    generate_file_operations(sb);
    generate_dir_operations(sb);
    generate_path_operations(sb);
    
    // Should have substantial content
    ASSERT(sb->length > 10000);
    
    // Verify key functions are present
    ASSERT(strstr(sb->buffer, "nl_abs") != NULL);
    ASSERT(strstr(sb->buffer, "char_at") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_file_read") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_dir_create") != NULL);
    ASSERT(strstr(sb->buffer, "nl_os_path_join") != NULL);
    
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Additional stdlib_runtime.c coverage
 * ============================================================================ */

void test_generate_timing_utilities() {
    StringBuilder *sb = sb_create();
    generate_timing_utilities(sb);
    ASSERT(sb->length > 0);
    free(sb->buffer);
    free(sb);
}

void test_generate_console_io_utilities() {
    StringBuilder *sb = sb_create();
    generate_console_io_utilities(sb);
    ASSERT(sb->length > 0);
    free(sb->buffer);
    free(sb);
}

void test_generate_profiling_system() {
    StringBuilder *sb = sb_create();
    generate_profiling_system(sb, "/tmp/test_transpiler.prof");
    free(sb->buffer);
    free(sb);
}

void test_generate_instrumented_profiling_system() {
    StringBuilder *sb = sb_create();
    generate_instrumented_profiling_system(sb);
    free(sb->buffer);
    free(sb);
}

void test_generate_flamegraph_profiling_system() {
    StringBuilder *sb = sb_create();
    generate_flamegraph_profiling_system(sb, "/tmp/test_transpiler.flamegraph");
    free(sb->buffer);
    free(sb);
}

void test_generate_coroutine_builtins() {
    StringBuilder *sb = sb_create();
    generate_coroutine_builtins(sb);
    ASSERT(sb->length > 0);
    free(sb->buffer);
    free(sb);
}

void test_generate_module_system_stubs() {
    StringBuilder *sb = sb_create();
    generate_module_system_stubs(sb);
    free(sb->buffer);
    free(sb);
}

/* ============================================================================
 * Suppress stderr helper
 * ============================================================================ */
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
 * transpile_to_c() integration tests
 * ============================================================================ */

/* Helper: parse + typecheck + transpile a nano program.
 * Returns generated C (caller must free), or NULL on failure. */
static char *transpile_src(const char *src) {
    int count = 0;
    Token *tokens = tokenize(src, &count);
    if (!tokens) return NULL;
    ASTNode *program = parse_program(tokens, count);
    free_tokens(tokens, count);
    if (!program) return NULL;
    clear_module_cache();
    Environment *env = create_environment();
    typecheck_set_current_file("<test>");
    suppress_stderr();
    bool ok = type_check(program, env);
    restore_stderr();
    if (!ok) { free_ast(program); free_environment(env); return NULL; }
    char *result = transpile_to_c(program, env, "<test>");
    free_ast(program);
    free_environment(env);
    return result;
}

void test_transpile_minimal() {
    char *c = transpile_src("fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "main") != NULL);
    free(c);
}

void test_transpile_arithmetic() {
    char *c = transpile_src(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn mul(a: int, b: int) -> int { return (* a b) }\n"
        "fn main() -> int {\n"
        "  let x: int = (add 3 (mul 2 4))\n"
        "  return x\n"
        "}");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "add") != NULL);
    free(c);
}

void test_transpile_struct() {
    char *c = transpile_src(
        "struct Point { x: int, y: int }\n"
        "fn make_point(a: int, b: int) -> Point {\n"
        "  let p: Point = Point { x: a, y: b }\n"
        "  return p\n"
        "}\n"
        "fn main() -> int {\n"
        "  let p: Point = (make_point 1 2)\n"
        "  return p.x\n"
        "}");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "Point") != NULL);
    free(c);
}

void test_transpile_enum() {
    char *c = transpile_src(
        "enum Color { Red, Green, Blue }\n"
        "fn to_int(c: Color) -> int {\n"
        "  if (== c Color.Red) { return 0 }\n"
        "  else { if (== c Color.Green) { return 1 } else { return 2 } }\n"
        "}\n"
        "fn main() -> int { return (to_int Color.Blue) }");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "Color") != NULL);
    free(c);
}

void test_transpile_union_match() {
    char *c = transpile_src(
        "union Shape { Circle { r: float }, Square { s: float } }\n"
        "fn area(sh: Shape) -> float {\n"
        "  match sh {\n"
        "    Circle(c) -> (* 3.14 (* c.r c.r)),\n"
        "    Square(sq) -> (* sq.s sq.s)\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "Shape") != NULL);
    free(c);
}

void test_transpile_string_ops() {
    char *c = transpile_src(
        "fn greet(name: string) -> string {\n"
        "  return (str_concat \"Hello, \" name)\n"
        "}\n"
        "fn main() -> int {\n"
        "  let s: string = (greet \"world\")\n"
        "  let n: int = (str_length s)\n"
        "  return n\n"
        "}");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_array_ops() {
    char *c = transpile_src(
        "fn sum_arr(arr: array<int>, n: int) -> int {\n"
        "  let mut total: int = 0\n"
        "  let mut i: int = 0\n"
        "  while (< i n) {\n"
        "    set total (+ total (array_get arr i))\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return total\n"
        "}\n"
        "fn main() -> int {\n"
        "  let arr: array<int> = [1, 2, 3]\n"
        "  return (sum_arr arr 3)\n"
        "}");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_for_range() {
    char *c = transpile_src(
        "fn count_sum(n: int) -> int {\n"
        "  let mut result: int = 0\n"
        "  for i in (range 0 n) {\n"
        "    set result (+ result i)\n"
        "  }\n"
        "  return result\n"
        "}\n"
        "fn main() -> int { return (count_sum 5) }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_if_else_chain() {
    char *c = transpile_src(
        "fn classify(n: int) -> string {\n"
        "  if (< n 0) { return \"negative\" }\n"
        "  else { if (== n 0) { return \"zero\" }\n"
        "  else { return \"positive\" } }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_constants() {
    char *c = transpile_src(
        "let MAX_SIZE: int = 100\n"
        "let PI: float = 3.14159\n"
        "fn get_max() -> int { return MAX_SIZE }\n"
        "fn main() -> int { return (get_max) }");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "MAX_SIZE") != NULL);
    free(c);
}

void test_transpile_recursive() {
    char *c = transpile_src(
        "fn fib(n: int) -> int {\n"
        "  if (< n 2) { return n }\n"
        "  else { return (+ (fib (- n 1)) (fib (- n 2))) }\n"
        "}\n"
        "fn main() -> int { return (fib 10) }");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "fib") != NULL);
    free(c);
}

void test_transpile_float_math() {
    char *c = transpile_src(
        "fn hyp(a: float, b: float) -> float {\n"
        "  return (sqrt (+ (* a a) (* b b)))\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_tuple() {
    char *c = transpile_src(
        "fn min_max(a: int, b: int) -> (int, int) {\n"
        "  if (< a b) { return (a, b) }\n"
        "  else { return (b, a) }\n"
        "}\n"
        "fn main() -> int {\n"
        "  let pair: (int, int) = (min_max 5 3)\n"
        "  return pair.0\n"
        "}");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_nested_structs() {
    char *c = transpile_src(
        "struct Vec2 { x: float, y: float }\n"
        "struct Rect { pos: Vec2, size: Vec2 }\n"
        "fn make_rect(x: float, y: float, w: float, h: float) -> Rect {\n"
        "  let pos: Vec2 = Vec2 { x: x, y: y }\n"
        "  let size: Vec2 = Vec2 { x: w, y: h }\n"
        "  return Rect { pos: pos, size: size }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    ASSERT(strstr(c, "Vec2") != NULL);
    ASSERT(strstr(c, "Rect") != NULL);
    free(c);
}

void test_transpile_shadow_tests() {
    char *c = transpile_src(
        "fn dbl(x: int) -> int { return (* x 2) }\n"
        "shadow dbl {\n"
        "  let result: int = (dbl 5)\n"
        "  assert (== result 10)\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_match_int() {
    char *c = transpile_src(
        "fn describe(x: int) -> string {\n"
        "  match x {\n"
        "    0 -> \"zero\",\n"
        "    1 -> \"one\",\n"
        "    _ -> \"other\"\n"
        "  }\n"
        "}\n"
        "fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_cond_expr() {
    char *c = transpile_src(
        "fn sign(n: int) -> int {\n"
        "  return (cond ((< n 0) (- 0 1)) ((== n 0) 0) (else 1))\n"
        "}\n"
        "fn main() -> int { return (sign 5) }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_pub_function() {
    char *c = transpile_src(
        "pub fn public_add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn main() -> int { return (public_add 1 2) }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_print_println() {
    char *c = transpile_src(
        "fn greet() -> int {\n"
        "  (print \"Hello\")\n"
        "  (println \" World\")\n"
        "  return 0\n"
        "}\n"
        "fn main() -> int { return (greet) }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_opaque_type() {
    /* Opaque type declaration at top-level, used in function signature */
    char *c = transpile_src(
        "opaque type Token\n"
        "fn main() -> int { return 0 }");
    ASSERT(c != NULL);
    free(c);
}

void test_transpile_null_inputs() {
    /* transpile_to_c with NULL program should return NULL */
    Environment *env = create_environment();
    suppress_stderr();
    char *r1 = transpile_to_c(NULL, env, "<test>");
    char *r2 = transpile_to_c(NULL, NULL, "<test>");
    restore_stderr();
    ASSERT(r1 == NULL);
    ASSERT(r2 == NULL);
    free_environment(env);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("Running transpiler component tests:\n");

    printf("\nStringBuilder Tests:\n");
    TEST(stringbuilder_create);
    TEST(stringbuilder_append_simple);
    TEST(stringbuilder_append_empty);
    TEST(stringbuilder_append_multiple);
    TEST(stringbuilder_large_append);
    TEST(stringbuilder_capacity_growth);
    TEST(stringbuilder_null_termination);

    printf("\nStdlib Runtime Generation Tests:\n");
    TEST(generate_string_operations);
    TEST(generate_file_operations);
    TEST(generate_dir_operations);
    TEST(generate_path_operations);
    TEST(generate_math_utility_builtins);
    TEST(stdlib_runtime_complete);
    TEST(generate_timing_utilities);
    TEST(generate_console_io_utilities);
    TEST(generate_profiling_system);
    TEST(generate_instrumented_profiling_system);
    TEST(generate_flamegraph_profiling_system);
    TEST(generate_coroutine_builtins);
    TEST(generate_module_system_stubs);

    printf("\nMemory Safety Tests:\n");
    TEST(stringbuilder_no_buffer_overflow);
    TEST(stringbuilder_empty_appends);

    printf("\nIntegration Tests:\n");
    TEST(generate_all_stdlib_functions);

    printf("\ntranspile_to_c() Tests:\n");
    TEST(transpile_minimal);
    TEST(transpile_arithmetic);
    TEST(transpile_struct);
    TEST(transpile_enum);
    TEST(transpile_union_match);
    TEST(transpile_string_ops);
    TEST(transpile_array_ops);
    TEST(transpile_for_range);
    TEST(transpile_if_else_chain);
    TEST(transpile_constants);
    TEST(transpile_recursive);
    TEST(transpile_float_math);
    TEST(transpile_tuple);
    TEST(transpile_nested_structs);
    TEST(transpile_shadow_tests);
    TEST(transpile_match_int);
    TEST(transpile_cond_expr);
    TEST(transpile_pub_function);
    TEST(transpile_print_println);
    TEST(transpile_opaque_type);
    TEST(transpile_null_inputs);

    printf("\n✓ All transpiler tests passed!\n");
    return 0;
}

