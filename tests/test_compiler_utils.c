/*
 * test_compiler_utils.c — unit tests for compiler utility modules
 *
 * Exercises: dwarf_info.c, nanocore_subset.c, nanocore_export.c,
 *            emit_typed_ast.c, reflection.c, bench.c
 */

#include "../src/nanolang.h"
#include "../src/dwarf_info.h"
#include "../src/nanocore_subset.h"
#include "../src/nanocore_export.h"
#include "../src/emit_typed_ast.h"
#include "../src/reflection.h"
#include "../src/bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static ASTNode *parse_nano(const char *src) {
    int n = 0;
    Token *t = tokenize(src, &n);
    if (!t) return NULL;
    ASTNode *p = parse_program(t, n);
    free_tokens(t, n);
    return p;
}

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

/* Redirect stdout to /dev/null for tests that emit to stdout */
static int s_saved_stdout_fd = -1;
static void suppress_stdout(void) {
    fflush(stdout);
    s_saved_stdout_fd = dup(STDOUT_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    close(devnull);
}
static void restore_stdout(void) {
    if (s_saved_stdout_fd >= 0) {
        fflush(stdout);
        dup2(s_saved_stdout_fd, STDOUT_FILENO);
        close(s_saved_stdout_fd);
        s_saved_stdout_fd = -1;
    }
}

/* ── DWARF info tests ────────────────────────────────────────────────────── */

static void test_dwarf_begin_free(void) {
    const char *test_name = "dwarf_begin/free";
    DwarfBuilder *db = dwarf_begin("test.nano");
    ASSERT(db != NULL, "dwarf_begin returned NULL");
    dwarf_free(db);
    PASS(test_name);
}

static void test_dwarf_add_function(void) {
    const char *test_name = "dwarf_function";
    DwarfBuilder *db = dwarf_begin("test.nano");
    ASSERT(db != NULL, "dwarf_begin returned NULL");
    dwarf_function(db, "add", 0, 1, 0, TYPE_INT);
    dwarf_function(db, "main", 0, 10, 0, TYPE_VOID);
    dwarf_free(db);
    PASS(test_name);
}

static void test_dwarf_add_variable(void) {
    const char *test_name = "dwarf_variable";
    DwarfBuilder *db = dwarf_begin("test.nano");
    ASSERT(db != NULL, "dwarf_begin returned NULL");
    dwarf_function(db, "add", 0, 1, 0, TYPE_INT);
    dwarf_variable(db, "x", TYPE_INT, 4, 2);
    dwarf_variable(db, "result", TYPE_INT, 8, 3);
    dwarf_free(db);
    PASS(test_name);
}

static void test_dwarf_add_lines(void) {
    const char *test_name = "dwarf_line";
    DwarfBuilder *db = dwarf_begin("test.nano");
    ASSERT(db != NULL, "dwarf_begin returned NULL");
    dwarf_line(db, 0, 1, 0, 0);
    dwarf_line(db, 0, 2, 4, 0);
    dwarf_line(db, 0, 3, 4, 0);
    dwarf_free(db);
    PASS(test_name);
}

static void test_dwarf_emit_sections(void) {
    const char *test_name = "dwarf_emit_asm_sections";
    DwarfBuilder *db = dwarf_begin("example.nano");
    ASSERT(db != NULL, "dwarf_begin returned NULL");
    dwarf_function(db, "square", 0, 1, 0, TYPE_INT);
    dwarf_variable(db, "n", TYPE_INT, 4, 2);
    dwarf_line(db, 0, 1, 0, 0);
    dwarf_line(db, 0, 2, 2, 0);

    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen failed");
    dwarf_emit_asm_sections(db, out);
    fclose(out);
    dwarf_free(db);
    PASS(test_name);
}

static void test_dwarf_emit_to_file(void) {
    const char *test_name = "dwarf_emit_asm_sections to file";
    DwarfBuilder *db = dwarf_begin("prog.nano");
    ASSERT(db != NULL, "dwarf_begin returned NULL");

    /* Add several functions and variables */
    dwarf_function(db, "add", 0, 1, 0, TYPE_INT);
    dwarf_variable(db, "a", TYPE_INT, 0, 2);
    dwarf_variable(db, "b", TYPE_INT, 4, 2);
    dwarf_line(db, 0, 1, 0, 0);
    dwarf_line(db, 0, 2, 2, 0);

    dwarf_function(db, "main", 0, 5, 0, TYPE_VOID);
    dwarf_line(db, 0, 5, 0, 0);
    dwarf_line(db, 0, 6, 4, 0);

    const char *out_path = "/tmp/test_compiler_utils_dwarf.s";
    FILE *out = fopen(out_path, "w");
    ASSERT(out != NULL, "fopen failed");
    dwarf_emit_asm_sections(db, out);
    fclose(out);
    dwarf_free(db);

    /* Verify file was written */
    FILE *check = fopen(out_path, "r");
    ASSERT(check != NULL, "output file not created");
    fclose(check);
    PASS(test_name);
}

/* ── nanocore_subset tests ───────────────────────────────────────────────── */

static void test_nanocore_is_subset_simple(void) {
    const char *test_name = "nanocore_is_subset: simple arithmetic";
    ASTNode *prog = parse_nano("fn add(a: int, b: int) -> int { return (+ a b) }\n");
    ASSERT(prog != NULL, "parse failed");
    Environment *env = create_environment();
    bool result = nanocore_is_subset(prog, env);
    (void)result; /* Just verify it doesn't crash */
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_nanocore_is_subset_complex(void) {
    const char *test_name = "nanocore_is_subset: complex program";
    const char *src =
        "fn factorial(n: int) -> int {\n"
        "  if (== n 0) { return 1 } else { return (* n (factorial (- n 1))) }\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();
    bool result = nanocore_is_subset(prog, env);
    (void)result;
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_nanocore_trust_report_empty(void) {
    const char *test_name = "nanocore_trust_report: empty program";
    ASTNode *prog = parse_nano("fn id(x: int) -> int { return x }\n");
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();
    TrustReport *report = nanocore_trust_report(prog, env);
    ASSERT(report != NULL, "trust_report returned NULL");
    nanocore_free_trust_report(report);
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_nanocore_trust_report_print(void) {
    const char *test_name = "nanocore_print_trust_report";
    ASTNode *prog = parse_nano(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn sub(a: int, b: int) -> int { return (- a b) }\n"
    );
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();
    TrustReport *report = nanocore_trust_report(prog, env);
    ASSERT(report != NULL, "trust_report returned NULL");
    suppress_stdout();
    nanocore_print_trust_report(report, "test.nano");
    restore_stdout();
    nanocore_free_trust_report(report);
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_nanocore_function_trust(void) {
    const char *test_name = "nanocore_function_trust: pure function";
    ASTNode *prog = parse_nano("fn square(x: int) -> int { return (* x x) }\n");
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();
    /* Get the first item (function node) */
    ASSERT(prog->as.program.count > 0, "program has no items");
    ASTNode *fn = prog->as.program.items[0];
    ASSERT(fn->type == AST_FUNCTION, "first item is not a function");
    TrustLevel level = nanocore_function_trust(fn, env);
    ASSERT(level >= TRUST_VERIFIED && level <= TRUST_UNSAFE, "invalid trust level");
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_nanocore_is_subset_number(void) {
    const char *test_name = "nanocore_is_subset: number literal";
    ASTNode *prog = parse_nano("fn get42() -> int { return 42 }\n");
    ASSERT(prog != NULL, "parse failed");
    Environment *env = create_environment();
    /* Check the return statement's value */
    ASTNode *fn = prog->as.program.items[0];
    bool result = nanocore_is_subset(fn, env);
    (void)result;
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

/* ── nanocore_export tests ───────────────────────────────────────────────── */

static void test_nanocore_export_number(void) {
    const char *test_name = "nanocore_export_sexpr: number";
    ASTNode *prog = parse_nano("fn get42() -> int { return 42 }\n");
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();
    char *sexpr = nanocore_export_sexpr(prog, env);
    /* May return NULL for programs not in subset — just don't crash */
    if (sexpr) free(sexpr);
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_nanocore_export_arithmetic(void) {
    const char *test_name = "nanocore_export_sexpr: arithmetic";
    ASTNode *prog = parse_nano("fn add(a: int, b: int) -> int { return (+ a b) }\n");
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();
    char *sexpr = nanocore_export_sexpr(prog, env);
    if (sexpr) free(sexpr);
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_nanocore_export_if(void) {
    const char *test_name = "nanocore_export_sexpr: if expression";
    const char *src = "fn abs_v(x: int) -> int {\n"
                      "  if (< x 0) { return (- 0 x) } else { return x }\n"
                      "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();
    char *sexpr = nanocore_export_sexpr(prog, env);
    if (sexpr) free(sexpr);
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

/* ── emit_typed_ast tests ────────────────────────────────────────────────── */

static void test_emit_typed_ast_simple(void) {
    const char *test_name = "emit_typed_ast_json: simple function";
    ASTNode *prog = parse_nano("fn id(x: int) -> int { return x }\n");
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    suppress_stdout();
    emit_typed_ast_json("<test>", prog, env);
    restore_stdout();

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_emit_typed_ast_arithmetic(void) {
    const char *test_name = "emit_typed_ast_json: arithmetic";
    ASTNode *prog = parse_nano(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn mul(a: int, b: int) -> int { return (* a b) }\n"
    );
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    suppress_stdout();
    emit_typed_ast_json("<test>", prog, env);
    restore_stdout();

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

/* ── reflection tests ───────────────────────────────────────────────────── */

static void test_reflection_simple(void) {
    const char *test_name = "emit_module_reflection: simple module";
    ASTNode *prog = parse_nano(
        "pub fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "pub fn sub(a: int, b: int) -> int { return (- a b) }\n"
    );
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    const char *out_path = "/tmp/test_compiler_utils_reflection.json";
    bool ok = emit_module_reflection(out_path, prog, env, "testmod");
    ASSERT(ok, "emit_module_reflection returned false");

    /* Verify file was created and has content */
    FILE *f = fopen(out_path, "r");
    ASSERT(f != NULL, "output file not created");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);
    ASSERT(size > 0, "output file is empty");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_reflection_struct(void) {
    const char *test_name = "emit_module_reflection: struct and enum";
    const char *src =
        "pub struct Point { x: int, y: int }\n"
        "pub enum Color { Red, Green, Blue }\n"
        "pub fn origin() -> Point { return Point { x: 0, y: 0 } }\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    const char *out_path = "/tmp/test_compiler_utils_reflection2.json";
    bool ok = emit_module_reflection(out_path, prog, env, "geom");
    ASSERT(ok, "emit_module_reflection returned false");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_reflection_private_only(void) {
    const char *test_name = "emit_module_reflection: private-only module";
    ASTNode *prog = parse_nano("fn helper(x: int) -> int { return x }\n");
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    const char *out_path = "/tmp/test_compiler_utils_reflection3.json";
    bool ok = emit_module_reflection(out_path, prog, env, "private");
    ASSERT(ok, "emit_module_reflection returned false");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_reflection_with_let_constant(void) {
    const char *test_name = "emit_module_reflection: let constant exported";
    const char *src =
        "let MAX_SIZE: int = 100\n"
        "pub fn get_max() -> int { return MAX_SIZE }\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    const char *out_path = "/tmp/test_reflection_const.json";
    bool ok = emit_module_reflection(out_path, prog, env, "consts");
    ASSERT(ok, "emit_module_reflection should succeed");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_reflection_underscore_fn_skipped(void) {
    const char *test_name = "emit_module_reflection: _private fn skipped";
    const char *src =
        "fn _internal(x: int) -> int { return x }\n"
        "pub fn public_fn(x: int) -> int { return x }\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    const char *out_path = "/tmp/test_reflection_underscore.json";
    bool ok = emit_module_reflection(out_path, prog, env, "mod");
    ASSERT(ok, "emit_module_reflection should succeed");

    /* Read output and verify _internal is not in the JSON */
    FILE *f = fopen(out_path, "r");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = malloc((size_t)sz + 1);
        if (buf) {
            if (fread(buf, 1, (size_t)sz, f) > 0) {
                buf[sz] = '\0';
                ASSERT(strstr(buf, "_internal") == NULL,
                       "_internal should not appear in reflection output");
            }
            free(buf);
        }
        fclose(f);
    }

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_reflection_opaque_type(void) {
    const char *test_name = "emit_module_reflection: opaque type exported";
    const char *src =
        "opaque type Handle\n"
        "pub fn create_handle() -> int { return 0 }\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");

    suppress_stderr();
    Environment *env = create_environment();
    type_check(prog, env);
    restore_stderr();

    const char *out_path = "/tmp/test_reflection_opaque.json";
    bool ok = emit_module_reflection(out_path, prog, env, "opaque_mod");
    ASSERT(ok, "emit_module_reflection should succeed");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_reflection_null_path(void) {
    const char *test_name = "emit_module_reflection: NULL output path fails";
    ASTNode *prog = parse_nano("pub fn f() -> int { return 0 }\n");
    ASSERT(prog != NULL, "parse failed");
    Environment *env = create_environment();
    suppress_stderr();
    bool ok = emit_module_reflection(NULL, prog, env, "mod");
    restore_stderr();
    ASSERT(!ok, "should fail with NULL path");
    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

/* ── bench.c tests ──────────────────────────────────────────────────────── */

static void test_bench_print_json(void) {
    const char *test_name = "bench_print_json: success result";
    BenchResult r = {
        .name       = "bench_add",
        .n          = 1000,
        .mean_ns    = 12.5,
        .stddev_ns  = 0.5,
        .min_ns     = 11.0,
        .max_ns     = 15.0,
        .ops_per_sec = 80000000.0,
        .backend    = "native",
        .source_file = "test.nano",
        .error      = NULL,
    };
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen failed");
    bench_print_json(&r, out);
    fclose(out);
    PASS(test_name);
}

static void test_bench_print_json_error(void) {
    const char *test_name = "bench_print_json: error result";
    BenchResult r = {
        .name       = "bench_fail",
        .n          = 0,
        .mean_ns    = 0.0,
        .stddev_ns  = 0.0,
        .min_ns     = 0.0,
        .max_ns     = 0.0,
        .ops_per_sec = 0.0,
        .backend    = "native",
        .source_file = "test.nano",
        .error      = "function not found",
    };
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen failed");
    bench_print_json(&r, out);
    fclose(out);
    PASS(test_name);
}

static void test_bench_print_human(void) {
    const char *test_name = "bench_print_human: success result";
    BenchResult r = {
        .name       = "bench_mul",
        .n          = 5000,
        .mean_ns    = 8.2,
        .stddev_ns  = 0.3,
        .min_ns     = 7.9,
        .max_ns     = 9.0,
        .ops_per_sec = 121951219.0,
        .backend    = "native",
        .source_file = "test.nano",
        .error      = NULL,
    };
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen failed");
    bench_print_human(&r, out);
    fclose(out);
    PASS(test_name);
}

static void test_bench_run_no_bench_fns(void) {
    const char *test_name = "bench_run_program: no @bench functions";
    /* Program with no bench_ functions should return error */
    ASTNode *prog = parse_nano("fn add(a: int, b: int) -> int { return (+ a b) }\n");
    ASSERT(prog != NULL, "parse failed");
    BenchOptions opts = {
        .n_iters = 0,
        .output_format = BENCH_FMT_HUMAN,
        .json_out_path = NULL,
        .backend = "native",
        .verbose = false,
    };
    suppress_stderr();
    int rc = bench_run_program(prog, &opts, "<test>", NULL);
    restore_stderr();
    ASSERT(rc != 0, "expected non-zero return with no bench functions");
    free_ast(prog);
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[compiler_utils] Compiler utility module tests...\n\n");

    printf("DWARF Debug Info:\n");
    test_dwarf_begin_free();
    test_dwarf_add_function();
    test_dwarf_add_variable();
    test_dwarf_add_lines();
    test_dwarf_emit_sections();
    test_dwarf_emit_to_file();

    printf("\nNanoCore Subset:\n");
    test_nanocore_is_subset_simple();
    test_nanocore_is_subset_complex();
    test_nanocore_is_subset_number();
    test_nanocore_trust_report_empty();
    test_nanocore_trust_report_print();
    test_nanocore_function_trust();

    printf("\nNanoCore Export:\n");
    test_nanocore_export_number();
    test_nanocore_export_arithmetic();
    test_nanocore_export_if();

    printf("\nTyped AST Emit:\n");
    test_emit_typed_ast_simple();
    test_emit_typed_ast_arithmetic();

    printf("\nModule Reflection:\n");
    test_reflection_simple();
    test_reflection_struct();
    test_reflection_private_only();
    test_reflection_with_let_constant();
    test_reflection_underscore_fn_skipped();
    test_reflection_opaque_type();
    test_reflection_null_path();

    printf("\nBenchmark Harness:\n");
    test_bench_print_json();
    test_bench_print_json_error();
    test_bench_print_human();
    test_bench_run_no_bench_fns();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
