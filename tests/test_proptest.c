/*
 * test_proptest.c — unit tests for proptest.c (property-based test runner)
 *
 * Tests proptest_run_program() with various nano programs containing
 * property functions (prop_* prefix), and error paths.
 */

#include "../src/nanolang.h"
#include "../src/proptest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* ── Helpers ─────────────────────────────────────────────────────────────── */

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

static ASTNode *parse_nano(const char *src) {
    int n = 0;
    Token *t = tokenize(src, &n);
    if (!t) return NULL;
    ASTNode *p = parse_program(t, n);
    free_tokens(t, n);
    return p;
}

static const PropTestOptions DEFAULT_OPTS = {
    .n_runs = 10,    /* keep tests fast */
    .verbose = false,
    .seed = 42       /* deterministic */
};

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_no_properties(void) {
    const char *test_name = "proptest_run_program: no prop_ functions returns 1";
    ASTNode *prog = parse_nano("fn add(a: int, b: int) -> int { return (+ a b) }\n");
    ASSERT(prog != NULL, "parse should succeed");

    Environment *env = create_environment();
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &DEFAULT_OPTS, "<test>");
    restore_stderr();
    ASSERT(rc == 1, "should return 1 when no property functions found");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_passing_property_commutative(void) {
    const char *test_name = "proptest_run_program: commutative add property passes";
    const char *src =
        "fn prop_add_commutative(a: int, b: int) -> bool {\n"
        "  return (== (+ a b) (+ b a))\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse should succeed");

    Environment *env = create_environment();
    type_check(prog, env);
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &DEFAULT_OPTS, "<test>");
    restore_stderr();
    ASSERT(rc == 0, "commutative add property should pass");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_failing_property(void) {
    const char *test_name = "proptest_run_program: always-false property fails";
    const char *src =
        "fn prop_always_false(a: int) -> bool {\n"
        "  return false\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse should succeed");

    Environment *env = create_environment();
    type_check(prog, env);
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &DEFAULT_OPTS, "<test>");
    restore_stderr();
    ASSERT(rc == 1, "always-false property should fail");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_passing_property_identity(void) {
    const char *test_name = "proptest_run_program: identity property passes";
    const char *src =
        "fn prop_add_zero(a: int) -> bool {\n"
        "  return (== (+ a 0) a)\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse should succeed");

    Environment *env = create_environment();
    type_check(prog, env);
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &DEFAULT_OPTS, "<test>");
    restore_stderr();
    ASSERT(rc == 0, "identity property should pass");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_multiple_properties_all_pass(void) {
    const char *test_name = "proptest_run_program: multiple passing properties";
    const char *src =
        "fn prop_comm(a: int, b: int) -> bool {\n"
        "  return (== (+ a b) (+ b a))\n"
        "}\n"
        "fn prop_identity(a: int) -> bool {\n"
        "  return (== (* a 1) a)\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse should succeed");

    Environment *env = create_environment();
    type_check(prog, env);
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &DEFAULT_OPTS, "<test>");
    restore_stderr();
    ASSERT(rc == 0, "all passing properties should return 0");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_mixed_pass_fail(void) {
    const char *test_name = "proptest_run_program: mixed pass/fail returns 1";
    const char *src =
        "fn prop_good(a: int) -> bool {\n"
        "  return (== (+ a 0) a)\n"
        "}\n"
        "fn prop_bad(a: int) -> bool {\n"
        "  return false\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse should succeed");

    Environment *env = create_environment();
    type_check(prog, env);
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &DEFAULT_OPTS, "<test>");
    restore_stderr();
    ASSERT(rc == 1, "mixed pass/fail should return 1");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_verbose_mode(void) {
    const char *test_name = "proptest_run_program: verbose mode doesn't crash";
    const char *src =
        "fn prop_simple(a: int) -> bool {\n"
        "  return (== a a)\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse should succeed");

    PropTestOptions opts = { .n_runs = 3, .verbose = true, .seed = 1 };
    Environment *env = create_environment();
    type_check(prog, env);
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &opts, "verbose_test.nano");
    restore_stderr();
    ASSERT(rc == 0, "verbose property should still pass");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

static void test_no_args_property(void) {
    const char *test_name = "proptest_run_program: zero-arg property function";
    const char *src =
        "fn prop_true() -> bool {\n"
        "  return true\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse should succeed");

    Environment *env = create_environment();
    type_check(prog, env);
    suppress_stderr();
    int rc = proptest_run_program(prog, env, &DEFAULT_OPTS, "<test>");
    restore_stderr();
    /* Zero-arg property: should run and pass */
    ASSERT(rc == 0, "zero-arg true property should pass");

    free_environment(env);
    free_ast(prog);
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[proptest] Property-based test runner tests...\n\n");

    test_no_properties();
    test_passing_property_commutative();
    test_failing_property();
    test_passing_property_identity();
    test_multiple_properties_all_pass();
    test_mixed_pass_fail();
    test_verbose_mode();
    test_no_args_property();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
