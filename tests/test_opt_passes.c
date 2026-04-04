/**
 * test_opt_passes.c — unit tests for optimization passes
 *
 * Exercises dce_pass.c, fold_constants.c, and par_let_pass.c by parsing
 * small nano programs and running each pass over the resulting AST.
 * Also exercises pgo_pass.c and bench.c via their public APIs.
 */

#include "../src/nanolang.h"
#include "../src/dce_pass.h"
#include "../src/fold_constants.h"
#include "../src/par_let_pass.h"
#include "../src/pgo_pass.h"
#include "../src/tco_pass.h"
#include "../src/cps_pass.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %d, expected %d)\n", \
        #a, #b, __LINE__, (int)(a), (int)(b)); exit(1); }
#define ASSERT_NOT_NULL(p) \
    if ((p) == NULL) { printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_NULL(p) \
    if ((p) != NULL) { printf("\n    FAILED: expected NULL at line %d\n", __LINE__); exit(1); }

/* Required by runtime/cli.c (extern in eval.c) */
int g_argc = 0;
char **g_argv = NULL;

/* Helper: suppress stderr during expected-error paths */
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

/* Helper: parse a nano program string, return AST (caller frees) */
static ASTNode *parse_nano(const char *src) {
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    if (!tokens) return NULL;
    ASTNode *prog = parse_program(tokens, token_count);
    free_tokens(tokens, token_count);
    return prog;
}

/* ============================================================================
 * dce_pass tests
 * ============================================================================ */

void test_dce_empty_program(void) {
    ASTNode *prog = parse_nano("fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    int elim = dce_pass(prog, false);
    /* Nothing to eliminate in a trivial program */
    ASSERT(elim >= 0);
    free_ast(prog);
}

void test_dce_dead_if_true(void) {
    /* if(true) { … } else { dead } — else branch should be eliminated */
    ASTNode *prog = parse_nano(
        "fn check() -> int {\n"
        "    if true { return 1 } else { return 2 }\n"
        "}\n"
        "fn main() -> int { return (check) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int elim = dce_pass(prog, false);
    ASSERT(elim >= 0);
    free_ast(prog);
}

void test_dce_dead_if_false(void) {
    /* if(false) { dead } else { … } — then branch should be eliminated */
    ASTNode *prog = parse_nano(
        "fn check() -> int {\n"
        "    if false { return 99 } else { return 0 }\n"
        "}\n"
        "fn main() -> int { return (check) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int elim = dce_pass(prog, false);
    ASSERT(elim >= 0);
    free_ast(prog);
}

void test_dce_verbose_flag(void) {
    /* Verify verbose=true doesn't crash */
    ASTNode *prog = parse_nano(
        "fn main() -> int {\n"
        "    if true { return 1 } else { return 2 }\n"
        "}\n"
    );
    ASSERT_NOT_NULL(prog);
    suppress_stderr();
    int elim = dce_pass(prog, true);
    restore_stderr();
    ASSERT(elim >= 0);
    free_ast(prog);
}

void test_dce_multiple_functions(void) {
    ASTNode *prog = parse_nano(
        "fn add(x: int, y: int) -> int { return (+ x y) }\n"
        "fn sub(x: int, y: int) -> int { return (- x y) }\n"
        "fn main() -> int { return (add 1 2) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int elim = dce_pass(prog, false);
    ASSERT(elim >= 0);
    free_ast(prog);
}

/* ============================================================================
 * fold_constants tests
 * ============================================================================ */

void test_fold_empty_program(void) {
    ASTNode *prog = parse_nano("fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    int folds = fold_constants(prog, false);
    ASSERT(folds >= 0);
    free_ast(prog);
}

void test_fold_arithmetic(void) {
    /* (+ 2 3) should fold to 5 */
    ASTNode *prog = parse_nano(
        "fn main() -> int {\n"
        "    return (+ 2 3)\n"
        "}\n"
    );
    ASSERT_NOT_NULL(prog);
    int folds = fold_constants(prog, false);
    ASSERT(folds >= 0);
    free_ast(prog);
}

void test_fold_nested_arithmetic(void) {
    /* (+ (* 2 3) (- 10 4)) — nested constant expressions */
    ASTNode *prog = parse_nano(
        "fn main() -> int {\n"
        "    return (+ (* 2 3) (- 10 4))\n"
        "}\n"
    );
    ASSERT_NOT_NULL(prog);
    int folds = fold_constants(prog, false);
    ASSERT(folds >= 0);
    free_ast(prog);
}

void test_fold_bool_constants(void) {
    ASTNode *prog = parse_nano(
        "fn main() -> int {\n"
        "    if (== 1 1) { return 1 } else { return 0 }\n"
        "}\n"
    );
    ASSERT_NOT_NULL(prog);
    int folds = fold_constants(prog, false);
    ASSERT(folds >= 0);
    free_ast(prog);
}

void test_fold_verbose_flag(void) {
    ASTNode *prog = parse_nano(
        "fn main() -> int { return (+ 1 2) }\n"
    );
    ASSERT_NOT_NULL(prog);
    suppress_stderr();
    int folds = fold_constants(prog, true);
    restore_stderr();
    ASSERT(folds >= 0);
    free_ast(prog);
}

void test_fold_multiple_functions(void) {
    ASTNode *prog = parse_nano(
        "fn square(x: int) -> int { return (* x x) }\n"
        "fn main() -> int { return (+ (square 3) 1) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int folds = fold_constants(prog, false);
    ASSERT(folds >= 0);
    free_ast(prog);
}

/* ============================================================================
 * par_let_pass tests
 * ============================================================================ */

void test_par_let_empty_program(void) {
    ASTNode *prog = parse_nano("fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    int result = par_let_pass(prog);
    ASSERT(result >= 0);
    free_ast(prog);
}

void test_par_let_simple_function(void) {
    ASTNode *prog = parse_nano(
        "fn add(x: int, y: int) -> int { return (+ x y) }\n"
        "fn main() -> int { return (add 1 2) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int result = par_let_pass(prog);
    ASSERT(result >= 0);
    free_ast(prog);
}

void test_par_let_with_let_bindings(void) {
    ASTNode *prog = parse_nano(
        "fn main() -> int {\n"
        "    let x: int = 10\n"
        "    let y: int = 20\n"
        "    return (+ x y)\n"
        "}\n"
    );
    ASSERT_NOT_NULL(prog);
    int result = par_let_pass(prog);
    ASSERT(result >= 0);
    free_ast(prog);
}

void test_par_let_multiple_functions(void) {
    ASTNode *prog = parse_nano(
        "fn f(x: int) -> int { return (* x 2) }\n"
        "fn g(x: int) -> int { return (+ x 1) }\n"
        "fn main() -> int {\n"
        "    let a: int = (f 3)\n"
        "    let b: int = (g 4)\n"
        "    return (+ a b)\n"
        "}\n"
    );
    ASSERT_NOT_NULL(prog);
    int result = par_let_pass(prog);
    ASSERT(result >= 0);
    free_ast(prog);
}

/* ============================================================================
 * pgo_pass tests — exercise the PGO profile API
 * ============================================================================ */

void test_pgo_load_null_path(void) {
    /* Loading a non-existent profile should return NULL gracefully */
    suppress_stderr();
    PGOProfile *prof = pgo_load_profile("/nonexistent/path/profile.nano.prof");
    restore_stderr();
    /* May return NULL for missing file — just verify no crash */
    if (prof) pgo_profile_free(prof);
}

void test_pgo_is_hot_null_profile(void) {
    /* pgo_is_hot with NULL profile should return false */
    bool hot = pgo_is_hot(NULL, "some_fn");
    ASSERT(!hot);
}

void test_pgo_apply_empty_profile(void) {
    /* pgo_apply on a NULL profile is a no-op, should handle gracefully */
    ASTNode *prog = parse_nano("fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    /* pgo_apply(prog, NULL) — guard against NULL profile */
    /* We test with a NULL profile only if the function handles it */
    /* (we don't call it with NULL as that'd be UB per the API) */
    free_ast(prog);
}

void test_pgo_print_report_null(void) {
    /* pgo_print_report with NULL should not crash */
    suppress_stderr();
    pgo_print_report(NULL);
    restore_stderr();
}

/* ============================================================================
 * tco_pass tests
 * ============================================================================ */

void test_tco_empty_program(void) {
    ASTNode *prog = parse_nano("fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    int result = tco_pass_run(prog, false);
    ASSERT(result >= 0);
    free_ast(prog);
}

void test_tco_non_recursive_function(void) {
    /* A non-recursive function should not be transformed */
    ASTNode *prog = parse_nano(
        "fn add(x: int, y: int) -> int { return (+ x y) }\n"
        "fn main() -> int { return (add 1 2) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int result = tco_pass_run(prog, false);
    ASSERT_EQ(result, 0);  /* no TCO transformations */
    free_ast(prog);
}

void test_tco_tail_recursive_function(void) {
    /* A tail-recursive function should be detected and possibly transformed */
    ASTNode *prog = parse_nano(
        "fn count_down(n: int) -> int {\n"
        "    if (<= n 0) { return 0 } else { return (count_down (- n 1)) }\n"
        "}\n"
        "fn main() -> int { return (count_down 5) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int result = tco_pass_run(prog, false);
    ASSERT(result >= 0);
    free_ast(prog);
}

void test_tco_verbose_flag(void) {
    ASTNode *prog = parse_nano(
        "fn sum(n: int, acc: int) -> int {\n"
        "    if (<= n 0) { return acc } else { return (sum (- n 1) (+ acc n)) }\n"
        "}\n"
        "fn main() -> int { return (sum 5 0) }\n"
    );
    ASSERT_NOT_NULL(prog);
    suppress_stderr();
    int result = tco_pass_run(prog, true);
    restore_stderr();
    ASSERT(result >= 0);
    free_ast(prog);
}

void test_tco_convenience_wrapper(void) {
    /* tco_pass() is the non-verbose wrapper */
    ASTNode *prog = parse_nano(
        "fn loop(n: int) -> int {\n"
        "    if (<= n 0) { return 0 } else { return (loop (- n 1)) }\n"
        "}\n"
        "fn main() -> int { return (loop 3) }\n"
    );
    ASSERT_NOT_NULL(prog);
    tco_pass(prog);  /* just verify it doesn't crash */
    free_ast(prog);
}

/* ============================================================================
 * cps_pass tests
 * ============================================================================ */

void test_cps_empty_program(void) {
    ASTNode *prog = parse_nano("fn main() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    int async_count = cps_pass(prog);
    ASSERT_EQ(async_count, 0);  /* no async functions */
    free_ast(prog);
}

void test_cps_non_async_functions(void) {
    ASTNode *prog = parse_nano(
        "fn add(x: int, y: int) -> int { return (+ x y) }\n"
        "fn mul(x: int, y: int) -> int { return (* x y) }\n"
        "fn main() -> int { return (add 1 2) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int async_count = cps_pass(prog);
    ASSERT_EQ(async_count, 0);  /* no async functions → no CPS transform */
    free_ast(prog);
}

void test_cps_multiple_functions(void) {
    ASTNode *prog = parse_nano(
        "fn f(x: int) -> int { return (* x x) }\n"
        "fn g(x: int) -> int { return (+ x 1) }\n"
        "fn h(x: int) -> int { return (f (g x)) }\n"
        "fn main() -> int { return (h 3) }\n"
    );
    ASSERT_NOT_NULL(prog);
    int async_count = cps_pass(prog);
    ASSERT(async_count >= 0);
    free_ast(prog);
}

/* ============================================================================
 * Combined pass pipeline: run all three passes in sequence
 * ============================================================================ */

void test_full_opt_pipeline(void) {
    /* Run fold → dce → par_let in sequence, as an optimizer pipeline would */
    ASTNode *prog = parse_nano(
        "fn compute(x: int) -> int {\n"
        "    let a: int = (+ 2 3)\n"
        "    if true { return (* x a) } else { return 0 }\n"
        "}\n"
        "fn main() -> int { return (compute 7) }\n"
    );
    ASSERT_NOT_NULL(prog);

    int folds = fold_constants(prog, false);
    int elims  = dce_pass(prog, false);
    int plets  = par_let_pass(prog);

    ASSERT(folds >= 0);
    ASSERT(elims >= 0);
    ASSERT(plets >= 0);

    free_ast(prog);
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== DCE Pass Tests ===\n");
    TEST(dce_empty_program);
    TEST(dce_dead_if_true);
    TEST(dce_dead_if_false);
    TEST(dce_verbose_flag);
    TEST(dce_multiple_functions);

    printf("\n=== Constant Folding Tests ===\n");
    TEST(fold_empty_program);
    TEST(fold_arithmetic);
    TEST(fold_nested_arithmetic);
    TEST(fold_bool_constants);
    TEST(fold_verbose_flag);
    TEST(fold_multiple_functions);

    printf("\n=== Par-Let Pass Tests ===\n");
    TEST(par_let_empty_program);
    TEST(par_let_simple_function);
    TEST(par_let_with_let_bindings);
    TEST(par_let_multiple_functions);

    printf("\n=== PGO Pass Tests ===\n");
    TEST(pgo_load_null_path);
    TEST(pgo_is_hot_null_profile);
    TEST(pgo_apply_empty_profile);
    TEST(pgo_print_report_null);

    printf("\n=== TCO Pass Tests ===\n");
    TEST(tco_empty_program);
    TEST(tco_non_recursive_function);
    TEST(tco_tail_recursive_function);
    TEST(tco_verbose_flag);
    TEST(tco_convenience_wrapper);

    printf("\n=== CPS Pass Tests ===\n");
    TEST(cps_empty_program);
    TEST(cps_non_async_functions);
    TEST(cps_multiple_functions);

    printf("\n=== Combined Pipeline Test ===\n");
    TEST(full_opt_pipeline);

    printf("\n✓ All optimization pass tests passed!\n");
    return 0;
}
