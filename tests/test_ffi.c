/**
 * test_ffi.c — unit tests for interpreter_ffi.c
 *
 * Exercises the FFI lifecycle and non-library-dependent paths:
 *   ffi_init, ffi_is_available, ffi_cleanup,
 *   ffi_load_module (error paths), ffi_call_extern (null/error paths)
 */

#include "../src/nanolang.h"
#include "../src/interpreter_ffi.h"
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

/* ============================================================================
 * Tests
 * ============================================================================ */

void test_ffi_init_and_cleanup(void) {
    bool ok = ffi_init(false);
    ASSERT(ok);
    ffi_cleanup();
}

void test_ffi_init_verbose(void) {
    suppress_stderr();
    bool ok = ffi_init(true);
    restore_stderr();
    ASSERT(ok);
    ffi_cleanup();
}

void test_ffi_is_available_after_init(void) {
    ffi_init(false);
    ASSERT(ffi_is_available());
    ffi_cleanup();
}

void test_ffi_is_available_before_init(void) {
    ffi_cleanup();
    /* After cleanup, should not be available */
    bool avail = ffi_is_available();
    (void)avail; /* Either state is acceptable; just shouldn't crash */
}

void test_ffi_load_nonexistent_module(void) {
    ffi_init(false);
    Environment *env = create_environment();
    suppress_stderr();
    bool loaded = ffi_load_module("nonexistent_mod", "/nonexistent/path/mod.so", env, false);
    restore_stderr();
    /* Should return false for a non-existent path, not crash */
    ASSERT(!loaded);
    free_environment(env);
    ffi_cleanup();
}

void test_ffi_load_null_args(void) {
    ffi_init(false);
    Environment *env = create_environment();
    suppress_stderr();
    bool loaded = ffi_load_module(NULL, "/some/path.so", env, false);
    restore_stderr();
    ASSERT(!loaded);
    free_environment(env);
    ffi_cleanup();
}

void test_ffi_call_extern_no_module(void) {
    ffi_init(false);
    Environment *env = create_environment();
    /* Calling a function from an unloaded module should fail gracefully */
    suppress_stderr();
    Value result = ffi_call_extern("nonexistent_fn", NULL, 0, NULL, env);
    restore_stderr();
    (void)result; /* Should not crash */
    free_environment(env);
    ffi_cleanup();
}

void test_ffi_double_init(void) {
    bool ok1 = ffi_init(false);
    bool ok2 = ffi_init(false);
    ASSERT(ok1);
    ASSERT(ok2);  /* Double init should be idempotent */
    ffi_cleanup();
}

void test_ffi_double_cleanup(void) {
    ffi_init(false);
    ffi_cleanup();
    ffi_cleanup();  /* Double cleanup should not crash */
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== FFI Tests ===\n");
    TEST(ffi_init_and_cleanup);
    TEST(ffi_init_verbose);
    TEST(ffi_is_available_after_init);
    TEST(ffi_is_available_before_init);
    TEST(ffi_load_nonexistent_module);
    TEST(ffi_load_null_args);
    TEST(ffi_call_extern_no_module);
    TEST(ffi_double_init);
    TEST(ffi_double_cleanup);

    printf("\n✓ All FFI tests passed!\n");
    return 0;
}
