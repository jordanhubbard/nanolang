/**
 * test_ffi.c — unit tests for interpreter_ffi.c
 *
 * Exercises the FFI lifecycle and non-library-dependent paths:
 *   ffi_init, ffi_is_available, ffi_cleanup,
 *   ffi_load_module (error paths), ffi_call_extern (null/error paths)
 */

#include "../src/nanolang.h"
#include "../src/interpreter_ffi.h"
#include "../src/runtime/ffi_loader.h"
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

void test_ffi_checked_status(void) {
    Environment *env = create_environment();
    Parameter param = {0};
    param.type = TYPE_INT;
    Function function = {0};
    function.params = &param;
    function.param_count = 1;
    function.return_type = TYPE_INT;
    Value arg = create_int(-42);
    bool success = true;
    ffi_cleanup();
    Value result = ffi_call_extern_checked("llabs", &arg, 1, &function, env, &success);
    ASSERT(!success && result.type == VAL_VOID);
    ASSERT(ffi_init(false));
    result = ffi_call_extern_checked("llabs", &arg, 1, &function, env, &success);
    ASSERT(success && result.type == VAL_INT && result.as.int_val == 42);

    result = ffi_call_extern_checked("nano_missing_checked_symbol", &arg, 1, &function, env, &success);
    ASSERT(!success && result.type == VAL_VOID);
    result = ffi_call_extern_checked("llabs", &arg, -1, &function, env, &success);
    ASSERT(!success && result.type == VAL_VOID);
    result = ffi_call_extern_checked("llabs", NULL, 1, &function, env, &success);
    ASSERT(!success && result.type == VAL_VOID);
    result = ffi_call_extern_checked("llabs", &arg, 0, &function, env, &success);
    ASSERT(!success && result.type == VAL_VOID);
    result = ffi_call_extern_checked("llabs", &arg, 1, NULL, env, &success);
    ASSERT(!success && result.type == VAL_VOID);

    arg = create_bool(true);
    result = ffi_call_extern_checked("llabs", &arg, 1, &function, env, &success);
    ASSERT(!success && result.type == VAL_VOID);
    param.type = TYPE_FLOAT;
    result = ffi_call_extern_checked("llabs", &arg, 1, &function, env, &success);
    ASSERT(!success && result.type == VAL_VOID);
    arg = create_float(0.0);
    function.return_type = TYPE_FLOAT;
    result = ffi_call_extern_checked("erf", &arg, 1, &function, env, &success);
    ASSERT(success && result.type == VAL_FLOAT && result.as.float_val == 0.0);

    param.type = TYPE_OPAQUE;
    function.return_type = TYPE_VOID;
    arg = create_int(0);
    result = ffi_call_extern_checked("free", &arg, 1, &function, env, &success);
    ASSERT(success && result.type == VAL_VOID);

    function.param_count = 0;
    function.return_type = TYPE_BOOL;
    result = ffi_call_extern_checked("___module_is_unsafe_missing", NULL, 0, &function, env, &success);
    ASSERT(success && result.type == VAL_BOOL && !result.as.bool_val);
    free_environment(env);
    ffi_cleanup();
}

void test_ffi_native_signatures(void) {
    ASSERT(ffi_init(false));
    ASSERT(ffi_loader_open("abi_test", "obj/test_interpreter_ffi_native.so"));
    Environment *env = create_environment();
    Parameter params[10] = {0};
    Function function = {0};
    function.params = params;
    Value args[10];
    bool success;
    int64_t pointed = 40;
    Type mixed[] = {TYPE_INT, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING, TYPE_OPAQUE};
    for (int i = 0; i < 5; i++) params[i].type = mixed[i];
    function.param_count = 5;
    function.return_type = TYPE_FLOAT;
    args[0] = create_int(2);
    args[1] = create_float(0.5);
    args[2] = create_bool(true);
    args[3] = create_string("native");
    args[4] = create_int((int64_t)(intptr_t)&pointed);
    Value result = ffi_call_extern_checked("ffi_test_mixed", args, 5, &function, env, &success);
    ASSERT(success && result.type == VAL_FLOAT && result.as.float_val == 42.5);

    function.param_count = 3;
    function.return_type = TYPE_BOOL;
    params[0].type = TYPE_BOOL; params[1].type = TYPE_INT; params[2].type = TYPE_FLOAT;
    args[0] = create_bool(true); args[1] = create_int(42); args[2] = create_float(1.25);
    result = ffi_call_extern_checked("ffi_test_bool", args, 3, &function, env, &success);
    ASSERT(success && result.type == VAL_BOOL && result.as.bool_val);
    args[0] = create_bool(false);
    result = ffi_call_extern_checked("ffi_test_bool", args, 3, &function, env, &success);
    ASSERT(success && result.type == VAL_BOOL && !result.as.bool_val);

    function.return_type = TYPE_VOID;
    params[0].type = TYPE_FLOAT; params[1].type = TYPE_INT; params[2].type = TYPE_BOOL;
    args[0] = create_float(1.25); args[1] = create_int(-42); args[2] = create_bool(true);
    result = ffi_call_extern_checked("ffi_test_void", args, 3, &function, env, &success);
    ASSERT(success && result.type == VAL_VOID);
    function.param_count = 0;
    function.return_type = TYPE_INT;
    result = ffi_call_extern_checked("ffi_test_observed", NULL, 0, &function, env, &success);
    ASSERT(success && result.type == VAL_INT && result.as.int_val == -42);
    function.return_type = TYPE_STRING;
    result = ffi_call_extern_checked("ffi_test_string", NULL, 0, &function, env, &success);
    ASSERT(success && result.type == VAL_STRING && !strcmp(result.as.string_val, "native"));

    function.param_count = 1;
    function.return_type = TYPE_OPAQUE;
    params[0].type = TYPE_OPAQUE;
    args[0] = create_int((int64_t)(intptr_t)&pointed);
    result = ffi_call_extern_checked("ffi_test_pointer", args, 1, &function, env, &success);
    ASSERT(success && result.type == VAL_INT && result.as.int_val == args[0].as.int_val);

    function.param_count = 10;
    function.return_type = TYPE_FLOAT;
    for (int i = 0; i < 10; i++) { params[i].type = TYPE_FLOAT; args[i] = create_float(i+1); }
    result = ffi_call_extern_checked("ffi_test_ten_doubles", args, 10, &function, env, &success);
    ASSERT(success && result.type == VAL_FLOAT && result.as.float_val == 385.0);
    function.return_type = TYPE_INT;
    for (int i = 0; i < 10; i++) { params[i].type = TYPE_INT; args[i] = create_int(i+1); }
    result = ffi_call_extern_checked("ffi_test_ten_ints", args, 10, &function, env, &success);
    ASSERT(success && result.type == VAL_INT && result.as.int_val == 385);
    free_environment(env);
    ffi_cleanup();
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
    TEST(ffi_checked_status);
    TEST(ffi_native_signatures);

    printf("\n✓ All FFI tests passed!\n");
    return 0;
}
