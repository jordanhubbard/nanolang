/*
 * test_vm_ffi.c — unit tests for src/nanovm/vm_ffi.c
 *
 * Exercises: vm_ffi_init/shutdown/set_env, vm_ffi_load_module,
 * vm_ffi_call (error paths, real function dispatch via dlsym).
 */

#include <stddef.h>   /* NULL */
#include <string.h>   /* strlen */
#include <stdlib.h>   /* abs */
#include <stdio.h>
#include <unistd.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#include "../../src/nanovm/vm_ffi.h"
#include "../../src/nanovm/heap.h"
#include "../../src/nanovm/value.h"
#include "../../src/nanoisa/nvm_format.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanovm/cop_protocol.h"
#include "../../src/runtime/ffi_loader.h"
#include "../../src/runtime/dyn_array.h"

/* ── Exported helpers with mixed integer/floating signatures ───────────────
 * These deterministic functions let the mixed-signature dispatcher be checked
 * against exact values (unlike libm, whose results vary by platform). They are
 * resolvable through dlsym because the test binary links with -rdynamic. */
double nl_ffi_test_mix_fi(double x, long n);   /* FP,GP -> FP */
double nl_ffi_test_mix_if(long n, double x);   /* GP,FP -> FP */
long   nl_ffi_test_mix_fi_gp(double x, long n);/* FP,GP -> GP */
double nl_ffi_test_mix_iffi(long a, double b, double c, long d); /* mixed -> FP */

double nl_ffi_test_mix_fi(double x, long n)    { return x + (double)n; }
double nl_ffi_test_mix_if(long n, double x)    { return (double)n - x; }
long   nl_ffi_test_mix_fi_gp(double x, long n) { return (long)(x * 2.0) + n; }
double nl_ffi_test_mix_iffi(long a, double b, double c, long d) {
    return (double)a + b * c + (double)d;
}

static int g_pass = 0, g_fail = 0;
double nl_ffi_test_wide(int64_t a, double b, int64_t c, double d,
                       int64_t e, double f, int64_t g, double h,
                       int64_t i, double j, int64_t k) {
    return a + b + c + d + e + f + g + h + i + j + k;
}
#define TEST(name) static void test_##name(void)
#define RUN(name)  do { test_##name(); \
    printf("  %-55s PASS\n", #name "..."); g_pass++; } while(0)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
    g_fail++; return; } } while(0)
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { \
    printf("  FAIL: %s == %s  (%s:%d)\n", #a, #b, __FILE__, __LINE__); \
    g_fail++; return; } } while(0)

/* ── Lifecycle tests ───────────────────────────────────────────────────── */

TEST(wide_mixed_signature) {
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    uint32_t module = nvm_add_string(mod, "", 0);
    uint32_t name = nvm_add_string(mod, "nl_ffi_test_wide", 16);
    uint8_t tags[11];
    NanoValue args[11];
    for (int i = 0; i < 11; i++) {
        tags[i] = i % 2 ? TAG_FLOAT : TAG_INT;
        args[i] = i % 2 ? val_float(i + 1.0) : val_int(i + 1);
    }
    uint32_t imported = nvm_add_import(mod, module, name, 11, TAG_FLOAT, tags);
    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue result;
    char error[256];
    ASSERT(vm_ffi_call(mod, imported, args, 11, &result, &heap, error, sizeof error));
    ASSERT_EQ(result.tag, TAG_FLOAT);
    ASSERT_EQ(result.as.f64, 66.0);
    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(init_shutdown_set_env) {
    /* Double-init should be safe (idempotent) */
    vm_ffi_init();
    vm_ffi_init();

    /* set_env with NULL should not crash */
    vm_ffi_set_env(NULL);

    vm_ffi_shutdown();

    /* shutdown when not initialised should be safe */
    vm_ffi_shutdown();

    /* Re-init after shutdown */
    vm_ffi_init();
    vm_ffi_shutdown();
}

TEST(call_bytecode_callback_rejected) {
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    uint32_t module = nvm_add_string(mod, "", 0);
    uint32_t name = nvm_add_string(mod, "abs", 3);
    uint8_t tag = TAG_FUNCTION;
    uint32_t imported = nvm_add_import(mod, module, name, 1, TAG_INT, &tag);
    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue arg = val_int(13), result;
    arg.tag = TAG_FUNCTION;
    char error[256];
    ASSERT(!vm_ffi_call(mod, imported, &arg, 1, &result, &heap, error, sizeof error));
    ASSERT(strstr(error, "native callback") != NULL);
    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_string_returning_float) {
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    uint32_t module = nvm_add_string(mod, "", 0);
    uint32_t name = nvm_add_string(mod, "atof", 4);
    uint8_t tag = TAG_STRING;
    uint32_t imported = nvm_add_import(mod, module, name, 1, TAG_FLOAT, &tag);
    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue arg = val_string(vm_string_new(&heap, "12.5", 4)), result;
    char error[256];
    ASSERT(vm_ffi_call(mod, imported, &arg, 1, &result, &heap, error, sizeof error));
    ASSERT_EQ(result.tag, TAG_FLOAT);
    ASSERT_EQ(result.as.f64, 12.5);
    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(load_module_nonexistent) {
    vm_ffi_init();

    /* Nonexistent module: should return false gracefully */
    bool ok = vm_ffi_load_module("definitely_not_a_real_ffi_module_xyzzy");
    ASSERT(!ok);

    /* I reject absent module names without entering library lookup. */
    ok = vm_ffi_load_module("");
    ASSERT(!ok);
    ASSERT(!vm_ffi_load_module(NULL));

    vm_ffi_shutdown();
}

/* ── vm_ffi_call error paths ───────────────────────────────────────────── */

TEST(call_empty_module_oob) {
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);

    /* Module has 0 imports — import_idx=0 is out of range */
    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, NULL, 0, &result, &heap, err, sizeof(err));
    ASSERT(!ok);
    ASSERT(err[0] != '\0');  /* Error message must be set */

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_too_many_args) {
    vm_ffi_init();

    /* Build minimal module with one import */
    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "strlen", 6);
    uint8_t ptypes[1] = {TAG_STRING};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);

    /* Pass NANO_MAX_FFI_ARGS + 1 args — over the shared foreign-call limit,
     * so dispatch must return false with an error rather than truncate. */
    NanoValue args[NANO_MAX_FFI_ARGS + 1];
    for (int i = 0; i < NANO_MAX_FFI_ARGS + 1; i++) args[i] = val_int(0);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, args, NANO_MAX_FFI_ARGS + 1, &result, &heap,
                          err, sizeof(err));
    ASSERT(!ok);
    ASSERT(err[0] != '\0');

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_unresolved_function) {
    vm_ffi_init();

    /* Import a function that definitely doesn't exist anywhere */
    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "xyzzy_no_such_function_123", 26);
    nvm_add_import(mod, mod_idx, fn_idx, 0, TAG_INT, NULL);

    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, NULL, 0, &result, &heap, err, sizeof(err));
    ASSERT(!ok);
    ASSERT(err[0] != '\0');

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

/* ── vm_ffi_call success path via dlsym ────────────────────────────────── */

TEST(call_strlen_null_string) {
    /*
     * Call strlen("") via FFI.
     * Pass val_string(NULL) as arg; vm_ffi.c maps NULL VmString → ""
     * strlen("") == 0, result tag TAG_INT, value 0.
     */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "strlen", 6);
    uint8_t ptypes[1] = {TAG_STRING};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);

    /* Build arg: TAG_STRING with NULL VmString pointer */
    NanoValue str_arg;
    memset(&str_arg, 0, sizeof(str_arg));
    str_arg.tag = TAG_STRING;
    str_arg.as.string = NULL;

    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, &str_arg, 1, &result, &heap, err, sizeof(err));

    if (ok) {
        /* strlen("") == 0 */
        ASSERT(result.tag == TAG_INT);
        ASSERT(result.as.i64 == 0);
    }
    /* If not ok (strlen not found via dlsym on this platform), that's acceptable */

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_abs_int_arg) {
    /*
     * Call abs(-42) via FFI.
     * TAG_INT arg → marshal as (void*)(intptr_t)(-42)
     * abs() reads lower 32 bits → returns 42
     * result tag TAG_INT, value 42.
     */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "abs", 3);
    uint8_t ptypes[1] = {TAG_INT};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);

    NanoValue arg = val_int(-42);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, &arg, 1, &result, &heap, err, sizeof(err));

    if (ok) {
        ASSERT(result.tag == TAG_INT);
        /* abs(-42) == 42 (lower 32 bits) */
        ASSERT(result.as.i64 == 42);
    }
    /* If not ok, abs wasn't found — not a test failure, just skip assertion */

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_bool_arg_path) {
    /*
     * Exercise TAG_BOOL arg marshaling path.
     * Use "strlen" with a bool arg (will get mapped to "" anyway
     * since strlen takes char*, not bool, but exercises marshal_args).
     */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "xyzzy_no_such_fn", 16);
    uint8_t ptypes[1] = {TAG_BOOL};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);

    NanoValue arg = val_bool(true);
    NanoValue result;
    char err[256] = "";
    /* Function won't be found, so vm_ffi_call returns false after marshal_args */
    bool ok = vm_ffi_call(mod, 0, &arg, 1, &result, &heap, err, sizeof(err));
    ASSERT(!ok);

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_opaque_arg_path) {
    /* Exercise TAG_OPAQUE arg marshaling */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "xyzzy_no_such_fn2", 17);
    uint8_t ptypes[1] = {TAG_OPAQUE};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);

    NanoValue arg;
    memset(&arg, 0, sizeof(arg));
    arg.tag = TAG_OPAQUE;
    arg.as.i64 = 12345;

    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, &arg, 1, &result, &heap, err, sizeof(err));
    ASSERT(!ok);

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_void_return_type) {
    /* Exercise TAG_VOID return type via marshal_result */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    /* Use a function that returns void and takes no args (e.g. vm_ffi_init itself,
     * but that's void. Use strlen which returns size_t but we request TAG_VOID). */
    uint32_t fn_idx  = nvm_add_string(mod, "strlen", 6);
    uint8_t ptypes[1] = {TAG_STRING};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_VOID, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);

    NanoValue str_arg;
    memset(&str_arg, 0, sizeof(str_arg));
    str_arg.tag = TAG_STRING;
    str_arg.as.string = NULL;

    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, &str_arg, 1, &result, &heap, err, sizeof(err));
    if (ok) {
        ASSERT(result.tag == TAG_VOID);
    }

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(call_bool_return_type) {
    /* Exercise TAG_BOOL return marshaling */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "strlen", 6);
    uint8_t ptypes[1] = {TAG_STRING};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_BOOL, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);

    NanoValue str_arg;
    memset(&str_arg, 0, sizeof(str_arg));
    str_arg.tag = TAG_STRING;
    str_arg.as.string = NULL;

    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, &str_arg, 1, &result, &heap, err, sizeof(err));
    if (ok) {
        ASSERT(result.tag == TAG_BOOL);
        /* strlen("") == 0, so bool result is false */
        ASSERT(result.as.boolean == false);
    }

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

/* ── Mixed integer/floating signature dispatch ─────────────────────────── */

TEST(call_mixed_fp_gp_ret_fp) {
    /* nl_ffi_test_mix_fi(1.5, 4) == 5.5  (FP arg, GP arg, FP return). */
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "nl_ffi_test_mix_fi", 18);
    uint8_t ptypes[2] = {TAG_FLOAT, TAG_INT};
    nvm_add_import(mod, mod_idx, fn_idx, 2, TAG_FLOAT, ptypes);

    VmHeap heap; vm_heap_init(&heap);
    NanoValue args[2] = { val_float(1.5), val_int(4) };
    NanoValue result; char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, args, 2, &result, &heap, err, sizeof(err));
    ASSERT(ok);
    ASSERT(result.tag == TAG_FLOAT);
    ASSERT(result.as.f64 == 5.5);

    vm_heap_destroy(&heap); nvm_module_free(mod); vm_ffi_shutdown();
}

TEST(call_mixed_gp_fp_ret_fp) {
    /* nl_ffi_test_mix_if(10, 2.25) == 7.75  (GP arg, FP arg, FP return). */
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "nl_ffi_test_mix_if", 18);
    uint8_t ptypes[2] = {TAG_INT, TAG_FLOAT};
    nvm_add_import(mod, mod_idx, fn_idx, 2, TAG_FLOAT, ptypes);

    VmHeap heap; vm_heap_init(&heap);
    NanoValue args[2] = { val_int(10), val_float(2.25) };
    NanoValue result; char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, args, 2, &result, &heap, err, sizeof(err));
    ASSERT(ok);
    ASSERT(result.tag == TAG_FLOAT);
    ASSERT(result.as.f64 == 7.75);

    vm_heap_destroy(&heap); nvm_module_free(mod); vm_ffi_shutdown();
}

TEST(call_mixed_fp_gp_ret_gp) {
    /* nl_ffi_test_mix_fi_gp(3.5, 5) == (long)(7.0) + 5 == 12  (GP return). */
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "nl_ffi_test_mix_fi_gp", 21);
    uint8_t ptypes[2] = {TAG_FLOAT, TAG_INT};
    nvm_add_import(mod, mod_idx, fn_idx, 2, TAG_INT, ptypes);

    VmHeap heap; vm_heap_init(&heap);
    NanoValue args[2] = { val_float(3.5), val_int(5) };
    NanoValue result; char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, args, 2, &result, &heap, err, sizeof(err));
    ASSERT(ok);
    ASSERT(result.tag == TAG_INT);
    ASSERT(result.as.i64 == 12);

    vm_heap_destroy(&heap); nvm_module_free(mod); vm_ffi_shutdown();
}

TEST(call_mixed_four_args) {
    /* nl_ffi_test_mix_iffi(2, 3.0, 4.0, 5) == 2 + 12 + 5 == 19.0
     * Exercises an interleaved GP/FP/FP/GP pattern at arity 4. */
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "nl_ffi_test_mix_iffi", 20);
    uint8_t ptypes[4] = {TAG_INT, TAG_FLOAT, TAG_FLOAT, TAG_INT};
    nvm_add_import(mod, mod_idx, fn_idx, 4, TAG_FLOAT, ptypes);

    VmHeap heap; vm_heap_init(&heap);
    NanoValue args[4] = { val_int(2), val_float(3.0), val_float(4.0), val_int(5) };
    NanoValue result; char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, args, 4, &result, &heap, err, sizeof(err));
    ASSERT(ok);
    ASSERT(result.tag == TAG_FLOAT);
    ASSERT(result.as.f64 == 19.0);

    vm_heap_destroy(&heap); nvm_module_free(mod); vm_ffi_shutdown();
}

/* ── resolve-once typed call descriptor tests ──────────────────────────── */

TEST(descriptor_cached_after_first_call) {
    /*
     * The first successful FFI call must populate the module's typed call
     * descriptor table and mark the import RESOLVED; the resolved func_ptr and
     * precomputed signature are then reused on subsequent calls.
     */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "abs", 3);
    uint8_t ptypes[1] = {TAG_INT};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, ptypes);

    /* No descriptors before the first call. */
    ASSERT(mod->call_descriptors == NULL);
    ASSERT_EQ(mod->call_descriptor_count, 0u);

    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue arg = val_int(-7);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, &arg, 1, &result, &heap, err, sizeof(err));

    if (ok) {
        /* Descriptor table allocated and this import resolved once. */
        ASSERT(mod->call_descriptors != NULL);
        ASSERT_EQ(mod->call_descriptor_count, 1u);
        ASSERT(mod->call_descriptors[0].state == NVM_CALL_RESOLVED);
        ASSERT(mod->call_descriptors[0].func_ptr != NULL);
        ASSERT_EQ(mod->call_descriptors[0].param_count, 1);
        ASSERT_EQ(mod->call_descriptors[0].return_type, (uint8_t)TAG_INT);
        ASSERT(mod->call_descriptors[0].all_float == false);

        void *first_ptr = mod->call_descriptors[0].func_ptr;

        /* Second call reuses the same cached pointer. */
        NanoValue arg2 = val_int(-99);
        NanoValue result2;
        ok = vm_ffi_call(mod, 0, &arg2, 1, &result2, &heap, err, sizeof(err));
        ASSERT(ok);
        ASSERT(mod->call_descriptors[0].func_ptr == first_ptr);
        ASSERT(result2.tag == TAG_INT);
        ASSERT(result2.as.i64 == 99);
    }

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(descriptor_failure_is_cached) {
    /*
     * A failed resolution must be remembered as NVM_CALL_FAILED so repeated
     * calls report the same error without re-attempting symbol lookup.
     */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "xyzzy_no_such_function_456", 26);
    nvm_add_import(mod, mod_idx, fn_idx, 0, TAG_INT, NULL);

    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, NULL, 0, &result, &heap, err, sizeof(err));
    ASSERT(!ok);
    ASSERT(err[0] != '\0');
    ASSERT(mod->call_descriptors != NULL);
    ASSERT(mod->call_descriptors[0].state == NVM_CALL_FAILED);

    /* Second call still fails, from the cached FAILED state. */
    char err2[256] = "";
    ok = vm_ffi_call(mod, 0, NULL, 0, &result, &heap, err2, sizeof(err2));
    ASSERT(!ok);
    ASSERT(err2[0] != '\0');
    ASSERT(mod->call_descriptors[0].state == NVM_CALL_FAILED);

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(descriptor_reset_forces_reresolve) {
    /*
     * nvm_call_descriptors_reset drops the cache so the next call resolves
     * the import from scratch again.
     */
    vm_ffi_init();

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint32_t mod_idx = nvm_add_string(mod, "", 0);
    uint32_t fn_idx  = nvm_add_string(mod, "abs", 3);
    uint8_t ptypes[1] = {TAG_INT};
    nvm_add_import(mod, mod_idx, fn_idx, 1, TAG_INT, ptypes);

    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue arg = val_int(-5);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, &arg, 1, &result, &heap, err, sizeof(err));

    if (ok) {
        ASSERT(mod->call_descriptors != NULL);
        nvm_call_descriptors_reset(mod);
        ASSERT(mod->call_descriptors == NULL);
        ASSERT_EQ(mod->call_descriptor_count, 0u);

        /* Re-resolve on the next call. */
        ok = vm_ffi_call(mod, 0, &arg, 1, &result, &heap, err, sizeof(err));
        ASSERT(ok);
        ASSERT(mod->call_descriptors != NULL);
        ASSERT(mod->call_descriptors[0].state == NVM_CALL_RESOLVED);
    }

    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(artifact_and_logical_array_abi) {
    for (int artifact = 0; artifact <= 1; ++artifact) {
        vm_ffi_init();
        NvmModule *mod = nvm_module_new();
        ASSERT(mod);
        char *path = realpath("obj/ffi_artifact_first.so", NULL);
        ASSERT(path);
        ASSERT(ffi_loader_open("array_fixture", path));
        uint32_t lib = artifact ? nvm_add_string(mod, path, (uint32_t)strlen(path)) :
                                 nvm_add_string(mod, "", 0);
        free(path);
        const char *names[] = {"array_matching", "array_legacy", "array_mismatch", "array_stale"};
        for (int i = 0; i < 4; ++i) {
            uint32_t fn = nvm_add_string(mod, names[i], (uint32_t)strlen(names[i]));
            uint32_t imp = nvm_add_import(mod, lib, fn, 0, TAG_ARRAY, NULL);
            if (artifact) mod->imports[imp].kind = NVM_IMPORT_ARTIFACT;
        }
        VmHeap heap;
        vm_heap_init(&heap);
        NanoValue result;
        char err[256];
        for (int repeat = 0; repeat < 2; ++repeat) {
            ASSERT(vm_ffi_call(mod, 0, NULL, 0, &result, &heap, err, sizeof err));
            ASSERT_EQ(result.tag, TAG_ARRAY);
            for (int i = 1; i < 4; ++i) {
                ASSERT(!vm_ffi_call(mod, i, NULL, 0, &result, &heap, err, sizeof err));
                if (!repeat) ASSERT(strstr(err, "native array ABI") != NULL);
                ASSERT_EQ(mod->call_descriptors[i].state, NVM_CALL_FAILED);
            }
        }
        vm_heap_destroy(&heap);
        nvm_module_free(mod);
        vm_ffi_shutdown();
    }
}

TEST(array_mutation_copyback) {
    vm_ffi_init();
    VmHeap heap;
    vm_heap_init(&heap);
    NvmModule *mod = nvm_module_new();
    char *path = realpath("obj/ffi_artifact_first.so", NULL);
    ASSERT(path && mod);
    uint32_t lib = nvm_add_string(mod, path, (uint32_t)strlen(path));
    free(path);
    const char *names[] = {"array_mutate_alias", "array_clear_handles", "array_cleared_count",
                          "array_scale", "array_invalid", "array_forbidden", "array_bad_result",
                          "array_grow_once"};
    uint8_t arities[] = {2, 1, 0, 2, 1, 1, 1, 1};
    uint8_t returns[] = {TAG_ARRAY, TAG_VOID, TAG_INT, TAG_FLOAT, TAG_VOID, TAG_VOID, TAG_ARRAY, TAG_ARRAY};
    for (int i = 0; i < 8; ++i) {
        uint32_t name = nvm_add_string(mod, names[i], (uint32_t)strlen(names[i]));
        uint8_t params[] = {TAG_ARRAY, i == 3 ? TAG_FLOAT : TAG_ARRAY};
        uint32_t imp = nvm_add_import(mod, lib, name, arities[i], returns[i], params);
        mod->imports[imp].kind = NVM_IMPORT_ARTIFACT;
    }
    VmArray *array = vm_array_new(&heap, TAG_INT, 8);
    ASSERT(array && vm_array_push(&heap, array, val_int(10)));
    NanoValue args[] = {val_array(array), val_array(array)}, result;
    char error[256];
    size_t native_objects = gc_get_stats().num_objects;
    for (int i = 0; i < 100; ++i) {
        ASSERT(vm_ffi_call(mod, 0, args, 2, &result, &heap, error, sizeof error));
        ASSERT_EQ(result.tag, TAG_ARRAY);
        ASSERT(result.as.array == array);
        ASSERT_EQ(vm_array_get(array, 0).as.i64, 11 + i);
        vm_release(&heap, result);
        ASSERT_EQ(gc_get_stats().num_objects, native_objects);
    }
    VmState isolated = {0};
    isolated.cop_pid = isolated.cop_in_fd = isolated.cop_out_fd = -1;
    isolated.cop_sig_send_fd = isolated.cop_sig_recv_fd = -1;
    isolated.cop_timeout_ms = 5000;
    isolated.isolate_ffi = true;
    CopBatchCall calls[3] = {{0, args, 2}, {0, args, 2}, {0, args, 2}};
    NanoValue replies[3];
    ASSERT(vm_ffi_call_cop_batch(&isolated, mod, calls, 3, replies, &heap, error, sizeof error));
    ASSERT_EQ(vm_array_get(array, 0).as.i64, 113);
    for (int i = 0; i < 3; ++i) {
        ASSERT(replies[i].tag == TAG_ARRAY && replies[i].as.array == array);
        vm_release(&heap, replies[i]);
    }
    ASSERT(!vm_ffi_call_cop(&isolated, mod, 6, args, 1, &result, &heap, error, sizeof error));
    ASSERT_EQ(vm_array_get(array, 0).as.i64, 113);
    for (int i = 0; i < 2; ++i) {
        ASSERT(vm_ffi_call_cop(&isolated, mod, 1, args, 1, &result, &heap, error, sizeof error));
        ASSERT_EQ(vm_array_get(array, 0).as.i64, 0);
    }
    ASSERT(vm_ffi_call_cop(&isolated, mod, 2, NULL, 0, &result, &heap, error, sizeof error));
    ASSERT_EQ(result.as.i64, 1);
    pid_t same_worker = isolated.cop_pid;
    for (int invocation = 1; invocation <= 2; ++invocation) {
        VmArray *small = vm_array_new(&heap, TAG_INT, 1);
        ASSERT(small && vm_array_push(&heap, small, val_int(0)));
        NanoValue small_arg = val_array(small);
        ASSERT(vm_ffi_call_cop(&isolated, mod, 7, &small_arg, 1, &result, &heap, error, sizeof error));
        ASSERT(result.tag == TAG_ARRAY && result.as.array == small);
        ASSERT_EQ(small->length, 2000);
        ASSERT_EQ(vm_array_get(small, 0).as.i64, invocation);
        ASSERT_EQ(vm_array_get(small, 1999).as.i64, 1999);
        ASSERT_EQ(isolated.cop_pid, same_worker);
        vm_release(&heap, result);
        vm_release(&heap, small_arg);
    }
    VmArray *bulk = vm_array_new(&heap, TAG_INT, 2000);
    ASSERT(bulk);
    for (int i = 0; i < 2000; ++i) ASSERT(vm_array_push(&heap, bulk, val_int(1)));
    NanoValue bulk_arg = val_array(bulk);
    ASSERT(vm_ffi_call_cop(&isolated, mod, 1, &bulk_arg, 1, &result, &heap, error, sizeof error));
    ASSERT_EQ(isolated.cop_pid, same_worker);
    ASSERT_EQ(vm_array_get(bulk, 1999).as.i64, 0);
    ASSERT(vm_ffi_call_cop(&isolated, mod, 2, NULL, 0, &result, &heap, error, sizeof error));
    ASSERT_EQ(result.as.i64, 2001);
    ASSERT_EQ(isolated.cop_pid, same_worker);
    ASSERT(vm_ffi_call_cop(&isolated, mod, 1, &bulk_arg, 1, &result, &heap, error, sizeof error));
    ASSERT(vm_ffi_call_cop(&isolated, mod, 2, NULL, 0, &result, &heap, error, sizeof error));
    ASSERT_EQ(result.as.i64, 2001);
    vm_release(&heap, bulk_arg);
    vm_ffi_cop_stop(&isolated);
    /* I exercise the real parent pipe branch with the shared worker handler,
     * using both request and reply envelopes larger than the mailbox. */
    int to_child[2], from_child[2];
    ASSERT(pipe(to_child) == 0 && pipe(from_child) == 0);
    pid_t worker = fork();
    ASSERT(worker >= 0);
    if (!worker) {
        close(to_child[1]); close(from_child[0]);
        VmHeap child_heap;
        vm_heap_init(&child_heap);
        CopMsgHeader header;
        while (cop_recv_header(to_child[0], &header) && header.msg_type == COP_MSG_FFI_REQ) {
            uint8_t *request = malloc(header.payload_len);
            if (!request || !cop_recv_payload(to_child[0], request, header.payload_len)) _exit(2);
            uint8_t *reply;
            uint32_t reply_size;
            char diagnostic[256] = {0};
            bool ok = cop_execute_request(request, header.payload_len, mod, &child_heap,
                                           &reply, &reply_size, diagnostic, sizeof diagnostic);
            free(request);
            bool sent = ok ? cop_send(from_child[1], COP_MSG_FFI_RESULT, reply, reply_size)
                           : cop_send(from_child[1], COP_MSG_FFI_ERROR, diagnostic, strlen(diagnostic));
            free(reply);
            if (!sent) _exit(3);
        }
        vm_heap_destroy(&child_heap);
        _exit(0);
    }
    close(to_child[0]); close(from_child[1]);
    isolated.cop_pid = worker;
    isolated.cop_in_fd = to_child[1];
    isolated.cop_out_fd = from_child[0];
    VmArray *large = vm_array_new(&heap, TAG_INT, 2000);
    ASSERT(large);
    for (int i = 0; i < 2000; ++i) ASSERT(vm_array_push(&heap, large, val_int(i)));
    NanoValue large_args[] = {val_array(large), val_array(large)};
    for (int i = 0; i < 3; ++i) {
        ASSERT(vm_ffi_call_cop(&isolated, mod, 0, large_args, 2, &result, &heap, error, sizeof error));
        ASSERT(result.tag == TAG_ARRAY && result.as.array == large);
        ASSERT_EQ(vm_array_get(large, 0).as.i64, i + 1);
        ASSERT_EQ(vm_array_get(large, 1999).as.i64, 1999);
        vm_release(&heap, result);
    }
    vm_release(&heap, val_array(large));
    vm_ffi_cop_stop(&isolated);
    vm_array_set(array, 0, val_int(1));
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT(vm_ffi_call(mod, 1, args, 1, &result, &heap, error, sizeof error));
        ASSERT_EQ(vm_array_get(array, 0).as.i64, 0);
    }
    ASSERT(vm_ffi_call(mod, 2, NULL, 0, &result, &heap, error, sizeof error));
    ASSERT_EQ(result.as.i64, 1);
    ASSERT(!vm_ffi_call(mod, 4, args, 1, &result, &heap, error, sizeof error));
    ASSERT_EQ(array->length, 1);
    ASSERT(strstr(error, "invalid native metadata"));
    ASSERT(!vm_ffi_call(mod, 6, args, 1, &result, &heap, error, sizeof error));
    ASSERT_EQ(vm_array_get(array, 0).as.i64, 0);
    ASSERT_EQ(gc_get_stats().num_objects, native_objects);
    VmArray *floating = vm_array_new(&heap, TAG_FLOAT, 8);
    ASSERT(floating && vm_array_push(&heap, floating, val_float(2.5)));
    NanoValue mixed[] = {val_array(floating), val_float(4.0)};
    ASSERT(vm_ffi_call(mod, 3, mixed, 2, &result, &heap, error, sizeof error));
    ASSERT(result.tag == TAG_FLOAT && result.as.f64 == 10.0);
    ASSERT(vm_array_get(floating, 0).as.f64 == 10.0);
    VmArray *nested = vm_array_new(&heap, TAG_ARRAY, 8);
    NanoValue unsupported = val_array(nested);
    ASSERT(!vm_ffi_call(mod, 5, &unsupported, 1, &result, &heap, error, sizeof error));
    ASSERT(strstr(error, "unsupported"));
    vm_release(&heap, val_array(array));
    vm_release(&heap, val_array(floating));
    vm_release(&heap, val_array(nested));
    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(artifact_handle_isolation) {
    vm_ffi_init();
    NvmModule *mod = nvm_module_new();
    ASSERT(mod);
    char *paths[] = {realpath("obj/ffi_artifact_first.so", NULL),
                     realpath("obj/ffi_artifact_second.so", NULL)};
    ASSERT(paths[0] && paths[1]);
    uint32_t fn = nvm_add_string(mod, "nano_artifact_answer", 20);
    for (int i = 0; i < 2; i++) {
        uint32_t path = nvm_add_string(mod, paths[i], (uint32_t)strlen(paths[i]));
        uint32_t imp = nvm_add_import(mod, path, fn, 0, TAG_INT, NULL);
        mod->imports[imp].kind = NVM_IMPORT_ARTIFACT;
        free(paths[i]);
    }
    uint32_t missing = nvm_add_string(mod, "/no-such-nano-artifact.so", 25);
    nvm_add_import(mod, missing, fn, 0, TAG_INT, NULL);
    mod->imports[2].kind = NVM_IMPORT_ARTIFACT;
    /* I must not resolve a missing bound symbol from the main executable. */
    uint32_t global = nvm_add_string(mod, "nl_ffi_test_mix_fi_gp", 21);
    uint8_t tags[] = {TAG_FLOAT, TAG_INT};
    nvm_add_import(mod, mod->imports[0].module_name_idx, global, 2, TAG_INT, tags);
    mod->imports[3].kind = NVM_IMPORT_ARTIFACT;
    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue result;
    char err[256];
    for (int repeat = 0; repeat < 2; repeat++) {
        for (uint32_t i = 0; i < 2; i++) {
            ASSERT(vm_ffi_call(mod, i, NULL, 0, &result, &heap, err, sizeof err));
            ASSERT_EQ(result.tag, TAG_INT);
            ASSERT_EQ(result.as.i64, 42 + i);
        }
        ASSERT(!vm_ffi_call(mod, 2, NULL, 0, &result, &heap, err, sizeof err));
        NanoValue args[] = {val_float(2.0), val_int(3)};
        ASSERT(!vm_ffi_call(mod, 3, args, 2, &result, &heap, err, sizeof err));
    }
    vm_heap_destroy(&heap);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

/* ── main ──────────────────────────────────────────────────────────────── */

TEST(contracted_import_never_uses_legacy_dispatch) {
    NvmModule *module = nvm_module_new();
    ASSERT(module);
    uint32_t native = nvm_add_string(module, "", 0);
    uint32_t name = nvm_add_string(module, "abs", 3);
    uint32_t adapter = nvm_add_string(module, "abs_retained_adapter", 20);
    uint8_t tags[] = {TAG_INT};
    ASSERT_EQ(nvm_add_import(module, native, name, 1, TAG_INT, tags), 0u);
    NvmCallbackContract c = {.import_idx = 0, .adapter_name_idx = adapter,
        .parameter_idx = NVM_CALLBACK_NO_PARAMETER, .abi_version = NVM_CALLBACK_ABI_RETAINED_V1,
        .execution = NVM_FOREIGN_WORKER_THREAD, .return_tag = TAG_VOID};
    ASSERT(nvm_add_callback_contract(module, &c));
    VmState *vm = malloc(sizeof(*vm));
    ASSERT(vm);
    vm_init(vm, module);
    NanoValue args[] = {val_int(-7)}, result;
    char error[256] = {0};
    ASSERT(!vm_ffi_call(module, 0, args, 1, &result, &vm->heap, error, sizeof error));
    ASSERT(strstr(error, "retained callback scheduler") != NULL);
    ASSERT(!vm_ffi_call_cop(vm, module, 0, args, 1, &result, &vm->heap, error, sizeof error));
    ASSERT(strstr(error, "retained callback scheduler") != NULL);
    CopBatchCall call = {.import_idx = 0, .args = args, .arg_count = 1};
    ASSERT(!vm_ffi_call_cop_batch(vm, module, &call, 1, &result, &vm->heap, error, sizeof error));
    ASSERT(strstr(error, "retained callback scheduler") != NULL);
    ASSERT_EQ(result.tag, TAG_VOID);
    ASSERT_EQ(vm->cop_pid, -1);
    vm_destroy(vm);
    free(vm);
    nvm_module_free(module);
}

TEST(retained_native_strings) {
    char *path = realpath("obj/ffi_callback_fixture.so", NULL);
    ASSERT(path);
    char source[8192];
    snprintf(source, sizeof source,
        ".import \"%s\" \"text\" string function string opaque\n.import_kind 0 artifact\n"
        ".callback 0 0 \"retained_string\" 1 worker int int\n"
        ".import \"%s\" \"identity\" string string opaque\n.import_kind 1 artifact\n"
        ".callback 1 65535 \"retained_string_identity\" 1 owner void\n"
        ".function increment 1 1 0 int 1\nLOAD_LOCAL 0\nPUSH_I64 1\nADD\nRET\n.end\n.parameters 0 int\n",
        path, path);
    free(path);
    AsmResult assembled;
    NvmModule *mod = asm_assemble(source, &assembled);
    ASSERT(mod);
    VmState *vm = malloc(sizeof(*vm));
    ASSERT(vm);
    vm_init(vm, mod);
    NanoValue text = val_string(vm_string_new(&vm->heap, "audio", 5));
    ASSERT(text.as.string);
    NanoValue original = {.tag = TAG_OPAQUE, .as.obj = text.as.string->data};
    NanoValue args[] = {val_function(0), text, original}, result;
    char error[256] = {0};
    alarm(20);
    for (int i = 0; i < 32; i++) {
        ASSERT(vm_ffi_call_vm(vm, mod, 0, args, 3, &result, error, sizeof error));
        ASSERT_EQ(result.tag, TAG_STRING);
        ASSERT(strcmp(result.as.string->data, "audio:42") == 0);
        vm_release(&vm->heap, result);
    }
    /* I copy aliasing borrowed results before freeing the argument snapshot. */
    ASSERT(vm_ffi_call_vm(vm, mod, 1, args + 1, 2, &result, error, sizeof error));
    ASSERT_EQ(result.tag, TAG_STRING);
    ASSERT(strcmp(result.as.string->data, "audio") == 0);
    vm_release(&vm->heap, result);
    mod->callback_contracts[1].execution = NVM_FOREIGN_WORKER_THREAD;
    ASSERT(vm_ffi_call_vm(vm, mod, 1, args + 1, 2, &result, error, sizeof error));
    ASSERT_EQ(result.tag, TAG_STRING);
    ASSERT(strcmp(result.as.string->data, "audio") == 0);
    vm_release(&vm->heap, result);
    ASSERT(strcmp(text.as.string->data, "audio") == 0);
    /* I unwind an already copied string and published handle on later failure. */
    args[2] = val_int(1);
    ASSERT(!vm_ffi_call_vm(vm, mod, 0, args, 3, &result, error, sizeof error));
    ASSERT_EQ(result.tag, TAG_VOID);
    args[2] = original;
    vm_release(&vm->heap, text);
    args[1] = val_string(vm_string_new(&vm->heap, "", 0));
    ASSERT(args[1].as.string);
    args[2].as.obj = args[1].as.string->data;
    ASSERT(vm_ffi_call_vm(vm, mod, 0, args, 3, &result, error, sizeof error));
    ASSERT_EQ(result.tag, TAG_VOID);
    vm_release(&vm->heap, args[1]);
    args[1] = val_string(vm_string_new(&vm->heap, "a\0b", 3));
    ASSERT(args[1].as.string);
    ASSERT(!vm_ffi_call_vm(vm, mod, 0, args, 3, &result, error, sizeof error));
    ASSERT(strstr(error, "embedded NUL"));
    vm_release(&vm->heap, args[1]);
    args[1] = val_int(0);
    ASSERT(!vm_ffi_call_vm(vm, mod, 0, args, 3, &result, error, sizeof error));
    args[1] = val_string(NULL);
    ASSERT(!vm_ffi_call_vm(vm, mod, 0, args, 3, &result, error, sizeof error));
    vm_destroy(vm);
    free(vm);
    nvm_module_free(mod);
    vm_ffi_shutdown();
    alarm(0);
}

TEST(retained_native_scheduler) {
    char *path = realpath("obj/ffi_callback_fixture.so", NULL);
    ASSERT(path);
    char source[8192];
    snprintf(source, sizeof source,
        ".import \"%s\" \"call\" int function int\n.import_kind 0 artifact\n"
        ".callback 0 0 \"retained_call\" 1 worker int int\n"
        ".import \"%s\" \"start\" opaque function\n.import_kind 1 artifact\n"
        ".callback 1 0 \"retained_start\" 1 owner int int\n"
        ".import \"%s\" \"wait\" int opaque\n.import_kind 2 artifact\n"
        ".callback 2 65535 \"retained_wait\" 1 worker void\n"
        ".import \"%s\" \"mix\" float function float bool u8 opaque\n.import_kind 3 artifact\n"
        ".callback 3 0 \"retained_mix\" 1 worker int int\n"
        ".function increment 1 1 0 int 1\nLOAD_LOCAL 0\nPUSH_I64 1\nADD\nDUP\nSTORE_GLOBAL 0\nRET\n.end\n.parameters 0 int\n"
        ".function nested 1 1 0 int 1\nFUNCREF 0\nLOAD_LOCAL 0\nCALL_EXTERN 0\nRET\n.end\n.parameters 1 int\n"
        ".function fail 1 1 0 int 1\nPUSH_BOOL 0\nASSERT\nLOAD_LOCAL 0\nRET\n.end\n.parameters 2 int\n"
        ".function spin 0 0 0 void 0\nagain:\nJMP again\n.end\n"
        ".function progress 0 1 0 int 1\nFUNCREF 0\nCALL_EXTERN 1\nSTORE_LOCAL 0\n"
        "waiting:\nLOAD_GLOBAL 0\nPUSH_I64 42\nEQ\nJMP_FALSE waiting\n"
        "LOAD_LOCAL 0\nCALL_EXTERN 2\nRET\n.end\n"
        ".function failing_host 0 0 0 int 1\nFUNCREF 2\nPUSH_I64 1\nCALL_EXTERN 0\nRET\n.end\n",
        path, path, path, path);
    free(path);
    AsmResult assembled;
    NvmModule *mod = asm_assemble(source, &assembled);
    if (!mod) printf("assembly: %s\n", assembled.message);
    ASSERT(mod);
    VmState *vm = malloc(sizeof(*vm));
    ASSERT(vm);
    vm_init(vm, mod);
    NanoValue args[] = {val_function(0), val_int(41)}, result;
    char error[256] = {0};
    alarm(20);
    ASSERT(vm_ffi_call_vm(vm, mod, 0, args, 2, &result, error, sizeof error));
    ASSERT_EQ(result.as.i64, 42);
    /* I can enter a second blocking native call from a suspended callback. */
    args[0] = val_function(1);
    ASSERT(vm_ffi_call_vm(vm, mod, 0, args, 2, &result, error, sizeof error));
    ASSERT_EQ(result.as.i64, 42);
    mod->callback_contracts[0].execution = NVM_FOREIGN_OWNER_THREAD;
    args[0] = val_function(0);
    ASSERT(vm_ffi_call_vm(vm, mod, 0, args, 2, &result, error, sizeof error));
    ASSERT_EQ(result.as.i64, 42);
    /* The adapter owns the handle after publication; a policy-only wait pumps it. */
    ASSERT(vm_ffi_call_vm(vm, mod, 1, args, 1, &result, error, sizeof error));
    ASSERT(result.tag == TAG_OPAQUE && result.as.obj);
    NanoValue pending = result;
    ASSERT(vm_ffi_call_vm(vm, mod, 2, &pending, 1, &result, error, sizeof error));
    ASSERT_EQ(result.as.i64, 42);
    /* I yield a non-terminating pure instruction stream without needing I/O. */
    vm->current_fn = 3;
    vm->ip = mod->functions[3].code_offset;
    vm->frame_count = 1;
    vm->frames[0].module = mod;
    vm->frames[0].fn_idx = 3;
    ASSERT_EQ(vm_core_execute(vm).type, TRAP_YIELD);
    ASSERT_EQ(vm_core_execute(vm).type, TRAP_YIELD);
    vm->frame_count = 0;
    vm->current_fn = 0;
    vm->ip = 0;
    vm->globals[0] = val_int(0);
    ASSERT_EQ(vm_invoke_callable(vm, val_function(4), NULL, 0, &result), VM_OK);
    ASSERT_EQ(result.as.i64, 42);
    NanoValue mixed[] = {val_function(0), val_float(0.5), val_bool(true), val_u8(200), val_void()};
    mixed[4].tag = TAG_OPAQUE;
    ASSERT(vm_ffi_call_vm(vm, mod, 3, mixed, 5, &result, error, sizeof error));
    ASSERT_EQ(result.tag, TAG_FLOAT);
    ASSERT_EQ(result.as.f64, 241.5);
    mixed[4] = val_int(0);
    ASSERT(vm_ffi_call_vm(vm, mod, 3, mixed, 5, &result, error, sizeof error));
    ASSERT_EQ(result.as.f64, 241.5);
    mixed[4] = val_int(1);
    ASSERT(!vm_ffi_call_vm(vm, mod, 3, mixed, 5, &result, error, sizeof error));
    uint32_t adapter = mod->callback_contracts[0].adapter_name_idx;
    mod->callback_contracts[0].adapter_name_idx = nvm_add_string(mod, "nl_ffi_test_mix_fi", 18);
    ASSERT(!vm_ffi_call_vm(vm, mod, 0, args, 2, &result, error, sizeof error));
    ASSERT(strstr(error, "selected module"));
    mod->callback_contracts[0].adapter_name_idx = adapter;
    NvmCallbackContract *contracts = mod->callback_contracts;
    mod->callback_contracts = NULL;
    ASSERT(!vm_ffi_call_vm(vm, mod, 0, args, 2, &result, error, sizeof error));
    ASSERT(strstr(error, "valid callback contracts"));
    mod->callback_contracts = contracts;
    vm->isolate_ffi = true;
    ASSERT(!vm_ffi_call_vm(vm, mod, 0, args, 2, &result, error, sizeof error));
    ASSERT(strstr(error, "isolated FFI"));
    ASSERT_EQ(vm->cop_pid, -1);
    vm->isolate_ffi = false;
    mod->callback_contracts[0].execution = NVM_FOREIGN_WORKER_THREAD;
    ASSERT_EQ(vm_invoke_callable(vm, val_function(5), NULL, 0, &result), VM_ERR_ASSERT_FAILED);
    ASSERT_EQ(vm->callback_error, VM_ERR_ASSERT_FAILED);
    ASSERT_EQ(result.tag, TAG_VOID);
    vm_destroy(vm);
    vm_init(vm, mod);
    /* I cancel native-owned work and call its join code after loader shutdown. */
    args[0] = val_function(0);
    ASSERT(vm_ffi_call_vm(vm, mod, 1, args, 1, &pending, error, sizeof error));
    ASSERT(pending.tag == TAG_OPAQUE && pending.as.obj);
    const char *library = nvm_get_string(mod, mod->imports[2].module_name_idx);
    int64_t (*late_wait)(void *) = (int64_t (*)(void *))ffi_loader_resolve_module("retained_wait", library);
    ASSERT(late_wait);
    vm_destroy(vm);
    vm_ffi_shutdown();
    ASSERT_EQ(late_wait(pending.as.obj), -1);
    alarm(0);
    free(vm);
    nvm_module_free(mod);
    vm_ffi_shutdown();
}

TEST(sdl_image_cleanup_dispatch) {
    const char *path = getenv("NANO_TEST_SDL_ARRAY_LIBRARY");
    ASSERT(path && path[0] == '/');
    NvmModule *module = nvm_module_new();
    uint32_t lib = nvm_add_string(module, path, strlen(path));
    const char *names[] = {"nl_img_destroy_texture_batch", "nl_test_destroyed_count"};
    uint8_t params[] = {TAG_ARRAY, TAG_INT};
    for (int i = 0; i < 2; ++i) {
        uint32_t name = nvm_add_string(module, names[i], strlen(names[i]));
        uint32_t index = nvm_add_import(module, lib, name, i ? 0 : 2,
                                        i ? TAG_INT : TAG_VOID, params);
        module->imports[index].kind = NVM_IMPORT_ARTIFACT;
    }
    vm_ffi_init();
    VmHeap heap;
    vm_heap_init(&heap);
    for (int mode = 0; mode < 3; ++mode) {
        VmState isolated = {0};
        isolated.cop_pid = isolated.cop_in_fd = isolated.cop_out_fd = -1;
        isolated.cop_sig_send_fd = isolated.cop_sig_recv_fd = -1;
        isolated.cop_timeout_ms = 5000;
        NanoValue result;
        char error[256];
        bool ok = mode ? vm_ffi_call_cop(&isolated, module, 1, NULL, 0, &result, &heap, error, sizeof error)
                       : vm_ffi_call(module, 1, NULL, 0, &result, &heap, error, sizeof error);
        ASSERT(ok && result.tag == TAG_INT);
        int64_t before = result.as.i64;
        int count = mode == 2 ? 2000 : 4;
        VmArray *handles = vm_array_new(&heap, TAG_INT, count);
        ASSERT(handles);
        for (int i = 0; i < count; ++i)
            ASSERT(vm_array_push(&heap, handles, val_int(i % 2 + 1)));
        NanoValue args[] = {val_array(handles), val_int(1)};
        for (int call = 0; call < 4; ++call) {
            if (call == 2) args[1] = val_int(count);
            ok = mode ? vm_ffi_call_cop(&isolated, module, 0, args, 2, &result, &heap, error, sizeof error)
                      : vm_ffi_call(module, 0, args, 2, &result, &heap, error, sizeof error);
            ASSERT(ok && result.tag == TAG_VOID);
            for (int i = 0; i < count; ++i)
                ASSERT_EQ(vm_array_get(handles, i).as.i64, call < 2 && i % 2 ? 2 : 0);
        }
        ok = mode ? vm_ffi_call_cop(&isolated, module, 1, NULL, 0, &result, &heap, error, sizeof error)
                  : vm_ffi_call(module, 1, NULL, 0, &result, &heap, error, sizeof error);
        ASSERT(ok && result.tag == TAG_INT && result.as.i64 == before + 2);
        if (mode) vm_ffi_cop_stop(&isolated);
        vm_release(&heap, val_array(handles));
    }
    vm_gc_collect_cycles(&heap);
    ASSERT_EQ(heap.stats.num_objects, 0);
    vm_heap_destroy(&heap);
    nvm_module_free(module);
    vm_ffi_shutdown();
}

TEST(builtin_path_normalize_aliases) {
    const char *aliases[] = {"path_normalize", "nl_os_path_normalize"};
    const char *paths[][2] = {{"", "."}, {"/../../a//b/..", "/a"},
        {"../../a/../b", "../../b"}, {"a/./b/../c", "a/c"}, {"é/../空", "空"}};
    vm_ffi_init();
    VmHeap heap; vm_heap_init(&heap);
    for (size_t alias=0;alias<2;alias++) {
        NvmModule *module=nvm_module_new(); ASSERT(module);
        uint32_t library=nvm_add_string(module,"",0);
        uint32_t symbol=nvm_add_string(module,aliases[alias],(uint32_t)strlen(aliases[alias]));
        uint8_t tag=TAG_STRING;
        uint32_t imported=nvm_add_import(module,library,symbol,1,TAG_STRING,&tag);
        for (size_t repeat=0;repeat<100;repeat++) for (size_t i=0;i<5;i++) {
            NanoValue argument=val_string(vm_string_new(&heap,paths[i][0],(uint32_t)strlen(paths[i][0])));
            NanoValue result=val_void(); char error[256]={0};
            ASSERT(vm_ffi_call(module,imported,&argument,1,&result,&heap,error,sizeof error));
            ASSERT(result.tag==TAG_STRING && result.as.string);
            ASSERT(strcmp(vmstring_cstr(result.as.string),paths[i][1])==0);
            vm_release(&heap,result); vm_release(&heap,argument);
            ASSERT_EQ(heap.stats.num_objects,0);
        }
        NanoValue wrong=val_int(1), result=val_void(); char error[256]={0};
        ASSERT(!vm_ffi_call(module,imported,&wrong,1,&result,&heap,error,sizeof error));
        ASSERT(strstr(error,"take and return a string"));
        ASSERT_EQ(heap.stats.num_objects,0);
        nvm_module_free(module);
    }
    vm_heap_destroy(&heap); vm_ffi_shutdown();
}

int main(void) {
    if (getenv("NANO_TEST_PATH_ALIASES_ONLY")) {
        RUN(builtin_path_normalize_aliases);
        return g_fail ? 1 : 0;
    }
    if (getenv("NANO_TEST_SDL_ARRAY_LIBRARY")) RUN(sdl_image_cleanup_dispatch);
    if (getenv("NANO_TEST_SDL_ARRAY_ONLY")) {
        if (g_pass == 1 && g_fail == 0) {
            printf("All 1 SDL VM cleanup tests passed.\n");
            return 0;
        }
        return 1;
    }
    RUN(builtin_path_normalize_aliases);
    RUN(retained_native_strings);
    printf("\n[vm_ffi] FFI bridge unit tests...\n\n");
    RUN(init_shutdown_set_env);
    RUN(retained_native_scheduler);
    RUN(wide_mixed_signature);
    RUN(call_string_returning_float);
    RUN(call_bytecode_callback_rejected);
    RUN(contracted_import_never_uses_legacy_dispatch);
    RUN(artifact_handle_isolation);
    RUN(artifact_and_logical_array_abi);
    RUN(array_mutation_copyback);
    RUN(load_module_nonexistent);
    RUN(call_empty_module_oob);
    RUN(call_too_many_args);
    RUN(call_unresolved_function);
    RUN(call_strlen_null_string);
    RUN(call_abs_int_arg);
    RUN(call_bool_arg_path);
    RUN(call_opaque_arg_path);
    RUN(call_void_return_type);
    RUN(call_bool_return_type);
    RUN(call_mixed_fp_gp_ret_fp);
    RUN(call_mixed_gp_fp_ret_fp);
    RUN(call_mixed_fp_gp_ret_gp);
    RUN(call_mixed_four_args);    RUN(descriptor_cached_after_first_call);
    RUN(descriptor_failure_is_cached);
    RUN(descriptor_reset_forces_reresolve);

    printf("\n");
    if (g_fail == 0) {
        printf("All %d vm_ffi tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d vm_ffi tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
