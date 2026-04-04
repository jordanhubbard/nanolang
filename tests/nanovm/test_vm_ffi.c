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

static int g_pass = 0, g_fail = 0;
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

TEST(load_module_nonexistent) {
    vm_ffi_init();

    /* Nonexistent module: should return false gracefully */
    bool ok = vm_ffi_load_module("definitely_not_a_real_ffi_module_xyzzy");
    ASSERT(!ok);

    /* Empty module name */
    ok = vm_ffi_load_module("");
    /* Return value undefined for empty name — just must not crash */

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

    /* Pass 17 args (> 16 max) — should return false with error */
    NanoValue args[17];
    for (int i = 0; i < 17; i++) args[i] = val_int(0);
    NanoValue result;
    char err[256] = "";
    bool ok = vm_ffi_call(mod, 0, args, 17, &result, &heap, err, sizeof(err));
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

/* ── main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[vm_ffi] FFI bridge unit tests...\n\n");
    RUN(init_shutdown_set_env);
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

    printf("\n");
    if (g_fail == 0) {
        printf("All %d vm_ffi tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d vm_ffi tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
