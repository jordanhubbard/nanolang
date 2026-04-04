/*
 * test_wrapper_gen.c — unit tests for src/nanovirt/wrapper_gen.c
 *
 * Exercises: find_obj_dir (via NANO_VIRT_LIB), wrapper_generate,
 * wrapper_generate_daemon.  Uses controlled environments to exercise
 * both failure and (where possible) success code paths.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#include "../../src/nanovirt/wrapper_gen.h"
#include "../../src/nanoisa/nvm_format.h"

static int g_pass = 0, g_fail = 0;
#define TEST(name) static void test_##name(void)
#define RUN(name)  do { test_##name(); \
    printf("  %-55s PASS\n", #name "..."); g_pass++; } while(0)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
    g_fail++; return; } } while(0)

/* ── find_obj_dir via env var (failure: path doesn't have vm.o) ────────── */

TEST(wrapper_generate_nonexistent_lib_path) {
    /*
     * Set NANO_VIRT_LIB to /tmp (exists, readable) but
     * /tmp/nanovm/vm.o does not exist → wrapper_generate returns false.
     * This covers find_obj_dir env-var path and the vm.o existence check.
     */
    setenv("NANO_VIRT_LIB", "/tmp", 1);

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint8_t blob[4] = {0x4e, 0x56, 0x4d, 0x01};

    bool ok = wrapper_generate(mod, blob, sizeof(blob),
                               "/tmp/test_wrapper_gen_out",
                               "test.nano", NULL, false);
    /* Expected false: /tmp/nanovm/vm.o does not exist */
    ASSERT(!ok);

    nvm_module_free(mod);
    unsetenv("NANO_VIRT_LIB");
}

TEST(wrapper_generate_daemon_nonexistent_lib) {
    /*
     * Same as above but for wrapper_generate_daemon.
     * /tmp/nanovm/vmd_client.o doesn't exist → returns false.
     */
    setenv("NANO_VIRT_LIB", "/tmp", 1);

    uint8_t blob[4] = {0x4e, 0x56, 0x4d, 0x01};
    bool ok = wrapper_generate_daemon(blob, sizeof(blob),
                                      "/tmp/test_daemon_gen_out", false);
    ASSERT(!ok);

    unsetenv("NANO_VIRT_LIB");
}

TEST(wrapper_generate_no_lib_path_no_obj) {
    /*
     * With NANO_VIRT_LIB unset and running from /tmp where obj/ doesn't
     * exist, find_obj_dir falls through all options and returns NULL.
     * wrapper_generate returns false at the very first check.
     */
    unsetenv("NANO_VIRT_LIB");

    /* Change to /tmp so that ./obj doesn't exist */
    char saved_cwd[4096];
    if (getcwd(saved_cwd, sizeof(saved_cwd)) == NULL) {
        printf("  SKIP: getcwd failed\n");
        return;
    }

    if (chdir("/tmp") != 0) {
        printf("  SKIP: chdir /tmp failed\n");
        return;
    }

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    uint8_t blob[4] = {0x4e, 0x56, 0x4d, 0x01};

    bool ok = wrapper_generate(mod, blob, sizeof(blob),
                               "/tmp/test_wrapper_noobjdir",
                               "test.nano", NULL, false);
    /* May be false (no obj/) or true (finds obj via /proc/self/exe) */
    /* Either way must not crash — just exercise the code paths */
    (void)ok;

    nvm_module_free(mod);

    /* Restore CWD */
    int rc = chdir(saved_cwd);
    (void)rc;
}

TEST(wrapper_generate_from_project_root) {
    /*
     * With NANO_VIRT_LIB unset, run from project root where ./obj/ exists.
     * find_obj_dir falls through to CWD fallback and finds ./obj.
     * Then wrapper_generate checks for obj/nanovm/vm.o.
     *
     * If obj/nanovm/vm.o exists (post-build), wrapper_generate proceeds to
     * write a temp C file, build the obj list, find src/, and try to compile.
     *
     * We call it with verbose=false and don't assert on success — the goal
     * is to exercise write_wrapper_c, build_obj_list, and the cc step.
     */
    unsetenv("NANO_VIRT_LIB");

    NvmModule *mod = nvm_module_new();
    ASSERT(mod != NULL);
    /* Use an empty module (no imports, no functions) */
    uint32_t *out_size = NULL;
    uint32_t bsize = 0;
    uint8_t *blob = nvm_serialize(mod, &bsize);
    if (!blob || bsize == 0) {
        /* Serialization failed — skip compilation test */
        nvm_module_free(mod);
        return;
    }

    /* Output to a temp path */
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "/tmp/test_wgen_%d", (int)getpid());

    bool ok = wrapper_generate(mod, blob, bsize, out_path, "test.nano",
                               NULL, false);
    /* ok may be true (compilation succeeded) or false (obj files missing,
     * cc not found, etc.) — both outcomes are valid for this test. */
    (void)ok;

    /* Clean up generated binary if it was created */
    remove(out_path);
    free(blob);
    nvm_module_free(mod);
    (void)out_size;
}

TEST(wrapper_generate_daemon_from_project_root) {
    /*
     * Exercise wrapper_generate_daemon from project root.
     * Checks for vmd_client.o — present only if 'make nano_vmd' was run.
     */
    unsetenv("NANO_VIRT_LIB");

    uint8_t blob[4] = {0x4e, 0x56, 0x4d, 0x01};
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "/tmp/test_wgen_daemon_%d", (int)getpid());

    bool ok = wrapper_generate_daemon(blob, sizeof(blob), out_path, false);
    /* ok may be true or false — just must not crash */
    (void)ok;

    remove(out_path);
}

/* ── main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[wrapper_gen] NanoVirt wrapper generation tests...\n\n");
    RUN(wrapper_generate_nonexistent_lib_path);
    RUN(wrapper_generate_daemon_nonexistent_lib);
    RUN(wrapper_generate_no_lib_path_no_obj);
    RUN(wrapper_generate_from_project_root);
    RUN(wrapper_generate_daemon_from_project_root);

    printf("\n");
    if (g_fail == 0) {
        printf("All %d wrapper_gen tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d wrapper_gen tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
