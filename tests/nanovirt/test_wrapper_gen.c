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
#include <sys/wait.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#include "../../src/nanovirt/wrapper_gen.h"
#include "../../src/nanoisa/nvm_format.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/nvm_v2_convert.h"

static int g_pass = 0, g_fail = 0;
#define TEST(name) static void test_##name(void)
#define RUN(name)  do { int before = g_fail; test_##name(); \
    if (g_fail == before) { printf("  %-55s PASS\n", #name "..."); g_pass++; } } while(0)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
    g_fail++; return; } } while(0)

/* I use current valid bytes so loader refusal cannot replace path/link checks. */
static bool ordinary_blob(NvmModule **module, uint8_t **blob, uint32_t *size) {
    AsmResult result = {0};
    NvmModule *mod = asm_assemble(".function main 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n.entry main\n", &result);
    if (!mod) return false;
    NvmV2Module v = {0};
    size_t length = 0;
    uint8_t *bytes = NULL;
    bool ok = nvm_v2_from_nvm_module(mod, &v) == NVM_V2_OK &&
              nvm_v2_module_serialize(&v, NULL, 0, &length) == NVM_V2_OK &&
              length <= UINT32_MAX;
    if (ok) {
        bytes = malloc(length);
        ok = bytes && nvm_v2_module_serialize(&v, bytes, length, &length) == NVM_V2_OK;
    }
    nvm_v2_module_free(&v);
    if (!ok) { free(bytes); nvm_module_free(mod); return false; }
    *module = mod; *blob = bytes; *size = (uint32_t)length;
    return true;
}

/* ── find_obj_dir via env var (failure: path doesn't have vm.o) ────────── */

TEST(wrapper_generate_nonexistent_lib_path) {
    /*
     * Set NANO_VIRT_LIB to /tmp (exists, readable) but
     * /tmp/nanovm/vm.o does not exist → wrapper_generate returns false.
     * This covers find_obj_dir env-var path and the vm.o existence check.
     */
    setenv("NANO_VIRT_LIB", "/tmp", 1);

    NvmModule *mod = NULL; uint8_t *blob = NULL; uint32_t bsize = 0;
    ASSERT(ordinary_blob(&mod, &blob, &bsize));

    bool ok = wrapper_generate(mod, blob, bsize,
                               "/tmp/test_wrapper_gen_out",
                               "test.nano", NULL, false);
    /* Expected false: /tmp/nanovm/vm.o does not exist */
    ASSERT(!ok);

    free(blob);
    nvm_module_free(mod);
    unsetenv("NANO_VIRT_LIB");
}

TEST(wrapper_generate_daemon_nonexistent_lib) {
    /*
     * Same as above but for wrapper_generate_daemon.
     * /tmp/nanovm/vmd_client.o doesn't exist → returns false.
     */
    setenv("NANO_VIRT_LIB", "/tmp", 1);

    NvmModule *mod = NULL; uint8_t *blob = NULL; uint32_t bsize = 0;
    ASSERT(ordinary_blob(&mod, &blob, &bsize));
    bool ok = wrapper_generate_daemon(blob, bsize,
                                      "/tmp/test_daemon_gen_out", false);
    ASSERT(!ok);
    free(blob);
    nvm_module_free(mod);

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

    NvmModule *mod = NULL; uint8_t *blob = NULL; uint32_t bsize = 0;
    ASSERT(ordinary_blob(&mod, &blob, &bsize));

    bool ok = wrapper_generate(mod, blob, bsize,
                               "/tmp/test_wrapper_noobjdir",
                               "test.nano", NULL, false);
    /* May be false (no obj/) or true (finds obj via /proc/self/exe) */
    /* Either way must not crash — just exercise the code paths */
    (void)ok;

    free(blob);
    nvm_module_free(mod);

    /* Restore CWD */
    int rc = chdir(saved_cwd);
    (void)rc;
}

TEST(wrapper_generate_from_project_root) {
    /* I require a real link from the object files built by this test target. */
    unsetenv("NANO_VIRT_LIB");

    NvmModule *mod = NULL; uint8_t *blob = NULL; uint32_t bsize = 0;
    ASSERT(ordinary_blob(&mod, &blob, &bsize));

    /* Output to a temp path */
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "/tmp/test_wgen_%d", (int)getpid());

    bool ok = wrapper_generate(mod, blob, bsize, out_path, "test.nano",
                               NULL, false);
    /* I require the positive linking gate to produce an executable. */
    ASSERT(ok);
    ASSERT(access(out_path, X_OK) == 0);

    pid_t child = fork();
    ASSERT(child >= 0);
    if (child == 0) { execl(out_path, out_path, (char *)NULL); _exit(127); }
    int status = 0;
    ASSERT(waitpid(child, &status, 0) == child);
    ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 7);

    /* Clean up generated binary if it was created */
    remove(out_path);
    free(blob);
    nvm_module_free(mod);
}

TEST(wrapper_generate_daemon_from_project_root) {
    /*
     * Exercise wrapper_generate_daemon from project root.
     * Checks for vmd_client.o — present only if 'make nano_vmd' was run.
     */
    unsetenv("NANO_VIRT_LIB");

    NvmModule *mod = NULL; uint8_t *blob = NULL; uint32_t bsize = 0;
    ASSERT(ordinary_blob(&mod, &blob, &bsize));
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "/tmp/test_wgen_daemon_%d", (int)getpid());

    bool ok = wrapper_generate_daemon(blob, bsize, out_path, false);
    ASSERT(ok);
    ASSERT(access(out_path, X_OK) == 0);

    remove(out_path);
    free(blob);
    nvm_module_free(mod);
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
