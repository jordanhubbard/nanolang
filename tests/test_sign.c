/*
 * test_sign.c — unit tests for sign.c (Ed25519 WASM module signing)
 *
 * Tests the sign, verify, and error paths of sign.c.
 */

#include "../src/nanolang.h"
#include "../src/sign.h"
#include "../src/wasm_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

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

/* Generate a simple WASM file and write it to path. Returns 0 on success. */
static int write_wasm_file(const char *path) {
    ASTNode *prog = parse_nano(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
    );
    if (!prog) return -1;
    suppress_stderr();
    int rc = wasm_backend_emit(prog, path, "<test>", NULL, false);
    restore_stderr();
    free_ast(prog);
    return rc;
}

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_sign_no_args(void) {
    const char *test_name = "nanoc_sign_cmd: no args";
    suppress_stderr();
    int rc = nanoc_sign_cmd(0, NULL);
    restore_stderr();
    ASSERT(rc != 0, "should return error with no args");
    PASS(test_name);
}

static void test_verify_no_args(void) {
    const char *test_name = "nanoc_verify_cmd: no args";
    suppress_stderr();
    int rc = nanoc_verify_cmd(0, NULL);
    restore_stderr();
    ASSERT(rc != 0, "should return error with no args");
    PASS(test_name);
}

static void test_verify_nonexistent_file(void) {
    const char *test_name = "nanoc_verify_cmd: nonexistent file";
    char *args[] = { "/tmp/does_not_exist_nano_test.wasm" };
    suppress_stderr();
    int rc = nanoc_verify_cmd(1, args);
    restore_stderr();
    ASSERT(rc != 0, "should return error for nonexistent file");
    PASS(test_name);
}

static void test_verify_unsigned_wasm(void) {
    const char *test_name = "nanoc_verify_cmd: unsigned WASM file";
    const char *wasm_path = "/tmp/test_sign_unsigned.wasm";
    int wrc = write_wasm_file(wasm_path);
    ASSERT(wrc == 0, "failed to create test WASM file");

    char *args[] = { (char *)wasm_path };
    suppress_stderr();
    int rc = nanoc_verify_cmd(1, args);
    restore_stderr();
    /* Unsigned file should fail verification */
    ASSERT(rc != 0, "unsigned file should fail verification");
    PASS(test_name);
}

static void test_sign_nonexistent_file(void) {
    const char *test_name = "wasm_sign_file: nonexistent file";
    suppress_stderr();
    int rc = wasm_sign_file("/tmp/does_not_exist_nano.wasm",
                             "/tmp/test_sign_key.key");
    restore_stderr();
    ASSERT(rc != 0, "should return error for nonexistent WASM file");
    PASS(test_name);
}

static void test_sign_and_verify(void) {
    const char *test_name = "sign + verify round-trip";
    const char *wasm_path = "/tmp/test_sign_roundtrip.wasm";
    const char *key_path  = "/tmp/test_sign_key.nanoc.key";

    /* Create a WASM file to sign */
    int wrc = write_wasm_file(wasm_path);
    ASSERT(wrc == 0, "failed to create WASM file");

    /* Sign it */
    suppress_stderr();
    int sign_rc = wasm_sign_file(wasm_path, key_path);
    restore_stderr();
    ASSERT(sign_rc == 0, "wasm_sign_file failed");

    /* Verify it */
    char *verify_args[] = { (char *)wasm_path };
    suppress_stderr();
    int verify_rc = nanoc_verify_cmd(1, verify_args);
    restore_stderr();
    ASSERT(verify_rc == 0, "nanoc_verify_cmd failed after signing");

    /* Clean up */
    unlink(wasm_path);
    unlink(key_path);
    PASS(test_name);
}

static void test_sign_already_signed(void) {
    const char *test_name = "wasm_sign_file: re-sign already-signed file";
    const char *wasm_path = "/tmp/test_sign_already.wasm";
    const char *key_path  = "/tmp/test_sign_key2.nanoc.key";

    int wrc = write_wasm_file(wasm_path);
    ASSERT(wrc == 0, "failed to create WASM file");

    /* Sign once */
    suppress_stderr();
    int rc1 = wasm_sign_file(wasm_path, key_path);
    restore_stderr();
    ASSERT(rc1 == 0, "first sign failed");

    /* Sign again (should strip old signature and re-sign) */
    suppress_stderr();
    int rc2 = wasm_sign_file(wasm_path, key_path);
    restore_stderr();
    ASSERT(rc2 == 0, "second sign failed");

    /* Verify still works */
    char *verify_args[] = { (char *)wasm_path };
    suppress_stderr();
    int verify_rc = nanoc_verify_cmd(1, verify_args);
    restore_stderr();
    ASSERT(verify_rc == 0, "verify failed after re-sign");

    unlink(wasm_path);
    unlink(key_path);
    PASS(test_name);
}

static void test_sign_cmd_with_file(void) {
    const char *test_name = "nanoc_sign_cmd: sign a file (uses default key path)";
    const char *wasm_path = "/tmp/test_sign_cmd.wasm";

    int wrc = write_wasm_file(wasm_path);
    ASSERT(wrc == 0, "failed to create WASM file");

    char *args[] = { (char *)wasm_path };
    /* This uses the default key path (~/.nanoc/signing.key), which may or
     * may not exist. We just verify it doesn't crash. */
    suppress_stderr();
    int rc = nanoc_sign_cmd(1, args);
    restore_stderr();
    (void)rc; /* Ignore return code - key path may not be writable */
    unlink(wasm_path);
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[sign] Ed25519 WASM signing tests...\n\n");

    test_sign_no_args();
    test_verify_no_args();
    test_verify_nonexistent_file();
    test_verify_unsigned_wasm();
    test_sign_nonexistent_file();
    test_sign_and_verify();
    test_sign_already_signed();
    test_sign_cmd_with_file();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
