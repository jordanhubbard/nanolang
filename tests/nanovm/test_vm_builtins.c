/*
 * test_vm_builtins.c — unit tests for nanovm/vm_builtins.c
 *
 * Exercises the C-callable VM built-in functions directly:
 * file system, string, character classification, process, and byte utilities.
 */

#include "nanovm/vm_builtins.h"
#include "runtime/dyn_array.h"
#include <stdio.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-55s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-55s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_vm_getcwd(void) {
    const char *test_name = "vm_getcwd: returns non-NULL";
    char *cwd = vm_getcwd();
    ASSERT(cwd != NULL, "vm_getcwd should return non-NULL");
    ASSERT(strlen(cwd) > 0, "cwd should be non-empty");
    free(cwd);
    PASS(test_name);
}

static void test_vm_file_exists_nonexistent(void) {
    const char *test_name = "vm_file_exists: nonexistent file returns 0";
    int64_t result = vm_file_exists("/tmp/does_not_exist_vm_builtins_test_xyz");
    ASSERT(result == 0, "nonexistent file should return 0");
    PASS(test_name);
}

static void test_vm_file_write_read(void) {
    const char *test_name = "vm_file_write/read: round-trip";
    const char *path = "/tmp/test_vm_builtins_rw.txt";
    int64_t wrc = vm_file_write(path, "hello vm builtins");
    ASSERT(wrc == 0, "vm_file_write should succeed");
    ASSERT(vm_file_exists(path) == 1, "file should exist after write");
    char *content = vm_file_read(path);
    ASSERT(content != NULL, "vm_file_read should return non-NULL");
    ASSERT(strncmp(content, "hello vm builtins", 17) == 0,
           "read content should match written content");
    free(content);
    unlink(path);
    PASS(test_name);
}

static void test_vm_file_read_nonexistent(void) {
    const char *test_name = "vm_file_read: nonexistent file returns NULL or empty";
    char *result = vm_file_read("/tmp/no_such_file_vm_builtins.txt");
    /* vm_file_read may return NULL or empty string for nonexistent files */
    if (result) free(result);
    PASS(test_name); /* Just verify no crash */
}

static void test_vm_dir_exists(void) {
    const char *test_name = "vm_dir_exists: /tmp exists";
    int64_t result = vm_dir_exists("/tmp");
    ASSERT(result == 1, "/tmp directory should exist");
    int64_t nodir = vm_dir_exists("/tmp/no_such_dir_vm_builtins_xyz");
    ASSERT(nodir == 0, "nonexistent dir should return 0");
    PASS(test_name);
}

static void test_vm_dir_create(void) {
    const char *test_name = "vm_dir_create: creates directory";
    const char *path = "/tmp/test_vm_builtins_dir";
    rmdir(path); /* clean up if exists */
    int64_t rc = vm_dir_create(path);
    ASSERT(rc == 0, "vm_dir_create should succeed");
    ASSERT(vm_dir_exists(path) == 1, "directory should exist after create");
    rmdir(path);
    PASS(test_name);
}

static void test_vm_dir_list(void) {
    const char *test_name = "vm_dir_list: /tmp returns non-NULL";
    DynArray *list = vm_dir_list("/tmp");
    ASSERT(list != NULL, "vm_dir_list should return non-NULL for /tmp");
    (void)list; /* DynArray has no free function; leak in test is acceptable */
    PASS(test_name);
}

static void test_vm_dir_list_nonexistent(void) {
    const char *test_name = "vm_dir_list: nonexistent dir returns NULL or empty";
    DynArray *list = vm_dir_list("/tmp/no_such_dir_vm_builtins");
    /* May return NULL or empty array — just verify no crash */
    (void)list;
    PASS(test_name);
}

static void test_vm_mktemp_dir(void) {
    const char *test_name = "vm_mktemp_dir: creates temp dir";
    char *dir = vm_mktemp_dir("vm_builtins_test");
    ASSERT(dir != NULL, "vm_mktemp_dir should return non-NULL");
    ASSERT(vm_dir_exists(dir) == 1, "temp dir should exist");
    rmdir(dir);
    free(dir);
    PASS(test_name);
}

static void test_vm_getenv(void) {
    const char *test_name = "vm_getenv: HOME is set";
    char *home = vm_getenv("HOME");
    ASSERT(home != NULL, "HOME env var should be set");
    free(home);
    PASS(test_name);
}

static void test_vm_str_index_of(void) {
    const char *test_name = "vm_str_index_of: finds substring";
    int64_t pos = vm_str_index_of("hello world", "world");
    ASSERT(pos == 6, "world should be at position 6");
    int64_t notfound = vm_str_index_of("hello", "xyz");
    ASSERT(notfound == -1, "not found should return -1");
    PASS(test_name);
}

static void test_vm_string_from_char(void) {
    const char *test_name = "vm_string_from_char: converts codepoint to string";
    char *s = vm_string_from_char('A');
    ASSERT(s != NULL, "should return non-NULL");
    ASSERT(s[0] == 'A', "first char should be 'A'");
    free(s);
    PASS(test_name);
}

static void test_vm_char_classification(void) {
    const char *test_name = "vm_is_digit/alpha/etc: character classification";
    ASSERT(vm_is_digit('5') == 1, "'5' should be digit");
    ASSERT(vm_is_digit('a') == 0, "'a' should not be digit");
    ASSERT(vm_is_alpha('z') == 1, "'z' should be alpha");
    ASSERT(vm_is_alpha('9') == 0, "'9' should not be alpha");
    ASSERT(vm_is_alnum('A') == 1, "'A' should be alnum");
    ASSERT(vm_is_space(' ') == 1, "' ' should be space");
    ASSERT(vm_is_upper('Z') == 1, "'Z' should be upper");
    ASSERT(vm_is_lower('a') == 1, "'a' should be lower");
    ASSERT(vm_is_whitespace('\t') == 1, "tab should be whitespace");
    PASS(test_name);
}

static void test_vm_digit_value(void) {
    const char *test_name = "vm_digit_value: converts char to int";
    ASSERT(vm_digit_value('0') == 0, "'0' has value 0");
    ASSERT(vm_digit_value('9') == 9, "'9' has value 9");
    ASSERT(vm_digit_value('a') == -1, "non-digit returns -1");
    PASS(test_name);
}

static void test_vm_char_case(void) {
    const char *test_name = "vm_char_to_lower/upper: case conversion";
    ASSERT(vm_char_to_lower('A') == 'a', "'A' should lower to 'a'");
    ASSERT(vm_char_to_upper('z') == 'Z', "'z' should upper to 'Z'");
    PASS(test_name);
}

static void test_vm_bytes_roundtrip(void) {
    const char *test_name = "vm_bytes_from_string/vm_string_from_bytes: round-trip";
    DynArray *bytes = vm_bytes_from_string("hello");
    ASSERT(bytes != NULL, "vm_bytes_from_string should succeed");
    char *back = vm_string_from_bytes(bytes);
    ASSERT(back != NULL, "vm_string_from_bytes should succeed");
    ASSERT(strcmp(back, "hello") == 0, "round-trip should preserve content");
    free(back);
    (void)bytes; /* no dyn_array_free */
    PASS(test_name);
}

static void test_vm_bstr_utf8(void) {
    const char *test_name = "vm_bstr_utf8_length: ASCII string";
    int64_t len = vm_bstr_utf8_length("hello");
    ASSERT(len == 5, "ASCII 'hello' has UTF-8 length 5");
    int64_t valid = vm_bstr_validate_utf8("hello");
    ASSERT(valid == 1, "ASCII string should be valid UTF-8");
    int64_t ch = vm_bstr_utf8_char_at("hello", 1);
    ASSERT(ch == 'e', "char at index 1 of 'hello' should be 'e'");
    PASS(test_name);
}

static void test_vm_process_run(void) {
    const char *test_name = "vm_process_run: echo command";
    DynArray *result = vm_process_run("echo hello");
    ASSERT(result != NULL, "vm_process_run should return non-NULL");
    (void)result; /* no dyn_array_free */
    PASS(test_name);
}

static void test_vm_file_write_null(void) {
    const char *test_name = "vm_file_write: NULL path returns error";
    int64_t rc = vm_file_write(NULL, "content");
    ASSERT(rc != 0, "NULL path should return error");
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[vm_builtins] NanoVM built-in function tests...\n\n");

    test_vm_getcwd();
    test_vm_file_exists_nonexistent();
    test_vm_file_write_read();
    test_vm_file_read_nonexistent();
    test_vm_dir_exists();
    test_vm_dir_create();
    test_vm_dir_list();
    test_vm_dir_list_nonexistent();
    test_vm_mktemp_dir();
    test_vm_getenv();
    test_vm_str_index_of();
    test_vm_string_from_char();
    test_vm_char_classification();
    test_vm_digit_value();
    test_vm_char_case();
    test_vm_bytes_roundtrip();
    test_vm_bstr_utf8();
    test_vm_process_run();
    test_vm_file_write_null();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
