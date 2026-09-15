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

static void test_vm_file_read_bytes(void) {
    const char *test_name = "vm_file_read_bytes: binary, empty and failed reads";
    char *path = vm_mktemp("nano-byte-read-");
    ASSERT(path && *path, "temporary file creation");
    FILE *file = fopen(path, "wb");
    ASSERT(file != NULL, "open binary fixture");
    unsigned char data[8193];
    for (size_t i = 0; i < sizeof(data); i++) data[i] = (unsigned char)i;
    ASSERT(fwrite(data, 1, sizeof(data), file) == sizeof(data), "write binary fixture");
    ASSERT(fclose(file) == 0, "close binary fixture");
    DynArray *bytes = vm_file_read_bytes(path);
    ASSERT(bytes && bytes->elem_type == ELEM_U8, "byte-typed result");
    ASSERT(bytes->length == sizeof(data), "complete binary length");
    for (size_t i = 0; i < sizeof(data); i++)
        ASSERT(dyn_array_get_u8(bytes, (int64_t)i) == data[i], "binary byte preserved");
    ASSERT(vm_file_write(path, "ABC") == 0, "write text fixture");
    char *text = vm_string_from_bytes(vm_file_read_bytes(path));
    ASSERT(text && !strcmp(text, "ABC"), "byte-typed string conversion");
    free(text);
    ASSERT(vm_file_write(path, "") == 0, "truncate fixture");
    bytes = vm_file_read_bytes(path);
    ASSERT(bytes && bytes->elem_type == ELEM_U8 && bytes->length == 0, "empty file");
    ASSERT(unlink(path) == 0, "remove fixture");
    bytes = vm_file_read_bytes(path);
    ASSERT(bytes && bytes->elem_type == ELEM_U8 && bytes->length == 0, "missing file");
    bytes = vm_file_read_bytes(NULL);
    ASSERT(bytes && bytes->length == 0, "null path");
    bytes = vm_file_read_bytes("/");
    ASSERT(bytes && bytes->length == 0, "directory read failure");
    free(path);
    PASS(test_name);
}

static void test_vm_trim_edges(void) {
    const char *test_name = "vm_str_trim_left/right: exact edge whitespace";
    const char *inputs[] = {NULL, "", " \t\r\n", " \tcafé\r\n", "\vcafé\f", "\xc2\xa0" "café" "\xc2\xa0"};
    const char *left[] = {"", "", "", "café\r\n", "\vcafé\f", "\xc2\xa0" "café" "\xc2\xa0"};
    const char *right[] = {"", "", "", " \tcafé", "\vcafé\f", "\xc2\xa0" "café" "\xc2\xa0"};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        char *l = vm_str_trim_left(inputs[i]);
        char *r = vm_str_trim_right(inputs[i]);
        ASSERT(l && r, "trim allocation");
        ASSERT(!strcmp(l, left[i]) && !strcmp(r, right[i]), "exact trimmed edge");
        free(l);
        free(r);
    }
    PASS(test_name);
}

static void test_vm_str_join(void) {
    const char *test_name = "vm_str_join: sized allocation and invalid arrays";
    DynArray *parts = dyn_array_new(ELEM_STRING);
    ASSERT(parts != NULL, "array allocation");
    char *joined = vm_str_join(parts, ",");
    ASSERT(joined && !strcmp(joined, ""), "empty result");
    free(joined);
    char word[5001];
    memset(word, 'x', sizeof(word) - 1);
    word[sizeof(word) - 1] = '\0';
    dyn_array_push_string(parts, word);
    dyn_array_push_string(parts, "");
    dyn_array_push_string(parts, word);
    joined = vm_str_join(parts, "--");
    ASSERT(joined && strlen(joined) == 10004, "long result length");
    ASSERT(!memcmp(joined, word, 5000) && !memcmp(joined + 5000, "----", 4) &&
           !strcmp(joined + 5004, word), "long result bytes");
    free(joined);
    ASSERT(vm_str_join(NULL, ",") == NULL, "null array rejected");
    ASSERT(vm_str_join(parts, NULL) == NULL, "null delimiter rejected");
    DynArray invalid = *parts;
    invalid.elem_type = ELEM_INT;
    ASSERT(vm_str_join(&invalid, ",") == NULL, "wrong element type rejected");
    invalid = *parts;
    invalid.length = -1;
    ASSERT(vm_str_join(&invalid, ",") == NULL, "negative length rejected");
    invalid = *parts;
    invalid.data = NULL;
    ASSERT(vm_str_join(&invalid, ",") == NULL, "missing storage rejected");
    PASS(test_name);
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
    ASSERT(vm_str_index_of("abc", "") == 0, "I find an empty needle at the start");
    ASSERT(vm_str_index_of(NULL, "a") == -1, "I reject a null haystack");
    ASSERT(vm_str_last_index_of("ababa", "aba") == 2, "I include overlapping matches");
    ASSERT(vm_str_last_index_of("abc", "") == 3, "I find an empty last needle at the end");
    ASSERT(vm_str_last_index_of("", "") == 0, "I search an empty string");
    ASSERT(vm_str_last_index_of("a", "ab") == -1, "I reject a longer needle");
    ASSERT(vm_str_last_index_of("abc", NULL) == -1, "I reject a null needle");
    ASSERT(vm_str_last_index_of(NULL, "a") == -1, "I reject a null haystack");
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
    ASSERT(!strcmp(dyn_array_get_string(result, 0), "0"), "exit status");
    ASSERT(!strcmp(dyn_array_get_string(result, 1), "hello\n"), "captured output");
    char command[10064];
    memset(command, ' ', 10000);
    memcpy(command, ": ", 2);
    strcpy(command + 10000, "; printf out; printf err >&2; exit 7");
    result = vm_process_run(command);
    ASSERT(result && !strcmp(dyn_array_get_string(result, 0), "7"), "long command status");
    ASSERT(!strcmp(dyn_array_get_string(result, 1), "out"), "long command stdout");
    ASSERT(!strcmp(dyn_array_get_string(result, 2), "err"), "long command stderr");
    result = vm_process_run("kill -TERM $$");
    ASSERT(result && !strcmp(dyn_array_get_string(result, 0), "-1"), "signal status");
    result = vm_process_run("printf '\\000'");
    ASSERT(result && !strcmp(dyn_array_get_string(result, 0), "-1"), "reject binary capture");
    result = vm_process_run(NULL);
    ASSERT(result && !strcmp(dyn_array_get_string(result, 0), "-1"), "reject null command");
    long limit = sysconf(_SC_ARG_MAX);
    ASSERT(limit > 0 && limit < 16 * 1024 * 1024, "bounded host argument limit");
    char *oversized = malloc((size_t)limit + 2);
    ASSERT(oversized, "allocate oversized command");
    memset(oversized, ' ', (size_t)limit + 1);
    memcpy(oversized, "printf UNEXPECTED;", 18);
    oversized[limit + 1] = '\0';
    result = vm_process_run(oversized);
    free(oversized);
    ASSERT(result && !strcmp(dyn_array_get_string(result, 0), "127"), "reject host argument overflow");
    ASSERT(!*dyn_array_get_string(result, 1), "do not execute a truncated command prefix");
    ASSERT(strstr(dyn_array_get_string(result, 2), "could not execute"), "exec failure diagnostic");
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

static void test_vm_format(void) {
    const char *test_name = "vm_format: substitutions, literals and invalid arrays";
    DynArray *args = dyn_array_new(ELEM_STRING);
    ASSERT(args, "allocate arguments");
    char *result = vm_format("%s %% %q %", args);
    ASSERT(result && !strcmp(result, "%s %% %q %"), "missing substitutions stay literal");
    free(result);
    dyn_array_push_string(args, "é🙂");
    dyn_array_push_string(args, "42");
    result = vm_format("%%s/%d/%f", args);
    ASSERT(result && !strcmp(result, "%é🙂/42/%f"), "scan placeholders, not printf escapes");
    free(result);
    result = vm_format("", args);
    ASSERT(result && !*result, "ignore extra arguments");
    free(result);
    char long_part[8193];
    memset(long_part, 'x', sizeof(long_part) - 1);
    long_part[sizeof(long_part) - 1] = '\0';
    dyn_array_push_string(args, long_part);
    result = vm_format("%s%d%g", args);
    ASSERT(result && strlen(result) == strlen("é🙂42") + 8192, "long result is not truncated");
    ASSERT(!strcmp(result + strlen("é🙂42"), long_part), "long substitution bytes");
    free(result);
    ASSERT(!vm_format(NULL, args) && !vm_format("x", NULL), "reject null inputs");
    DynArray malformed = *args;
    malformed.elem_type = ELEM_INT;
    ASSERT(!vm_format("x", &malformed), "reject wrong element type");
    malformed = *args;
    malformed.length = -1;
    ASSERT(!vm_format("x", &malformed), "reject negative length");
    malformed = *args;
    malformed.data = NULL;
    ASSERT(!vm_format("x", &malformed), "reject missing storage");
    PASS(test_name);
}

int main(void) {
    test_vm_format();
    test_vm_str_join();
    test_vm_trim_edges();
    test_vm_file_read_bytes();
    printf("\n[vm_builtins] NanoVM built-in function tests...\n\n");

    test_vm_getcwd();
    test_vm_file_exists_nonexistent();
    test_vm_file_write_read();
    test_vm_dir_exists();
    test_vm_dir_create();
    test_vm_dir_list();
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
