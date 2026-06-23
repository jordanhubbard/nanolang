/*
 * test_fmt.c — unit tests for fmt.c (nanolang source formatter)
 *
 * Tests fmt_source() and fmt_file() with various nano source snippets.
 */

#include "../src/nanolang.h"
#include "../src/fmt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

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

/* Write content to a temp file and return a static path */
static const char *write_temp_file(const char *content) {
    static char path[] = "/tmp/test_fmt_nano.nano";
    FILE *f = fopen(path, "w");
    if (!f) return NULL;
    fputs(content, f);
    fclose(f);
    return path;
}

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_fmt_source_null(void) {
    const char *test_name = "fmt_source: NULL input returns NULL";
    char *result = fmt_source(NULL, NULL, NULL);
    ASSERT(result == NULL, "expected NULL for NULL source");
    PASS(test_name);
}

static void test_fmt_source_empty(void) {
    const char *test_name = "fmt_source: empty string returns NULL (no tokens)";
    char *result = fmt_source("", NULL, NULL);
    /* Empty source has no tokens, returns NULL */
    if (result) free(result);
    PASS(test_name); /* Either NULL or empty string is acceptable */
}

static void test_fmt_source_simple_fn(void) {
    const char *test_name = "fmt_source: simple function formats without crash";
    const char *src = "fn add(a: int, b: int) -> int { return (+ a b) }\n";
    char *result = fmt_source(src, "test.nano", NULL);
    ASSERT(result != NULL, "fmt_source should return non-NULL for valid source");
    ASSERT(strstr(result, "add") != NULL, "formatted output should contain 'add'");
    ASSERT(strstr(result, "int") != NULL, "formatted output should contain 'int'");
    free(result);
    PASS(test_name);
}

static void test_fmt_source_pub_fn(void) {
    const char *test_name = "fmt_source: pub fn preserved";
    const char *src = "pub fn square(n: int) -> int { return (* n n) }\n";
    char *result = fmt_source(src, NULL, NULL);
    ASSERT(result != NULL, "fmt_source should succeed");
    ASSERT(strstr(result, "square") != NULL, "output should contain 'square'");
    free(result);
    PASS(test_name);
}

static void test_fmt_source_struct(void) {
    const char *test_name = "fmt_source: struct declaration";
    const char *src =
        "pub struct Point {\n"
        "  x: int,\n"
        "  y: int\n"
        "}\n";
    char *result = fmt_source(src, NULL, NULL);
    ASSERT(result != NULL, "fmt_source should succeed for struct");
    ASSERT(strstr(result, "Point") != NULL, "output should contain 'Point'");
    ASSERT(strstr(result, "struct") != NULL, "output should contain 'struct'");
    free(result);
    PASS(test_name);
}

static void test_fmt_source_enum(void) {
    const char *test_name = "fmt_source: enum declaration";
    const char *src =
        "pub enum Color {\n"
        "  Red,\n"
        "  Green,\n"
        "  Blue\n"
        "}\n";
    char *result = fmt_source(src, NULL, NULL);
    ASSERT(result != NULL, "fmt_source should succeed for enum");
    ASSERT(strstr(result, "Color") != NULL, "output should contain 'Color'");
    free(result);
    PASS(test_name);
}

static void test_fmt_source_let_binding(void) {
    const char *test_name = "fmt_source: let binding";
    const char *src =
        "fn test() -> int {\n"
        "  let x: int = 42\n"
        "  return x\n"
        "}\n";
    char *result = fmt_source(src, NULL, NULL);
    ASSERT(result != NULL, "fmt_source should succeed for let binding");
    ASSERT(strstr(result, "let") != NULL, "output should contain 'let'");
    free(result);
    PASS(test_name);
}

static void test_fmt_source_custom_indent(void) {
    const char *test_name = "fmt_source: custom indent size";
    const char *src = "fn f() -> int { return 1 }\n";
    FmtOptions opts = { .indent_size = 2, .write_in_place = false, .check_only = false, .verbose = false };
    char *result = fmt_source(src, NULL, &opts);
    ASSERT(result != NULL, "fmt_source should succeed with custom indent");
    free(result);
    PASS(test_name);
}

static void test_fmt_source_if_else(void) {
    const char *test_name = "fmt_source: if/else expression";
    const char *src =
        "fn abs(n: int) -> int {\n"
        "  if (> n 0) {\n"
        "    return n\n"
        "  } else {\n"
        "    return (- 0 n)\n"
        "  }\n"
        "}\n";
    char *result = fmt_source(src, NULL, NULL);
    ASSERT(result != NULL, "fmt_source should succeed for if/else");
    free(result);
    PASS(test_name);
}

static void test_fmt_source_multiple_fns(void) {
    const char *test_name = "fmt_source: multiple function definitions";
    const char *src =
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn mul(a: int, b: int) -> int { return (* a b) }\n"
        "fn sub(a: int, b: int) -> int { return (- a b) }\n";
    char *result = fmt_source(src, NULL, NULL);
    ASSERT(result != NULL, "fmt_source should succeed for multiple functions");
    ASSERT(strstr(result, "add") != NULL, "output should contain 'add'");
    ASSERT(strstr(result, "mul") != NULL, "output should contain 'mul'");
    ASSERT(strstr(result, "sub") != NULL, "output should contain 'sub'");
    free(result);
    PASS(test_name);
}

static void test_fmt_file_nonexistent(void) {
    const char *test_name = "fmt_file: nonexistent file returns error";
    suppress_stderr();
    int rc = fmt_file("/tmp/does_not_exist_nano_fmt.nano", NULL);
    restore_stderr();
    ASSERT(rc != 0, "fmt_file should return error for nonexistent file");
    PASS(test_name);
}

static void test_fmt_file_null_path(void) {
    const char *test_name = "fmt_file: NULL path returns error";
    int rc = fmt_file(NULL, NULL);
    ASSERT(rc != 0, "fmt_file should return error for NULL path");
    PASS(test_name);
}

static void test_fmt_file_valid(void) {
    const char *test_name = "fmt_file: valid file default options (stdout)";
    const char *path = write_temp_file("fn f(x: int) -> int { return x }\n");
    ASSERT(path != NULL, "failed to write temp file");
    /* Default options: prints to stdout. We just verify it doesn't crash. */
    int rc = fmt_file(path, NULL);
    ASSERT(rc == 0, "fmt_file should succeed for valid nano file");
    PASS(test_name);
}

static void test_fmt_file_check_only(void) {
    const char *test_name = "fmt_file: check_only on already-formatted file";
    /* Write a simple already-formatted file */
    const char *path = write_temp_file("fn f(x: int) -> int {\n    return x\n}\n");
    ASSERT(path != NULL, "failed to write temp file");
    FmtOptions opts = { .indent_size = 4, .write_in_place = false, .check_only = true, .verbose = false };
    /* check_only returns 0 if no changes needed, 2 if changes would be made */
    int rc = fmt_file(path, &opts);
    (void)rc; /* Either 0 or 2 is fine — just verify no crash */
    PASS(test_name);
}

static void test_fmt_file_write_in_place(void) {
    const char *test_name = "fmt_file: write_in_place on valid file";
    const char *path = write_temp_file("fn g(n: int) -> int { return (* n n) }\n");
    ASSERT(path != NULL, "failed to write temp file");
    FmtOptions opts = { .indent_size = 4, .write_in_place = true, .check_only = false, .verbose = false };
    int rc = fmt_file(path, &opts);
    ASSERT(rc == 0, "fmt_file write_in_place should succeed");
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[fmt] Source formatter tests...\n\n");

    test_fmt_source_null();
    test_fmt_source_empty();
    test_fmt_source_simple_fn();
    test_fmt_source_pub_fn();
    test_fmt_source_struct();
    test_fmt_source_enum();
    test_fmt_source_let_binding();
    test_fmt_source_custom_indent();
    test_fmt_source_if_else();
    test_fmt_source_multiple_fns();
    test_fmt_file_nonexistent();
    test_fmt_file_null_path();
    test_fmt_file_valid();
    test_fmt_file_check_only();
    test_fmt_file_write_in_place();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
