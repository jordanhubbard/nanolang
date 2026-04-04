/**
 * test_diagnostics.c — unit tests for json_diagnostics.c and toon_output.c
 *
 * Exercises the diagnostic accumulation, accessor, and output APIs without
 * requiring a full compiler pipeline.
 */

#include "../src/json_diagnostics.h"
#include "../src/toon_output.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %d, expected %d)\n", \
        #a, #b, __LINE__, (int)(a), (int)(b)); exit(1); }
#define ASSERT_STR_EQ(a, b) \
    if (strcmp((a), (b)) != 0) { printf("\n    FAILED: %s == %s at line %d\n    got: \"%s\"\n    expected: \"%s\"\n", \
        #a, #b, __LINE__, (a), (b)); exit(1); }
#define ASSERT_NULL(p) \
    if ((p) != NULL) { printf("\n    FAILED: expected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_NOT_NULL(p) \
    if ((p) == NULL) { printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_CONTAINS(haystack, needle) \
    if (strstr((haystack), (needle)) == NULL) { \
        printf("\n    FAILED: expected \"%s\" in output at line %d\n    got: \"%s\"\n", \
            (needle), __LINE__, (haystack)); exit(1); }

/* Required by runtime/cli.c (extern in eval.c) */
int g_argc = 0;
char **g_argv = NULL;

/* Helper: read a small file into a static buffer */
static char s_file_buf[8192];
static const char *read_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return "";
    size_t n = fread(s_file_buf, 1, sizeof(s_file_buf) - 1, f);
    fclose(f);
    s_file_buf[n] = '\0';
    return s_file_buf;
}

static char s_tmp_path[256];
static const char *make_tmp_path(const char *suffix) {
    const char *td = getenv("TMPDIR");
    if (!td) td = "/tmp";
    snprintf(s_tmp_path, sizeof(s_tmp_path), "%s/nano_diag_test_%d%s", td, getpid(), suffix);
    return s_tmp_path;
}

/* ============================================================================
 * json_diagnostics tests
 * ============================================================================ */

void test_json_init_empty(void) {
    json_diagnostics_init();
    ASSERT_EQ(json_diag_count(), 0);
    json_diagnostics_cleanup();
}

void test_json_add_when_disabled(void) {
    json_diagnostics_init();
    g_json_output_enabled = false;
    json_diagnostics_add(DIAG_ERROR, "E001", "ignored", "f.nano", 1, 1, NULL);
    ASSERT_EQ(json_diag_count(), 0);
    json_diagnostics_cleanup();
}

void test_json_enable_and_add_error(void) {
    json_diagnostics_init();
    json_diagnostics_enable();
    json_error("E001", "undefined variable 'x'", "main.nano", 10, 5, "Did you mean 'xx'?");
    ASSERT_EQ(json_diag_count(), 1);
    ASSERT_EQ(json_diag_severity(0), (int)DIAG_ERROR);
    ASSERT_STR_EQ(json_diag_code(0), "E001");
    ASSERT_STR_EQ(json_diag_message(0), "undefined variable 'x'");
    ASSERT_STR_EQ(json_diag_file(0), "main.nano");
    ASSERT_EQ(json_diag_line(0), 10);
    ASSERT_EQ(json_diag_column(0), 5);
    json_diagnostics_cleanup();
    g_json_output_enabled = false;
}

void test_json_add_warning(void) {
    json_diagnostics_init();
    json_diagnostics_enable();
    json_warning("W001", "unused variable", "lib.nano", 3, 2, "remove it");
    ASSERT_EQ(json_diag_count(), 1);
    ASSERT_EQ(json_diag_severity(0), (int)DIAG_WARNING);
    ASSERT_STR_EQ(json_diag_code(0), "W001");
    json_diagnostics_cleanup();
    g_json_output_enabled = false;
}

void test_json_all_severities(void) {
    json_diagnostics_init();
    json_diagnostics_enable();
    json_diagnostics_add(DIAG_ERROR,   "E001", "error",   "a.nano", 1, 1, NULL);
    json_diagnostics_add(DIAG_WARNING, "W001", "warning", "b.nano", 2, 2, NULL);
    json_diagnostics_add(DIAG_INFO,    "I001", "info",    "c.nano", 3, 3, NULL);
    json_diagnostics_add(DIAG_HINT,    "H001", "hint",    "d.nano", 4, 4, NULL);
    ASSERT_EQ(json_diag_count(), 4);
    ASSERT_EQ(json_diag_severity(0), (int)DIAG_ERROR);
    ASSERT_EQ(json_diag_severity(1), (int)DIAG_WARNING);
    ASSERT_EQ(json_diag_severity(2), (int)DIAG_INFO);
    ASSERT_EQ(json_diag_severity(3), (int)DIAG_HINT);
    ASSERT_STR_EQ(json_diag_file(2), "c.nano");
    ASSERT_EQ(json_diag_line(3), 4);
    ASSERT_EQ(json_diag_column(3), 4);
    json_diagnostics_cleanup();
    g_json_output_enabled = false;
}

void test_json_out_of_bounds_accessors(void) {
    json_diagnostics_init();
    json_diagnostics_enable();
    /* Empty: all accessors should return safe defaults */
    ASSERT_EQ(json_diag_count(), 0);
    ASSERT_EQ(json_diag_severity(-1), 0);
    ASSERT_EQ(json_diag_severity(0), 0);
    ASSERT_NULL(json_diag_code(0));
    ASSERT_NULL(json_diag_message(0));
    ASSERT_NULL(json_diag_file(0));
    ASSERT_EQ(json_diag_line(0), 0);
    ASSERT_EQ(json_diag_column(0), 0);
    json_diagnostics_cleanup();
    g_json_output_enabled = false;
}

void test_json_capacity_expansion(void) {
    json_diagnostics_init();
    json_diagnostics_enable();
    /* Add more than the initial capacity (10) to trigger realloc */
    for (int i = 0; i < 25; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "error %d", i);
        json_error("E001", msg, "f.nano", i + 1, 1, NULL);
    }
    ASSERT_EQ(json_diag_count(), 25);
    ASSERT_EQ(json_diag_line(24), 25);
    json_diagnostics_cleanup();
    g_json_output_enabled = false;
}

void test_json_null_suggestion(void) {
    json_diagnostics_init();
    json_diagnostics_enable();
    json_error("E002", "type mismatch", "t.nano", 7, 1, NULL);
    ASSERT_EQ(json_diag_count(), 1);
    /* No assertion on suggestion - just verify it doesn't crash */
    json_diagnostics_cleanup();
    g_json_output_enabled = false;
}

/* ============================================================================
 * toon_output tests
 * ============================================================================ */

/* IMPORTANT: toon_diagnostics_enable() sets a static flag that cannot be reset.
 * Tests that check the disabled state MUST run before any test calls enable(). */

void test_toon_disabled_state_and_add(void) {
    /* Must run first — verifies initial disabled state and add-while-disabled path */
    ASSERT(!toon_diagnostics_enabled());

    /* Adding while disabled should be silently ignored */
    toon_diagnostics_add("error", "E001", "not added", "f.nano", 1, 1);

    /* Now enable */
    toon_diagnostics_enable();
    ASSERT(toon_diagnostics_enabled());

    /* Output to temp file — diagnostic_count must be 0 (add was ignored) */
    const char *path = make_tmp_path(".toon");
    bool ok = toon_diagnostics_output_to_file(path, "f.nano", "out", 0);
    ASSERT(ok);
    const char *content = read_file(path);
    remove(path);
    ASSERT_CONTAINS(content, "diagnostic_count: 0");
    ASSERT_CONTAINS(content, "has_diagnostics: false");

    toon_diagnostics_cleanup();
}

void test_toon_add_and_output(void) {
    /* toon is already enabled from previous test */
    toon_diagnostics_add("error",   "E001", "undefined 'x'", "main.nano", 10, 5);
    toon_diagnostics_add("warning", "W002", "unused var",    "lib.nano",   3, 1);

    const char *path = make_tmp_path("2.toon");
    bool ok = toon_diagnostics_output_to_file(path, "main.nano", "out", 1);
    ASSERT(ok);
    const char *content = read_file(path);
    remove(path);

    ASSERT_CONTAINS(content, "tool: nanoc_c");
    ASSERT_CONTAINS(content, "success: false");  /* exit_code=1 */
    ASSERT_CONTAINS(content, "diagnostic_count: 2");
    ASSERT_CONTAINS(content, "has_diagnostics: true");
    ASSERT_CONTAINS(content, "E001");
    ASSERT_CONTAINS(content, "W002");
    ASSERT_CONTAINS(content, "main.nano");

    toon_diagnostics_cleanup();
}

void test_toon_null_path_returns_false(void) {
    bool ok = toon_diagnostics_output_to_file(NULL, "in.nano", "out", 0);
    ASSERT(!ok);
}

void test_toon_empty_output(void) {
    /* Cleanup left count at 0 */
    const char *path = make_tmp_path("3.toon");
    bool ok = toon_diagnostics_output_to_file(path, NULL, NULL, 0);
    ASSERT(ok);
    const char *content = read_file(path);
    remove(path);
    ASSERT_CONTAINS(content, "diagnostic_count: 0");
    ASSERT_CONTAINS(content, "success: true");
    toon_diagnostics_cleanup();
}

void test_toon_special_char_escaping(void) {
    toon_diagnostics_add("error", "E003", "msg\twith\ttabs\nand newline\\slash", "f.nano", 1, 1);

    const char *path = make_tmp_path("4.toon");
    toon_diagnostics_output_to_file(path, "f.nano", "out", 1);
    const char *content = read_file(path);
    remove(path);

    /* Tab and newline should be escaped, backslash doubled */
    ASSERT_CONTAINS(content, "\\t");
    ASSERT_CONTAINS(content, "\\n");
    ASSERT_CONTAINS(content, "\\\\");

    toon_diagnostics_cleanup();
}

void test_toon_max_diagnostics_no_overflow(void) {
    /* Fill up close to MAX_TOON_DIAGNOSTICS (256) without exceeding */
    for (int i = 0; i < 50; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "diag %d", i);
        toon_diagnostics_add("info", "I001", msg, "f.nano", i + 1, 1);
    }

    const char *path = make_tmp_path("5.toon");
    bool ok = toon_diagnostics_output_to_file(path, "f.nano", "out", 0);
    ASSERT(ok);
    const char *content = read_file(path);
    remove(path);
    ASSERT_CONTAINS(content, "diagnostic_count: 50");

    toon_diagnostics_cleanup();
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== JSON Diagnostics Tests ===\n");
    TEST(json_init_empty);
    TEST(json_add_when_disabled);
    TEST(json_enable_and_add_error);
    TEST(json_add_warning);
    TEST(json_all_severities);
    TEST(json_out_of_bounds_accessors);
    TEST(json_capacity_expansion);
    TEST(json_null_suggestion);

    printf("\n=== TOON Output Tests ===\n");
    /* NOTE: toon tests are order-dependent — disabled-state test MUST run first */
    TEST(toon_disabled_state_and_add);
    TEST(toon_add_and_output);
    TEST(toon_null_path_returns_false);
    TEST(toon_empty_output);
    TEST(toon_special_char_escaping);
    TEST(toon_max_diagnostics_no_overflow);

    printf("\n✓ All diagnostics tests passed!\n");
    return 0;
}
