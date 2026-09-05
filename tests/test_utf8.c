#include "utf8.h"
#include "diag_id.h"

#include <stdio.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_utf8_ok(void) {
    const char *test_name = "utf8: empty ascii and supplementary";
    const char *world = "Hi \xE4\xB8\x96\xE7\x95\x8C"; /* Hi 世界 */
    const char emoji[] = "\xF0\x9F\x8C\x8D"; /* 🌍 */
    size_t off = 99;
    if (!nl_utf8_validate("", 0, &off) || off != 0)
        { FAIL(test_name, "empty"); return; }
    if (!nl_utf8_validate("hello", 5, &off))
        { FAIL(test_name, "ascii"); return; }
    if (!nl_utf8_validate(world, strlen(world), &off))
        { FAIL(test_name, "world"); return; }
    if (!nl_utf8_validate(emoji, 4, &off))
        { FAIL(test_name, "emoji"); return; }
    PASS(test_name);
}

static void test_utf8_reject(void) {
    const char *test_name = "utf8: reject overlong surrogate truncated and junk";
    size_t off = 0;
    const char overlong[] = { (char)0xC0, (char)0x80 };
    const char surrogate[] = { (char)0xED, (char)0xA0, (char)0x80 };
    const char trunc[] = { (char)0xE0 };
    const char junk[] = { (char)0xFF };
    const char too_big[] = { (char)0xF4, (char)0x90, (char)0x80, (char)0x80 };
    if (nl_utf8_validate(overlong, 2, &off) || off != 0)
        { FAIL(test_name, "overlong"); return; }
    if (nl_utf8_validate(surrogate, 3, &off))
        { FAIL(test_name, "surrogate"); return; }
    if (nl_utf8_validate(trunc, 1, &off) || off != 0)
        { FAIL(test_name, "truncated"); return; }
    if (nl_utf8_validate(junk, 1, &off))
        { FAIL(test_name, "0xFF"); return; }
    if (nl_utf8_validate(too_big, 4, &off))
        { FAIL(test_name, "above U+10FFFF"); return; }
    PASS(test_name);
}

static void test_diag_ids(void) {
    const char *test_name = "diag: pipeline ids lookup English and stay stable";
    if (strcmp(NL_DIAG_SRC_UTF8, "CSRC01") != 0)
        { FAIL(test_name, "CSRC01"); return; }
    if (strcmp(nl_diag_en(NL_DIAG_IO_OPEN), "Could not open input file") != 0)
        { FAIL(test_name, "CIO01 en"); return; }
    if (strcmp(nl_diag_en(NL_DIAG_SRC_UTF8), "Source is not valid UTF-8") != 0)
        { FAIL(test_name, "CSRC01 en"); return; }
    if (nl_diag_en("not-an-id") != NULL)
        { FAIL(test_name, "unknown"); return; }
    if (nl_diag_en(NULL) != NULL)
        { FAIL(test_name, "NULL"); return; }
    PASS(test_name);
}

static void test_utf8_helpers(void) {
    const char *test_name = "utf8: cstr helpers combining and bidi";
    const char comb[] = "e\xCC\x81"; /* e + combining acute */
    const char rlm[] = "\xE2\x80\x8F"; /* U+200F RLM */
    const char junk[] = { (char)0xFF, 0 };
    if (nl_utf8_ok_cstr(NULL))
        { FAIL(test_name, "NULL"); return; }
    if (!nl_utf8_ok_cstr(""))
        { FAIL(test_name, "empty"); return; }
    if (strcmp(nl_utf8_cstr_or_marker(NULL), "") != 0)
        { FAIL(test_name, "NULL marker"); return; }
    if (strcmp(nl_utf8_cstr_or_marker(junk), "<invalid UTF-8>") != 0)
        { FAIL(test_name, "junk marker"); return; }
    if (!nl_utf8_ok_cstr(comb))
        { FAIL(test_name, "combining"); return; }
    if (!nl_utf8_ok_cstr(rlm))
        { FAIL(test_name, "rlm"); return; }
    PASS(test_name);
}

static void test_ascii_and_sanitize(void) {
    const char *test_name = "utf8: ASCII ctype and log sanitize";
    char out[64];
    char payload[32];
    if (!nl_ascii_isspace(' ') || nl_ascii_isspace('A') || nl_ascii_isspace(-1))
        { FAIL(test_name, "space"); return; }
    if (!nl_ascii_isdigit('0') || nl_ascii_isdigit(0xE9))
        { FAIL(test_name, "digit"); return; }
    if (!nl_ascii_isalpha('Z') || nl_ascii_isalpha('0'))
        { FAIL(test_name, "alpha"); return; }
    if (nl_ascii_toupper('a') != 'A' || nl_ascii_tolower('A') != 'a')
        { FAIL(test_name, "case"); return; }
    snprintf(payload, sizeof payload, "\xE2\x80\xAEx\x1B[31my");
    nl_utf8_sanitize_log(payload, out, sizeof out);
    if (strchr(out, '\x1B') || strstr(out, "\xE2\x80\xAE"))
        { FAIL(test_name, "sanitize"); return; }
    if (!strchr(out, 'x') || !strchr(out, 'y'))
        { FAIL(test_name, out); return; }
    PASS(test_name);
}

int main(void) {
    printf("UTF-8 and diagnostic id tests\n");
    test_utf8_ok();
    test_utf8_reject();
    test_utf8_helpers();
    test_diag_ids();
    test_ascii_and_sanitize();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
