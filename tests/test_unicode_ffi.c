/*
 * test_unicode_ffi.c — grapheme, normalize, case fold, and display width
 * through the utf8proc FFI. These are C bindings; unicode.nano is extern.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

int64_t nl_str_byte_length(const char *str);
int64_t nl_str_grapheme_length(const char *str);
int64_t nl_str_codepoint_at(const char *str, int64_t byte_index);
char *nl_str_grapheme_at(const char *str, int64_t grapheme_index);
char *nl_str_to_lowercase(const char *str);
char *nl_str_to_uppercase(const char *str);
char *nl_str_normalize(const char *str, int64_t form);
char *nl_str_casefold(const char *str);
int64_t nl_str_display_width(const char *str);
bool nl_str_is_ascii(const char *str);
bool nl_str_is_valid_utf8(const char *str);

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_byte_vs_codepoint_vs_grapheme(void) {
    const char *test_name = "unicode: byte codepoint grapheme stay distinct";
    const char comb[] = "e\xCC\x81"; /* e + combining acute */
    const char *world = "\xE4\xB8\x96\xE7\x95\x8C"; /* 世界 */
    if (nl_str_byte_length(comb) != 3)
        { FAIL(test_name, "combining bytes"); return; }
    if (nl_str_grapheme_length(comb) != 1)
        { FAIL(test_name, "combining grapheme"); return; }
    if (nl_str_byte_length(world) != 6)
        { FAIL(test_name, "world bytes"); return; }
    if (nl_str_grapheme_length(world) != 2)
        { FAIL(test_name, "world graphemes"); return; }
    if (nl_str_byte_length("Hello") != nl_str_grapheme_length("Hello"))
        { FAIL(test_name, "ascii equal"); return; }
    PASS(test_name);
}

static void test_normalize_nfc_nfd(void) {
    const char *test_name = "unicode: NFC of decomposed acute e";
    const char nfd[] = "e\xCC\x81";
    const char nfc[] = "\xC3\xA9"; /* U+00E9 */
    char *composed = nl_str_normalize(nfd, 0);
    char *decomposed = nl_str_normalize(nfc, 1);
    if (!composed || strcmp(composed, nfc) != 0)
        { FAIL(test_name, "NFC"); free(composed); free(decomposed); return; }
    if (!decomposed || strcmp(decomposed, nfd) != 0)
        { FAIL(test_name, "NFD"); free(composed); free(decomposed); return; }
    free(composed);
    free(decomposed);
    PASS(test_name);
}

static void test_case_and_fold(void) {
    const char *test_name = "unicode: lowercase uppercase and casefold";
    char *low = nl_str_to_lowercase("Hello");
    char *up = nl_str_to_uppercase("Hello");
    const char *ss = "\xC3\x9F"; /* ß */
    char *folded = nl_str_casefold(ss);
    if (!low || strcmp(low, "hello") != 0)
        { FAIL(test_name, "lower"); free(low); free(up); free(folded); return; }
    if (!up || strcmp(up, "HELLO") != 0)
        { FAIL(test_name, "upper"); free(low); free(up); free(folded); return; }
    if (!folded || strcmp(folded, "ss") != 0)
        { FAIL(test_name, "casefold ß"); free(low); free(up); free(folded); return; }
    free(low);
    free(up);
    free(folded);
    PASS(test_name);
}

static void test_display_width(void) {
    const char *test_name = "unicode: display width combining and CJK";
    const char comb[] = "e\xCC\x81";
    const char *han = "\xE4\xBD\xA0"; /* 你 */
    if (nl_str_display_width("A") != 1)
        { FAIL(test_name, "A"); return; }
    if (nl_str_display_width(comb) != 1)
        { FAIL(test_name, "combining"); return; }
    if (nl_str_display_width(han) != 2)
        { FAIL(test_name, "CJK"); return; }
    if (nl_str_display_width(NULL) != 0)
        { FAIL(test_name, "NULL"); return; }
    PASS(test_name);
}

static void test_ascii_and_utf8(void) {
    const char *test_name = "unicode: ascii and utf8 predicates";
    const char junk[] = { (char)0xFF, 0 };
    if (!nl_str_is_ascii("ok"))
        { FAIL(test_name, "ascii"); return; }
    if (nl_str_is_ascii("\xC3\xA9"))
        { FAIL(test_name, "not ascii"); return; }
    if (!nl_str_is_valid_utf8("\xC3\xA9"))
        { FAIL(test_name, "valid"); return; }
    if (nl_str_is_valid_utf8(junk))
        { FAIL(test_name, "invalid"); return; }
    PASS(test_name);
}

int main(void) {
    printf("Unicode FFI tests\n");
    test_byte_vs_codepoint_vs_grapheme();
    test_normalize_nfc_nfd();
    test_case_and_fold();
    test_display_width();
    test_ascii_and_utf8();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
