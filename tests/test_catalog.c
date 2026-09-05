#include "catalog.h"
#include "diag_id.h"
#include "utf8.h"
#include "locale.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_en_catalog(void) {
    const char *test_name = "catalog: load en and look up CIO01";
    if (!nl_catalog_load_path("catalogs/messages/en.json"))
        { FAIL(test_name, "load"); return; }
    if (strcmp(nl_catalog_get("CIO01"), "Could not open input file") != 0)
        { FAIL(test_name, "CIO01"); nl_catalog_clear(); return; }
    if (strcmp(nl_catalog_lang(), "en") != 0)
        { FAIL(test_name, "lang"); nl_catalog_clear(); return; }
    if (nl_catalog_missing_count() != 0)
        { FAIL(test_name, "missing"); nl_catalog_clear(); return; }
    PASS(test_name);
    nl_catalog_clear();
}

static void test_zh_and_fallback(void) {
    const char *test_name = "catalog: zh lookup and English fallback for missing";
    if (!nl_catalog_load_path("catalogs/messages/zh.json"))
        { FAIL(test_name, "load"); return; }
    if (!nl_catalog_get("CSRC01") || strstr(nl_catalog_get("CSRC01"), "UTF-8") == NULL)
        { FAIL(test_name, "zh CSRC01"); nl_catalog_clear(); return; }
    if (strcmp(nl_catalog_text("NOPE"), "NOPE") != 0)
        { FAIL(test_name, "unknown"); nl_catalog_clear(); return; }
    if (nl_catalog_missing_count() < 1)
        { FAIL(test_name, "missing count"); nl_catalog_clear(); return; }
    PASS(test_name);
    nl_catalog_clear();
}

static void test_invalid_utf8_catalog(void) {
    const char *test_name = "catalog: reject invalid UTF-8 JSON";
    const char *path = "/tmp/nl_bad_catalog.json";
    FILE *fp = fopen(path, "wb");
    if (!fp) { FAIL(test_name, "open"); return; }
    fputs("{\"CIO01\":\"bad \xff\"}", fp);
    fclose(fp);
    if (nl_catalog_load_path(path))
        { FAIL(test_name, "accepted junk"); unlink(path); return; }
    unlink(path);
    PASS(test_name);
}

static void test_locale_chain(void) {
    const char *test_name = "catalog: load_locale zh-Hans-CN falls to zh.json";
    NlResolvedLocale loc;
    if (!nl_locale_resolve("zh-Hans-CN", &loc))
        { FAIL(test_name, "resolve"); return; }
    if (!nl_catalog_load_locale(&loc, "catalogs/messages"))
        { FAIL(test_name, "load"); return; }
    if (strcmp(nl_catalog_lang(), "zh") != 0)
        { FAIL(test_name, nl_catalog_lang()); nl_catalog_clear(); return; }
    PASS(test_name);
    nl_catalog_clear();
}

static void test_format_reorder_plural_quote_list(void) {
    const char *test_name = "catalog: format reorder plural quote list date number";
    char buf[256];
    const char *args[] = { "2", "file", NULL };
    const char *one[] = { "1", NULL };
    const char *quoted[] = { "x", NULL };
    const char *listed[] = { "a|b|c", NULL };
    const char *date[] = { "2026-09-05", NULL };

    nl_catalog_format(buf, sizeof buf, "{1} {0}", "en", args);
    if (strcmp(buf, "file 2") != 0)
        { FAIL(test_name, buf); return; }

    nl_catalog_format(buf, sizeof buf, "{0,plural,one{one file}other{many files}}", "en", one);
    if (strcmp(buf, "one file") != 0)
        { FAIL(test_name, buf); return; }

    nl_catalog_format(buf, sizeof buf, "{0,plural,one{one file}other{many files}}", "en", args);
    if (strcmp(buf, "many files") != 0)
        { FAIL(test_name, buf); return; }

    nl_catalog_format(buf, sizeof buf, "{0,plural,other{个}}", "zh", args);
    if (strcmp(buf, "个") != 0)
        { FAIL(test_name, buf); return; }

    nl_catalog_format(buf, sizeof buf, "{0,quote}", "en", quoted);
    if (strstr(buf, "x") == NULL)
        { FAIL(test_name, buf); return; }

    nl_catalog_format(buf, sizeof buf, "{0,list}", "en", listed);
    if (!strstr(buf, "a") || !strstr(buf, "and") || !strstr(buf, "c"))
        { FAIL(test_name, buf); return; }

    nl_catalog_format(buf, sizeof buf, "{0,date} {0,number}", "en", date);
    if (strcmp(buf, "2026-09-05 2026-09-05") != 0)
        { FAIL(test_name, buf); return; }

    PASS(test_name);
}

static int catalog_key_count(const char *path, char keys[][16], int max) {
    /* Re-load via API and walk known ids. */
    const char *ids[] = {
        "CIO01","CLEX01","CPARSE01","CIMPORT01","CTYPE01","CMOD01","CSHADOW01",
        "CTRANS01","CC01","CCC02","CCC01","CSRC01","CAT01","L0003","L0004",
        "L0005","L0006","L0007","L0008","P0001","P0002","LOG01","LOG02","LOG03","LOG04",
        NULL
    };
    int n = 0;
    int i;
    if (!nl_catalog_load_path(path)) return -1;
    for (i = 0; ids[i] && n < max; i++) {
        if (!nl_catalog_get(ids[i])) {
            nl_catalog_clear();
            return -2;
        }
        snprintf(keys[n], 16, "%s", ids[i]);
        n++;
    }
    nl_catalog_clear();
    return n;
}

static void test_completeness(void) {
    const char *test_name = "catalog: six languages share the English key set";
    const char *langs[] = { "en", "zh", "hi", "es", "ar", "fr" };
    char keys[32][16];
    int expect;
    int i;
    char path[64];

    snprintf(path, sizeof path, "catalogs/messages/en.json");
    expect = catalog_key_count(path, keys, 32);
    if (expect < 20)
        { FAIL(test_name, "en keys"); return; }
    for (i = 0; i < 6; i++) {
        int n;
        snprintf(path, sizeof path, "catalogs/messages/%s.json", langs[i]);
        n = catalog_key_count(path, keys, 32);
        if (n != expect)
            { FAIL(test_name, langs[i]); return; }
        if (!nl_utf8_ok_cstr(path))
            { FAIL(test_name, "path"); return; }
    }
    PASS(test_name);
}

int main(void) {
    printf("Message catalog tests\n");
    test_en_catalog();
    test_zh_and_fallback();
    test_invalid_utf8_catalog();
    test_locale_chain();
    test_format_reorder_plural_quote_list();
    test_completeness();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    nl_catalog_clear();
    return g_fail != 0;
}
