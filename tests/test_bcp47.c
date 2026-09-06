#include "bcp47.h"
#include "locale.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_parse_initial_languages(void) {
    const char *test_name = "bcp47: parse en zh hi es ar fr";
    NlLocale loc;
    if (!nl_bcp47_parse("en", &loc) || strcmp(loc.language, "en") != 0)
        { FAIL(test_name, "en"); return; }
    if (loc.direction != NL_TEXT_LTR) { FAIL(test_name, "en ltr"); return; }
    if (!nl_bcp47_parse("zh-Hans-CN", &loc)
            || strcmp(loc.language, "zh") != 0
            || strcmp(loc.script, "Hans") != 0
            || strcmp(loc.region, "CN") != 0
            || strcmp(loc.tag, "zh-Hans-CN") != 0)
        { FAIL(test_name, "zh-Hans-CN"); return; }
    if (!nl_bcp47_parse("hi-IN", &loc) || strcmp(loc.region, "IN") != 0)
        { FAIL(test_name, "hi-IN"); return; }
    if (!nl_bcp47_parse("es-419", &loc) || strcmp(loc.region, "419") != 0)
        { FAIL(test_name, "es-419"); return; }
    if (!nl_bcp47_parse("ar", &loc) || loc.direction != NL_TEXT_RTL)
        { FAIL(test_name, "ar rtl"); return; }
    if (!nl_bcp47_parse("fr-FR", &loc) || strcmp(loc.language, "fr") != 0)
        { FAIL(test_name, "fr-FR"); return; }
    PASS(test_name);
}

static void test_reject_malformed(void) {
    const char *test_name = "bcp47: reject empty overlong and junk";
    NlLocale loc;
    if (nl_bcp47_parse("", &loc)) { FAIL(test_name, "empty"); return; }
    if (nl_bcp47_parse(NULL, &loc)) { FAIL(test_name, "NULL"); return; }
    if (nl_bcp47_parse("e", &loc)) { FAIL(test_name, "one letter"); return; }
    if (nl_bcp47_parse("en--US", &loc)) { FAIL(test_name, "double dash"); return; }
    if (nl_bcp47_parse("zh_Hans", &loc)) { FAIL(test_name, "underscore"); return; }
    PASS(test_name);
}

static void test_fallback_and_axes(void) {
    const char *test_name = "bcp47: fallback chain and separate axes";
    NlLocale loc;
    char chain[NL_BCP47_FALLBACK_MAX][NL_BCP47_TAG];
    int n;
    if (!nl_bcp47_parse("zh-Hans-CN", &loc)) { FAIL(test_name, "parse"); return; }
    n = nl_bcp47_fallback_chain(&loc, chain, NL_BCP47_FALLBACK_MAX);
    if (n != 4 || strcmp(chain[0], "zh-Hans-CN") != 0
            || strcmp(chain[1], "zh-Hans") != 0
            || strcmp(chain[2], "zh") != 0
            || strcmp(chain[3], "en") != 0)
        { FAIL(test_name, "zh fallback"); return; }
    if (strcmp(nl_bcp47_encoding_name(loc.encoding), "utf-8") != 0)
        { FAIL(test_name, "utf-8"); return; }
    if (strcmp(nl_bcp47_collation_name(loc.collation), "unicode") != 0)
        { FAIL(test_name, "unicode collation"); return; }
    if (strcmp(nl_bcp47_direction_name(loc.direction), "ltr") != 0)
        { FAIL(test_name, "zh ltr"); return; }
    loc.encoding = NL_ENCODING_BINARY;
    loc.collation = NL_COLLATION_BYTE;
    if (strcmp(nl_bcp47_encoding_name(loc.encoding), "binary") != 0)
        { FAIL(test_name, "binary"); return; }
    if (strcmp(nl_bcp47_collation_name(loc.collation), "byte") != 0)
        { FAIL(test_name, "byte collation"); return; }
    PASS(test_name);
}

static char *dup_env(const char *key) {
    const char *v = getenv(key);
    if (!v) return NULL;
    return strdup(v);
}

static void set_or_unset(const char *key, const char *value) {
    if (value)
        setenv(key, value, 1);
    else
        unsetenv(key);
}

static void clear_locale_env(void) {
    unsetenv("NANO_LOCALE");
    unsetenv("LC_ALL");
    unsetenv("LANG");
}

static void test_posix_to_bcp47(void) {
    const char *test_name = "locale: POSIX LANG to BCP 47";
    char tag[NL_BCP47_TAG];
    if (!nl_posix_locale_to_bcp47("C", tag, sizeof tag) || strcmp(tag, "en") != 0)
        { FAIL(test_name, "C"); return; }
    if (!nl_posix_locale_to_bcp47("C.UTF-8", tag, sizeof tag) || strcmp(tag, "en") != 0)
        { FAIL(test_name, "C.UTF-8"); return; }
    if (!nl_posix_locale_to_bcp47("POSIX", tag, sizeof tag) || strcmp(tag, "en") != 0)
        { FAIL(test_name, "POSIX"); return; }
    if (!nl_posix_locale_to_bcp47("POSIX.UTF-8", tag, sizeof tag) || strcmp(tag, "en") != 0)
        { FAIL(test_name, "POSIX.UTF-8"); return; }
    if (!nl_posix_locale_to_bcp47("en_US.UTF-8", tag, sizeof tag)
            || strcmp(tag, "en-US") != 0)
        { FAIL(test_name, "en_US.UTF-8"); return; }
    if (!nl_posix_locale_to_bcp47("zh_CN.UTF-8@euro", tag, sizeof tag)
            || strcmp(tag, "zh-CN") != 0)
        { FAIL(test_name, "zh_CN"); return; }
    if (nl_posix_locale_to_bcp47("", tag, sizeof tag))
        { FAIL(test_name, "empty"); return; }
    if (nl_posix_locale_to_bcp47(NULL, tag, sizeof tag))
        { FAIL(test_name, "NULL"); return; }
    PASS(test_name);
}

static void test_locale_resolve_order(void) {
    const char *test_name = "locale: resolve cli NANO_LOCALE LC_ALL LANG default";
    char *saved_nano = dup_env("NANO_LOCALE");
    char *saved_lc = dup_env("LC_ALL");
    char *saved_lang = dup_env("LANG");
    NlResolvedLocale r;

    clear_locale_env();
    if (!nl_locale_resolve(NULL, &r) || strcmp(r.locale.tag, "en") != 0
            || strcmp(r.source, "default") != 0)
        { FAIL(test_name, "default"); goto restore; }

    setenv("LANG", "fr_FR.UTF-8", 1);
    if (!nl_locale_resolve(NULL, &r) || strcmp(r.locale.tag, "fr-FR") != 0
            || strcmp(r.source, "LANG") != 0)
        { FAIL(test_name, "LANG"); goto restore; }

    setenv("LC_ALL", "es_MX.UTF-8", 1);
    if (!nl_locale_resolve(NULL, &r) || strcmp(r.locale.tag, "es-MX") != 0
            || strcmp(r.source, "LC_ALL") != 0)
        { FAIL(test_name, "LC_ALL"); goto restore; }

    setenv("NANO_LOCALE", "zh-Hans-CN", 1);
    if (!nl_locale_resolve(NULL, &r) || strcmp(r.locale.tag, "zh-Hans-CN") != 0
            || strcmp(r.source, "NANO_LOCALE") != 0)
        { FAIL(test_name, "NANO_LOCALE"); goto restore; }

    if (!nl_locale_resolve("ar", &r) || strcmp(r.locale.tag, "ar") != 0
            || strcmp(r.source, "cli") != 0
            || r.locale.direction != NL_TEXT_RTL)
        { FAIL(test_name, "cli"); goto restore; }

    if (nl_locale_resolve("en--US", &r))
        { FAIL(test_name, "invalid cli"); goto restore; }
    if (nl_locale_resolve("", &r))
        { FAIL(test_name, "empty cli"); goto restore; }

    clear_locale_env();
    setenv("NANO_LOCALE", "not-a-tag!!!!", 1);
    if (nl_locale_resolve(NULL, &r))
        { FAIL(test_name, "invalid NANO_LOCALE"); goto restore; }

    clear_locale_env();
    setenv("LANG", "!!!!", 1);
    if (!nl_locale_resolve(NULL, &r) || strcmp(r.locale.tag, "en") != 0
            || strcmp(r.source, "default") != 0)
        { FAIL(test_name, "garbage LANG"); goto restore; }

    PASS(test_name);
restore:
    set_or_unset("NANO_LOCALE", saved_nano);
    set_or_unset("LC_ALL", saved_lc);
    set_or_unset("LANG", saved_lang);
    free(saved_nano);
    free(saved_lc);
    free(saved_lang);
}

static void test_locale_format_axes(void) {
    const char *test_name = "locale: format prints separate axes";
    NlResolvedLocale r;
    char buf[1024];
    if (!nl_bcp47_parse("ar", &r.locale)) { FAIL(test_name, "parse"); return; }
    snprintf(r.source, sizeof r.source, "cli");
    nl_locale_format(&r, buf, sizeof buf);
    if (!strstr(buf, "tag: ar\n") || !strstr(buf, "language: ar\n")
            || !strstr(buf, "direction: rtl\n")
            || !strstr(buf, "encoding: utf-8\n")
            || !strstr(buf, "collation: unicode\n")
            || !strstr(buf, "source: cli\n")
            || !strstr(buf, "fallback: ar en\n"))
        { FAIL(test_name, buf); return; }
    PASS(test_name);
}

int main(void) {
    printf("BCP 47 locale tag tests\n");
    test_parse_initial_languages();
    test_reject_malformed();
    test_fallback_and_axes();
    test_posix_to_bcp47();
    test_locale_resolve_order();
    test_locale_format_axes();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
