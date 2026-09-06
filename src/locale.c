#include "locale.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void copy_src(char *dst, const char *s) {
    size_t i;
    for (i = 0; i + 1 < NL_LOCALE_SOURCE && s[i]; i++) dst[i] = s[i];
    dst[i] = '\0';
}

bool nl_posix_locale_to_bcp47(const char *posix, char *out, size_t out_sz) {
    size_t i, o;
    if (!posix || !out || out_sz < 3) return false;
    if (posix[0] == '\0') return false;

    if ((posix[0] == 'C' && (posix[1] == '\0' || posix[1] == '.' || posix[1] == '@'))
            || strcmp(posix, "POSIX") == 0
            || strncmp(posix, "POSIX.", 6) == 0
            || strncmp(posix, "POSIX@", 6) == 0) {
        snprintf(out, out_sz, "en");
        return true;
    }

    o = 0;
    for (i = 0; posix[i] && posix[i] != '.' && posix[i] != '@'; i++) {
        char c = posix[i];
        if (c == '_') c = '-';
        if (o + 1 >= out_sz) return false;
        out[o++] = c;
    }
    out[o] = '\0';
    return o >= 2;
}

static int fill_en(NlResolvedLocale *out, const char *source) {
    if (!nl_bcp47_parse("en", &out->locale)) return 0;
    copy_src(out->source, source);
    return 1;
}

bool nl_locale_resolve(const char *cli_tag, NlResolvedLocale *out) {
    const char *env;
    char converted[NL_BCP47_TAG];

    if (!out) return false;
    memset(out, 0, sizeof *out);

    if (cli_tag) {
        if (cli_tag[0] == '\0' || !nl_bcp47_parse(cli_tag, &out->locale))
            return false;
        copy_src(out->source, "cli");
        return true;
    }

    env = getenv("NANO_LOCALE");
    if (env && env[0] != '\0') {
        if (!nl_bcp47_parse(env, &out->locale)) return false;
        copy_src(out->source, "NANO_LOCALE");
        return true;
    }

    env = getenv("LC_ALL");
    if (env && env[0] != '\0' && nl_posix_locale_to_bcp47(env, converted, sizeof converted) &&
        nl_bcp47_parse(converted, &out->locale)) {
        copy_src(out->source, "LC_ALL");
        return true;
    }

    env = getenv("LANG");
    if (env && env[0] != '\0' && nl_posix_locale_to_bcp47(env, converted, sizeof converted) &&
        nl_bcp47_parse(converted, &out->locale)) {
        copy_src(out->source, "LANG");
        return true;
    }

    return fill_en(out, "default");
}

void nl_locale_format(const NlResolvedLocale *r, char *out, size_t out_sz) {
    char chain[NL_BCP47_FALLBACK_MAX][NL_BCP47_TAG];
    int n, i;
    char fallback[NL_BCP47_TAG * NL_BCP47_FALLBACK_MAX];
    size_t fp = 0;

    if (!r || !out || out_sz == 0) return;
    n = nl_bcp47_fallback_chain(&r->locale, chain, NL_BCP47_FALLBACK_MAX);
    fallback[0] = '\0';
    for (i = 0; i < n; i++) {
        int w = snprintf(fallback + fp, sizeof fallback - fp, "%s%s",
                         i ? " " : "", chain[i]);
        if (w < 0) break;
        fp += (size_t)w;
        if (fp >= sizeof fallback) break;
    }

    snprintf(out, out_sz,
             "tag: %s\n"
             "language: %s\n"
             "script: %s\n"
             "region: %s\n"
             "variant: %s\n"
             "direction: %s\n"
             "encoding: %s\n"
             "collation: %s\n"
             "source: %s\n"
             "fallback: %s\n",
             r->locale.tag,
             r->locale.language,
             r->locale.script,
             r->locale.region,
             r->locale.variant,
             nl_bcp47_direction_name(r->locale.direction),
             nl_bcp47_encoding_name(r->locale.encoding),
             nl_bcp47_collation_name(r->locale.collation),
             r->source,
             fallback);
}
