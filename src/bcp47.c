#include "bcp47.h"
#include "utf8.h"

#include <string.h>

static int is_alpha(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static int is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static void lower_copy(char *dst, size_t dst_sz, const char *src, size_t n) {
    size_t i;
    if (n >= dst_sz) n = dst_sz - 1;
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)src[i];
        dst[i] = (char)nl_ascii_tolower((int)c);
    }
    dst[n] = '\0';
}

static void title_copy(char *dst, size_t dst_sz, const char *src, size_t n) {
    size_t i;
    if (n >= dst_sz) n = dst_sz - 1;
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)src[i];
        dst[i] = (i == 0) ? (char)nl_ascii_toupper((int)c) : (char)nl_ascii_tolower((int)c);
    }
    dst[n] = '\0';
}

static void upper_copy(char *dst, size_t dst_sz, const char *src, size_t n) {
    size_t i;
    if (n >= dst_sz) n = dst_sz - 1;
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)src[i];
        dst[i] = (char)nl_ascii_toupper((int)c);
    }
    dst[n] = '\0';
}

static int subtag_alpha(const char *s, size_t n) {
    size_t i;
    if (n == 0) return 0;
    for (i = 0; i < n; i++) {
        if (!is_alpha((unsigned char)s[i])) return 0;
    }
    return 1;
}

static int subtag_alnum(const char *s, size_t n) {
    size_t i;
    if (n == 0) return 0;
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (!is_alpha(c) && !is_digit(c)) return 0;
    }
    return 1;
}

static NlTextDirection direction_for_language(const char *lang) {
    if (strcmp(lang, "ar") == 0 || strcmp(lang, "he") == 0
            || strcmp(lang, "fa") == 0 || strcmp(lang, "ur") == 0)
        return NL_TEXT_RTL;
    return NL_TEXT_LTR;
}

static void rebuild_tag(NlLocale *out) {
    size_t used = 0;
    out->tag[0] = '\0';
    if (out->language[0] == '\0') return;
    used = (size_t)strlen(out->language);
    memcpy(out->tag, out->language, used + 1);
    if (out->script[0] != '\0') {
        if (used + 1 + strlen(out->script) >= NL_BCP47_TAG) return;
        out->tag[used] = '-';
        memcpy(out->tag + used + 1, out->script, strlen(out->script) + 1);
        used = strlen(out->tag);
    }
    if (out->region[0] != '\0') {
        if (used + 1 + strlen(out->region) >= NL_BCP47_TAG) return;
        out->tag[used] = '-';
        memcpy(out->tag + used + 1, out->region, strlen(out->region) + 1);
        used = strlen(out->tag);
    }
    if (out->variant[0] != '\0') {
        if (used + 1 + strlen(out->variant) >= NL_BCP47_TAG) return;
        out->tag[used] = '-';
        memcpy(out->tag + used + 1, out->variant, strlen(out->variant) + 1);
    }
}

static void copy_tag(char *dst, const char *src) {
    size_t n = strlen(src);
    if (n >= NL_BCP47_TAG) n = NL_BCP47_TAG - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

bool nl_bcp47_parse(const char *tag, NlLocale *out) {
    const char *p;
    const char *part;
    size_t n;
    int idx;

    if (!tag || !out) return false;
    memset(out, 0, sizeof(*out));
    out->encoding = NL_ENCODING_UTF8;
    out->collation = NL_COLLATION_UNICODE;
    if (tag[0] == '\0' || strlen(tag) >= NL_BCP47_TAG) return false;

    p = tag;
    idx = 0;
    while (*p) {
        part = p;
        while (*p && *p != '-') p++;
        n = (size_t)(p - part);
        if (n == 0 || n >= NL_BCP47_SUBTAG) return false;
        if (idx == 0) {
            if (n < 2 || n > 8 || !subtag_alpha(part, n)) return false;
            lower_copy(out->language, sizeof(out->language), part, n);
        } else if (idx == 1 && n == 4 && subtag_alpha(part, n)) {
            title_copy(out->script, sizeof(out->script), part, n);
        } else if ((idx == 1 || idx == 2) && out->region[0] == '\0'
                   && ((n == 2 && subtag_alpha(part, n))
                       || (n == 3 && is_digit((unsigned char)part[0])
                           && is_digit((unsigned char)part[1])
                           && is_digit((unsigned char)part[2])))) {
            if (n == 2) upper_copy(out->region, sizeof(out->region), part, n);
            else lower_copy(out->region, sizeof(out->region), part, n);
        } else if (out->variant[0] == '\0' && n >= 4 && n <= 8
                   && subtag_alnum(part, n)) {
            lower_copy(out->variant, sizeof(out->variant), part, n);
        } else {
            return false;
        }
        idx++;
        if (*p == '-') p++;
    }
    if (out->language[0] == '\0') return false;
    out->direction = direction_for_language(out->language);
    rebuild_tag(out);
    return true;
}

int nl_bcp47_fallback_chain(const NlLocale *loc, char out[][NL_BCP47_TAG],
                            int max_out) {
    char cur[NL_BCP47_TAG];
    char *dash;
    int n = 0;
    int has_en = 0;

    if (!loc || !out || max_out <= 0 || loc->tag[0] == '\0') return 0;
    copy_tag(cur, loc->tag);
    while (n < max_out) {
        copy_tag(out[n], cur);
        if (strcmp(out[n], "en") == 0) has_en = 1;
        n++;
        dash = strrchr(cur, '-');
        if (!dash) break;
        *dash = '\0';
    }
    if (!has_en && n < max_out && strcmp(loc->language, "en") != 0) {
        copy_tag(out[n], "en");
        n++;
    }
    return n;
}

const char *nl_bcp47_encoding_name(NlTextEncoding encoding) {
    if (encoding == NL_ENCODING_BINARY) return "binary";
    return "utf-8";
}

const char *nl_bcp47_collation_name(NlCollation collation) {
    if (collation == NL_COLLATION_BYTE) return "byte";
    return "unicode";
}

const char *nl_bcp47_direction_name(NlTextDirection direction) {
    if (direction == NL_TEXT_RTL) return "rtl";
    return "ltr";
}
