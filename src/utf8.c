#include "utf8.h"

#include <stdint.h>
#include <string.h>

bool nl_utf8_validate(const char *data, size_t n, size_t *error_offset) {
    size_t i = 0;

    if (error_offset) *error_offset = 0;
    if (n == 0) return true;
    if (!data) return false;

    while (i < n) {
        unsigned char c = (unsigned char)data[i];
        size_t need = 0;
        unsigned char lo = 0x80;
        unsigned char hi = 0xBF;

        if (c <= 0x7F) {
            i++;
            continue;
        }

        if (c >= 0xC2 && c <= 0xDF) {
            need = 2;
        } else if (c == 0xE0) {
            need = 3;
            lo = 0xA0;
        } else if (c >= 0xE1 && c <= 0xEC) {
            need = 3;
        } else if (c == 0xED) {
            need = 3;
            hi = 0x9F;
        } else if (c == 0xEE || c == 0xEF) {
            need = 3;
        } else if (c == 0xF0) {
            need = 4;
            lo = 0x90;
        } else if (c >= 0xF1 && c <= 0xF3) {
            need = 4;
        } else if (c == 0xF4) {
            need = 4;
            hi = 0x8F;
        } else {
            if (error_offset) *error_offset = i;
            return false;
        }

        if (i + need > n) {
            if (error_offset) *error_offset = i;
            return false;
        }

        {
            unsigned char c1 = (unsigned char)data[i + 1];
            if (c1 < lo || c1 > hi) {
                if (error_offset) *error_offset = i;
                return false;
            }
        }
        {
            size_t k;
            for (k = 2; k < need; k++) {
                unsigned char ck = (unsigned char)data[i + k];
                if (ck < 0x80 || ck > 0xBF) {
                    if (error_offset) *error_offset = i;
                    return false;
                }
            }
        }
        i += need;
    }
    return true;
}

bool nl_utf8_ok_cstr(const char *s) {
    if (!s) return false;
    return nl_utf8_validate(s, strlen(s), NULL);
}

const char *nl_utf8_cstr_or_marker(const char *s) {
    if (!s) return "";
    if (nl_utf8_ok_cstr(s)) return s;
    return "<invalid UTF-8>";
}

int nl_ascii_isspace(int c) {
    unsigned char u = (unsigned char)c;
    if (c < 0) return 0;
    return u == ' ' || (u >= 0x09 && u <= 0x0D);
}

int nl_ascii_isdigit(int c) {
    return c >= '0' && c <= '9';
}

int nl_ascii_isalpha(int c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

int nl_ascii_isalnum(int c) {
    return nl_ascii_isalpha(c) || nl_ascii_isdigit(c);
}

int nl_ascii_isupper(int c) {
    return c >= 'A' && c <= 'Z';
}

int nl_ascii_islower(int c) {
    return c >= 'a' && c <= 'z';
}

int nl_ascii_isprint(int c) {
    return c >= 0x20 && c <= 0x7E;
}

int nl_ascii_toupper(int c) {
    if (c >= 'a' && c <= 'z') return c - ('a' - 'A');
    return c;
}

int nl_ascii_tolower(int c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    return c;
}

static int utf8_cp_at(const unsigned char *s, size_t n, size_t i, uint32_t *cp, size_t *adv) {
    unsigned char c;
    if (i >= n) return 0;
    c = s[i];
    if (c <= 0x7F) {
        *cp = c;
        *adv = 1;
        return 1;
    }
    if (c >= 0xC2 && c <= 0xDF && i + 1 < n) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | (s[i + 1] & 0x3F);
        *adv = 2;
        return 1;
    }
    if (c >= 0xE0 && c <= 0xEF && i + 2 < n) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
        *adv = 3;
        return 1;
    }
    if (c >= 0xF0 && c <= 0xF4 && i + 3 < n) {
        *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12)
            | ((uint32_t)(s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
        *adv = 4;
        return 1;
    }
    return 0;
}

static int is_bidi_control(uint32_t cp) {
    return cp == 0x061C || cp == 0x200E || cp == 0x200F
        || (cp >= 0x202A && cp <= 0x202E)
        || (cp >= 0x2066 && cp <= 0x2069);
}

void nl_utf8_sanitize_log(const char *s, char *out, size_t out_n) {
    const char *marker = "<invalid UTF-8>";
    const unsigned char *in;
    size_t n, i, o;

    if (!out || out_n == 0) return;
    out[0] = '\0';
    if (!s) return;
    if (!nl_utf8_ok_cstr(s)) {
        strncpy(out, marker, out_n - 1);
        out[out_n - 1] = '\0';
        return;
    }

    in = (const unsigned char *)s;
    n = strlen(s);
    i = 0;
    o = 0;
    while (i < n && o + 1 < out_n) {
        uint32_t cp = 0;
        size_t adv = 1;

        if (in[i] == 0x1B) {
            i++;
            if (i < n && in[i] == '[') {
                i++;
                while (i < n && !(in[i] >= 0x40 && in[i] <= 0x7E)) i++;
                if (i < n) i++;
            } else if (i < n && in[i] == ']') {
                i++;
                while (i < n && in[i] != 0x07) {
                    if (in[i] == 0x1B && i + 1 < n && in[i + 1] == '\\') {
                        i += 2;
                        break;
                    }
                    i++;
                }
                if (i < n && in[i] == 0x07) i++;
            }
            continue;
        }

        if (!utf8_cp_at(in, n, i, &cp, &adv)) break;
        if (is_bidi_control(cp)) {
            i += adv;
            continue;
        }
        if (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r') {
            i += adv;
            continue;
        }
        if (o + adv >= out_n) break;
        memcpy(out + o, s + i, adv);
        o += adv;
        i += adv;
    }
    out[o] = '\0';
}
