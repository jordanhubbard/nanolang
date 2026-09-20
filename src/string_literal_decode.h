/* I share source escape decoding and its complete byte count. */
#ifndef NANOLANG_STRING_LITERAL_DECODE_H
#define NANOLANG_STRING_LITERAL_DECODE_H
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static inline bool nl_string_escape_byte(char code, char *out) {
    switch (code) {
        case 'n': *out = '\n'; return true;
        case 't': *out = '\t'; return true;
        case 'r': *out = '\r'; return true;
        case '0': *out = '\0'; return true;
        case '\\': case '\'': case '"': *out = code; return true;
        default: return false;
    }
}

static inline size_t nl_string_literal_value_bytes(const char *raw) {
    size_t out = 0;
    for (size_t i = 0; raw[i]; i++) {
        char byte;
        if (raw[i] == '\\' && raw[i + 1] &&
            nl_string_escape_byte(raw[i + 1], &byte)) i++;
        else if (raw[i] == '\\' && raw[i + 1]) { i++; out++; }
        out++;
    }
    return out;
}

static inline char *nl_decode_string_literal(const char *raw) {
    size_t len = strlen(raw);
    char *buf = malloc(len + 1);
    if (!buf) return NULL;
    size_t out = 0;
    for (size_t i = 0; i < len; i++) {
        char byte = raw[i];
        if (byte == '\\' && i + 1 < len) {
            char code = raw[++i];
            if (!nl_string_escape_byte(code, &byte)) {
                buf[out++] = '\\'; byte = code;
            }
        }
        buf[out++] = byte;
    }
    buf[out] = '\0';
    return buf;
}
#endif
