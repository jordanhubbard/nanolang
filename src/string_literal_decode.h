/* I share existing source escape decoding without changing its byte contract. */
#ifndef NANOLANG_STRING_LITERAL_DECODE_H
#define NANOLANG_STRING_LITERAL_DECODE_H
#include <stdlib.h>
#include <string.h>

static inline char *nl_decode_string_literal(const char *raw) {
    size_t len = strlen(raw);
    char *buf = malloc(len + 1);
    if (!buf) return NULL;
    size_t out = 0;
    for (size_t i = 0; i < len; i++) {
        if (raw[i] == '\\' && i + 1 < len) {
            i++;
            switch (raw[i]) {
                case 'n':  buf[out++] = '\n'; break;
                case 't':  buf[out++] = '\t'; break;
                case 'r':  buf[out++] = '\r'; break;
                case '0':  buf[out++] = '\0'; break;
                case '\\': buf[out++] = '\\'; break;
                case '\'': buf[out++] = '\''; break;
                case '"':  buf[out++] = '"';  break;
                default:   buf[out++] = '\\'; buf[out++] = raw[i]; break;
            }
        } else {
            buf[out++] = raw[i];
        }
    }
    buf[out] = '\0';
    return buf;
}
#endif
