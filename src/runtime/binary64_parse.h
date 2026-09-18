#ifndef NL_RUNTIME_BINARY64_PARSE_H
#define NL_RUNTIME_BINARY64_PARSE_H
#include "../nanoisa/binary64_parse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* I expose the same prefix value and checked endpoint to legacy C-string
 * consumers. Strict callers decide whether the entire string was consumed. */
static inline int nl_binary64_parse(const char *text, double *out, uint32_t *consumed) {
    if (!text || !out) return 0;
    size_t length = strlen(text);
    uint64_t bits;
    uint32_t end;
    if (length > UINT32_MAX ||
        !nbp_parse_end((const unsigned char *)text, (uint32_t)length, &bits, &end)) return 0;
    memcpy(out, &bits, sizeof bits);
    if (consumed) *consumed = end;
    return 1;
}
static inline double nl_binary64_prefix(const char *text) {
    double result;
    if (!nl_binary64_parse(text, &result, NULL)) {
        fputs("I cannot finish the checked binary64 conversion.\n", stderr);
        exit(EXIT_FAILURE);
    }
    return result;
}
#endif
