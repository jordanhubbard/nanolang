/* I share these generated-code helpers across my C emitters. */
#ifndef NANOLANG_STRING_EDGES_H
#define NANOLANG_STRING_EDGES_H

#include <stdbool.h>
#include <string.h>
#include "gc.h"

static inline bool nl_str_ends_with(const char *s, const char *suffix) {
    if (!s || !suffix) return false;
    size_t slen = strnlen(s, 64 * 1024 * 1024);
    size_t suflen = strnlen(suffix, 64 * 1024 * 1024);
    if (suflen > slen) return false;
    if (suflen == 0) return true;
    return strncmp(s + slen - suflen, suffix, suflen) == 0;
}

static inline const char *nl_str_trim(const char *s) {
    if (!s) return "";
    size_t len = strnlen(s, 64 * 1024 * 1024);
    size_t start = 0;
    while (start < len && (s[start] == ' ' || s[start] == '\t' ||
                          s[start] == '\n' || s[start] == '\r')) start++;
    size_t end = len;
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t' ||
                          s[end - 1] == '\n' || s[end - 1] == '\r')) end--;
    size_t new_len = end - start;
    char *result = gc_alloc_string(new_len);
    if (!result) return "";
    memcpy(result, s + start, new_len);
    result[new_len] = '\0';
    return result;
}
#endif
