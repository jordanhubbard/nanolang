/* I return byte offsets in NUL-terminated strings, not Unicode positions. */
#ifndef NANOLANG_STRING_SEARCH_H
#define NANOLANG_STRING_SEARCH_H

#include <stdint.h>
#include <string.h>

static inline int64_t nl_str_index_of(const char* haystack, const char* needle) {
    if (!haystack || !needle) return -1;
    const char* match = strstr(haystack, needle);
    return match ? (int64_t)(match - haystack) : -1;
}

static inline int64_t nl_str_last_index_of(const char* haystack, const char* needle) {
    if (!haystack || !needle) return -1;
    size_t length = strlen(haystack), wanted = strlen(needle);
    if (wanted > length || length > INT64_MAX) return -1;
    size_t position = length - wanted;
    for (;;) {
        if (memcmp(haystack + position, needle, wanted) == 0)
            return (int64_t)position;
        if (position == 0) return -1;
        --position;
    }
}

#endif
