#ifndef NL_STRING_SEARCH_H
#define NL_STRING_SEARCH_H

#include <stdint.h>
#include <string.h>

/* NUL-terminated byte search shared by generated C and runtime wrappers. */
static inline int64_t nl_str_index_of(const char *haystack, const char *needle) {
    if (!haystack || !needle) return -1;
    const char *match = strstr(haystack, needle);
    return match ? (int64_t)(match - haystack) : -1;
}

static inline int64_t nl_str_last_index_of(const char *haystack, const char *needle) {
    if (!haystack || !needle) return -1;

    size_t haystack_len = strlen(haystack);
    size_t needle_len = strlen(needle);
    if (needle_len == 0) return (int64_t)haystack_len;
    if (needle_len > haystack_len) return -1;

    for (size_t offset = haystack_len - needle_len + 1; offset-- > 0;) {
        if (memcmp(haystack + offset, needle, needle_len) == 0) {
            return (int64_t)offset;
        }
    }
    return -1;
}

#endif /* NL_STRING_SEARCH_H */
