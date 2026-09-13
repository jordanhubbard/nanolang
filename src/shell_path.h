#ifndef NANO_SHELL_PATH_H
#define NANO_SHELL_PATH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static inline bool shell_append_text(char *dst, size_t cap, size_t *pos,
                                     const char *text) {
    if (!dst || !pos || !text || *pos >= cap) return false;
    size_t len = strlen(text);
    if (len >= cap - *pos) return false;
    memcpy(dst + *pos, text, len + 1);
    *pos += len;
    return true;
}

/* Append one literal POSIX-shell word, including empty strings. */
static inline bool shell_append_word(char *dst, size_t cap, size_t *pos,
                                     const char *word) {
    if (!shell_append_text(dst, cap, pos, " '")) return false;
    for (const char *p = word ? word : ""; *p; p++) {
        if (*p == '\'') {
            if (!shell_append_text(dst, cap, pos, "'\\''")) return false;
        } else {
            char byte[2] = {*p, '\0'};
            if (!shell_append_text(dst, cap, pos, byte)) return false;
        }
    }
    return shell_append_text(dst, cap, pos, "'");
}

static inline bool shell_append_joined_word(char *dst, size_t cap, size_t *pos,
                                            const char *prefix,
                                            const char *value) {
    char joined[4096];
    int n = snprintf(joined, sizeof(joined), "%s%s", prefix ? prefix : "",
                     value ? value : "");
    return n >= 0 && (size_t)n < sizeof(joined) &&
           shell_append_word(dst, cap, pos, joined);
}

#endif
