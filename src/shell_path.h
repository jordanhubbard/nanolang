#ifndef NL_SHELL_PATH_H
#define NL_SHELL_PATH_H
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* I preserve paths as one literal shell word; compiler command configuration
 * remains a command fragment for compatibility with wrappers such as ccache. */
static inline char *module_quote_path(const char *path) {
    size_t length = strlen(path);
    if (length > (SIZE_MAX - 3) / 4) return NULL;
    char *quoted = malloc(length * 4 + 3);
    if (!quoted) return NULL;
    char *out = quoted;
    *out++ = '\'';
    for (const char *p = path; *p; p++) {
        if (*p == '\'') {
            memcpy(out, "'\\''", 4);
            out += 4;
        } else {
            *out++ = *p;
        }
    }
    *out++ = '\'';
    *out = '\0';
    return quoted;
}

static inline bool module_append_path_flag(char *buffer, size_t capacity, const char *prefix, const char *path) {
    char *quoted = module_quote_path(path);
    if (!quoted) return false;
    size_t used = strlen(buffer);
    int written = snprintf(buffer + used, capacity - used, " %s%s", prefix, quoted);
    free(quoted);
    return written >= 0 && (size_t)written < capacity - used;
}

static inline bool module_append_include(char *buffer, size_t capacity, const char *path) {
    return module_append_path_flag(buffer, capacity, "-I", path);
}

static inline bool module_append_unique_object(char *buffer, size_t capacity, const char *path) {
    char *quoted = module_quote_path(path);
    if (!quoted) return false;
    size_t length = strlen(quoted);
    for (const char *found = strstr(buffer, quoted); found; found = strstr(found + 1, quoted)) {
        if ((found == buffer || found[-1] == ' ') &&
            (found[length] == '\0' || found[length] == ' ')) {
            free(quoted);
            return true;
        }
    }
    size_t used = strlen(buffer);
    int written = snprintf(buffer + used, capacity - used, "%s%s", used ? " " : "", quoted);
    free(quoted);
    return written >= 0 && (size_t)written < capacity - used;
}
#endif
