#ifndef NL_SHELL_PATH_H
#define NL_SHELL_PATH_H
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* I bound complete module object/shared links to the link-query grammar. */
#define NL_MODULE_LINK_COMMAND_CAPACITY 65537u

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

/* I grow the complete link closure without silently dropping later inputs. */
static inline bool module_append_fragment(char **buffer, const char *fragment) {
    size_t used = *buffer ? strlen(*buffer) : 0;
    size_t length = strlen(fragment);
    if (length > SIZE_MAX - used - 2) return false;
    char *grown = realloc(*buffer, used + length + 2);
    if (!grown) return false;
    if (used) grown[used++] = ' ';
    memcpy(grown + used, fragment, length + 1);
    *buffer = grown;
    return true;
}

static inline bool module_append_unique_object(char **buffer, const char *path) {
    char *quoted = module_quote_path(path);
    if (!quoted) return false;
    size_t length = strlen(quoted);
    for (const char *found = *buffer ? strstr(*buffer, quoted) : NULL; found; found = strstr(found + 1, quoted)) {
        if ((found == *buffer || found[-1] == ' ') &&
            (found[length] == '\0' || found[length] == ' ')) {
            free(quoted);
            return true;
        }
    }
    bool appended = module_append_fragment(buffer, quoted);
    free(quoted);
    return appended;
}
#endif
