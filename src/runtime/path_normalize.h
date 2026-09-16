#ifndef NANOLANG_PATH_NORMALIZE_H
#define NANOLANG_PATH_NORMALIZE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* I return owned lexical normalization, or NULL on allocation/size failure.
 * I do not resolve symlinks or consult the filesystem. */
static inline char *nl_normalize_path(const char *path) {
    if (!path) path = "";
    size_t length = strlen(path), slots = length / 2 + 1;
    if (length > SIZE_MAX - 2 || slots > SIZE_MAX / sizeof(size_t)) return NULL;
    char *out = malloc(length + 2);
    size_t *bases = malloc(slots * sizeof *bases);
    if (!out || !bases) { free(out); free(bases); return NULL; }
    int absolute = path[0] == '/';
    size_t used = 0, count = 0, cursor = 0;
    if (absolute) out[used++] = '/';
    while (cursor < length) {
        if (path[cursor] == '/') { ++cursor; continue; }
        size_t start = cursor;
        while (cursor < length && path[cursor] != '/') ++cursor;
        size_t size = cursor - start;
        if (size == 1 && path[start] == '.') continue;
        if (size == 2 && path[start] == '.' && path[start + 1] == '.') {
            if (count) {
                size_t previous = bases[count - 1];
                if (out[previous] == '/') ++previous;
                if (!(used - previous == 2 && out[previous] == '.' && out[previous + 1] == '.')) {
                    used = bases[--count]; continue;
                }
            }
            if (absolute) continue;
        }
        if (count >= slots) { free(out); free(bases); return NULL; }
        bases[count++] = used;
        if (used && out[used - 1] != '/') out[used++] = '/';
        memcpy(out + used, path + start, size); used += size;
    }
    if (!used) out[used++] = '.';
    out[used] = 0;
    free(bases);
    return out;
}

#endif
