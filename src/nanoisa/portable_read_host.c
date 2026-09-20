#include "portable_read_host.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t length;
    uint8_t bytes[NPR_PATH_LIMIT + 1u];
} NprStoredPath;
struct NprFileHost {
    uint32_t count;
    bool active;
    NprStoredPath paths[NPR_ALLOWLIST_LIMIT];
};

/* My sole retained allocation is exactly sizeof(NprFileHost). */
NprStatus npr_file_host_create(const NprPath *paths, uint32_t count,
                               NprFileHost **out) {
    if (!out || (count && !paths)) return NPR_INVALID;
    if (count > NPR_ALLOWLIST_LIMIT) return NPR_LIMIT;
    for (uint32_t i = 0; i < count; ++i) {
        if (!paths[i].length || !paths[i].data) return NPR_INVALID;
        if (paths[i].length > NPR_PATH_LIMIT) return NPR_LIMIT;
        if (memchr(paths[i].data, 0, paths[i].length)) return NPR_INVALID;
    }
    NprFileHost *host = calloc(1, sizeof(*host));
    if (!host) return NPR_MEMORY;
    host->count = count;
    for (uint32_t i = 0; i < count; ++i) {
        host->paths[i].length = paths[i].length;
        memcpy(host->paths[i].bytes, paths[i].data, paths[i].length);
    }
    *out = host;
    return NPR_OK;
}

NprStatus npr_file_host_destroy(NprFileHost *host) {
    if (!host) return NPR_OK;
    if (host->active) return NPR_INVALID;
    free(host);
    return NPR_OK;
}

typedef struct { uintptr_t start, end; } NprRange;
static bool npr_range(const void *pointer, size_t length, NprRange *out) {
    uintptr_t start = (uintptr_t)pointer;
    if ((length && !pointer) || length > UINTPTR_MAX - start) return false;
    out->start = start;
    out->end = start + length;
    return true;
}
static bool npr_overlap(NprRange a, NprRange b) {
    return a.start != a.end && b.start != b.end &&
           a.start < b.end && b.start < a.end;
}

int32_t npr_file_read(void *context, const uint8_t *path, uint32_t path_length,
                      uint8_t *destination, uint32_t capacity,
                      uint32_t *length_out) {
    NprFileHost *host = context;
    if (!host) return NPR_DENIED;
    if (path_length > NPR_PATH_LIMIT || capacity > NPR_TEXT_LIMIT) return NPR_LIMIT;
    if (!length_out) return NPR_INVALID;
    NprRange regions[4];
    if (!npr_range(path, path_length, &regions[0]) ||
        !npr_range(destination, capacity, &regions[1]) ||
        !npr_range(length_out, sizeof(*length_out), &regions[2]) ||
        !npr_range(host, sizeof(*host), &regions[3])) return NPR_INVALID;
    for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = i + 1; j < 4; ++j)
            if (npr_overlap(regions[i], regions[j])) return NPR_INVALID;
    if (host->active) return NPR_INVALID;
    if (path_length && memchr(path, 0, path_length)) return NPR_INVALID;
    bool allowed = false;
    for (uint32_t i = 0; i < host->count; ++i) {
        if (host->paths[i].length == path_length &&
            path_length && !memcmp(host->paths[i].bytes, path, path_length)) {
            allowed = true;
            break;
        }
    }
    if (!allowed) return NPR_DENIED;
    char terminated[NPR_PATH_LIMIT + 1u];
    memcpy(terminated, path, path_length);
    terminated[path_length] = 0;
    host->active = true;
    FILE *file = fopen(terminated, "rb");
    NprStatus status = NPR_OK;
    uint32_t length = 0;
    bool failed = false;
    if (file) {
        while (length < capacity) {
            size_t got = fread(destination + length, 1, capacity - length, file);
            length += (uint32_t)got;
            if (ferror(file)) { failed = true; break; }
            if (feof(file)) break;
            /* A stream that makes no progress is not successful partial text. */
            if (!got) { failed = true; break; }
        }
        if (!failed && length == capacity) {
            int excess = fgetc(file);
            if (excess != EOF) status = NPR_LIMIT;
            else if (ferror(file)) failed = true;
        }
        /* I always close once; a prior LIMIT survives a close error. */
        if (fclose(file) != 0) failed = true;
    }
    host->active = false;
    if (status != NPR_OK) return status;
    if (failed || (length && memchr(destination, 0, length))) length = 0;
    *length_out = length;
    return NPR_OK;
}
