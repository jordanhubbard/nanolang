/* I inject only a selected REPL save stream's write or close failure. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int selected(FILE *stream, const char *operation) {
    const char *mode = getenv("NANO_REPL_FAIL_SAVE");
    const char *target = getenv("NANO_REPL_FAIL_SAVE_TARGET");
    if (!mode || !target || strcmp(mode, operation)) return 0;
    char link[64], path[PATH_MAX];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fileno(stream));
    ssize_t bytes = readlink(link, path, sizeof path - 1);
    if (bytes < 0) return 0;
    path[bytes] = '\0'; size_t size = strlen(target);
    return !strncmp(path, target, size) && !strncmp(path + size, ".nano-repl-", 11);
}

size_t fwrite(const void *data, size_t size, size_t count, FILE *stream) {
    size_t (*original)(const void *, size_t, size_t, FILE *) = dlsym(RTLD_NEXT, "fwrite");
    if (selected(stream, "write")) { errno = ENOSPC; return count ? count - 1 : 0; }
    return original(data, size, count, stream);
}

int fclose(FILE *stream) {
    int (*original)(FILE *) = dlsym(RTLD_NEXT, "fclose");
    int fail = selected(stream, "close"), result = original(stream);
    if (fail) { errno = EIO; return EOF; }
    return result;
}
