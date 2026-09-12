#include "runtime/module_build_dir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

static void sanitize_module_dir(const char *module_dir, char *out, size_t out_size) {
    const char *p = module_dir ? module_dir : "";
    while (p[0] == '.' && p[1] == '/') {
        p += 2;
    }
    size_t oi = 0;
    for (; *p != '\0' && oi + 1 < out_size; p++) {
        char c = *p;
        if (c == '/') {
            c = '_';
        }
        out[oi++] = c;
    }
    out[oi] = '\0';
    if (oi == 0 && out_size > 0) {
        snprintf(out, out_size, "unknown");
    }
}

bool nano_module_build_dir(const char *module_dir, char *dest, size_t dest_size) {
    if (!dest || dest_size == 0) {
        return false;
    }

    const char *cache = getenv("NANO_BUILD_CACHE");
    if (cache && cache[0] != '\0') {
        char key[1024];
        sanitize_module_dir(module_dir, key, sizeof(key));
        int n = snprintf(dest, dest_size, "%s/%s", cache, key);
        return n >= 0 && (size_t)n < dest_size;
    }

    if (!module_dir || module_dir[0] == '\0') {
        return false;
    }
    int n = snprintf(dest, dest_size, "%s/.build", module_dir);
    return n >= 0 && (size_t)n < dest_size;
}

bool nano_module_artifact_dir(const char *module_dir, char *dest, size_t dest_size) {
    char root[2048], pointer[2048], name[64], generation[2048];
    if (!dest || !dest_size || !nano_module_build_dir(module_dir, root, sizeof(root))) return false;
    int n = snprintf(pointer, sizeof(pointer), "%s/current", root);
    if (n < 0 || (size_t)n >= sizeof(pointer)) return false;
    ssize_t length = readlink(pointer, name, sizeof(name) - 1);
    if (length < 0) {
        if (errno != ENOENT) return false;
        n = snprintf(dest, dest_size, "%s", root);
        return n >= 0 && (size_t)n < dest_size;
    }
    name[length] = '\0';
    if (length != 16 || strncmp(name, ".nano-gen-", 10) != 0 ||
        strspn(name + 10, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") != 6) return false;
    n = snprintf(generation, sizeof(generation), "%s/%s", root, name);
    if (n < 0 || (size_t)n >= sizeof(generation)) return false;
    struct stat st;
    if (lstat(generation, &st) != 0 || !S_ISDIR(st.st_mode)) return false;
    char *resolved = realpath(generation, NULL);
    if (!resolved) return false;
    n = snprintf(dest, dest_size, "%s", resolved);
    free(resolved);
    return n >= 0 && (size_t)n < dest_size;
}
