#include "runtime/module_build_dir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        snprintf(dest, dest_size, "%s/%s", cache, key);
        return true;
    }

    if (!module_dir || module_dir[0] == '\0') {
        return false;
    }
    snprintf(dest, dest_size, "%s/.build", module_dir);
    return true;
}
