#include "runtime/module_build_dir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <openssl/sha.h>

bool nano_module_build_dir(const char *module_dir, char *dest, size_t dest_size) {
    if (!dest || dest_size == 0) {
        return false;
    }
    dest[0] = '\0';
    if (!module_dir || !module_dir[0]) return false;

    const char *cache = getenv("NANO_BUILD_CACHE");
    if (cache && cache[0] != '\0') {
        char *canonical = realpath(module_dir, NULL);
        struct stat st;
        if (!canonical || stat(canonical, &st) != 0 || !S_ISDIR(st.st_mode)) {
            free(canonical);
            return false;
        }
        unsigned char digest[SHA256_DIGEST_LENGTH];
        bool hashed = SHA256((const unsigned char *)canonical, strlen(canonical), digest) != NULL;
        free(canonical);
        if (!hashed) return false;
        char key[SHA256_DIGEST_LENGTH * 2 + 1];
        const char *hex = "0123456789abcdef";
        for (size_t i = 0; i < sizeof(digest); i++) {
            key[i * 2] = hex[digest[i] >> 4];
            key[i * 2 + 1] = hex[digest[i] & 15];
        }
        key[sizeof(key) - 1] = '\0';
        /* I never read an ambiguous slash-to-underscore legacy namespace. */
        int n = snprintf(dest, dest_size, "%s/v2-%s", cache, key);
        if (n >= 0 && (size_t)n < dest_size) return true;
        dest[0] = '\0';
        return false;
    }

    int n = snprintf(dest, dest_size, "%s/.build", module_dir);
    if (n >= 0 && (size_t)n < dest_size) return true;
    dest[0] = '\0';
    return false;
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
