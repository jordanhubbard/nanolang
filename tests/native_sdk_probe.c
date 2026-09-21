#define _GNU_SOURCE 1
#include "runtime/module_build_dir.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void report(const char *name, const char *value) {
    printf("%s ", name);
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) printf("%02x", *p);
    putchar('\n');
}
int main(int argc, char **argv) {
    assert(argc == 4);
    char path[4096], before[sizeof(path)];
    memset(path, 0xa5, sizeof(path)); memcpy(before, path, sizeof(path));
    bool installed = false;
    NanoSdkStatus expected = (NanoSdkStatus)strtol(argv[2], NULL, 10);
    size_t capacity = !strcmp(argv[1], "short") ? 1 : sizeof(path);
    NanoSdkStatus status = nano_native_sdk_root(path, capacity, &installed);
    printf("STATUS %d\n", (int)status);
    assert(status == expected);
    if (status != NANO_SDK_OK) {
        assert(!memcmp(path, before, sizeof(path)) && !installed);
        puts("PASS unchanged root outputs");
        return 0;
    }
    assert(installed == (strcmp(argv[3], "installed") == 0));
    report("ROOT", path);
    if (!strcmp(argv[1], "objects") || !strcmp(argv[1], "lists")) {
        assert(nano_native_sdk_prepare() == NANO_SDK_OK);
        char first[4096], second[4096];
        assert(nano_native_module_objects_dir(first, sizeof(first)) == NANO_SDK_OK);
        assert(nano_native_module_objects_dir(second, sizeof(second)) == NANO_SDK_OK);
        assert(strcmp(first, second));
        struct stat st;
        assert(!stat(first, &st) && S_ISDIR(st.st_mode));
        assert(!stat(second, &st) && S_ISDIR(st.st_mode));
        report("FIRST", first); report("SECOND", second);
        const char *cache = getenv("NANO_BUILD_CACHE");
        report("CACHE", cache ? cache : "");
        if (!strcmp(argv[1], "lists")) {
            assert(!nano_native_generate_list(path, first, "invalid;name", "SdkPair"));
            assert(!nano_native_generate_list(path, first, "SdkPair", "invalid type"));
            assert(nano_native_generate_list(path, first, "SdkPair", "SdkPair"));
            nano_native_retain_private_work();
        }
    }
    puts("PASS native SDK root and private lifetime");
    return 0;
}
