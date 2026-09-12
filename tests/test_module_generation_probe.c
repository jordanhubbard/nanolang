/* I expose production cache paths so tests can hold them across publication. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#include "module_builder.h"
#include "runtime/module_build_dir.h"
#include "runtime/ffi_loader.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* I inject a final-pointer rename failure without damaging the old pointer.
 * This hook is confined to the test translation unit. */
static int generation_test_rename(const char *source, const char *target) {
    const char *leaf = strrchr(target, '/');
    if (leaf && strcmp(leaf, "/current") == 0 && getenv("NANO_TEST_POINTER_FAILURE")) {
        errno = EIO;
        return -1;
    }
    return rename(source, target);
}
#define rename generation_test_rename
#include "../src/module_builder.c"
#undef rename

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    if (strcmp(argv[1], "directory") == 0) {
        char path[2048];
        if (!nano_module_artifact_dir(argv[2], path, sizeof(path))) return 1;
        puts(path);
        return 0;
    }
    ModuleBuildMetadata *meta = module_load_metadata(argv[2]);
    if (!meta) return 1;
    int status = 1;
    if (strcmp(argv[1], "build") == 0) {
        ModuleBuildInfo *info = module_build(NULL, meta);
        if (info && info->object_file) {
            puts(info->object_file);
            status = 0;
        }
        module_build_info_free(info);
    } else if (strcmp(argv[1], "library") == 0) {
        char path[2048];
        if (ffi_loader_find_library(meta->name, argv[2], path, sizeof(path))) {
            puts(path);
            status = 0;
        }
    }
    module_metadata_free(meta);
    return status;
}
