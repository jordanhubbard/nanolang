#ifndef NANO_MODULE_BUILD_DIR_H
#define NANO_MODULE_BUILD_DIR_H

#include <stddef.h>
#include <stdbool.h>

/**
 * I return the cache root, not a pinned artifact generation.
 *
 * If NANO_BUILD_CACHE is set (Makefile exports this as obj/module_cache),
 * I use $NANO_BUILD_CACHE/v2-<sha256(realpath(module_dir))> when configured,
 * otherwise <module_dir>/.build. Readers use nano_module_artifact_dir.
 * Shared-cache directory identity requires an existing module directory.
 * I reject an output buffer that cannot hold the complete path and NUL.
 */
bool nano_module_build_dir(const char *module_dir, char *dest, size_t dest_size);

/* I resolve current once to a retained generation. With no current pointer,
 * I expose the legacy flat directory for runtime compatibility. */
bool nano_module_artifact_dir(const char *module_dir, char *dest, size_t dest_size);

/* I discover immutable compiler inputs, not language or service authority.
 * I leave both outputs unchanged on failure. I count path bytes including the
 * terminating NUL against capacity. The caller serializes process environment
 * changes and retains normal C buffer lifetime/exclusion obligations. */
typedef enum {
    NANO_SDK_OK = 0, NANO_SDK_INVALID = 1, NANO_SDK_IO = 2,
    NANO_SDK_LIMIT = 3, NANO_SDK_MEMORY = 4
} NanoSdkStatus;
NanoSdkStatus nano_native_sdk_root(char *dest, size_t capacity, bool *installed);
/* I create installed private work and select it as the writable cache only
 * without an explicit NANO_BUILD_CACHE. I retain work through process shutdown. */
NanoSdkStatus nano_native_sdk_prepare(void);
/* I retain generated C module objects privately through the final native link. */
NanoSdkStatus nano_native_module_objects_dir(char *dest, size_t capacity);
/* I retain requested C diagnostics or an unexpected generated-source failure. */
void nano_native_retain_private_work(void);
/* I accept a caller-owned private directory and original C identifiers. I use
 * argv directly, so path bytes never become shell syntax. The caller's existing
 * compiler/shadow supervisor bounds the child process. */
bool nano_native_generate_list(const char *root, const char *directory,
                               const char *name, const char *c_type);
/* I remove only a private tree whose ownership the caller retains. */
void nano_native_remove_private_tree(const char *directory);

#endif
