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

#endif
