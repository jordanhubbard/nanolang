#ifndef NANO_MODULE_BUILD_DIR_H
#define NANO_MODULE_BUILD_DIR_H

#include <stddef.h>
#include <stdbool.h>

/**
 * Directory where a module's compiled C artifacts live.
 *
 * If NANO_BUILD_CACHE is set (Makefile exports this as obj/module_cache),
 * artifacts go under $NANO_BUILD_CACHE/<sanitized_module_dir>.
 * Otherwise they go in <module_dir>/.build (the documented default).
 */
bool nano_module_build_dir(const char *module_dir, char *dest, size_t dest_size);

#endif
