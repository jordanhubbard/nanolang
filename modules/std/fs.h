#ifndef NANOLANG_STD_FS_H
#define NANOLANG_STD_FS_H

#include <stdint.h>
#include "../../src/runtime/dyn_array.h"

/* Walk directory tree recursively, returning all file paths */
DynArray* fs_walkdir(const char* root);

/* Normalize path (resolve . and .., remove redundant slashes) */
const char* path_normalize(const char* path);

/* Join two path components */
const char* path_join(const char* a, const char* b);

/* Get basename of path */
const char* path_basename(const char* path);

/* Get dirname of path */
const char* path_dirname(const char* path);

#endif /* NANOLANG_STD_FS_H */

