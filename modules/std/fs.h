#ifndef NANOLANG_STD_FS_H
#define NANOLANG_STD_FS_H

#include <stdint.h>
#include <stdbool.h>
#include "../../src/runtime/dyn_array.h"

/* Walk directory tree recursively, returning all file paths */
DynArray* std_fs__fs_walkdir(const char* root);

/* Normalize path (resolve . and .., remove redundant slashes) */
const char* std_fs__path_normalize(const char* path);

/* Join two path components */
const char* std_fs__path_join(const char* a, const char* b);

/* Get basename of path */
const char* std_fs__path_basename(const char* path);

/* Get dirname of path */
const char* std_fs__path_dirname(const char* path);

/* Read file content as string */
char* std_fs__file_read(const char* path);

/* Write string to file */
int64_t std_fs__file_write(const char* path, const char* content);

/* Append string to file */
int64_t std_fs__file_append(const char* path, const char* content);

/* Check if file exists */
bool std_fs__file_exists(const char* path);

/* Delete file */
int64_t std_fs__file_delete(const char* path);

#endif /* NANOLANG_STD_FS_H */

