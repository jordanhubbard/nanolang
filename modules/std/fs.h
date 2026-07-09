#ifndef NANOLANG_STD_FS_H
#define NANOLANG_STD_FS_H

#include <stdint.h>
#include <stdbool.h>
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

/* Compute relative path from base to target */
const char* path_relpath(const char* target, const char* base);

/* Read file content as string */
const char* file_read(const char* path);

/* Write string to file */
int64_t file_write(const char* path, const char* content);

/* Append string to file */
int64_t file_append(const char* path, const char* content);

/* Check if file exists */
bool file_exists(const char* path);

/* Delete file */
int64_t file_delete(const char* path);

/* Create directory and parents (mkdir -p) */
int64_t fs_mkdir_p(const char* path);

/* Copy a single file (binary-safe) */
int64_t file_copy(const char* src, const char* dst);

/* Copy a directory tree recursively */
int64_t dir_copy(const char* src, const char* dst);

#endif /* NANOLANG_STD_FS_H */

