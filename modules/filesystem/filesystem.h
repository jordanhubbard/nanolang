#ifndef NANOLANG_FILESYSTEM_H
#define NANOLANG_FILESYSTEM_H

#include <stdint.h>
#include "../../src/runtime/dyn_array.h"

// List files in directory with optional extension filter
DynArray* nl_fs_list_files(const char* path, const char* extension);

// List files in directory with optional extension filter (case-insensitive)
DynArray* nl_fs_list_files_ci(const char* path, const char* extension);

// List directories in directory
DynArray* nl_fs_list_dirs(const char* path);

// I consume one exclusive string listing after its values have been copied.
bool nl_fs_list_release(DynArray *result);

// I return a borrowed static parent path, "." for null/empty input, or NULL
// for inputs of 2048 bytes or longer. My static path buffers are not thread-safe.
const char* nl_fs_parent_dir(const char* path);

// I return 0 for null, unavailable, or non-directory paths.
int64_t nl_fs_is_directory(const char* path);

// I return 0 for null or unavailable paths.
int64_t nl_fs_file_exists(const char* path);

// I return -1 for null or unavailable paths.
int64_t nl_fs_file_size(const char* path);

// I join null/empty components as empty strings in a borrowed static buffer.
// I return NULL if the joined path exceeds 2047 bytes, without truncation.
const char* nl_fs_join_path(const char* dir, const char* filename);

#endif // NANOLANG_FILESYSTEM_H
