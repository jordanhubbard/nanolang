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

// Get parent directory of path. Returns a pointer to an internal static buffer.
const char* nl_fs_parent_dir(const char* path);

// Check if path is a directory
int64_t nl_fs_is_directory(const char* path);

// Check if file exists
int64_t nl_fs_file_exists(const char* path);

// Get file size
int64_t nl_fs_file_size(const char* path);

// Join path components
const char* nl_fs_join_path(const char* dir, const char* filename);

#endif // NANOLANG_FILESYSTEM_H
