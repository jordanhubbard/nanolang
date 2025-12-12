#ifndef NANOLANG_FILESYSTEM_H
#define NANOLANG_FILESYSTEM_H

#include <stdint.h>

// Check if array type already defined
#ifndef NL_ARRAY_T_DEFINED
#define NL_ARRAY_T_DEFINED
typedef struct {
    void** data;
    int64_t length;
    int64_t capacity;
} nl_array_t;
#endif

// List files in directory with optional extension filter
nl_array_t* nl_fs_list_files(const char* path, const char* extension);

// Check if path is a directory
int64_t nl_fs_is_directory(const char* path);

// Check if file exists
int64_t nl_fs_file_exists(const char* path);

// Get file size
int64_t nl_fs_file_size(const char* path);

// Join path components
const char* nl_fs_join_path(const char* dir, const char* filename);

#endif // NANOLANG_FILESYSTEM_H
