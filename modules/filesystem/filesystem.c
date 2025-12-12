#include "filesystem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

// Helper: Create nanolang array
static nl_array_t* create_array(int64_t initial_capacity) {
    nl_array_t* arr = (nl_array_t*)malloc(sizeof(nl_array_t));
    if (!arr) return NULL;
    
    arr->length = 0;
    arr->capacity = initial_capacity;
    arr->elem_type = 3;  // ELEM_STRING = 3
    arr->elem_size = sizeof(char*);
    arr->data = calloc(initial_capacity, sizeof(char*));
    
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    
    return arr;
}

// Helper: Append string to array
static void array_append(nl_array_t* arr, const char* str) {
    if (!arr || !str) return;
    
    // Grow if needed
    if (arr->length >= arr->capacity) {
        int64_t new_capacity = arr->capacity * 2;
        char** new_data = (char**)realloc(arr->data, new_capacity * sizeof(char*));
        if (!new_data) return;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    
    // Duplicate string and add to array
    char* dup = strdup(str);
    if (dup) {
        ((char**)arr->data)[arr->length++] = dup;
    }
}

// Helper: Check if string ends with extension
static int ends_with(const char* str, const char* suffix) {
    if (!str || !suffix) return 0;
    
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    
    if (suffix_len > str_len) return 0;
    
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

// List files in directory
nl_array_t* nl_fs_list_files(const char* path, const char* extension) {
    nl_array_t* result = create_array(32);
    if (!result) return NULL;
    
    DIR* dir = opendir(path);
    if (!dir) {
        return result; // Return empty array
    }
    
    struct dirent* entry;
    int filter_by_ext = (extension && strlen(extension) > 0);
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Check if it's a regular file
        // Build full path for stat check
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
        
        struct stat st;
        if (stat(full_path, &st) == 0) {
            // Only include regular files
            if (S_ISREG(st.st_mode)) {
                // Filter by extension if specified
                if (!filter_by_ext || ends_with(entry->d_name, extension)) {
                    array_append(result, entry->d_name);
                }
            }
        }
    }
    
    closedir(dir);
    return result;
}

// Check if path is directory
int64_t nl_fs_is_directory(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

// Check if file exists
int64_t nl_fs_file_exists(const char* path) {
    return (access(path, F_OK) == 0) ? 1 : 0;
}

// Get file size
int64_t nl_fs_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    return (int64_t)st.st_size;
}

// Join path components
const char* nl_fs_join_path(const char* dir, const char* filename) {
    static char result[2048];
    
    // Handle empty inputs
    if (!dir || strlen(dir) == 0) {
        snprintf(result, sizeof(result), "%s", filename ? filename : "");
        return result;
    }
    
    if (!filename || strlen(filename) == 0) {
        snprintf(result, sizeof(result), "%s", dir);
        return result;
    }
    
    // Check if dir ends with /
    size_t dir_len = strlen(dir);
    if (dir[dir_len - 1] == '/') {
        snprintf(result, sizeof(result), "%s%s", dir, filename);
    } else {
        snprintf(result, sizeof(result), "%s/%s", dir, filename);
    }
    
    return result;
}
