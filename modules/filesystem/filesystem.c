#include "filesystem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <ctype.h>

// Helper: Create nanolang array for strings
static DynArray* create_string_array(int64_t initial_capacity) {
    DynArray* arr = (DynArray*)malloc(sizeof(DynArray));
    if (!arr) return NULL;
    
    arr->length = 0;
    arr->capacity = initial_capacity;
    arr->elem_type = ELEM_STRING;  // ElementType enum from dyn_array.h
    arr->elem_size = sizeof(char*);
    arr->data = calloc(initial_capacity, sizeof(char*));
    
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    
    return arr;
}

// Helper: Append string to array
static void array_append_string(DynArray* arr, const char* str) {
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

static int cmp_cstr_ptr(const void *a, const void *b) {
    const char *sa = *(const char * const *)a;
    const char *sb = *(const char * const *)b;
    if (!sa && !sb) return 0;
    if (!sa) return -1;
    if (!sb) return 1;
    return strcmp(sa, sb);
}

static void sort_string_array(DynArray *arr) {
    if (!arr || arr->length <= 1) return;
    qsort(arr->data, (size_t)arr->length, sizeof(char*), cmp_cstr_ptr);
}

// Helper: Check if string ends with extension
static int ends_with(const char* str, const char* suffix) {
    if (!str || !suffix) return 0;
    
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    
    if (suffix_len > str_len) return 0;
    
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

static int ends_with_ci(const char *str, const char *suffix) {
    if (!str || !suffix) return 0;
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return 0;
    const char *a = str + (str_len - suffix_len);
    for (size_t i = 0; i < suffix_len; i++) {
        unsigned char ca = (unsigned char)a[i];
        unsigned char cb = (unsigned char)suffix[i];
        if (tolower(ca) != tolower(cb)) return 0;
    }
    return 1;
}

// List files in directory
DynArray* nl_fs_list_files(const char* path, const char* extension) {
    DynArray* result = create_string_array(32);
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
                    array_append_string(result, entry->d_name);
                }
            }
        }
    }
    
    closedir(dir);
    sort_string_array(result);
    return result;
}

DynArray* nl_fs_list_files_ci(const char* path, const char* extension) {
    DynArray* result = create_string_array(32);
    if (!result) return NULL;

    DIR* dir = opendir(path);
    if (!dir) {
        return result;
    }

    struct dirent* entry;
    int filter_by_ext = (extension && strlen(extension) > 0);

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);

        struct stat st;
        if (stat(full_path, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                if (!filter_by_ext || ends_with_ci(entry->d_name, extension)) {
                    array_append_string(result, entry->d_name);
                }
            }
        }
    }

    closedir(dir);
    sort_string_array(result);
    return result;
}

// List directories in directory
DynArray* nl_fs_list_dirs(const char* path) {
    DynArray* result = create_string_array(32);
    if (!result) return NULL;
    
    DIR* dir = opendir(path);
    if (!dir) {
        return result; // Return empty array
    }
    
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Build full path for stat check
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
        
        struct stat st;
        if (stat(full_path, &st) == 0) {
            // Only include directories
            if (S_ISDIR(st.st_mode)) {
                array_append_string(result, entry->d_name);
            }
        }
    }
    
    closedir(dir);
    sort_string_array(result);
    return result;
}

const char* nl_fs_parent_dir(const char* path) {
    static char out[2048];
    if (!path || path[0] == 0) {
        snprintf(out, sizeof(out), ".");
        return out;
    }

    /* Copy and trim trailing slashes (except root). */
    snprintf(out, sizeof(out), "%s", path);
    size_t n = strlen(out);
    while (n > 1 && out[n - 1] == '/') {
        out[n - 1] = 0;
        n--;
    }

    char *last = strrchr(out, '/');
    if (!last) {
        snprintf(out, sizeof(out), ".");
        return out;
    }

    if (last == out) {
        /* Parent of "/x" is "/" */
        out[1] = 0;
        return out;
    }

    *last = 0;
    if (out[0] == 0) {
        snprintf(out, sizeof(out), ".");
    }
    return out;
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
