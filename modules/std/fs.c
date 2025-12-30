#include "fs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>

/* Helper: Create nanolang array for strings */
static DynArray* create_string_array(int64_t initial_capacity) {
    DynArray* arr = (DynArray*)malloc(sizeof(DynArray));
    if (!arr) return NULL;
    
    arr->length = 0;
    arr->capacity = initial_capacity;
    arr->elem_type = ELEM_STRING;
    arr->elem_size = sizeof(char*);
    arr->data = calloc(initial_capacity, sizeof(char*));
    
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    
    return arr;
}

/* Helper: Append string to array */
static void array_append_string(DynArray* arr, const char* str) {
    if (!arr || !str) return;
    
    /* Grow if needed */
    if (arr->length >= arr->capacity) {
        int64_t new_capacity = arr->capacity * 2;
        char** new_data = (char**)realloc(arr->data, new_capacity * sizeof(char*));
        if (!new_data) return;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    
    /* Duplicate string and add to array */
    char* dup = strdup(str);
    if (dup) {
        ((char**)arr->data)[arr->length++] = dup;
    }
}

/* Recursive directory walker */
static void walkdir_recursive(const char* path, DynArray* result) {
    DIR* dir = opendir(path);
    if (!dir) return;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        /* Build full path */
        char full_path[2048];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
        
        struct stat st;
        if (stat(full_path, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                /* Add file to result */
                array_append_string(result, full_path);
            } else if (S_ISDIR(st.st_mode)) {
                /* Recurse into directory */
                walkdir_recursive(full_path, result);
            }
        }
    }
    
    closedir(dir);
}

/* Walk directory tree, returning all file paths */
DynArray* fs_walkdir(const char* root) {
    DynArray* result = create_string_array(128);
    if (!result) return NULL;
    
    walkdir_recursive(root, result);
    return result;
}

/* Normalize path (resolve . and .., remove redundant slashes) */
const char* path_normalize(const char* path) {
    static char result[2048];
    
    if (!path || path[0] == '\0') {
        snprintf(result, sizeof(result), ".");
        return result;
    }
    
    int is_absolute = (path[0] == '/');
    
    /* Copy path for tokenization */
    char* copy = strdup(path);
    if (!copy) {
        snprintf(result, sizeof(result), "%s", path);
        return result;
    }
    
    /* Split path into components */
    const char* parts[512];
    int count = 0;
    
    char* saveptr = NULL;
    char* token = strtok_r(copy, "/", &saveptr);
    while (token) {
        if (strcmp(token, "") == 0 || strcmp(token, ".") == 0) {
            /* Skip empty and current directory */
        } else if (strcmp(token, "..") == 0) {
            /* Go up one level if possible */
            if (count > 0 && strcmp(parts[count - 1], "..") != 0) {
                count--;
            } else if (!is_absolute) {
                /* Keep .. for relative paths */
                if (count < 512) parts[count++] = token;
            }
        } else {
            /* Normal component */
            if (count < 512) parts[count++] = token;
        }
        token = strtok_r(NULL, "/", &saveptr);
    }
    
    /* Build result */
    result[0] = '\0';
    if (is_absolute) {
        strcat(result, "/");
    }
    
    for (int i = 0; i < count; i++) {
        if (i > 0 || is_absolute) {
            if (result[strlen(result) - 1] != '/') {
                strcat(result, "/");
            }
        }
        strcat(result, parts[i]);
    }
    
    /* Handle empty result */
    if (result[0] == '\0') {
        snprintf(result, sizeof(result), ".");
    }
    
    free(copy);
    return result;
}

/* Join two path components */
const char* path_join(const char* a, const char* b) {
    static char result[2048];
    
    if (!a || a[0] == '\0') {
        snprintf(result, sizeof(result), "%s", b ? b : "");
        return result;
    }
    
    if (!b || b[0] == '\0') {
        snprintf(result, sizeof(result), "%s", a);
        return result;
    }
    
    /* Check if a ends with / */
    size_t a_len = strlen(a);
    if (a[a_len - 1] == '/') {
        snprintf(result, sizeof(result), "%s%s", a, b);
    } else {
        snprintf(result, sizeof(result), "%s/%s", a, b);
    }
    
    return result;
}

/* Get basename of path */
const char* path_basename(const char* path) {
    static char result[2048];
    
    if (!path || path[0] == '\0') {
        snprintf(result, sizeof(result), ".");
        return result;
    }
    
    /* Use POSIX basename (modifies input, so copy first) */
    char* copy = strdup(path);
    if (!copy) {
        snprintf(result, sizeof(result), "%s", path);
        return result;
    }
    
    char* base = basename(copy);
    snprintf(result, sizeof(result), "%s", base);
    free(copy);
    
    return result;
}

/* Get dirname of path */
const char* path_dirname(const char* path) {
    static char result[2048];
    
    if (!path || path[0] == '\0') {
        snprintf(result, sizeof(result), ".");
        return result;
    }
    
    /* Use POSIX dirname (modifies input, so copy first) */
    char* copy = strdup(path);
    if (!copy) {
        snprintf(result, sizeof(result), "%s", path);
        return result;
    }
    
    char* dir = dirname(copy);
    snprintf(result, sizeof(result), "%s", dir);
    free(copy);
    
    return result;
}

