#include "fs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>

/* Forward declarations */
extern char* nl_str_concat(const char* s1, const char* s2);
extern DynArray* dyn_array_new_with_capacity(ElementType elem_type, int64_t initial_capacity);
extern DynArray* dyn_array_push_string_copy(DynArray* arr, const char* value);

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
                dyn_array_push_string_copy(result, full_path);
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
    DynArray* result = dyn_array_new_with_capacity(ELEM_STRING, 128);
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

/* Read file content as string */
char* file_read(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return "";
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* buffer = malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return "";
    }
    
    size_t read = fread(buffer, 1, size, f);
    buffer[read] = '\0';
    fclose(f);
    
    return buffer;
}

/* Write string to file */
int64_t file_write(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    
    size_t written = fwrite(content, 1, strlen(content), f);
    fclose(f);
    
    return (written == strlen(content)) ? 0 : -1;
}

/* Append string to file */
int64_t file_append(const char* path, const char* content) {
    FILE* f = fopen(path, "a");
    if (!f) return -1;
    
    size_t written = fwrite(content, 1, strlen(content), f);
    fclose(f);
    
    return (written == strlen(content)) ? 0 : -1;
}

/* Check if file exists */
bool file_exists(const char* path) {
    return access(path, F_OK) == 0;
}

/* Delete file */
int64_t file_delete(const char* path) {
    return remove(path);
}

