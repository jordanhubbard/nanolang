#define _POSIX_C_SOURCE 200809L  /* For strdup(), strtok_r() */

#include "fs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>
#include <errno.h>

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

/* Internal helper: normalize path into caller-provided buffer */
static void path_normalize_into(const char* path, char* result, size_t result_size) {
    if (!path || path[0] == '\0') {
        snprintf(result, result_size, ".");
        return;
    }

    int is_absolute = (path[0] == '/');

    /* Copy path for tokenization */
    char* copy = strdup(path);
    if (!copy) {
        snprintf(result, result_size, "%s", path);
        return;
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
                strncat(result, "/", result_size - strlen(result) - 1);
            }
        }
        strncat(result, parts[i], result_size - strlen(result) - 1);
    }

    /* Handle empty result */
    if (result[0] == '\0') {
        snprintf(result, result_size, ".");
    }

    free(copy);
}

/* Normalize path (resolve . and .., remove redundant slashes) */
const char* path_normalize(const char* path) {
    char result[2048];
    path_normalize_into(path, result, sizeof(result));
    return strdup(result);
}

/* Join two path components */
const char* path_join(const char* a, const char* b) {
    char result[2048];

    if (!a || a[0] == '\0') {
        snprintf(result, sizeof(result), "%s", b ? b : "");
        return strdup(result);
    }

    if (!b || b[0] == '\0') {
        snprintf(result, sizeof(result), "%s", a);
        return strdup(result);
    }

    /* Check if a ends with / */
    size_t a_len = strlen(a);
    if (a[a_len - 1] == '/') {
        snprintf(result, sizeof(result), "%s%s", a, b);
    } else {
        snprintf(result, sizeof(result), "%s/%s", a, b);
    }

    return strdup(result);
}

/* Get basename of path */
const char* path_basename(const char* path) {
    if (!path || path[0] == '\0') {
        return strdup(".");
    }

    /* Use POSIX basename (modifies input, so copy first) */
    char* copy = strdup(path);
    if (!copy) {
        return strdup(path);
    }

    char* base = basename(copy);
    const char* result = strdup(base);
    free(copy);

    return result;
}

/* Get dirname of path */
const char* path_dirname(const char* path) {
    if (!path || path[0] == '\0') {
        return strdup(".");
    }

    /* Use POSIX dirname (modifies input, so copy first) */
    char* copy = strdup(path);
    if (!copy) {
        return strdup(path);
    }

    char* dir = dirname(copy);
    const char* result = strdup(dir);
    free(copy);

    return result;
}

static void path_append(char* out, size_t out_size, const char* part) {
    if (!out || !part) return;
    if (out[0] != '\0') {
        strncat(out, "/", out_size - strlen(out) - 1);
    }
    strncat(out, part, out_size - strlen(out) - 1);
}

/* Compute relative path from base to target */
const char* path_relpath(const char* target, const char* base) {
    char result[4096];

    if (!target || !base) {
        return strdup(".");
    }

    char target_norm[2048];
    char base_norm[2048];
    path_normalize_into(target, target_norm, sizeof(target_norm));
    path_normalize_into(base, base_norm, sizeof(base_norm));

    char target_copy[2048];
    char base_copy[2048];
    snprintf(target_copy, sizeof(target_copy), "%s", target_norm);
    snprintf(base_copy, sizeof(base_copy), "%s", base_norm);

    char* target_parts[256];
    char* base_parts[256];
    int target_count = 0;
    int base_count = 0;

    char* saveptr = NULL;
    char* token = strtok_r(target_copy, "/", &saveptr);
    while (token && target_count < 256) {
        target_parts[target_count++] = token;
        token = strtok_r(NULL, "/", &saveptr);
    }

    saveptr = NULL;
    token = strtok_r(base_copy, "/", &saveptr);
    while (token && base_count < 256) {
        base_parts[base_count++] = token;
        token = strtok_r(NULL, "/", &saveptr);
    }

    int common = 0;
    while (common < target_count && common < base_count &&
           strcmp(target_parts[common], base_parts[common]) == 0) {
        common++;
    }

    result[0] = '\0';

    for (int i = common; i < base_count; i++) {
        path_append(result, sizeof(result), "..");
    }

    for (int i = common; i < target_count; i++) {
        path_append(result, sizeof(result), target_parts[i]);
    }

    if (result[0] == '\0') {
        snprintf(result, sizeof(result), ".");
    }

    return strdup(result);
}

/* Read file content as string */
const char* file_read(const char* path) {
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

/* Create directory and parents (mkdir -p) */
int64_t fs_mkdir_p(const char* path) {
    if (!path || path[0] == '\0') return -1;

    char tmp[2048];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);

    if (len == 0) return -1;
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                *p = '/';
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

/* Copy a single file (binary-safe) */
int64_t file_copy(const char* src, const char* dst) {
    FILE* in = fopen(src, "rb");
    if (!in) return -1;
    FILE* out = fopen(dst, "wb");
    if (!out) {
        fclose(in);
        return -1;
    }

    char buffer[8192];
    size_t n = 0;
    while ((n = fread(buffer, 1, sizeof(buffer), in)) > 0) {
        if (fwrite(buffer, 1, n, out) != n) {
            fclose(in);
            fclose(out);
            return -1;
        }
    }

    if (ferror(in)) {
        fclose(in);
        fclose(out);
        return -1;
    }

    fclose(in);
    fclose(out);
    return 0;
}

/* Copy a directory tree recursively */
int64_t dir_copy(const char* src, const char* dst) {
    struct stat st;
    if (stat(src, &st) != 0 || !S_ISDIR(st.st_mode)) {
        return -1;
    }

    if (fs_mkdir_p(dst) != 0) {
        return -1;
    }

    DIR* dir = opendir(src);
    if (!dir) return -1;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char src_path[2048];
        char dst_path[2048];
        snprintf(src_path, sizeof(src_path), "%s/%s", src, entry->d_name);
        snprintf(dst_path, sizeof(dst_path), "%s/%s", dst, entry->d_name);

        if (stat(src_path, &st) != 0) {
            closedir(dir);
            return -1;
        }

        if (S_ISDIR(st.st_mode)) {
            if (dir_copy(src_path, dst_path) != 0) {
                closedir(dir);
                return -1;
            }
        } else if (S_ISREG(st.st_mode)) {
            if (file_copy(src_path, dst_path) != 0) {
                closedir(dir);
                return -1;
            }
        }
    }

    closedir(dir);
    return 0;
}
