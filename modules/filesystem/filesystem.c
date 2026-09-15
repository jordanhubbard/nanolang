#define _POSIX_C_SOURCE 200809L
#include "filesystem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <ctype.h>

NANO_EXPORT_ARRAY_ABI(nl_fs_list_files);
NANO_EXPORT_ARRAY_ABI(nl_fs_list_files_ci);
NANO_EXPORT_ARRAY_ABI(nl_fs_list_dirs);

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

/* I resolve entries relative to the open directory, without truncating a
 * joined path. Flags zero preserves the existing symlink-following behavior. */
static DynArray *list_entries(const char *path, const char *extension,
                               bool directories, bool case_insensitive) {
    DynArray *result = dyn_array_new_with_capacity(ELEM_STRING, 32);
    if (!result || !path) return result;
    DIR *dir = opendir(path);
    if (!dir) return result;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        struct stat st;
        if (fstatat(dirfd(dir), entry->d_name, &st, 0) != 0) continue;
        if (directories ? !S_ISDIR(st.st_mode) : !S_ISREG(st.st_mode)) continue;
        if (!directories && extension && *extension &&
            !(case_insensitive ? ends_with_ci(entry->d_name, extension)
                               : ends_with(entry->d_name, extension))) continue;
        dyn_array_push_string_copy(result, entry->d_name);
    }
    closedir(dir);
    sort_string_array(result);
    return result;
}

DynArray *nl_fs_list_files(const char *path, const char *extension) {
    return list_entries(path, extension, false, false);
}

DynArray *nl_fs_list_files_ci(const char *path, const char *extension) {
    return list_entries(path, extension, false, true);
}

DynArray *nl_fs_list_dirs(const char *path) {
    return list_entries(path, NULL, true, false);
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
