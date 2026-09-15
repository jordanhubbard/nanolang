#define _POSIX_C_SOURCE 200809L  /* For strdup(), strtok_r() */
#define _XOPEN_SOURCE 700       /* For realpath() */

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

#include "../../src/runtime/directory_walk.h"
#include "../../src/runtime/file_text.h"
#include "../../src/runtime/file_write.h"
#include "../../src/runtime/path_normalize.h"

NANO_EXPORT_ARRAY_ABI(fs_walkdir);

DynArray* fs_walkdir(const char* root) {
    return nl_fs_walkdir(root);
}

bool fs_walkdir_release(DynArray* result) {
    if (!result || !gc_is_managed(result)) return false;
    GCHeader *header = gc_get_header(result);
    if (header->type != GC_TYPE_ARRAY || header->ref_count != 1 ||
        !dyn_array_has_storage(result, ELEM_STRING, sizeof(char*), 0)) return false;
    for (int64_t i = 0; i < result->length; i++) {
        free(((char**)result->data)[i]);
        ((char**)result->data)[i] = NULL;
    }
    result->length = 0;
    gc_release(result);
    return true;
}

/* I resolve existing paths physically; failure is an empty string, never a
 * lexical approximation of the requested identity. */
const char* path_canonical(const char* path) {
    if (!path || !path[0]) return strdup("");
    char *resolved = realpath(path, NULL);
    return resolved ? resolved : strdup("");
}

/* I compare existing file identities without opening either file for writing.
 * A missing candidate is distinct; unavailable source identity is an error.
 * This is a snapshot check, not protection against concurrent path replacement. */
int64_t file_compare_identity(const char* source, const char* candidate) {
    struct stat source_stat, candidate_stat;
    if (!source || !source[0] || !candidate || !candidate[0]) return -1;
    if (stat(source, &source_stat) != 0) return -1;
    if (stat(candidate, &candidate_stat) != 0)
        return errno == ENOENT || errno == ENOTDIR ? 0 : -1;
    return source_stat.st_dev == candidate_stat.st_dev &&
           source_stat.st_ino == candidate_stat.st_ino ? 1 : 0;
}

/* I distinguish absent entries from dangling links and lookup failures. */
static int destination_stat(const char* path, struct stat* info) {
    if (stat(path, info) == 0) return 1;
    if (errno != ENOENT) return -1;
    if (lstat(path, info) == 0 || errno != ENOENT) return -1;
    return 0;
}

/* I compare stable destination entries, including files not yet created.
 * An exclusive empty-directory probe asks the filesystem about absent names.
 * I remove it before returning; unresolved identity or cleanup fails closed.
 * This is not protection against concurrent namespace replacement. */
int64_t file_compare_destinations(const char* first, const char* second) {
    if (!first || !first[0] || !second || !second[0]) return -1;
    struct stat a, b;
    int a_exists = destination_stat(first, &a);
    int b_exists = destination_stat(second, &b);
    if (a_exists < 0 || b_exists < 0) return -1;
    if (a_exists && b_exists)
        return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
    if (a_exists || b_exists) return 0;

    if (mkdir(first, 0700) != 0) return -1;
    int64_t result = -1;
    if (stat(first, &a) == 0) {
        b_exists = destination_stat(second, &b);
        if (b_exists == 0) result = 0;
        else if (b_exists > 0)
            result = a.st_dev == b.st_dev && a.st_ino == b.st_ino;
    }
    if (rmdir(first) != 0) return -1;
    return result;
}


/* Normalize path (resolve . and .., remove redundant slashes) */
const char* path_normalize(const char* path) {
    return nl_normalize_path(path);
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

/* I compare normalized lexical components without fixed path/token buffers.
 * Mixed roots and unresolved parent components retain the existing lexical
 * comparison rules; I do not resolve either path against a working directory. */
const char* path_relpath(const char* target, const char* base) {
    if (!target || !base) return strdup(".");
    char *target_norm = nl_normalize_path(target);
    char *base_norm = nl_normalize_path(base);
    if (!target_norm || !base_norm) {
        free(target_norm); free(base_norm); return NULL;
    }
    size_t target_length = strlen(target_norm), base_length = strlen(base_norm);
    if (target_length > SIZE_MAX - 2 || base_length > (SIZE_MAX - target_length - 2) / 3) {
        free(target_norm); free(base_norm); return NULL;
    }
    char *result = malloc(target_length + base_length * 3 + 2);
    if (!result) { free(target_norm); free(base_norm); return NULL; }
    char *target_save = NULL, *base_save = NULL;
    char *a = strtok_r(target_norm, "/", &target_save);
    char *b = strtok_r(base_norm, "/", &base_save);
    while (a && b && strcmp(a, b) == 0) {
        a = strtok_r(NULL, "/", &target_save);
        b = strtok_r(NULL, "/", &base_save);
    }
    size_t used = 0;
    while (b) {
        if (used) result[used++] = '/';
        result[used++] = '.'; result[used++] = '.';
        b = strtok_r(NULL, "/", &base_save);
    }
    while (a) {
        if (used) result[used++] = '/';
        size_t length = strlen(a);
        memcpy(result + used, a, length); used += length;
        a = strtok_r(NULL, "/", &target_save);
    }
    if (!used) result[used++] = '.';
    result[used] = 0;
    free(target_norm); free(base_norm);
    return result;
}

/* Read file content as string */
const char* file_read(const char* path) {
    char *text = nl_read_file_text(path);
    return text ? text : "";
}

/* Write string to file */
int64_t file_write(const char* path, const char* content) {
    return nl_write_file_text(path, content, "w");
}

/* Append string to file */
int64_t file_append(const char* path, const char* content) {
    return nl_write_file_text(path, content, "a");
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
