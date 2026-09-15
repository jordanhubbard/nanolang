#ifndef NANOLANG_STD_FS_H
#define NANOLANG_STD_FS_H

#include <stdint.h>
#include <stdbool.h>
#include "../../src/runtime/dyn_array.h"

/* I return regular-file paths in unspecified order, following symlinks and
 * visiting each opened directory identity once. Aliased directories use the
 * first encountered spelling. Missing/unreadable roots yield an empty array;
 * inaccessible/disappearing entries and subtrees are omitted. This is a
 * best-effort walk, not a complete snapshot or a confinement boundary. Paths
 * have no internal fixed-size limit; host filesystem limits still apply.
 * Copied strings retain process-lifetime storage unless the caller explicitly
 * consumes the unmodified result with fs_walkdir_release below. Ordinary
 * native array collection does not free these strings. */
DynArray* fs_walkdir(const char* root);

/* I consume only an unmodified result returned by fs_walkdir in this library.
 * The caller must copy any escaping strings first. No borrowed element pointer
 * may survive success. I refuse arrays with additional GC owners. This opt-in
 * operation does not change ordinary native array element ownership. */
bool fs_walkdir_release(DynArray* result);

/* Normalize path (resolve . and .., remove redundant slashes) */
const char* path_normalize(const char* path);

/* I return an allocated physical absolute path, or an empty string on failure. */
const char* path_canonical(const char* path);

/* I return 1 for the same file, 0 for distinct/missing candidate, -1 on error. */
int64_t file_compare_identity(const char* source, const char* candidate);

/* I return 1 for colliding destinations, 0 for distinct entries, -1 on error.
 * When both are missing, I create and remove an empty directory at first to
 * query filesystem name equivalence. Cleanup failure can leave that probe. */
int64_t file_compare_destinations(const char* first, const char* second);

/* Join two path components */
const char* path_join(const char* a, const char* b);

/* Get basename of path */
const char* path_basename(const char* path);

/* Get dirname of path */
const char* path_dirname(const char* path);

/* Compute relative path from base to target */
const char* path_relpath(const char* target, const char* base);

/* Read file content as string */
const char* file_read(const char* path);

/* Write string to file */
int64_t file_write(const char* path, const char* content);

/* Append string to file */
int64_t file_append(const char* path, const char* content);

/* Check if file exists */
bool file_exists(const char* path);

/* Delete file */
int64_t file_delete(const char* path);

/* Create directory and parents (mkdir -p) */
int64_t fs_mkdir_p(const char* path);

/* Copy a single file (binary-safe) */
int64_t file_copy(const char* src, const char* dst);

/* Copy a directory tree recursively */
int64_t dir_copy(const char* src, const char* dst);

#endif /* NANOLANG_STD_FS_H */
