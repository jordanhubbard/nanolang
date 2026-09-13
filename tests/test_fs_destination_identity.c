#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static int deny_create, deny_lookup, deny_cleanup, probe_active;

static int probe_mkdir(const char* path, mode_t mode) {
    if (deny_create) { errno = EACCES; return -1; }
    int result = mkdir(path, mode);
    if (result == 0) probe_active = 1;
    return result;
}

static int probe_stat(const char* path, struct stat* info) {
    if (deny_lookup && probe_active) { errno = EIO; return -1; }
    return stat(path, info);
}

static int probe_rmdir(const char* path) {
    if (deny_cleanup) { errno = EACCES; return -1; }
    int result = rmdir(path);
    if (result == 0) probe_active = 0;
    return result;
}

/* I inject failures into the production helper, not a copied implementation. */
#define mkdir probe_mkdir
#define stat(path, info) probe_stat(path, info)
#define rmdir probe_rmdir
#include "../modules/std/fs.c"
#undef mkdir
#undef stat
#undef rmdir

int main(int argc, char** argv) {
    assert(argc == 2);
    char first[4096], second[4096];
    assert(snprintf(first, sizeof(first), "%s/output", argv[1]) < (int)sizeof(first));
    assert(snprintf(second, sizeof(second), "%s/report", argv[1]) < (int)sizeof(second));
    struct stat info;
    assert(file_compare_destinations(first, second) == 0);
    assert(!probe_active && lstat(first, &info) != 0 && errno == ENOENT);
    assert(file_compare_destinations(first, first) == 1);
    assert(!probe_active && lstat(first, &info) != 0 && errno == ENOENT);

    deny_create = 1;
    assert(file_compare_destinations(first, second) == -1);
    assert(!probe_active && lstat(first, &info) != 0 && errno == ENOENT);
    deny_create = 0;

    deny_lookup = 1;
    assert(file_compare_destinations(first, second) == -1);
    assert(!probe_active && lstat(first, &info) != 0 && errno == ENOENT);
    deny_lookup = 0;

    deny_cleanup = 1;
    assert(file_compare_destinations(first, second) == -1);
    assert(probe_active && stat(first, &info) == 0 && S_ISDIR(info.st_mode));
    /* I remove only the empty probe I deliberately prevented from cleaning up. */
    assert(rmdir(first) == 0);
    probe_active = deny_cleanup = 0;

    assert(file_compare_destinations(first, second) == 0);
    assert(!probe_active && lstat(first, &info) != 0 && errno == ENOENT);
    return 0;
}
