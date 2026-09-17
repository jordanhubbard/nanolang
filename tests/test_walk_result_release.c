#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#include "../modules/std/fs.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

int main(void) {
    gc_init();
    size_t initial = gc_get_stats().num_objects;
    char directory[] = "/tmp/nano-walk-release-XXXXXX";
    assert(mkdtemp(directory));
    char path[256];
    snprintf(path, sizeof path, "%s/retained.txt", directory);
    FILE *file = fopen(path, "wx");
    assert(file && fclose(file) == 0);
    DynArray *result = fs_walkdir(directory);
    assert(result && result->length == 1);
    const char *borrowed = dyn_array_get_string(result, 0);
    char *escaped = strdup(borrowed);
    assert(escaped && strcmp(escaped, path) == 0);
    gc_retain(result);
    assert(!fs_walkdir_release(result));
    assert(result->length == 1 && strcmp(borrowed, path) == 0);
    gc_release(result);
    assert(fs_walkdir_release(result));
    assert(gc_get_stats().num_objects == initial);
    assert(strcmp(escaped, path) == 0);
    free(escaped);
    assert(!fs_walkdir_release(NULL));
    DynArray unmanaged = {0};
    assert(!fs_walkdir_release(&unmanaged));
    DynArray *integers = dyn_array_new(ELEM_INT);
    assert(integers && !fs_walkdir_release(integers));
    gc_release(integers);
    assert(unlink(path) == 0);
    result = fs_walkdir(directory);
    assert(result && result->length == 0 && fs_walkdir_release(result));
    assert(gc_get_stats().num_objects == initial);
    assert(rmdir(directory) == 0);
    gc_shutdown();
    puts("I released owned walk results and preserved escaped copies and shared arrays.");
    return 0;
}
