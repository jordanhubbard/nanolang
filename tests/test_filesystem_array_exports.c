#include "../modules/filesystem/filesystem.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

extern const uint32_t nl_fs_list_files__nano_array_abi;
extern const uint32_t nl_fs_list_files_ci__nano_array_abi;
extern const uint32_t nl_fs_list_dirs__nano_array_abi;

static void check(DynArray *a, int64_t count) {
    assert(a && a->elem_type == ELEM_STRING && a->elem_size == sizeof(char *));
    assert(a->length == count && a->capacity >= count);
    for (int64_t i = 1; i < count; ++i)
        assert(strcmp(dyn_array_get_string(a, i - 1), dyn_array_get_string(a, i)) <= 0);
}
static void dispose(DynArray *a) {
    /* I free known copied fixture elements; runtime ownership is separate. */
    for (int64_t i = 0; i < a->length; ++i) free((void *)dyn_array_get_string(a, i));
    gc_release(a);
}
static void paths(void) {
    assert(!nl_fs_is_directory(NULL));
    assert(!nl_fs_file_exists(NULL));
    assert(nl_fs_file_size(NULL) == -1);
    assert(!strcmp(nl_fs_parent_dir(NULL), "."));
    assert(!strcmp(nl_fs_parent_dir(""), "."));
    assert(!strcmp(nl_fs_parent_dir("/"), "/"));
    assert(!strcmp(nl_fs_parent_dir("///"), "/"));
    assert(!strcmp(nl_fs_parent_dir("/a///"), "/"));
    assert(!strcmp(nl_fs_parent_dir("a/b/"), "a"));
    const char *parent = nl_fs_parent_dir("a/b/c");
    assert(!strcmp(nl_fs_parent_dir(parent), "a"));
    assert(!strcmp(nl_fs_join_path(NULL, NULL), ""));
    assert(!strcmp(nl_fs_join_path(NULL, "b"), "b"));
    assert(!strcmp(nl_fs_join_path("a", NULL), "a"));
    assert(!strcmp(nl_fs_join_path("a/", "b"), "a/b"));
    const char *joined = nl_fs_join_path("a", "b");
    assert(!strcmp(nl_fs_join_path(joined, "c"), "a/b/c"));
    joined = nl_fs_join_path("a", "b");
    assert(!strcmp(nl_fs_join_path("prefix", joined), "prefix/a/b"));
    char limit[2049];
    memset(limit, 'x', sizeof(limit));
    limit[2047] = 0;
    assert(strlen(nl_fs_join_path(limit, NULL)) == 2047);
    assert(!nl_fs_join_path(limit, "y"));
    assert(!nl_fs_join_path("y", limit));
    assert(!strcmp(nl_fs_parent_dir(limit), "."));
    limit[2045] = '/'; limit[2046] = 'z';
    assert(strlen(nl_fs_parent_dir(limit)) == 2045);
    limit[2047] = 'x'; limit[2048] = 0;
    assert(!nl_fs_parent_dir(limit));
    assert(!nl_fs_join_path(limit, NULL));
    assert(!nl_fs_join_path(NULL, limit));
    limit[2045] = 0;
    assert(strlen(nl_fs_join_path(limit, "y")) == 2047);
}
int main(int argc, char **argv) {
    assert(argc == 3);
    gc_init();
    paths();
    assert(nl_fs_is_directory(argv[1]));
    assert(nl_fs_file_exists(argv[1]));
    assert(nl_fs_file_size(nl_fs_join_path(argv[1], "a.txt")) == 0);
    assert(nl_fs_list_files__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_fs_list_files_ci__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_fs_list_dirs__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    DynArray *a = nl_fs_list_files(argv[1], ".txt");
    check(a, 2);
    assert(!strcmp(dyn_array_get_string(a, 0), "a.txt"));
    assert(!strcmp(dyn_array_get_string(a, 1), argv[2]));
    dispose(a);
    a = nl_fs_list_files_ci(argv[1], ".txt"); check(a, 4); dispose(a);
    a = nl_fs_list_files(argv[1], NULL); check(a, 4); dispose(a);
    a = nl_fs_list_dirs(argv[1]); check(a, 2);
    assert(!strcmp(dyn_array_get_string(a, 0), "folder"));
    assert(!strcmp(dyn_array_get_string(a, 1), "linked"));
    dispose(a);
    a = nl_fs_list_files(NULL, ".txt"); check(a, 0); dispose(a);
    a = nl_fs_list_files_ci("", NULL); check(a, 0); dispose(a);
    a = nl_fs_list_dirs("/no-such-nanolang-array-fixture"); check(a, 0); dispose(a);
    assert(gc_get_stats().num_objects == 0);
    gc_shutdown();
    return 0;
}
