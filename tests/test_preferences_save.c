#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../modules/preferences/preferences.h"

static int fail_write, fail_close;
static int fail_array, fail_table_at, tables, fail_read_at, reads;
static DynArray *checked_array(ElementType type, int64_t capacity) {
    return fail_array ? NULL : dyn_array_new_with_capacity(type, capacity);
}
static void *checked_table(void *ptr, size_t bytes) {
    return ++tables == fail_table_at ? NULL : realloc(ptr, bytes);
}
static ssize_t checked_line(char **line, size_t *size, FILE *f) {
    return ++reads == fail_read_at ? -1 : getline(line, size, f);
}
static int checked_print(FILE *f, const char *format, ...) {
    if (fail_write) return -1;
    va_list args;
    va_start(args, format);
    int result = vfprintf(f, format, args);
    va_end(args);
    return result;
}
static int checked_close(FILE *f) {
    int result = fclose(f);
    return fail_close ? EOF : result;
}
#define fprintf checked_print
#define fclose checked_close
#define realloc checked_table
#define getline checked_line
#define dyn_array_new_with_capacity checked_array
#include "../modules/preferences/preferences.c"
#undef fprintf
#undef fclose
#undef realloc
#undef getline
#undef dyn_array_new_with_capacity

static void release_lines(DynArray *a) {
    assert(a && a->elem_type == ELEM_STRING && a->elem_size == sizeof(char*));
    for (int64_t i = 0; i < a->length; i++) free((void *)dyn_array_get_string(a, i));
    gc_release(a);
}

static void expect_file(const char *path, const char *expected) {
    FILE *f = fopen(path, "r");
    assert(f);
    char text[128] = {0};
    size_t n = fread(text, 1, sizeof(text) - 1, f);
    assert(!ferror(f) && n == strlen(expected));
    assert(strcmp(text, expected) == 0);
    assert(fclose(f) == 0);
}
int main(void) {
    gc_init();
    assert(nl_prefs_load_playlist__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_prefs_save_playlist__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    char path[] = "/tmp/nano-prefs-save-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    assert(write(fd, "original", 8) == 8);
    close(fd);
    char *data[] = {"one", "two"};
    DynArray valid = {.length=2, .capacity=2, .elem_type=ELEM_STRING,
                      .elem_size=sizeof(char*), .data=data};
    for (int test = 0; test < 10; test++) {
        DynArray a = valid;
        int64_t count = 2;
        switch (test) {
        case 0: a.elem_type = ELEM_INT; break;
        case 1: a.elem_size = 1; break;
        case 2: a.length = -1; break;
        case 3: a.capacity = 1; break;
        case 4: count = -1; break;
        case 5: count = 3; break;
        case 6: a.data = NULL; break;
        case 7: data[1] = NULL; break;
        case 8: data[1] = "two\nthree"; break;
        case 9: a.capacity = INT64_MAX; break;
        }
        assert(nl_prefs_save_playlist(path, &a, count) == 0);
        expect_file(path, "original");
        data[1] = "two";
    }
    assert(!nl_prefs_save_playlist(path, NULL, 0));
    assert(!nl_prefs_save_playlist(NULL, &valid, 2));
    expect_file(path, "original");
    assert(nl_prefs_save_playlist(path, &valid, 1));
    expect_file(path, "one\n");
    assert(nl_prefs_save_playlist(path, &valid, 2));
    expect_file(path, "one\ntwo\n");
    fail_write = 1;
    assert(!nl_prefs_save_playlist(path, &valid, 2));
    fail_write = 0;
    fail_close = 1;
    assert(!nl_prefs_save_playlist(path, &valid, 2));
    fail_close = 0;
    assert(nl_prefs_save_playlist(path, &valid, 0));
    expect_file(path, "");
    DynArray *loaded = nl_prefs_load_playlist(path);
    assert(loaded && loaded->length == 0);
    release_lines(loaded);
    FILE *f = fopen(path, "w");
    assert(f);
    for (int i = 0; i < 6000; i++) assert(fputc('x', f) != EOF);
    assert(fputs("\n\n", f) >= 0);
    for (int i = 0; i < 40; i++) assert(fputs("entry\n", f) >= 0);
    assert(fputs("last", f) >= 0 && fclose(f) == 0);
    loaded = nl_prefs_load_playlist(path);
    assert(loaded && loaded->length == 42);
    assert(strlen(dyn_array_get_string(loaded, 0)) == 6000);
    assert(!strcmp(dyn_array_get_string(loaded, 41), "last"));
    release_lines(loaded);
    for (int i = 1; i <= 2; i++) {
        tables = 0; fail_table_at = i;
        assert(!nl_prefs_load_playlist(path));
        fail_table_at = 0;
    }
    for (int i = 1; i <= 4; i++) {
        reads = 0; fail_read_at = i;
        assert(!nl_prefs_load_playlist(path));
        fail_read_at = 0;
    }
    fail_array = 1;
    assert(!nl_prefs_load_playlist(path));
    fail_array = 0;
    fail_close = 1;
    assert(!nl_prefs_load_playlist(path));
    fail_close = 0;
    assert(!nl_prefs_load_playlist(NULL));
    assert(!nl_prefs_load_playlist("/"));
    f = fopen(path, "w");
    assert(f && fwrite("a\0b\n", 1, 4, f) == 4 && fclose(f) == 0);
    assert(!nl_prefs_load_playlist(path));
    assert(unlink(path) == 0);
    loaded = nl_prefs_load_playlist(path);
    assert(loaded && loaded->length == 0);
    release_lines(loaded);
    assert(setenv("HOME", "/tmp/prefs-test", 1) == 0);
    assert(!strcmp(nl_prefs_get_path("app"), "/tmp/prefs-test/.app_prefs"));
    assert(!nl_prefs_get_path(NULL));
    char long_name[2048];
    memset(long_name, 'a', sizeof(long_name) - 1);
    long_name[sizeof(long_name) - 1] = 0;
    assert(!nl_prefs_get_path(long_name));
    assert(setenv("HOME", long_name, 1) == 0);
    assert(!nl_prefs_get_path("app"));
    assert(unsetenv("HOME") == 0);
    assert(!strcmp(nl_prefs_get_path("app"), "/tmp/.app_prefs"));
    assert(gc_get_stats().num_objects == 0);
    gc_shutdown();
    return 0;
}
