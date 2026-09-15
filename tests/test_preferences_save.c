#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fail_write, fail_close;
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
#include "../modules/preferences/preferences.c"
#undef fprintf
#undef fclose

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
    assert(unlink(path) == 0);
    return 0;
}
