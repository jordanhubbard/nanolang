#include "runtime/dyn_array.h"
#include "runtime/gc.h"
#include <assert.h>
#include <stdio.h>

int g_argc = 0;
char **g_argv = NULL;
static int fail_read, fail_close, reads, closes;

static size_t checked_read(void *ptr, size_t size, size_t count, FILE *file) {
    reads++;
    if (fail_read && reads > 1) return 0;
    return fread(ptr, size, count, file);
}
static int checked_error(FILE *file) {
    return (fail_read && reads > 1) || ferror(file);
}
static int checked_close(FILE *file) {
    closes++;
    int status = fclose(file);
    return fail_close ? EOF : status;
}
#define fread checked_read
#define ferror checked_error
#define fclose checked_close
#include "runtime/file_bytes.h"
#undef fread
#undef ferror
#undef fclose

int main(void) {
    for (int mode = 0; mode < 3; mode++) {
        FILE *file = tmpfile();
        assert(file);
        for (int i = 0; i < 8193; i++) assert(fputc(i % 256, file) != EOF);
        rewind(file);
        fail_read = mode == 1;
        fail_close = mode == 2;
        reads = closes = 0;
        DynArray *bytes = nl_read_byte_stream(file);
        assert(bytes && bytes->elem_type == ELEM_U8);
        assert(closes == 1);
        assert(bytes->length == (mode == 0 ? 8193 : 0));
        for (int64_t i = 0; i < bytes->length; i++)
            assert(dyn_array_get_u8(bytes, i) == i % 256);
        gc_release(bytes);
    }
    closes = 0;
    DynArray *missing = nl_read_file_bytes(NULL);
    assert(missing && missing->elem_type == ELEM_U8 && missing->length == 0);
    assert(closes == 0);
    gc_release(missing);
    puts("I passed byte-stream, read-error and close-error checks.");
    return 0;
}
