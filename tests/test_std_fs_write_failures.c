#include "../modules/std/fs.h"

#include <stdio.h>
#include <stdlib.h>

static int fail_fwrite;
static int fail_fclose;
static int fclose_calls;
int g_argc;
char **g_argv;

size_t __real_fwrite(const void *ptr, size_t size, size_t count, FILE *stream);
int __real_fclose(FILE *stream);

size_t __wrap_fwrite(const void *ptr, size_t size, size_t count, FILE *stream) {
    return fail_fwrite ? 0 : __real_fwrite(ptr, size, count, stream);
}

int __wrap_fclose(FILE *stream) {
    fclose_calls++;
    int result = __real_fclose(stream);
    return fail_fclose ? EOF : result;
}

#define ASSERT(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAILED: %s at line %d\n", #condition, __LINE__); \
        return 1; \
    } \
} while (0)

int main(void) {
    const char *path = "/tmp/test_std_fs_write_failures.txt";

    fail_fwrite = 1;
    fclose_calls = 0;
    ASSERT(file_write(path, "content") == -1);
    ASSERT(fclose_calls == 1);
    fail_fwrite = 0;

    fail_fclose = 1;
    ASSERT(file_append(path, "content") == -1);
    fail_fclose = 0;

    remove(path);
    puts("std fs write failure tests passed");
    return 0;
}
