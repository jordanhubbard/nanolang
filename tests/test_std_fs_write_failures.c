#include "../modules/std/fs.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int fail_fwrite;
static int fail_fclose;
static int fclose_calls;
int g_argc;
char **g_argv;

size_t nano_test_fwrite(const void *ptr, size_t size, size_t count, FILE *stream) {
    return fail_fwrite ? 0 : fwrite(ptr, size, count, stream);
}

int nano_test_fclose(FILE *stream) {
    fclose_calls++;
    int result = fclose(stream);
    return fail_fclose ? EOF : result;
}

#define ASSERT(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAILED: %s at line %d\n", #condition, __LINE__); \
        return 1; \
    } \
} while (0)

int main(void) {
    char path[] = "/tmp/test_std_fs_write_failures.XXXXXX";
    int fd = mkstemp(path);
    ASSERT(fd >= 0);
    ASSERT(close(fd) == 0);

    fail_fwrite = 1;
    fclose_calls = 0;
    ASSERT(file_write(path, "content") == -1);
    ASSERT(fclose_calls == 1);
    fail_fwrite = 0;

    fail_fclose = 1;
    fclose_calls = 0;
    ASSERT(file_append(path, "content") == -1);
    ASSERT(fclose_calls == 1);
    fail_fclose = 0;

    remove(path);
    puts("std fs write failure tests passed");
    return 0;
}
