#define _POSIX_C_SOURCE 200809L
#include "../modules/std/process.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>

static int temp_calls, fail_temp, fork_calls, fail_fork, reads, fail_read;
static int allocations, fail_allocation;
static void *allocate(size_t size) { return ++allocations == fail_allocation ? NULL : malloc(size); }
static FILE *temporary(void) { return ++temp_calls == fail_temp ? NULL : tmpfile(); }
static pid_t spawn(void) { ++fork_calls; return fail_fork ? -1 : fork(); }
static ssize_t capture_read(int fd, void *data, size_t size, off_t offset) {
    if (++reads == fail_read) { errno = EIO; return -1; }
    return pread(fd, data, size, offset);
}
#define tmpfile temporary
#define fork spawn
#define pread capture_read
#define malloc allocate
#include "../modules/std/process.c"
#undef tmpfile
#undef fork
#undef pread
#undef malloc

static void release_result(DynArray *a) {
    assert(a && a->length == 3 && a->elem_type == ELEM_STRING && a->elem_size == sizeof(char *));
    for (int i = 0; i < 3; ++i) free((void *)dyn_array_get_string(a, i));
    gc_release(a);
}
static void expect(const char *command, const char *status, const char *out, const char *err) {
    DynArray *a = nl_os_process_run(command);
    assert(a && !strcmp(dyn_array_get_string(a, 0), status));
    assert(!strcmp(dyn_array_get_string(a, 1), out));
    assert(!strcmp(dyn_array_get_string(a, 2), err));
    release_result(a);
}
int main(void) {
    gc_init();
    assert(nl_os_process_run__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    expect("printf first; printf second; printf error >&2; exit 7", "7", "firstsecond", "error");
    char command[6200];
    memset(command, ' ', 6000);
    strcpy(command + 6000, "printf long");
    expect(command, "0", "long", "");
    expect("kill -TERM $$", "-1", "", "");
    DynArray *a = nl_os_process_run("i=0; while [ $i -lt 9000 ]; do printf abcdefgh; i=$((i+1)); done");
    assert(a && !strcmp(dyn_array_get_string(a, 0), "0"));
    assert(strlen(dyn_array_get_string(a, 1)) == 72000);
    release_result(a);
    a = nl_os_process_run(NULL);
    assert(a && !strcmp(dyn_array_get_string(a, 0), "-1")); release_result(a);
    a = nl_os_process_run("printf '\\000'");
    assert(a && !strcmp(dyn_array_get_string(a, 0), "-1")); release_result(a);
    for (int failure = 1; failure <= 2; ++failure) {
        temp_calls = fork_calls = 0; fail_temp = failure;
        a = nl_os_process_run("exit 0");
        assert(a && !strcmp(dyn_array_get_string(a, 0), "-1") && !fork_calls);
        release_result(a);
    }
    fail_temp = 0; fail_fork = 1;
    a = nl_os_process_run("exit 0");
    assert(a && !strcmp(dyn_array_get_string(a, 0), "-1")); release_result(a);
    fail_fork = 0; reads = 0; fail_read = 1;
    a = nl_os_process_run("printf data");
    assert(a && !strcmp(dyn_array_get_string(a, 0), "-1")); release_result(a);
    fail_read = 0;
    for (int failure = 1; failure <= 3; ++failure) {
        allocations = 0; fail_allocation = failure;
        a = nl_os_process_run("printf data");
        if (failure == 3) assert(!a);
        else { assert(a && !strcmp(dyn_array_get_string(a, 0), "-1")); release_result(a); }
        assert(gc_get_stats().num_objects == 0);
    }
    fail_allocation = 0;
    close(STDOUT_FILENO); close(STDERR_FILENO);
    expect("printf closed", "0", "closed", "");
    assert(gc_get_stats().num_objects == 0);
    gc_shutdown();
    return 0;
}
