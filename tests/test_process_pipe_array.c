#define _POSIX_C_SOURCE 200809L
#include "../modules/std/process.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>

static int allocation, fail_alloc, pipes, fail_pipe, forks, fail_fork, fail_flags;
static int opened[4], opened_count;
static int fail_array;
static DynArray *array_alloc(ElementType type, int64_t capacity) {
    return fail_array ? NULL : dyn_array_new_with_capacity(type, capacity);
}
static void *allocate(size_t size) {
    return ++allocation == fail_alloc ? NULL : malloc(size);
}
static int create_pipe(int pair[2]) {
    if (++pipes == fail_pipe) { errno = EMFILE; return -1; }
    int rc = pipe(pair);
    if (!rc) { opened[opened_count++] = pair[0]; opened[opened_count++] = pair[1]; }
    return rc;
}
static pid_t spawn(void) { ++forks; return fail_fork ? -1 : fork(); }
static int set_flags(int fd, int cmd, int flags) {
    return (fail_flags == 2 || (fail_flags == 1 && cmd == F_SETFL)) ? -1 : fcntl(fd, cmd, flags);
}
#define malloc allocate
#define pipe create_pipe
#define fork spawn
#define fcntl set_flags
#define dyn_array_new_with_capacity array_alloc
#include "../modules/std/process.c"
#undef malloc
#undef pipe
#undef fork
#undef fcntl
#undef dyn_array_new_with_capacity

static void release_result(DynArray *a) {
    assert(a && a->length == 3 && a->elem_type == ELEM_STRING && a->elem_size == sizeof(char *));
    for (int i = 0; i < 3; ++i) free((void *)dyn_array_get_string(a, i));
    gc_release(a);
}
int main(void) {
    gc_init();
    assert(nl_os_process_spawn_with_pipes__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    fail_array = 1;
    assert(!nl_os_process_spawn_with_pipes("exit 0") && !pipes && !forks);
    fail_array = 0;
    for (int i = 1; i <= 3; ++i) {
        allocation = 0; fail_alloc = i;
        assert(!nl_os_process_spawn_with_pipes("exit 0"));
        assert(!pipes && !forks && gc_get_stats().num_objects == 0);
    }
    fail_alloc = 0;
    for (int failure = 0; failure < 5; ++failure) {
        pipes = forks = opened_count = 0;
        fail_pipe = failure < 2 ? failure + 1 : 0;
        fail_flags = failure == 2 ? 1 : failure == 4 ? 2 : 0;
        fail_fork = failure == 3;
        DynArray *a = nl_os_process_spawn_with_pipes("exit 0");
        for (int i = 0; i < 3; ++i) assert(!strcmp(dyn_array_get_string(a, i), "-1"));
        for (int i = 0; i < opened_count; ++i) assert(fcntl(opened[i], F_GETFD) == -1 && errno == EBADF);
        assert(forks == (failure == 3));
        release_result(a);
    }
    fail_pipe = fail_flags = fail_fork = 0;
    pipes = forks = opened_count = 0;
    DynArray *a = nl_os_process_spawn_with_pipes("printf out; printf err >&2");
    pid_t pid = (pid_t)strtol(dyn_array_get_string(a, 0), NULL, 10);
    int out = atoi(dyn_array_get_string(a, 1)), err = atoi(dyn_array_get_string(a, 2));
    assert(pid > 0 && out >= 0 && err >= 0);
    assert(fcntl(out, F_GETFL) & O_NONBLOCK);
    assert(fcntl(err, F_GETFL) & O_NONBLOCK);
    assert(fcntl(out, F_GETFD) & FD_CLOEXEC);
    assert(fcntl(err, F_GETFD) & FD_CLOEXEC);
    int status;
    assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status));
    char bytes[4] = {0};
    assert(read(out, bytes, 3) == 3 && !strcmp(bytes, "out"));
    assert(read(err, bytes, 3) == 3 && !strcmp(bytes, "err"));
    close(out); close(err); release_result(a);
    a = nl_os_process_spawn_with_pipes(NULL);
    assert(!strcmp(dyn_array_get_string(a, 0), "-1")); release_result(a);
    pid = fork();
    assert(pid >= 0);
    if (!pid) {
        close(STDOUT_FILENO); close(STDERR_FILENO);
        opened_count = 0;
        a = nl_os_process_spawn_with_pipes("printf ok");
        assert(a);
        pid_t nested = atoi(dyn_array_get_string(a, 0));
        out = atoi(dyn_array_get_string(a, 1)); err = atoi(dyn_array_get_string(a, 2));
        assert(nested > 0 && out > 2 && err > 2);
        assert(waitpid(nested, &status, 0) == nested && WIFEXITED(status) && !WEXITSTATUS(status));
        assert(read(out, bytes, 2) == 2 && bytes[0] == 'o' && bytes[1] == 'k');
        close(out); close(err); release_result(a);
        _exit(0);
    }
    assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status));
    assert(gc_get_stats().num_objects == 0);
    gc_shutdown();
    return 0;
}
