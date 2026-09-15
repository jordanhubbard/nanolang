#include "vmd_server.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int bind_calls;
static char bound_path[sizeof(((struct sockaddr_un *)0)->sun_path)];
int g_argc;
char **g_argv;

static int fake_bind(int fd, const struct sockaddr *address, socklen_t length) {
    (void)fd;
    (void)length;
    const struct sockaddr_un *unix_address = (const struct sockaddr_un *)address;
    bind_calls++;
    memcpy(bound_path, unix_address->sun_path, sizeof(bound_path));
    return -1;
}

static void fill_path(char *path, size_t length) {
    assert(length > 5);
    memcpy(path, "/tmp/", 5);
    memset(path + 5, 'x', length - 5);
    path[length] = '\0';
}

static int run_with_path(const char *socket_path, const char *pid_path) {
    VmdServerConfig config = {
        .idle_timeout_sec = 0,
        .foreground = true,
        .verbose = false,
        .socket_path = socket_path,
        .pid_path = pid_path,
        .bind_fn = fake_bind,
    };
    return vmd_server_run(&config);
}

int main(void) {
    const size_t capacity = sizeof(((struct sockaddr_un *)0)->sun_path);
    char path[sizeof(((struct sockaddr_un *)0)->sun_path) + 2];
    char pid_path[128];
    snprintf(pid_path, sizeof(pid_path), "/tmp/nanolang_vmd_path_test_%ld.pid", (long)getpid());

    fill_path(path, capacity - 1);
    bind_calls = 0;
    memset(bound_path, 0, sizeof(bound_path));
    assert(run_with_path(path, pid_path) == 1);
    assert(bind_calls == 1);
    assert(memcmp(bound_path, path, capacity) == 0);

    for (size_t length = capacity; length <= capacity + 1; length++) {
        fill_path(path, length);
        FILE *old_socket = fopen(path, "w");
        if (old_socket) fclose(old_socket);
        bind_calls = 0;
        assert(run_with_path(path, pid_path) == 1);
        assert(bind_calls == 0);
        assert(access(path, F_OK) == 0);
        unlink(path);
    }

    unlink(pid_path);
    puts("vmd server path tests passed");
    return 0;
}
