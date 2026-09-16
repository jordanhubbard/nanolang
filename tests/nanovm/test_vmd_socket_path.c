/* I exercise the real server's path boundary before any unlink or bind. */
#include "nanovm/vmd_server.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int g_argc;
char **g_argv;
static char selected_path[256];
static int bound;

static void test_socket_path(char *buffer, size_t size) {
    assert(strlen(selected_path) < size);
    memcpy(buffer, selected_path, strlen(selected_path) + 1);
}
static void test_pid_path(char *buffer, size_t size) {
    assert(size);
    buffer[0] = 0;
}
static int test_bind(int fd, const struct sockaddr *address, socklen_t length) {
    (void)fd; (void)length;
    const struct sockaddr_un *local = (const struct sockaddr_un *)address;
    assert(!strcmp(local->sun_path, selected_path));
    bound++;
    errno = EADDRINUSE;
    return -1;
}
#define vmd_socket_path test_socket_path
#define vmd_pid_path test_pid_path
#define bind test_bind
#include "../../src/nanovm/vmd_server.c"
#undef bind
#undef vmd_pid_path
#undef vmd_socket_path

int main(void) {
    char directory[] = "/tmp/nano-vmd-path-XXXXXX";
    assert(mkdtemp(directory));
    size_t capacity = sizeof(((struct sockaddr_un *)0)->sun_path);
    for (size_t size = capacity - 1; size <= capacity + 1; size++) {
        size_t prefix = strlen(directory);
        memcpy(selected_path, directory, prefix);
        selected_path[prefix++] = '/';
        assert(size > prefix && size < sizeof(selected_path));
        memset(selected_path + prefix, 'x', size - prefix);
        selected_path[size] = 0;
        if (size >= capacity) {
            FILE *file = fopen(selected_path, "wb");
            assert(file && fputs("preserved", file) >= 0 && !fclose(file));
        }
        bound = 0;
        VmdServerConfig config = {.foreground = true};
        assert(vmd_server_run(&config) == 1);
        assert(bound == (size < capacity));
        if (size >= capacity) {
            char contents[10] = {0};
            FILE *file = fopen(selected_path, "rb");
            assert(file && fread(contents, 1, 9, file) == 9 && !fclose(file));
            assert(!strcmp(contents, "preserved"));
            assert(!unlink(selected_path));
        }
    }
    assert(!rmdir(directory));
    puts("I passed all three daemon socket-path boundaries.");
    return 0;
}
