/* I keep explicitly selected daemon endpoints separate from ambient defaults. */
#include "nanovm/vmd_protocol.h"
#include "nanovm/vmd_client.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char socket_path[256], pid_path[256], expected[256];
    unsetenv("NANOVMD_SOCKET");
    vmd_socket_path(socket_path, sizeof(socket_path));
    snprintf(expected, sizeof(expected), "/tmp/nanolang_vm_%u.sock", (unsigned)getuid());
    assert(!strcmp(socket_path, expected));
    char directory[] = "/tmp/nano-vmd-config-XXXXXX";
    assert(mkdtemp(directory));
    snprintf(expected, sizeof(expected), "%s/vm.sock", directory);
    assert(!setenv("NANOVMD_SOCKET", expected, 1));
    assert(!setenv("NANOVMD_NO_AUTOSTART", "1", 1));
    vmd_socket_path(socket_path, sizeof(socket_path));
    assert(!strcmp(socket_path, expected));
    vmd_pid_path(pid_path, sizeof(pid_path));
    assert(!strncmp(pid_path, expected, strlen(expected)));
    assert(!strcmp(pid_path + strlen(expected), ".pid"));
    assert(!vmd_connect(10));
    assert(access(pid_path, F_OK) != 0);
    assert(access(socket_path, F_OK) != 0);
    char small[4] = "abc";
    vmd_socket_path(small, sizeof(small));
    assert(!small[0]);
    vmd_pid_path(small, sizeof(small));
    assert(!small[0]);
    char oversized[512];
    memset(oversized, 'x', sizeof(oversized) - 1);
    oversized[sizeof(oversized) - 1] = 0;
    assert(!setenv("NANOVMD_SOCKET", oversized, 1));
    vmd_socket_path(socket_path, sizeof(socket_path));
    assert(!socket_path[0]);
    assert(!vmd_connect(10));
    assert(!setenv("NANOVMD_SOCKET", "", 1));
    assert(!vmd_connect(10));
    assert(!rmdir(directory));
    puts("I passed daemon endpoint configuration and no-autostart checks.");
    return 0;
}
