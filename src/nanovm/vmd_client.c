/*
 * vmd_client.c - NanoVM Daemon Client Library
 *
 * Connects to nano_vmd via Unix domain socket, with lazy-launch support.
 * If no daemon is running, forks and execs nano_vmd, then polls for the
 * socket to appear (with flock-based race protection).
 */

#define _GNU_SOURCE  /* For strdup(), readlink(), flock() */

#include "vmd_client.h"
#include "vmd_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/file.h>
#include <sys/wait.h>

/* ========================================================================
 * Client handle
 * ======================================================================== */

struct VmdClient {
    int fd;
};

/* ========================================================================
 * Connection
 * ======================================================================== */

static int try_connect(const char *sock_path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    return fd;
}

/* Find the nano_vmd binary. Searches:
 * 1. Same directory as current executable
 * 2. PATH via execlp fallback */
static bool find_daemon_binary(char *buf, size_t size) {
#ifdef __APPLE__
    extern int _NSGetExecutablePath(char *, uint32_t *);
    char exe_path[4096];
    uint32_t exe_size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &exe_size) == 0) {
        /* Replace last component with nano_vmd */
        char *slash = strrchr(exe_path, '/');
        if (slash) {
            slash[1] = '\0';
            snprintf(buf, size, "%snano_vmd", exe_path);
            if (access(buf, X_OK) == 0) return true;
        }
    }
#else
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        char *slash = strrchr(exe_path, '/');
        if (slash) {
            slash[1] = '\0';
            snprintf(buf, size, "%snano_vmd", exe_path);
            if (access(buf, X_OK) == 0) return true;
        }
    }
#endif
    /* Fallback: assume it's in PATH */
    snprintf(buf, size, "nano_vmd");
    return true;
}

static bool launch_daemon(const char *pid_path) {
    /* Use flock on PID file to prevent multiple launchers racing */
    int lock_fd = open(pid_path, O_CREAT | O_WRONLY, 0600);
    if (lock_fd < 0) return false;

    if (flock(lock_fd, LOCK_EX | LOCK_NB) < 0) {
        /* Another process is launching — just wait for it */
        close(lock_fd);
        return true;  /* Caller will poll for socket */
    }

    char daemon_bin[4096];
    find_daemon_binary(daemon_bin, sizeof(daemon_bin));

    pid_t pid = fork();
    if (pid < 0) {
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
        return false;
    }

    if (pid == 0) {
        /* Child: exec the daemon */
        flock(lock_fd, LOCK_UN);
        close(lock_fd);

        /* Detach from controlling terminal */
        setsid();

        /* Redirect stdio to /dev/null */
        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            /* Keep stderr for diagnostics */
            close(devnull);
        }

        execlp(daemon_bin, daemon_bin, (char *)NULL);
        _exit(127);  /* exec failed */
    }

    /* Parent: release lock and wait for daemon to start */
    flock(lock_fd, LOCK_UN);
    close(lock_fd);

    /* Don't wait for child — it's the daemon */
    return true;
}

VmdClient *vmd_connect(int timeout_ms) {
    char sock_path[256], pid_path[256];
    vmd_socket_path(sock_path, sizeof(sock_path));
    vmd_pid_path(pid_path, sizeof(pid_path));

    /* Try connecting first */
    int fd = try_connect(sock_path);
    if (fd >= 0) goto connected;

    /* Launch daemon and poll for socket */
    if (!launch_daemon(pid_path)) {
        fprintf(stderr, "[vmd-client] Failed to launch daemon\n");
        return NULL;
    }

    /* Poll for socket availability */
    int elapsed = 0;
    int interval = 50;  /* ms */
    while (elapsed < timeout_ms) {
        usleep((useconds_t)(interval * 1000));
        elapsed += interval;

        fd = try_connect(sock_path);
        if (fd >= 0) goto connected;

        /* Back off slightly */
        if (interval < 200) interval += 50;
    }

    fprintf(stderr, "[vmd-client] Timeout waiting for daemon\n");
    return NULL;

connected: {
    VmdClient *client = malloc(sizeof(VmdClient));
    if (!client) {
        close(fd);
        return NULL;
    }
    client->fd = fd;
    return client;
    }
}

/* ========================================================================
 * Execution
 * ======================================================================== */

int vmd_execute(VmdClient *client, const uint8_t *blob, uint32_t blob_size) {
    if (!client || client->fd < 0) return -1;

    /* Send LOAD_EXEC with .nvm blob */
    if (!vmd_msg_send(client->fd, VMD_MSG_LOAD_EXEC, blob, blob_size)) {
        return -1;
    }

    /* Receive responses until EXIT_CODE */
    for (;;) {
        VmdMsgHeader hdr;
        if (!vmd_msg_recv_header(client->fd, &hdr)) {
            return -1;
        }

        switch (hdr.msg_type) {
        case VMD_MSG_OUTPUT: {
            if (hdr.payload_len == 0) break;
            /* Stream output to stdout in chunks */
            char buf[8192];
            uint32_t remaining = hdr.payload_len;
            while (remaining > 0) {
                uint32_t chunk = remaining < sizeof(buf) ? remaining : sizeof(buf);
                if (!vmd_msg_recv_payload(client->fd, buf, chunk)) return -1;
                fwrite(buf, 1, chunk, stdout);
                remaining -= chunk;
            }
            break;
        }

        case VMD_MSG_ERROR: {
            if (hdr.payload_len == 0) break;
            char *msg = malloc(hdr.payload_len + 1);
            if (!msg) return -1;
            if (!vmd_msg_recv_payload(client->fd, msg, hdr.payload_len)) {
                free(msg);
                return -1;
            }
            msg[hdr.payload_len] = '\0';
            fprintf(stderr, "%s\n", msg);
            free(msg);
            break;
        }

        case VMD_MSG_EXIT_CODE: {
            if (hdr.payload_len != sizeof(int32_t)) return -1;
            int32_t code;
            if (!vmd_msg_recv_payload(client->fd, &code, sizeof(code))) return -1;
            /* Convert from little-endian */
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
            code = (int32_t)__builtin_bswap32((uint32_t)code);
#endif
            return (int)code;
        }

        default:
            /* Skip unknown message payloads */
            if (hdr.payload_len > 0) {
                char skip[4096];
                uint32_t remaining = hdr.payload_len;
                while (remaining > 0) {
                    uint32_t chunk = remaining < sizeof(skip) ? remaining : sizeof(skip);
                    if (!vmd_msg_recv_payload(client->fd, skip, chunk)) return -1;
                    remaining -= chunk;
                }
            }
            break;
        }
    }
}

/* ========================================================================
 * Utilities
 * ======================================================================== */

void vmd_disconnect(VmdClient *client) {
    if (!client) return;
    if (client->fd >= 0) close(client->fd);
    free(client);
}

bool vmd_ping(VmdClient *client) {
    if (!client || client->fd < 0) return false;

    if (!vmd_msg_send_simple(client->fd, VMD_MSG_PING))
        return false;

    VmdMsgHeader hdr;
    if (!vmd_msg_recv_header(client->fd, &hdr))
        return false;

    return hdr.msg_type == VMD_MSG_PONG;
}
