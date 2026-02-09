/*
 * vmd_server.c - NanoVM Daemon Server Implementation
 *
 * Accepts connections on a Unix domain socket. Each client connection
 * spawns a thread that deserializes the .nvm blob, runs it through
 * vm_execute() with output redirected over the socket, and sends
 * the exit code back.
 *
 * Key insight: vm_call_function() already uses vm_out(vm) for TRAP_PRINT.
 * We create a custom FILE* (via funopen) that sends VMD_MSG_OUTPUT messages
 * over the socket, then set vm->output to that FILE*. The core execution
 * path is completely unchanged — validating the co-processor trap model.
 */

#include "vmd_server.h"
#include "vmd_protocol.h"
#include "vm.h"
#include "vm_ffi.h"
#include "../nanoisa/nvm_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <fcntl.h>

/* ========================================================================
 * Globals
 * ======================================================================== */

static volatile sig_atomic_t g_shutdown = 0;
static pthread_mutex_t g_ffi_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_client_count_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_active_clients = 0;

/* ========================================================================
 * Signal handling
 * ======================================================================== */

static void signal_handler(int sig) {
    (void)sig;
    g_shutdown = 1;
}

static void setup_signals(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    /* Ignore SIGPIPE — we handle write errors via return codes */
    sa.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa, NULL);
}

/* ========================================================================
 * Socket-backed FILE* for output streaming
 *
 * On macOS we use funopen(); on Linux we use fopencookie().
 * The write callback sends each chunk as a VMD_MSG_OUTPUT message.
 * ======================================================================== */

typedef struct {
    int fd;
} SocketCookie;

#ifdef __APPLE__

static int socket_write(void *cookie, const char *buf, int len) {
    SocketCookie *sc = cookie;
    if (len > 0) {
        vmd_msg_send_output(sc->fd, buf, (uint32_t)len);
    }
    return len;
}

static int socket_close(void *cookie) {
    free(cookie);
    return 0;
}

static FILE *socket_fopen(int fd) {
    SocketCookie *sc = malloc(sizeof(SocketCookie));
    if (!sc) return NULL;
    sc->fd = fd;
    FILE *f = funopen(sc, NULL, socket_write, NULL, socket_close);
    if (!f) { free(sc); return NULL; }
    /* Line-buffered so each print statement flushes */
    setvbuf(f, NULL, _IOLBF, 0);
    return f;
}

#else /* Linux */

static ssize_t socket_write_cookie(void *cookie, const char *buf, size_t len) {
    SocketCookie *sc = cookie;
    if (len > 0) {
        vmd_msg_send_output(sc->fd, buf, (uint32_t)len);
    }
    return (ssize_t)len;
}

static int socket_close_cookie(void *cookie) {
    free(cookie);
    return 0;
}

static FILE *socket_fopen(int fd) {
    SocketCookie *sc = malloc(sizeof(SocketCookie));
    if (!sc) return NULL;
    sc->fd = fd;
    cookie_io_functions_t funcs = {
        .read  = NULL,
        .write = socket_write_cookie,
        .seek  = NULL,
        .close = socket_close_cookie,
    };
    FILE *f = fopencookie(sc, "w", funcs);
    if (!f) { free(sc); return NULL; }
    setvbuf(f, NULL, _IOLBF, 0);
    return f;
}

#endif

/* ========================================================================
 * Per-client execution (runs in its own thread)
 * ======================================================================== */

typedef struct {
    int client_fd;
    bool verbose;
} ClientCtx;

static void *client_thread(void *arg) {
    ClientCtx *ctx = arg;
    int fd = ctx->client_fd;
    bool verbose = ctx->verbose;
    free(ctx);

    pthread_mutex_lock(&g_client_count_mutex);
    g_active_clients++;
    pthread_mutex_unlock(&g_client_count_mutex);

    /* Read one message from the client */
    VmdMsgHeader hdr;
    if (!vmd_msg_recv_header(fd, &hdr)) {
        if (verbose) fprintf(stderr, "[vmd] Client read error\n");
        goto done;
    }

    switch (hdr.msg_type) {
    case VMD_MSG_PING:
        vmd_msg_send_simple(fd, VMD_MSG_PONG);
        break;

    case VMD_MSG_SHUTDOWN:
        g_shutdown = 1;
        vmd_msg_send_simple(fd, VMD_MSG_PONG);
        break;

    case VMD_MSG_STATUS: {
        pthread_mutex_lock(&g_client_count_mutex);
        int n = g_active_clients;
        pthread_mutex_unlock(&g_client_count_mutex);
        char status[128];
        snprintf(status, sizeof(status), "active_clients=%d", n);
        vmd_msg_send(fd, VMD_MSG_STATUS_RSP, status, (uint32_t)strlen(status));
        break;
    }

    case VMD_MSG_LOAD_EXEC: {
        if (hdr.payload_len == 0 || hdr.payload_len > VMD_MAX_PAYLOAD) {
            vmd_msg_send_error(fd, "Invalid payload size");
            break;
        }

        /* Read .nvm blob */
        uint8_t *blob = malloc(hdr.payload_len);
        if (!blob) {
            vmd_msg_send_error(fd, "Out of memory");
            break;
        }

        if (!vmd_msg_recv_payload(fd, blob, hdr.payload_len)) {
            free(blob);
            vmd_msg_send_error(fd, "Payload read error");
            break;
        }

        /* Deserialize */
        NvmModule *module = nvm_deserialize(blob, hdr.payload_len);
        free(blob);

        if (!module) {
            vmd_msg_send_error(fd, "Invalid .nvm format");
            break;
        }

        if (verbose) {
            fprintf(stderr, "[vmd] Executing program (%u bytes, %u functions)\n",
                    hdr.payload_len, module->function_count);
        }

        /* Initialize FFI for this module's imports */
        if (module->header.flags & NVM_FLAG_NEEDS_EXTERN) {
            pthread_mutex_lock(&g_ffi_mutex);
            vm_ffi_init();
            for (uint32_t i = 0; i < module->import_count; i++) {
                const char *mod_name = nvm_get_string(module,
                                        module->imports[i].module_name_idx);
                if (mod_name && mod_name[0]) {
                    vm_ffi_load_module(mod_name);
                }
            }
            pthread_mutex_unlock(&g_ffi_mutex);
        }

        /* Create socket-backed FILE* for output streaming */
        FILE *sock_out = socket_fopen(fd);

        /* Execute — vm_execute uses vm_out(vm) which returns vm->output */
        VmState vm;
        vm_init(&vm, module);
        vm.output = sock_out;  /* Redirect output over socket */

        VmResult result = vm_execute(&vm);

        /* Flush any remaining output */
        if (sock_out) fflush(sock_out);

        int32_t exit_code = 0;
        if (result != VM_OK) {
            exit_code = 1;
            /* Format error like standalone: "Runtime error: <type>\n  <detail>" */
            char errbuf[512];
            if (vm.error_msg[0]) {
                snprintf(errbuf, sizeof(errbuf), "Runtime error: %s\n  %s",
                         vm_error_string(result), vm.error_msg);
            } else {
                snprintf(errbuf, sizeof(errbuf), "Runtime error: %s",
                         vm_error_string(result));
            }
            vmd_msg_send_error(fd, errbuf);
        }

        vmd_msg_send_exit(fd, exit_code);

        vm_destroy(&vm);
        if (sock_out) fclose(sock_out);
        nvm_module_free(module);
        break;
    }

    default:
        vmd_msg_send_error(fd, "Unknown message type");
        break;
    }

done:
    close(fd);
    pthread_mutex_lock(&g_client_count_mutex);
    g_active_clients--;
    pthread_mutex_unlock(&g_client_count_mutex);
    return NULL;
}

/* ========================================================================
 * PID file management
 * ======================================================================== */

static bool write_pid_file(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return false;
    fprintf(f, "%d\n", (int)getpid());
    fclose(f);
    return true;
}

static void remove_pid_file(const char *path) {
    unlink(path);
}

/* Check if a daemon is already running. Returns its PID or 0. */
static pid_t check_pid_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int pid = 0;
    if (fscanf(f, "%d", &pid) != 1) {
        fclose(f);
        return 0;
    }
    fclose(f);
    if (pid > 0 && kill((pid_t)pid, 0) == 0) {
        return (pid_t)pid;
    }
    return 0;
}

/* ========================================================================
 * Main server loop
 * ======================================================================== */

int vmd_server_run(const VmdServerConfig *cfg) {
    char sock_path[256], pid_file[256];
    vmd_socket_path(sock_path, sizeof(sock_path));
    vmd_pid_path(pid_file, sizeof(pid_file));

    /* Check for existing daemon */
    pid_t existing = check_pid_file(pid_file);
    if (existing) {
        fprintf(stderr, "[vmd] Daemon already running (pid %d)\n", (int)existing);
        return 1;
    }

    setup_signals();

    /* Create socket */
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("[vmd] socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    /* Remove stale socket */
    unlink(sock_path);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[vmd] bind");
        close(server_fd);
        return 1;
    }

    /* Restrict socket to owner only */
    chmod(sock_path, 0700);

    if (listen(server_fd, 16) < 0) {
        perror("[vmd] listen");
        close(server_fd);
        unlink(sock_path);
        return 1;
    }

    /* Write PID file */
    if (!write_pid_file(pid_file)) {
        perror("[vmd] pid file");
        close(server_fd);
        unlink(sock_path);
        return 1;
    }

    if (cfg->foreground || cfg->verbose) {
        fprintf(stderr, "[vmd] Listening on %s (pid %d)\n", sock_path, (int)getpid());
    }

    /* Accept loop with poll() for idle timeout */
    int idle_timeout_ms = cfg->idle_timeout_sec > 0
                        ? cfg->idle_timeout_sec * 1000
                        : -1;  /* infinite */

    while (!g_shutdown) {
        struct pollfd pfd = { .fd = server_fd, .events = POLLIN };
        int nready = poll(&pfd, 1, idle_timeout_ms);

        if (nready < 0) {
            if (errno == EINTR) continue;
            perror("[vmd] poll");
            break;
        }

        if (nready == 0) {
            /* Idle timeout — check if any clients are still active */
            pthread_mutex_lock(&g_client_count_mutex);
            int active = g_active_clients;
            pthread_mutex_unlock(&g_client_count_mutex);

            if (active == 0) {
                if (cfg->verbose) {
                    fprintf(stderr, "[vmd] Idle timeout, shutting down\n");
                }
                break;
            }
            continue;
        }

        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("[vmd] accept");
            continue;
        }

        ClientCtx *ctx = malloc(sizeof(ClientCtx));
        if (!ctx) {
            close(client_fd);
            continue;
        }
        ctx->client_fd = client_fd;
        ctx->verbose = cfg->verbose;

        pthread_t tid;
        if (pthread_create(&tid, NULL, client_thread, ctx) != 0) {
            perror("[vmd] pthread_create");
            close(client_fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }

    /* Cleanup */
    close(server_fd);
    unlink(sock_path);
    remove_pid_file(pid_file);

    if (cfg->foreground || cfg->verbose) {
        fprintf(stderr, "[vmd] Shutdown complete\n");
    }

    return 0;
}
