#if defined(__APPLE__)
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#endif

#if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__) || defined(__OpenBSD__)
#ifndef __BSD_VISIBLE
#define __BSD_VISIBLE 1
#endif
#endif

#define _XOPEN_SOURCE 700
#define _GNU_SOURCE

#include "pty.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <termios.h>

#ifdef __linux__
#  include <pty.h>
#elif defined(__APPLE__)
#  include <util.h>
#endif

#define READ_BUF_SIZE 4096

/* Persistent read buffer returned by nl_pty_read.
 * Sized to READ_BUF_SIZE + 1 for the NUL terminator. */
static char nl_pty_read_buf[READ_BUF_SIZE + 1];

int64_t nl_pty_open(int64_t rows, int64_t cols) {
    int master_fd = posix_openpt(O_RDWR | O_NOCTTY);
    if (master_fd < 0) return -1;

    if (grantpt(master_fd) < 0 || unlockpt(master_fd) < 0) {
        close(master_fd);
        return -1;
    }

    /* Set initial terminal dimensions */
    struct winsize ws = {
        .ws_row = (unsigned short)rows,
        .ws_col = (unsigned short)cols,
        .ws_xpixel = 0,
        .ws_ypixel = 0
    };
    ioctl(master_fd, TIOCSWINSZ, &ws);

    /* Put master in non-blocking mode */
    int flags = fcntl(master_fd, F_GETFL, 0);
    fcntl(master_fd, F_SETFL, flags | O_NONBLOCK);

    return (int64_t)master_fd;
}

/* Split args_str on spaces into a NULL-terminated argv array.
 * Returns pointer to a static buffer — single-use only. */
static char **build_argv(const char *prog, const char *args_str) {
    static char arg_copy[4096];
    static char *argv_buf[128];

    argv_buf[0] = (char *)prog;
    int argc = 1;

    if (args_str && args_str[0] != '\0') {
        strncpy(arg_copy, args_str, sizeof(arg_copy) - 1);
        arg_copy[sizeof(arg_copy) - 1] = '\0';

        char *tok = strtok(arg_copy, " ");
        while (tok && argc < 126) {
            argv_buf[argc++] = tok;
            tok = strtok(NULL, " ");
        }
    }

    argv_buf[argc] = NULL;
    return argv_buf;
}

int64_t nl_pty_fork_exec(int64_t master_fd, const char *prog,
                          const char *args_str,
                          const char *envkey, const char *envval) {
    char *slave_name = ptsname((int)master_fd);
    if (!slave_name) return -1;

    pid_t pid = fork();
    if (pid < 0) return -1;

    if (pid == 0) {
        /* Child */
        close((int)master_fd);

        /* New session so we can acquire a controlling terminal */
        if (setsid() < 0) _exit(1);

        int slave_fd = open(slave_name, O_RDWR);
        if (slave_fd < 0) _exit(1);

#ifdef TIOCSCTTY
        ioctl(slave_fd, TIOCSCTTY, 0);
#endif

        dup2(slave_fd, STDIN_FILENO);
        dup2(slave_fd, STDOUT_FILENO);
        dup2(slave_fd, STDERR_FILENO);
        if (slave_fd > STDERR_FILENO) close(slave_fd);

        /* Set optional environment variable */
        if (envkey && envkey[0] != '\0') {
            setenv(envkey, envval ? envval : "", 1);
        }

        /* Make sure TERM is set to something the child recognises */
        if (!getenv("TERM")) setenv("TERM", "ansi", 1);

        char **argv = build_argv(prog, args_str);
        execvp(prog, argv);
        _exit(127);
    }

    /* Parent — return child PID */
    return (int64_t)pid;
}

const char *nl_pty_read(int64_t master_fd) {
    ssize_t n = read((int)master_fd, nl_pty_read_buf, READ_BUF_SIZE);
    if (n <= 0) {
        nl_pty_read_buf[0] = '\0';
        return nl_pty_read_buf;
    }
    nl_pty_read_buf[n] = '\0';
    return nl_pty_read_buf;
}

int64_t nl_pty_write(int64_t master_fd, const char *data) {
    if (!data || data[0] == '\0') return 0;
    ssize_t n = write((int)master_fd, data, strlen(data));
    return (int64_t)n;
}

void nl_pty_resize(int64_t master_fd, int64_t pid, int64_t rows, int64_t cols) {
    struct winsize ws = {
        .ws_row = (unsigned short)rows,
        .ws_col = (unsigned short)cols,
        .ws_xpixel = 0,
        .ws_ypixel = 0
    };
    ioctl((int)master_fd, TIOCSWINSZ, &ws);
    if (pid > 0) kill((pid_t)pid, SIGWINCH);
}

void nl_pty_close(int64_t master_fd) {
    close((int)master_fd);
}

int64_t nl_pty_is_alive(int64_t pid) {
    if (pid <= 0) return 0;
    int status = 0;
    pid_t result = waitpid((pid_t)pid, &status, WNOHANG);
    if (result == 0) return 1;   /* still running */
    return 0;                    /* exited or error */
}
