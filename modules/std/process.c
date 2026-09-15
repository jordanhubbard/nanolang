#define _POSIX_C_SOURCE 200809L
#include "process.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <errno.h>

/* Use runtime DynArray API */
extern DynArray* dyn_array_new_with_capacity(ElementType elem_type, int64_t initial_capacity);
extern DynArray* dyn_array_push_string_copy(DynArray* arr, const char* value);

NANO_EXPORT_ARRAY_ABI(nl_os_process_spawn_with_pipes);
NANO_EXPORT_ARRAY_ABI(nl_os_process_run);

/* Run a command and capture stdout/stderr
 * Returns array<string> with [exit_code, stdout, stderr]
 */
static DynArray *run_result(DynArray *result, int code, char *out, char *err) {
    char *status = malloc(32);
    if (!status || !out || !err) {
        free(status); free(out); free(err); gc_release(result);
        return NULL;
    }
    snprintf(status, 32, "%d", code);
    dyn_array_push_string(result, status);
    dyn_array_push_string(result, out);
    dyn_array_push_string(result, err);
    return result;
}

static DynArray *run_error(DynArray *result, const char *message) {
    return run_result(result, -1, strdup(""), strdup(message));
}

/* I retain anonymous temporary-file descriptors, never reopen capture paths.
 * Moving them above stderr also handles callers with closed standard fds. */
static int capture_file(void) {
    FILE *file = tmpfile();
    if (!file) return -1;
    int fd = fcntl(fileno(file), F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    fclose(file);
    return fd;
}

static char *read_capture(int fd) {
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 0 || (uintmax_t)st.st_size >= SIZE_MAX) return NULL;
    size_t length = (size_t)st.st_size;
    char *text = malloc(length + 1);
    if (!text) return NULL;
    size_t done = 0;
    while (done < length) {
        size_t chunk = length - done;
        if (chunk > 65536) chunk = 65536;
        ssize_t n = pread(fd, text + done, chunk, (off_t)done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { free(text); return NULL; }
        done += (size_t)n;
    }
    if (memchr(text, 0, length)) { free(text); return NULL; }
    text[length] = 0;
    return text;
}

DynArray* nl_os_process_run(const char* command) {
    DynArray *result = dyn_array_new_with_capacity(ELEM_STRING, 3);
    if (!result) return NULL;
    if (!command) return run_error(result, "I require a command");
    int out = capture_file(), err = capture_file();
    if (out < 0 || err < 0) {
        if (out >= 0) close(out);
        if (err >= 0) close(err);
        return run_error(result, "I could not create capture files");
    }
    pid_t child = fork();
    if (child == 0) {
        if (dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0) _exit(127);
        close(out); close(err);
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
        _exit(127);
    }
    if (child < 0) {
        close(out); close(err);
        return run_error(result, "I could not start the command");
    }
    int status;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    if (waited != child) {
        close(out); close(err);
        return run_error(result, "I could not wait for the command");
    }
    char *stdout_text = read_capture(out), *stderr_text = read_capture(err);
    close(out); close(err);
    if (!stdout_text || !stderr_text) {
        free(stdout_text); free(stderr_text);
        return run_error(result, "I could not read captured text (I reject embedded NUL bytes)");
    }
    return run_result(result, WIFEXITED(status) ? WEXITSTATUS(status) : -1,
                      stdout_text, stderr_text);
}


/* Spawn a process non-blocking
 * Returns process ID (pid) or -1 on error
 */
int64_t nl_os_process_spawn(const char* command) {
    pid_t pid = fork();
    
    if (pid < 0) {
        /* Fork failed */
        return -1;
    } else if (pid == 0) {
        /* Child process */
        setpgid(0, 0);
        /* Use sh -c to execute command string */
        execl("/bin/sh", "sh", "-c", command, (char*)NULL);
        /* If exec fails, exit child */
        _exit(127);
    } else {
        /* Parent process - return child PID */
        setpgid(pid, pid);
        return (int64_t)pid;
    }
}

int64_t nl_os_process_kill_group(int64_t pid, int64_t signal_number) {
    if (pid <= 0 || signal_number <= 0) return -1;
    return kill(-(pid_t)pid, (int)signal_number) == 0 ? 0 : -1;
}

/* Check if a process is still running
 * Returns 1 if running, 0 if exited, -1 on error
 */
int64_t nl_os_process_is_running(int64_t pid) {
    if (pid <= 0) return -1;

    /* Observe termination without consuming the status. The caller follows
     * this probe with nl_os_process_wait(), which owns the actual reap and
     * must still be able to return the child's real exit code. */
    siginfo_t info;
    memset(&info, 0, sizeof(info));
    if (waitid(P_PID, (id_t)pid, &info, WEXITED | WNOHANG | WNOWAIT) != 0) {
        return -1;
    }
    return info.si_pid == 0 ? 1 : 0;
}

/* Wait for a process to complete
 * Returns exit code of the process, or -1 on error
 */
int64_t nl_os_process_wait(int64_t pid) {
    if (pid <= 0) return -1;

    int status;
    pid_t result = waitpid((pid_t)pid, &status, 0);

    if (result == (pid_t)pid) {
        if (WIFEXITED(status)) {
            return (int64_t)WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            /* Process terminated by signal - return negative signal number */
            return -(int64_t)WTERMSIG(status);
        } else {
            return -1;
        }
    } else {
        /* waitpid failed */
        return -1;
    }
}

/* Spawn a process non-blocking with pipes for stdout/stderr capture.
 * Returns array<string> with [pid, stdout_fd, stderr_fd].
 * Both fds are set non-blocking so fd_read_available() never blocks the caller.
 * Caller must close fds with fd_close() when done (typically on process exit).
 * Returns ["-1", "-1", "-1"] on error.
 */
static bool prepare_process_pipe(int descriptors[2]) {
    for (int i = 0; i < 2; ++i) {
        if (descriptors[i] <= STDERR_FILENO) {
            int moved = fcntl(descriptors[i], F_DUPFD, STDERR_FILENO + 1);
            if (moved < 0) return false;
            close(descriptors[i]);
            descriptors[i] = moved;
        }
        if (fcntl(descriptors[i], F_SETFD, FD_CLOEXEC) < 0) return false;
    }
    return true;
}

DynArray* nl_os_process_spawn_with_pipes(const char* command) {
    DynArray* result = dyn_array_new_with_capacity(ELEM_STRING, 3);
    if (!result) return NULL;
    /* I finish every fallible result allocation before acquiring descriptors
     * or spawning a child. Error triples reuse these same owned copies. */
    for (int i = 0; i < 3; ++i) {
        char *field = malloc(32);
        if (!field) {
            for (int64_t j = 0; j < result->length; ++j)
                free((void *)dyn_array_get_string(result, j));
            gc_release(result);
            return NULL;
        }
        strcpy(field, "-1");
        dyn_array_push_string(result, field);
    }
    if (!command) return result;

    int out_pipe[2] = {-1, -1}, err_pipe[2] = {-1, -1};
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0 ||
        !prepare_process_pipe(out_pipe) || !prepare_process_pipe(err_pipe) ||
        fcntl(out_pipe[0], F_SETFL, O_NONBLOCK) < 0 ||
        fcntl(err_pipe[0], F_SETFL, O_NONBLOCK) < 0) {
        for (int i = 0; i < 2; ++i) {
            if (out_pipe[i] >= 0) close(out_pipe[i]);
            if (err_pipe[i] >= 0) close(err_pipe[i]);
        }
        return result;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        return result;
    }

    if (pid == 0) {
        /* Child: wire stdout/stderr into the pipes and exec */
        setpgid(0, 0);
        close(out_pipe[0]);
        close(err_pipe[0]);
        if (dup2(out_pipe[1], STDOUT_FILENO) < 0 ||
            dup2(err_pipe[1], STDERR_FILENO) < 0) _exit(127);
        close(out_pipe[1]);
        close(err_pipe[1]);
        execl("/bin/sh", "sh", "-c", command, (char*)NULL);
        _exit(127);
    }

    setpgid(pid, pid);
    /* Parent: keep the already-configured read ends and close write ends. */
    close(out_pipe[1]);
    close(err_pipe[1]);
    snprintf((char *)dyn_array_get_string(result, 0), 32, "%d", (int)pid);
    snprintf((char *)dyn_array_get_string(result, 1), 32, "%d", out_pipe[0]);
    snprintf((char *)dyn_array_get_string(result, 2), 32, "%d", err_pipe[0]);
    return result;
}

/* Non-blocking read from a file descriptor.
 * Returns whatever bytes are currently available, or "" if none.
 * The fd must have been opened non-blocking (as returned by spawn_with_pipes).
 */
const char* nl_os_fd_read_available(int64_t fd_val) {
    /* Static buffer is safe for the single-threaded SDL launcher;
     * NanoLang copies string values into struct fields before the next call. */
    static char buf[65536];
    if (fd_val < 0) return "";
    ssize_t n = read((int)fd_val, buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = '\0';
    return buf;
}

/* Close a file descriptor (pipe read-end).
 * Returns 0 on success, -1 on error.
 */
int64_t nl_os_fd_close(int64_t fd_val) {
    if (fd_val < 0) return -1;
    return (int64_t)close((int)fd_val);
}
