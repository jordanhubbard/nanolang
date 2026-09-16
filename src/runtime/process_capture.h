#ifndef NANO_PROCESS_CAPTURE_H
#define NANO_PROCESS_CAPTURE_H
/* I share shell invocation and file-backed output capture across backends. */
#include "dyn_array.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <errno.h>

/* Run a command and capture stdout/stderr
 * Returns array<string> with [exit_code, stdout, stderr]
 */
static DynArray *nl_process_capture_result(DynArray *result, int code, char *out, char *err) {
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

static DynArray *nl_process_capture_error(DynArray *result, const char *message) {
    return nl_process_capture_result(result, -1, strdup(""), strdup(message));
}

/* I retain anonymous temporary-file descriptors, never reopen capture paths.
 * Moving them above stderr also handles callers with closed standard fds. */
static int nl_process_capture_file(void) {
    FILE *file = tmpfile();
    if (!file) return -1;
    int fd = fcntl(fileno(file), F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    fclose(file);
    return fd;
}

static char *nl_process_read_capture(int fd) {
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

static inline DynArray* nl_process_run_capture(const char* command) {
    DynArray *result = dyn_array_new_with_capacity(ELEM_STRING, 3);
    if (!result) return NULL;
    if (!command) return nl_process_capture_error(result, "I require a command");
    int out = nl_process_capture_file(), err = nl_process_capture_file();
    if (out < 0 || err < 0) {
        if (out >= 0) close(out);
        if (err >= 0) close(err);
        return nl_process_capture_error(result, "I could not create capture files");
    }
    pid_t child = fork();
    if (child == 0) {
        if (dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0) _exit(127);
        close(out); close(err);
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
        const char message[] = "I could not execute the shell command.\n";
        ssize_t diagnostic_written = write(STDERR_FILENO, message, sizeof(message) - 1);
        (void)diagnostic_written;
        _exit(127);
    }
    if (child < 0) {
        close(out); close(err);
        return nl_process_capture_error(result, "I could not start the command");
    }
    int status;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    if (waited != child) {
        close(out); close(err);
        return nl_process_capture_error(result, "I could not wait for the command");
    }
    char *stdout_text = nl_process_read_capture(out), *stderr_text = nl_process_read_capture(err);
    close(out); close(err);
    if (!stdout_text || !stderr_text) {
        free(stdout_text); free(stderr_text);
        return nl_process_capture_error(result, "I could not read captured text (I reject embedded NUL bytes)");
    }
    return nl_process_capture_result(result, WIFEXITED(status) ? WEXITSTATUS(status) : -1,
                      stdout_text, stderr_text);
}
#endif
