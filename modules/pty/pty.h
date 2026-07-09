#ifndef NL_PTY_H
#define NL_PTY_H

#include <stdint.h>

/* Open a new PTY master and set the initial terminal size.
 * Returns the master file descriptor, or -1 on failure. */
int64_t nl_pty_open(int64_t rows, int64_t cols);

/* Fork a child process and connect it to the PTY slave.
 * prog     - path to executable (e.g. "bin/nano_vm")
 * args_str - space-separated arguments (e.g. "bin/nl_forth_interpreter_vm")
 * envkey/envval - one optional environment variable to set in child ("" to skip)
 * Returns the child PID, or -1 on failure. */
int64_t nl_pty_fork_exec(int64_t master_fd, const char *prog,
                          const char *args_str,
                          const char *envkey, const char *envval);

/* Non-blocking read from the PTY master.
 * Returns a heap-allocated string (caller must NOT free — managed by runtime).
 * Returns an empty string if no data is available (EAGAIN). */
const char *nl_pty_read(int64_t master_fd);

/* Write a string to the PTY master (goes to the child's stdin).
 * Returns the number of bytes written, or -1 on error. */
int64_t nl_pty_write(int64_t master_fd, const char *data);

/* Send SIGWINCH to notify the child of a terminal resize. */
void nl_pty_resize(int64_t master_fd, int64_t pid, int64_t rows, int64_t cols);

/* Close the PTY master file descriptor. */
void nl_pty_close(int64_t master_fd);

/* Returns 1 if the child process is still alive, 0 if it has exited. */
int64_t nl_pty_is_alive(int64_t pid);

#endif /* NL_PTY_H */
