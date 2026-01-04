#ifndef NANOLANG_STD_PROCESS_H
#define NANOLANG_STD_PROCESS_H

#include <stdint.h>
#include "../../src/runtime/dyn_array.h"

/* Run a command and capture stdout/stderr
 * Returns array<string> with [exit_code, stdout, stderr]
 */
DynArray* process_run(const char* command);

/* Spawn a process non-blocking
 * Returns process ID (pid) or -1 on error
 */
int64_t nl_os_process_spawn(const char* command);

/* Check if a process is still running
 * Returns 1 if running, 0 if exited, -1 on error
 */
int64_t nl_os_process_is_running(int64_t pid);

/* Wait for a process to complete
 * Returns exit code of the process, or -1 on error
 */
int64_t nl_os_process_wait(int64_t pid);

#endif /* NANOLANG_STD_PROCESS_H */

