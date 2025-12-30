#ifndef NANOLANG_STD_PROCESS_H
#define NANOLANG_STD_PROCESS_H

#include <stdint.h>
#include "../../src/runtime/dyn_array.h"

/* Run a command and capture stdout/stderr
 * Returns array<string> with [exit_code, stdout, stderr]
 */
DynArray* process_run(const char* command);

#endif /* NANOLANG_STD_PROCESS_H */

