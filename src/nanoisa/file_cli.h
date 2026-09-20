#ifndef NANOISA_FILE_CLI_H
#define NANOISA_FILE_CLI_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
/* CLI input/output I/O is not a bytecode service grant. I publish reader outputs
 * only after a complete bounded read/close; I replace -o only after staged close. */
bool nvm_file_cli_read(const char *, uint8_t **, size_t *, char *, size_t);
bool nvm_file_cli_write(const char *, const char *, char *, size_t);
#endif
