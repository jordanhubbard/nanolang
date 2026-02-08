/*
 * VM Built-in Functions
 * C-callable implementations for the NanoVM FFI bridge.
 */
#ifndef NANOVM_BUILTINS_H
#define NANOVM_BUILTINS_H

#include <stdint.h>
#include "runtime/dyn_array.h"

/* OS / File System */
char *vm_getcwd(void);
int64_t vm_chdir(const char *path);
char *vm_file_read(const char *path);
int64_t vm_file_write(const char *path, const char *content);
int64_t vm_file_exists(const char *path);
int64_t vm_dir_exists(const char *path);
int64_t vm_dir_create(const char *path);
DynArray *vm_dir_list(const char *path);
char *vm_mktemp_dir(const char *prefix);
char *vm_getenv(const char *name);
int64_t vm_setenv(const char *name, const char *value);

/* String */
int64_t vm_str_index_of(const char *haystack, const char *needle);

/* Process */
int64_t vm_process_run(const char *cmd);

#endif /* NANOVM_BUILTINS_H */
