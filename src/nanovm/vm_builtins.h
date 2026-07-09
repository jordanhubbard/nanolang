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
char *vm_string_from_char(int64_t code);

/* Binary string */
DynArray *vm_bytes_from_string(const char *str);
char *vm_string_from_bytes(DynArray *arr);

/* Character classification */
int64_t vm_is_digit(int64_t c);
int64_t vm_is_alpha(int64_t c);
int64_t vm_is_alnum(int64_t c);
int64_t vm_is_space(int64_t c);
int64_t vm_is_upper(int64_t c);
int64_t vm_is_lower(int64_t c);
int64_t vm_is_whitespace(int64_t c);
int64_t vm_digit_value(int64_t c);
int64_t vm_char_to_lower(int64_t c);
int64_t vm_char_to_upper(int64_t c);
int64_t vm_bstr_utf8_length(const char *str);
int64_t vm_bstr_utf8_char_at(const char *str, int64_t char_index);
int64_t vm_bstr_validate_utf8(const char *str);

/* Process */
DynArray *vm_process_run(const char *cmd);

#endif /* NANOVM_BUILTINS_H */
