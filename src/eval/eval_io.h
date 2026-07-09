#ifndef EVAL_IO_H
#define EVAL_IO_H

#include "../nanolang.h"

/* File operation built-in functions */
Value builtin_file_read(Value *args);
Value builtin_file_read_bytes(Value *args);
Value builtin_bytes_from_string(Value *args);
Value builtin_string_from_bytes(Value *args);
Value builtin_file_write(Value *args);
Value builtin_file_append(Value *args);
Value builtin_file_remove(Value *args);
Value builtin_file_rename(Value *args);
Value builtin_file_exists(Value *args);
Value builtin_file_size(Value *args);
Value builtin_tmp_dir(Value *args);
Value builtin_mktemp(Value *args);
Value builtin_mktemp_dir(Value *args);

/* Directory operation built-in functions */
Value builtin_dir_create(Value *args);
Value builtin_dir_remove(Value *args);
Value builtin_dir_list(Value *args);
Value builtin_dir_exists(Value *args);
Value builtin_getcwd(Value *args);
Value builtin_chdir(Value *args);

/* Path operation built-in functions */
Value builtin_path_isfile(Value *args);
Value builtin_path_isdir(Value *args);
Value builtin_path_join(Value *args);
Value builtin_path_basename(Value *args);
Value builtin_path_dirname(Value *args);
Value builtin_path_normalize(Value *args);

/* Process and system operation built-in functions */
Value builtin_fs_walkdir(Value *args);
Value builtin_system(Value *args);
Value builtin_exit(Value *args);
Value builtin_getenv(Value *args);
Value builtin_setenv(Value *args);
Value builtin_unsetenv(Value *args);
Value builtin_process_run(Value *args);

/* Result type built-in functions */
Value builtin_result_is_ok(Value *args);
Value builtin_result_is_err(Value *args);
Value builtin_result_unwrap(Value *args);
Value builtin_result_unwrap_err(Value *args);
Value builtin_result_unwrap_or(Value *args);
Value builtin_result_map(Value *args, Environment *env);
Value builtin_result_and_then(Value *args, Environment *env);

/* Helper function */
char* nl_path_normalize(const char* path);

#endif /* EVAL_IO_H */
