/**
 * interpreter_ffi.h - Foreign Function Interface for Interpreter
 * 
 * Enables the interpreter to call extern functions from compiled modules
 * via dynamic loading (.so/.dylib files).
 */

#ifndef NANOLANG_INTERPRETER_FFI_H
#define NANOLANG_INTERPRETER_FFI_H

#include "nanolang.h"

/* Initialize FFI system - must be called before using FFI functions */
bool ffi_init(bool verbose);

/* Cleanup FFI system - unloads all modules */
void ffi_cleanup(void);

/* Load a module's shared library for FFI access */
bool ffi_load_module(const char *module_name, const char *module_path, Environment *env, bool verbose);

/* Call an extern function via FFI 
 * Returns: Value from the function, or VAL_VOID on error */
Value ffi_call_extern(const char *function_name, Value *args, int arg_count, 
                      Function *func_info, Environment *env);

/* Check if FFI is available and initialized */
bool ffi_is_available(void);

#endif /* NANOLANG_INTERPRETER_FFI_H */

