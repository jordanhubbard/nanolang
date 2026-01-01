/**
 * @file interpreter_ffi.h
 * @brief Foreign Function Interface for nanolang interpreter
 *
 * Enables the interpreter to dynamically load and call C functions from compiled
 * modules at runtime, providing true compiler/interpreter parity for extern functions.
 * Uses platform-specific dynamic loading (dlopen on Unix, LoadLibrary on Windows).
 */

#ifndef NANOLANG_INTERPRETER_FFI_H
#define NANOLANG_INTERPRETER_FFI_H

#include "nanolang.h"

/**
 * @brief Initialize the FFI system
 * @param verbose Enable verbose logging of FFI operations
 * @return true on successful initialization, false on error
 *
 * Must be called before any other FFI functions. Sets up internal tracking
 * for loaded modules and prepares the FFI subsystem. Safe to call multiple times.
 */
bool ffi_init(bool verbose);

/**
 * @brief Cleanup and shutdown FFI system
 *
 * Unloads all dynamically loaded modules and frees FFI data structures.
 * Should be called during interpreter shutdown. Safe to call even if
 * ffi_init() was never called.
 */
void ffi_cleanup(void);

/**
 * @brief Load a module's shared library for FFI calls
 * @param module_name Module name (e.g., "sqlite", "filesystem")
 * @param module_path Module directory path (reserved for future use)
 * @param env Environment context (reserved for future use)
 * @param verbose Enable verbose logging for this operation
 * @return true if module loaded successfully, false on error
 *
 * Dynamically loads the module's shared library (.so on Linux, .dylib on macOS)
 * and resolves extern function symbols. Idempotent - loading the same module
 * multiple times is safe and returns success.
 */
bool ffi_load_module(const char *module_name, const char *module_path, Environment *env, bool verbose);

/**
 * @brief Call an extern function via FFI
 * @param function_name Extern function name to call
 * @param args Array of nanolang Value arguments
 * @param arg_count Number of arguments in args array
 * @param func_info Function metadata (types, signature)
 * @param env Environment context (reserved for future use)
 * @return Function return value as nanolang Value, or VAL_VOID on error
 *
 * Marshals nanolang Values to C types (int64_t, double, bool, char*, void*),
 * invokes the native function via function pointer from dlsym(), then marshals
 * the C return value back to a nanolang Value. Handles type conversions
 * automatically based on func_info metadata.
 *
 * Supported types: int, float, bool, string, opaque
 * Unsupported: arrays, structs, unions (future work)
 */
Value ffi_call_extern(const char *function_name, Value *args, int arg_count, 
                      Function *func_info, Environment *env);

/**
 * @brief Check if FFI is available and initialized
 * @return true if FFI system is ready for use, false otherwise
 *
 * Useful for conditional logic when FFI support may not be available
 * (e.g., on platforms without dlopen support).
 */
bool ffi_is_available(void);

#endif /* NANOLANG_INTERPRETER_FFI_H */

