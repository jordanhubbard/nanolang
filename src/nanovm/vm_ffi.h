/*
 * NanoVM FFI Bridge - Call native C functions from the VM
 *
 * Uses dlopen/dlsym to load shared libraries and call functions.
 * Marshals between NanoValue (VM type) and C types.
 */

#ifndef NANOVM_FFI_H
#define NANOVM_FFI_H

#include "value.h"
#include "heap.h"
#include "../nanoisa/nvm_format.h"
#include "../nanolang.h"
#include <stdbool.h>

/* Initialize the VM FFI subsystem */
void vm_ffi_init(void);

/* Set the environment for module introspection (___module_* functions) */
void vm_ffi_set_env(Environment *env);

/* Shutdown and unload all modules */
void vm_ffi_shutdown(void);

/* Load a module's shared library by name.
 * Searches standard module paths. Returns true on success. */
bool vm_ffi_load_module(const char *module_name);

/* Call an extern function.
 * import_idx: index into the NVM module's import table
 * args: array of NanoValue arguments (already popped from stack)
 * arg_count: number of arguments
 * result: output NanoValue (caller pushes onto stack)
 * Returns true on success, false on error. */
bool vm_ffi_call(const NvmModule *module, uint32_t import_idx,
                 NanoValue *args, int arg_count,
                 NanoValue *result, VmHeap *heap,
                 char *error_msg, size_t error_msg_size);

/* ========================================================================
 * Co-Process FFI Isolation
 *
 * When enabled, FFI calls are dispatched to a separate process (nano_cop)
 * via pipes, providing complete address-space isolation.
 * ======================================================================== */

/* VmState is needed for per-VM co-process state */
#include "vm.h"

/* Start the co-process for FFI isolation.
 * Forks, pipes, and execs nano_cop, then sends COP_MSG_INIT.
 * Stores cop_pid/cop_in_fd/cop_out_fd in vm.
 * Returns true on success. */
bool vm_ffi_cop_start(VmState *vm, const NvmModule *module);

/* Stop the co-process (send shutdown, close pipes, waitpid). */
void vm_ffi_cop_stop(VmState *vm);

/* Call an extern function via the co-process.
 * Same signature as vm_ffi_call() but dispatches over pipes.
 * Falls back to in-process vm_ffi_call() if cop is not active. */
bool vm_ffi_call_cop(VmState *vm, const NvmModule *module, uint32_t import_idx,
                     NanoValue *args, int arg_count,
                     NanoValue *result, VmHeap *heap,
                     char *error_msg, size_t error_msg_size);

#endif /* NANOVM_FFI_H */
