/*
 * NanoVM - Virtual Machine Execution Engine
 *
 * Loads an NvmModule and executes its bytecode via switch dispatch.
 */

#ifndef NANOVM_VM_H
#define NANOVM_VM_H

#include "value.h"
#include "heap.h"
#include "../nanoisa/isa.h"
#include "../nanoisa/nvm_format.h"

/* ========================================================================
 * VM Configuration
 * ======================================================================== */

#define VM_STACK_INITIAL    4096
#define VM_MAX_FRAMES       1024
#define VM_MAX_GLOBALS      4096

/* ========================================================================
 * Call Frame
 * ======================================================================== */

typedef struct {
    uint32_t fn_idx;          /* Function table index */
    uint32_t return_ip;       /* Instruction pointer to return to */
    uint32_t stack_base;      /* Stack index where this frame's locals begin */
    uint16_t local_count;     /* Number of locals (including params) */
    VmClosure *closure;       /* Non-NULL if this is a closure call */
} VmCallFrame;

/* ========================================================================
 * VM Execution Result
 * ======================================================================== */

typedef enum {
    VM_OK = 0,
    VM_ERR_STACK_OVERFLOW,
    VM_ERR_STACK_UNDERFLOW,
    VM_ERR_CALL_DEPTH,
    VM_ERR_INVALID_OPCODE,
    VM_ERR_TYPE_ERROR,
    VM_ERR_OUT_OF_BOUNDS,
    VM_ERR_DIV_ZERO,         /* Not used (div by zero = 0) but reserved */
    VM_ERR_ASSERT_FAILED,
    VM_ERR_UNDEFINED_GLOBAL,
    VM_ERR_UNDEFINED_FUNCTION,
    VM_ERR_NOT_IMPLEMENTED,
    VM_ERR_MEMORY,
    VM_ERR_DECODE
} VmResult;

/* ========================================================================
 * VM State
 * ======================================================================== */

typedef struct {
    /* Module being executed */
    const NvmModule *module;

    /* Operand stack */
    NanoValue *stack;
    uint32_t stack_size;
    uint32_t stack_capacity;

    /* Call stack */
    VmCallFrame frames[VM_MAX_FRAMES];
    uint32_t frame_count;

    /* Current execution state */
    uint32_t ip;              /* Instruction pointer (byte offset in code) */
    uint32_t current_fn;      /* Current function index */

    /* Global variables */
    NanoValue globals[VM_MAX_GLOBALS];
    uint32_t global_count;

    /* GC Heap */
    VmHeap heap;

    /* Output capture (NULL = stdout) */
    FILE *output;

    /* Error info */
    VmResult last_error;
    char error_msg[256];
} VmState;

/* ========================================================================
 * VM API
 * ======================================================================== */

/* Initialize VM state for a module */
void vm_init(VmState *vm, const NvmModule *module);

/* Destroy VM state (free stack, heap, etc.) */
void vm_destroy(VmState *vm);

/* Execute from the module's entry point. Returns VM_OK on success. */
VmResult vm_execute(VmState *vm);

/* Execute a specific function by index. Returns VM_OK on success. */
VmResult vm_call_function(VmState *vm, uint32_t fn_idx, NanoValue *args, uint16_t arg_count);

/* Get the return value (top of stack after execution) */
NanoValue vm_get_result(VmState *vm);

/* Get error message string */
const char *vm_error_string(VmResult result);

#endif /* NANOVM_VM_H */
