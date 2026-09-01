/*
 * NVM Bytecode Verifier
 *
 * Validates .nvm bytecode before execution to ensure memory safety.
 * Checks jump targets, index bounds, function ranges, and type consistency.
 *
 * Call nvm_verify() after nvm_deserialize() and before vm_execute().
 * Returns true if the module is safe to execute; on failure, sets
 * error_msg with a human-readable description of the first violation.
 */

#ifndef NANOISA_VERIFIER_H
#define NANOISA_VERIFIER_H

#include "nvm_format.h"
#include <stdbool.h>

#define NVM_VERIFY_ERROR_SIZE 256

/* Static safety bounds proven by the verifier before execution.
 *
 * The VM allocates a bounded operand stack and a fixed call-frame array
 * (see VM_STACK_INITIAL and VM_MAX_FRAMES in vm.h). The verifier proves,
 * per function, that the inferred operand-stack depth and per-frame local
 * footprint stay within these limits so the running VM can never exceed
 * them. Frame-count (recursion) depth is dynamic and enforced at call time;
 * the verifier bounds the per-frame contribution that feeds it. */
#define NVM_MAX_OPERAND_DEPTH 4096u   /* mirrors VM_STACK_INITIAL */
#define NVM_MAX_FRAME_LOCALS  4096u   /* per-frame local slots; mirrors VM_STACK_INITIAL */

typedef struct {
    bool ok;
    char error_msg[NVM_VERIFY_ERROR_SIZE];
} NvmVerifyResult;

/* Validate a deserialized NVM module for safe execution.
 * Checks:
 *   - Function code_offset/code_length within code section bounds
 *   - All jump targets land on instruction boundaries in the originating function
 *   - All OP_CALL/OP_TAIL_CALL indices and tail-call results are valid
 *   - All OP_PUSH_STR string indices < string_count
 *   - All OP_CALL_EXTERN import indices < import_count
 *   - All OP_CLOSURE_NEW function indices < function_count
 *   - All struct/enum/union definition indices are valid
 *   - All opcodes are recognized
 *   - Entry point is a valid function index
 *   - Return shape: every OP_RET leaves exactly result_count values
 *   - Maximum operand depth stays within NVM_MAX_OPERAND_DEPTH
 *   - Per-frame local footprint stays within NVM_MAX_FRAME_LOCALS
 *   - Ownership effects balance: no reachable path leaks or
 *     under-runs owned operands (return shape + stack-height merge)
 *   - Explicit termination: every reachable path ends in RET/TAIL_CALL/HALT
 */
NvmVerifyResult nvm_verify(const NvmModule *mod);

/* Validate one function after an incremental compiler appends it. */
NvmVerifyResult nvm_verify_function(const NvmModule *mod, uint32_t fn_idx);

#endif /* NANOISA_VERIFIER_H */
