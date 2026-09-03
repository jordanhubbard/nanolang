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
 *   - Calls and aggregate constructors have enough stack operands
 *   - Local, global, and flattened-upvalue operands are in bounds
 *   - Encoded value tags and import signatures are valid
 *   - All OP_CLOSURE_NEW function indices < function_count
 *   - All struct/enum/union definition indices are valid
 *   - All type-tag operands (ARR_NEW/ARR_LITERAL/HM_NEW/TYPE_CHECK) < TAG_COUNT
 *   - Every opcode family is covered: an opcode with an unhandled table operand
 *     is a verifier bug, not silently accepted
 *   - All opcodes are recognized
 *   - Entry point is a valid function index
 *
 * OP_CALL_MODULE operands are checked structurally here and fully resolved by
 * nvm_verify_linked() once the linked-module table is known.
 */
NvmVerifyResult nvm_verify(const NvmModule *mod);

/* Validate one function after an incremental compiler appends it. */
NvmVerifyResult nvm_verify_function(const NvmModule *mod, uint32_t fn_idx);

/* Validate a module together with the table of modules it is linked against.
 *
 * nvm_verify() checks each module in isolation: an OP_CALL_MODULE operand pair
 * (module index, function index) cannot be fully resolved because the target
 * module table is not known until link time. This entry point closes that gap.
 * It first runs the full single-module nvm_verify(), then, for every
 * OP_CALL_MODULE in every function, confirms that:
 *   - the module index is < linked_count,
 *   - the referenced linked module is non-NULL,
 *   - the callee function index is < that module's function_count.
 * linked_modules[i] may be mod itself (self reference) or any peer; a NULL
 * slot is treated as an unresolved link and rejected. Passing linked_count == 0
 * with a module that contains no OP_CALL_MODULE is valid and reduces to
 * nvm_verify().
 */
NvmVerifyResult nvm_verify_linked(const NvmModule *mod,
                                  const NvmModule *const *linked_modules,
                                  uint32_t linked_count);

#endif /* NANOISA_VERIFIER_H */
