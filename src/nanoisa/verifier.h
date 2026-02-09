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
 *   - All jump targets land within the originating function's code range
 *   - All OP_CALL function indices < function_count
 *   - All OP_PUSH_STR string indices < string_count
 *   - All OP_CALL_EXTERN import indices < import_count
 *   - All OP_CLOSURE_NEW function indices < function_count
 *   - All struct/enum/union definition indices are valid
 *   - All opcodes are recognized
 *   - Entry point is a valid function index
 */
NvmVerifyResult nvm_verify(const NvmModule *mod);

#endif /* NANOISA_VERIFIER_H */
