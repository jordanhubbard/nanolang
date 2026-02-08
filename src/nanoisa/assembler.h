/*
 * NanoISA Text Assembler
 *
 * Converts text assembly (.nasm) to NVM binary modules.
 *
 * Assembly format:
 *   .string "hello"          ; add to string pool
 *   .function main 0 2 0     ; name arity locals upvalues
 *   label:
 *     PUSH_I64 42
 *     PUSH_I64 10
 *     ADD
 *     PRINT
 *     HALT
 *   .end
 */

#ifndef NANOISA_ASSEMBLER_H
#define NANOISA_ASSEMBLER_H

#include "nvm_format.h"

/* Assembler error codes */
typedef enum {
    ASM_OK = 0,
    ASM_ERR_SYNTAX,         /* Syntax error in assembly text */
    ASM_ERR_UNKNOWN_OPCODE, /* Unknown instruction mnemonic */
    ASM_ERR_BAD_OPERAND,    /* Invalid operand value or type */
    ASM_ERR_UNDEFINED_LABEL,/* Jump to undefined label */
    ASM_ERR_DUPLICATE_LABEL,/* Label defined more than once */
    ASM_ERR_NO_FUNCTION,    /* Instruction outside .function/.end */
    ASM_ERR_MEMORY,         /* Allocation failure */
    ASM_ERR_IO              /* File I/O error */
} AsmError;

/* Assembler result */
typedef struct {
    AsmError error;
    uint32_t line;       /* Line number where error occurred (1-based) */
    char message[256];   /* Human-readable error message */
} AsmResult;

/* Assemble text source into an NVM module.
 * Returns a new NvmModule on success (caller frees with nvm_module_free).
 * On error, returns NULL and fills result with error info. */
NvmModule *asm_assemble(const char *source, AsmResult *result);

/* Assemble from a file path.
 * Returns a new NvmModule on success, NULL on error. */
NvmModule *asm_assemble_file(const char *path, AsmResult *result);

#endif /* NANOISA_ASSEMBLER_H */
