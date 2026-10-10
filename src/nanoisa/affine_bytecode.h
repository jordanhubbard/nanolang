#ifndef NANOISA_AFFINE_BYTECODE_H
#define NANOISA_AFFINE_BYTECODE_H
#include "nvm_format.h"

#define NVM_AFFINE_MAX_INSTRUCTIONS 4096u
#define NVM_AFFINE_MAX_LOCALS 256u
#define NVM_AFFINE_MAX_STACK 256u

typedef struct {
    bool ok;
    uint32_t byte_offset; /* Function-relative refusal position. */
    uint32_t reachable; /* Distinct processed instructions. */
    uint32_t visits; /* Includes rechecks after initialization or selected-arm facts weaken. */
    char message[192];
} NvmAffineAnalysis;

typedef struct {
    bool reachable;
    uint8_t top_tag; /* TAG_VOID when the entry stack is empty. */
    uint16_t stack_depth;
    uint16_t unpack_count; /* Zero except at destructive local unpack. */
    uint32_t byte_offset;
} NvmAffineInstructionFact;

/* I publish one fact per decoded instruction only after convergence succeeds.
 * Capacity must equal the decoded instruction count. Failure leaves the entire
 * caller buffer unchanged. These facts do not grant executable admission. */
NvmAffineAnalysis nvm_affine_analyze_instructions(const NvmModule *module,
    uint32_t function, NvmAffineInstructionFact *facts, uint32_t capacity);

/* I check every function's bounded value signature and all direct-call edges,
 * including unreachable code. This graph check alone grants no execution. */
bool nvm_affine_value_call_graph(const NvmModule *module);

/* I analyze the documented scalar/record-observation/selected-union/owned-transfer subset without changing
 * the module. Success is NOT executable verification. Caller alias binding,
 * reference opcodes and standalone runtime eligibility remain separate.
 * With globals, helper success is conditional on its inferred initialization
 * preconditions. Entry-zero analysis composes all called helper requirements
 * and checks them against initially uninitialized slots. Only that entry proof
 * establishes global read/write ordering for the reachable call graph. */
NvmAffineAnalysis nvm_affine_analyze_function(const NvmModule *module,
                                              uint32_t function);
#endif
