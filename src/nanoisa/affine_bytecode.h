#ifndef NANOISA_AFFINE_BYTECODE_H
#define NANOISA_AFFINE_BYTECODE_H
#include "nvm_format.h"

#define NVM_AFFINE_MAX_INSTRUCTIONS 4096u
#define NVM_AFFINE_MAX_LOCALS 256u
#define NVM_AFFINE_MAX_STACK 256u

typedef struct {
    bool ok;
    uint32_t byte_offset; /* Function-relative refusal position. */
    uint32_t reachable;
    char message[192];
} NvmAffineAnalysis;

/* I analyze the documented scalar/record-observation/owned-transfer subset without changing
 * the module. Success is NOT executable verification. Caller alias binding,
 * reference opcodes and standalone runtime eligibility remain separate. */
NvmAffineAnalysis nvm_affine_analyze_function(const NvmModule *module,
                                              uint32_t function);
#endif
