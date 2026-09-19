#ifndef NANOISA_FILE_CODE_H
#define NANOISA_FILE_CODE_H
#include "file_flow.h"
#include "isa.h"

/* Private declaration/decode/DAG preparation only. Success is NOT a checked
 * stack/owner/body/exit certificate and grants no execution or host authority.
 * Input storage is valid and immutable during preparation; output storage is
 * disjoint from input and plan storage. Calls require external serialization,
 * as do the underlying ISA metadata and File declaration APIs. */
#define NVM_FILE_CODE_BYTES 65536u
#define NVM_FILE_CODE_FUNCTION_INSTRUCTIONS 256u
#define NVM_FILE_CODE_INSTRUCTIONS 4096u
#define NVM_FILE_CODE_NO_SUCCESSOR UINT16_MAX
typedef struct NvmFileCodePlan NvmFileCodePlan;
typedef struct {
    uint32_t byte_offset; /* Checked function-relative logical site. */
    DecodedInstruction decoded;
    uint16_t successors[2]; /* Function-local instruction indices. */
    uint8_t successor_count;
    uint32_t catalog_ordinal; /* Constructor/service identity, else NO_INDEX. */
} NvmFileCodeInstruction;
typedef struct {
    uint32_t code_offset, code_length;
    uint16_t instruction_count;
    NvmFileFlowFunction declaration;
} NvmFileCodeFunction;
NvmFileFlowStatus nvm_file_code_prepare(const NvmModule *, NvmFileCodePlan **out);
void nvm_file_code_free(NvmFileCodePlan *);
uint32_t nvm_file_code_function_count(const NvmFileCodePlan *);
/* Getters copy facts; failure preserves output. No input buffers are borrowed. */
bool nvm_file_code_function(const NvmFileCodePlan *, uint32_t, NvmFileCodeFunction *);
bool nvm_file_code_local(const NvmFileCodePlan *, uint32_t, uint16_t,
                         NvmFileFlowDeclaration *);
bool nvm_file_code_instruction(const NvmFileCodePlan *, uint32_t, uint16_t,
                               NvmFileCodeInstruction *);
bool nvm_file_code_instruction_order(const NvmFileCodePlan *, uint32_t, uint16_t,
                                     uint16_t *);
/* Complete callee-before-caller order, including uncalled functions. */
bool nvm_file_code_function_order(const NvmFileCodePlan *, uint32_t, uint32_t *);
#endif
