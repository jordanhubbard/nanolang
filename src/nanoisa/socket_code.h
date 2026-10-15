#ifndef NANOISA_SOCKET_CODE_H
#define NANOISA_SOCKET_CODE_H
#include "socket_flow.h"
#include "isa.h"

/* Private declaration/decode/DAG preparation only. Success is NOT a checked
 * stack/owner/body/exit certificate and grants no execution or host authority.
 * Input storage is valid and immutable during preparation; output storage is
 * disjoint from input and plan storage. Calls require external serialization,
 * as do the underlying ISA metadata and Socket declaration APIs. */
#define NVM_SOCKET_CODE_BYTES 65536u
#define NVM_SOCKET_CODE_FUNCTION_INSTRUCTIONS 256u
#define NVM_SOCKET_CODE_INSTRUCTIONS 4096u
#define NVM_SOCKET_CODE_NO_SUCCESSOR UINT16_MAX
typedef struct NvmSocketCodePlan NvmSocketCodePlan;
typedef struct {
    uint32_t byte_offset; /* Checked function-relative logical site. */
    DecodedInstruction decoded;
    uint16_t successors[2]; /* Function-local instruction indices. */
    uint8_t successor_count;
    uint16_t call_references[NVM_SOCKET_FLOW_LOCALS]; /* One per declared parameter; NO_REFERENCE for values. */
    uint32_t catalog_ordinal; /* Constructor/service identity, else NO_INDEX. */
} NvmSocketCodeInstruction;
typedef struct {
    uint32_t code_offset, code_length;
    uint16_t instruction_count;
    NvmSocketFlowFunction declaration;
} NvmSocketCodeFunction;
NvmSocketFlowStatus nvm_socket_code_prepare(const NvmModule *, NvmSocketCodePlan **out);
void nvm_socket_code_free(NvmSocketCodePlan *);
uint32_t nvm_socket_code_function_count(const NvmSocketCodePlan *);
/* Getters copy facts; failure preserves output. No input buffers are borrowed. */
bool nvm_socket_code_function(const NvmSocketCodePlan *, uint32_t, NvmSocketCodeFunction *);
bool nvm_socket_code_local(const NvmSocketCodePlan *, uint32_t, uint16_t,
                         NvmSocketFlowDeclaration *);
bool nvm_socket_code_instruction(const NvmSocketCodePlan *, uint32_t, uint16_t,
                               NvmSocketCodeInstruction *);
bool nvm_socket_code_instruction_order(const NvmSocketCodePlan *, uint32_t, uint16_t,
                                     uint16_t *);
/* Complete callee-before-caller order, including uncalled functions. */
bool nvm_socket_code_function_order(const NvmSocketCodePlan *, uint32_t, uint32_t *);
#endif
