#ifndef NANOISA_WEBSOCKET_CODE_H
#define NANOISA_WEBSOCKET_CODE_H
#include "websocket_flow.h"
#include "isa.h"

/* Private declaration/decode/DAG preparation only. Success is NOT a checked
 * stack/owner/body/exit certificate and grants no execution or host authority.
 * Input storage is valid and immutable during preparation; output storage is
 * disjoint from input and plan storage. Calls require external serialization,
 * as do the underlying ISA metadata and WebSocket declaration APIs. */
#define NVM_WEBSOCKET_CODE_BYTES 65536u
#define NVM_WEBSOCKET_CODE_FUNCTION_INSTRUCTIONS 256u
#define NVM_WEBSOCKET_CODE_INSTRUCTIONS 4096u
#define NVM_WEBSOCKET_CODE_NO_SUCCESSOR UINT16_MAX
typedef struct NvmWebSocketCodePlan NvmWebSocketCodePlan;
typedef struct {
    uint32_t byte_offset; /* Checked function-relative logical site. */
    DecodedInstruction decoded;
    uint16_t successors[2]; /* Function-local instruction indices. */
    uint8_t successor_count;
    uint16_t call_references[NVM_WEBSOCKET_FLOW_LOCALS]; /* One per declared parameter; NO_REFERENCE for values. */
    uint32_t catalog_ordinal; /* Constructor/service identity, else NO_INDEX. */
} NvmWebSocketCodeInstruction;
typedef struct {
    uint32_t code_offset, code_length;
    uint16_t instruction_count;
    NvmWebSocketFlowFunction declaration;
} NvmWebSocketCodeFunction;
NvmWebSocketFlowStatus nvm_websocket_code_prepare(const NvmModule *, NvmWebSocketCodePlan **out);
void nvm_websocket_code_free(NvmWebSocketCodePlan *);
uint32_t nvm_websocket_code_function_count(const NvmWebSocketCodePlan *);
/* Getters copy facts; failure preserves output. No input buffers are borrowed. */
bool nvm_websocket_code_function(const NvmWebSocketCodePlan *, uint32_t, NvmWebSocketCodeFunction *);
bool nvm_websocket_code_local(const NvmWebSocketCodePlan *, uint32_t, uint16_t,
                         NvmWebSocketFlowDeclaration *);
bool nvm_websocket_code_instruction(const NvmWebSocketCodePlan *, uint32_t, uint16_t,
                               NvmWebSocketCodeInstruction *);
bool nvm_websocket_code_instruction_order(const NvmWebSocketCodePlan *, uint32_t, uint16_t,
                                     uint16_t *);
/* Complete callee-before-caller order, including uncalled functions. */
bool nvm_websocket_code_function_order(const NvmWebSocketCodePlan *, uint32_t, uint32_t *);
#endif
