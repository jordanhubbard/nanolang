#ifndef NANOISA_SERVICES_CODE_H
#define NANOISA_SERVICES_CODE_H
#include "services_flow.h"
#include "isa.h"

/* Private declaration/decode/DAG preparation only. Success is NOT a checked
 * stack/owner/body/exit certificate and grants no execution or host authority.
 * Input storage is valid and immutable during preparation; output storage is
 * disjoint from input and plan storage. Calls require external serialization,
 * as do the underlying ISA metadata and Services declaration APIs. */
#define NVM_SERVICES_CODE_BYTES 65536u
#define NVM_SERVICES_CODE_FUNCTION_INSTRUCTIONS 256u
#define NVM_SERVICES_CODE_INSTRUCTIONS 4096u
#define NVM_SERVICES_CODE_NO_SUCCESSOR UINT16_MAX
typedef struct NvmServicesCodePlan NvmServicesCodePlan;
typedef struct {
    uint32_t byte_offset; /* Checked function-relative logical site. */
    DecodedInstruction decoded;
    uint16_t successors[2]; /* Function-local instruction indices. */
    uint8_t successor_count;
    uint16_t call_references[NVM_SERVICES_FLOW_LOCALS]; /* One per declared parameter; NO_REFERENCE for values. */
    uint32_t catalog_ordinal; /* instance*9+type for constructors, instance*5+method for services; otherwise NO_INDEX. */
} NvmServicesCodeInstruction;
typedef struct {
    uint32_t code_offset, code_length;
    uint16_t instruction_count;
    NvmServicesFlowFunction declaration;
} NvmServicesCodeFunction;
NvmServicesFlowStatus nvm_services_code_prepare(const NvmModule *, NvmServicesCodePlan **out);
void nvm_services_code_free(NvmServicesCodePlan *);
uint32_t nvm_services_code_function_count(const NvmServicesCodePlan *);
/* Getters copy facts; failure preserves output. No input buffers are borrowed. */
bool nvm_services_code_function(const NvmServicesCodePlan *, uint32_t, NvmServicesCodeFunction *);
bool nvm_services_code_local(const NvmServicesCodePlan *, uint32_t, uint16_t,
                         NvmServicesFlowDeclaration *);
bool nvm_services_code_instruction(const NvmServicesCodePlan *, uint32_t, uint16_t,
                               NvmServicesCodeInstruction *);
bool nvm_services_code_instruction_order(const NvmServicesCodePlan *, uint32_t, uint16_t,
                                     uint16_t *);
/* Complete callee-before-caller order, including uncalled functions. */
bool nvm_services_code_function_order(const NvmServicesCodePlan *, uint32_t, uint32_t *);
#endif
