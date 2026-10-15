#ifndef NANOISA_WEBSOCKET_HOSTED_H
#define NANOISA_WEBSOCKET_HOSTED_H
#include "websocket_body.h"
#define NVM_WEBSOCKET_HOSTED_INPUT_BYTES (16u*1024u*1024u)
#define NVM_WEBSOCKET_HOSTED_BYTES (64u*1024u*1024u)
typedef struct NvmWebSocketHostedPlan NvmWebSocketHostedPlan;
typedef struct {
    uint32_t entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t allocation_bound;
} NvmWebSocketHostedStartup;
typedef struct {
    NvmWebSocketCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
} NvmWebSocketHostedFunction;
/* Private serialized-v2 startup/storage facts only. No host/service operation,
 * public admission or concrete carrier ABI certification. Input stays immutable
 * during the call; output is disjoint. Reports own all facts; failure preserves
 * output. Calls require the underlying private query's external serialization.
 * Value-slot upper bounds exclude bytes inside a runtime carrier/host resource;
 * a later matched runtime must separately check those extents and cleanup. */
NvmWebSocketFlowStatus nvm_websocket_hosted_prepare(const uint8_t *,size_t,NvmWebSocketHostedPlan **);
void nvm_websocket_hosted_free(NvmWebSocketHostedPlan *);
bool nvm_websocket_hosted_startup(const NvmWebSocketHostedPlan *,NvmWebSocketHostedStartup *);
bool nvm_websocket_hosted_function(const NvmWebSocketHostedPlan *,uint32_t,NvmWebSocketHostedFunction *);
bool nvm_websocket_hosted_local(const NvmWebSocketHostedPlan *,uint32_t,uint16_t,NvmWebSocketFlowDeclaration *);
bool nvm_websocket_hosted_instruction(const NvmWebSocketHostedPlan *,uint32_t,uint16_t,
                                NvmWebSocketCodeInstruction *,NvmWebSocketBodyInstruction *);
/* Read-only exact nominal/catalog maps owned by this same hosted plan. */
bool nvm_websocket_hosted_type(const NvmWebSocketHostedPlan *,uint32_t,NvmWebSocketNominalLayout *);
bool nvm_websocket_hosted_import(const NvmWebSocketHostedPlan *,uint32_t,uint32_t *);
#endif
