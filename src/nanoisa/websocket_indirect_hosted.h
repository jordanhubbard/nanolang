#ifndef NANOISA_WEBSOCKET_INDIRECT_HOSTED_H
#define NANOISA_WEBSOCKET_INDIRECT_HOSTED_H
#include "websocket_indirect_flow.h"
#include "websocket_hosted.h"

#define NVM_WEBSOCKET_INDIRECT_HOSTED_REVISION 1u
/* Private serialized preparation only. No dispatcher coverage, runtime admission,
 * grant, fuel, service execution or old single-state plan conversion. Inputs
 * stay immutable during preparation; output storage is disjoint from all input
 * and plan storage. External serialization matches the underlying queries.
 * Success owns every byte/fact. Failure and invalid getters preserve outputs. */
typedef struct NvmWebSocketIndirectHostedPlan NvmWebSocketIndirectHostedPlan;
typedef struct {
    uint32_t revision, entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t input_bytes, allocation_bound, query_storage_peak, retained_bound;
    bool runtime_admitted; /* Always false in this nonexecuting checkpoint. */
} NvmWebSocketIndirectHostedStartup;
typedef struct {
    NvmWebSocketCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint16_t frame_owner_peak, frame_reference_peak, frame_region_peak;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
    uint8_t entry_variant;
} NvmWebSocketIndirectHostedFunction;
/* allocation_bound conservatively includes simultaneous copied wire/bridge,
 * full query budget, retained plan and transient preparation allocations under
 * NVM_WEBSOCKET_HOSTED_BYTES. retained_bound may overestimate retained query memory
 * by its preparation peak. Runtime/host resource bytes are not certified. */
NvmWebSocketFlowStatus nvm_websocket_indirect_hosted_prepare(const uint8_t *,size_t,NvmWebSocketIndirectHostedPlan **);
void nvm_websocket_indirect_hosted_free(NvmWebSocketIndirectHostedPlan *);
bool nvm_websocket_indirect_hosted_query_summary(const NvmWebSocketIndirectHostedPlan *,NvmWebSocketIndirectFlowSummary *);
bool nvm_websocket_indirect_hosted_startup(const NvmWebSocketIndirectHostedPlan *,NvmWebSocketIndirectHostedStartup *);
bool nvm_websocket_indirect_hosted_function(const NvmWebSocketIndirectHostedPlan *,uint32_t,NvmWebSocketIndirectHostedFunction *);
bool nvm_websocket_indirect_hosted_function_order(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint32_t *);
bool nvm_websocket_indirect_hosted_local(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,NvmWebSocketFlowDeclaration *);
bool nvm_websocket_indirect_hosted_instruction(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,NvmWebSocketCodeInstruction *);
bool nvm_websocket_indirect_hosted_component(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint16_t *);
bool nvm_websocket_indirect_hosted_variant_count(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_websocket_indirect_hosted_variant(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,NvmWebSocketCyclicVariant *);
bool nvm_websocket_indirect_hosted_input_local(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowValue *);
bool nvm_websocket_indirect_hosted_input_stack(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowValue *);
bool nvm_websocket_indirect_hosted_input_reference(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowReference *);
bool nvm_websocket_indirect_hosted_input_region(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_websocket_indirect_hosted_type(const NvmWebSocketIndirectHostedPlan *,uint32_t,NvmWebSocketNominalLayout *);
/* Catalog method ordinal -> original module import, like the query accessor. */
bool nvm_websocket_indirect_hosted_import(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint32_t *);
/* Copies an exact serialized input span; zero count permits NULL output. */
bool nvm_websocket_indirect_hosted_bytes(const NvmWebSocketIndirectHostedPlan *,size_t,void *,size_t);
bool nvm_websocket_indirect_hosted_call(const NvmWebSocketIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,NvmWebSocketIndirectFlowCall *);
/* I borrow counted constant bytes only while the immutable plan lives. */
bool nvm_websocket_indirect_hosted_string(const NvmWebSocketIndirectHostedPlan *,uint32_t,const uint8_t **,size_t *);
#endif
