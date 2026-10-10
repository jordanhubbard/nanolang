#ifndef NANOISA_WEBSOCKET_CYCLIC_HOSTED_H
#define NANOISA_WEBSOCKET_CYCLIC_HOSTED_H
#include "websocket_cyclic.h"
#include "websocket_hosted.h"

#define NVM_WEBSOCKET_CYCLIC_HOSTED_REVISION 1u
/* Trusted serialized preparation only. No target coverage, runtime admission,
 * grant, fuel, service execution or old single-state plan conversion. Inputs
 * stay immutable during preparation; output storage is disjoint from all input
 * and plan storage. External serialization matches the underlying queries.
 * Success owns every byte/fact. Failure and invalid getters preserve outputs. */
typedef struct NvmWebSocketCyclicHostedPlan NvmWebSocketCyclicHostedPlan;
typedef struct {
    uint32_t revision, entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t input_bytes, allocation_bound, query_storage_peak, retained_bound;
    bool runtime_admitted; /* Always false: descriptive preparation is not admission. */
} NvmWebSocketCyclicHostedStartup;
typedef struct {
    NvmWebSocketCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint16_t frame_owner_peak, frame_reference_peak, frame_region_peak;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
    uint8_t entry_variant;
} NvmWebSocketCyclicHostedFunction;
/* allocation_bound conservatively includes simultaneous copied wire/bridge,
 * full query budget, retained plan and transient preparation allocations under
 * NVM_WEBSOCKET_HOSTED_BYTES. retained_bound may overestimate retained query memory
 * by its preparation peak. Runtime/host resource bytes are not certified. */
NvmWebSocketFlowStatus nvm_websocket_cyclic_hosted_prepare(const uint8_t *,size_t,NvmWebSocketCyclicHostedPlan **);
void nvm_websocket_cyclic_hosted_free(NvmWebSocketCyclicHostedPlan *);
bool nvm_websocket_cyclic_hosted_query_summary(const NvmWebSocketCyclicHostedPlan *,NvmWebSocketCyclicSummary *);
bool nvm_websocket_cyclic_hosted_startup(const NvmWebSocketCyclicHostedPlan *,NvmWebSocketCyclicHostedStartup *);
bool nvm_websocket_cyclic_hosted_function(const NvmWebSocketCyclicHostedPlan *,uint32_t,NvmWebSocketCyclicHostedFunction *);
bool nvm_websocket_cyclic_hosted_function_order(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint32_t *);
bool nvm_websocket_cyclic_hosted_local(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,NvmWebSocketFlowDeclaration *);
bool nvm_websocket_cyclic_hosted_instruction(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,NvmWebSocketCodeInstruction *);
bool nvm_websocket_cyclic_hosted_component(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,uint16_t *);
bool nvm_websocket_cyclic_hosted_variant_count(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_websocket_cyclic_hosted_variant(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,NvmWebSocketCyclicVariant *);
bool nvm_websocket_cyclic_hosted_input_local(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowValue *);
bool nvm_websocket_cyclic_hosted_input_stack(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowValue *);
bool nvm_websocket_cyclic_hosted_input_reference(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowReference *);
bool nvm_websocket_cyclic_hosted_input_region(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_websocket_cyclic_hosted_type(const NvmWebSocketCyclicHostedPlan *,uint32_t,NvmWebSocketNominalLayout *);
/* Catalog method ordinal -> original module import, like the query accessor. */
bool nvm_websocket_cyclic_hosted_import(const NvmWebSocketCyclicHostedPlan *,uint32_t,uint32_t *);
/* Copies an exact serialized input span; zero count permits NULL output. */
bool nvm_websocket_cyclic_hosted_bytes(const NvmWebSocketCyclicHostedPlan *,size_t,void *,size_t);
#endif
