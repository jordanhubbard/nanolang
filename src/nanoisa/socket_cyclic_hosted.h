#ifndef NANOISA_SOCKET_CYCLIC_HOSTED_H
#define NANOISA_SOCKET_CYCLIC_HOSTED_H
#include "socket_cyclic.h"
#include "socket_hosted.h"

#define NVM_SOCKET_CYCLIC_HOSTED_REVISION 1u
/* Trusted serialized preparation only. No target coverage, runtime admission,
 * grant, fuel, service execution or old single-state plan conversion. Inputs
 * stay immutable during preparation; output storage is disjoint from all input
 * and plan storage. External serialization matches the underlying queries.
 * Success owns every byte/fact. Failure and invalid getters preserve outputs. */
typedef struct NvmSocketCyclicHostedPlan NvmSocketCyclicHostedPlan;
typedef struct {
    uint32_t revision, entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t input_bytes, allocation_bound, query_storage_peak, retained_bound;
    bool runtime_admitted; /* Always false: descriptive preparation is not admission. */
} NvmSocketCyclicHostedStartup;
typedef struct {
    NvmSocketCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint16_t frame_owner_peak, frame_reference_peak, frame_region_peak;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
    uint8_t entry_variant;
} NvmSocketCyclicHostedFunction;
/* allocation_bound conservatively includes simultaneous copied wire/bridge,
 * full query budget, retained plan and transient preparation allocations under
 * NVM_SOCKET_HOSTED_BYTES. retained_bound may overestimate retained query memory
 * by its preparation peak. Runtime/host resource bytes are not certified. */
NvmSocketFlowStatus nvm_socket_cyclic_hosted_prepare(const uint8_t *,size_t,NvmSocketCyclicHostedPlan **);
void nvm_socket_cyclic_hosted_free(NvmSocketCyclicHostedPlan *);
bool nvm_socket_cyclic_hosted_query_summary(const NvmSocketCyclicHostedPlan *,NvmSocketCyclicSummary *);
bool nvm_socket_cyclic_hosted_startup(const NvmSocketCyclicHostedPlan *,NvmSocketCyclicHostedStartup *);
bool nvm_socket_cyclic_hosted_function(const NvmSocketCyclicHostedPlan *,uint32_t,NvmSocketCyclicHostedFunction *);
bool nvm_socket_cyclic_hosted_function_order(const NvmSocketCyclicHostedPlan *,uint32_t,uint32_t *);
bool nvm_socket_cyclic_hosted_local(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,NvmSocketFlowDeclaration *);
bool nvm_socket_cyclic_hosted_instruction(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,NvmSocketCodeInstruction *);
bool nvm_socket_cyclic_hosted_component(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,uint16_t *);
bool nvm_socket_cyclic_hosted_variant_count(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_socket_cyclic_hosted_variant(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,NvmSocketCyclicVariant *);
bool nvm_socket_cyclic_hosted_input_local(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmSocketFlowValue *);
bool nvm_socket_cyclic_hosted_input_stack(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmSocketFlowValue *);
bool nvm_socket_cyclic_hosted_input_reference(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmSocketFlowReference *);
bool nvm_socket_cyclic_hosted_input_region(const NvmSocketCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_socket_cyclic_hosted_type(const NvmSocketCyclicHostedPlan *,uint32_t,NvmSocketNominalLayout *);
/* Catalog method ordinal -> original module import, like the query accessor. */
bool nvm_socket_cyclic_hosted_import(const NvmSocketCyclicHostedPlan *,uint32_t,uint32_t *);
/* Copies an exact serialized input span; zero count permits NULL output. */
bool nvm_socket_cyclic_hosted_bytes(const NvmSocketCyclicHostedPlan *,size_t,void *,size_t);
#endif
