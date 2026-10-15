#ifndef NANOISA_SERVICES_CYCLIC_HOSTED_H
#define NANOISA_SERVICES_CYCLIC_HOSTED_H
#include "services_cyclic.h"
#include "services_hosted.h"

#define NVM_SERVICES_CYCLIC_HOSTED_REVISION 1u
/* Trusted serialized preparation only. No target coverage, runtime admission,
 * grant, fuel, service execution or old single-state plan conversion. Inputs
 * stay immutable during preparation; output storage is disjoint from all input
 * and plan storage. External serialization matches the underlying queries.
 * Success owns every byte/fact. Failure and invalid getters preserve outputs. */
typedef struct NvmServicesCyclicHostedPlan NvmServicesCyclicHostedPlan;
typedef struct {
    uint32_t revision, entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t input_bytes, allocation_bound, query_storage_peak, retained_bound;
    bool runtime_admitted; /* Always false: descriptive preparation is not admission. */
} NvmServicesCyclicHostedStartup;
typedef struct {
    NvmServicesCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint16_t frame_owner_peak, frame_reference_peak, frame_region_peak;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
    uint8_t entry_variant;
} NvmServicesCyclicHostedFunction;
/* allocation_bound conservatively includes simultaneous copied wire/bridge,
 * full query budget, retained plan and transient preparation allocations under
 * NVM_SERVICES_HOSTED_BYTES. retained_bound may overestimate retained query memory
 * by its preparation peak. Runtime/host resource bytes are not certified. */
NvmServicesFlowStatus nvm_services_cyclic_hosted_prepare(const uint8_t *,size_t,NvmServicesCyclicHostedPlan **);
void nvm_services_cyclic_hosted_free(NvmServicesCyclicHostedPlan *);
bool nvm_services_cyclic_hosted_query_summary(const NvmServicesCyclicHostedPlan *,NvmServicesCyclicSummary *);
bool nvm_services_cyclic_hosted_startup(const NvmServicesCyclicHostedPlan *,NvmServicesCyclicHostedStartup *);
bool nvm_services_cyclic_hosted_function(const NvmServicesCyclicHostedPlan *,uint32_t,NvmServicesCyclicHostedFunction *);
bool nvm_services_cyclic_hosted_function_order(const NvmServicesCyclicHostedPlan *,uint32_t,uint32_t *);
bool nvm_services_cyclic_hosted_local(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,NvmServicesFlowDeclaration *);
bool nvm_services_cyclic_hosted_instruction(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,NvmServicesCodeInstruction *);
bool nvm_services_cyclic_hosted_component(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,uint16_t *);
bool nvm_services_cyclic_hosted_variant_count(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_services_cyclic_hosted_variant(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,NvmServicesCyclicVariant *);
bool nvm_services_cyclic_hosted_input_local(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowValue *);
bool nvm_services_cyclic_hosted_input_stack(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowValue *);
bool nvm_services_cyclic_hosted_input_reference(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowReference *);
bool nvm_services_cyclic_hosted_input_region(const NvmServicesCyclicHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_services_cyclic_hosted_type(const NvmServicesCyclicHostedPlan *,uint32_t,NvmServicesNominalLayout *);
/* Catalog method ordinal -> original module import, like the query accessor. */
bool nvm_services_cyclic_hosted_import(const NvmServicesCyclicHostedPlan *,uint32_t,uint32_t *);
/* Copies an exact serialized input span; zero count permits NULL output. */
bool nvm_services_cyclic_hosted_bytes(const NvmServicesCyclicHostedPlan *,size_t,void *,size_t);
#endif
