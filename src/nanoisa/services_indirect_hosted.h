#ifndef NANOISA_SERVICES_INDIRECT_HOSTED_H
#define NANOISA_SERVICES_INDIRECT_HOSTED_H
#include "services_indirect_flow.h"
#include "services_hosted.h"

#define NVM_SERVICES_INDIRECT_HOSTED_REVISION 1u
/* Private serialized preparation only. No dispatcher coverage, runtime admission,
 * grant, fuel, service execution or old single-state plan conversion. Inputs
 * stay immutable during preparation; output storage is disjoint from all input
 * and plan storage. External serialization matches the underlying queries.
 * Success owns every byte/fact. Failure and invalid getters preserve outputs. */
typedef struct NvmServicesIndirectHostedPlan NvmServicesIndirectHostedPlan;
typedef struct {
    uint32_t revision, entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t input_bytes, allocation_bound, query_storage_peak, retained_bound;
    bool runtime_admitted; /* Always false in this nonexecuting checkpoint. */
} NvmServicesIndirectHostedStartup;
typedef struct {
    NvmServicesCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint16_t frame_owner_peak, frame_reference_peak, frame_region_peak;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
    uint8_t entry_variant;
} NvmServicesIndirectHostedFunction;
/* allocation_bound conservatively includes simultaneous copied wire/bridge,
 * full query budget, retained plan and transient preparation allocations under
 * NVM_SERVICES_HOSTED_BYTES. retained_bound may overestimate retained query memory
 * by its preparation peak. Runtime/host resource bytes are not certified. */
NvmServicesFlowStatus nvm_services_indirect_hosted_prepare(const uint8_t *,size_t,NvmServicesIndirectHostedPlan **);
void nvm_services_indirect_hosted_free(NvmServicesIndirectHostedPlan *);
bool nvm_services_indirect_hosted_query_summary(const NvmServicesIndirectHostedPlan *,NvmServicesIndirectFlowSummary *);
bool nvm_services_indirect_hosted_startup(const NvmServicesIndirectHostedPlan *,NvmServicesIndirectHostedStartup *);
bool nvm_services_indirect_hosted_function(const NvmServicesIndirectHostedPlan *,uint32_t,NvmServicesIndirectHostedFunction *);
bool nvm_services_indirect_hosted_function_order(const NvmServicesIndirectHostedPlan *,uint32_t,uint32_t *);
bool nvm_services_indirect_hosted_local(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,NvmServicesFlowDeclaration *);
bool nvm_services_indirect_hosted_instruction(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,NvmServicesCodeInstruction *);
bool nvm_services_indirect_hosted_component(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint16_t *);
bool nvm_services_indirect_hosted_variant_count(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_services_indirect_hosted_variant(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,NvmServicesCyclicVariant *);
bool nvm_services_indirect_hosted_input_local(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowValue *);
bool nvm_services_indirect_hosted_input_stack(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowValue *);
bool nvm_services_indirect_hosted_input_reference(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowReference *);
bool nvm_services_indirect_hosted_input_region(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_services_indirect_hosted_type(const NvmServicesIndirectHostedPlan *,uint32_t,NvmServicesNominalLayout *);
/* instance*5+method -> original module import, like the query accessor. */
bool nvm_services_indirect_hosted_import(const NvmServicesIndirectHostedPlan *,uint32_t,uint32_t *);
/* Copies an exact serialized input span; zero count permits NULL output. */
bool nvm_services_indirect_hosted_bytes(const NvmServicesIndirectHostedPlan *,size_t,void *,size_t);
bool nvm_services_indirect_hosted_call(const NvmServicesIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,NvmServicesIndirectFlowCall *);
/* I borrow counted constant bytes only while the immutable plan lives. */
bool nvm_services_indirect_hosted_string(const NvmServicesIndirectHostedPlan *,uint32_t,const uint8_t **,size_t *);
#endif
