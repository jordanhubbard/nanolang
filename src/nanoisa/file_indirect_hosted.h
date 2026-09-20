#ifndef NANOISA_FILE_INDIRECT_HOSTED_H
#define NANOISA_FILE_INDIRECT_HOSTED_H
#include "file_indirect_flow.h"
#include "file_hosted.h"

#define NVM_FILE_INDIRECT_HOSTED_REVISION 1u
/* Private serialized preparation only. No dispatcher coverage, runtime admission,
 * grant, fuel, service execution or old single-state plan conversion. Inputs
 * stay immutable during preparation; output storage is disjoint from all input
 * and plan storage. External serialization matches the underlying queries.
 * Success owns every byte/fact. Failure and invalid getters preserve outputs. */
typedef struct NvmFileIndirectHostedPlan NvmFileIndirectHostedPlan;
typedef struct {
    uint32_t revision, entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t input_bytes, allocation_bound, query_storage_peak, retained_bound;
    bool runtime_admitted; /* Always false in this nonexecuting checkpoint. */
} NvmFileIndirectHostedStartup;
typedef struct {
    NvmFileCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint16_t frame_owner_peak, frame_reference_peak, frame_region_peak;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
    uint8_t entry_variant;
} NvmFileIndirectHostedFunction;
/* allocation_bound conservatively includes simultaneous copied wire/bridge,
 * full query budget, retained plan and transient preparation allocations under
 * NVM_FILE_HOSTED_BYTES. retained_bound may overestimate retained query memory
 * by its preparation peak. Runtime/host resource bytes are not certified. */
NvmFileFlowStatus nvm_file_indirect_hosted_prepare(const uint8_t *,size_t,NvmFileIndirectHostedPlan **);
void nvm_file_indirect_hosted_free(NvmFileIndirectHostedPlan *);
bool nvm_file_indirect_hosted_query_summary(const NvmFileIndirectHostedPlan *,NvmFileIndirectFlowSummary *);
bool nvm_file_indirect_hosted_startup(const NvmFileIndirectHostedPlan *,NvmFileIndirectHostedStartup *);
bool nvm_file_indirect_hosted_function(const NvmFileIndirectHostedPlan *,uint32_t,NvmFileIndirectHostedFunction *);
bool nvm_file_indirect_hosted_function_order(const NvmFileIndirectHostedPlan *,uint32_t,uint32_t *);
bool nvm_file_indirect_hosted_local(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,NvmFileFlowDeclaration *);
bool nvm_file_indirect_hosted_instruction(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,NvmFileCodeInstruction *);
bool nvm_file_indirect_hosted_component(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint16_t *);
bool nvm_file_indirect_hosted_variant_count(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_file_indirect_hosted_variant(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,NvmFileCyclicVariant *);
bool nvm_file_indirect_hosted_input_local(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmFileFlowValue *);
bool nvm_file_indirect_hosted_input_stack(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmFileFlowValue *);
bool nvm_file_indirect_hosted_input_reference(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,NvmFileFlowReference *);
bool nvm_file_indirect_hosted_input_region(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_file_indirect_hosted_type(const NvmFileIndirectHostedPlan *,uint32_t,NvmFileNominalLayout *);
/* Catalog method ordinal -> original module import, like the query accessor. */
bool nvm_file_indirect_hosted_import(const NvmFileIndirectHostedPlan *,uint32_t,uint32_t *);
/* Copies an exact serialized input span; zero count permits NULL output. */
bool nvm_file_indirect_hosted_bytes(const NvmFileIndirectHostedPlan *,size_t,void *,size_t);
bool nvm_file_indirect_hosted_call(const NvmFileIndirectHostedPlan *,uint32_t,uint16_t,uint8_t,NvmFileIndirectFlowCall *);
#endif
