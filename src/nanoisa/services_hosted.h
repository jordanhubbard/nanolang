#ifndef NANOISA_SERVICES_HOSTED_H
#define NANOISA_SERVICES_HOSTED_H
#include "services_body.h"
#define NVM_SERVICES_HOSTED_INPUT_BYTES (16u*1024u*1024u)
#define NVM_SERVICES_HOSTED_BYTES (64u*1024u*1024u)
typedef struct NvmServicesHostedPlan NvmServicesHostedPlan;
typedef struct {
    uint32_t entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t allocation_bound;
} NvmServicesHostedStartup;
typedef struct {
    NvmServicesCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
} NvmServicesHostedFunction;
/* Private serialized-v2 startup/storage facts only. No host/service operation,
 * public admission or concrete carrier ABI certification. Input stays immutable
 * during the call; output is disjoint. Reports own all facts; failure preserves
 * output. Calls require the underlying private query's external serialization.
 * Value-slot upper bounds exclude bytes inside a runtime carrier/host resource;
 * a later matched runtime must separately check those extents and cleanup. */
NvmServicesFlowStatus nvm_services_hosted_prepare(const uint8_t *,size_t,NvmServicesHostedPlan **);
void nvm_services_hosted_free(NvmServicesHostedPlan *);
bool nvm_services_hosted_startup(const NvmServicesHostedPlan *,NvmServicesHostedStartup *);
bool nvm_services_hosted_function(const NvmServicesHostedPlan *,uint32_t,NvmServicesHostedFunction *);
bool nvm_services_hosted_local(const NvmServicesHostedPlan *,uint32_t,uint16_t,NvmServicesFlowDeclaration *);
bool nvm_services_hosted_instruction(const NvmServicesHostedPlan *,uint32_t,uint16_t,
                                NvmServicesCodeInstruction *,NvmServicesBodyInstruction *);
/* Read-only exact nominal/catalog maps owned by this same hosted plan. */
bool nvm_services_hosted_type(const NvmServicesHostedPlan *,uint32_t,NvmServicesNominalLayout *);
bool nvm_services_hosted_import(const NvmServicesHostedPlan *,uint32_t,uint32_t *);
#endif
