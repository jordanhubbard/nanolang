#ifndef NANOISA_FILE_HOSTED_H
#define NANOISA_FILE_HOSTED_H
#include "file_body.h"
#define NVM_FILE_HOSTED_INPUT_BYTES (16u*1024u*1024u)
#define NVM_FILE_HOSTED_BYTES (64u*1024u*1024u)
typedef struct NvmFileHostedPlan NvmFileHostedPlan;
typedef struct {
    uint32_t entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t allocation_bound;
} NvmFileHostedStartup;
typedef struct {
    NvmFileCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
} NvmFileHostedFunction;
/* Private serialized-v2 startup/storage facts only. No host/service operation,
 * public admission or concrete carrier ABI certification. Input stays immutable
 * during the call; output is disjoint. Reports own all facts; failure preserves
 * output. Calls require the underlying private query's external serialization.
 * Value-slot upper bounds exclude bytes inside a runtime carrier/host resource;
 * a later matched runtime must separately check those extents and cleanup. */
NvmFileFlowStatus nvm_file_hosted_prepare(const uint8_t *,size_t,NvmFileHostedPlan **);
void nvm_file_hosted_free(NvmFileHostedPlan *);
bool nvm_file_hosted_startup(const NvmFileHostedPlan *,NvmFileHostedStartup *);
bool nvm_file_hosted_function(const NvmFileHostedPlan *,uint32_t,NvmFileHostedFunction *);
bool nvm_file_hosted_local(const NvmFileHostedPlan *,uint32_t,uint16_t,NvmFileFlowDeclaration *);
bool nvm_file_hosted_instruction(const NvmFileHostedPlan *,uint32_t,uint16_t,
                                NvmFileCodeInstruction *,NvmFileBodyInstruction *);
/* Read-only exact nominal/catalog maps owned by this same hosted plan. */
bool nvm_file_hosted_type(const NvmFileHostedPlan *,uint32_t,NvmFileNominalLayout *);
bool nvm_file_hosted_import(const NvmFileHostedPlan *,uint32_t,uint32_t *);
#endif
