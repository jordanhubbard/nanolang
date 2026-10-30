#ifndef NANOISA_MANAGED_UNION_FIELD_EXECUTION_PRIVATE_H
#define NANOISA_MANAGED_UNION_FIELD_EXECUTION_PRIVATE_H
#include "managed_union_field_proof_private.h"
#include "managed_record_array_execution.h"
/* Copied descriptive recipes only. No execution selector accepts these owners.
 * Module storage is exclusively stable during preparation. Writable outputs
 * must be disjoint from all input/owner storage (managed C API precondition).
 * Immutable getters copy in four logical probes including misses; callers
 * precharge. Byte-copy getter additionally costs count logical byte copies.
 * Failure preserves outputs; copied values outlive the owner. NOT_ESTABLISHED
 * never permits execution, and these recipes do not enroll any executor. */
enum NvmUnionSiteState { UFS_NOT_REQUIRED, UFS_NOT_ESTABLISHED, UFS_PROVED };
enum NvmUnionSiteOperation {
    UFS_NONE, UFS_RECORD_NEW, UFS_RECORD_PACK, UFS_VARIANT_PACK,
    UFS_RECORD_GET, UFS_PAYLOAD_GET, UFS_RECORD_SET, UFS_TAG, UFS_HALT
};
typedef struct {
    uint32_t function,pc;
    NvmRecordValueOrigins receiver,input,result;
    uint32_t nominal,layout;
    uint16_t member;
    uint8_t variant,operation,state,reserved[3];
} NvmUnionFieldSite;
#define NVM_UF_INVALID_BASE_RECIPE UINT8_MAX
typedef struct NvmUnionFieldExecutionPlan NvmUnionFieldExecutionPlan;
typedef struct {
    NvmRecordArrayExecutionInstruction base;
    NvmUnionFieldSite site;
} NvmUnionFieldExecutionInstruction;
NvmArrayEligibilityResult nvm_prepare_union_field_execution(const NvmModule *,NvmUnionFieldExecutionPlan **);
void nvm_union_field_execution_free(NvmUnionFieldExecutionPlan *);
bool nvm_union_field_execution_counts(const NvmUnionFieldExecutionPlan *,NvmRecordArrayExecutionCounts *);
bool nvm_union_field_execution_instruction(const NvmUnionFieldExecutionPlan *,uint32_t,NvmUnionFieldExecutionInstruction *);
bool nvm_union_field_execution_function(const NvmUnionFieldExecutionPlan *,uint32_t,NvmRecordArrayExecutionFunction *);
bool nvm_union_field_execution_parameter(const NvmUnionFieldExecutionPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_union_field_execution_header(const NvmUnionFieldExecutionPlan *,NvmHeader *);
bool nvm_union_field_execution_size(const NvmUnionFieldExecutionPlan *,NvmRecordArraySnapshotKind,uint32_t,uint32_t *);
bool nvm_union_field_execution_bytes(const NvmUnionFieldExecutionPlan *,NvmRecordArraySnapshotKind,uint32_t,uint32_t,uint32_t,void *);
bool nvm_union_field_execution_proof_counts(const NvmUnionFieldExecutionPlan *,NvmUnionFieldProofCounts *);
bool nvm_union_field_execution_origin(const NvmUnionFieldExecutionPlan *,uint32_t,NvmUnionFieldOrigin *);
bool nvm_union_field_execution_global(const NvmUnionFieldExecutionPlan *,uint32_t,NvmRecordValueOrigins *);
bool nvm_union_field_execution_result(const NvmUnionFieldExecutionPlan *,uint32_t,NvmRecordValueOrigins *);
bool nvm_union_field_execution_layout(const NvmUnionFieldExecutionPlan *,uint32_t,NvmDeclarationLayout *);
bool nvm_union_field_execution_field(const NvmUnionFieldExecutionPlan *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_union_field_execution_variant(const NvmUnionFieldExecutionPlan *,uint32_t,uint16_t,NvmUnionVariantFact *);
bool nvm_union_field_execution_nominal(const NvmUnionFieldExecutionPlan *,uint8_t,uint32_t,uint32_t *);
#endif
