#ifndef NANOISA_UNION_FIELD_SNAPSHOT_PRIVATE_H
#define NANOISA_UNION_FIELD_SNAPSHOT_PRIVATE_H
#include "managed_union_field_execution_private.h"
/* Descriptive independently reconstructed owner only, never execution authority.
 * Source is exclusively stable managed constructor-owned storage. Writable
 * outputs are disjoint from source/all owned storage as a C API precondition.
 * Failure preserves outputs; free(NULL) is valid. All copied queries have the
 * execution-plan four-probe bound; bytes additionally cost count copies.
 * There is no borrowed plan/module getter. Numeric and byte copies outlive us. */
typedef struct NvmUnionFieldImage NvmUnionFieldImage;
typedef struct {
    uint64_t source_retained_bytes,preparation_bytes_reserved,preparation_work_reserved;
} NvmUnionFieldImageCost;
NvmArrayEligibilityResult nvm_union_field_image_copy(const NvmUnionFieldExecutionPlan *,NvmUnionFieldImage **);
void nvm_union_field_image_free(NvmUnionFieldImage *);
bool nvm_union_field_image_cost(const NvmUnionFieldImage *,NvmUnionFieldImageCost *);
bool nvm_union_field_image_counts(const NvmUnionFieldImage *,NvmRecordArrayExecutionCounts *);
bool nvm_union_field_image_instruction(const NvmUnionFieldImage *,uint32_t,NvmUnionFieldExecutionInstruction *);
bool nvm_union_field_image_function(const NvmUnionFieldImage *,uint32_t,NvmRecordArrayExecutionFunction *);
bool nvm_union_field_image_parameter(const NvmUnionFieldImage *,uint32_t,uint16_t,uint8_t *);
bool nvm_union_field_image_header(const NvmUnionFieldImage *,NvmHeader *);
bool nvm_union_field_image_size(const NvmUnionFieldImage *,NvmRecordArraySnapshotKind,uint32_t,uint32_t *);
bool nvm_union_field_image_bytes(const NvmUnionFieldImage *,NvmRecordArraySnapshotKind,uint32_t,uint32_t,uint32_t,void *);
bool nvm_union_field_image_proof_counts(const NvmUnionFieldImage *,NvmUnionFieldProofCounts *);
bool nvm_union_field_image_origin(const NvmUnionFieldImage *,uint32_t,NvmUnionFieldOrigin *);
bool nvm_union_field_image_global(const NvmUnionFieldImage *,uint32_t,NvmRecordValueOrigins *);
bool nvm_union_field_image_result(const NvmUnionFieldImage *,uint32_t,NvmRecordValueOrigins *);
bool nvm_union_field_image_layout(const NvmUnionFieldImage *,uint32_t,NvmDeclarationLayout *);
bool nvm_union_field_image_field(const NvmUnionFieldImage *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_union_field_image_variant(const NvmUnionFieldImage *,uint32_t,uint16_t,NvmUnionVariantFact *);
bool nvm_union_field_image_nominal(const NvmUnionFieldImage *,uint8_t,uint32_t,uint32_t *);
#endif
