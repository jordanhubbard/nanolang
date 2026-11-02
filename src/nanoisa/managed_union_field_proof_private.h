#ifndef NANOISA_MANAGED_UNION_FIELD_PROOF_PRIVATE_H
#define NANOISA_MANAGED_UNION_FIELD_PROOF_PRIVATE_H
#include "managed_record_shapes.h"
#include "ownership_union_fields_private.h"
#define NVM_UNION_FIELD_ORIGIN 3u
/* Descriptive copies only; these facts never select an execution route.
 * Module/backing storage is exclusively stable during preparation. Every
 * writable output is disjoint from the module and all owner storage. Each
 * immutable getter has a four-logical-probe bound including misses; callers
 * precharge when needed. Failure leaves outputs unchanged. */
typedef struct NvmUnionFieldProof NvmUnionFieldProof;
typedef struct {
    uint32_t function, pc, nominal_ordinal, layout_index;
    uint8_t kind, variant;
    uint16_t member_count;
    NvmRecordValueOrigins member;
} NvmUnionFieldOrigin;
typedef struct {
    uint32_t origins, globals, functions;
    uint64_t bytes_reserved, work_reserved;
} NvmUnionFieldProofCounts;
NvmArrayEligibilityResult nvm_union_field_proof_prepare(const NvmModule *,NvmUnionFieldProof **);
void nvm_union_field_proof_free(NvmUnionFieldProof *);
bool nvm_union_field_proof_counts(const NvmUnionFieldProof *,NvmUnionFieldProofCounts *);
bool nvm_union_field_proof_origin(const NvmUnionFieldProof *,uint32_t,NvmUnionFieldOrigin *);
bool nvm_union_field_proof_global(const NvmUnionFieldProof *,uint32_t,NvmRecordValueOrigins *);
bool nvm_union_field_proof_result(const NvmUnionFieldProof *,uint32_t,NvmRecordValueOrigins *);
bool nvm_union_field_proof_stack(const NvmUnionFieldProof *,uint32_t,uint16_t *);
bool nvm_union_field_proof_layout(const NvmUnionFieldProof *,uint32_t,NvmDeclarationLayout *);
bool nvm_union_field_proof_field(const NvmUnionFieldProof *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_union_field_proof_variant(const NvmUnionFieldProof *,uint32_t,uint16_t,NvmUnionVariantFact *);
bool nvm_union_field_proof_nominal(const NvmUnionFieldProof *,uint8_t,uint32_t,uint32_t *);
#endif
