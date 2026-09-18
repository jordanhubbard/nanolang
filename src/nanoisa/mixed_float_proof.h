#ifndef NANOISA_MIXED_FLOAT_PROOF_H
#define NANOISA_MIXED_FLOAT_PROOF_H
#include "mixed_layout_view.h"
#define NVM_MIXED_FLOAT_ORIGINS 64u
#define NVM_MIXED_FLOAT_INSTRUCTIONS 4096u
/* Shape success is NOT executable scalar/affine authority. A later conjunction
 * must independently check lifetimes, scalar obligations and the runtime profile. */
typedef enum {
    NVM_MIXED_SHAPE_PROVED, NVM_MIXED_SHAPE_UNRESOLVED,
    NVM_MIXED_SHAPE_INVALID, NVM_MIXED_SHAPE_LIMIT, NVM_MIXED_SHAPE_MEMORY
} NvmMixedShapeStatus;
typedef struct {
    NvmMixedShapeStatus status;
    uint32_t function, pc;
    const char *message;
} NvmMixedShapeResult;
typedef struct {
    uint16_t tags;
    uint64_t origins;
} NvmMixedFieldFact;
typedef struct {
    uint32_t function, pc, global_layout, source_record, field_start;
    uint16_t field_count;
    uint8_t tag;
} NvmMixedFloatOrigin;
typedef struct {
    uint32_t function, pc;
    uint16_t read_tags; /* ARR_GET remains FLOAT|VOID. */
    uint16_t required_tags, actual_tags; /* Nonzero mismatch needs scalar checking. */
} NvmMixedScalarObligation;
typedef struct {
    NvmMixedLayoutView *view; /* Owned descriptive facts; classes stay unchanged. */
    uint32_t origin_count, field_count, obligation_count, checked_writes;
    bool requires_affine_verification; /* Always true: shape never proves ownership. */
    NvmMixedFloatOrigin origins[NVM_MIXED_FLOAT_ORIGINS];
    NvmMixedFieldFact *fields;
    NvmMixedScalarObligation obligations[NVM_MIXED_FLOAT_INSTRUCTIONS];
} NvmMixedFloatProof;
/* I borrow an immutable module for this query only. Failure preserves *out.
 * No public verifier, selector, source lowering or execution is invoked. */
NvmMixedShapeResult nvm_analyze_mixed_float_origins(const NvmModule *, NvmMixedFloatProof **);
void nvm_mixed_float_proof_free(NvmMixedFloatProof *);
#endif
