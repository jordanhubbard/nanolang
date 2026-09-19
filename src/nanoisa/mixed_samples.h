#ifndef NANOISA_MIXED_SAMPLES_H
#define NANOISA_MIXED_SAMPLES_H
#include "mixed_float_proof.h"
#define NVM_MIXED_SCALAR_CHECKS (2u * NVM_MIXED_FLOAT_INSTRUCTIONS)
typedef enum { NVM_MIXED_CHECK_FLOAT, NVM_MIXED_COMPARE_VALUE } NvmMixedScalarPolicy;
typedef struct {
    uint32_t function, pc;
    uint16_t operand, actual_tags, required_tags;
    NvmMixedScalarPolicy policy;
} NvmMixedScalarCheck;
typedef struct {
    NvmMixedFloatProof *shape; /* Owned; descriptive classes remain unchanged. */
    uint32_t checked_functions, checked_instructions, visits, check_count;
    uint16_t max_stack[8];
    uint32_t managed_count;
    uint32_t global_to_managed[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t managed_to_global[NVM_RECORD_PLAN_MAX_LAYOUTS];
    NvmMixedScalarCheck checks[NVM_MIXED_SCALAR_CHECKS];
    bool affine_checked;
    bool runtime_admitted; /* Always false; checks have NOT run in a VM/native backend. */
} NvmMixedSamplesProof;
/* I borrow one immutable module, construct all facts internally, and leave *out
 * unchanged on failure. PROVED means private composed analysis only. No public
 * verification, selector, lowering or execution is invoked. FLOAT|VOID typed
 * uses retain explicit runtime checks, never an inferred exact FLOAT guarantee. */
NvmMixedShapeResult nvm_analyze_mixed_samples(const NvmModule *, NvmMixedSamplesProof **);
void nvm_mixed_samples_proof_free(NvmMixedSamplesProof *);
#endif
