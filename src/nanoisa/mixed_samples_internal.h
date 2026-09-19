#ifndef NANOISA_MIXED_SAMPLES_INTERNAL_H
#define NANOISA_MIXED_SAMPLES_INTERNAL_H
#include "mixed_samples.h"
/* Internal preparation only. I expose neither an unchecked state constructor nor
 * executable admission. The immutable source module must outlive preparation;
 * returned facts/snapshots are owned by the opaque plan until its destruction. */
typedef struct NvmMixedSamplesPlan NvmMixedSamplesPlan;
typedef enum {
    NVM_MIXED_VALUE_UNAVAILABLE, NVM_MIXED_VALUE_SCALAR,
    NVM_MIXED_VALUE_OWNER, NVM_MIXED_VALUE_ORDINARY
} NvmMixedValueCategory;
typedef struct {
    uint8_t tag, mode;
    uint32_t global_layout;
    NvmMixedValueCategory category;
} NvmMixedDeclaration;
typedef struct {
    uint16_t parameter_count, local_count, max_stack;
    NvmMixedDeclaration parameters[8], result;
    uint16_t result_fields;
} NvmMixedSignature;
typedef struct {
    uint32_t source_record, global_layout, managed_record, field_count;
} NvmMixedRecordIdentity;
typedef struct {
    const uint8_t *layouts, *ownership;
    uint32_t layout_size, ownership_size;
} NvmMixedTransport;
/* PROVED means prepared analysis, NOT executable authority. No consumer accepts
 * a caller-built certificate. Failure preserves *out and all query outputs. */
NvmMixedShapeResult nvm_mixed_samples_prepare(const NvmModule *, NvmMixedSamplesPlan **out);
void nvm_mixed_samples_plan_free(NvmMixedSamplesPlan *);
bool nvm_mixed_samples_signature(const NvmMixedSamplesPlan *, uint32_t function,
                                 NvmMixedSignature *out);
bool nvm_mixed_samples_local(const NvmMixedSamplesPlan *, uint32_t function,
                             uint16_t local, NvmMixedDeclaration *out);
bool nvm_mixed_samples_record(const NvmMixedSamplesPlan *, uint32_t source_record,
                              NvmMixedRecordIdentity *out);
bool nvm_mixed_samples_transport(const NvmMixedSamplesPlan *, NvmMixedTransport *out);
/* Read-only obligations remain unsatisfied runtime work. No plan is admitted. */
const NvmMixedSamplesProof *nvm_mixed_samples_obligations(const NvmMixedSamplesPlan *);
#endif
