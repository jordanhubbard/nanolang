#ifndef NANOISA_MIXED_LAYOUT_VIEW_H
#define NANOISA_MIXED_LAYOUT_VIEW_H
#include "managed_record_plan.h"
#define NVM_MIXED_MAX_OWNERSHIP_BYTES (16u * 1024u * 1024u)
/* I describe declarations only; these classes grant no execution authority. */
typedef enum {
    NVM_MIXED_UNKNOWN, NVM_MIXED_RESOURCE,
    NVM_MIXED_ORDINARY_STRUCTURAL, NVM_MIXED_PENDING_ARRAY_PROOF
} NvmMixedLayoutClass;
typedef struct {
    NvmV2Layouts layouts;
    uint32_t record_count, managed_count;
    uint32_t source_to_global[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t global_to_source[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t managed_to_global[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t global_to_managed[NVM_RECORD_PLAN_MAX_LAYOUTS];
    NvmMixedLayoutClass classes[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint8_t *ownership; /* Exact owned transport, including descriptor/path order. */
    uint32_t ownership_size;
} NvmMixedLayoutView;
/* Every failure leaves *out unchanged. No verifier or selector is invoked. */
NvmRecordPlanResult nvm_describe_mixed_layouts(const NvmModule *, NvmMixedLayoutView **);
void nvm_mixed_layout_view_free(NvmMixedLayoutView *);
#endif
