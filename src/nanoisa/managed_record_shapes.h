#ifndef NANOISA_MANAGED_RECORD_SHAPES_H
#define NANOISA_MANAGED_RECORD_SHAPES_H
#include "managed_array_shapes.h"
#include "managed_record_plan.h"

/* A separate non-admitting query: existing array reports/selectors are unchanged. */
typedef struct {
    uint64_t origins;
    uint16_t tags;
    uint8_t unknown;
} NvmRecordValueOrigins;
typedef enum { NVM_HEAP_ORIGIN_ARRAY, NVM_HEAP_ORIGIN_RECORD } NvmHeapOriginKind;
typedef struct {
    uint32_t function, pc;
    uint32_t record_ordinal, layout_index, field_start;
    uint16_t field_count;
    uint8_t kind, declared_tag, packed;
    NvmRecordValueOrigins children; /* Array contents; empty for records. */
} NvmRecordHeapOrigin;
typedef struct {
    uint32_t origin_count, field_value_count, checked_field_writes;
    uint32_t checked_array_writes, runtime_tag_checks;
    NvmRecordHeapOrigin origins[64];
    NvmRecordValueOrigins *fields; /* Owned flattened per-origin field summaries. */
} NvmRecordEligibilityReport;
/* I require checked COMPLETE ordinary declarations. Success establishes only
 * the bounded field/origin obligations, never execution eligibility or lifetime.
 * Every failure preserves *out and the borrowed module. */
NvmArrayEligibilityResult nvm_analyze_managed_records(
    const NvmModule *module, NvmRecordEligibilityReport **out);
void nvm_record_eligibility_free(NvmRecordEligibilityReport *report);
/* Shared checked selection. I publish a complete owned plan only on success.
 * The caller still checks supported instructions/signatures and runtime lowering. */
typedef enum { NVM_MANAGED_LEAF, NVM_MANAGED_ARRAY_GRAPH, NVM_MANAGED_RECORD } NvmManagedHeapMode;
typedef struct {
    NvmManagedHeapMode mode;
    NvmRecordPlan *records;
    NvmRecordEligibilityReport *fields;
} NvmManagedHeapPlan;
NvmArrayEligibilityResult nvm_select_managed_heap(const NvmModule *, int mutable_arrays,
                                                 NvmManagedHeapPlan **out);
void nvm_managed_heap_plan_free(NvmManagedHeapPlan *);
#endif
