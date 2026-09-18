#ifndef NANOISA_MANAGED_ARRAY_SHAPES_H
#define NANOISA_MANAGED_ARRAY_SHAPES_H
#include "nvm_format.h"
#include <stdint.h>

/* Shape evidence only: callers must also enforce their execution profile and runtime checks. */
typedef enum {
    NVM_ARRAY_ELIGIBLE, NVM_ARRAY_UNRESOLVED, NVM_ARRAY_INVALID,
    NVM_ARRAY_LIMIT, NVM_ARRAY_MEMORY
} NvmArrayEligibilityStatus;
typedef struct {
    uint32_t function, pc;
    uint16_t child_tags;
    uint8_t declared_tag, packed;
} NvmArrayOrigin;
typedef struct {
    uint32_t origin_count, checked_writes;
    /* Wrong receiver/index tags still require the existing runtime checks. */
    uint32_t runtime_tag_checks;
    NvmArrayOrigin origins[64];
} NvmArrayEligibilityReport;
/* I keep this graph report separate from the existing leaf ABI/consumer. */
typedef struct {
    NvmArrayEligibilityReport arrays;
    uint64_t child_origins[64];
    uint8_t child_unknown[64];
} NvmArrayGraphEligibilityReport;
typedef struct {
    NvmArrayEligibilityStatus status;
    uint32_t function, pc;
    char message[192];
} NvmArrayEligibilityResult;
/* On failure, *out is untouched. The module is borrowed and never mutated. */
NvmArrayEligibilityResult nvm_analyze_managed_arrays(
    const NvmModule *module, NvmArrayEligibilityReport **out);
void nvm_array_eligibility_free(NvmArrayEligibilityReport *report);
/* This query grants no executable profile admission or collection safe point. */
NvmArrayEligibilityResult nvm_analyze_managed_array_graphs(
    const NvmModule *module, NvmArrayGraphEligibilityReport **out);
void nvm_array_graph_eligibility_free(NvmArrayGraphEligibilityReport *report);
/* I select graph lifetime only after leaf UNRESOLVED and graph ELIGIBLE.
 * Other failures are final; on failure *graph_required stays unchanged.
 * Selection remains shape evidence: the profile and emitter supply lifetime. */
NvmArrayEligibilityResult nvm_select_managed_array_mode(
    const NvmModule *module, int *graph_required);
#ifdef NMA_TESTING
void nvm_array_analysis_fail_after(uint64_t allocations);
#endif
#endif
