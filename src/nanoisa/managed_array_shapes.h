#ifndef NANOISA_MANAGED_ARRAY_SHAPES_H
#define NANOISA_MANAGED_ARRAY_SHAPES_H
#include "nvm_format.h"
#include <stdint.h>

/* Private analysis only: no verifier profile or executable admission is changed. */
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
typedef struct {
    NvmArrayEligibilityStatus status;
    uint32_t function, pc;
    char message[192];
} NvmArrayEligibilityResult;
/* On failure, *out is untouched. The module is borrowed and never mutated. */
NvmArrayEligibilityResult nvm_analyze_managed_arrays(
    const NvmModule *module, NvmArrayEligibilityReport **out);
void nvm_array_eligibility_free(NvmArrayEligibilityReport *report);
#ifdef NMA_TESTING
void nvm_array_analysis_fail_after(uint64_t allocations);
#endif
#endif
