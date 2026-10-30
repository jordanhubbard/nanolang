#ifndef NANOISA_MANAGED_RECORD_MAP_BOUNDARY_H
#define NANOISA_MANAGED_RECORD_MAP_BOUNDARY_H
#include "managed_record_map_execution.h"
/* Immutable data only; no public execution or imported-object admission.
 * Output storage must not overlap the input module, owner or any owned storage.
 * Getters allocate nothing, preserve output on failure and cost four logical
 * probes. Borrowed execution data expires when this owner is freed. */
typedef struct NvmRecordMapBoundaryPlan NvmRecordMapBoundaryPlan;
typedef struct {
    uint32_t globals;
    uint64_t peak_bytes_reserved, work_reserved;
} NvmRecordMapBoundaryCounts;
NvmArrayEligibilityResult nvm_prepare_record_map_boundary(
    const NvmModule *, NvmRecordMapBoundaryPlan **);
void nvm_record_map_boundary_free(NvmRecordMapBoundaryPlan *);
bool nvm_record_map_boundary_counts(const NvmRecordMapBoundaryPlan *,NvmRecordMapBoundaryCounts *);
bool nvm_record_map_boundary_global(const NvmRecordMapBoundaryPlan *,uint32_t,NvmRecordValueOrigins *);
const NvmRecordMapExecutionPlan *nvm_record_map_boundary_execution(const NvmRecordMapBoundaryPlan *);
#endif
