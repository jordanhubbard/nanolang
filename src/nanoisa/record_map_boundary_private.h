#ifndef NANOISA_RECORD_MAP_BOUNDARY_PRIVATE_H
#define NANOISA_RECORD_MAP_BOUNDARY_PRIVATE_H
#include "managed_record_map_boundary.h"
/* Positive construction reservation only. No execution authority or credits.
 * Normal public construction retains its zero-addition schedule. */
NvmArrayEligibilityResult nvm_prepare_record_map_boundary_reserved_private(
    const NvmModule *,NvmRecordMapBoundaryPlan **,uint64_t,uint64_t);
#endif
