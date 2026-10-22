#ifndef NANOISA_RECORD_MAP_NATIVE_TARGET_INTERNAL_H
#define NANOISA_RECORD_MAP_NATIVE_TARGET_INTERNAL_H
#include "record_map_snapshot_private.h"
/* Internal immutable-image classification only. This grants no execution or
 * source-generation capability. The caller owns and stabilizes the image. */
typedef struct {
    NvmArrayEligibilityResult eligibility;
    uint64_t work;
} NvmRecordMapTargetResult;
/* Optional disjoint output is published only on full ELIGIBLE validation. */
NvmRecordMapTargetResult nvm_record_map_native_target_pair(const NvmRecordMapImage *,uint64_t allowance,uint8_t *);
NvmRecordMapTargetResult nvm_record_map_native_target(const NvmRecordMapImage *,uint64_t allowance);
#endif
