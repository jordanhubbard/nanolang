#ifndef NANOISA_RECORD_MAP_NATIVE_TARGET_INTERNAL_H
#define NANOISA_RECORD_MAP_NATIVE_TARGET_INTERNAL_H
#include "record_map_snapshot_private.h"
/* Internal immutable-image classification only. This grants no execution or
 * source-generation capability. The caller owns and stabilizes the image. */
typedef struct {
    NvmArrayEligibilityResult eligibility;
    uint64_t work;
} NvmRecordMapTargetResult;
NvmRecordMapTargetResult nvm_record_map_native_target(const NvmRecordMapImage *,uint64_t allowance);
#endif
