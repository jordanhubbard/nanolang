#ifndef NANOISA_RECORD_MAP_SNAPSHOT_PRIVATE_H
#define NANOISA_RECORD_MAP_SNAPSHOT_PRIVATE_H
#include "managed_record_map_execution.h"
typedef struct NvmRecordMapImage NvmRecordMapImage;
typedef struct {
    uint64_t source_retained_bytes,preparation_bytes_reserved,preparation_work_reserved;
} NvmRecordMapImageCost;
NvmArrayEligibilityResult nvm_record_map_image_copy(const NvmRecordMapExecutionPlan *,NvmRecordMapImage **);
NvmArrayEligibilityResult nvm_record_map_image_copy_reserved(const NvmRecordMapExecutionPlan *,
    uint64_t extra_bytes,uint64_t extra_work,NvmRecordMapImage **);
const NvmRecordMapExecutionPlan *nvm_record_map_image_plan(const NvmRecordMapImage *);
void nvm_record_map_image_free(NvmRecordMapImage *);
bool nvm_record_map_image_cost(const NvmRecordMapImage *,NvmRecordMapImageCost *);
/* Private lifetime-bound borrow only; this grants no executable admission. */
const NvmModule *nvm_record_map_image_borrow(const NvmRecordMapImage *);
bool nvm_record_map_image_counts(const NvmRecordMapImage *,NvmRecordMapExecutionCounts *);
bool nvm_record_map_image_instruction(const NvmRecordMapImage *,uint32_t,NvmRecordMapExecutionInstruction *);
bool nvm_record_map_image_field(const NvmRecordMapImage *,uint32_t,uint16_t,uint8_t *,uint8_t *);
bool nvm_record_map_image_origin(const NvmRecordMapImage *,uint32_t,uint8_t *,uint8_t *);
#endif
