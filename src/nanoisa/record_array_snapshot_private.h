#ifndef NANOISA_RECORD_ARRAY_SNAPSHOT_PRIVATE_H
#define NANOISA_RECORD_ARRAY_SNAPSHOT_PRIVATE_H
#include "managed_record_array_execution.h"
/* I copy a prepared image; I confer no public execution authority. */
typedef struct NvmRecordArrayImage NvmRecordArrayImage;
typedef struct { uint64_t bytes_reserved, work_charged; } NvmRecordArrayImageCost;
bool nvm_record_array_image_bound(const NvmRecordArrayExecutionPlan *,NvmRecordArrayImageCost *);
NvmArrayEligibilityResult nvm_record_array_image_copy(
    const NvmRecordArrayExecutionPlan *, NvmRecordArrayImage **);
const NvmModule *nvm_record_array_image_borrow(const NvmRecordArrayImage *);
bool nvm_record_array_image_cost(const NvmRecordArrayImage *, NvmRecordArrayImageCost *);
void nvm_record_array_image_free(NvmRecordArrayImage *);
#endif
