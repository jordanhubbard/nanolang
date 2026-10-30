#ifndef NANOISA_UNION_FIELD_VM_IMAGE_PRIVATE_H
#define NANOISA_UNION_FIELD_VM_IMAGE_PRIVATE_H
#include "union_field_snapshot_private.h"
/* Owning VM preparation and macro-private native target TUs only. Positive reservations never authorize execution.
 * Module borrow is immutable, exclusively stable and image-lifetime bound.
 * No caller-supplied image/plan is accepted by a VM execution entry. */
NvmArrayEligibilityResult nvm_union_field_image_copy_reserved(
    const NvmUnionFieldExecutionPlan *,uint64_t,uint64_t,NvmUnionFieldImage **);
const NvmModule *nvm_union_field_image_module_internal(const NvmUnionFieldImage *);
#endif
