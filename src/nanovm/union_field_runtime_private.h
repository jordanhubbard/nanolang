#ifndef NANOVM_UNION_FIELD_RUNTIME_PRIVATE_H
#define NANOVM_UNION_FIELD_RUNTIME_PRIVATE_H
#include "vm.h"
#include "../nanoisa/union_field_snapshot_private.h"
#ifdef NANO_UNION_FIELD_PRIVATE_PREPARATION
/* Preparation only. No VM/plan/image getter and no run entry. Stable module and
 * output disjointness from all input/owned storage are C API preconditions.
 * Failure preserves output. Destroy requires the creating thread. */
typedef struct VmUnionFieldPrivate VmUnionFieldPrivate;
typedef struct {
    uint64_t runtime_bytes,runtime_work,image_overlap_bytes,image_overlap_work;
    uint64_t source_bytes_reserved,source_work_reserved,owner_size,instruction_size;
    uint32_t functions,instructions,globals,stack_capacity;
} VmUnionFieldPreparation;
NvmArrayEligibilityResult vm_union_field_private_prepare(const NvmModule *,VmUnionFieldPrivate **);
void vm_union_field_private_destroy(VmUnionFieldPrivate *);
bool vm_union_field_private_preparation(const VmUnionFieldPrivate *,VmUnionFieldPreparation *);
#endif
#endif
