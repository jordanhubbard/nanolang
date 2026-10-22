#ifndef NANOVM_UNION_FIELD_RUNTIME_PRIVATE_H
#define NANOVM_UNION_FIELD_RUNTIME_PRIVATE_H
#include "vm.h"
#include "../nanoisa/union_field_snapshot_private.h"
#ifdef NANO_UNION_FIELD_PRIVATE_PREPARATION
/* No VM/plan/image getter. Stable module and
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
typedef enum { VM_UNION_FIELD_RETURNED,VM_UNION_FIELD_HALTED } VmUnionFieldRunOutcome;
typedef struct { VmUnionFieldRunOutcome outcome;NanoValue result; } VmUnionFieldRunResult;
/* Closed synchronous execution on the creating thread. The opaque owner is
 * exclusively stable; execute/destroy must not race or reenter. Output must be
 * disjoint from all owner/image/heap storage. Error preserves output bytes;
 * guest mutations of retained globals are not rolled back. Only RETURNED INT
 * or HALTED VOID is published, with no heap edge transferred to the caller. */
VmResult vm_union_field_private_execute(VmUnionFieldPrivate *,VmUnionFieldRunResult *);
#endif
#endif
