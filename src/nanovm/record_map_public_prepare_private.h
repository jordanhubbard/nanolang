#ifndef NANOVM_RECORD_MAP_PUBLIC_PREPARE_PRIVATE_H
#define NANOVM_RECORD_MAP_PUBLIC_PREPARE_PRIVATE_H
#include "record_map_import_private.h"
#include "../nanoisa/managed_record_map_boundary.h"
/* Const preparation query only. Managed live storage/exclusive stability are
 * preconditions. Output storage must be disjoint from ALL module, VM, heap,
 * decoded/dispatch cache, stack, globals, constants and other owned storage.
 * Bounded checks cover accessible root ranges; this is not arbitrary allocation
 * discovery. Failure preserves output and VM state. Result function/PC fields
 * are diagnostics, not reservation counters; exact final costs belong to the
 * opaque preparation owner and are measured only by internal whitebox tests.
 * No borrowed owner can authorize execution. */
typedef struct VmRecordMapPublicPreparation VmRecordMapPublicPreparation;
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
#define VM_RMP_API
#else
#define VM_RMP_API static
#endif
VM_RMP_API NvmArrayEligibilityResult vm_record_map_public_prepare(const VmState *,VmRecordMapPublicPreparation **);
VM_RMP_API void vm_record_map_public_preparation_free(VmRecordMapPublicPreparation *);
#endif
