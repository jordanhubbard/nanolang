#ifndef NANOVM_RECORD_ARRAY_RUNTIME_PRIVATE_H
#define NANOVM_RECORD_ARRAY_RUNTIME_PRIVATE_H
#include "vm.h"
#include "../nanoisa/managed_record_array_execution.h"
/* My private qualification interface is absent from ordinary VM builds. */
typedef struct VmRecordArrayPrivate VmRecordArrayPrivate;
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
#define VM_RECORD_ARRAY_EXTRA_BYTES (UINT64_C(128)*1024*1024)
#define VM_RECORD_ARRAY_EXTRA_STEPS UINT64_C(33554432)
#define VM_RECORD_ARRAY_ROOT_CAPACITY (UINT32_C(1024)*512+1)
#define VM_RECORD_ARRAY_PATH_MAX 257u
typedef enum { VM_RA_RESULT, VM_RA_GLOBAL } VmRecordArrayRoot;
typedef struct {
    uint64_t epoch, identity, scalar_bits;
    uint32_t length, layout;
    uint8_t tag, element_tag;
} VmRecordArrayObservation;
typedef struct {
    uint64_t preparation_bytes, preparation_steps, epoch;
    uint64_t heap_objects, heap_live_bytes;
    uint32_t active_frames, active_stack, maximum_frames;
    VmResult last_status;
    bool has_result;
} VmRecordArrayPrivateStats;
NvmArrayEligibilityResult vm_record_array_private_create(const NvmModule *,VmRecordArrayPrivate **);
VmResult vm_record_array_private_run(VmRecordArrayPrivate *);
void vm_record_array_private_destroy(VmRecordArrayPrivate *);
/* Observations own no heap edge. Identity is meaningful only in the same
 * instance/epoch; every invocation invalidates prior observation epochs. */
bool vm_record_array_private_observe(const VmRecordArrayPrivate *,VmRecordArrayRoot,
    uint32_t,const uint32_t *,uint16_t,VmRecordArrayObservation *);
bool vm_record_array_private_string(const VmRecordArrayPrivate *,VmRecordArrayRoot,
    uint32_t,const uint32_t *,uint16_t,uint32_t,uint32_t,void *);
bool vm_record_array_private_stats(const VmRecordArrayPrivate *,VmRecordArrayPrivateStats *);
#endif
#endif
