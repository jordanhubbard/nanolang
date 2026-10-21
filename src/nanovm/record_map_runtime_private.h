#ifndef NANOVM_RECORD_MAP_RUNTIME_PRIVATE_H
#define NANOVM_RECORD_MAP_RUNTIME_PRIVATE_H
#include "vm.h"
#include "../nanoisa/record_map_snapshot_private.h"
/* Private preparation, with a separately gated private invocation API.
 * Ordinary public execution admission is never supplied by this header. */
typedef struct VmRecordMapPrivate VmRecordMapPrivate;
typedef struct VmRecordMapInvocation VmRecordMapInvocation;
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
#if !defined(NANO_RECORD_MAP_PRIVATE_PREPARATION) || !defined(NANO_RECORD_MAP_PRIVATE_HEAP) || !defined(NANO_RECORD_ARRAY_PRIVATE_RUNTIME)
#error "private map invocation requires consistent preparation, heap and DAG macros"
#endif
typedef struct {
    uint64_t epoch,heap_objects,heap_live_bytes;
    uint32_t active_frames,active_stack,maximum_frames;
    VmResult last_status;
    bool has_result;
} VmRecordMapPrivateStats;
VmResult vm_record_map_private_run(VmRecordMapPrivate *);
bool vm_record_map_private_stats(const VmRecordMapPrivate *,VmRecordMapPrivateStats *);
#endif
#ifdef NANO_RECORD_MAP_PRIVATE_PREPARATION
typedef struct {
    uint64_t runtime_bytes,runtime_work,image_overlap_bytes,image_overlap_work;
    uint64_t owner_size,field_size,layout_size;
    uint32_t functions,instructions,fields;
} VmRecordMapPreparation;
NvmArrayEligibilityResult vm_record_map_private_create(const NvmRecordMapExecutionPlan *,VmRecordMapPrivate **);
void vm_record_map_private_destroy(VmRecordMapPrivate *);
bool vm_record_map_private_preparation(const VmRecordMapPrivate *,VmRecordMapPreparation *);
bool vm_record_map_private_instruction(const VmRecordMapPrivate *,uint32_t,NvmRecordMapExecutionInstruction *);
#endif
#endif
