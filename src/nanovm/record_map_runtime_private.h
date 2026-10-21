#ifndef NANOVM_RECORD_MAP_RUNTIME_PRIVATE_H
#define NANOVM_RECORD_MAP_RUNTIME_PRIVATE_H
#include "vm.h"
#include "../nanoisa/record_map_snapshot_private.h"
/* Preparation/query only. No run API or executable admission. */
typedef struct VmRecordMapPrivate VmRecordMapPrivate;
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
