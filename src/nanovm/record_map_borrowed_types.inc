/* Internal copied preparation and call-local view; no VmState storage. */
#include "record_map_public_prepare_private.h"
struct VmRecordMapPublicPreparation {
    NvmRecordMapBoundaryPlan *boundary;
    const NvmRecordMapExecutionPlan *plan;
    const NvmModule *module;
    const NanoValue *original_stack;
    NvmRecordMapExecutionInstruction *instructions;
    VmRecordMapRuntimeField *fields;
    NvmRecordMapExecutionCounts counts;
    NvmRecordArrayExecutionFunction functions[256];
    VmRecordMapRuntimeLayout layouts[256];
    uint32_t record_layouts[256];
    uint32_t layout_count,field_count;
    uint32_t original_stack_capacity,rounded_stack_capacity;
    uint64_t bytes,work;
};
struct VmRecordMapBorrowedInvocation {
    const VmRecordMapPublicPreparation *preparation;
    VmState *vm;
    const NvmModule *module;
    uint32_t active_root,root_base;
    bool active;
};
_Static_assert(sizeof(VmRecordMapBorrowedInvocation)==40,"borrowed view prepaid ledger");
