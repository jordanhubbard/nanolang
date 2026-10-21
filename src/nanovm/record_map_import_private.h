#ifndef NANOVM_RECORD_MAP_IMPORT_PRIVATE_H
#define NANOVM_RECORD_MAP_IMPORT_PRIVATE_H
#include "vm.h"
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
/* Managed live storage and exclusive owner-thread access are preconditions.
 * This copied observation is not execution authority. Output must be disjoint
 * from module/state/heap storage. Failure leaves output and VmState unchanged. */
typedef enum {
    VM_RM_IMPORT_OK, VM_RM_IMPORT_INVALID, VM_RM_IMPORT_UNRESOLVED,
    VM_RM_IMPORT_MEMORY, VM_RM_IMPORT_LIMIT
} VmRmImportStatus;
typedef struct {
    uint64_t preparation_bytes,preparation_work;
    uint64_t graph_objects,collector_objects,traversed_edges;
} VmRmImportReport;
VmRmImportStatus vm_record_map_import_check(const NvmModule *,const VmState *,VmRmImportReport *);
#endif
#endif
