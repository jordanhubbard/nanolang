#ifndef NANOVM_RECORD_MAP_HEAP_PRIVATE_H
#define NANOVM_RECORD_MAP_HEAP_PRIVATE_H
#include "heap.h"
#include "../nanoisa/record_map_admission.h"
/* Private heap qualification only; never an execution/admission token. */
#if defined(NANO_RECORD_MAP_PRIVATE_HEAP) && !defined(NANO_RECORD_ARRAY_PRIVATE_RUNTIME)
#error "private map teardown requires the existing private DAG heap layout"
#endif
#define VM_RM_OBSERVATION_WORK_MAX UINT64_C(33554432)
typedef enum {
    VM_RM_HEAP_OK, VM_RM_HEAP_INVALID, VM_RM_HEAP_MEMORY,
    VM_RM_HEAP_RETAIN_OVERFLOW, VM_RM_HEAP_LIMIT
} VmRmHeapStatus;
typedef struct { uint64_t used,limit;bool limited; } VmRmObservationWork;
/* Fast adapters require closed constructor-owned storage, immutable counted
 * STRINGs and no callbacks/concurrent mutation. Header checks are not a proof
 * for arbitrary forged tables. Inputs are borrowed; output/work memory must
 * not overlap the input graph. No guest fuel budget is introduced. */
#ifdef NANO_RECORD_MAP_PRIVATE_HEAP
VmRmHeapStatus vm_rm_heap_new(VmHeap *,VmHashMap **);
VmRmHeapStatus vm_rm_heap_set(VmHeap *,VmHashMap *,NanoValue,NanoValue);
VmRmHeapStatus vm_rm_heap_get(const VmHashMap *,NanoValue,NanoValue *);
VmRmHeapStatus vm_rm_heap_has(const VmHashMap *,NanoValue,bool *);
VmRmHeapStatus vm_rm_heap_len(const VmHashMap *,uint32_t *);
VmRmHeapStatus vm_rm_heap_validate(const VmHashMap *,VmRmObservationWork *);
#endif
static inline bool vm_rm_private_dag(const VmHeap *heap) {
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    return heap->private_record_dag;
#else
    (void)heap;return false;
#endif
}
/* Ordinary adapters require fresh full topology/collector validation within
 * an exclusive closed invocation, or a map created by this constructor. No
 * callback/concurrent mutation is allowed. Every map/key must belong to the
 * supplied managed heap; no heap identity registry is inferred. Buffered PURPLE containers retain
 * their collector membership. Inputs are borrowed; outputs must be disjoint
 * from all input graph/storage. These are not executable trust APIs. */
VmRmHeapStatus vm_rm_ordinary_new(VmHeap *,VmHashMap **);
VmRmHeapStatus vm_rm_ordinary_set(VmHeap *,VmHashMap *,NanoValue,NanoValue);
VmRmHeapStatus vm_rm_ordinary_get(const VmHeap *,const VmHashMap *,NanoValue,NanoValue *);
VmRmHeapStatus vm_rm_ordinary_has(const VmHeap *,const VmHashMap *,NanoValue,bool *);
VmRmHeapStatus vm_rm_ordinary_len(const VmHeap *,const VmHashMap *,uint32_t *);
/* Physical counted-byte topology only; no execution or GC membership authority.
 * The internal module-taking import query checks ordinary collector membership. */
VmRmHeapStatus vm_rm_heap_import_validate(const VmHashMap *,VmRmObservationWork *);
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
/* Full topology validation and byte probe occur in one read-only call. No
 * caller-provided flag can bypass validation; result is only INT or VOID. */
VmRmHeapStatus vm_rm_heap_observe_lookup(const VmHashMap *,const void *,uint32_t,VmRmObservationWork *,NanoValue *);
#endif
#endif
