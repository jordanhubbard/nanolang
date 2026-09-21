/* I own binding metadata and cell edges, never addresses into a movable stack. */
#ifndef NANOVM_BINDING_STATE_H
#define NANOVM_BINDING_STATE_H
#include "heap.h"

typedef struct {
    VmTuple *cell; /* Internal owned one-slot tuple, not a source tuple value. */
    bool initialized, shared;
} VmBindingSlot;

typedef struct {
    VmHeap *heap;
    size_t bytes;
    uint16_t count;
    VmBindingSlot slots[];
} VmBindingState;

typedef enum {
    VM_BINDING_OK, VM_BINDING_INVALID, VM_BINDING_LIMIT, VM_BINDING_MEMORY
} VmBindingResult;

/* I copy modes 0/1; parameters are the first arity slots. locals at entry must
 * own the supplied arguments and contain VOID elsewhere. limit bounds existing
 * accounted live heap bytes plus this allocation, not process resident memory.
 * Failure preserves *out. The heap outlives this state and its owned values. */
VmBindingResult vm_binding_state_new(VmHeap *heap, const uint8_t *modes,
    uint16_t count, uint16_t arity, size_t limit, VmBindingState **out);

/* I require the current count-element locals array. Inputs/outputs must use
 * distinct storage outside that array and cell storage. An incoming operand is
 * independently owned; I move it and clear *incoming only on success. A read
 * publishes one newly retained owner only on success. No output may replace an
 * existing owner without the caller first releasing that owner. */
VmBindingResult vm_binding_read(VmBindingState *state, NanoValue *locals,
    uint16_t slot, NanoValue *out);
VmBindingResult vm_binding_initialize(VmBindingState *state, NanoValue *locals,
    uint16_t slot, NanoValue *incoming);
VmBindingResult vm_binding_assign(VmBindingState *state, NanoValue *locals,
    uint16_t slot, NanoValue *incoming);
VmBindingResult vm_binding_clear(VmBindingState *state, NanoValue *locals,
    uint16_t slot);
/* I require a valid created state/current locals. I detach/release every local
 * or cell edge and leave locals VOID before existing frame-stack teardown. */
void vm_binding_state_destroy(VmBindingState *state, NanoValue *locals);
#endif
