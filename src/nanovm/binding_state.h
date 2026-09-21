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

/* A local source names state/current locals/slot. A forwarded source has NULL
 * state/locals, slot zero, and an independently rooted borrowed value: ordinary
 * for mode0, an internal one-slot tuple cell for mode1. Nothing is consumed. */
typedef struct {
    VmBindingState *state;
    NanoValue *locals;
    NanoValue value;
    uint16_t slot;
    uint8_t mode;
} VmBindingSource;

/* I require resolved, rooted sources and validated target identity/modes. The
 * caller reserves the result stack and excludes stack growth/callbacks while I
 * run. Repeated state pointers require the same current locals pointer. Inputs
 * and out are distinct, readable/writable for their extents. Failure preserves
 * bindings and *out; success returns one owned closure. Limit covers accounted
 * live heap bytes plus scratch/cells/closure. Work charges source and duplicate
 * scan visits; diagnostic allocation/retain/release counters may advance on
 * refusal. This helper does not validate a wire site or admit VM execution. */
VmBindingResult vm_binding_closure(VmHeap *heap, uint32_t module_id,
    uint32_t function, const uint8_t *target_modes,
    const VmBindingSource *sources, uint16_t count, size_t limit,
    size_t work_limit, VmClosure **out);

/* I borrow the exact immutable modes of a previously validated target.
 * Shape alone does not establish declaration authority. NULL closure denotes
 * a raw function and is accepted only when count is zero. */
VmBindingResult vm_binding_environment(const VmClosure *closure,
    uint32_t module_id, uint32_t function, const uint8_t *modes, uint16_t count);
/* Entry validation precedes access. The rooted closure and selected cell stay
 * live; operands/outputs are distinct from environment and cell storage. Read
 * publishes a retained ordinary value; assignment moves an owned operand only
 * on success, as for local bindings. Neither operation exposes an internal cell. */
VmBindingResult vm_binding_upvalue_read(VmHeap *heap, VmClosure *closure,
    uint32_t module_id, uint32_t function, const uint8_t *modes,
    uint16_t count, uint16_t index, NanoValue *out);
VmBindingResult vm_binding_upvalue_assign(VmHeap *heap, VmClosure *closure,
    uint32_t module_id, uint32_t function, const uint8_t *modes,
    uint16_t count, uint16_t index, NanoValue *incoming);
#endif
