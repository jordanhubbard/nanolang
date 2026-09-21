#include "binding_state.h"
#include <assert.h>
#include <stdlib.h>

VmBindingResult vm_binding_state_new(VmHeap *heap, const uint8_t *modes,
    uint16_t count, uint16_t arity, size_t limit, VmBindingState **out) {
    if (!heap || !out || arity > count || (count && !modes) ||
        heap->stats.freed > heap->stats.allocated) return VM_BINDING_INVALID;
    for (uint16_t i = 0; i < count; ++i)
        if (modes[i] > 1) return VM_BINDING_INVALID;
    size_t slot_count = count;
    if (slot_count > (SIZE_MAX - sizeof(VmBindingState)) / sizeof(VmBindingSlot))
        return VM_BINDING_LIMIT;
    size_t bytes = sizeof(VmBindingState) + (size_t)count * sizeof(VmBindingSlot);
    size_t live = heap->stats.allocated - heap->stats.freed;
    if (live > limit || bytes > limit - live ||
        bytes > SIZE_MAX - heap->stats.allocated ||
        heap->stats.allocation_calls == UINT64_MAX) return VM_BINDING_LIMIT;
    VmBindingState *state = calloc(1, bytes);
    if (!state) return VM_BINDING_MEMORY;
    state->heap = heap; state->bytes = bytes; state->count = count;
    for (uint16_t i = 0; i < count; ++i) {
        state->slots[i].initialized = i < arity;
        state->slots[i].shared = modes[i] != 0;
    }
    heap->stats.allocated += bytes;
    heap->stats.allocation_calls++;
    *out = state;
    return VM_BINDING_OK;
}

static VmBindingSlot *binding_slot(VmBindingState *state, NanoValue *locals,
    uint16_t index) {
    if (!state || !state->heap || !locals || index >= state->count) return NULL;
    VmBindingSlot *slot = &state->slots[index];
    if (slot->cell && (!slot->initialized || !slot->shared ||
        slot->cell->header.obj_type != TAG_TUPLE || slot->cell->count != 1 ||
        !slot->cell->header.ref_count || locals[index].tag != TAG_VOID)) return NULL;
    return slot;
}

static VmBindingResult binding_retain(VmHeap *heap, NanoValue value) {
    if (!val_is_heap_obj(value) || !value.as.obj) return VM_BINDING_OK;
    VmHeapHeader *header = value.as.obj;
    if (!header->ref_count) return VM_BINDING_INVALID;
    if (header->ref_count == UINT32_MAX || heap->stats.retain_calls == UINT64_MAX)
        return VM_BINDING_LIMIT;
    vm_retain(heap, value);
    return VM_BINDING_OK;
}

VmBindingResult vm_binding_read(VmBindingState *state, NanoValue *locals,
    uint16_t index, NanoValue *out) {
    VmBindingSlot *slot = binding_slot(state, locals, index);
    if (!slot || !slot->initialized || !out) return VM_BINDING_INVALID;
    NanoValue value = slot->cell ? slot->cell->elements[0] : locals[index];
    VmBindingResult result = binding_retain(state->heap, value);
    if (result != VM_BINDING_OK) return result;
    *out = value;
    return VM_BINDING_OK;
}

VmBindingResult vm_binding_initialize(VmBindingState *state, NanoValue *locals,
    uint16_t index, NanoValue *incoming) {
    VmBindingSlot *slot = binding_slot(state, locals, index);
    if (!slot || !incoming) return VM_BINDING_INVALID;
    NanoValue previous = slot->cell ? val_tuple(slot->cell) : locals[index];
    locals[index] = *incoming;
    *incoming = val_void();
    slot->cell = NULL;
    slot->initialized = true;
    vm_release(state->heap, previous);
    return VM_BINDING_OK;
}

VmBindingResult vm_binding_assign(VmBindingState *state, NanoValue *locals,
    uint16_t index, NanoValue *incoming) {
    VmBindingSlot *slot = binding_slot(state, locals, index);
    if (!slot || !slot->initialized || !slot->shared || !incoming)
        return VM_BINDING_INVALID;
    NanoValue *destination = slot->cell ? &slot->cell->elements[0] : &locals[index];
    NanoValue previous = *destination;
    *destination = *incoming;
    *incoming = val_void();
    vm_release(state->heap, previous);
    return VM_BINDING_OK;
}

VmBindingResult vm_binding_clear(VmBindingState *state, NanoValue *locals,
    uint16_t index) {
    VmBindingSlot *slot = binding_slot(state, locals, index);
    if (!slot) return VM_BINDING_INVALID;
    NanoValue previous = slot->cell ? val_tuple(slot->cell) : locals[index];
    slot->cell = NULL;
    slot->initialized = false;
    locals[index] = val_void();
    vm_release(state->heap, previous);
    return VM_BINDING_OK;
}

void vm_binding_state_destroy(VmBindingState *state, NanoValue *locals) {
    if (!state) return;
    assert(locals || !state->count);
    for (uint16_t i = 0; i < state->count; ++i) {
        VmBindingResult result = vm_binding_clear(state, locals, i);
        assert(result == VM_BINDING_OK);
        (void)result;
    }
    state->heap->stats.freed += state->bytes;
    free(state);
}
