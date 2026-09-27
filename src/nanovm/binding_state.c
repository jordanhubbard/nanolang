#include "binding_state.h"
#include <assert.h>
#include <stdlib.h>

VmBindingResult vm_binding_state_new(VmHeap *heap, const uint8_t *modes,
    uint16_t count, uint16_t arity, size_t limit, VmBindingState **out) {
    return vm_binding_state_new_range(heap, modes, count, 0, arity, limit, out);
}

VmBindingResult vm_binding_state_new_range(VmHeap *heap, const uint8_t *modes,
    uint16_t count, uint16_t start, uint16_t initialized_count,
    size_t limit, VmBindingState **out) {
    if (!heap || !out || start > count || initialized_count > count - start ||
        (count && !modes) ||
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
        state->slots[i].initialized = i >= start && i - start < initialized_count;
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

static bool binding_environment_identity(const VmClosure *closure,
    uint32_t module_id, uint32_t function, const uint8_t *modes, uint16_t count) {
    if (!module_id || (count && !modes)) return false;
    if (!closure) return count == 0;
    return closure->header.obj_type == TAG_CLOSURE && closure->header.ref_count &&
        closure->callable_module == module_id && closure->fn_idx == function &&
        closure->capture_count == count;
}

static bool binding_capture_shape(NanoValue capture, uint8_t mode) {
    if (mode == 0) return true;
    return mode == 1 && capture.tag == TAG_TUPLE && capture.as.tuple &&
        capture.as.tuple->header.obj_type == TAG_TUPLE &&
        capture.as.tuple->header.ref_count && capture.as.tuple->count == 1;
}

VmBindingResult vm_binding_environment(const VmClosure *closure,
    uint32_t module_id, uint32_t function, const uint8_t *modes, uint16_t count) {
    if (!binding_environment_identity(closure, module_id, function, modes, count))
        return VM_BINDING_INVALID;
    for (uint16_t i = 0; i < count; ++i)
        if (!binding_capture_shape(closure->captures[i], modes[i]))
            return VM_BINDING_INVALID;
    return VM_BINDING_OK;
}

static NanoValue *binding_upvalue(VmClosure *closure, uint32_t module_id,
    uint32_t function, const uint8_t *modes, uint16_t count, uint16_t index) {
    if (!closure || index >= count ||
        !binding_environment_identity(closure, module_id, function, modes, count) ||
        !binding_capture_shape(closure->captures[index], modes[index])) return NULL;
    return modes[index] ? &closure->captures[index].as.tuple->elements[0] :
        &closure->captures[index];
}

VmBindingResult vm_binding_upvalue_read(VmHeap *heap, VmClosure *closure,
    uint32_t module_id, uint32_t function, const uint8_t *modes,
    uint16_t count, uint16_t index, NanoValue *out) {
    if (!heap || !out) return VM_BINDING_INVALID;
    NanoValue *value = binding_upvalue(closure, module_id, function, modes, count, index);
    if (!value) return VM_BINDING_INVALID;
    VmBindingResult result = binding_retain(heap, *value);
    if (result != VM_BINDING_OK) return result;
    *out = *value;
    return VM_BINDING_OK;
}

VmBindingResult vm_binding_upvalue_assign(VmHeap *heap, VmClosure *closure,
    uint32_t module_id, uint32_t function, const uint8_t *modes,
    uint16_t count, uint16_t index, NanoValue *incoming) {
    if (!heap || !incoming) return VM_BINDING_INVALID;
    NanoValue *value = binding_upvalue(closure, module_id, function, modes, count, index);
    if (!value || modes[index] != 1) return VM_BINDING_INVALID;
    NanoValue previous = *value;
    *value = *incoming;
    *incoming = val_void();
    vm_release(heap, previous);
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

typedef struct {
    NanoValue value;
    VmTuple *new_cell; /* Only the representative owns this staged edge. */
    uint16_t representative;
    bool fresh;
} BindingCaptureStage;

static bool binding_room(VmHeap *heap, size_t bytes, size_t objects,
    uint64_t calls, size_t limit) {
    if (heap->stats.freed > heap->stats.allocated) return false;
    size_t live = heap->stats.allocated - heap->stats.freed;
    return live <= limit && bytes <= limit - live &&
        bytes <= SIZE_MAX - heap->stats.allocated &&
        objects <= SIZE_MAX - heap->stats.num_objects &&
        calls <= UINT64_MAX - heap->stats.allocation_calls;
}

VmBindingResult vm_binding_closure(VmHeap *heap, uint32_t module_id,
    uint32_t function, const uint8_t *target_modes,
    const VmBindingSource *sources, uint16_t count, size_t limit,
    size_t work_limit, VmClosure **out) {
    if (!heap || !module_id || !out || (count && (!sources || !target_modes)) ||
        heap->stats.freed > heap->stats.allocated) return VM_BINDING_INVALID;
    if ((uint64_t)count > UINT64_MAX - heap->stats.release_calls)
        return VM_BINDING_LIMIT;
#ifdef NANO_RECORD_ARRAY_PRIVATE_RUNTIME
    if (heap->private_record_dag) return VM_BINDING_INVALID;
#endif
    size_t n = count;
    if (n > SIZE_MAX / sizeof(BindingCaptureStage)) return VM_BINDING_LIMIT;
    size_t scratch_bytes = n * sizeof(BindingCaptureStage);
    if (!binding_room(heap, scratch_bytes, 0, count ? 1 : 0, limit))
        return VM_BINDING_LIMIT;
    BindingCaptureStage *stages = count ? calloc(n, sizeof(*stages)) : NULL;
    if (count && !stages) return VM_BINDING_MEMORY;
    heap->stats.allocated += scratch_bytes;
    heap->stats.allocation_calls += count ? 1 : 0;
    VmBindingResult result = VM_BINDING_INVALID;
    VmClosure *closure = NULL;
    size_t unique = 0;
    for (uint16_t i = 0; i < count; ++i) {
        if (!work_limit) { result = VM_BINDING_LIMIT; goto fail; }
        --work_limit;
        const VmBindingSource *source = &sources[i];
        BindingCaptureStage *stage = &stages[i];
        stage->representative = i;
        if (source->mode > 1 || target_modes[i] != source->mode) goto fail;
        if (source->state) {
            VmBindingSlot *slot = binding_slot(source->state, source->locals, source->slot);
            if (!slot || source->state->heap != heap || !slot->initialized ||
                slot->shared != (source->mode != 0)) goto fail;
            for (uint16_t j = 0; j < i; ++j) {
                if (!work_limit) { result = VM_BINDING_LIMIT; goto fail; }
                --work_limit;
                if (sources[j].state != source->state) continue;
                if (sources[j].locals != source->locals) goto fail;
                if (source->mode && !slot->cell && sources[j].slot == source->slot)
                    stage->representative = stages[j].representative;
            }
            if (source->mode) {
                if (slot->cell) stage->value = val_tuple(slot->cell);
                else {
                    stage->fresh = true;
                    if (stage->representative == i) ++unique;
                }
            } else stage->value = source->locals[source->slot];
        } else {
            if (source->locals || source->slot) goto fail;
            stage->value = source->value;
            if (source->mode && (source->value.tag != TAG_TUPLE ||
                !source->value.as.tuple ||
                source->value.as.tuple->header.obj_type != TAG_TUPLE ||
                !source->value.as.tuple->header.ref_count ||
                source->value.as.tuple->count != 1)) goto fail;
        }
    }
    if (n > (SIZE_MAX - sizeof(VmClosure)) / sizeof(NanoValue)) {
        result = VM_BINDING_LIMIT; goto fail;
    }
    size_t bytes = sizeof(VmClosure) + n * sizeof(NanoValue);
    size_t cell_bytes = sizeof(VmTuple) + sizeof(NanoValue);
    if (unique > (SIZE_MAX - bytes) / cell_bytes ||
        !binding_room(heap, bytes + unique * cell_bytes, unique + 1,
                      (uint64_t)unique + 1, limit)) {
        result = VM_BINDING_LIMIT; goto fail;
    }
    closure = vm_closure_new(heap, function, count);
    if (!closure) { result = VM_BINDING_MEMORY; goto fail; }
    closure->callable_module = module_id;
    for (uint16_t i = 0; i < count; ++i) {
        BindingCaptureStage *stage = &stages[i];
        if (stage->fresh && stage->representative == i) {
            stage->new_cell = vm_tuple_new(heap, 1);
            if (!stage->new_cell) { result = VM_BINDING_MEMORY; goto fail; }
        }
    }
    for (uint16_t i = 0; i < count; ++i) {
        NanoValue value = stages[i].fresh ?
            val_tuple(stages[stages[i].representative].new_cell) : stages[i].value;
        result = binding_retain(heap, value);
        if (result != VM_BINDING_OK) goto fail;
        closure->captures[i] = value;
    }
    /* I have acquired every edge. Publication cannot allocate or release. */
    for (uint16_t i = 0; i < count; ++i) {
        VmTuple *cell = stages[i].new_cell;
        if (!cell) continue;
        const VmBindingSource *source = &sources[i];
        cell->elements[0] = source->locals[source->slot];
        source->locals[source->slot] = val_void();
        source->state->slots[source->slot].cell = cell;
        stages[i].new_cell = NULL; /* The binding now owns the staged edge. */
    }
    heap->stats.freed += scratch_bytes;
    free(stages);
    *out = closure;
    return VM_BINDING_OK;
fail:
    if (closure) {
        /* Every captured edge was acquired here. Its prior owner is still
         * rooted; undoing the acquire cannot create a new unreachable cycle.
         * Ordinary release would add suspect-buffer allocations and defer
         * private cell reclamation beyond this transaction's budget. */
        for (uint16_t i = 0; i < count; ++i) {
            NanoValue value = closure->captures[i];
            if (!val_is_heap_obj(value) || !value.as.obj) continue;
            VmHeapHeader *header = value.as.obj;
            assert(header->ref_count > 1);
            --header->ref_count;
            ++heap->stats.release_calls;
        }
        heap->stats.freed += sizeof(VmClosure) + n * sizeof(NanoValue);
        --heap->stats.num_objects;
        free(closure);
    }
    for (uint16_t i = 0; i < count; ++i) {
        VmTuple *cell = stages[i].new_cell;
        if (!cell) continue;
        assert(cell->header.ref_count == 1 && !cell->header.buffered &&
               cell->elements[0].tag == TAG_VOID);
        heap->stats.freed += sizeof(VmTuple) + sizeof(NanoValue);
        --heap->stats.num_objects;
        free(cell);
    }
    heap->stats.freed += scratch_bytes;
    free(stages);
    return result;
}
