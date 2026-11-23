/* I account for carrier ownership independently of public runtime layouts. */
#include "carrier_heap.h"
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    void *object;
    NlCarrierTrace trace;
    NlCarrierDestroy destroy;
    void *context;
    bool marked;
} CarrierEntry;

typedef struct {
    CarrierEntry *entry;
    bool deleted;
} CarrierSlot;

struct NlCarrierHeap {
    CarrierSlot *slots;
    size_t capacity;
    size_t used;
    bool collecting;
    NlCarrierHeapStats stats;
};

static size_t carrier_hash(const void *object) {
    uintptr_t value = (uintptr_t)object >> 3;
    value ^= value >> 16;
    value *= (uintptr_t)0x9e3779b1u;
    value ^= value >> 16;
    return (size_t)value;
}

static size_t carrier_slot(const NlCarrierHeap *heap, const void *object,
                           bool insertion) {
    size_t slot = carrier_hash(object) & (heap->capacity - 1);
    size_t deleted = SIZE_MAX;
    for (size_t probe = 0; probe < heap->capacity; ++probe) {
        CarrierSlot *item = &heap->slots[slot];
        if (item->entry) {
            if (item->entry->object == object) return slot;
        } else if (item->deleted) {
            if (deleted == SIZE_MAX) deleted = slot;
        } else {
            return insertion ? (deleted == SIZE_MAX ? slot : deleted) : SIZE_MAX;
        }
        slot = (slot + 1) & (heap->capacity - 1);
    }
    return insertion ? deleted : SIZE_MAX;
}

static bool carrier_resize(NlCarrierHeap *heap, size_t capacity) {
    if (capacity > SIZE_MAX / sizeof(CarrierSlot)) return false;
    CarrierSlot *slots = calloc(capacity, sizeof *slots);
    if (!slots) return false;
    NlCarrierHeap next = *heap;
    next.slots = slots;
    next.capacity = capacity;
    for (size_t i = 0; i < heap->capacity; ++i) {
        CarrierEntry *entry = heap->slots[i].entry;
        if (entry) next.slots[carrier_slot(&next, entry->object, true)].entry = entry;
    }
    free(heap->slots);
    heap->slots = slots;
    heap->capacity = capacity;
    heap->used = heap->stats.live;
    return true;
}

NlCarrierHeap *nl_carrier_heap_new(void) {
    NlCarrierHeap *heap = calloc(1, sizeof *heap);
    if (heap && !carrier_resize(heap, 32)) {
        free(heap);
        return NULL;
    }
    return heap;
}

bool nl_carrier_heap_contains(const NlCarrierHeap *heap, const void *object) {
    return heap && object && carrier_slot(heap, object, false) != SIZE_MAX;
}

bool nl_carrier_heap_track(NlCarrierHeap *heap, void *object,
                           NlCarrierTrace trace, NlCarrierDestroy destroy,
                           void *context) {
    if (!heap || heap->collecting || !object || !destroy ||
        heap->stats.registered == SIZE_MAX ||
        nl_carrier_heap_contains(heap, object)) return false;
    if (heap->used + 1 > heap->capacity * 3 / 4) {
        size_t capacity = heap->capacity;
        if (heap->stats.live + 1 > capacity / 2) {
            if (capacity > SIZE_MAX / 2) return false;
            capacity *= 2;
        }
        if (!carrier_resize(heap, capacity)) return false;
    }
    CarrierEntry *entry = malloc(sizeof *entry);
    if (!entry) return false;
    *entry = (CarrierEntry){object, trace, destroy, context, false};
    size_t slot = carrier_slot(heap, object, true);
    if (!heap->slots[slot].deleted) heap->used++;
    heap->slots[slot].entry = entry;
    heap->slots[slot].deleted = false;
    heap->stats.registered++;
    heap->stats.live++;
    return true;
}

bool nl_carrier_heap_forget(NlCarrierHeap *heap, const void *object) {
    if (!heap || heap->collecting || !object) return false;
    size_t slot = carrier_slot(heap, object, false);
    if (slot == SIZE_MAX) return false;
    free(heap->slots[slot].entry);
    heap->slots[slot].entry = NULL;
    heap->slots[slot].deleted = true;
    heap->stats.transferred++;
    heap->stats.live--;
    return true;
}

typedef struct {
    NlCarrierHeap *heap;
    CarrierEntry **queue;
    size_t count;
} CarrierMark;

static bool carrier_mark(const void *object, void *visitor) {
    CarrierMark *mark = visitor;
    if (!object) return false;
    size_t slot = carrier_slot(mark->heap, object, false);
    if (slot == SIZE_MAX) return false;
    CarrierEntry *entry = mark->heap->slots[slot].entry;
    if (!entry->marked) {
        entry->marked = true;
        mark->queue[mark->count++] = entry;
    }
    return true;
}

/* I remove ownership before invoking its nonrecursive destructor. */
static void carrier_destroy_slot(NlCarrierHeap *heap, size_t slot) {
    CarrierEntry *entry = heap->slots[slot].entry;
    heap->slots[slot].entry = NULL;
    heap->slots[slot].deleted = true;
    heap->stats.live--;
    heap->stats.destroyed++;
    entry->destroy(entry->object, entry->context);
    free(entry);
}

bool nl_carrier_heap_collect(NlCarrierHeap *heap,
                             const void *const *roots, size_t root_count) {
    if (!heap || heap->collecting || (root_count && !roots)) return false;
    if (heap->stats.live > SIZE_MAX / sizeof(CarrierEntry *)) return false;
    CarrierEntry **queue = heap->stats.live ?
        malloc(heap->stats.live * sizeof *queue) : NULL;
    if (heap->stats.live && !queue) return false;
    heap->collecting = true;
    for (size_t i = 0; i < heap->capacity; ++i)
        if (heap->slots[i].entry) heap->slots[i].entry->marked = false;
    CarrierMark mark = {heap, queue, 0};
    for (size_t i = 0; i < root_count; ++i) carrier_mark(roots[i], &mark);
    for (size_t i = 0; i < mark.count; ++i) {
        CarrierEntry *entry = queue[i];
        if (entry->trace &&
            !entry->trace(entry->object, carrier_mark, &mark, entry->context)) {
            heap->collecting = false;
            free(queue);
            return false;
        }
    }
    for (size_t i = 0; i < heap->capacity; ++i) {
        CarrierEntry *entry = heap->slots[i].entry;
        if (entry && !entry->marked) carrier_destroy_slot(heap, i);
    }
    heap->collecting = false;
    free(queue);
    return true;
}

bool nl_carrier_heap_clear(NlCarrierHeap *heap) {
    if (!heap || heap->collecting) return false;
    heap->collecting = true;
    for (size_t i = 0; i < heap->capacity; ++i)
        if (heap->slots[i].entry) carrier_destroy_slot(heap, i);
    heap->collecting = false;
    return true;
}

bool nl_carrier_heap_free(NlCarrierHeap *heap) {
    if (!heap) return true;
    if (!nl_carrier_heap_clear(heap)) return false;
    free(heap->slots);
    free(heap);
    return true;
}

NlCarrierHeapStats nl_carrier_heap_stats(const NlCarrierHeap *heap) {
    return heap ? heap->stats : (NlCarrierHeapStats){0};
}
