/* I adapt actual interpreter storage without changing its public layouts. */
#include "carrier_values.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const void *pointer;
    unsigned kind;
} Seen;
typedef struct {
    struct NlCarrierValues *owner;
    NlCarrierRef ref;
    NlCarrierRef *edges;
    size_t edge_count;
} CarrierValue;
struct NlCarrierValues {
    NlCarrierHeap *engine;
    NlCarrierRef *pending;
    size_t pending_count, pending_capacity;
    Seen *seen;
    size_t seen_count, seen_capacity;
    const void **keys;
    size_t key_count, key_capacity;
    NlCarrierVisit mark;
    void *visitor;
    bool busy, transfer;
};

NlCarrierRef nl_carrier_value(Value value) {
    return (NlCarrierRef){value, false};
}
NlCarrierRef nl_carrier_record_list(DynArray *list) {
    return (NlCarrierRef){create_int((long long)(intptr_t)list), true};
}

static void *value_key(NlCarrierRef ref) {
    if (ref.record_list)
        return ref.value.type == VAL_INT ? (void *)(intptr_t)ref.value.as.int_val : NULL;
    switch (ref.value.type) {
        case VAL_STRUCT: return ref.value.as.struct_val;
        case VAL_UNION: return ref.value.as.union_val;
        case VAL_TUPLE: return ref.value.as.tuple_val;
        case VAL_ARRAY: return ref.value.as.array_val;
        default: return NULL;
    }
}
static unsigned value_kind(NlCarrierRef ref) {
    return ref.record_list ? 256 : (unsigned)ref.value.type;
}
static bool valid_slots(int count, const void *slots) {
    return count >= 0 && (count == 0 || slots);
}
static bool valid_value(NlCarrierRef ref) {
    if (ref.record_list && ref.value.type != VAL_INT) return false;
    void *key = value_key(ref);
    if (!key) return true;
    if (ref.record_list) {
        DynArray *list = key;
        return list->elem_type == ELEM_INT && list->elem_size == sizeof(int64_t) &&
            list->length >= 0 && list->length <= list->capacity && (!list->length || list->data);
    }
    switch (ref.value.type) {
        case VAL_STRUCT: {
            StructValue *r = key; return valid_slots(r->field_count, r->field_values);
        }
        case VAL_UNION: {
            UnionValue *u = key; return valid_slots(u->field_count, u->field_values);
        }
        case VAL_TUPLE: {
            TupleValue *t = key; return valid_slots(t->element_count, t->elements);
        }
        case VAL_ARRAY: {
            Array *a = key; return valid_slots(a->length, a->data) && a->length <= a->capacity;
        }
        default: return false;
    }
}
static void *grow(void *data, size_t *capacity, size_t count, size_t width) {
    if (count <= *capacity) return data;
    size_t next = *capacity ? *capacity : 16;
    while (next < count) {
        if (next > SIZE_MAX / 2) return NULL;
        next *= 2;
    }
    if (next > SIZE_MAX / width) return NULL;
    void *grown = realloc(data, next * width);
    if (!grown) return NULL;
    *capacity = next; return grown;
}
static bool push_ref(NlCarrierValues *heap, NlCarrierRef ref) {
    if (heap->pending_count == SIZE_MAX) return false;
    NlCarrierRef *pending = grow(heap->pending, &heap->pending_capacity,
                               heap->pending_count + 1, sizeof *heap->pending);
    if (!pending) return false;
    heap->pending = pending;
    heap->pending[heap->pending_count++] = ref; return true;
}
static size_t seen_hash(Seen key) {
    uintptr_t bits = (uintptr_t)key.pointer >> 3;
    bits ^= bits >> 16; bits *= (uintptr_t)0x9e3779b1u;
    return (size_t)(bits ^ key.kind);
}
static bool seen_resize(NlCarrierValues *heap, size_t capacity) {
    if (capacity > SIZE_MAX / sizeof(Seen)) return false;
    Seen *next = calloc(capacity, sizeof *next);
    if (!next) return false;
    for (size_t i = 0; i < heap->seen_capacity; ++i) {
        Seen item = heap->seen[i]; if (!item.pointer) continue;
        size_t slot = seen_hash(item) & (capacity - 1);
        while (next[slot].pointer) slot = (slot + 1) & (capacity - 1);
        next[slot] = item;
    }
    free(heap->seen); heap->seen = next; heap->seen_capacity = capacity; return true;
}
static int seen_add(NlCarrierValues *heap, NlCarrierRef ref) {
    if (!heap->seen_capacity || heap->seen_count + 1 > heap->seen_capacity / 2) {
        size_t next = heap->seen_capacity ? heap->seen_capacity * 2 : 32;
        if (next < heap->seen_capacity || !seen_resize(heap, next)) return -1;
    }
    Seen key = {value_key(ref), value_kind(ref)};
    size_t slot = seen_hash(key) & (heap->seen_capacity - 1);
    while (heap->seen[slot].pointer) {
        if (heap->seen[slot].pointer == key.pointer && heap->seen[slot].kind == key.kind) return 0;
        slot = (slot + 1) & (heap->seen_capacity - 1);
    }
    heap->seen[slot] = key; heap->seen_count++; return 1;
}
static bool push_children(NlCarrierValues *heap, NlCarrierRef ref) {
    if (ref.record_list) {
        DynArray *list = value_key(ref);
        for (int64_t i = 0; i < list->length; ++i) {
            Value record = create_void(); record.type = VAL_STRUCT;
            record.as.struct_val = (StructValue *)(intptr_t)dyn_array_get_int(list, i);
            if (!push_ref(heap, nl_carrier_value(record))) return false;
        }
        return true;
    }
    Value *fields = NULL; int count = 0;
    switch (ref.value.type) {
        case VAL_STRUCT: fields = ref.value.as.struct_val->field_values; count = ref.value.as.struct_val->field_count; break;
        case VAL_UNION: fields = ref.value.as.union_val->field_values; count = ref.value.as.union_val->field_count; break;
        case VAL_TUPLE: fields = ref.value.as.tuple_val->elements; count = ref.value.as.tuple_val->element_count; break;
        case VAL_ARRAY: {
            Array *a = ref.value.as.array_val;
            for (int i = 0; i < a->length; ++i) {
                Value child = create_void(); child.type = a->element_type;
                switch (a->element_type) {
                    case VAL_ARRAY: child = ((Value *)a->data)[i]; break;
                    case VAL_STRUCT: child.as.struct_val = ((StructValue **)a->data)[i]; break;
                    case VAL_UNION: child.as.union_val = ((UnionValue **)a->data)[i]; break;
                    case VAL_TUPLE: child.as.tuple_val = ((TupleValue **)a->data)[i]; break;
                    default: continue;
                }
                if (!push_ref(heap, nl_carrier_value(child))) return false;
            }
            break;
        }
        default: break;
    }
    for (int i = 0; i < count; ++i)
        if (!push_ref(heap, nl_carrier_value(fields[i]))) return false;
    return true;
}
static bool append_key(NlCarrierValues *heap, const void *key) {
    if (heap->key_count == SIZE_MAX) return false;
    const void **keys = grow(heap->keys, &heap->key_capacity,
                             heap->key_count + 1, sizeof *heap->keys);
    if (!keys) return false;
    heap->keys = keys;
    heap->keys[heap->key_count++] = key; return true;
}
static bool walk(NlCarrierValues *heap) {
    while (heap->pending_count) {
        NlCarrierRef ref = heap->pending[--heap->pending_count];
        void *key = value_key(ref);
        if (!key) { if (ref.record_list && ref.value.type != VAL_INT) return false; continue; }
        CarrierValue *owned = nl_carrier_heap_context(heap->engine, key);
        if (owned && value_kind(owned->ref) != value_kind(ref)) return false;
        if (!valid_value(ref)) return false;
        int fresh = seen_add(heap, ref); if (fresh < 0) return false; if (!fresh) continue;
        if (owned) {
            if (heap->mark) { if (!heap->mark(key, heap->visitor)) return false; }
            else if (!append_key(heap, key)) return false;
            if (!heap->transfer) continue;
            for (size_t i = 0; i < owned->edge_count; ++i)
                if (!push_ref(heap, owned->edges[i])) return false;
        }
        if (!push_children(heap, ref)) return false;
    }
    return true;
}
static bool trace_value(const void *object, NlCarrierVisit mark, void *visitor, void *context) {
    CarrierValue *value = context; NlCarrierValues *heap = value->owner;
    (void)object;
    heap->mark = mark; heap->visitor = visitor;
    if (!valid_value(value->ref) || !push_children(heap, value->ref)) return false;
    for (size_t i = 0; i < value->edge_count; ++i)
        if (!push_ref(heap, value->edges[i])) return false;
    return walk(heap);
}
static void release_scalar(Value value) {
    if (value.type == VAL_STRING) {
        if (gc_is_managed(value.as.string_val)) gc_release(value.as.string_val);
        else free(value.as.string_val);
    } else if (value.type == VAL_FUNCTION) {
        free(value.as.function_val.function_name);
        free_function_signature(value.as.function_val.signature);
    }
}
static void release_fields(Value *fields, int count) {
    for (int i = 0; i < count; ++i) release_scalar(fields[i]);
}
static void destroy_value(void *object, void *context) {
    CarrierValue *value = context;
    if (value->ref.record_list) gc_release(object);
    else switch (value->ref.value.type) {
        case VAL_STRUCT: {
            StructValue *r = object; release_fields(r->field_values, r->field_count);
            release_struct_carrier_names(r); free(r->field_values); free(r); break;
        }
        case VAL_UNION: {
            UnionValue *u = object; release_fields(u->field_values, u->field_count);
            for (int i = 0; i < u->field_count; ++i) free(u->field_names[i]);
            free(u->field_names); free(u->field_values); free(u->union_name); free(u->variant_name); free(u); break;
        }
        case VAL_TUPLE: {
            TupleValue *t = object; release_fields(t->elements, t->element_count);
            free(t->elements); free(t); break;
        }
        case VAL_ARRAY: {
            Array *a = object;
            if (a->element_type == VAL_STRING) {
                for (int i = 0; i < a->length; ++i) {
                    Value item = create_void(); item.type = VAL_STRING;
                    item.as.string_val = ((char **)a->data)[i]; release_scalar(item);
                }
            } else if (a->element_type == VAL_FUNCTION) release_fields(a->data, a->length);
            free(a->data); free(a); break;
        }
        default: break;
    }
    free(value->edges); free(value);
}
NlCarrierValues *nl_carrier_values_new(void) {
    NlCarrierValues *heap = calloc(1, sizeof *heap);
    if (heap && !(heap->engine = nl_carrier_heap_new())) { free(heap); return NULL; }
    return heap;
}
bool nl_carrier_values_adopt(NlCarrierValues *heap, NlCarrierRef ref) {
    if (!heap || heap->busy || !value_key(ref) || !valid_value(ref) ||
        (ref.record_list && (!gc_is_managed(value_key(ref)) ||
                            gc_get_header(value_key(ref))->type != GC_TYPE_ARRAY))) return false;
    CarrierValue *value = calloc(1, sizeof *value); if (!value) return false;
    value->owner = heap; value->ref = ref;
    if (!nl_carrier_heap_track(heap->engine, value_key(ref), trace_value, destroy_value, value)) {
        free(value); return false;
    }
    return true;
}
bool nl_carrier_values_edges(NlCarrierValues *heap, NlCarrierRef ref,
                             const NlCarrierRef *edges, size_t count) {
    if (!heap || heap->busy || (count && !edges) || count > SIZE_MAX / sizeof *edges) return false;
    CarrierValue *value = nl_carrier_heap_context(heap->engine, value_key(ref));
    if (!value || value_kind(value->ref) != value_kind(ref)) return false;
    NlCarrierRef *copy = count ? malloc(count * sizeof *copy) : NULL;
    if (count && !copy) return false;
    if (count) memcpy(copy, edges, count * sizeof *copy);
    free(value->edges); value->edges = copy; value->edge_count = count; return true;
}
static bool begin_walk(NlCarrierValues *heap, const NlCarrierRef *roots, size_t count, bool transfer) {
    if (!heap || heap->busy || (count && !roots)) return false;
    heap->pending_count = heap->key_count = heap->seen_count = 0;
    if (heap->seen_capacity) memset(heap->seen, 0, heap->seen_capacity * sizeof *heap->seen);
    heap->mark = NULL; heap->visitor = NULL; heap->transfer = transfer; heap->busy = true;
    for (size_t i = 0; i < count; ++i) if (!push_ref(heap, roots[i])) return false;
    return walk(heap);
}
bool nl_carrier_values_collect(NlCarrierValues *heap, const NlCarrierRef *roots, size_t count) {
    if (!heap || heap->busy) return false;
    bool result = begin_walk(heap, roots, count, false);
    if (result) result = nl_carrier_heap_collect(heap->engine, heap->keys, heap->key_count);
    heap->busy = false; return result;
}
bool nl_carrier_values_transfer(NlCarrierValues *heap, const NlCarrierRef *roots, size_t count) {
    if (!heap || heap->busy) return false;
    bool result = begin_walk(heap, roots, count, true);
    if (result) for (size_t i = 0; i < heap->key_count; ++i) {
        CarrierValue *value = nl_carrier_heap_context(heap->engine, heap->keys[i]);
        if (!nl_carrier_heap_forget(heap->engine, heap->keys[i])) { result = false; break; }
        free(value->edges); free(value);
    }
    heap->busy = false; return result;
}
bool nl_carrier_values_clear(NlCarrierValues *heap) {
    if (!heap || heap->busy) return false;
    heap->busy = true;
    bool result = nl_carrier_heap_clear(heap->engine);
    heap->busy = false; return result;
}
bool nl_carrier_values_contains(const NlCarrierValues *heap, NlCarrierRef ref) {
    if (!heap) return false;
    CarrierValue *value = nl_carrier_heap_context(heap->engine, value_key(ref));
    return value && value_kind(value->ref) == value_kind(ref);
}
bool nl_carrier_values_free(NlCarrierValues *heap) {
    if (!heap) return true;
    if (heap->busy) return false;
    heap->busy = true;
    if (!nl_carrier_heap_free(heap->engine)) { heap->busy = false; return false; }
    free(heap->pending); free(heap->seen); free(heap->keys); free(heap); return true;
}
NlCarrierHeapStats nl_carrier_values_stats(const NlCarrierValues *heap) {
    return nl_carrier_heap_stats(heap ? heap->engine : NULL);
}
