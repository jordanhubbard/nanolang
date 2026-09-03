/*
 * NanoVM Heap - reference counting GC and heap object implementations
 */

#include "heap.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ========================================================================
 * Heap Init / Destroy
 * ======================================================================== */

#define VM_INTERN_INITIAL_BUCKETS 256u

void vm_heap_init(VmHeap *heap) {
    memset(heap, 0, sizeof(*heap));
    heap->intern_bucket_count = VM_INTERN_INITIAL_BUCKETS;
    heap->intern_count = 0;
    heap->intern_buckets = calloc(heap->intern_bucket_count, sizeof(VmString *));
}

void vm_heap_destroy(VmHeap *heap) {
    /* Reclaim cycles before tearing anything else down, so their members get
     * their normal release path rather than being abandoned. */
    vm_gc_collect_cycles(heap);
    vm_gc_cycle_buffer_free(heap);
    /* Release all interned strings by walking each bucket chain. */
    for (uint32_t b = 0; b < heap->intern_bucket_count; b++) {
        VmString *s = heap->intern_buckets ? heap->intern_buckets[b] : NULL;
        while (s) {
            VmString *next = s->intern_next;
            /* Force free regardless of ref_count */
            free(s);
            s = next;
        }
    }
    free(heap->intern_buckets);
    heap->intern_buckets = NULL;
    heap->intern_bucket_count = 0;
    heap->intern_count = 0;
}

/* ========================================================================
 * String Hashing
 * ======================================================================== */

static uint32_t fnv1a(const char *data, uint32_t len) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        hash ^= (uint8_t)data[i];
        hash *= 16777619u;
    }
    return hash;
}

/* ========================================================================
 * String Interning Table (chained hash buckets)
 * ======================================================================== */

/* Load factor threshold above which the bucket array is doubled. */
#define VM_INTERN_MAX_LOAD_NUM 3u
#define VM_INTERN_MAX_LOAD_DEN 4u

/* Grow the bucket array (doubling) and rehash existing strings. No-op on
 * allocation failure: the table stays correct, only denser. */
static void vm_intern_maybe_grow(VmHeap *heap) {
    if ((uint64_t)heap->intern_count * VM_INTERN_MAX_LOAD_DEN <
        (uint64_t)heap->intern_bucket_count * VM_INTERN_MAX_LOAD_NUM) {
        return;
    }
    uint32_t new_count = heap->intern_bucket_count ? heap->intern_bucket_count * 2 : 256;
    VmString **new_buckets = calloc(new_count, sizeof(VmString *));
    if (!new_buckets) return;
    for (uint32_t b = 0; b < heap->intern_bucket_count; b++) {
        VmString *s = heap->intern_buckets[b];
        while (s) {
            VmString *next = s->intern_next;
            uint32_t idx = s->hash & (new_count - 1);
            s->intern_next = new_buckets[idx];
            new_buckets[idx] = s;
            s = next;
        }
    }
    free(heap->intern_buckets);
    heap->intern_buckets = new_buckets;
    heap->intern_bucket_count = new_count;
}

/* Look up an existing interned string with matching hash/length/content. */
static VmString *vm_intern_lookup(VmHeap *heap, uint32_t hash,
                                  const char *data, uint32_t length) {
    if (heap->intern_bucket_count == 0) return NULL;
    uint32_t idx = hash & (heap->intern_bucket_count - 1);
    for (VmString *s = heap->intern_buckets[idx]; s; s = s->intern_next) {
        if (s->hash == hash && s->length == length &&
            memcmp(s->data, data, length) == 0) {
            return s;
        }
    }
    return NULL;
}

/* Insert a freshly allocated string into its bucket chain. */
static void vm_intern_insert(VmHeap *heap, VmString *s) {
    vm_intern_maybe_grow(heap);
    uint32_t idx = s->hash & (heap->intern_bucket_count - 1);
    s->intern_next = heap->intern_buckets[idx];
    heap->intern_buckets[idx] = s;
    heap->intern_count++;
}

/* Remove a string from its bucket chain in O(1) expected time. */
static void vm_intern_unlink(VmHeap *heap, VmString *s) {
    if (heap->intern_bucket_count == 0) return;
    uint32_t idx = s->hash & (heap->intern_bucket_count - 1);
    VmString **link = &heap->intern_buckets[idx];
    while (*link) {
        if (*link == s) {
            *link = s->intern_next;
            s->intern_next = NULL;
            heap->intern_count--;
            return;
        }
        link = &(*link)->intern_next;
    }
}

/* ========================================================================
 * Reference Counting
 * ======================================================================== */

void vm_retain(VmHeap *heap, NanoValue v) {
    if (!val_is_heap_obj(v)) return;
    void *ptr = v.as.obj;
    if (!ptr) return;
    VmHeapHeader *hdr = (VmHeapHeader *)ptr;
    if (heap) heap->stats.retain_calls++;
    hdr->ref_count++;
}

static void release_array(VmHeap *heap, VmArray *a);
static void release_struct(VmHeap *heap, VmStruct *s);
static void release_union(VmHeap *heap, VmUnion *u);
static void release_tuple(VmHeap *heap, VmTuple *t);
static void release_closure(VmHeap *heap, VmClosure *c);
static void release_hashmap(VmHeap *heap, VmHashMap *m);

void vm_release(VmHeap *heap, NanoValue v) {
    if (!val_is_heap_obj(v)) return;
    void *ptr = v.as.obj;
    if (!ptr) return;
    if (heap) heap->stats.release_calls++;
    VmHeapHeader *hdr = (VmHeapHeader *)ptr;
    if (hdr->ref_count == 0) return; /* already freed or static */
    hdr->ref_count--;
    if (hdr->ref_count > 0) {
        /* Still referenced -- but possibly only from inside a cycle, which
         * refcounting alone can never resolve. Remember it as a candidate;
         * vm_gc_collect_cycles decides. */
        if (heap) vm_gc_note_suspect(heap, v);
        return;
    }

    /* Reached zero while sitting on the suspect buffer. The buffer holds a
     * bare pointer, so freeing now leaves it dangling and the next collection
     * reads a freed header -- which is what AddressSanitizer caught. The
     * collector frees it instead, on the pass that finds it at zero. */
    if (heap && hdr->buffered) {
        hdr->colour = VM_GC_BLACK;
        return;
    }

    /* ref_count reached 0 - free the object */
    switch (v.tag) {
        case TAG_STRING: {
            VmString *s = v.as.string;
            heap->stats.freed += sizeof(VmString) + s->length + 1;
            heap->stats.num_objects--;
            /* Remove from intern table (O(1): unlink from its bucket chain). */
            vm_intern_unlink(heap, s);
            free(s);
            break;
        }
        case TAG_ARRAY:
            release_array(heap, v.as.array);
            break;
        case TAG_STRUCT:
            release_struct(heap, v.as.sval);
            break;
        case TAG_UNION:
            release_union(heap, v.as.uval);
            break;
        case TAG_TUPLE:
            release_tuple(heap, v.as.tuple);
            break;
        case TAG_HASHMAP:
            release_hashmap(heap, v.as.hashmap);
            break;
        case TAG_CLOSURE:
            release_closure(heap, v.as.closure);
            break;
        default:
            break;
    }
}

static void release_array(VmHeap *heap, VmArray *a) {
    if (a->unboxed) {
        /* Unboxed elements are plain payloads with no owned references. */
        heap->stats.freed += sizeof(VmArray) +
            (size_t)a->capacity * vm_array_elem_size(a->elem_type);
        heap->stats.num_objects--;
        free(a->packed);
        free(a);
        return;
    }
    for (uint32_t i = 0; i < a->length; i++) {
        vm_release(heap, a->elements[i]);
    }
    heap->stats.freed += sizeof(VmArray) + a->capacity * sizeof(NanoValue);
    heap->stats.num_objects--;
    free(a->elements);
    free(a);
}

static void release_struct(VmHeap *heap, VmStruct *s) {
    for (uint32_t i = 0; i < s->field_count; i++) {
        vm_release(heap, s->fields[i]);
    }
    if (s->field_names) {
        for (uint32_t i = 0; i < s->field_count; i++) {
            if (s->field_names[i]) {
                NanoValue sv = val_string(s->field_names[i]);
                vm_release(heap, sv);
            }
        }
        free(s->field_names);
    }
    heap->stats.freed += sizeof(VmStruct) + s->field_count * sizeof(NanoValue);
    heap->stats.num_objects--;
    free(s->fields);
    free(s);
}

static void release_union(VmHeap *heap, VmUnion *u) {
    for (uint32_t i = 0; i < u->field_count; i++) {
        vm_release(heap, u->fields[i]);
    }
    heap->stats.freed += sizeof(VmUnion) + u->field_count * sizeof(NanoValue);
    heap->stats.num_objects--;
    free(u->fields);
    free(u);
}

static void release_tuple(VmHeap *heap, VmTuple *t) {
    for (uint32_t i = 0; i < t->count; i++) {
        vm_release(heap, t->elements[i]);
    }
    size_t sz = sizeof(VmTuple) + t->count * sizeof(NanoValue);
    heap->stats.freed += sz;
    heap->stats.num_objects--;
    free(t);
}

static void release_closure(VmHeap *heap, VmClosure *c) {
    for (uint16_t i = 0; i < c->capture_count; i++) {
        vm_release(heap, c->captures[i]);
    }
    size_t sz = sizeof(VmClosure) + c->capture_count * sizeof(NanoValue);
    heap->stats.freed += sz;
    heap->stats.num_objects--;
    free(c);
}

static void release_hashmap(VmHeap *heap, VmHashMap *m) {
    for (uint32_t i = 0; i < m->bucket_count; i++) {
        VmHMEntry *entry = &m->entries[i];
        if (entry->state == 1) {
            vm_release(heap, entry->key);
            vm_release(heap, entry->value);
        }
    }
    heap->stats.freed += sizeof(VmHashMap) + m->bucket_count * sizeof(VmHMEntry);
    heap->stats.num_objects--;
    free(m->entries);
    free(m);
}

/* ========================================================================
 * String Allocation
 * ======================================================================== */

VmString *vm_string_new(VmHeap *heap, const char *data, uint32_t length) {
    uint32_t hash = fnv1a(data, length);

    /* Check intern table for dedup (O(1) expected via bucket chain). */
    VmString *existing = vm_intern_lookup(heap, hash, data, length);
    if (existing) {
        existing->header.ref_count++;
        return existing;
    }

    /* Allocate new string */
    size_t sz = sizeof(VmString) + length + 1;
    VmString *s = malloc(sz);
    if (!s) return NULL;
    s->header.ref_count = 1;
    s->header.obj_type = TAG_STRING;
    s->header.colour = VM_GC_BLACK;
    s->header.buffered = 0;
    s->length = length;
    s->hash = hash;
    s->intern_next = NULL;
    memcpy(s->data, data, length);
    s->data[length] = '\0';

    heap->stats.allocated += sz;
    heap->stats.allocation_calls++;
    heap->stats.num_objects++;

    /* Add to intern table (O(1) amortized, doubling to bound load factor). */
    vm_intern_insert(heap, s);

    return s;
}

VmString *vm_string_concat(VmHeap *heap, VmString *a, VmString *b) {
    uint32_t new_len = a->length + b->length;
    char *buf = malloc(new_len);
    if (!buf) return NULL;
    memcpy(buf, a->data, a->length);
    memcpy(buf + a->length, b->data, b->length);
    VmString *result = vm_string_new(heap, buf, new_len);
    free(buf);
    return result;
}

VmString *vm_string_substr(VmHeap *heap, VmString *s, uint32_t start, uint32_t len) {
    if (start >= s->length) return vm_string_new(heap, "", 0);
    if (start + len > s->length) len = s->length - start;
    return vm_string_new(heap, s->data + start, len);
}

/* String operations */

const char *vmstring_cstr(VmString *s) {
    return s ? s->data : "";
}

uint32_t vmstring_len(VmString *s) {
    return s ? s->length : 0;
}

bool vmstring_equal(VmString *a, VmString *b) {
    if (a == b) return true;
    if (a->length != b->length) return false;
    if (a->hash != b->hash) return false;
    return memcmp(a->data, b->data, a->length) == 0;
}

int vmstring_compare(VmString *a, VmString *b) {
    uint32_t min_len = a->length < b->length ? a->length : b->length;
    int cmp = memcmp(a->data, b->data, min_len);
    if (cmp != 0) return cmp;
    if (a->length < b->length) return -1;
    if (a->length > b->length) return 1;
    return 0;
}

/* Length-aware substring search. Unlike strstr(), this honors the stored
 * lengths of both strings and therefore matches correctly across embedded
 * zero bytes in the haystack or needle. */
int64_t vmstring_find(VmString *haystack, VmString *needle) {
    uint32_t nlen = needle->length;
    if (nlen == 0) return 0;
    uint32_t hlen = haystack->length;
    if (nlen > hlen) return -1;
    const char *hay = haystack->data;
    const char *nee = needle->data;
    uint32_t last = hlen - nlen;
    for (uint32_t i = 0; i <= last; i++) {
        if (memcmp(hay + i, nee, nlen) == 0) return (int64_t)i;
    }
    return -1;
}

bool vmstring_contains(VmString *haystack, VmString *needle) {
    return vmstring_find(haystack, needle) >= 0;
}

VmString *vmstring_char_at(VmHeap *heap, VmString *s, uint32_t index) {
    if (index >= s->length) return vm_string_new(heap, "", 0);
    return vm_string_new(heap, &s->data[index], 1);
}

VmString *vm_string_from_int(VmHeap *heap, int64_t v) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%lld", (long long)v);
    return vm_string_new(heap, buf, (uint32_t)len);
}

VmString *vm_string_from_float(VmHeap *heap, double v) {
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%g", v);
    return vm_string_new(heap, buf, (uint32_t)len);
}

VmString *vm_string_from_bool(VmHeap *heap, bool v) {
    return v ? vm_string_new(heap, "true", 4) : vm_string_new(heap, "false", 5);
}

/* ========================================================================
 * Array Allocation
 * ======================================================================== */

bool vm_array_type_unboxable(uint8_t elem_type) {
    return elem_type == TAG_INT || elem_type == TAG_FLOAT ||
           elem_type == TAG_BOOL || elem_type == TAG_U8;
}

/* Size in bytes of one unboxed element for a given element type. Returns 0
 * for boxed types (callers must not use packed storage for those). */
size_t vm_array_elem_size(uint8_t elem_type) {
    switch (elem_type) {
        case TAG_INT:   return sizeof(int64_t);
        case TAG_FLOAT: return sizeof(double);
        case TAG_BOOL:  return sizeof(uint8_t);
        case TAG_U8:    return sizeof(uint8_t);
        default:        return 0;
    }
}

/* Read the unboxed element at `index` and materialize it as a NanoValue. */
static NanoValue packed_load(const VmArray *a, uint32_t index) {
    switch (a->elem_type) {
        case TAG_INT:   return val_int(((const int64_t *)a->packed)[index]);
        case TAG_FLOAT: return val_float(((const double *)a->packed)[index]);
        case TAG_BOOL:  return val_bool(((const uint8_t *)a->packed)[index] != 0);
        case TAG_U8:    return val_u8(((const uint8_t *)a->packed)[index]);
        default:        return val_void();
    }
}

/* Store a NanoValue into unboxed slot `index`. Values whose tag does not match
 * the array element type are coerced where it is lossless (int<->u8) and
 * otherwise stored as their raw payload; callers are expected to push matching
 * types. */
static void packed_store(VmArray *a, uint32_t index, NanoValue v) {
    switch (a->elem_type) {
        case TAG_INT:
            ((int64_t *)a->packed)[index] =
                (v.tag == TAG_U8) ? (int64_t)v.as.u8 : v.as.i64;
            break;
        case TAG_FLOAT:
            ((double *)a->packed)[index] =
                (v.tag == TAG_INT) ? (double)v.as.i64 : v.as.f64;
            break;
        case TAG_BOOL:
            ((uint8_t *)a->packed)[index] = v.as.boolean ? 1u : 0u;
            break;
        case TAG_U8:
            ((uint8_t *)a->packed)[index] =
                (v.tag == TAG_INT) ? (uint8_t)v.as.i64 : v.as.u8;
            break;
        default:
            break;
    }
}

VmArray *vm_array_new(VmHeap *heap, uint8_t elem_type, uint32_t initial_capacity) {
    if (initial_capacity < 8) initial_capacity = 8;
    VmArray *a = malloc(sizeof(VmArray));
    if (!a) return NULL;
    a->header.ref_count = 1;
    a->header.obj_type = TAG_ARRAY;
    a->header.colour = VM_GC_BLACK;
    a->header.buffered = 0;
    a->elem_type = elem_type;
    a->unboxed = vm_array_type_unboxable(elem_type) ? 1u : 0u;
    a->length = 0;
    a->capacity = initial_capacity;
    a->elements = NULL;
    a->packed = NULL;
    if (a->unboxed) {
        size_t esz = vm_array_elem_size(elem_type);
        a->packed = calloc(initial_capacity, esz);
        if (!a->packed) { free(a); return NULL; }
        heap->stats.allocated += sizeof(VmArray) + (size_t)initial_capacity * esz;
    } else {
        a->elements = calloc(initial_capacity, sizeof(NanoValue));
        if (!a->elements) { free(a); return NULL; }
        heap->stats.allocated += sizeof(VmArray) + initial_capacity * sizeof(NanoValue);
    }
    heap->stats.allocation_calls++;
    heap->stats.num_objects++;
    return a;
}

/* Returns true on success, false if the array could not grow (overflow or
 * OOM). Callers MUST NOT write past a->length when this returns false. */
static bool array_grow(VmArray *a) {
    uint32_t new_cap = a->capacity ? a->capacity * 2 : 4;
    size_t esz = a->unboxed ? vm_array_elem_size(a->elem_type) : sizeof(NanoValue);
    /* capacity*2 wraps to 0 for a 2^31-element array; reject rather than
     * realloc(0) and then write out of bounds. */
    if (new_cap <= a->capacity || new_cap > (UINT32_MAX / esz)) {
        return false;
    }
    if (a->unboxed) {
        void *new_buf = realloc(a->packed, (size_t)new_cap * esz);
        if (!new_buf) return false;
        a->packed = new_buf;
    } else {
        NanoValue *new_elems = realloc(a->elements, new_cap * sizeof(NanoValue));
        if (!new_elems) return false;
        a->elements = new_elems;
    }
    a->capacity = new_cap;
    return true;
}

void vm_array_push(VmHeap *heap, VmArray *a, NanoValue v) {
    if (a->length >= a->capacity) {
        /* On grow failure keep the array intact and drop the push rather than
         * writing past the buffer (heap overflow). */
        if (!array_grow(a)) return;
    }
    if (a->unboxed) {
        packed_store(a, a->length, v);
        a->length++;
        /* Unboxed payloads own no references; nothing to retain. */
        return;
    }
    a->elements[a->length++] = v;
    vm_retain(heap, v);
}

NanoValue vm_array_pop(VmArray *a) {
    if (a->length == 0) return val_void();
    a->length--;
    if (a->unboxed) {
        return packed_load(a, a->length);
    }
    NanoValue v = a->elements[a->length];
    /* Don't release - caller takes ownership */
    return v;
}

NanoValue vm_array_get(VmArray *a, uint32_t index) {
    if (index >= a->length) return val_void();
    if (a->unboxed) return packed_load(a, index);
    return a->elements[index];
}

void vm_array_set(VmArray *a, uint32_t index, NanoValue v) {
    if (index >= a->length) return;
    if (a->unboxed) {
        packed_store(a, index, v);
        return;
    }
    a->elements[index] = v;
}

VmArray *vm_array_slice(VmHeap *heap, VmArray *a, uint32_t start, uint32_t end) {
    if (start >= a->length) start = a->length;
    if (end > a->length) end = a->length;
    if (end <= start) return vm_array_new(heap, a->elem_type, 8);

    uint32_t new_len = end - start;
    VmArray *result = vm_array_new(heap, a->elem_type, new_len);
    if (!result) return NULL;
    if (a->unboxed) {
        size_t esz = vm_array_elem_size(a->elem_type);
        memcpy(result->packed,
               (const char *)a->packed + (size_t)start * esz,
               (size_t)new_len * esz);
        result->length = new_len;
        return result;
    }
    for (uint32_t i = 0; i < new_len; i++) {
        result->elements[i] = a->elements[start + i];
        vm_retain(heap, result->elements[i]);
    }
    result->length = new_len;
    return result;
}

void vm_array_remove(VmHeap *heap, VmArray *a, uint32_t index) {
    if (index >= a->length) return;
    if (a->unboxed) {
        size_t esz = vm_array_elem_size(a->elem_type);
        char *base = (char *)a->packed;
        memmove(base + (size_t)index * esz,
                base + (size_t)(index + 1) * esz,
                (size_t)(a->length - 1 - index) * esz);
        a->length--;
        return;
    }
    /* The array owned a reference to the element being dropped; nothing else
     * will decrement it, so this is the only place it can happen. */
    vm_release(heap, a->elements[index]);
    /* Shift elements left */
    for (uint32_t i = index; i < a->length - 1; i++) {
        a->elements[i] = a->elements[i + 1];
    }
    a->length--;
}

/* ========================================================================
 * Struct Allocation
 * ======================================================================== */

VmStruct *vm_struct_new(VmHeap *heap, uint32_t def_idx, uint32_t field_count) {
    VmStruct *s = malloc(sizeof(VmStruct));
    if (!s) return NULL;
    s->header.ref_count = 1;
    s->header.obj_type = TAG_STRUCT;
    s->header.colour = VM_GC_BLACK;
    s->header.buffered = 0;
    s->def_idx = def_idx;
    s->field_count = field_count;
    s->field_names = NULL;
    s->fields = calloc(field_count, sizeof(NanoValue));
    heap->stats.allocated += sizeof(VmStruct) + field_count * sizeof(NanoValue);
    heap->stats.allocation_calls++;
    heap->stats.num_objects++;
    return s;
}

/* ========================================================================
 * Union Allocation
 * ======================================================================== */

VmUnion *vm_union_new(VmHeap *heap, uint32_t def_idx, uint16_t variant, uint16_t field_count) {
    VmUnion *u = malloc(sizeof(VmUnion));
    if (!u) return NULL;
    u->header.ref_count = 1;
    u->header.obj_type = TAG_UNION;
    u->header.colour = VM_GC_BLACK;
    u->header.buffered = 0;
    u->def_idx = def_idx;
    u->variant = variant;
    u->field_count = field_count;
    u->fields = calloc(field_count, sizeof(NanoValue));
    heap->stats.allocated += sizeof(VmUnion) + field_count * sizeof(NanoValue);
    heap->stats.allocation_calls++;
    heap->stats.num_objects++;
    return u;
}

/* ========================================================================
 * Tuple Allocation
 * ======================================================================== */

VmTuple *vm_tuple_new(VmHeap *heap, uint32_t count) {
    size_t sz = sizeof(VmTuple) + count * sizeof(NanoValue);
    VmTuple *t = calloc(1, sz);
    if (!t) return NULL;
    t->header.ref_count = 1;
    t->header.obj_type = TAG_TUPLE;
    t->header.colour = VM_GC_BLACK;
    t->header.buffered = 0;
    t->count = count;
    heap->stats.allocated += sz;
    heap->stats.allocation_calls++;
    heap->stats.num_objects++;
    return t;
}

/* ========================================================================
 * Closure Allocation
 * ======================================================================== */

VmClosure *vm_closure_new(VmHeap *heap, uint32_t fn_idx, uint16_t capture_count) {
    size_t sz = sizeof(VmClosure) + capture_count * sizeof(NanoValue);
    VmClosure *c = calloc(1, sz);
    if (!c) return NULL;
    c->header.ref_count = 1;
    c->header.obj_type = TAG_CLOSURE;
    c->header.colour = VM_GC_BLACK;
    c->header.buffered = 0;
    c->fn_idx = fn_idx;
    c->capture_count = capture_count;
    heap->stats.allocated += sz;
    heap->stats.allocation_calls++;
    heap->stats.num_objects++;
    return c;
}

/* ========================================================================
 * HashMap
 * ======================================================================== */

#define HM_INITIAL_BUCKETS 16
#define HM_LOAD_FACTOR 0.75

static uint32_t hash_value(NanoValue v) {
    switch (v.tag) {
        case TAG_INT:    return (uint32_t)(v.as.i64 ^ (v.as.i64 >> 32));
        case TAG_STRING: return v.as.string ? v.as.string->hash : 0;
        case TAG_BOOL:   return v.as.boolean ? 1 : 0;
        case TAG_ENUM:   return (uint32_t)v.as.enum_val;
        default:         return 0;
    }
}

VmHashMap *vm_hashmap_new(VmHeap *heap, uint8_t key_type, uint8_t val_type) {
    VmHashMap *m = malloc(sizeof(VmHashMap));
    if (!m) return NULL;
    m->header.ref_count = 1;
    m->header.obj_type = TAG_HASHMAP;
    m->header.colour = VM_GC_BLACK;
    m->header.buffered = 0;
    m->key_type = key_type;
    m->val_type = val_type;
    m->count = 0;
    m->tombstone_count = 0;
    m->bucket_count = HM_INITIAL_BUCKETS;
    m->entries = calloc(HM_INITIAL_BUCKETS, sizeof(VmHMEntry));
    if (!m->entries) {
        free(m);
        return NULL;
    }
    heap->stats.allocated += sizeof(VmHashMap) + HM_INITIAL_BUCKETS * sizeof(VmHMEntry);
    heap->stats.allocation_calls++;
    heap->stats.num_objects++;
    return m;
}

static bool hm_find_slot(VmHashMap *m, NanoValue key, uint32_t *slot, bool *found) {
    uint32_t idx = hash_value(key) % m->bucket_count;
    uint32_t first_tombstone = UINT32_MAX;
    for (uint32_t probes = 0; probes < m->bucket_count; probes++) {
        VmHMEntry *entry = &m->entries[idx];
        if (entry->state == 0) {
            *slot = first_tombstone != UINT32_MAX ? first_tombstone : idx;
            *found = false;
            return true;
        }
        if (entry->state == 2) {
            if (first_tombstone == UINT32_MAX) first_tombstone = idx;
        } else if (val_equal(entry->key, key)) {
            *slot = idx;
            *found = true;
            return true;
        }
        idx = (idx + 1) % m->bucket_count;
    }
    if (first_tombstone != UINT32_MAX) {
        *slot = first_tombstone;
        *found = false;
        return true;
    }
    return false;
}

static bool hm_resize(VmHashMap *m, uint32_t new_count) {
    VmHMEntry *old_entries = m->entries;
    uint32_t old_count = m->bucket_count;
    VmHMEntry *new_entries = calloc(new_count, sizeof(VmHMEntry));
    if (!new_entries) return false;

    m->entries = new_entries;
    m->bucket_count = new_count;
    m->count = 0;
    m->tombstone_count = 0;
    for (uint32_t i = 0; i < old_count; i++) {
        if (old_entries[i].state == 1) {
            uint32_t slot;
            bool found;
            (void)hm_find_slot(m, old_entries[i].key, &slot, &found);
            m->entries[slot] = old_entries[i];
            m->entries[slot].state = 1;
            m->count++;
        }
    }
    free(old_entries);
    return true;
}

NanoValue vm_hashmap_get(VmHashMap *m, NanoValue key) {
    uint32_t slot;
    bool found;
    if (hm_find_slot(m, key, &slot, &found) && found) return m->entries[slot].value;
    return val_void();
}

void vm_hashmap_set(VmHeap *heap, VmHashMap *m, NanoValue key, NanoValue value) {
    if ((double)(m->count + m->tombstone_count + 1) / (double)m->bucket_count > HM_LOAD_FACTOR) {
        if (m->bucket_count > UINT32_MAX / 2 || !hm_resize(m, m->bucket_count * 2)) return;
    }

    uint32_t slot;
    bool found;
    if (!hm_find_slot(m, key, &slot, &found)) return;
    VmHMEntry *entry = &m->entries[slot];
    if (found) {
        vm_release(heap, entry->value);
        entry->value = value;
        vm_retain(heap, value);
        return;
    }
    if (entry->state == 2) m->tombstone_count--;
    entry->state = 1;
    entry->key = key;
    entry->value = value;
    vm_retain(heap, key);
    vm_retain(heap, value);
    m->count++;
}

bool vm_hashmap_has(VmHashMap *m, NanoValue key) {
    uint32_t slot;
    bool found;
    return hm_find_slot(m, key, &slot, &found) && found;
}

void vm_hashmap_delete(VmHeap *heap, VmHashMap *m, NanoValue key) {
    uint32_t slot;
    bool found;
    if (!hm_find_slot(m, key, &slot, &found) || !found) return;
    VmHMEntry *entry = &m->entries[slot];
    vm_release(heap, entry->key);
    vm_release(heap, entry->value);
    entry->state = 2;
    m->count--;
    m->tombstone_count++;
}

VmArray *vm_hashmap_keys(VmHeap *heap, VmHashMap *m) {
    VmArray *result = vm_array_new(heap, m->key_type, m->count > 0 ? m->count : 8);
    for (uint32_t i = 0; i < m->bucket_count; i++) {
        VmHMEntry *entry = &m->entries[i];
        if (entry->state == 1) vm_array_push(heap, result, entry->key);
    }
    return result;
}

VmArray *vm_hashmap_values(VmHeap *heap, VmHashMap *m) {
    VmArray *result = vm_array_new(heap, m->val_type, m->count > 0 ? m->count : 8);
    for (uint32_t i = 0; i < m->bucket_count; i++) {
        VmHMEntry *entry = &m->entries[i];
        if (entry->state == 1) vm_array_push(heap, result, entry->value);
    }
    return result;
}
