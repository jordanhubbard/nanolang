/*
 * NanoVM Heap - GC-managed heap objects
 *
 * Reference counting GC. Every heap object has a VmHeapHeader prepended.
 * When ref_count reaches 0, the object is freed (with recursive release
 * of any contained references).
 */

#ifndef NANOVM_HEAP_H
#define NANOVM_HEAP_H

#include "value.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ========================================================================
 * Heap Header (prepended to every heap object)
 * ======================================================================== */

typedef struct VmHeapHeader {
    uint32_t ref_count;
    uint8_t  obj_type;  /* NanoValueTag */
} VmHeapHeader;

/* ========================================================================
 * Heap Object Types
 * ======================================================================== */

/* String: immutable, reference counted */
struct VmString {
    VmHeapHeader header;
    uint32_t length;
    uint32_t hash;       /* Cached hash for interning */
    char data[];         /* Flexible array member */
};

/* Array: dynamic, growable */
struct VmArray {
    VmHeapHeader header;
    uint8_t  elem_type;  /* Expected element type tag */
    uint32_t length;
    uint32_t capacity;
    NanoValue *elements;
};

/* Struct: named fields */
struct VmStruct {
    VmHeapHeader header;
    uint32_t    def_idx;     /* Struct definition index */
    uint32_t    field_count;
    VmString   **field_names; /* Optional: field name strings */
    NanoValue   *fields;
};

/* Union: tagged union with variant fields */
struct VmUnion {
    VmHeapHeader header;
    uint32_t def_idx;
    uint16_t variant;
    uint16_t field_count;
    NanoValue *fields;
};

/* Tuple: fixed-size heterogeneous */
struct VmTuple {
    VmHeapHeader header;
    uint32_t count;
    NanoValue elements[];  /* Flexible array member */
};

/* Closure: function + captured environment */
struct VmClosure {
    VmHeapHeader header;
    uint32_t fn_idx;
    uint16_t capture_count;
    NanoValue captures[];  /* Flexible array member */
};

/* HashMap entry */
typedef struct VmHMEntry {
    NanoValue key;
    NanoValue value;
    struct VmHMEntry *next;  /* Chaining for collisions */
} VmHMEntry;

/* HashMap: key-value map */
struct VmHashMap {
    VmHeapHeader header;
    uint8_t key_type;
    uint8_t val_type;
    uint32_t count;
    uint32_t bucket_count;
    VmHMEntry **buckets;
};

/* ========================================================================
 * Heap State
 * ======================================================================== */

typedef struct {
    size_t allocated;
    size_t freed;
    size_t num_objects;
} VmHeapStats;

typedef struct {
    VmHeapStats stats;
    /* String interning table */
    VmString **intern_table;
    uint32_t intern_count;
    uint32_t intern_capacity;
} VmHeap;

/* ========================================================================
 * Heap API
 * ======================================================================== */

/* Initialize/destroy heap */
void vm_heap_init(VmHeap *heap);
void vm_heap_destroy(VmHeap *heap);

/* Reference counting */
void vm_retain(NanoValue v);
void vm_release(VmHeap *heap, NanoValue v);

/* String allocation */
VmString *vm_string_new(VmHeap *heap, const char *data, uint32_t length);
VmString *vm_string_concat(VmHeap *heap, VmString *a, VmString *b);
VmString *vm_string_substr(VmHeap *heap, VmString *s, uint32_t start, uint32_t len);

/* String operations */
const char *vmstring_cstr(VmString *s);
uint32_t vmstring_len(VmString *s);
bool vmstring_equal(VmString *a, VmString *b);
int vmstring_compare(VmString *a, VmString *b);
bool vmstring_contains(VmString *haystack, VmString *needle);
VmString *vmstring_char_at(VmHeap *heap, VmString *s, uint32_t index);

/* Array allocation */
VmArray *vm_array_new(VmHeap *heap, uint8_t elem_type, uint32_t initial_capacity);
void vm_array_push(VmArray *a, NanoValue v);
NanoValue vm_array_pop(VmArray *a);
NanoValue vm_array_get(VmArray *a, uint32_t index);
void vm_array_set(VmArray *a, uint32_t index, NanoValue v);
VmArray *vm_array_slice(VmHeap *heap, VmArray *a, uint32_t start, uint32_t end);
void vm_array_remove(VmArray *a, uint32_t index);

/* Struct allocation */
VmStruct *vm_struct_new(VmHeap *heap, uint32_t def_idx, uint32_t field_count);

/* Union allocation */
VmUnion *vm_union_new(VmHeap *heap, uint32_t def_idx, uint16_t variant, uint16_t field_count);

/* Tuple allocation */
VmTuple *vm_tuple_new(VmHeap *heap, uint32_t count);

/* Closure allocation */
VmClosure *vm_closure_new(VmHeap *heap, uint32_t fn_idx, uint16_t capture_count);

/* HashMap allocation */
VmHashMap *vm_hashmap_new(VmHeap *heap, uint8_t key_type, uint8_t val_type);
NanoValue vm_hashmap_get(VmHashMap *m, NanoValue key);
void vm_hashmap_set(VmHeap *heap, VmHashMap *m, NanoValue key, NanoValue value);
bool vm_hashmap_has(VmHashMap *m, NanoValue key);
void vm_hashmap_delete(VmHeap *heap, VmHashMap *m, NanoValue key);
VmArray *vm_hashmap_keys(VmHeap *heap, VmHashMap *m);
VmArray *vm_hashmap_values(VmHeap *heap, VmHashMap *m);

/* String conversion helpers (for STR_FROM_INT, STR_FROM_FLOAT, CAST_STRING) */
VmString *vm_string_from_int(VmHeap *heap, int64_t v);
VmString *vm_string_from_float(VmHeap *heap, double v);
VmString *vm_string_from_bool(VmHeap *heap, bool v);

#endif /* NANOVM_HEAP_H */
