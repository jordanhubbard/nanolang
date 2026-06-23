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

void vm_heap_init(VmHeap *heap) {
    memset(heap, 0, sizeof(*heap));
    heap->intern_capacity = 256;
    heap->intern_table = calloc(heap->intern_capacity, sizeof(VmString *));
}

void vm_heap_destroy(VmHeap *heap) {
    /* Release all interned strings */
    for (uint32_t i = 0; i < heap->intern_count; i++) {
        if (heap->intern_table[i]) {
            /* Force free regardless of ref_count */
            free(heap->intern_table[i]);
        }
    }
    free(heap->intern_table);
    heap->intern_table = NULL;
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
 * Reference Counting
 * ======================================================================== */

void vm_retain(NanoValue v) {
    if (!val_is_heap_obj(v) && v.tag != TAG_FUNCTION) return;
    void *ptr = v.as.obj;
    if (!ptr) return;
    VmHeapHeader *hdr = (VmHeapHeader *)ptr;
    hdr->ref_count++;
}

static void release_array(VmHeap *heap, VmArray *a);
static void release_struct(VmHeap *heap, VmStruct *s);
static void release_union(VmHeap *heap, VmUnion *u);
static void release_tuple(VmHeap *heap, VmTuple *t);
static void release_closure(VmHeap *heap, VmClosure *c);
static void release_hashmap(VmHeap *heap, VmHashMap *m);

void vm_release(VmHeap *heap, NanoValue v) {
    if (!val_is_heap_obj(v) && v.tag != TAG_FUNCTION) return;
    void *ptr = v.as.obj;
    if (!ptr) return;
    VmHeapHeader *hdr = (VmHeapHeader *)ptr;
    if (hdr->ref_count == 0) return; /* already freed or static */
    hdr->ref_count--;
    if (hdr->ref_count > 0) return;

    /* ref_count reached 0 - free the object */
    switch (v.tag) {
        case TAG_STRING: {
            VmString *s = v.as.string;
            heap->stats.freed += sizeof(VmString) + s->length + 1;
            heap->stats.num_objects--;
            /* Remove from intern table if present */
            for (uint32_t i = 0; i < heap->intern_count; i++) {
                if (heap->intern_table[i] == s) {
                    heap->intern_table[i] = heap->intern_table[--heap->intern_count];
                    break;
                }
            }
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
        case TAG_FUNCTION:
            /* Could be a closure */
            if (v.as.closure && v.as.closure->header.obj_type == TAG_FUNCTION) {
                release_closure(heap, v.as.closure);
            }
            break;
        default:
            break;
    }
}

static void release_array(VmHeap *heap, VmArray *a) {
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
        VmHMEntry *entry = m->buckets[i];
        while (entry) {
            VmHMEntry *next = entry->next;
            vm_release(heap, entry->key);
            vm_release(heap, entry->value);
            free(entry);
            entry = next;
        }
    }
    heap->stats.freed += sizeof(VmHashMap) + m->bucket_count * sizeof(VmHMEntry *);
    heap->stats.num_objects--;
    free(m->buckets);
    free(m);
}

/* ========================================================================
 * String Allocation
 * ======================================================================== */

VmString *vm_string_new(VmHeap *heap, const char *data, uint32_t length) {
    uint32_t hash = fnv1a(data, length);

    /* Check intern table for dedup */
    for (uint32_t i = 0; i < heap->intern_count; i++) {
        VmString *s = heap->intern_table[i];
        if (s && s->hash == hash && s->length == length &&
            memcmp(s->data, data, length) == 0) {
            s->header.ref_count++;
            return s;
        }
    }

    /* Allocate new string */
    size_t sz = sizeof(VmString) + length + 1;
    VmString *s = malloc(sz);
    if (!s) return NULL;
    s->header.ref_count = 1;
    s->header.obj_type = TAG_STRING;
    s->length = length;
    s->hash = hash;
    memcpy(s->data, data, length);
    s->data[length] = '\0';

    heap->stats.allocated += sz;
    heap->stats.num_objects++;

    /* Add to intern table */
    if (heap->intern_count >= heap->intern_capacity) {
        uint32_t new_cap = heap->intern_capacity * 2;
        VmString **new_table = realloc(heap->intern_table, new_cap * sizeof(VmString *));
        if (new_table) {
            heap->intern_table = new_table;
            heap->intern_capacity = new_cap;
        }
    }
    if (heap->intern_count < heap->intern_capacity) {
        heap->intern_table[heap->intern_count++] = s;
    }

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

bool vmstring_contains(VmString *haystack, VmString *needle) {
    if (needle->length == 0) return true;
    if (needle->length > haystack->length) return false;
    return strstr(haystack->data, needle->data) != NULL;
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

VmArray *vm_array_new(VmHeap *heap, uint8_t elem_type, uint32_t initial_capacity) {
    if (initial_capacity < 8) initial_capacity = 8;
    VmArray *a = malloc(sizeof(VmArray));
    if (!a) return NULL;
    a->header.ref_count = 1;
    a->header.obj_type = TAG_ARRAY;
    a->elem_type = elem_type;
    a->length = 0;
    a->capacity = initial_capacity;
    a->elements = calloc(initial_capacity, sizeof(NanoValue));
    heap->stats.allocated += sizeof(VmArray) + initial_capacity * sizeof(NanoValue);
    heap->stats.num_objects++;
    return a;
}

static void array_grow(VmArray *a) {
    uint32_t new_cap = a->capacity * 2;
    NanoValue *new_elems = realloc(a->elements, new_cap * sizeof(NanoValue));
    if (!new_elems) return;
    a->elements = new_elems;
    a->capacity = new_cap;
}

void vm_array_push(VmArray *a, NanoValue v) {
    if (a->length >= a->capacity) array_grow(a);
    a->elements[a->length++] = v;
    vm_retain(v);
}

NanoValue vm_array_pop(VmArray *a) {
    if (a->length == 0) return val_void();
    a->length--;
    NanoValue v = a->elements[a->length];
    /* Don't release - caller takes ownership */
    return v;
}

NanoValue vm_array_get(VmArray *a, uint32_t index) {
    if (index >= a->length) return val_void();
    return a->elements[index];
}

void vm_array_set(VmArray *a, uint32_t index, NanoValue v) {
    if (index >= a->length) return;
    a->elements[index] = v;
}

VmArray *vm_array_slice(VmHeap *heap, VmArray *a, uint32_t start, uint32_t end) {
    if (start >= a->length) start = a->length;
    if (end > a->length) end = a->length;
    if (end <= start) return vm_array_new(heap, a->elem_type, 8);

    uint32_t new_len = end - start;
    VmArray *result = vm_array_new(heap, a->elem_type, new_len);
    for (uint32_t i = 0; i < new_len; i++) {
        result->elements[i] = a->elements[start + i];
        vm_retain(result->elements[i]);
    }
    result->length = new_len;
    return result;
}

void vm_array_remove(VmArray *a, uint32_t index) {
    if (index >= a->length) return;
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
    s->def_idx = def_idx;
    s->field_count = field_count;
    s->field_names = NULL;
    s->fields = calloc(field_count, sizeof(NanoValue));
    heap->stats.allocated += sizeof(VmStruct) + field_count * sizeof(NanoValue);
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
    u->def_idx = def_idx;
    u->variant = variant;
    u->field_count = field_count;
    u->fields = calloc(field_count, sizeof(NanoValue));
    heap->stats.allocated += sizeof(VmUnion) + field_count * sizeof(NanoValue);
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
    t->count = count;
    heap->stats.allocated += sz;
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
    c->header.obj_type = TAG_FUNCTION;
    c->fn_idx = fn_idx;
    c->capture_count = capture_count;
    heap->stats.allocated += sz;
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
    m->key_type = key_type;
    m->val_type = val_type;
    m->count = 0;
    m->bucket_count = HM_INITIAL_BUCKETS;
    m->buckets = calloc(HM_INITIAL_BUCKETS, sizeof(VmHMEntry *));
    heap->stats.allocated += sizeof(VmHashMap) + HM_INITIAL_BUCKETS * sizeof(VmHMEntry *);
    heap->stats.num_objects++;
    return m;
}

static void hm_resize(VmHeap *heap, VmHashMap *m) {
    uint32_t new_count = m->bucket_count * 2;
    VmHMEntry **new_buckets = calloc(new_count, sizeof(VmHMEntry *));
    if (!new_buckets) return;

    for (uint32_t i = 0; i < m->bucket_count; i++) {
        VmHMEntry *entry = m->buckets[i];
        while (entry) {
            VmHMEntry *next = entry->next;
            uint32_t idx = hash_value(entry->key) % new_count;
            entry->next = new_buckets[idx];
            new_buckets[idx] = entry;
            entry = next;
        }
    }
    free(m->buckets);
    m->buckets = new_buckets;
    m->bucket_count = new_count;
    (void)heap;
}

NanoValue vm_hashmap_get(VmHashMap *m, NanoValue key) {
    uint32_t idx = hash_value(key) % m->bucket_count;
    VmHMEntry *entry = m->buckets[idx];
    while (entry) {
        if (val_equal(entry->key, key)) return entry->value;
        entry = entry->next;
    }
    return val_void();
}

void vm_hashmap_set(VmHeap *heap, VmHashMap *m, NanoValue key, NanoValue value) {
    uint32_t idx = hash_value(key) % m->bucket_count;
    VmHMEntry *entry = m->buckets[idx];
    while (entry) {
        if (val_equal(entry->key, key)) {
            vm_release(heap, entry->value);
            entry->value = value;
            vm_retain(value);
            return;
        }
        entry = entry->next;
    }

    /* New entry */
    VmHMEntry *new_entry = malloc(sizeof(VmHMEntry));
    if (!new_entry) return;
    new_entry->key = key;
    new_entry->value = value;
    vm_retain(key);
    vm_retain(value);
    new_entry->next = m->buckets[idx];
    m->buckets[idx] = new_entry;
    m->count++;

    if ((double)m->count / (double)m->bucket_count > HM_LOAD_FACTOR) {
        hm_resize(heap, m);
    }
}

bool vm_hashmap_has(VmHashMap *m, NanoValue key) {
    uint32_t idx = hash_value(key) % m->bucket_count;
    VmHMEntry *entry = m->buckets[idx];
    while (entry) {
        if (val_equal(entry->key, key)) return true;
        entry = entry->next;
    }
    return false;
}

void vm_hashmap_delete(VmHeap *heap, VmHashMap *m, NanoValue key) {
    uint32_t idx = hash_value(key) % m->bucket_count;
    VmHMEntry **prev = &m->buckets[idx];
    VmHMEntry *entry = *prev;
    while (entry) {
        if (val_equal(entry->key, key)) {
            *prev = entry->next;
            vm_release(heap, entry->key);
            vm_release(heap, entry->value);
            free(entry);
            m->count--;
            return;
        }
        prev = &entry->next;
        entry = entry->next;
    }
}

VmArray *vm_hashmap_keys(VmHeap *heap, VmHashMap *m) {
    VmArray *result = vm_array_new(heap, m->key_type, m->count > 0 ? m->count : 8);
    for (uint32_t i = 0; i < m->bucket_count; i++) {
        VmHMEntry *entry = m->buckets[i];
        while (entry) {
            vm_array_push(result, entry->key);
            entry = entry->next;
        }
    }
    return result;
}

VmArray *vm_hashmap_values(VmHeap *heap, VmHashMap *m) {
    VmArray *result = vm_array_new(heap, m->val_type, m->count > 0 ? m->count : 8);
    for (uint32_t i = 0; i < m->bucket_count; i++) {
        VmHMEntry *entry = m->buckets[i];
        while (entry) {
            vm_array_push(result, entry->value);
            entry = entry->next;
        }
    }
    return result;
}
