/*
 * NanoVM Value Representation
 *
 * Tagged values for the VM operand stack and heap objects.
 * Each value is 16 bytes: 1-byte tag + 15-byte payload.
 */

#ifndef NANOVM_VALUE_H
#define NANOVM_VALUE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include "../nanoisa/isa.h"

/* ========================================================================
 * Forward declarations for heap objects
 * ======================================================================== */

typedef struct VmString VmString;
typedef struct VmArray VmArray;
typedef struct VmStruct VmStruct;
typedef struct VmUnion VmUnion;
typedef struct VmTuple VmTuple;
typedef struct VmHashMap VmHashMap;
typedef struct VmClosure VmClosure;

/* ========================================================================
 * NanoValue - the core tagged value
 * ======================================================================== */

typedef struct {
    uint8_t tag;  /* NanoValueTag */
    union {
        int64_t  i64;
        double   f64;
        uint8_t  u8;
        bool     boolean;
        int32_t  enum_val;    /* Enum variant index */
        uint32_t fn_idx;      /* Function table index */
        uint32_t proxy_id;    /* Opaque proxy ID */

        /* Heap object pointers */
        VmString  *string;
        VmArray   *array;
        VmStruct  *sval;
        VmUnion   *uval;
        VmTuple   *tuple;
        VmHashMap *hashmap;
        VmClosure *closure;
        void      *obj;       /* Generic heap object pointer */
    } as;
} NanoValue;

/* ========================================================================
 * Value constructors
 * ======================================================================== */

static inline NanoValue val_void(void) {
    NanoValue v = {0};
    v.tag = TAG_VOID;
    return v;
}

static inline NanoValue val_int(int64_t n) {
    NanoValue v = {0};
    v.tag = TAG_INT;
    v.as.i64 = n;
    return v;
}

static inline NanoValue val_float(double d) {
    NanoValue v = {0};
    v.tag = TAG_FLOAT;
    v.as.f64 = d;
    return v;
}

static inline NanoValue val_bool(bool b) {
    NanoValue v = {0};
    v.tag = TAG_BOOL;
    v.as.boolean = b;
    return v;
}

static inline NanoValue val_u8(uint8_t b) {
    NanoValue v = {0};
    v.tag = TAG_U8;
    v.as.u8 = b;
    return v;
}

static inline NanoValue val_string(VmString *s) {
    NanoValue v = {0};
    v.tag = TAG_STRING;
    v.as.string = s;
    return v;
}

static inline NanoValue val_array(VmArray *a) {
    NanoValue v = {0};
    v.tag = TAG_ARRAY;
    v.as.array = a;
    return v;
}

static inline NanoValue val_struct(VmStruct *s) {
    NanoValue v = {0};
    v.tag = TAG_STRUCT;
    v.as.sval = s;
    return v;
}

static inline NanoValue val_enum(int32_t variant) {
    NanoValue v = {0};
    v.tag = TAG_ENUM;
    v.as.enum_val = variant;
    return v;
}

static inline NanoValue val_union(VmUnion *u) {
    NanoValue v = {0};
    v.tag = TAG_UNION;
    v.as.uval = u;
    return v;
}

static inline NanoValue val_function(uint32_t fn_idx) {
    NanoValue v = {0};
    v.tag = TAG_FUNCTION;
    v.as.fn_idx = fn_idx;
    return v;
}

static inline NanoValue val_tuple(VmTuple *t) {
    NanoValue v = {0};
    v.tag = TAG_TUPLE;
    v.as.tuple = t;
    return v;
}

static inline NanoValue val_hashmap(VmHashMap *h) {
    NanoValue v = {0};
    v.tag = TAG_HASHMAP;
    v.as.hashmap = h;
    return v;
}

static inline NanoValue val_closure(VmClosure *c) {
    NanoValue v = {0};
    v.tag = TAG_FUNCTION;
    v.as.closure = c;
    return v;
}

/* ========================================================================
 * Type checking
 * ======================================================================== */

static inline bool val_is_int(NanoValue v)      { return v.tag == TAG_INT; }
static inline bool val_is_float(NanoValue v)    { return v.tag == TAG_FLOAT; }
static inline bool val_is_bool(NanoValue v)     { return v.tag == TAG_BOOL; }
static inline bool val_is_string(NanoValue v)   { return v.tag == TAG_STRING; }
static inline bool val_is_array(NanoValue v)    { return v.tag == TAG_ARRAY; }
static inline bool val_is_struct(NanoValue v)   { return v.tag == TAG_STRUCT; }
static inline bool val_is_union(NanoValue v)    { return v.tag == TAG_UNION; }
static inline bool val_is_tuple(NanoValue v)    { return v.tag == TAG_TUPLE; }
static inline bool val_is_hashmap(NanoValue v)  { return v.tag == TAG_HASHMAP; }
static inline bool val_is_function(NanoValue v) { return v.tag == TAG_FUNCTION; }
static inline bool val_is_void(NanoValue v)     { return v.tag == TAG_VOID; }
static inline bool val_is_heap_obj(NanoValue v) {
    return v.tag == TAG_STRING || v.tag == TAG_ARRAY || v.tag == TAG_STRUCT ||
           v.tag == TAG_UNION || v.tag == TAG_TUPLE || v.tag == TAG_HASHMAP;
}

/* ========================================================================
 * Value operations (implemented in value.c)
 * ======================================================================== */

/* Print a value to a file stream */
void val_print(NanoValue v, FILE *out);

/* Print a value to stdout followed by newline */
void val_println(NanoValue v);

/* Compare two values for equality. Returns true if equal. */
bool val_equal(NanoValue a, NanoValue b);

/* Compare two values for ordering. Returns <0, 0, or >0. */
int val_compare(NanoValue a, NanoValue b);

/* Convert a value to boolean (truthiness) */
bool val_truthy(NanoValue v);

/* Convert a value to a string representation (heap-allocated, caller frees) */
char *val_to_cstring(NanoValue v);

#endif /* NANOVM_VALUE_H */
