/*
 * Dynamic Arrays for nanolang
 * GC-managed variable-length arrays
 */

#ifndef NANOLANG_DYN_ARRAY_H
#define NANOLANG_DYN_ARRAY_H

#include "gc.h"
#include <stdint.h>
#include <stdbool.h>

/* Element type enum (matches nanolang Value types) */
typedef enum {
    ELEM_INT = 1,
    ELEM_FLOAT = 2,
    ELEM_STRING = 3,
    ELEM_BOOL = 4,
    ELEM_ARRAY = 5,     /* Arrays (for nested arrays) */
    ELEM_STRUCT = 6,    /* GC Structs (for arrays of structs) */
    ELEM_POINTER = 7    /* Generic pointer (for GC objects) */
} ElementType;

/* Dynamic array structure */
typedef struct {
    int64_t length;        /* Current number of elements */
    int64_t capacity;      /* Allocated capacity */
    ElementType elem_type; /* Element type */
    uint8_t elem_size;     /* Size of each element in bytes */
    void* data;            /* Element storage */
} DynArray;

/* Create new empty dynamic array */
DynArray* dyn_array_new(ElementType elem_type);

/* Push element to end of array */
DynArray* dyn_array_push_int(DynArray* arr, int64_t value);
DynArray* dyn_array_push_float(DynArray* arr, double value);
DynArray* dyn_array_push_bool(DynArray* arr, bool value);
DynArray* dyn_array_push_string(DynArray* arr, const char* value);
DynArray* dyn_array_push_array(DynArray* arr, DynArray* value);  /* For nested arrays */

/* Pop element from end of array */
int64_t dyn_array_pop_int(DynArray* arr, bool* success);
double dyn_array_pop_float(DynArray* arr, bool* success);
bool dyn_array_pop_bool(DynArray* arr, bool* success);
const char* dyn_array_pop_string(DynArray* arr, bool* success);
DynArray* dyn_array_pop_array(DynArray* arr, bool* success);  /* For nested arrays */

/* Get element at index */
int64_t dyn_array_get_int(DynArray* arr, int64_t index);
double dyn_array_get_float(DynArray* arr, int64_t index);
bool dyn_array_get_bool(DynArray* arr, int64_t index);
const char* dyn_array_get_string(DynArray* arr, int64_t index);
DynArray* dyn_array_get_array(DynArray* arr, int64_t index);  /* For nested arrays */

/* Set element at index */
void dyn_array_set_int(DynArray* arr, int64_t index, int64_t value);
void dyn_array_set_float(DynArray* arr, int64_t index, double value);
void dyn_array_set_bool(DynArray* arr, int64_t index, bool value);
void dyn_array_set_string(DynArray* arr, int64_t index, const char* value);
void dyn_array_set_array(DynArray* arr, int64_t index, DynArray* value);  /* For nested arrays */

/* Remove element at index (shifts remaining elements) */
DynArray* dyn_array_remove_at(DynArray* arr, int64_t index);

/* Insert element at index (shifts remaining elements) */
DynArray* dyn_array_insert_int(DynArray* arr, int64_t index, int64_t value);
DynArray* dyn_array_insert_float(DynArray* arr, int64_t index, double value);
DynArray* dyn_array_insert_bool(DynArray* arr, int64_t index, bool value);
DynArray* dyn_array_insert_string(DynArray* arr, int64_t index, const char* value);

/* Clear all elements (keeps capacity) */
void dyn_array_clear(DynArray* arr);

/* Get length */
int64_t dyn_array_length(DynArray* arr);

/* Get capacity */
int64_t dyn_array_capacity(DynArray* arr);

/* Get element type */
ElementType dyn_array_get_elem_type(DynArray* arr);

/* Reserve capacity (pre-allocate) */
void dyn_array_reserve(DynArray* arr, int64_t new_capacity);

/* Clone array (deep copy) */
DynArray* dyn_array_clone(DynArray* arr);

/* Struct array operations */
DynArray* dyn_array_push_struct(DynArray* arr, const void* struct_ptr, size_t struct_size);
void* dyn_array_get_struct(DynArray* arr, int64_t index);
void dyn_array_set_struct(DynArray* arr, int64_t index, const void* struct_ptr, size_t struct_size);
void dyn_array_pop_struct(DynArray* arr, void* out_struct, size_t struct_size, bool* success);

#endif /* NANOLANG_DYN_ARRAY_H */

