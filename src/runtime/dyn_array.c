/*
 * Dynamic Array Implementation
 */

#include "dyn_array.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* Dynamic array configuration */
#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Get element size for type */
static size_t get_element_size(ElementType type) {
    switch (type) {
        case ELEM_INT:    return sizeof(int64_t);
        case ELEM_FLOAT:  return sizeof(double);
        case ELEM_STRING: return sizeof(char*);
        case ELEM_BOOL:   return sizeof(bool);
        case ELEM_POINTER: return sizeof(int64_t);  /* Pointers stored as int64_t */
        default:          return sizeof(void*);
    }
}

/* Create new dynamic array */
DynArray* dyn_array_new(ElementType elem_type) {
    DynArray* arr = (DynArray*)gc_alloc(sizeof(DynArray), GC_TYPE_ARRAY);
    if (arr == NULL) {
        return NULL;
    }
    
    arr->length = 0;
    arr->capacity = INITIAL_CAPACITY;
    arr->elem_type = elem_type;
    arr->elem_size = get_element_size(elem_type);
    
    /* For structs, we don't know the size yet - delay allocation */
    if (elem_type == ELEM_STRUCT) {
        arr->elem_size = 0;  /* Will be set on first push */
        arr->data = NULL;     /* Allocate on first push */
    } else {
        arr->data = malloc(arr->capacity * arr->elem_size);
        if (arr->data == NULL) {
            gc_release(arr);
            return NULL;
        }
    }
    
    return arr;
}

/* Grow array capacity */
static void dyn_array_grow(DynArray* arr) {
    int64_t new_capacity = arr->capacity * GROWTH_FACTOR;
    void* new_data = realloc(arr->data, new_capacity * arr->elem_size);
    
    if (new_data == NULL) {
        fprintf(stderr, "DynArray: Out of memory growing array\n");
        return;
    }
    
    arr->data = new_data;
    arr->capacity = new_capacity;
}

/* Push int */
DynArray* dyn_array_push_int(DynArray* arr, int64_t value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_INT && "DynArray: Type mismatch");
    
    if (arr->length >= arr->capacity) {
        dyn_array_grow(arr);
    }
    
    ((int64_t*)arr->data)[arr->length] = value;
    arr->length++;
    
    return arr;
}

/* Push float */
DynArray* dyn_array_push_float(DynArray* arr, double value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_FLOAT && "DynArray: Type mismatch");
    
    if (arr->length >= arr->capacity) {
        dyn_array_grow(arr);
    }
    
    ((double*)arr->data)[arr->length] = value;
    arr->length++;
    
    return arr;
}

/* Push bool */
DynArray* dyn_array_push_bool(DynArray* arr, bool value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_BOOL && "DynArray: Type mismatch");
    
    if (arr->length >= arr->capacity) {
        dyn_array_grow(arr);
    }
    
    ((bool*)arr->data)[arr->length] = value;
    arr->length++;
    
    return arr;
}

/* Push string */
DynArray* dyn_array_push_string(DynArray* arr, const char* value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_STRING && "DynArray: Type mismatch");
    
    if (arr->length >= arr->capacity) {
        dyn_array_grow(arr);
    }
    
    /* Store string pointer (caller responsible for string lifetime) */
    ((const char**)arr->data)[arr->length] = value;
    arr->length++;
    
    return arr;
}

/* Pop int */
int64_t dyn_array_pop_int(DynArray* arr, bool* success) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_INT && "DynArray: Type mismatch");
    
    if (arr->length == 0) {
        if (success) *success = false;
        return 0;
    }
    
    arr->length--;
    if (success) *success = true;
    return ((int64_t*)arr->data)[arr->length];
}

/* Pop float */
double dyn_array_pop_float(DynArray* arr, bool* success) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_FLOAT && "DynArray: Type mismatch");
    
    if (arr->length == 0) {
        if (success) *success = false;
        return 0.0;
    }
    
    arr->length--;
    if (success) *success = true;
    return ((double*)arr->data)[arr->length];
}

/* Pop bool */
bool dyn_array_pop_bool(DynArray* arr, bool* success) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_BOOL && "DynArray: Type mismatch");
    
    if (arr->length == 0) {
        if (success) *success = false;
        return false;
    }
    
    arr->length--;
    if (success) *success = true;
    return ((bool*)arr->data)[arr->length];
}

/* Pop string */
const char* dyn_array_pop_string(DynArray* arr, bool* success) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_STRING && "DynArray: Type mismatch");
    
    if (arr->length == 0) {
        if (success) *success = false;
        return NULL;
    }
    
    arr->length--;
    if (success) *success = true;
    return ((const char**)arr->data)[arr->length];
}

/* Get int */
int64_t dyn_array_get_int(DynArray* arr, int64_t index) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_INT && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    return ((int64_t*)arr->data)[index];
}

/* Get float */
double dyn_array_get_float(DynArray* arr, int64_t index) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_FLOAT && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    return ((double*)arr->data)[index];
}

/* Get bool */
bool dyn_array_get_bool(DynArray* arr, int64_t index) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_BOOL && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    return ((bool*)arr->data)[index];
}

/* Get string */
const char* dyn_array_get_string(DynArray* arr, int64_t index) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_STRING && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    return ((const char**)arr->data)[index];
}

/* Set int */
void dyn_array_set_int(DynArray* arr, int64_t index, int64_t value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_INT && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    ((int64_t*)arr->data)[index] = value;
}

/* Set float */
void dyn_array_set_float(DynArray* arr, int64_t index, double value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_FLOAT && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    ((double*)arr->data)[index] = value;
}

/* Set bool */
void dyn_array_set_bool(DynArray* arr, int64_t index, bool value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_BOOL && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    ((bool*)arr->data)[index] = value;
}

/* Set string */
void dyn_array_set_string(DynArray* arr, int64_t index, const char* value) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_STRING && "DynArray: Type mismatch");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    ((const char**)arr->data)[index] = value;
}

/* Remove element at index */
DynArray* dyn_array_remove_at(DynArray* arr, int64_t index) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(index >= 0 && index < arr->length && "DynArray: Index out of bounds");
    
    /* Shift elements down */
    if (index < arr->length - 1) {
        char* data_ptr = (char*)arr->data;
        memmove(
            data_ptr + (index * arr->elem_size),
            data_ptr + ((index + 1) * arr->elem_size),
            (arr->length - index - 1) * arr->elem_size
        );
    }
    
    arr->length--;
    return arr;
}

/* Clear all elements */
void dyn_array_clear(DynArray* arr) {
    assert(arr != NULL && "DynArray: NULL array");
    arr->length = 0;
}

/* Get length */
int64_t dyn_array_length(DynArray* arr) {
    assert(arr != NULL && "DynArray: NULL array");
    return arr->length;
}

/* Get capacity */
int64_t dyn_array_capacity(DynArray* arr) {
    assert(arr != NULL && "DynArray: NULL array");
    return arr->capacity;
}

/* Get element type */
ElementType dyn_array_get_elem_type(DynArray* arr) {
    assert(arr != NULL && "DynArray: NULL array");
    return arr->elem_type;
}

/* Reserve capacity */
void dyn_array_reserve(DynArray* arr, int64_t new_capacity) {
    assert(arr != NULL && "DynArray: NULL array");
    
    if (new_capacity <= arr->capacity) {
        return;
    }
    
    void* new_data = realloc(arr->data, new_capacity * arr->elem_size);
    if (new_data == NULL) {
        fprintf(stderr, "DynArray: Out of memory reserving capacity\n");
        return;
    }
    
    arr->data = new_data;
    arr->capacity = new_capacity;
}

/* Clone array */
DynArray* dyn_array_clone(DynArray* arr) {
    assert(arr != NULL && "DynArray: NULL array");
    
    DynArray* new_arr = dyn_array_new(arr->elem_type);
    if (new_arr == NULL) {
        return NULL;
    }
    
    /* Reserve capacity and copy data */
    dyn_array_reserve(new_arr, arr->length);
    memcpy(new_arr->data, arr->data, arr->length * arr->elem_size);
    new_arr->length = arr->length;
    
    return new_arr;
}

/* Push struct - makes a copy of the struct */
DynArray* dyn_array_push_struct(DynArray* arr, const void* struct_ptr, size_t struct_size) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(struct_ptr != NULL && "DynArray: NULL struct pointer");
    assert(arr->elem_type == ELEM_STRUCT && "DynArray: Type mismatch");
    
    /* Set struct size and allocate on first push */
    if (arr->elem_size == 0) {
        arr->elem_size = struct_size;
        arr->data = malloc(arr->capacity * arr->elem_size);
        if (arr->data == NULL) {
            fprintf(stderr, "DynArray: Out of memory allocating struct array\n");
            return arr;
        }
    }
    
    assert(arr->elem_size == struct_size && "DynArray: Struct size mismatch");
    
    if (arr->length >= arr->capacity) {
        dyn_array_grow(arr);
    }
    
    /* Copy struct into array */
    void* dest = (uint8_t*)arr->data + (arr->length * arr->elem_size);
    memcpy(dest, struct_ptr, struct_size);
    arr->length++;
    
    return arr;
}

/* Get struct - returns pointer to struct in array (not a copy) */
void* dyn_array_get_struct(DynArray* arr, int64_t index) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(arr->elem_type == ELEM_STRUCT && "DynArray: Type mismatch");
    
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "DynArray: Index out of bounds: %lld (length: %lld)\n", 
                (long long)index, (long long)arr->length);
        return NULL;
    }
    
    /* Return pointer to struct in array */
    return (uint8_t*)arr->data + (index * arr->elem_size);
}

/* Set struct - copies the struct into the array */
void dyn_array_set_struct(DynArray* arr, int64_t index, const void* struct_ptr, size_t struct_size) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(struct_ptr != NULL && "DynArray: NULL struct pointer");
    assert(arr->elem_type == ELEM_STRUCT && "DynArray: Type mismatch");
    assert(arr->elem_size == struct_size && "DynArray: Struct size mismatch");
    
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "DynArray: Index out of bounds: %lld (length: %lld)\n", 
                (long long)index, (long long)arr->length);
        return;
    }
    
    /* Copy struct into array */
    void* dest = (uint8_t*)arr->data + (index * arr->elem_size);
    memcpy(dest, struct_ptr, struct_size);
}

/* Pop struct - copies the last struct into out_struct and removes it */
void dyn_array_pop_struct(DynArray* arr, void* out_struct, size_t struct_size, bool* success) {
    assert(arr != NULL && "DynArray: NULL array");
    assert(out_struct != NULL && "DynArray: NULL output pointer");
    assert(arr->elem_type == ELEM_STRUCT && "DynArray: Type mismatch");
    assert(arr->elem_size == struct_size && "DynArray: Struct size mismatch");
    
    if (arr->length == 0) {
        if (success) *success = false;
        return;
    }
    
    /* Copy last struct to output */
    void* src = (uint8_t*)arr->data + ((arr->length - 1) * arr->elem_size);
    memcpy(out_struct, src, struct_size);
    arr->length--;
    
    if (success) *success = true;
}

