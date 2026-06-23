/*
 * GC-Managed Struct Implementation
 * 
 * Provides heap-allocated structs with automatic memory management
 */

#ifndef NANOLANG_GC_STRUCT_H
#define NANOLANG_GC_STRUCT_H

#include "gc.h"
#include <stdint.h>
#include <stdbool.h>

/* GC-managed struct */
typedef struct {
    char* struct_name;       /* Type name (e.g., "Player", "Enemy") */
    int field_count;         /* Number of fields */
    char** field_names;      /* Field names */
    void** field_values;     /* Field values (can be primitive or GC objects) */
    uint8_t* field_gc_flags; /* Which fields are GC objects (need retain/release) */
    uint8_t* field_types;    /* Field types (for type checking) */
} GCStruct;

/* Field type enumeration */
typedef enum {
    FIELD_INT = 1,
    FIELD_FLOAT = 2,
    FIELD_BOOL = 3,
    FIELD_STRING = 4,
    FIELD_ARRAY = 5,
    FIELD_STRUCT = 6,
    FIELD_UNION = 7
} FieldType;

/* Create new GC struct */
GCStruct* gc_struct_new(const char* struct_name, int field_count);

/* Free GC struct (called by GC when ref_count hits 0) */
void gc_struct_free(GCStruct* s);

/* Set field value (handles retain/release automatically) */
void gc_struct_set_field(GCStruct* s, int field_index, 
                         const char* field_name, void* value, 
                         FieldType type, bool is_gc_object);

/* Get field value */
void* gc_struct_get_field(GCStruct* s, int field_index);

/* Get field by name */
void* gc_struct_get_field_by_name(GCStruct* s, const char* field_name);

/* Get field index by name */
int gc_struct_get_field_index(GCStruct* s, const char* field_name);

/* Clone struct (deep copy) */
GCStruct* gc_struct_clone(GCStruct* s);

#endif /* NANOLANG_GC_STRUCT_H */

