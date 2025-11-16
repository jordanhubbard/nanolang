/*
 * GC-Managed Struct Implementation
 */

#include "gc_struct.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* Create new GC struct */
GCStruct* gc_struct_new(const char* struct_name, int field_count) {
    GCStruct* s = (GCStruct*)gc_alloc(sizeof(GCStruct), GC_TYPE_STRUCT);
    if (!s) {
        fprintf(stderr, "GC: Failed to allocate struct '%s'\n", struct_name);
        return NULL;
    }
    
    s->struct_name = strdup(struct_name);
    s->field_count = field_count;
    s->field_names = (char**)calloc(field_count, sizeof(char*));
    s->field_values = (void**)calloc(field_count, sizeof(void*));
    s->field_gc_flags = (uint8_t*)calloc(field_count, sizeof(uint8_t));
    s->field_types = (uint8_t*)calloc(field_count, sizeof(uint8_t));
    
    if (!s->field_names || !s->field_values || !s->field_gc_flags || !s->field_types) {
        fprintf(stderr, "GC: Failed to allocate struct fields\n");
        gc_struct_free(s);
        return NULL;
    }
    
    return s;
}

/* Free GC struct (called by GC) */
void gc_struct_free(GCStruct* s) {
    if (!s) return;
    
    /* Release all GC object fields */
    for (int i = 0; i < s->field_count; i++) {
        if (s->field_gc_flags[i] && s->field_values[i]) {
            gc_release(s->field_values[i]);
        }
        if (s->field_names[i]) {
            free(s->field_names[i]);
        }
    }
    
    /* Free arrays */
    if (s->struct_name) free(s->struct_name);
    if (s->field_names) free(s->field_names);
    if (s->field_values) free(s->field_values);
    if (s->field_gc_flags) free(s->field_gc_flags);
    if (s->field_types) free(s->field_types);
}

/* Set field value */
void gc_struct_set_field(GCStruct* s, int field_index, 
                         const char* field_name, void* value,
                         FieldType type, bool is_gc_object) {
    assert(s != NULL && "GC: NULL struct");
    assert(field_index >= 0 && field_index < s->field_count && "GC: Field index out of bounds");
    
    /* If old field was a GC object, release it */
    if (s->field_gc_flags[field_index] && s->field_values[field_index]) {
        gc_release(s->field_values[field_index]);
    }
    
    /* Set field name if not already set */
    if (!s->field_names[field_index]) {
        s->field_names[field_index] = strdup(field_name);
    }
    
    /* Set new field value */
    s->field_values[field_index] = value;
    s->field_gc_flags[field_index] = is_gc_object ? 1 : 0;
    s->field_types[field_index] = (uint8_t)type;
    
    /* If new field is a GC object, retain it */
    if (is_gc_object && value) {
        gc_retain(value);
    }
}

/* Get field value */
void* gc_struct_get_field(GCStruct* s, int field_index) {
    assert(s != NULL && "GC: NULL struct");
    if (field_index < 0 || field_index >= s->field_count) {
        fprintf(stderr, "GC: Field index %d out of bounds [0..%d)\n", 
                field_index, s->field_count);
        return NULL;
    }
    return s->field_values[field_index];
}

/* Get field index by name */
int gc_struct_get_field_index(GCStruct* s, const char* field_name) {
    assert(s != NULL && "GC: NULL struct");
    assert(field_name != NULL && "GC: NULL field name");
    
    for (int i = 0; i < s->field_count; i++) {
        if (s->field_names[i] && strcmp(s->field_names[i], field_name) == 0) {
            return i;
        }
    }
    return -1;
}

/* Get field by name */
void* gc_struct_get_field_by_name(GCStruct* s, const char* field_name) {
    int index = gc_struct_get_field_index(s, field_name);
    if (index < 0) {
        fprintf(stderr, "GC: Field '%s' not found in struct '%s'\n", 
                field_name, s->struct_name);
        return NULL;
    }
    return gc_struct_get_field(s, index);
}

/* Clone struct (deep copy) */
GCStruct* gc_struct_clone(GCStruct* s) {
    assert(s != NULL && "GC: NULL struct");
    
    GCStruct* clone = gc_struct_new(s->struct_name, s->field_count);
    if (!clone) return NULL;
    
    /* Copy all fields */
    for (int i = 0; i < s->field_count; i++) {
        /* For GC objects, we share the reference (increment ref count) */
        /* For primitives, we copy the value */
        bool is_gc = s->field_gc_flags[i];
        FieldType type = (FieldType)s->field_types[i];
        
        if (is_gc && s->field_values[i]) {
            /* Share GC object (retain increases ref count) */
            gc_struct_set_field(clone, i, s->field_names[i], 
                              s->field_values[i], type, true);
        } else {
            /* Copy primitive value directly */
            clone->field_names[i] = strdup(s->field_names[i]);
            clone->field_values[i] = s->field_values[i];
            clone->field_gc_flags[i] = 0;
            clone->field_types[i] = s->field_types[i];
        }
    }
    
    return clone;
}

