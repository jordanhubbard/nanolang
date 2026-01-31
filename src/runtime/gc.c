/*
 * Garbage Collector Implementation
 * Reference counting with optional cycle detection
 */

#include "gc.h"
#include "dyn_array.h"
#include "gc_struct.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* GC State */
static struct {
    GCHeader* all_objects;      /* Linked list of all allocated objects */
    GCStats stats;              /* GC statistics */
    size_t threshold;           /* Memory threshold for auto-collection */
    bool cycle_detection_enabled;
    bool initialized;
} gc_state = {
    .all_objects = NULL,
    .stats = {0},
    .threshold = 10 * 1024 * 1024,  /* 10MB default */
    .cycle_detection_enabled = true,
    .initialized = false
};

static void gc_destroy_object(GCHeader *header, bool release_children) {
    if (!header) return;
    void *obj = gc_header_to_ptr(header);
    switch (header->type) {
        case GC_TYPE_ARRAY: {
            DynArray *arr = (DynArray *)obj;
            if (arr && arr->data) {
                free(arr->data);
                arr->data = NULL;
            }
            break;
        }
        case GC_TYPE_STRUCT: {
            GCStruct *s = (GCStruct *)obj;
            if (!s) break;

            if (release_children) {
                gc_struct_free(s);
            } else {
                for (int i = 0; i < s->field_count; i++) {
                    if (s->field_names && s->field_names[i]) {
                        free(s->field_names[i]);
                    }
                }
                if (s->struct_name) free(s->struct_name);
                if (s->field_names) free(s->field_names);
                if (s->field_values) free(s->field_values);
                if (s->field_gc_flags) free(s->field_gc_flags);
                if (s->field_types) free(s->field_types);
            }
            break;
        }
        default:
            break;
    }
}

/* Initialize GC */
void gc_init(void) {
    if (gc_state.initialized) {
        return;
    }
    
    memset(&gc_state.stats, 0, sizeof(GCStats));
    gc_state.all_objects = NULL;
    gc_state.threshold = 10 * 1024 * 1024;
    gc_state.cycle_detection_enabled = true;
    gc_state.initialized = true;
}

/* Shutdown GC */
void gc_shutdown(void) {
    if (!gc_state.initialized) {
        return;
    }
    
    /* Free all remaining objects */
    GCHeader* current = gc_state.all_objects;
    while (current != NULL) {
        GCHeader* next = (GCHeader*)current->next;
        gc_destroy_object(current, false);
        free(current);
        current = next;
    }
    
    gc_state.all_objects = NULL;
    gc_state.initialized = false;
}

/* Allocate GC-managed object */
void* gc_alloc(size_t size, GCObjectType type) {
    if (!gc_state.initialized) {
        gc_init();
    }
    
    /* Allocate space for header + object */
    size_t total_size = sizeof(GCHeader) + size;
    GCHeader* header = (GCHeader*)malloc(total_size);
    
    if (header == NULL) {
        fprintf(stderr, "GC: Out of memory (requested %zu bytes)\n", total_size);
        return NULL;
    }
    
    /* Initialize header */
    header->ref_count = 1;      /* Start with ref count 1 */
    header->type = type;
    header->marked = 0;
    header->flags = 0;
    header->size = total_size;
    
    /* Add to linked list */
    header->next = gc_state.all_objects;
    gc_state.all_objects = header;
    
    /* Update statistics */
    gc_state.stats.total_allocated += total_size;
    gc_state.stats.current_usage += total_size;
    gc_state.stats.num_objects++;
    
    /* Check if we should trigger collection */
    if (gc_state.stats.current_usage > gc_state.threshold) {
        if (gc_state.cycle_detection_enabled) {
            gc_collect_cycles();
        }
    }
    
    /* Return pointer to object (after header) */
    return gc_header_to_ptr(header);
}

/* Increment reference count */
void gc_retain(void* ptr) {
    if (ptr == NULL) {
        return;
    }
    
    GCHeader* header = gc_get_header(ptr);
    header->ref_count++;
}

/* Decrement reference count and potentially free */
void gc_release(void* ptr) {
    if (ptr == NULL) {
        return;
    }
    
    GCHeader* header = gc_get_header(ptr);
    
    assert(header->ref_count > 0 && "GC: Double free detected!");
    
    header->ref_count--;
    
    /* If ref count reaches zero, free immediately */
    if (header->ref_count == 0) {
        /* Remove from linked list */
        if (gc_state.all_objects == header) {
            gc_state.all_objects = (GCHeader*)header->next;
        } else {
            GCHeader* current = gc_state.all_objects;
            while (current != NULL && current->next != header) {
                current = (GCHeader*)current->next;
            }
            if (current != NULL) {
                current->next = header->next;
            }
        }
        
        /* Update statistics */
        gc_state.stats.total_freed += header->size;
        gc_state.stats.current_usage -= header->size;
        gc_state.stats.num_objects--;
        
        gc_destroy_object(header, true);
        
        /* Free the object */
        free(header);
    }
}

/* Mark phase of mark-and-sweep */
static void gc_mark(GCHeader* header) {
    if (header == NULL || header->marked) {
        return;
    }
    
    header->marked = 1;
    
    /* Get object pointer */
    void* obj = gc_header_to_ptr(header);
    
    /* Recursively mark nested GC objects */
    switch (header->type) {
        case GC_TYPE_ARRAY: {
            DynArray* arr = (DynArray*)obj;
            ElementType elem_type = dyn_array_get_elem_type(arr);
            
            /* If array contains GC objects (arrays or structs), mark them */
            /* Note: Arrays of GC objects store pointers to those objects */
            if (elem_type == ELEM_ARRAY || elem_type == ELEM_STRUCT) {
                int64_t len = dyn_array_length(arr);
                /* For object arrays, data is an array of pointers */
                void** ptr_data = (void**)arr->data;
                for (int64_t i = 0; i < len; i++) {
                    void* elem = ptr_data[i];
                    if (elem && gc_is_managed(elem)) {
                        gc_mark(gc_get_header(elem));
                    }
                }
            }
            break;
        }
        
        case GC_TYPE_STRUCT: {
            GCStruct* s = (GCStruct*)obj;
            /* Mark all GC object fields */
            for (int i = 0; i < s->field_count; i++) {
                if (s->field_gc_flags[i] && s->field_values[i]) {
                    if (gc_is_managed(s->field_values[i])) {
                        gc_mark(gc_get_header(s->field_values[i]));
                    }
                }
            }
            break;
        }
        
        case GC_TYPE_STRING:
        case GC_TYPE_CLOSURE:
        default:
            /* These types don't have nested objects */
            break;
    }
}

/* Sweep phase of mark-and-sweep */
static void gc_sweep(void) {
    GCHeader** current_ptr = &gc_state.all_objects;
    
    while (*current_ptr != NULL) {
        GCHeader* header = *current_ptr;
        
        if (!header->marked && header->ref_count == 0) {
            /* Unreachable object - free it */
            *current_ptr = (GCHeader*)header->next;
            
            gc_state.stats.total_freed += header->size;
            gc_state.stats.current_usage -= header->size;
            gc_state.stats.num_objects--;
            
            gc_destroy_object(header, true);
            free(header);
        } else {
            /* Clear mark for next collection */
            header->marked = 0;
            current_ptr = (GCHeader**)&header->next;
        }
    }
}

/* Run cycle detection */
void gc_collect_cycles(void) {
    if (!gc_state.initialized || !gc_state.cycle_detection_enabled) {
        return;
    }
    
    /* Mark phase: mark all objects reachable from roots */
    /* In a simple implementation, objects with ref_count > 0 are roots */
    GCHeader* current = gc_state.all_objects;
    while (current != NULL) {
        if (current->ref_count > 0) {
            gc_mark(current);
        }
        current = (GCHeader*)current->next;
    }
    
    /* Sweep phase: free unmarked objects */
    gc_sweep();
    
    gc_state.stats.num_collections++;
}

/* Force immediate collection */
void gc_collect_all(void) {
    gc_collect_cycles();
}

/* Get GC statistics */
GCStats gc_get_stats(void) {
    return gc_state.stats;
}

/* Print GC statistics */
void gc_print_stats(void) {
    printf("=== GC Statistics ===\n");
    printf("Total allocated: %zu bytes\n", gc_state.stats.total_allocated);
    printf("Total freed:     %zu bytes\n", gc_state.stats.total_freed);
    printf("Current usage:   %zu bytes\n", gc_state.stats.current_usage);
    printf("Live objects:    %zu\n", gc_state.stats.num_objects);
    printf("Collections:     %zu\n", gc_state.stats.num_collections);
    printf("====================\n");
}

/* Check if pointer is GC-managed */
bool gc_is_managed(void* ptr) {
    if (ptr == NULL || !gc_state.initialized) {
        return false;
    }
    
    /* Search in linked list */
    GCHeader* current = gc_state.all_objects;
    while (current != NULL) {
        if (gc_header_to_ptr(current) == ptr) {
            return true;
        }
        current = (GCHeader*)current->next;
    }
    
    return false;
}

/* Enable/disable cycle detection */
void gc_set_cycle_detection_enabled(bool enabled) {
    gc_state.cycle_detection_enabled = enabled;
}

/* Set GC threshold */
void gc_set_threshold(size_t bytes) {
    gc_state.threshold = bytes;
}

