/*
 * Garbage Collector Implementation
 * Reference counting with optional cycle detection
 * 
 * OPTIMIZATION (2026-02):
 * - Hash table for O(1) gc_is_managed() lookup
 * - Doubly-linked list for O(1) object removal
 */

#include "gc.h"
#include "dyn_array.h"
#include "gc_struct.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ============================================================================
 * Hash Table for O(1) pointer lookup
 * ============================================================================ */

#define GC_HASH_SIZE 16384  /* Power of 2 for fast modulo */
#define GC_HASH_MASK (GC_HASH_SIZE - 1)

/* Hash bucket entry */
typedef struct GCHashEntry {
    void* ptr;
    struct GCHashEntry* next;
} GCHashEntry;

/* Hash function for pointers (FNV-1a inspired) */
static inline size_t gc_hash_ptr(void* ptr) {
    size_t h = (size_t)ptr;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h & GC_HASH_MASK;
}

/* GC State */
static struct {
    GCHeader* all_objects;      /* Doubly-linked list of all allocated objects */
    GCHashEntry* hash_table[GC_HASH_SIZE];  /* Hash set for O(1) lookup */
    GCStats stats;              /* GC statistics */
    size_t threshold;           /* Memory threshold for auto-collection */
    size_t last_collection_usage; /* Usage at last collection (hysteresis) */
    bool cycle_detection_enabled;
    bool initialized;
} gc_state = {
    .all_objects = NULL,
    .stats = {0},
    .threshold = 10 * 1024 * 1024,  /* 10MB default */
    .last_collection_usage = 0,
    .cycle_detection_enabled = true,
    .initialized = false
};

/* ============================================================================
 * Hash Table Operations - O(1) average
 * ============================================================================ */

/* Add pointer to hash table */
static void gc_hash_add(void* ptr) {
    size_t idx = gc_hash_ptr(ptr);
    GCHashEntry* entry = (GCHashEntry*)malloc(sizeof(GCHashEntry));
    if (entry) {
        entry->ptr = ptr;
        entry->next = gc_state.hash_table[idx];
        gc_state.hash_table[idx] = entry;
    }
}

/* Remove pointer from hash table */
static void gc_hash_remove(void* ptr) {
    size_t idx = gc_hash_ptr(ptr);
    GCHashEntry** current = &gc_state.hash_table[idx];
    
    while (*current != NULL) {
        if ((*current)->ptr == ptr) {
            GCHashEntry* to_free = *current;
            *current = (*current)->next;
            free(to_free);
            return;
        }
        current = &(*current)->next;
    }
}

/* Check if pointer is in hash table - O(1) average */
static inline bool gc_hash_contains(void* ptr) {
    size_t idx = gc_hash_ptr(ptr);
    GCHashEntry* current = gc_state.hash_table[idx];
    
    while (current != NULL) {
        if (current->ptr == ptr) {
            return true;
        }
        current = current->next;
    }
    return false;
}

/* Clear entire hash table */
static void gc_hash_clear(void) {
    for (size_t i = 0; i < GC_HASH_SIZE; i++) {
        GCHashEntry* current = gc_state.hash_table[i];
        while (current != NULL) {
            GCHashEntry* next = current->next;
            free(current);
            current = next;
        }
        gc_state.hash_table[i] = NULL;
    }
}

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
    memset(gc_state.hash_table, 0, sizeof(gc_state.hash_table));
    gc_state.all_objects = NULL;
    gc_state.threshold = 10 * 1024 * 1024;
    gc_state.last_collection_usage = 0;
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
        GCHeader* next = current->next;
        gc_destroy_object(current, false);
        free(current);
        current = next;
    }
    
    /* Clear hash table */
    gc_hash_clear();
    
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
    
    /* Add to doubly-linked list - O(1) */
    header->prev = NULL;
    header->next = gc_state.all_objects;
    if (gc_state.all_objects != NULL) {
        gc_state.all_objects->prev = header;
    }
    gc_state.all_objects = header;
    
    /* Get pointer to object */
    void* ptr = gc_header_to_ptr(header);
    
    /* Add to hash table for O(1) lookup */
    gc_hash_add(ptr);
    
    /* Update statistics */
    gc_state.stats.total_allocated += total_size;
    gc_state.stats.current_usage += total_size;
    gc_state.stats.num_objects++;
    
    /* Check if we should trigger collection
     * Use hysteresis: only collect if we've allocated at least 1MB MORE
     * since the last collection. This prevents collecting on every allocation
     * when memory usage hovers around the threshold.
     */
    if (gc_state.stats.current_usage > gc_state.threshold &&
        gc_state.stats.current_usage > gc_state.last_collection_usage + (1024 * 1024)) {
        if (gc_state.cycle_detection_enabled) {
            gc_state.last_collection_usage = gc_state.stats.current_usage;
            gc_collect_cycles();
        }
    }
    
    /* Return pointer to object (after header) */
    return ptr;
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
        /* Remove from doubly-linked list - O(1) */
        if (header->prev != NULL) {
            header->prev->next = header->next;
        } else {
            gc_state.all_objects = header->next;
        }
        if (header->next != NULL) {
            header->next->prev = header->prev;
        }
        
        /* Remove from hash table - O(1) average */
        gc_hash_remove(ptr);
        
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
    GCHeader* current = gc_state.all_objects;
    
    while (current != NULL) {
        GCHeader* next = current->next;
        
        if (!current->marked && current->ref_count == 0) {
            /* Unreachable object - free it */
            
            /* Remove from doubly-linked list - O(1) */
            if (current->prev != NULL) {
                current->prev->next = current->next;
            } else {
                gc_state.all_objects = current->next;
            }
            if (current->next != NULL) {
                current->next->prev = current->prev;
            }
            
            /* Remove from hash table - O(1) average */
            gc_hash_remove(gc_header_to_ptr(current));
            
            gc_state.stats.total_freed += current->size;
            gc_state.stats.current_usage -= current->size;
            gc_state.stats.num_objects--;
            
            gc_destroy_object(current, true);
            free(current);
        } else {
            /* Clear mark for next collection */
            current->marked = 0;
        }
        
        current = next;
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
        current = current->next;
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

/* Check if pointer is GC-managed - O(1) via hash table */
bool gc_is_managed(void* ptr) {
    if (ptr == NULL || !gc_state.initialized) {
        return false;
    }
    
    /* O(1) hash table lookup instead of O(n) linked list search */
    return gc_hash_contains(ptr);
}

/* Enable/disable cycle detection */
void gc_set_cycle_detection_enabled(bool enabled) {
    gc_state.cycle_detection_enabled = enabled;
}

/* Set GC threshold */
void gc_set_threshold(size_t bytes) {
    gc_state.threshold = bytes;
}

/* Allocate GC-managed string */
char* gc_alloc_string(size_t length) {
    /* Allocate space for the string content + null terminator */
    char* str = (char*)gc_alloc(length + 1, GC_TYPE_STRING);
    if (str) {
        str[length] = '\0';  /* Ensure null termination */
    }
    return str;
}

