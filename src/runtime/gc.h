/*
 * Garbage Collector for nanolang
 * 
 * Simple reference counting GC with optional cycle detection
 * Manages dynamic arrays, strings, and future heap objects
 */

#ifndef NANOLANG_GC_H
#define NANOLANG_GC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Object types tracked by GC */
typedef enum {
    GC_TYPE_ARRAY = 1,
    GC_TYPE_STRING = 2,
    GC_TYPE_STRUCT = 3,
    GC_TYPE_CLOSURE = 4
} GCObjectType;

/* GC Header (prepended to all GC-managed objects) */
typedef struct {
    uint32_t ref_count;      /* Reference count */
    uint8_t type;            /* Object type (GCObjectType) */
    uint8_t marked;          /* Mark bit for cycle collection */
    uint16_t flags;          /* Additional flags */
    size_t size;             /* Object size in bytes (including header) */
    void* next;              /* Next object in allocation list */
} GCHeader;

/* GC Statistics */
typedef struct {
    size_t total_allocated;   /* Total bytes allocated */
    size_t total_freed;        /* Total bytes freed */
    size_t current_usage;      /* Current memory usage */
    size_t num_objects;        /* Number of live objects */
    size_t num_collections;    /* Number of GC cycles run */
} GCStats;

/* Initialize GC system */
void gc_init(void);

/* Shutdown GC system (free all remaining objects) */
void gc_shutdown(void);

/* Allocate GC-managed object */
void* gc_alloc(size_t size, GCObjectType type);

/* Increment reference count */
void gc_retain(void* ptr);

/* Decrement reference count (may free object) */
void gc_release(void* ptr);

/* Run cycle detection (mark-and-sweep) */
void gc_collect_cycles(void);

/* Force immediate collection (for testing/debugging) */
void gc_collect_all(void);

/* Get GC statistics */
GCStats gc_get_stats(void);

/* Print GC statistics (for debugging) */
void gc_print_stats(void);

/* Get header from object pointer */
static inline GCHeader* gc_get_header(void* ptr) {
    return (GCHeader*)((char*)ptr - sizeof(GCHeader));
}

/* Get object pointer from header */
static inline void* gc_header_to_ptr(GCHeader* header) {
    return (void*)((char*)header + sizeof(GCHeader));
}

/* Check if pointer is GC-managed (for safety) */
bool gc_is_managed(void* ptr);

/* Enable/disable cycle detection */
void gc_set_cycle_detection_enabled(bool enabled);

/* Set GC threshold (trigger collection when memory usage exceeds this) */
void gc_set_threshold(size_t bytes);

/* Allocate GC-managed string */
char* gc_alloc_string(size_t length);

#endif /* NANOLANG_GC_H */

