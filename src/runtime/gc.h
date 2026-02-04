/*
 * Garbage Collector for nanolang
 * 
 * Reference counting GC with optional cycle detection
 * Manages dynamic arrays, strings, and future heap objects
 * 
 * OPTIMIZATION (2026-02): 
 * - Hash table for O(1) gc_is_managed() lookup (was O(n))
 * - Doubly-linked list for O(1) object removal (was O(n))
 * - Expected improvement: 46% overhead -> 2-3% overhead
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
    GC_TYPE_CLOSURE = 4,
    GC_TYPE_OPAQUE = 5  /* Opaque handles (regex, hashmap, etc.) with finalizers */
} GCObjectType;

/* Finalizer function type - called when object is freed */
typedef void (*GCFinalizer)(void* object);

/* GC Header (prepended to all GC-managed objects) */
typedef struct GCHeader {
    uint32_t ref_count;      /* Reference count */
    uint8_t type;            /* Object type (GCObjectType) */
    uint8_t marked;          /* Mark bit for cycle collection */
    uint16_t flags;          /* Additional flags */
    size_t size;             /* Object size in bytes (including header) */
    struct GCHeader* next;   /* Next object in allocation list */
    struct GCHeader* prev;   /* Previous object for O(1) removal */
    GCFinalizer finalizer;   /* Optional cleanup function (NULL if none) */
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

/* Allocate GC-managed opaque object with finalizer */
void* gc_alloc_opaque(size_t size, GCFinalizer finalizer);

#endif /* NANOLANG_GC_H */

