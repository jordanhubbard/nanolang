#ifndef NANO_EVAL_CARRIER_HEAP_H
#define NANO_EVAL_CARRIER_HEAP_H

#include <stdbool.h>
#include <stddef.h>

/* I own only explicitly registered allocations. My adapter supplies actual
 * edges; an integer or an unknown pointer never grants ownership. */
typedef struct NlCarrierHeap NlCarrierHeap;
typedef bool (*NlCarrierVisit)(const void *object, void *visitor);
typedef bool (*NlCarrierTrace)(const void *object, NlCarrierVisit visit,
                               void *visitor, void *context);
typedef void (*NlCarrierDestroy)(void *object, void *context);

typedef struct {
    size_t registered;
    size_t destroyed;
    size_t transferred;
    size_t live;
} NlCarrierHeapStats;

NlCarrierHeap *nl_carrier_heap_new(void);
/* I reject mutation/reentrant destruction during tracing or destruction. */
bool nl_carrier_heap_free(NlCarrierHeap *heap);
bool nl_carrier_heap_clear(NlCarrierHeap *heap);
/* A successful registration transfers destruction responsibility to me.
 * Failure (including duplicate/unknown destruction policy) transfers none. */
bool nl_carrier_heap_track(NlCarrierHeap *heap, void *object,
                           NlCarrierTrace trace, NlCarrierDestroy destroy,
                           void *context);
bool nl_carrier_heap_contains(const NlCarrierHeap *heap, const void *object);
/* I expose borrowed adapter metadata only until the next mutation. */
void *nl_carrier_heap_context(const NlCarrierHeap *heap, const void *object);
/* I transfer one allocation out without destroying it. The adapter must
 * explicitly transfer every owned descendant crossing its API boundary. */
bool nl_carrier_heap_forget(NlCarrierHeap *heap, const void *object);
/* I destroy exactly one explicitly owned allocation immediately. Unknown
 * pointers and reentrant calls fail without changing ownership or counters. */
bool nl_carrier_heap_destroy(NlCarrierHeap *heap, const void *object);
/* I follow registered objects iteratively, preserve their identity, then
 * destroy each unreachable allocation once. Destructors retire only their
 * own storage/scalars; they must not recursively destroy registered children.
 * My visitor returns false for unregistered objects. An adapter following
 * legacy borrowed containers must inspect their valid edges separately.
 * A tracer returning false aborts before any destruction; all allocations
 * remain owned. Trace/destructor callbacks cannot mutate this heap or start collection.
 * This is a per-interpreter engine, not a thread-safe public allocator. */
bool nl_carrier_heap_collect(NlCarrierHeap *heap,
                             const void *const *roots, size_t root_count);
NlCarrierHeapStats nl_carrier_heap_stats(const NlCarrierHeap *heap);

#endif
