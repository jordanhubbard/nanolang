#ifndef NANO_EVAL_CARRIER_VALUES_H
#define NANO_EVAL_CARRIER_VALUES_H
#include "../nanolang.h"
#include "carrier_heap.h"

typedef struct NlCarrierValues NlCarrierValues;
typedef struct {
    Value value;
    bool record_list; /* I require a producer-supplied typed-list fact. */
} NlCarrierRef;

NlCarrierRef nl_carrier_value(Value value);
NlCarrierRef nl_carrier_record_list(DynArray *list);
NlCarrierValues *nl_carrier_values_new(void);
bool nl_carrier_values_free(NlCarrierValues *heap);
bool nl_carrier_values_clear(NlCarrierValues *heap);
bool nl_carrier_values_contains(const NlCarrierValues *heap, NlCarrierRef value);
/* I adopt exactly one header and its owned STRING/FUNCTION scalar storage.
 * Aggregate children remain separately owned. A list must be a managed
 * ELEM_INT record-pointer list; I do not own its referenced records implicitly.
 * I accept only explicit owned storage, never an arbitrary borrowed C value. */
bool nl_carrier_values_adopt(NlCarrierValues *heap, NlCarrierRef value);
/* I copy explicit additional root facts, for fields whose public INT storage
 * cannot carry typed-list provenance. Reaching a child grants no ownership. */
bool nl_carrier_values_edges(NlCarrierValues *heap, NlCarrierRef owner,
                             const NlCarrierRef *edges, size_t count);
/* I trace both registered values and valid legacy borrowed containers. */
bool nl_carrier_values_collect(NlCarrierValues *heap,
                               const NlCarrierRef *roots, size_t count);
/* I gather the complete reachable graph before transferring registered
 * allocations. Failure transfers none. Caller storage is never adopted. */
bool nl_carrier_values_transfer(NlCarrierValues *heap,
                                const NlCarrierRef *roots, size_t count);
NlCarrierHeapStats nl_carrier_values_stats(const NlCarrierValues *heap);
#endif
