#ifndef NANO_EVAL_CARRIER_VALUES_H
#define NANO_EVAL_CARRIER_VALUES_H
#include "../nanolang.h"
#include "carrier_heap.h"
#include "../runtime/list_int.h"

typedef struct NlCarrierValues NlCarrierValues;
enum { NL_CARRIER_RECORD_LIST = 1, NL_CARRIER_LEGACY_RECORD_LIST = 2 };
typedef struct {
    Value value;
    /* I keep this one-byte private fact: 0 ordinary, 1 DynArray/checked
     * logical list, 2 explicitly identified legacy List_int record storage. */
    unsigned char record_list;
} NlCarrierRef;

NlCarrierRef nl_carrier_value(Value value);
NlCarrierRef nl_carrier_record_list(DynArray *list);
NlCarrierRef nl_carrier_legacy_record_list(List_int *list);
/* My source adapter may identify typed slots from checked metadata. This
 * borrowed, read-only callback grants edges, never ownership of an INT. */
typedef bool (*NlCarrierRecordListSlot)(Value owner, int index, void *context);
void nl_carrier_values_set_slots(NlCarrierValues *heap, NlCarrierRecordListSlot slots, void *context);
NlCarrierValues *nl_carrier_values_new(void);
bool nl_carrier_values_free(NlCarrierValues *heap);
bool nl_carrier_values_clear(NlCarrierValues *heap);
bool nl_carrier_values_contains(const NlCarrierValues *heap, NlCarrierRef value);
/* I adopt exactly one header and its owned STRING/FUNCTION scalar storage.
 * Aggregate children remain separately owned. A list is explicitly identified
 * as either managed ELEM_INT DynArray or malloc-owned List_int record storage;
 * I do not own its referenced records implicitly.
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
size_t nl_carrier_values_reclaimed_arrays(void);
#endif
