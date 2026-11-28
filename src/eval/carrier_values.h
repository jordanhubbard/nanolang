#ifndef NANO_EVAL_CARRIER_VALUES_H
#define NANO_EVAL_CARRIER_VALUES_H
#include "../nanolang.h"
#include "carrier_heap.h"
#include "../runtime/list_int.h"

typedef struct NlCarrierValues NlCarrierValues;
enum { NL_CARRIER_RECORD_LIST = 1, NL_CARRIER_LEGACY_RECORD_LIST = 2,
       NL_CARRIER_LEGACY_SCALAR_LIST = 3 };
typedef struct {
    Value value;
    /* I keep this one-byte private fact: 0 ordinary, 1 DynArray/checked
     * logical list, 2 explicit legacy List_int record storage, 3 explicit legacy
     * List_int scalar storage. Scalar cells grant no record edges. */
    unsigned char record_list;
} NlCarrierRef;

NlCarrierRef nl_carrier_value(Value value);
NlCarrierRef nl_carrier_record_list(DynArray *list);
NlCarrierRef nl_carrier_legacy_record_list(List_int *list);
NlCarrierRef nl_carrier_legacy_scalar_list(List_int *list);
/* My source adapter may identify typed slots from checked metadata. This
 * borrowed, read-only callback grants edges, never ownership of an INT. */
typedef bool (*NlCarrierRecordListSlot)(Value owner, int index, void *context);
void nl_carrier_values_set_slots(NlCarrierValues *heap, NlCarrierRecordListSlot slots, void *context);
/* I borrow declaration metadata for one immediate field traversal only.
 * Names map by their first match, including reordered runtime fields.
 * This read-only resolver grants list edges, never ownership or cached roots. */
typedef struct {
    char *const *names;
    const Type *types;
    int count;
} NlCarrierRecordListFields;
typedef NlCarrierRecordListFields (*NlCarrierRecordListResolver)(Value owner, void *context);
/* I select either this resolver or the existing per-slot callback. */
void nl_carrier_values_set_fields(NlCarrierValues *heap, NlCarrierRecordListResolver fields, void *context);
NlCarrierValues *nl_carrier_values_new(void);
bool nl_carrier_values_free(NlCarrierValues *heap);
bool nl_carrier_values_clear(NlCarrierValues *heap);
bool nl_carrier_values_contains(const NlCarrierValues *heap, NlCarrierRef value);
/* I adopt exactly one header and its owned STRING/FUNCTION scalar storage.
 * Aggregate children remain separately owned. A list is explicitly identified
 * as managed ELEM_INT DynArray, malloc-owned List_int record storage,
 * or explicitly identified List_int integer/enum storage. Scalar cells grant
 * no child edges. I do not own referenced records implicitly.
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
/* I consume malloc-backed legacy storage at an explicit free boundary. I free
 * its header/data now, not its separately owned record children. A repeated
 * consumption fails before reading freed storage. The caller grants this
 * exact destruction; ordinary INT facts never call this operation. */
bool nl_carrier_values_consume_legacy_list(NlCarrierValues *heap, List_int *list);
/* I start an explicitly new caller-owned lifetime at a reused address. The
 * caller supplies the actual canonical or legacy layout and replaces old
 * caller root facts. Storage remains valid until those roots are withdrawn.
 * I adopt nothing; owned addresses, ordinary INT facts and malformed layouts
 * are refused. This explicit boundary is not inferred from pointer bits. */
bool nl_carrier_values_begin_list_borrow(NlCarrierValues *heap, NlCarrierRef list);
/* My consumed facts hold addresses only, never released payloads. */
size_t nl_carrier_values_consumed_count(const NlCarrierValues *heap);
size_t nl_carrier_values_consumed_capacity(const NlCarrierValues *heap);
NlCarrierHeapStats nl_carrier_values_stats(const NlCarrierValues *heap);
size_t nl_carrier_values_reclaimed_arrays(void);
#endif
