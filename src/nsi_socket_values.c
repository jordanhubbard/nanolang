#include "nsi_socket_values.h"
#include "nsi_cap_private.h"
#include <stdlib.h>
#include <string.h>

#if NL_SOCKET_VALUE_SLOTS != NL_CAP_PRIVATE_SLOTS
#error "I require matching Socket value and adapter capacities"
#endif
typedef enum { SV_EMPTY, SV_CONNECTION, SV_CONNECT_OK, SV_CONNECT_ERROR } SvKind;
typedef struct {
    uint64_t generation, borrow_epoch;
    SvKind kind;
    bool borrowed, pending;
    NlSocketToken token;
    NlSocketResult error;
} SvSlot;
struct NlSocketValues {
    uint64_t identity;
    NlSocketService *service;
    bool finished;
    NlSocketValuesFinish report;
    SvSlot slots[NL_SOCKET_VALUE_SLOTS];
};
static uint64_t sv_identity;

bool nl_socket_values_storage_bound(size_t *out) {
    size_t service;
    if (!out || !nl_socket_service_storage_bound(&service) ||
        service > SIZE_MAX - sizeof(NlSocketValues)) return false;
    *out = sizeof(NlSocketValues) + service;
    return true;
}
static bool sv_empty(NlSocketValue v) {
    return !v.invocation && !v.generation && !v.slot;
}
static bool sv_disjoint(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x >= na : x - y >= nb;
}
static NlSocketValueStatus sv_context(NlSocketValues *s) {
    return !s ? NL_SOCKET_VALUE_ARGUMENT : s->finished ? NL_SOCKET_VALUE_DISPOSED : NL_SOCKET_VALUE_OK;
}
static NlSocketValueStatus sv_resolve(NlSocketValues *s, const NlSocketValue *v, SvSlot **out) {
    NlSocketValueStatus status = sv_context(s);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (!v) return NL_SOCKET_VALUE_ARGUMENT;
    if (v->invocation != s->identity || v->slot >= NL_SOCKET_VALUE_SLOTS || !v->generation)
        return NL_SOCKET_VALUE_STALE;
    SvSlot *slot = &s->slots[v->slot];
    if (slot->kind == SV_EMPTY || slot->generation != v->generation) return NL_SOCKET_VALUE_STALE;
    *out = slot;
    return NL_SOCKET_VALUE_OK;
}
static NlSocketValue sv_handle(NlSocketValues *s, uint32_t index) {
    return (NlSocketValue){s->identity, s->slots[index].generation, index};
}
static void sv_clear(SvSlot *slot) {
    uint64_t generation = slot->generation, epoch = slot->borrow_epoch;
    memset(slot, 0, sizeof(*slot));
    slot->generation = generation;
    slot->borrow_epoch = epoch;
}
static void sv_cleanup(NlSocketValues *s, NlSocketResult r) {
    if (r.status == NL_SOCKET_OK && !r.cleanup_failed && !r.closure_unknown) return;
    if (!s->report.cleanup_failures) s->report.first_cleanup = r;
    else if (s->report.cleanup_failures == 1) s->report.next_cleanup = r;
    if (s->report.cleanup_failures < UINT64_MAX) s->report.cleanup_failures++;
}
static NlSocketValueStatus sv_close_slot(NlSocketValues *s, SvSlot *slot, NlSocketResult *out) {
    NlSocketResult r = nl_socket_consume_close(s->service, &slot->token);
    sv_cleanup(s, r);
    if (!r.consumed) return NL_SOCKET_VALUE_STATE;
    sv_clear(slot);
    *out = r;
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_values_create(NlSocketValues **out) {
    if (!out || *out) return NL_SOCKET_VALUE_ARGUMENT;
    if (sv_identity == UINT64_MAX) return NL_SOCKET_VALUE_LIMIT;
    NlSocketValues *s = calloc(1, sizeof(*s));
    if (!s) return NL_SOCKET_VALUE_MEMORY;
    NlSocketResult r = nl_socket_service_create(&s->service);
    if (r.status != NL_SOCKET_OK) {
        free(s);
        return r.status == NL_SOCKET_LIMIT ? NL_SOCKET_VALUE_LIMIT :
               r.status == NL_SOCKET_MEMORY ? NL_SOCKET_VALUE_MEMORY : NL_SOCKET_VALUE_STATE;
    }
    s->identity = ++sv_identity;
    *out = s;
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_values_connect(NlSocketValues *s, const NlSocketAddress *address, NlSocketValue *out) {
    NlSocketValueStatus status = sv_context(s);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (!address || !out || !sv_disjoint(address, sizeof(*address), out, sizeof(*out)) || !sv_empty(*out))
        return NL_SOCKET_VALUE_ARGUMENT;
    uint32_t i = 0;
    while (i < NL_SOCKET_VALUE_SLOTS && (s->slots[i].kind != SV_EMPTY ||
           s->slots[i].generation == UINT64_MAX || s->slots[i].borrow_epoch == UINT64_MAX)) i++;
    if (i == NL_SOCKET_VALUE_SLOTS) return NL_SOCKET_VALUE_LIMIT;
    NlSocketToken token = {0};
    NlSocketResult r = nl_socket_acquire_tcp(s->service, address,
        NL_CAP_READ | NL_CAP_WRITE | NL_CAP_TRANSFER, &token);
    SvSlot *slot = &s->slots[i];
    slot->generation++;
    if (r.status == NL_SOCKET_OK) {
        slot->kind = SV_CONNECT_OK;
        slot->token = token;
        slot->pending = r.connect_pending;
    } else {
        slot->kind = SV_CONNECT_ERROR;
        slot->error = r;
        if (r.cleanup_failed || r.closure_unknown) sv_cleanup(s, r);
    }
    *out = sv_handle(s, i);
    return NL_SOCKET_VALUE_OK;
}
bool nl_socket_endpoint_decode(const NlSocketEndpoint *endpoint, NlSocketAddress *out) {
    if (!endpoint || !out || !sv_disjoint(endpoint, sizeof(*endpoint), out, sizeof(*out))) return false;
    if ((endpoint->family != NL_SOCKET_IPV4 && endpoint->family != NL_SOCKET_IPV6) ||
        endpoint->port < 1 || endpoint->port > UINT16_MAX ||
        endpoint->scope_id < 0 || endpoint->scope_id > UINT32_MAX) return false;
    int64_t words[4] = {endpoint->address0, endpoint->address1, endpoint->address2, endpoint->address3};
    for (unsigned i = 0; i < 4; i++) if (words[i] < 0 || words[i] > UINT32_MAX) return false;
    if (endpoint->family == NL_SOCKET_IPV4 &&
        (words[1] || words[2] || words[3] || endpoint->scope_id)) return false;
    NlSocketAddress address = {.family = (NlSocketFamily)endpoint->family,
                              .port = (uint16_t)endpoint->port,
                              .scope_id = (uint32_t)endpoint->scope_id};
    for (unsigned i = 0; i < 4; i++)
        for (unsigned j = 0; j < 4; j++)
            address.address[4 * i + j] = (uint8_t)((uint32_t)words[i] >> (24 - 8 * j));
    *out = address;
    return true;
}
NlSocketValueStatus nl_socket_values_begin_connect(NlSocketValues *s, const NlSocketEndpoint *endpoint,
                                                  NlSocketValue *out) {
    if (!endpoint || !out || !sv_disjoint(endpoint, sizeof(*endpoint), out, sizeof(*out)))
        return NL_SOCKET_VALUE_ARGUMENT;
    NlSocketAddress address = {0};
    /* Decode failure leaves a deliberately invalid address. The ordinary value
     * acquisition path publishes its Error arm without acquiring a descriptor. */
    (void)nl_socket_endpoint_decode(endpoint, &address);
    return nl_socket_values_connect(s, &address, out);
}
static NlSocketValueStatus sv_move(NlSocketValues *s, NlSocketValue *v, NlSocketValue *out, bool take_ok) {
    if (!v || !out || !sv_disjoint(v, sizeof(*v), out, sizeof(*out)) || !sv_empty(*out))
        return NL_SOCKET_VALUE_ARGUMENT;
    SvSlot *slot;
    NlSocketValueStatus status = sv_resolve(s, v, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (take_ok && slot->kind != SV_CONNECT_OK) return NL_SOCKET_VALUE_TYPE;
    if (slot->borrowed) return NL_SOCKET_VALUE_BORROWED;
    if (slot->generation == UINT64_MAX) return NL_SOCKET_VALUE_LIMIT;
    uint32_t index = v->slot;
    slot->generation++;
    if (take_ok) slot->kind = SV_CONNECTION;
    *v = (NlSocketValue){0};
    *out = sv_handle(s, index);
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_move(NlSocketValues *s, NlSocketValue *v, NlSocketValue *out) {
    return sv_move(s, v, out, false);
}
NlSocketValueStatus nl_socket_connect_take_ok(NlSocketValues *s, NlSocketValue *v, NlSocketValue *out) {
    return sv_move(s, v, out, true);
}
NlSocketValueStatus nl_socket_connect_view(NlSocketValues *s, const NlSocketValue *v, NlSocketConnectView *out) {
    if (!out || !v || !sv_disjoint(v, sizeof(*v), out, sizeof(*out))) return NL_SOCKET_VALUE_ARGUMENT;
    SvSlot *slot;
    NlSocketValueStatus status = sv_resolve(s, v, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (slot->kind != SV_CONNECT_OK && slot->kind != SV_CONNECT_ERROR) return NL_SOCKET_VALUE_TYPE;
    NlSocketConnectView view = {0};
    view.ok = slot->kind == SV_CONNECT_OK;
    if (view.ok) view.pending = slot->pending;
    else view.error = slot->error;
    *out = view;
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_connect_take_error(NlSocketValues *s, NlSocketValue *v, NlSocketResult *out) {
    if (!out || !v || !sv_disjoint(v, sizeof(*v), out, sizeof(*out))) return NL_SOCKET_VALUE_ARGUMENT;
    SvSlot *slot;
    NlSocketValueStatus status = sv_resolve(s, v, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (slot->kind != SV_CONNECT_ERROR) return NL_SOCKET_VALUE_TYPE;
    NlSocketResult error = slot->error;
    sv_clear(slot);
    *v = (NlSocketValue){0};
    *out = error;
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_borrow(NlSocketValues *s, const NlSocketValue *v, NlSocketValueBorrow *out) {
    if (!out || !v || !sv_disjoint(v, sizeof(*v), out, sizeof(*out)) || !sv_empty(out->value) || out->epoch)
        return NL_SOCKET_VALUE_ARGUMENT;
    SvSlot *slot;
    NlSocketValueStatus status = sv_resolve(s, v, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (slot->kind != SV_CONNECTION) return NL_SOCKET_VALUE_TYPE;
    if (slot->borrowed) return NL_SOCKET_VALUE_BORROWED;
    if (slot->borrow_epoch == UINT64_MAX) return NL_SOCKET_VALUE_LIMIT;
    slot->borrow_epoch++;
    slot->borrowed = true;
    *out = (NlSocketValueBorrow){*v, slot->borrow_epoch};
    return NL_SOCKET_VALUE_OK;
}
static NlSocketValueStatus sv_borrow(NlSocketValues *s, const NlSocketValueBorrow *b, SvSlot **out) {
    if (!b) return NL_SOCKET_VALUE_ARGUMENT;
    SvSlot *slot;
    NlSocketValueStatus status = sv_resolve(s, &b->value, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (slot->kind != SV_CONNECTION || !slot->borrowed || !b->epoch || b->epoch != slot->borrow_epoch)
        return NL_SOCKET_VALUE_STALE;
    *out = slot;
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_end_borrow(NlSocketValues *s, NlSocketValueBorrow *b) {
    SvSlot *slot;
    NlSocketValueStatus status = sv_borrow(s, b, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    slot->borrowed = false;
    *b = (NlSocketValueBorrow){0};
    return NL_SOCKET_VALUE_OK;
}
static NlSocketScalarResult sv_scalar(NlSocketScalarKind kind, NlSocketResult r) {
    NlSocketScalarResult out = {0};
    out.kind = kind;
    out.ok = r.status == NL_SOCKET_OK || (kind == NL_SOCKET_VALUE_RECEIVE && r.status == NL_SOCKET_EOF);
    out.detail = r;
    return out;
}
static NlSocketValueStatus sv_scalar_borrow(NlSocketValues *s, const NlSocketValueBorrow *b,
                                           NlSocketScalarResult *out, SvSlot **slot) {
    if (!out || !b || !sv_disjoint(b, sizeof(*b), out, sizeof(*out))) return NL_SOCKET_VALUE_ARGUMENT;
    return sv_borrow(s, b, slot);
}
NlSocketValueStatus nl_socket_value_finish_connect(NlSocketValues *s, const NlSocketValueBorrow *b, NlSocketScalarResult *out) {
    SvSlot *slot;
    NlSocketValueStatus status = sv_scalar_borrow(s, b, out, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    *out = sv_scalar(NL_SOCKET_VALUE_CONNECT, nl_socket_finish_connect(s->service, &slot->token));
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_send_byte(NlSocketValues *s, const NlSocketValueBorrow *b, int64_t byte, NlSocketScalarResult *out) {
    SvSlot *slot;
    NlSocketValueStatus status = sv_scalar_borrow(s, b, out, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    NlSocketResult r = {.status = NL_SOCKET_ARGUMENT};
    if (byte >= 0 && byte <= 255) r = nl_socket_send_byte(s->service, &slot->token, (uint8_t)byte);
    NlSocketScalarResult result = sv_scalar(NL_SOCKET_VALUE_SEND, r);
    if (result.ok) result.value = (int64_t)r.bytes;
    *out = result;
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_receive_byte(NlSocketValues *s, const NlSocketValueBorrow *b, NlSocketScalarResult *out) {
    SvSlot *slot;
    NlSocketValueStatus status = sv_scalar_borrow(s, b, out, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    uint8_t byte = 0;
    NlSocketResult r = nl_socket_receive_byte(s->service, &slot->token, &byte);
    NlSocketScalarResult result = sv_scalar(NL_SOCKET_VALUE_RECEIVE, r);
    if (result.ok) { result.value = byte; result.eof = r.eof; }
    *out = result;
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_close(NlSocketValues *s, NlSocketValue *v, NlSocketScalarResult *out) {
    if (!out || !v || !sv_disjoint(v, sizeof(*v), out, sizeof(*out))) return NL_SOCKET_VALUE_ARGUMENT;
    SvSlot *slot;
    NlSocketValueStatus status = sv_resolve(s, v, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (slot->kind != SV_CONNECTION) return NL_SOCKET_VALUE_TYPE;
    if (slot->borrowed) return NL_SOCKET_VALUE_BORROWED;
    NlSocketResult r;
    status = sv_close_slot(s, slot, &r);
    if (status != NL_SOCKET_VALUE_OK) return status;
    *v = (NlSocketValue){0};
    *out = sv_scalar(NL_SOCKET_VALUE_CLOSE, r);
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_drop(NlSocketValues *s, NlSocketValue *v) {
    NlSocketValueStatus status = sv_context(s);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (!v) return NL_SOCKET_VALUE_ARGUMENT;
    if (sv_empty(*v)) return NL_SOCKET_VALUE_OK;
    SvSlot *slot;
    status = sv_resolve(s, v, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (slot->borrowed) return NL_SOCKET_VALUE_BORROWED;
    if (slot->kind == SV_CONNECT_ERROR) sv_clear(slot);
    else {
        NlSocketResult r;
        status = sv_close_slot(s, slot, &r);
        if (status != NL_SOCKET_VALUE_OK) return status;
    }
    *v = (NlSocketValue){0};
    return NL_SOCKET_VALUE_OK;
}
NlSocketValueStatus nl_socket_value_validate(NlSocketValues *s, const NlSocketValue *v, bool connect_result) {
    SvSlot *slot;
    NlSocketValueStatus status = sv_resolve(s, v, &slot);
    if (status != NL_SOCKET_VALUE_OK) return status;
    return (connect_result ? (slot->kind == SV_CONNECT_OK || slot->kind == SV_CONNECT_ERROR) :
            slot->kind == SV_CONNECTION) ? NL_SOCKET_VALUE_OK : NL_SOCKET_VALUE_TYPE;
}
NlSocketValueStatus nl_socket_value_borrow_validate(NlSocketValues *s, const NlSocketValueBorrow *b) {
    SvSlot *slot;
    return sv_borrow(s, b, &slot);
}
NlSocketValueStatus nl_socket_values_live_slots(NlSocketValues *s, uint64_t *owners, uint64_t *borrowed) {
    NlSocketValueStatus status = sv_context(s);
    if (status != NL_SOCKET_VALUE_OK) return status;
    if (!owners || !borrowed || !sv_disjoint(owners, sizeof(*owners), borrowed, sizeof(*borrowed)))
        return NL_SOCKET_VALUE_ARGUMENT;
    uint64_t live = 0, held = 0;
    for (unsigned i = 0; i < NL_SOCKET_VALUE_SLOTS; i++) {
        if (s->slots[i].kind != SV_EMPTY) live |= UINT64_C(1) << i;
        if (s->slots[i].borrowed) held |= UINT64_C(1) << i;
    }
    *owners = live; *borrowed = held;
    return NL_SOCKET_VALUE_OK;
}
bool nl_socket_values_report(const NlSocketValues *s, NlSocketValuesFinish *out) {
    if (!s || !out) return false;
    *out = s->report;
    return true;
}
NlSocketValuesFinish nl_socket_values_finish(NlSocketValues *s, NlSocketValueStatus error) {
    if (!s) return (NlSocketValuesFinish){.execution = NL_SOCKET_VALUE_ARGUMENT};
    if (s->finished) return s->report;
    if ((unsigned)error >= NL_SOCKET_VALUE_STATUS_COUNT)
        return (NlSocketValuesFinish){.execution = NL_SOCKET_VALUE_ARGUMENT};
    s->report.execution = error;
    for (unsigned i = 0; i < NL_SOCKET_VALUE_SLOTS; i++) {
        SvSlot *slot = &s->slots[i];
        if (slot->kind == SV_CONNECTION || slot->kind == SV_CONNECT_OK) {
            NlSocketResult r;
            NlSocketValueStatus status = sv_close_slot(s, slot, &r);
            if (status != NL_SOCKET_VALUE_OK && s->report.execution == NL_SOCKET_VALUE_OK)
                s->report.execution = status;
        }
        sv_clear(slot);
    }
    /* The service retains ambiguous-close history; one terminal report suffices. */
    sv_cleanup(s, nl_socket_service_destroy(s->service));
    s->service = NULL;
    s->finished = true;
    return s->report;
}
NlSocketValuesFinish nl_socket_values_destroy(NlSocketValues *s, NlSocketValueStatus error) {
    NlSocketValuesFinish report = nl_socket_values_finish(s, error);
    if (s && s->finished) free(s);
    return report;
}
