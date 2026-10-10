#include "nsi_websocket_values.h"
#include <stdlib.h>
#include <string.h>

typedef enum { WV_EMPTY, WV_CONNECTION, WV_CONNECT_OK, WV_CONNECT_ERROR } WvKind;
typedef struct {
    uint64_t generation, borrow_epoch;
    WvKind kind;
    bool borrowed;
    NlWsTransport *transport;
    NlWsTransportResult error;
} WvSlot;
struct NlWsValues {
    uint64_t identity;
    size_t storage_limit,transport_bound;
    NlWsTransportPolicy policy;
    char helper[4096];
    bool finished;
    NlWsValuesFinish report;
    WvSlot slots[NL_WS_VALUE_SLOTS];
};
static uint64_t wv_identity;

static bool wv_empty(NlWsValue v) {
    return !v.invocation && !v.generation && !v.slot;
}
static bool wv_disjoint(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x >= na : x - y >= nb;
}
static NlWsValueStatus wv_context(NlWsValues *s) {
    return !s ? NL_WS_VALUE_ARGUMENT : s->finished ? NL_WS_VALUE_DISPOSED : NL_WS_VALUE_OK;
}
static NlWsValueStatus wv_resolve(NlWsValues *s, const NlWsValue *v, WvSlot **out) {
    NlWsValueStatus status = wv_context(s);
    if (status != NL_WS_VALUE_OK) return status;
    if (!v) return NL_WS_VALUE_ARGUMENT;
    if (v->invocation != s->identity || v->slot >= NL_WS_VALUE_SLOTS || !v->generation)
        return NL_WS_VALUE_STALE;
    WvSlot *slot = &s->slots[v->slot];
    if (slot->kind == WV_EMPTY || slot->generation != v->generation) return NL_WS_VALUE_STALE;
    *out = slot;
    return NL_WS_VALUE_OK;
}
static NlWsValue wv_handle(NlWsValues *s, uint32_t index) {
    return (NlWsValue){s->identity, s->slots[index].generation, index};
}
static void wv_clear(WvSlot *slot) {
    uint64_t generation = slot->generation, epoch = slot->borrow_epoch;
    memset(slot, 0, sizeof(*slot));
    slot->generation = generation;
    slot->borrow_epoch = epoch;
}
static void wv_cleanup(NlWsValues *s, NlWsTransportResult r) {
    if (!r.cleanup_failed && !r.closure_unknown) return;
    if (!s->report.cleanup_failures) s->report.first_cleanup = r;
    else if (s->report.cleanup_failures == 1) s->report.next_cleanup = r;
    if (s->report.cleanup_failures < UINT64_MAX) s->report.cleanup_failures++;
}
static NlWsTransportResult wv_close_slot(NlWsValues *s,WvSlot *slot,int64_t timeout,bool abort_io) {
    NlWsTransport *transport=slot->transport;
    wv_clear(slot);
    if(abort_io)(void)nl_ws_transport_abort(transport);
    unsigned bounded=timeout<0 || timeout>60000?60001u:(unsigned)timeout;
    NlWsTransportResult r=nl_ws_transport_close(transport,bounded);
    wv_cleanup(s,r);
    return r;
}
bool nl_ws_values_minimum_storage(size_t *out) {
    if(!out)return false;
    *out=sizeof(NlWsValues);return true;
}
NlWsValueStatus nl_ws_values_create_bounded(const NlWsTransportPolicy *policy,size_t limit,NlWsValues **out) {
    if(!out || *out || !policy || policy->max_timeout_ms>60000)return NL_WS_VALUE_ARGUMENT;
    size_t length=policy->resolver_helper?strlen(policy->resolver_helper):0;
    if(length>=4096 || (policy->resolver_helper && (!length || policy->resolver_helper[0]!='/')))
        return NL_WS_VALUE_ARGUMENT;
    size_t transport;
    if(limit<sizeof(NlWsValues) || !nl_ws_transport_storage_bound(&transport))return NL_WS_VALUE_LIMIT;
    if(wv_identity==UINT64_MAX)return NL_WS_VALUE_LIMIT;
    NlWsValues *s=calloc(1,sizeof *s);if(!s)return NL_WS_VALUE_MEMORY;
    s->policy=*policy;s->storage_limit=limit;s->transport_bound=transport;
    if(policy->resolver_helper){memcpy(s->helper,policy->resolver_helper,length+1);s->policy.resolver_helper=s->helper;}
    s->identity=++wv_identity;*out=s;return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_values_create(const NlWsTransportPolicy *policy,NlWsValues **out) {
    return nl_ws_values_create_bounded(policy,SIZE_MAX,out);
}
NlWsValueStatus nl_ws_values_connect(NlWsValues *s,const void *url,size_t length,int64_t timeout,NlWsValue *out) {
    NlWsValueStatus status=wv_context(s);if(status!=NL_WS_VALUE_OK)return status;
    if(!out || !wv_empty(*out) || (!url && length) ||
        (url && !wv_disjoint(url,length,out,sizeof *out)))return NL_WS_VALUE_ARGUMENT;
    uint32_t i=0;
    while(i<NL_WS_VALUE_SLOTS && (s->slots[i].kind!=WV_EMPTY ||
        s->slots[i].generation==UINT64_MAX || s->slots[i].borrow_epoch==UINT64_MAX))i++;
    if(i==NL_WS_VALUE_SLOTS)return NL_WS_VALUE_LIMIT;
    NlWsTransport *transport=NULL;
    NlWsTransportResult r={.status=NL_WS_TRANSPORT_LIMIT};
    size_t live=0;
    for(uint32_t n=0;n<NL_WS_VALUE_SLOTS;n++)if(s->slots[n].transport)live++;
    if(live<(s->storage_limit-sizeof *s)/s->transport_bound && timeout>=0 && timeout<=60000)
        r=nl_ws_transport_connect(url,length,&s->policy,(unsigned)timeout,&transport);
    WvSlot *slot=&s->slots[i];slot->generation++;
    slot->kind=r.status==NL_WS_TRANSPORT_OK?WV_CONNECT_OK:WV_CONNECT_ERROR;
    slot->transport=transport;slot->error=r;wv_cleanup(s,r);
    *out=wv_handle(s,i);return NL_WS_VALUE_OK;
}
static NlWsValueStatus wv_move(NlWsValues *s, NlWsValue *v, NlWsValue *out, bool take_ok) {
    if (!v || !out || !wv_disjoint(v, sizeof(*v), out, sizeof(*out)) || !wv_empty(*out))
        return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;
    NlWsValueStatus status = wv_resolve(s, v, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    if (take_ok && slot->kind != WV_CONNECT_OK) return NL_WS_VALUE_TYPE;
    if (slot->borrowed) return NL_WS_VALUE_BORROWED;
    if (slot->generation == UINT64_MAX) return NL_WS_VALUE_LIMIT;
    uint32_t index = v->slot;
    slot->generation++;
    if (take_ok) slot->kind = WV_CONNECTION;
    *v = (NlWsValue){0};
    *out = wv_handle(s, index);
    return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_move(NlWsValues *s, NlWsValue *v, NlWsValue *out) {
    return wv_move(s, v, out, false);
}
NlWsValueStatus nl_ws_connect_take_ok(NlWsValues *s, NlWsValue *v, NlWsValue *out) {
    return wv_move(s, v, out, true);
}
NlWsValueStatus nl_ws_connect_view(NlWsValues *s, const NlWsValue *v, NlWsConnectView *out) {
    if (!out || !v || !wv_disjoint(v, sizeof(*v), out, sizeof(*out))) return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;
    NlWsValueStatus status = wv_resolve(s, v, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    if (slot->kind != WV_CONNECT_OK && slot->kind != WV_CONNECT_ERROR) return NL_WS_VALUE_TYPE;
    NlWsConnectView view = {0};
    view.ok = slot->kind == WV_CONNECT_OK;
    if (!view.ok) view.error = slot->error;
    *out = view;
    return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_connect_take_error(NlWsValues *s, NlWsValue *v, NlWsTransportResult *out) {
    if (!out || !v || !wv_disjoint(v, sizeof(*v), out, sizeof(*out))) return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;
    NlWsValueStatus status = wv_resolve(s, v, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    if (slot->kind != WV_CONNECT_ERROR) return NL_WS_VALUE_TYPE;
    NlWsTransportResult error = slot->error;
    wv_clear(slot);
    *v = (NlWsValue){0};
    *out = error;
    return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_borrow(NlWsValues *s, const NlWsValue *v, NlWsValueBorrow *out) {
    if (!out || !v || !wv_disjoint(v, sizeof(*v), out, sizeof(*out)) || !wv_empty(out->value) || out->epoch)
        return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;
    NlWsValueStatus status = wv_resolve(s, v, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    if (slot->kind != WV_CONNECTION) return NL_WS_VALUE_TYPE;
    if (slot->borrowed) return NL_WS_VALUE_BORROWED;
    if (slot->borrow_epoch == UINT64_MAX) return NL_WS_VALUE_LIMIT;
    slot->borrow_epoch++;
    slot->borrowed = true;
    *out = (NlWsValueBorrow){*v, slot->borrow_epoch};
    return NL_WS_VALUE_OK;
}
static NlWsValueStatus wv_borrow(NlWsValues *s, const NlWsValueBorrow *b, WvSlot **out) {
    if (!b) return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;
    NlWsValueStatus status = wv_resolve(s, &b->value, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    if (slot->kind != WV_CONNECTION || !slot->borrowed || !b->epoch || b->epoch != slot->borrow_epoch)
        return NL_WS_VALUE_STALE;
    *out = slot;
    return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_end_borrow(NlWsValues *s, NlWsValueBorrow *b) {
    WvSlot *slot;
    NlWsValueStatus status = wv_borrow(s, b, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    slot->borrowed = false;
    *b = (NlWsValueBorrow){0};
    return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_send(NlWsValues *s,const NlWsValueBorrow *b,bool binary,
    const void *bytes,size_t length,int64_t timeout,NlWsTransportResult *out) {
    if(!out || !b || (!bytes && length) || !wv_disjoint(b,sizeof *b,out,sizeof *out) ||
        (bytes && !wv_disjoint(bytes,length,out,sizeof *out)))return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;NlWsValueStatus status=wv_borrow(s,b,&slot);if(status!=NL_WS_VALUE_OK)return status;
    unsigned bounded=timeout<0 || timeout>60000?60001u:(unsigned)timeout;
    *out=nl_ws_transport_send(slot->transport,binary,bytes,length,bounded);return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_receive(NlWsValues *s,const NlWsValueBorrow *b,int64_t timeout,
    NlWsMessage *message,NlWsTransportResult *out) {
    if(!out || !b || !message || message->bytes || message->length || message->binary ||
        !wv_disjoint(b,sizeof *b,out,sizeof *out) || !wv_disjoint(b,sizeof *b,message,sizeof *message) ||
        !wv_disjoint(out,sizeof *out,message,sizeof *message))return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;NlWsValueStatus status=wv_borrow(s,b,&slot);if(status!=NL_WS_VALUE_OK)return status;
    unsigned bounded=timeout<0 || timeout>60000?60001u:(unsigned)timeout;
    *out=nl_ws_transport_receive(slot->transport,bounded,message);return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_close(NlWsValues *s,NlWsValue *v,int64_t timeout,NlWsTransportResult *out) {
    if(!out || !v || !wv_disjoint(v,sizeof *v,out,sizeof *out))return NL_WS_VALUE_ARGUMENT;
    WvSlot *slot;NlWsValueStatus status=wv_resolve(s,v,&slot);if(status!=NL_WS_VALUE_OK)return status;
    if(slot->kind!=WV_CONNECTION)return NL_WS_VALUE_TYPE;
    if(slot->borrowed)return NL_WS_VALUE_BORROWED;
    *v=(NlWsValue){0};*out=wv_close_slot(s,slot,timeout,false);return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_drop(NlWsValues *s, NlWsValue *v) {
    NlWsValueStatus status = wv_context(s);
    if (status != NL_WS_VALUE_OK) return status;
    if (!v) return NL_WS_VALUE_ARGUMENT;
    if (wv_empty(*v)) return NL_WS_VALUE_OK;
    WvSlot *slot;
    status = wv_resolve(s, v, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    if (slot->borrowed) return NL_WS_VALUE_BORROWED;
    if (slot->kind == WV_CONNECT_ERROR) wv_clear(slot);
    else {
        (void)wv_close_slot(s,slot,0,true);
    }
    *v = (NlWsValue){0};
    return NL_WS_VALUE_OK;
}
NlWsValueStatus nl_ws_value_validate(NlWsValues *s, const NlWsValue *v, bool connect_result) {
    WvSlot *slot;
    NlWsValueStatus status = wv_resolve(s, v, &slot);
    if (status != NL_WS_VALUE_OK) return status;
    return (connect_result ? (slot->kind == WV_CONNECT_OK || slot->kind == WV_CONNECT_ERROR) :
            slot->kind == WV_CONNECTION) ? NL_WS_VALUE_OK : NL_WS_VALUE_TYPE;
}
NlWsValueStatus nl_ws_value_borrow_validate(NlWsValues *s, const NlWsValueBorrow *b) {
    WvSlot *slot;
    return wv_borrow(s, b, &slot);
}
NlWsValueStatus nl_ws_values_live_slots(NlWsValues *s, uint64_t *owners, uint64_t *borrowed) {
    NlWsValueStatus status = wv_context(s);
    if (status != NL_WS_VALUE_OK) return status;
    if (!owners || !borrowed || !wv_disjoint(owners, sizeof(*owners), borrowed, sizeof(*borrowed)))
        return NL_WS_VALUE_ARGUMENT;
    uint64_t live = 0, held = 0;
    for (unsigned i = 0; i < NL_WS_VALUE_SLOTS; i++) {
        if (s->slots[i].kind != WV_EMPTY) live |= UINT64_C(1) << i;
        if (s->slots[i].borrowed) held |= UINT64_C(1) << i;
    }
    *owners = live; *borrowed = held;
    return NL_WS_VALUE_OK;
}
bool nl_ws_values_report(const NlWsValues *s, NlWsValuesFinish *out) {
    if (!s || !out) return false;
    *out = s->report;
    return true;
}
NlWsValuesFinish nl_ws_values_finish(NlWsValues *s, NlWsValueStatus error) {
    if (!s) return (NlWsValuesFinish){.execution = NL_WS_VALUE_ARGUMENT};
    if (s->finished) return s->report;
    if ((unsigned)error >= NL_WS_VALUE_STATUS_COUNT)
        return (NlWsValuesFinish){.execution = NL_WS_VALUE_ARGUMENT};
    s->report.execution = error;
    for (unsigned i = 0; i < NL_WS_VALUE_SLOTS; i++) {
        WvSlot *slot = &s->slots[i];
        if (slot->kind == WV_CONNECTION || slot->kind == WV_CONNECT_OK) {
            (void)wv_close_slot(s,slot,0,true);
        }
        wv_clear(slot);
    }
    s->finished = true;
    return s->report;
}
NlWsValuesFinish nl_ws_values_destroy(NlWsValues *s, NlWsValueStatus error) {
    NlWsValuesFinish report = nl_ws_values_finish(s, error);
    if (s && s->finished) free(s);
    return report;
}
