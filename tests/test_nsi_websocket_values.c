#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../src/nsi_websocket_values.h"

static unsigned connects,closes,aborts,sends,receives,live;
static bool fail_alloc,fail_connect,fail_cleanup;
static NlWsTransportResult close_result;
struct NlWsTransport { int marker; };
static void *wv_test_calloc(size_t n,size_t size) { return fail_alloc?NULL:calloc(n,size); }
#define calloc wv_test_calloc
#include "../src/nsi_websocket_values.c"
#undef calloc

bool nl_ws_transport_storage_bound(size_t *out){if(!out)return false;*out=4096;return true;}
NlWsTransportResult nl_ws_transport_connect(const void *url,size_t length,
    const NlWsTransportPolicy *p,unsigned timeout,NlWsTransport **out) {
    (void)url;(void)length;(void)timeout;connects++;
    assert(!strcmp(p->resolver_helper,"/my/resolver"));
    if(!p->allow_network)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_RIGHTS};
    if(fail_connect)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_IO,
        .cleanup_failed=fail_cleanup,.cleanup_errno=9};
    *out=malloc(sizeof **out);assert(*out);(*out)->marker=123;live++;
    return (NlWsTransportResult){0};
}
NlWsTransportResult nl_ws_transport_close(NlWsTransport *t,unsigned timeout) {
    assert(t && t->marker==123);t->marker=0;free(t);live--;closes++;
    NlWsTransportResult r=close_result;r.terminal=true;
    if(timeout>60000)r.status=NL_WS_TRANSPORT_LIMIT;
    return r;
}
NlWsTransportResult nl_ws_transport_abort(NlWsTransport *t) {
    assert(t && t->marker==123);aborts++;return (NlWsTransportResult){.terminal=true};
}
NlWsTransportResult nl_ws_transport_send(NlWsTransport *t,bool binary,const void *b,size_t n,unsigned timeout) {
    assert(t && t->marker==123);(void)binary;(void)b;sends++;
    return (NlWsTransportResult){.status=timeout>60000?NL_WS_TRANSPORT_LIMIT:NL_WS_TRANSPORT_OK,.bytes=n};
}
NlWsTransportResult nl_ws_transport_receive(NlWsTransport *t,unsigned timeout,NlWsMessage *out) {
    assert(t && t->marker==123);(void)out;receives++;
    return (NlWsTransportResult){.status=timeout>60000?NL_WS_TRANSPORT_LIMIT:NL_WS_TRANSPORT_TIMEOUT};
}
static NlWsValues *context(bool network) {
    char path[]="/my/resolver";
    NlWsTransportPolicy p={network,true,path,60000};NlWsValues *s=NULL;
    assert(nl_ws_values_create(&p,&s)==NL_WS_VALUE_OK);memset(path,'x',sizeof path-1);
    return s;
}
static NlWsValue connect_value(NlWsValues *s) {
    NlWsValue v={0};assert(nl_ws_values_connect(s,"ws://test",9,100,&v)==NL_WS_VALUE_OK);return v;
}
static NlWsValue owner(NlWsValues *s) {
    NlWsValue result=connect_value(s),v={0};
    assert(nl_ws_connect_take_ok(s,&result,&v)==NL_WS_VALUE_OK && !result.invocation);return v;
}
int main(void) {
    NlWsTransportPolicy p={true,true,"/my/resolver",60000};NlWsValues *s=NULL;
    size_t base=0;assert(nl_ws_values_minimum_storage(&base) && base==sizeof(NlWsValues));
    assert(nl_ws_values_create_bounded(&p,base-1,&s)==NL_WS_VALUE_LIMIT && !s);
    assert(nl_ws_values_create_bounded(&p,base+4096,&s)==NL_WS_VALUE_OK);
    NlWsValue first=owner(s),blocked=connect_value(s);NlWsConnectView capacity;
    assert(connects==1 && live==1 && nl_ws_connect_view(s,&blocked,&capacity)==NL_WS_VALUE_OK && !capacity.ok && capacity.error.status==NL_WS_TRANSPORT_LIMIT);
    assert(nl_ws_value_drop(s,&blocked)==NL_WS_VALUE_OK);
    assert(nl_ws_value_drop(s,&first)==NL_WS_VALUE_OK);
    first=owner(s);assert(connects==2 && live==1);
    nl_ws_values_destroy(s,NL_WS_VALUE_OK);s=NULL;assert(!live);connects=closes=aborts=0;
    fail_alloc=true;assert(nl_ws_values_create(&p,&s)==NL_WS_VALUE_MEMORY && !s);fail_alloc=false;
    p.resolver_helper="relative";assert(nl_ws_values_create(&p,&s)==NL_WS_VALUE_ARGUMENT && !s);
    s=context(true);NlWsValues *other=context(true);
    NlWsValue result=connect_value(s),old=result,moved={0},v={0};NlWsConnectView view={0};
    assert(nl_ws_value_validate(other,&result,true)==NL_WS_VALUE_STALE);
    assert(nl_ws_connect_view(s,&result,&view)==NL_WS_VALUE_OK && view.ok);
    assert(nl_ws_value_move(s,&result,&moved)==NL_WS_VALUE_OK && !result.invocation);
    assert(nl_ws_value_validate(s,&old,true)==NL_WS_VALUE_STALE);
    old=moved;assert(nl_ws_connect_take_ok(s,&moved,&v)==NL_WS_VALUE_OK && !moved.invocation);
    assert(nl_ws_value_validate(s,&old,true)==NL_WS_VALUE_STALE);
    assert(nl_ws_value_validate(s,&v,true)==NL_WS_VALUE_TYPE);
    assert(nl_ws_value_move(s,&v,&v)==NL_WS_VALUE_ARGUMENT);
    NlWsValueBorrow b={0},second={0};NlWsTransportResult output={.status=NL_WS_TRANSPORT_CRYPTO};
    assert(nl_ws_value_borrow(s,&v,&b)==NL_WS_VALUE_OK);
    assert(nl_ws_value_borrow(s,&v,&second)==NL_WS_VALUE_BORROWED && !second.epoch);
    assert(nl_ws_value_move(s,&v,&moved)==NL_WS_VALUE_BORROWED);
    assert(nl_ws_value_drop(s,&v)==NL_WS_VALUE_BORROWED);
    assert(nl_ws_value_close(s,&v,0,&output)==NL_WS_VALUE_BORROWED && output.status==NL_WS_TRANSPORT_CRYPTO);
    assert(nl_ws_value_send(s,&b,false,"a\0b",3,1,&output)==NL_WS_VALUE_OK && output.bytes==3);
    NlWsMessage message={0};
    assert(nl_ws_value_receive(s,&b,1,&message,&output)==NL_WS_VALUE_OK && output.status==NL_WS_TRANSPORT_TIMEOUT && !message.bytes);
    assert(nl_ws_value_send(s,&b,false,NULL,0,-1,&output)==NL_WS_VALUE_OK && output.status==NL_WS_TRANSPORT_LIMIT);
    second=b;assert(nl_ws_value_end_borrow(s,&b)==NL_WS_VALUE_OK);
    assert(nl_ws_value_borrow(s,&v,&b)==NL_WS_VALUE_OK && b.epoch!=second.epoch);
    assert(nl_ws_value_send(s,&second,false,NULL,0,1,&output)==NL_WS_VALUE_STALE);
    assert(nl_ws_value_end_borrow(s,&second)==NL_WS_VALUE_STALE);
    uint64_t owners=0,borrowed=0;
    assert(nl_ws_values_live_slots(s,&owners,&borrowed)==NL_WS_VALUE_OK && owners==borrowed && owners);
    assert(nl_ws_value_end_borrow(s,&b)==NL_WS_VALUE_OK);
    old=v;assert(nl_ws_value_close(s,&v,-1,&output)==NL_WS_VALUE_OK && output.status==NL_WS_TRANSPORT_LIMIT && !v.invocation);
    assert(nl_ws_value_validate(s,&old,false)==NL_WS_VALUE_STALE);
    assert(nl_ws_value_drop(s,&v)==NL_WS_VALUE_OK);
    fail_connect=true;result=connect_value(s);fail_connect=false;
    assert(nl_ws_connect_view(s,&result,&view)==NL_WS_VALUE_OK && !view.ok && view.error.status==NL_WS_TRANSPORT_IO);
    assert(nl_ws_connect_take_ok(s,&result,&v)==NL_WS_VALUE_TYPE && !v.invocation);
    old=result;assert(nl_ws_connect_take_error(s,&result,&output)==NL_WS_VALUE_OK && !result.invocation);
    assert(nl_ws_value_validate(s,&old,true)==NL_WS_VALUE_STALE);
    v=owner(s);old=v;
    s->slots[v.slot].generation=UINT64_MAX;v.generation=UINT64_MAX;
    assert(nl_ws_value_move(s,&v,&moved)==NL_WS_VALUE_LIMIT && !moved.invocation);
    s->slots[v.slot].borrow_epoch=UINT64_MAX;
    assert(nl_ws_value_borrow(s,&v,&b)==NL_WS_VALUE_LIMIT && !b.epoch);
    assert(nl_ws_value_close(s,&v,0,&output)==NL_WS_VALUE_OK && !v.invocation);
    assert(nl_ws_value_validate(s,&old,false)==NL_WS_VALUE_STALE);
    /* I restore one exhausted test slot only to exercise full capacity below. */
    s->slots[old.slot].generation=0;s->slots[old.slot].borrow_epoch=0;
    result=(NlWsValue){0};unsigned before=connects;
    assert(nl_ws_values_connect(s,"x",1,INT64_MAX,&result)==NL_WS_VALUE_OK && connects==before);
    assert(nl_ws_connect_take_error(s,&result,&output)==NL_WS_VALUE_OK && output.status==NL_WS_TRANSPORT_LIMIT);
    NlWsValue slots[NL_WS_VALUE_SLOTS];
    for(unsigned i=0;i<NL_WS_VALUE_SLOTS;i++)slots[i]=connect_value(s);
    before=connects;result=(NlWsValue){0};
    assert(nl_ws_values_connect(s,"x",1,1,&result)==NL_WS_VALUE_LIMIT && connects==before && !result.invocation);
    for(unsigned i=0;i<NL_WS_VALUE_SLOTS;i++)assert(nl_ws_value_drop(s,&slots[i])==NL_WS_VALUE_OK);
    assert(aborts==NL_WS_VALUE_SLOTS && !live);
    v=owner(s);assert(nl_ws_value_borrow(s,&v,&b)==NL_WS_VALUE_OK);
    result=connect_value(s);
    close_result=(NlWsTransportResult){.status=NL_WS_TRANSPORT_IO,.cleanup_failed=true,.closure_unknown=true,.cleanup_errno=9};
    NlWsValuesFinish report=nl_ws_values_finish(s,NL_WS_VALUE_STATE);
    assert(report.execution==NL_WS_VALUE_STATE && report.cleanup_failures==2 && !live);
    assert(report.first_cleanup.cleanup_errno==9 && report.next_cleanup.closure_unknown);
    before=closes;report=nl_ws_values_finish(s,NL_WS_VALUE_OK);assert(closes==before && report.execution==NL_WS_VALUE_STATE);
    assert(nl_ws_value_validate(s,&v,false)==NL_WS_VALUE_DISPOSED);
    nl_ws_values_destroy(s,NL_WS_VALUE_OK);nl_ws_values_destroy(other,NL_WS_VALUE_OK);
    close_result=(NlWsTransportResult){0};s=context(false);result=connect_value(s);
    assert(nl_ws_connect_take_error(s,&result,&output)==NL_WS_VALUE_OK && output.status==NL_WS_TRANSPORT_RIGHTS);
    report=nl_ws_values_destroy(s,(NlWsValueStatus)-1);assert(report.execution==NL_WS_VALUE_ARGUMENT);
    assert(nl_ws_values_report(s,&report));nl_ws_values_destroy(s,NL_WS_VALUE_OK);
    s=context(true);fail_connect=true;fail_cleanup=true;result=connect_value(s);
    fail_connect=false;fail_cleanup=false;
    assert(nl_ws_connect_take_error(s,&result,&output)==NL_WS_VALUE_OK && output.cleanup_failed);
    report=nl_ws_values_destroy(s,NL_WS_VALUE_OK);
    assert(report.cleanup_failures==1 && report.first_cleanup.cleanup_errno==9);
    s=context(true);wv_identity=UINT64_MAX;p.resolver_helper="/my/resolver";other=NULL;
    assert(nl_ws_values_create(&p,&other)==NL_WS_VALUE_LIMIT && !other);
    for(unsigned i=0;i<NL_WS_VALUE_SLOTS;i++)s->slots[i].generation=UINT64_MAX;
    result=(NlWsValue){0};before=connects;
    assert(nl_ws_values_connect(s,"x",1,1,&result)==NL_WS_VALUE_LIMIT && before==connects);
    nl_ws_values_destroy(s,NL_WS_VALUE_OK);
    assert(!live && sends==2 && receives==1);
    return 0;
}
