/* I fault the transport boundary while retaining real File/TCP lifetime cores. */
#include "../src/nsi_services_values.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks,connects,sends,receives,closes,aborts,file_closes,live;
static long budget=-1,allocations;
static bool fail_cleanup;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %u: %s\n",__LINE__,#x);exit(1);}}while(0)
#define OK(x) CHECK((x)==NL_SERVICES_VALUE_OK)
void *mixed_values_calloc(size_t n,size_t width){
 if(!budget)return NULL;
 if(budget>0)budget--;
 void *p=calloc(n,width);if(p)allocations++;return p;
}
void mixed_values_free(void *p){if(p){CHECK(allocations>0);allocations--;}free(p);}
int mixed_values_fclose(FILE *stream){int result=fclose(stream);file_closes++;if(fail_cleanup){errno=EIO;return EOF;}return result;}
struct NlWsTransport {unsigned ceiling;};
bool nl_ws_transport_storage_bound(size_t *out){if(!out)return false;*out=4096;return true;}
NlWsTransportResult nl_ws_transport_connect(const void *url,size_t size,const NlWsTransportPolicy *p,unsigned timeout,NlWsTransport **out){
 connects++;CHECK(size==8 && !memcmp(url,"ws://one",8));
 CHECK(p->resolver_helper && !strcmp(p->resolver_helper,"/copied/resolver"));
 if(!p->allow_network || !p->allow_lookup)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_RIGHTS};
 if(timeout>p->max_timeout_ms)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_LIMIT};
 NlWsTransport *t=mixed_values_calloc(1,sizeof *t);if(!t)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_MEMORY};
 t->ceiling=p->max_timeout_ms;live++;*out=t;return (NlWsTransportResult){0};
}
NlWsTransportResult nl_ws_transport_send(NlWsTransport *t,bool binary,const void *bytes,size_t n,unsigned timeout){
 CHECK(t);sends++;if(timeout>t->ceiling)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_LIMIT};
 CHECK(binary && n==3 && !memcmp(bytes,"a\0b",3));return (NlWsTransportResult){.bytes=n};
}
NlWsTransportResult nl_ws_transport_receive(NlWsTransport *t,unsigned timeout,NlWsMessage *out){
 CHECK(t);receives++;if(timeout>t->ceiling)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_LIMIT};
 unsigned char *p=mixed_values_calloc(3,1);if(!p)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_MEMORY};
 memcpy(p,"a\0b",3);*out=(NlWsMessage){true,p,3};return (NlWsTransportResult){.bytes=3};
}
void nl_ws_message_free(NlWsMessage *message){if(message){mixed_values_free(message->bytes);*message=(NlWsMessage){0};}}
NlWsTransportResult nl_ws_transport_abort(NlWsTransport *t){CHECK(t);aborts++;return (NlWsTransportResult){.terminal=true};}
NlWsTransportResult nl_ws_transport_close(NlWsTransport *t,unsigned timeout){
 CHECK(t && live);unsigned ceiling=t->ceiling;mixed_values_free(t);live--;closes++;
 return (NlWsTransportResult){.status=timeout>ceiling?NL_WS_TRANSPORT_LIMIT:fail_cleanup?NL_WS_TRANSPORT_IO:NL_WS_TRANSPORT_OK,
  .cleanup_failed=fail_cleanup,.closure_unknown=fail_cleanup,.cleanup_errno=fail_cleanup?EIO:0,.terminal=true};
}
static void configurations(NlServicesValueConfig *c,unsigned count){
 size_t minimum=0;CHECK(nl_ws_values_minimum_storage(&minimum));
 for(unsigned i=0;i<count;i++){
  NlServicesCatalog catalog=count==4?(i<2?(NlServicesCatalog)(i+1):NL_SERVICES_WEBSOCKET):(NlServicesCatalog)(i%3+1);
  c[i]=(NlServicesValueConfig){.catalog=catalog};
  if(catalog==NL_SERVICES_WEBSOCKET){c[i].websocket=(NlWsTransportPolicy){true,true,"/copied/resolver",i==3?29:17};c[i].websocket_storage_limit=minimum+4096;}
 }
}
static NlServicesValues *context(void){
 NlServicesValueConfig configs[4];configurations(configs,4);char helper[]="/copied/resolver";
 configs[2].websocket.resolver_helper=helper;configs[3].websocket.resolver_helper=helper;
 NlServicesValues *c=NULL;OK(nl_services_values_create_config(configs,4,&c));
 memset(configs,0,sizeof configs);memset(helper,'x',sizeof helper-1);return c;
}
static NlServicesValue ws_result(NlServicesValues *c,unsigned i,int64_t timeout){
 NlServicesValue result={0};OK(nl_services_values_websocket_connect(c,i,"ws://one",8,timeout,&result));
 CHECK(result.instance==i+1 && result.catalog==NL_SERVICES_WEBSOCKET);return result;
}
static NlServicesValue ws_owner(NlServicesValues *c,unsigned i){
 NlServicesValue result=ws_result(c,i,1),owner={0};NlServicesOpenView view;
 OK(nl_services_value_view(c,&result,&view));CHECK(view.catalog==NL_SERVICES_WEBSOCKET && view.ok && !view.pending);
 OK(nl_services_value_take_ok(c,&result,&owner));CHECK(!result.instance);return owner;
}
static void finish(NlServicesValues **c){NlServicesFinish report;OK(nl_services_values_destroy(c,NL_SERVICES_VALUE_OK,&report));CHECK(!*c && !report.cleanup_failures);}
static void boundaries(void){
 NlServicesCatalog old[]={NL_SERVICES_WEBSOCKET};NlServicesValues *c=NULL;size_t bound=123;
 CHECK(!nl_services_values_storage_bound(old,1,&bound) && bound==123);
 CHECK(nl_services_values_create(old,1,&c)==NL_SERVICES_VALUE_ARGUMENT && !c);
 NlServicesValueConfig configs[4];configurations(configs,4);
 CHECK(nl_services_values_config_storage_bound(configs,4,&bound));size_t original=bound;
 configs[2].websocket_storage_limit+=512;CHECK(nl_services_values_config_storage_bound(configs,4,&bound) && bound==original+512);
 configs[2].websocket_storage_limit=SIZE_MAX;bound=123;CHECK(!nl_services_values_config_storage_bound(configs,4,&bound) && bound==123);
 for(unsigned defect=0;defect<7;defect++){
  configurations(configs,4);
  switch(defect){case 0:configs[3].catalog=99;break;case 1:configs[3].websocket.max_timeout_ms=60001;break;
   case 2:configs[3].websocket.resolver_helper="relative";break;case 3:configs[3].websocket_storage_limit=0;break;
   case 4:configs[0].websocket.allow_network=true;break;case 5:configs[1].websocket_storage_limit=1;break;
   default:configs[3].websocket.resolver_helper="";break;}
  CHECK(nl_services_values_create_config(configs,4,&c)==NL_SERVICES_VALUE_ARGUMENT && !c && !allocations);
 }
 NlServicesValue value={0};c=context();
 CHECK(nl_services_values_websocket_connect(c,0,"ws://one",8,1,&value)==NL_SERVICES_VALUE_TYPE && !value.instance);
 CHECK(nl_services_values_acquire(c,2,NULL,&value)==NL_SERVICES_VALUE_TYPE && !value.instance);
 value=ws_result(c,2,18);NlServicesOpenView view;
 OK(nl_services_value_view(c,&value,&view));CHECK(!view.ok && view.error.websocket.status==NL_WS_TRANSPORT_LIMIT);
 NlServicesValue moved={0},old_value=value;OK(nl_services_value_move(c,&value,&moved));CHECK(!value.instance);
 CHECK(nl_services_value_validate(c,&old_value,true)==NL_SERVICES_VALUE_STALE);
 OK(nl_services_value_take_error(c,&moved,&view));CHECK(!moved.instance && view.catalog==NL_SERVICES_WEBSOCKET && view.error.websocket.status==NL_WS_TRANSPORT_LIMIT);
 finish(&c);CHECK(!allocations);
 configurations(configs,4);configs[2].websocket.allow_network=false;configs[3].websocket.allow_lookup=false;
 OK(nl_services_values_create_config(configs,4,&c));
 for(unsigned i=2;i<4;i++){value=ws_result(c,i,1);OK(nl_services_value_take_error(c,&value,&view));CHECK(view.error.websocket.status==NL_WS_TRANSPORT_RIGHTS);}
 finish(&c);CHECK(!allocations);
}
static void lifecycle(bool cleanup_fault){
 NlServicesValues *c=context(),*other=context();NlServicesValue values[4]={{0}};NlServicesOpenView view;
 OK(nl_services_values_acquire(c,0,NULL,&values[0]));NlServicesValue file={0};OK(nl_services_value_take_ok(c,&values[0],&file));values[0]=file;
 NlSocketEndpoint invalid={0};OK(nl_services_values_acquire(c,1,&invalid,&values[1]));OK(nl_services_value_take_error(c,&values[1],&view));CHECK(view.error.tcp.status==NL_SOCKET_ARGUMENT);
 values[2]=ws_owner(c,2);values[3]=ws_owner(c,3);NlServicesBorrow borrow[4]={{0}};
 for(unsigned i=2;i<4;i++){
  NlServicesValue saved=values[i],moved={0};OK(nl_services_value_move(c,&values[i],&moved));CHECK(!values[i].instance);
  CHECK(nl_services_value_validate(c,&saved,false)==NL_SERVICES_VALUE_STALE);values[i]=moved;
  CHECK(nl_services_value_validate(other,&values[i],false)==NL_SERVICES_VALUE_STALE);
  NlServicesValue forged=values[i];forged.instance=i==2?4:3;CHECK(nl_services_value_validate(c,&forged,false)==NL_SERVICES_VALUE_STALE);
  OK(nl_services_value_borrow(c,&values[i],&borrow[i]));OK(nl_services_borrow_validate(c,&borrow[i]));
  uint64_t owners=0,borrowed=0;OK(nl_services_values_live_slots(c,i,&owners,&borrowed));CHECK(owners && owners==borrowed);
  NlWsTransportResult result={.status=NL_WS_TRANSPORT_CRYPTO},prior=result;NlWsMessage message={0};unsigned before=sends,before_receive=receives;
  CHECK(nl_services_value_websocket_send(c,i==2?3:2,&borrow[i],true,"a\0b",3,1,&result)==NL_SERVICES_VALUE_TYPE && sends==before);
  CHECK(!memcmp(&result,&prior,sizeof result));
  NlServicesBorrow forged_borrow=borrow[i];forged_borrow.instance=i==2?4:3;
  CHECK(nl_services_value_websocket_send(c,i==2?3:2,&forged_borrow,true,"a\0b",3,1,&result)==NL_SERVICES_VALUE_STALE && sends==before);
  CHECK(nl_services_value_websocket_receive(c,i==2?3:2,&borrow[i],1,&message,&result)==NL_SERVICES_VALUE_TYPE && !message.bytes && receives==before_receive);
  CHECK(nl_services_value_websocket_close(c,i==2?3:2,&values[i],1,&result)==NL_SERVICES_VALUE_TYPE);
  CHECK(nl_services_value_websocket_close(c,i,&values[i],1,&result)==NL_SERVICES_VALUE_BORROWED);
  CHECK(nl_services_value_drop(c,&values[i])==NL_SERVICES_VALUE_BORROWED);
  NlServicesScalarResult scalar={.catalog=99};
  CHECK(nl_services_value_call(c,i,1,&borrow[i],1,&scalar)==NL_SERVICES_VALUE_TYPE && scalar.catalog==99);
  CHECK(nl_services_value_close(c,i,&values[i],&scalar)==NL_SERVICES_VALUE_TYPE && scalar.catalog==99);
  OK(nl_services_value_websocket_send(c,i,&borrow[i],true,"a\0b",3,18,&result));CHECK(result.status==(i==2?NL_WS_TRANSPORT_LIMIT:NL_WS_TRANSPORT_OK));
 }
 NlWsMessage message={0};NlWsTransportResult result;OK(nl_services_value_websocket_receive(c,2,&borrow[2],1,&message,&result));CHECK(result.status==NL_WS_TRANSPORT_OK && message.binary && message.length==3 && !memcmp(message.bytes,"a\0b",3));
 NlWsMessage saved_message=message;CHECK(nl_services_value_websocket_receive(c,2,&borrow[2],1,&message,&result)==NL_SERVICES_VALUE_ARGUMENT && message.bytes==saved_message.bytes);
 unsigned close_before=closes,abort_before=aborts,file_before=file_closes;
 if(!cleanup_fault){
  NlServicesBorrow stale=borrow[2];OK(nl_services_value_end_borrow(c,&borrow[2]));CHECK(!borrow[2].instance);
  CHECK(nl_services_borrow_validate(c,&stale)==NL_SERVICES_VALUE_STALE);
  NlServicesValue stale_value=values[2];OK(nl_services_value_websocket_close(c,2,&values[2],-1,&result));CHECK(!values[2].instance && result.status==NL_WS_TRANSPORT_LIMIT);
  CHECK(nl_services_value_validate(c,&stale_value,false)==NL_SERVICES_VALUE_STALE);
 }
 fail_cleanup=cleanup_fault;NlServicesFinish first,again;
 OK(nl_services_values_finish(c,NL_SERVICES_VALUE_STATE,&first));fail_cleanup=false;
 CHECK(first.execution==NL_SERVICES_VALUE_STATE && first.count==4 && first.cleanup_failures==(cleanup_fault?3u:0u));
 CHECK(closes==close_before+2 && aborts==abort_before+(cleanup_fault?2:1));
#ifdef MIXED_VALUES_INSTRUMENT
 CHECK(file_closes==file_before+1);
#else
 (void)file_before;
#endif
 if(cleanup_fault){CHECK(first.instances[2].finish.websocket.cleanup_failures==1 && first.instances[3].finish.websocket.cleanup_failures==1);CHECK(first.instances[2].finish.websocket.first_cleanup.cleanup_errno==EIO);}
 OK(nl_services_values_destroy(&c,NL_SERVICES_VALUE_OK,&again));CHECK(!c && !memcmp(&first,&again,sizeof first));CHECK(closes==close_before+2);
 CHECK(message.length==3 && !memcmp(message.bytes,"a\0b",3));nl_ws_message_free(&message);CHECK(!message.bytes);finish(&other);CHECK(!live && !allocations);
}
static void capacity(void){
 NlServicesValues *c=context();NlServicesValue first=ws_owner(c,2);unsigned before=connects;
 NlServicesValue values[64]={{0}};NlServicesOpenView view;
 for(unsigned i=0;i<63;i++){values[i]=ws_result(c,2,1);OK(nl_services_value_view(c,&values[i],&view));CHECK(!view.ok && view.error.websocket.status==NL_WS_TRANSPORT_LIMIT);}
 CHECK(connects==before);CHECK(nl_services_values_websocket_connect(c,2,"ws://one",8,1,&values[63])==NL_SERVICES_VALUE_LIMIT && !values[63].instance);
 NlServicesValue second=ws_owner(c,3);CHECK(second.instance==4);OK(nl_services_value_drop(c,&second));
 for(unsigned i=0;i<63;i++)OK(nl_services_value_drop(c,&values[i]));
 OK(nl_services_value_drop(c,&first));
 first=ws_owner(c,2);OK(nl_services_value_drop(c,&first));finish(&c);CHECK(!live && !allocations);
}
#ifdef MIXED_VALUES_INSTRUMENT
static void allocation_failures(void){
 NlServicesValueConfig configs[64];configurations(configs,64);bool succeeded=false;
 for(int n=0;n<256;n++){
  NlServicesValues *c=NULL;CHECK(!allocations);budget=n;NlServicesValueStatus status=nl_services_values_create_config(configs,64,&c);budget=-1;
  if(status==NL_SERVICES_VALUE_OK){finish(&c);succeeded=true;printf("I checked %d mixed creation allocation prefixes.\n",n);break;}
  CHECK(status==NL_SERVICES_VALUE_MEMORY && !c && !allocations);
 }
 CHECK(succeeded && !allocations);
 NlServicesValues *c=context();NlServicesValue result={0};budget=0;
 OK(nl_services_values_websocket_connect(c,2,"ws://one",8,1,&result));budget=-1;NlServicesOpenView view;
 OK(nl_services_value_take_error(c,&result,&view));CHECK(view.error.websocket.status==NL_WS_TRANSPORT_MEMORY);
 NlServicesValue owner=ws_owner(c,2);NlServicesBorrow borrow={0};OK(nl_services_value_borrow(c,&owner,&borrow));NlWsMessage message={0};NlWsTransportResult error;
 budget=0;OK(nl_services_value_websocket_receive(c,2,&borrow,1,&message,&error));budget=-1;CHECK(error.status==NL_WS_TRANSPORT_MEMORY && !message.bytes);
 finish(&c);CHECK(!live && !allocations);
}
#endif
int main(void){boundaries();lifecycle(false);capacity();
#ifdef MIXED_VALUES_INSTRUMENT
 lifecycle(true);allocation_failures();
#endif
 CHECK(!live && !allocations);printf("PASS %u mixed WebSocket value checks with controlled transport\n",checks);return 0;}
