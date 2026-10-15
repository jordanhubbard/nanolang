#ifndef MIXED_WEBSOCKET_RUNTIME_HOST_H
#define MIXED_WEBSOCKET_RUNTIME_HOST_H
#include "../../src/nanoisa/services_indirect_public.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
static unsigned mixed_connects,mixed_sends,mixed_receives,mixed_closes,mixed_aborts,mixed_live,mixed_mode;
static unsigned mixed_instance_calls[64];
#define MIXED_CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL transport %d: %s\n",__LINE__,#x);exit(1);}}while(0)
static long mixed_budget=-1;
static unsigned mixed_faults;
void *mixed_runtime_malloc(size_t n){
 if(!mixed_budget){mixed_faults++;return NULL;}
 if(mixed_budget>0)mixed_budget--;
 return malloc(n);
}
void *mixed_runtime_calloc(size_t n,size_t size){
 if(!mixed_budget){mixed_faults++;return NULL;}
 if(mixed_budget>0)mixed_budget--;
 return calloc(n,size);
}
static void mixed_budget_start(void){const char *p=getenv("NANOLANG_TEST_RUNTIME_BUDGET");mixed_budget=p?strtol(p,NULL,10):-1;}
#ifndef MIXED_RUNTIME_REAL_TRANSPORT
struct NlWsTransport {unsigned ceiling;};
bool nl_ws_transport_storage_bound(size_t *out){if(!out)return false;*out=4096;return true;}
NlWsTransportResult nl_ws_transport_connect(const void *url,size_t length,const NlWsTransportPolicy *p,unsigned timeout,NlWsTransport **out){
 mixed_connects++;MIXED_CHECK(length==19 && !memcmp(url,"ws://127.0.0.1/test",19));
 MIXED_CHECK(p->allow_network && p->max_timeout_ms>=17 && p->max_timeout_ms<81);
 unsigned instance=p->max_timeout_ms-17;char path[32];snprintf(path,sizeof path,"/resolver/%u",instance);
 MIXED_CHECK(p->allow_lookup==(bool)(instance%2) && p->resolver_helper && !strcmp(p->resolver_helper,path));
 mixed_instance_calls[instance]++;
 if(timeout>p->max_timeout_ms)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_LIMIT};
 NlWsTransport *t=calloc(1,sizeof *t);if(!t)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_MEMORY};
 t->ceiling=p->max_timeout_ms;*out=t;mixed_live++;return (NlWsTransportResult){0};
}
NlWsTransportResult nl_ws_transport_send(NlWsTransport *t,bool binary,const void *bytes,size_t n,unsigned timeout){
 MIXED_CHECK(t && binary && n==3 && !memcmp(bytes,"a\0b",3) && timeout==0);mixed_sends++;
 return (NlWsTransportResult){.status=mixed_mode==5?NL_WS_TRANSPORT_IO:NL_WS_TRANSPORT_OK,.bytes=n};
}
NlWsTransportResult nl_ws_transport_receive(NlWsTransport *t,unsigned timeout,NlWsMessage *out){
 MIXED_CHECK(t && !timeout);mixed_receives++;
 if(mixed_mode==6)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_IO,.host_errno=EIO,.resolver_error=3,.supervisor_status=4,.close_code=1002};
 unsigned char *bytes=malloc(3);if(!bytes)return (NlWsTransportResult){.status=NL_WS_TRANSPORT_MEMORY};
 memcpy(bytes,"r\0b",3);*out=(NlWsMessage){true,bytes,3};return (NlWsTransportResult){.bytes=3};
}
void nl_ws_message_free(NlWsMessage *m){if(m){free(m->bytes);*m=(NlWsMessage){0};}}
NlWsTransportResult nl_ws_transport_abort(NlWsTransport *t){MIXED_CHECK(t);mixed_aborts++;return (NlWsTransportResult){.terminal=true};}
NlWsTransportResult nl_ws_transport_close(NlWsTransport *t,unsigned timeout){
 MIXED_CHECK(t && mixed_live);bool bad=timeout>t->ceiling;MIXED_CHECK(mixed_mode!=8 || bad);free(t);mixed_closes++;mixed_live--;
 return (NlWsTransportResult){.status=bad?NL_WS_TRANSPORT_LIMIT:mixed_mode==7?NL_WS_TRANSPORT_IO:NL_WS_TRANSPORT_OK,
  .cleanup_failed=mixed_mode==7,.cleanup_errno=mixed_mode==7?EIO:0,.terminal=true};
}
#endif
static unsigned mixed_catalogs(unsigned shape,unsigned *catalogs){
 static const unsigned cases[3][5]={{3},{1,2,3,3},{3,3,3,3,3}};
 unsigned count=shape==0?1:shape==1?4:5;
 for(unsigned i=0;i<count;i++)catalogs[i]=cases[shape][i];
 return count;
}
static NvmServicesHostGrant *mixed_grant(unsigned shape,unsigned mode){
 unsigned catalogs[5],count=mixed_catalogs(shape,catalogs);NvmServicesHostConfig configs[5]={0};char paths[5][32];
 for(unsigned i=0;i<count;i++){
  configs[i]=(NvmServicesHostConfig){.revision=1,.catalog=(NvmServicesHostCatalog)catalogs[i],.allowed=true};
  if(catalogs[i]==3){snprintf(paths[i],sizeof paths[i],"/resolver/%u",i);configs[i].allow_lookup=i%2;configs[i].max_timeout_ms=17+i;configs[i].resolver_helper=paths[i];
#ifdef MIXED_RUNTIME_REAL_TRANSPORT
   configs[i].max_timeout_ms=2000;
#endif
  }
 }
 if(mode==1)configs[count-1].allowed=false;
 if(mode==2)configs[count-1]=(NvmServicesHostConfig){.revision=1,.catalog=NVM_SERVICES_HOST_FILE,.allowed=true};
 NvmServicesHostGrant *g=NULL;MIXED_CHECK(nvm_services_host_grant_create_config(configs,count,&g)==NVM_SERVICES_HOST_OK);
 memset(paths,0,sizeof paths);memset(configs,0,sizeof configs);
 if(mode==3)MIXED_CHECK(nvm_services_host_grant_revoke(g)==NVM_SERVICES_HOST_OK);
 return g;
}
static void mixed_report(NvmServicesIndirectExecutionReport r,NvmServicesScalar value){
 MIXED_CHECK(!mixed_live);
 printf("{\"faults\":%u,\"status\":%u,\"acquired\":%u,\"value\":%lld,\"cleanup\":%llu,\"steps\":%llu,\"connects\":%u,\"sends\":%u,\"receives\":%u,\"closes\":%u,\"aborts\":%u,\"instances\":[%u,%u,%u,%u,%u]}\n",
  mixed_faults,r.runtime.status,r.runtime.acquired,(long long)value.value,(unsigned long long)r.runtime.cleanup.cleanup_failures,
  (unsigned long long)r.instructions_started,mixed_connects,mixed_sends,mixed_receives,mixed_closes,mixed_aborts,
  mixed_instance_calls[0],mixed_instance_calls[1],mixed_instance_calls[2],mixed_instance_calls[3],mixed_instance_calls[4]);
}
#endif
