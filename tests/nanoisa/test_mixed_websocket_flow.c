/* I check all three catalogs in one graph before runtime admission. */
#define SERVICES_FLOW_MAIN file_tcp_flow_fixture_main
#include "test_services_flow.c"
#undef SERVICES_FLOW_MAIN
#include "../../src/nanoisa/services_indirect_public.h"
#include "../../src/nanoisa/services_host_grant_internal.h"
#include "../../src/nanoisa/file_host_grant_internal.h"
#include "../../src/nanoisa/services_indirect_runtime.h"
#include "../../src/runtime/service_policy.h"

static void ws_contracts(NvmModule *m,const NvmMultiNominalBindings *b){
 unsigned count=b->count,functions=1+2*count,layouts=rd32(m->layout_data);
 free(m->function_param_types[0]);m->function_param_types[0]=NULL;
 for(unsigned f=0;f<functions;f++){
  bool send=f>count;unsigned instance=f?(f-1)%count:0;
  unsigned params=f?(send?(b->instances[instance].catalog==3?3:2):1):0;
  NvmFunctionEntry fn=m->functions[0];fn.arity=(uint8_t)params;fn.local_count=(uint16_t)(f?params:3*count);
  fn.result_tag=!f?TAG_INT:send?TAG_UNION:TAG_STRUCT;fn.result_count=1;
  char name[32];snprintf(name,sizeof name,"function_%u",f);fn.name_idx=str(m,name);
  if(!f)m->functions[0]=fn;else CHECK(nvm_add_function(m,&fn)==f);
  uint8_t tags[]={TAG_STRUCT,params==3?TAG_STRUCT:TAG_INT,TAG_INT};
  CHECK(nvm_set_function_param_types(m,f,tags,(uint16_t)params));
 }
 unsigned start=(8+layouts+3)&~3u;size_t size=start+4;
 for(unsigned f=0;f<functions;f++)size+=12+8*m->functions[f].local_count;
 free(m->ownership_data);m->ownership_data=calloc(1,size);CHECK(m->ownership_data);m->ownership_size=(uint32_t)size;
 uint8_t *o=m->ownership_data;wr32(o,1);wr32(o+4,layouts);wr32(o+start,functions);
 for(unsigned i=0;i<count;i++)for(unsigned j=0;j<types(&b->instances[i]);j++)o[8+b->instances[i].layouts[j]]=(j==0||j==3)?3:1;
 size_t at=start+4;
 for(unsigned f=0;f<functions;f++){
  const NvmFunctionEntry *fn=&m->functions[f];bool send=f>count;unsigned instance=f?(f-1)%count:0;
  o[at]=(uint8_t)fn->local_count;o[at+2]=fn->arity;
  descriptor(o+at+4,fn->result_tag,0,f?b->instances[instance].layouts[send?4:0]:NVM_V2_NO_INDEX);
  for(unsigned j=0;j<fn->local_count;j++){
   uint8_t tag=TAG_STRUCT,mode=0;uint32_t layout;
   if(!f){tag=j>=count && j<2*count?TAG_STRUCT:TAG_UNION;layout=b->instances[j%count].layouts[j<count?3:j<2*count?0:5];}
   else if(!send || !j){layout=b->instances[instance].layouts[0];mode=send?2:0;}
   else if(fn->arity==3 && j==1)layout=b->instances[instance].layouts[2];
   else{layout=NVM_V2_NO_INDEX;tag=TAG_INT;}
   descriptor(o+at+12+8*j,tag,mode,layout);
  }
  at+=12+8*fn->local_count;
 }
 CHECK(at==size);
}
static void push_text(Code *c,NvmModule *m,const char *bytes,unsigned size){
 byte(c,OP_PUSH_STR);dword(c,nvm_add_string(m,bytes,size));
}
static unsigned ws_fixture_timeout;
static const char *ws_fixture_reply="a\0b";
static bool ws_fixture_require_success;
static void ws_deadline(Code *c){byte(c,OP_PUSH_I64);for(unsigned n=0;n<8;n++)byte(c,(uint8_t)((uint64_t)ws_fixture_timeout>>(8*n)));}
static void ws_require_ok(Code *c){byte(c,OP_UNION_TAG);number(c);byte(c,OP_EQ);byte(c,OP_ASSERT);}
static void ws_forbid_error(Code *c){byte(c,OP_PUSH_BOOL);byte(c,0);byte(c,OP_ASSERT);}
static NvmModule *ws_program(const uint32_t *catalogs,unsigned count,bool indirect,bool loop,bool reverse,unsigned defect,NvmMultiNominalBindings *b){
 NvmModule *m=fixture_catalogs(reverse,b,catalogs,count);ws_contracts(m,b);
 NvmServicesNominalPlan *p=NULL;CHECK(nvm_services_nominal_plan(m,&p)==NVM_MULTI_NOMINAL_DESCRIBED);
 Code code[11]={0};Code *c=&code[0];
 for(unsigned i=0;i<count;i++){
  bool ws=catalogs[i]==3;NvmServicesNominalLayout type;
  if(catalogs[i]==2){
   for(unsigned j=0;j<7;j++)number(c);
   CHECK(nvm_services_nominal_type(p,i*9+8,&type));byte(c,OP_AGG_PACK);byte(c,AGG_RECORD);dword(c,type.source_ordinal);word(c,0);word(c,7);
  }else if(ws){push_text(c,m,"ws://127.0.0.1/test",19);ws_deadline(c);}
  svc(c,b,i,0,UINT16_MAX);local(c,OP_OWN_STORE_LOCAL,(uint16_t)i);
  uint32_t error=jump(c,OP_FILE_RESULT_BRANCH,(uint16_t)i);
  take(c,(uint16_t)i,0);local(c,OP_OWN_STORE_LOCAL,(uint16_t)(count+i));
  byte(c,OP_REGION_BEGIN);byte(c,OP_BORROW_LOCAL_EXCLUSIVE);word(c,20);word(c,(uint16_t)(count+i));
  if(ws){
   byte(c,OP_PUSH_BOOL);byte(c,1);push_text(c,m,"a\0b",3);
   unsigned message_instance=defect==1 && i==count-1?i-1:i;
   CHECK(nvm_services_nominal_type(p,message_instance*9+2,&type));
   byte(c,OP_AGG_PACK);byte(c,AGG_RECORD);dword(c,type.source_ordinal);word(c,0);word(c,2);
  }
  if(defect==2 && ws){byte(c,OP_PUSH_BOOL);byte(c,0);}else if(ws)ws_deadline(c);else number(c);
  unsigned target=count+1+i;
  if(defect==3 && i==count-1)target--;
  if(indirect){
   const char refs[]={20,0,(char)255,(char)255,(char)255,(char)255};unsigned params=ws?3:2;
   uint32_t map=nvm_add_string(m,refs,2*params);byte(c,OP_FUNCREF);dword(c,target);
   byte(c,OP_FILE_CALL_INDIRECT_REFS);word(c,(uint16_t)params);word(c,1);dword(c,map);
  }else{byte(c,OP_CALL_REF);dword(c,target);word(c,20);}
  byte(c,OP_POP);
  if(ws)ws_deadline(c);
  svc(c,b,i,2,20);
  if(ws){
   local(c,OP_STORE_LOCAL,(uint16_t)(2*count+i));uint32_t receive_error=jump(c,OP_FILE_RESULT_BRANCH,(uint16_t)(2*count+i));
   take(c,(uint16_t)(2*count+i),0);byte(c,OP_AGG_GET);word(c,1);byte(c,OP_DUP);byte(c,OP_STR_LEN);
   byte(c,OP_PUSH_I64);for(unsigned n=0;n<8;n++)byte(c,n?0:3);byte(c,OP_EQ);byte(c,OP_ASSERT);
   push_text(c,m,ws_fixture_reply,3);byte(c,OP_STR_EQ);byte(c,OP_ASSERT);
   uint32_t receive_join=jump(c,OP_JMP,0);destination(c,receive_error,c->n);take(c,(uint16_t)(2*count+i),1);byte(c,OP_POP);if(ws_fixture_require_success)ws_forbid_error(c);destination(c,receive_join,c->n);
  }else{byte(c,OP_POP);svc(c,b,i,3,20);byte(c,OP_POP);}
  if(!ws || !ws_fixture_require_success)local(c,OP_FILE_END_BORROW,20);
  byte(c,OP_REGION_END);
  local(c,OP_OWN_MOVE_LOCAL,(uint16_t)(count+i));owned_call(c,1+i,indirect);
  if(ws && defect!=4){
   if(defect==6){byte(c,OP_PUSH_I64);for(unsigned n=0;n<8;n++)byte(c,255);}
   else ws_deadline(c);
  }
  unsigned close_instance=defect==5 && i==count-1?i-1:i;
  svc(c,b,close_instance,ws?3:4,UINT16_MAX);if(ws && ws_fixture_require_success)ws_require_ok(c);else byte(c,OP_POP);
  uint32_t join=jump(c,OP_JMP,0);destination(c,error,c->n);take(c,(uint16_t)i,1);byte(c,OP_POP);if(ws && ws_fixture_require_success)ws_forbid_error(c);destination(c,join,c->n);
 }
 if(loop){byte(c,OP_PUSH_BOOL);byte(c,0);uint32_t back=jump(c,OP_JMP_TRUE,0);destination(c,back,0);}
 number(c);byte(c,OP_RET);
 for(unsigned i=0;i<count;i++){
  c=&code[1+i];local(c,OP_OWN_MOVE_LOCAL,0);byte(c,OP_RET);
  c=&code[1+count+i];for(unsigned j=1;j<m->functions[1+count+i].arity;j++)local(c,OP_LOAD_LOCAL,(uint16_t)j);
  svc(c,b,i,1,0);if(catalogs[i]==3 && ws_fixture_require_success){byte(c,OP_DUP);ws_require_ok(c);}
  byte(c,OP_RET);
 }
 nvm_services_nominal_plan_free(p);
 uint32_t size=0;for(unsigned f=0;f<m->function_count;f++)size+=code[f].n;
 free(m->code);m->code=malloc(size);CHECK(m->code);m->code_size=m->code_capacity=size;
 uint32_t at=0;for(unsigned f=0;f<m->function_count;f++){
  m->functions[f].code_offset=at;m->functions[f].code_length=code[f].n;memcpy(m->code+at,code[f].data,code[f].n);at+=code[f].n;
 }
 return m;
}
static uint8_t *ws_wire(NvmModule *m,size_t *size){
 NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);
 CHECK(nvm_v2_module_serialize(&v,NULL,0,size)==NVM_V2_OK);uint8_t *bytes=malloc(*size);CHECK(bytes);
 CHECK(nvm_v2_module_serialize(&v,bytes,*size,size)==NVM_V2_OK);nvm_v2_module_free(&v);return bytes;
}
static void ws_facts(NvmServicesIndirectFlow *r,const NvmMultiNominalBindings *b){
 unsigned methods[5]={0},timeouts=0;
 NvmServicesIndirectFlowSummary summary;CHECK(nvm_services_indirect_flow_summary(r,&summary) && !summary.runtime_admitted);
 for(unsigned f=0;f<summary.targets.functions;f++){
  NvmServicesCodeFunction fn;CHECK(nvm_services_indirect_flow_function(r,f,&fn));
  for(uint16_t i=0;i<fn.instruction_count;i++){
   NvmServicesCodeInstruction in;CHECK(nvm_services_indirect_flow_instruction(r,f,i,&in));if(in.decoded.opcode!=OP_FILE_SERVICE)continue;
   unsigned instance=in.catalog_ordinal/5,method=in.catalog_ordinal%5;CHECK(instance<b->count);methods[instance]++;
   CHECK(in.decoded.operands[0].u32==b->instances[instance].imports[method]);
   uint8_t variants;CHECK(nvm_services_indirect_flow_variant_count(r,f,i,&variants) && variants);
   for(uint8_t v=0;v<variants;v++){
    NvmServicesCyclicVariant fact;CHECK(nvm_services_indirect_flow_variant(r,f,i,v,&fact));
    CHECK(fact.body.has_obligation && fact.body.obligation.result.global_index==b->instances[instance].layouts[3+method]);
    bool ws=b->instances[instance].catalog==3;
    CHECK(!!(fact.body.pending_checks & NVM_SERVICES_FLOW_CHECK_TIMEOUT)==ws);
    CHECK(!!(fact.body.pending_checks & NVM_SERVICES_FLOW_CHECK_BYTE)==(!ws && method==1));
    if(ws){timeouts++;CHECK(!(fact.body.pending_checks & NVM_SERVICES_FLOW_CHECK_ENDPOINT));
     NvmServicesFlowInputState disposition=method==0?NVM_SERVICES_FLOW_INPUT_NONE:method==3?NVM_SERVICES_FLOW_INPUT_CONSUMED:NVM_SERVICES_FLOW_INPUT_PRESERVED;
     CHECK(fact.body.obligation.outcomes[0]==disposition && fact.body.obligation.outcomes[1]==disposition);
     CHECK(fact.body.obligation.owned_inputs==(method==3));CHECK(fact.body.obligation.borrowed_inputs==(method==1 || method==2));}
   }
  }
 }
 CHECK(timeouts);for(unsigned i=0;i<b->count;i++)CHECK(methods[i]==(b->instances[i].catalog==3?4u:5u));
}
static void ws_runtime_refusal(const uint8_t *bytes,size_t size,NvmServicesIndirectHostedPlan *p,const NvmMultiNominalBindings *b){
 NlServicePolicy policy={0};
 CHECK(nl_service_policy_read(bytes,size,true,true,true,&policy) && policy.profile==3 && policy.allowed && policy.requires_websocket);
 CHECK(policy.count==b->count);
 CHECK(nl_service_policy_read(bytes,size,true,true,false,&policy) && !policy.allowed && policy.requires_websocket);
 NvmWebSocketHostPolicy ws_policy={1,true,false,1000,NULL};
 NvmServicesHostGrant *policy_grant=NULL;
 CHECK(nl_service_policy_grant(&policy,&ws_policy,&policy_grant)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_grant_destroy(&policy_grant)==NVM_SERVICES_HOST_OK);
 CHECK(nl_service_policy_read(bytes,size,true,true,true,&policy));
 ws_policy.allow_lookup=true;
 CHECK(nl_service_policy_grant(&policy,&ws_policy,&policy_grant)==NVM_SERVICES_HOST_INVALID && !policy_grant);
 ws_policy.resolver_helper="/unused/helper";
 CHECK(nl_service_policy_grant(&policy,&ws_policy,&policy_grant)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_grant_destroy(&policy_grant)==NVM_SERVICES_HOST_OK);
 NvmServicesIndirectOptions options={1,100000};
 for(unsigned mode=0;mode<2;mode++){
  NvmServicesRuntime *runtime=NULL;
  CHECK(nvm_services_runtime_indirect_create(bytes,size,mode?NVM_SERVICES_RUNTIME_NATIVE:NVM_SERVICES_RUNTIME_VM,&options,&runtime)==NVM_SERVICES_RUNTIME_OK && runtime);
  CHECK(nvm_services_runtime_begin(runtime)==NVM_SERVICES_RUNTIME_INVALID);
  NvmServicesIndirectExecutionReport missing=nvm_services_runtime_indirect_destroy(&runtime,NULL);
  CHECK(!runtime && !missing.runtime.acquired && missing.runtime.status==NVM_SERVICES_RUNTIME_INVALID);
  CHECK(nvm_services_runtime_indirect_create(bytes,size,mode?NVM_SERVICES_RUNTIME_NATIVE:NVM_SERVICES_RUNTIME_VM,&options,&runtime)==NVM_SERVICES_RUNTIME_OK);
  NlWsTransportPolicy policy={.max_timeout_ms=60001};
  CHECK(nvm_services_runtime_websocket_policy(runtime,b->count,&policy)==NVM_SERVICES_RUNTIME_TYPE);
  for(unsigned i=0;i<b->count;i++){
   if(b->instances[i].catalog!=3){CHECK(nvm_services_runtime_websocket_policy(runtime,i,&policy)==NVM_SERVICES_RUNTIME_TYPE);continue;}
   CHECK(nvm_services_runtime_websocket_policy(runtime,i,NULL)==NVM_SERVICES_RUNTIME_INVALID);
   CHECK(nvm_services_runtime_websocket_policy(runtime,i,&policy)==NVM_SERVICES_RUNTIME_INVALID);
   policy=(NlWsTransportPolicy){.allow_lookup=true};
   CHECK(nvm_services_runtime_websocket_policy(runtime,i,&policy)==NVM_SERVICES_RUNTIME_INVALID);
   policy=(NlWsTransportPolicy){.resolver_helper="relative"};
   CHECK(nvm_services_runtime_websocket_policy(runtime,i,&policy)==NVM_SERVICES_RUNTIME_INVALID);
   char helper[]="/copied/helper";policy=(NlWsTransportPolicy){.resolver_helper=helper};
   CHECK(nvm_services_runtime_websocket_policy(runtime,i,&policy)==NVM_SERVICES_RUNTIME_OK);
   memset(helper,'?',sizeof helper);
   CHECK(nvm_services_runtime_websocket_policy(runtime,i,&policy)==NVM_SERVICES_RUNTIME_STATE);
   policy=(NlWsTransportPolicy){.max_timeout_ms=60001};
  }
  CHECK(nvm_services_runtime_begin(runtime)==NVM_SERVICES_RUNTIME_OK);
  CHECK(nvm_services_runtime_websocket_policy(runtime,0,&policy)==NVM_SERVICES_RUNTIME_STATE);
  missing=nvm_services_runtime_indirect_destroy(&runtime,NULL);
  CHECK(!runtime && missing.runtime.acquired && !missing.runtime.cleanup.cleanup_failures);
 }
 char *text=(void *)&checks,diagnostic[256];
 CHECK(nvm2c_emit_services_indirect_bytes(bytes,size,"mixed_ws",&text,diagnostic,sizeof diagnostic)==NVM_SERVICES_RUNTIME_OK && text!=(void *)&checks);
 free(text);
 NvmServicesHostPolicy policies[5];for(unsigned i=0;i<b->count;i++)policies[i]=(NvmServicesHostPolicy){b->instances[i].catalog==2?NVM_SERVICES_HOST_TCP:NVM_SERVICES_HOST_FILE,true};
 NvmServicesHostGrant *grant=NULL;CHECK(nvm_services_host_grant_create(policies,b->count,&grant)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_enter(grant,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_authorize(grant,p)==NVM_SERVICES_HOST_UNRESOLVED);nvm_services_host_leave();
 NvmServicesScalar scalar={TAG_INT,123};NvmServicesIndirectExecutionReport report=nvm_services_execute_indirect_bytes(grant,bytes,size,&options,&scalar);
 CHECK(report.runtime.status!=NVM_SERVICES_RUNTIME_OK && !report.runtime.acquired && scalar.tag==TAG_INT && scalar.value==123);
 CHECK(nvm_services_host_grant_destroy(&grant)==NVM_SERVICES_HOST_OK && !grant);
}
static NvmServicesHostConfig ws_policy(unsigned catalog,unsigned index) {
 return (NvmServicesHostConfig){.revision=NVM_SERVICES_HOST_POLICY_REVISION,
  .catalog=(NvmServicesHostCatalog)catalog,.allowed=true,.max_timeout_ms=catalog==3?17+index:0};
}
static void ws_grant_boundaries(void){
 NvmServicesHostGrant *g=NULL;NvmServicesHostConfig p=ws_policy(3,0);
 CHECK(nvm_services_host_grant_create_config(NULL,1,&g)==NVM_SERVICES_HOST_INVALID && !g);
 CHECK(nvm_services_host_grant_create_config(&p,0,&g)==NVM_SERVICES_HOST_INVALID && !g);
 CHECK(nvm_services_host_grant_create_config(&p,65,&g)==NVM_SERVICES_HOST_INVALID && !g);
 CHECK(nvm_services_host_grant_create_config(&p,1,NULL)==NVM_SERVICES_HOST_INVALID);
 g=(void *)&checks;CHECK(nvm_services_host_grant_create_config(&p,1,&g)==NVM_SERVICES_HOST_INVALID && g==(void *)&checks);g=NULL;
 NvmServicesHostPolicy legacy={NVM_SERVICES_HOST_WEBSOCKET,true};
 CHECK(nvm_services_host_grant_create(&legacy,1,&g)==NVM_SERVICES_HOST_INVALID && !g);
 char long_path[4097];memset(long_path,'x',sizeof long_path);long_path[0]='/';long_path[4096]=0;
 for(unsigned bad=0;bad<10;bad++){
  p=ws_policy(3,0);
  switch(bad){
   case 0:p.revision=0;break;case 1:p.catalog=99;break;case 2:p.max_timeout_ms=60001;break;
   case 3:p.allow_lookup=true;break;case 4:p.resolver_helper="";break;
   case 5:p.resolver_helper="relative";break;case 6:p.resolver_helper=long_path;break;
   case 7:p=ws_policy(1,0);p.allow_lookup=true;break;
   case 8:p=ws_policy(2,0);p.resolver_helper="/resolver";break;
   case 9:p=ws_policy(1,0);p.max_timeout_ms=1;break;
  }
  CHECK(nvm_services_host_grant_create_config(&p,1,&g)==NVM_SERVICES_HOST_INVALID && !g);
 }
 long_path[4095]=0;NvmServicesHostConfig all[64];
 for(unsigned i=0;i<64;i++){all[i]=ws_policy(3,i);all[i].allow_lookup=i%2;all[i].resolver_helper=long_path;}
 all[63].max_timeout_ms=60000;
#ifdef SERVICE_ALLOC_TEST
 budget=0;CHECK(nvm_services_host_grant_create_config(all,64,&g)==NVM_SERVICES_HOST_MEMORY && !g);budget=-1;
 budget=1;CHECK(nvm_services_host_grant_create_config(all,64,&g)==NVM_SERVICES_HOST_OK && g);budget=-1;
#else
 CHECK(nvm_services_host_grant_create_config(all,64,&g)==NVM_SERVICES_HOST_OK && g);
#endif
 memset(long_path,0,sizeof long_path);memset(all,0,sizeof all);
 CHECK(nvm_services_host_enter(g,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);
 CHECK(nvm_services_host_grant_create_config((void *)1,65,(void *)1)==NVM_SERVICES_HOST_BUSY);
 CHECK(nvm_services_host_grant_create((void *)1,65,(void *)1)==NVM_SERVICES_HOST_BUSY);
 CHECK(nvm_services_host_grant_revoke_instance((void *)1,99)==NVM_SERVICES_HOST_BUSY);
 CHECK(nvm_services_host_grant_revoke((void *)1)==NVM_SERVICES_HOST_BUSY);
 CHECK(nvm_services_host_grant_destroy((void *)1)==NVM_SERVICES_HOST_BUSY);
 for(unsigned i=0;i<64;i++){
  NlWsTransportPolicy actual={0};CHECK(nvm_services_host_websocket_policy(g,i,&actual)==NVM_SERVICES_HOST_OK);
  CHECK(actual.allow_network && actual.allow_lookup==(bool)(i%2));
  CHECK(actual.max_timeout_ms==(i==63?60000:17+i));
  CHECK(actual.resolver_helper && actual.resolver_helper[0]=='/' && strlen(actual.resolver_helper)==4095);
 }
 NlWsTransportPolicy out={.allow_network=true,.max_timeout_ms=123},saved=out;
 CHECK(nvm_services_host_websocket_policy(g,64,&out)==NVM_SERVICES_HOST_INVALID && !memcmp(&out,&saved,sizeof out));
 CHECK(nvm_services_host_websocket_policy(g,0,NULL)==NVM_SERVICES_HOST_INVALID);
 nvm_services_host_leave();
 CHECK(nvm_services_host_grant_revoke_instance(g,63)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_grant_revoke_instance(g,63)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_grant_revoke_instance(g,64)==NVM_SERVICES_HOST_INVALID);
 CHECK(nvm_services_host_enter(g,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_websocket_policy(g,63,&out)==NVM_SERVICES_HOST_OK && !out.allow_network);
 CHECK(nvm_services_host_websocket_policy(g,62,&out)==NVM_SERVICES_HOST_OK && out.allow_network);
 nvm_services_host_leave();
 CHECK(nvm_services_host_grant_revoke(g)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_grant_revoke(g)==NVM_SERVICES_HOST_OK);
 out=saved;CHECK(nvm_services_host_enter_query()==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_websocket_policy(g,0,&out)==NVM_SERVICES_HOST_STATE && !memcmp(&out,&saved,sizeof out));
 nvm_services_host_leave();CHECK(nvm_services_host_enter(g,1,3)==NVM_SERVICES_HOST_STATE);
 CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);
 CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);
 /* I retain separate connections and lookup, including an all-denied policy. */
 p=ws_policy(3,0);p.allowed=false;p.max_timeout_ms=0;
 CHECK(nvm_services_host_grant_create_config(&p,1,&g)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_enter(g,1,3)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_websocket_policy(g,0,&out)==NVM_SERVICES_HOST_OK && !out.allow_network && !out.allow_lookup && !out.resolver_helper && !out.max_timeout_ms);
 nvm_services_host_leave();CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK);
}
static void ws_grant_plan(NvmServicesIndirectHostedPlan *plan,const NvmMultiNominalBindings *b){
 NvmServicesHostConfig configs[6]={0};char paths[5][32];NvmServicesHostGrant *g=NULL;
 for(unsigned i=0;i<b->count;i++){
  configs[i]=ws_policy(b->instances[i].catalog,i);
  if(configs[i].catalog==NVM_SERVICES_HOST_WEBSOCKET){
   snprintf(paths[i],sizeof paths[i],"/resolver/%u",i);configs[i].resolver_helper=paths[i];configs[i].allow_lookup=i%2;
  }
 }
 CHECK(nvm_services_host_grant_create_config(configs,b->count,&g)==NVM_SERVICES_HOST_OK);
 memset(paths,'?',sizeof paths);
 CHECK(nvm_services_host_enter(g,1,3)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_authorize(g,plan)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_authorize(g,NULL)==NVM_SERVICES_HOST_INVALID);
 for(unsigned i=0;i<b->count;i++){
  NlWsTransportPolicy out={.max_timeout_ms=123},saved=out;
  if(configs[i].catalog==NVM_SERVICES_HOST_WEBSOCKET){
   char expected[32];snprintf(expected,sizeof expected,"/resolver/%u",i);
   CHECK(nvm_services_host_websocket_policy(g,i,&out)==NVM_SERVICES_HOST_OK);
   CHECK(out.allow_network && out.allow_lookup==(bool)(i%2) && out.max_timeout_ms==17+i && !strcmp(out.resolver_helper,expected));
  }else CHECK(nvm_services_host_websocket_policy(g,i,&out)==NVM_SERVICES_HOST_UNRESOLVED && !memcmp(&out,&saved,sizeof out));
 }
 nvm_services_host_leave();
 for(unsigned i=0;i<b->count;i++){
  CHECK(nvm_services_host_grant_revoke_instance(g,i)==NVM_SERVICES_HOST_OK);
  CHECK(nvm_services_host_enter(g,1,3)==NVM_SERVICES_HOST_OK);
  CHECK(nvm_services_host_authorize(g,plan)==NVM_SERVICES_HOST_STATE);nvm_services_host_leave();
 }
 CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK);
 for(unsigned i=0;i<b->count;i++){configs[i].resolver_helper=NULL;configs[i].allow_lookup=false;}
 for(unsigned i=0;i<b->count;i++){
  configs[i].allowed=false;
  CHECK(nvm_services_host_grant_create_config(configs,b->count,&g)==NVM_SERVICES_HOST_OK);
  CHECK(nvm_services_host_enter(g,1,3)==NVM_SERVICES_HOST_OK);
  CHECK(nvm_services_host_authorize(g,plan)==NVM_SERVICES_HOST_STATE);nvm_services_host_leave();
  CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK);configs[i].allowed=true;
 }
 /* I refuse a missing instance, an extra instance and each wrong catalog. */
 for(unsigned bad=0;bad<b->count+2;bad++){
  if(!bad && b->count==1)continue;
  NvmServicesHostConfig altered[6];memcpy(altered,configs,sizeof altered);unsigned count=b->count;
  if(!bad)count--;else if(bad==1){altered[count]=ws_policy(3,count);count++;}
  else{unsigned i=bad-2;altered[i]=ws_policy(configs[i].catalog==NVM_SERVICES_HOST_FILE?3:1,i);}
  CHECK(nvm_services_host_grant_create_config(altered,count,&g)==NVM_SERVICES_HOST_OK);
  CHECK(nvm_services_host_enter(g,1,3)==NVM_SERVICES_HOST_OK);
  CHECK(nvm_services_host_authorize(g,plan)==NVM_SERVICES_HOST_UNRESOLVED);nvm_services_host_leave();
  CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK);
 }
}
static void ws_checked(const uint32_t *catalogs,unsigned count,bool indirect,bool loop,bool reverse){
 NvmMultiNominalBindings b;NvmModule *m=ws_program(catalogs,count,indirect,loop,reverse,0,&b);
 NvmServicesIndirectFlow *r=NULL;NvmServicesFlowStatus status=nvm_services_indirect_flow_analyze(m,&r);
 if(status!=NVM_SERVICES_FLOW_OK)fprintf(stderr,"mixed WS flow status=%u instances=%u indirect=%u loop=%u reverse=%u\n",status,count,indirect,loop,reverse);
 OK(status);ws_facts(r,&b);nvm_services_indirect_flow_free(r);
 if(!indirect){NvmServicesCyclicReport *cyclic=NULL;OK(nvm_services_cyclic_analyze(m,&cyclic));nvm_services_cyclic_free(cyclic);
  if(!loop){NvmServicesBodyReport *body=NULL;OK(nvm_services_body_analyze(m,&body));nvm_services_body_free(body);}}
 size_t size=0;uint8_t *bytes=ws_wire(m,&size);NvmServicesIndirectHostedPlan *p=NULL;OK(nvm_services_indirect_hosted_prepare(bytes,size,&p));
 NvmServicesIndirectHostedStartup startup;CHECK(nvm_services_indirect_hosted_startup(p,&startup) && !startup.runtime_admitted);
 if(!indirect){NvmServicesCyclicHostedPlan *cyclic=NULL;OK(nvm_services_cyclic_hosted_prepare(bytes,size,&cyclic));nvm_services_cyclic_hosted_free(cyclic);
  if(!loop){NvmServicesHostedPlan *hosted=NULL;OK(nvm_services_hosted_prepare(bytes,size,&hosted));nvm_services_hosted_free(hosted);}}
 ws_runtime_refusal(bytes,size,p,&b);ws_grant_plan(p,&b);
 uint32_t literal=UINT32_MAX;
 for(uint32_t i=0;i<m->string_count;i++)if(m->string_lengths[i]==3 && !memcmp(m->strings[i],"a\0b",3)){literal=i;break;}
 CHECK(literal!=UINT32_MAX);
 memset(bytes,0,size);free(bytes);nvm_module_free(m);
 const uint8_t *retained=NULL;size_t length=0;
 CHECK(nvm_services_indirect_hosted_string(p,literal,&retained,&length) && length==3 && !memcmp(retained,"a\0b",3));
 for(unsigned i=0;i<count;i++){
  for(unsigned j=0;j<(catalogs[i]==3?4u:5u);j++){uint32_t index;CHECK(nvm_services_indirect_hosted_import(p,5*i+j,&index) && index==b.instances[i].imports[j]);}
  if(catalogs[i]==3){uint32_t index=123;CHECK(!nvm_services_indirect_hosted_import(p,5*i+4,&index) && index==123);}
 }
 nvm_services_indirect_hosted_free(p);
}
static void ws_unused(void){
 const uint32_t catalogs[]={3};NvmMultiNominalBindings b;NvmModule *m=fixture_catalogs(false,&b,catalogs,1);
 m->functions[0].arity=0;m->functions[0].local_count=0;CHECK(nvm_set_function_param_types(m,0,NULL,0));
 free(m->ownership_data);m->ownership_size=32;m->ownership_data=calloc(1,32);CHECK(m->ownership_data);
 wr32(m->ownership_data,1);wr32(m->ownership_data+4,7);
 for(unsigned i=0;i<7;i++)m->ownership_data[8+i]=(i==0||i==3)?3:1;
 wr32(m->ownership_data+16,1);descriptor(m->ownership_data+24,TAG_INT,0,NVM_V2_NO_INDEX);
 size_t size=0;uint8_t *bytes=ws_wire(m,&size);NvmServicesIndirectHostedPlan *p=NULL;OK(nvm_services_indirect_hosted_prepare(bytes,size,&p));
 /* No service instruction can hide an unsupported declared runtime catalog. */
 ws_runtime_refusal(bytes,size,p,&b);ws_grant_plan(p,&b);nvm_services_indirect_hosted_free(p);free(bytes);nvm_module_free(m);
}
#ifdef SERVICE_ALLOC_TEST
static void ws_allocations(void){
 const uint32_t catalogs[]={1,2,3,3};NvmMultiNominalBindings b;NvmModule *m=ws_program(catalogs,4,true,true,true,0,&b);
 bool success=false;
 for(int n=0;n<2048;n++){
  NvmServicesIndirectFlow *p=(void *)&checks;budget=n;NvmServicesFlowStatus status=nvm_services_indirect_flow_analyze(m,&p);budget=-1;
  if(status==NVM_SERVICES_FLOW_OK){nvm_services_indirect_flow_free(p);success=true;printf("mixed WS flow allocation prefixes: %d\n",n);break;}
  CHECK(p==(void *)&checks);
 }
 CHECK(success);size_t size=0;uint8_t *bytes=ws_wire(m,&size);success=false;
 for(int n=0;n<4096;n++){
  NvmServicesIndirectHostedPlan *p=(void *)&checks;budget=n;NvmServicesFlowStatus status=nvm_services_indirect_hosted_prepare(bytes,size,&p);budget=-1;
  if(status==NVM_SERVICES_FLOW_OK){nvm_services_indirect_hosted_free(p);success=true;printf("mixed WS hosted allocation prefixes: %d\n",n);break;}
  CHECK(p==(void *)&checks);
 }
 CHECK(success);free(bytes);nvm_module_free(m);
}
#endif
#ifndef MIXED_WEBSOCKET_FLOW_MAIN
#define MIXED_WEBSOCKET_FLOW_MAIN main
#endif
int MIXED_WEBSOCKET_FLOW_MAIN(void){
 ws_grant_boundaries();ws_unused();
 const uint32_t one[]={3},mixed[]={1,2,3,3},five[]={3,3,3,3,3};
 for(unsigned reverse=0;reverse<2;reverse++)for(unsigned indirect=0;indirect<2;indirect++)for(unsigned loop=0;loop<2;loop++){
  ws_checked(one,1,indirect,loop,reverse);ws_checked(mixed,4,indirect,loop,reverse);ws_checked(five,5,indirect,loop,reverse);
 }
 for(unsigned indirect=0;indirect<2;indirect++)for(unsigned defect=1;defect<=5;defect++){
  NvmMultiNominalBindings b;NvmModule *m=ws_program(mixed,4,indirect,true,true,defect,&b);NvmServicesIndirectFlow *p=(void *)&checks;
  CHECK(nvm_services_indirect_flow_analyze(m,&p)!=NVM_SERVICES_FLOW_OK && p==(void *)&checks);nvm_module_free(m);
 }
#ifdef SERVICE_ALLOC_TEST
 ws_allocations();
#endif
 printf("PASS %u mixed WebSocket flow and runtime refusal checks\n",checks);return 0;
}
