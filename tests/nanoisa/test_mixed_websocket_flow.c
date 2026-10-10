/* I check all three catalogs in one graph before runtime admission. */
#define SERVICES_FLOW_MAIN file_tcp_flow_fixture_main
#include "test_services_flow.c"
#undef SERVICES_FLOW_MAIN
#include "../../src/nanoisa/services_indirect_public.h"
#include "../../src/nanoisa/services_host_grant_internal.h"
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
static NvmModule *ws_program(const uint32_t *catalogs,unsigned count,bool indirect,bool loop,bool reverse,unsigned defect,NvmMultiNominalBindings *b){
 NvmModule *m=fixture_catalogs(reverse,b,catalogs,count);ws_contracts(m,b);
 NvmServicesNominalPlan *p=NULL;CHECK(nvm_services_nominal_plan(m,&p)==NVM_MULTI_NOMINAL_DESCRIBED);
 Code code[11]={0};Code *c=&code[0];
 for(unsigned i=0;i<count;i++){
  bool ws=catalogs[i]==3;NvmServicesNominalLayout type;
  if(catalogs[i]==2){
   for(unsigned j=0;j<7;j++)number(c);
   CHECK(nvm_services_nominal_type(p,i*9+8,&type));byte(c,OP_AGG_PACK);byte(c,AGG_RECORD);dword(c,type.source_ordinal);word(c,0);word(c,7);
  }else if(ws){push_text(c,m,"ws://127.0.0.1/test",19);number(c);}
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
  if(defect==2 && ws){byte(c,OP_PUSH_BOOL);byte(c,0);}else number(c);
  unsigned target=count+1+i;
  if(defect==3 && i==count-1)target--;
  if(indirect){
   const char refs[]={20,0,(char)255,(char)255,(char)255,(char)255};unsigned params=ws?3:2;
   uint32_t map=nvm_add_string(m,refs,2*params);byte(c,OP_FUNCREF);dword(c,target);
   byte(c,OP_FILE_CALL_INDIRECT_REFS);word(c,(uint16_t)params);word(c,1);dword(c,map);
  }else{byte(c,OP_CALL_REF);dword(c,target);word(c,20);}
  byte(c,OP_POP);
  if(ws)number(c);
  svc(c,b,i,2,20);
  if(ws){
   local(c,OP_STORE_LOCAL,(uint16_t)(2*count+i));uint32_t receive_error=jump(c,OP_FILE_RESULT_BRANCH,(uint16_t)(2*count+i));
   take(c,(uint16_t)(2*count+i),0);byte(c,OP_AGG_GET);word(c,1);byte(c,OP_STR_LEN);byte(c,OP_POP);
   uint32_t receive_join=jump(c,OP_JMP,0);destination(c,receive_error,c->n);take(c,(uint16_t)(2*count+i),1);byte(c,OP_POP);destination(c,receive_join,c->n);
  }else{byte(c,OP_POP);svc(c,b,i,3,20);byte(c,OP_POP);}
  local(c,OP_FILE_END_BORROW,20);byte(c,OP_REGION_END);
  local(c,OP_OWN_MOVE_LOCAL,(uint16_t)(count+i));owned_call(c,1+i,indirect);
  if(ws && defect!=4)number(c);
  unsigned close_instance=defect==5 && i==count-1?i-1:i;
  svc(c,b,close_instance,ws?3:4,UINT16_MAX);byte(c,OP_POP);
  uint32_t join=jump(c,OP_JMP,0);destination(c,error,c->n);take(c,(uint16_t)i,1);byte(c,OP_POP);destination(c,join,c->n);
 }
 if(loop){byte(c,OP_PUSH_BOOL);byte(c,0);uint32_t back=jump(c,OP_JMP_TRUE,0);destination(c,back,0);}
 number(c);byte(c,OP_RET);
 for(unsigned i=0;i<count;i++){
  c=&code[1+i];local(c,OP_OWN_MOVE_LOCAL,0);byte(c,OP_RET);
  c=&code[1+count+i];for(unsigned j=1;j<m->functions[1+count+i].arity;j++)local(c,OP_LOAD_LOCAL,(uint16_t)j);
  svc(c,b,i,1,0);byte(c,OP_RET);
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
 NlServicePolicy policy={0},saved_policy=policy;
 CHECK(!nl_service_policy_read(bytes,size,true,true,true,&policy) && !memcmp(&policy,&saved_policy,sizeof policy));
 NvmServicesIndirectOptions options={1,100000};
 for(unsigned mode=0;mode<2;mode++){
  NvmServicesRuntime *runtime=(void *)&checks;
  CHECK(nvm_services_runtime_indirect_create(bytes,size,mode?NVM_SERVICES_RUNTIME_NATIVE:NVM_SERVICES_RUNTIME_VM,&options,&runtime)!=NVM_SERVICES_RUNTIME_OK && runtime==(void *)&checks);
 }
 char *text=(void *)&checks,diagnostic[256];
 CHECK(nvm2c_emit_services_indirect_bytes(bytes,size,"mixed_ws",&text,diagnostic,sizeof diagnostic)!=NVM_SERVICES_RUNTIME_OK && text==(void *)&checks);
 NvmServicesHostPolicy policies[5];for(unsigned i=0;i<b->count;i++)policies[i]=(NvmServicesHostPolicy){b->instances[i].catalog==2?NVM_SERVICES_HOST_TCP:NVM_SERVICES_HOST_FILE,true};
 NvmServicesHostGrant *grant=NULL;CHECK(nvm_services_host_grant_create(policies,b->count,&grant)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_enter(grant,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_authorize(grant,p)==NVM_SERVICES_HOST_UNRESOLVED);nvm_services_host_leave();
 NvmServicesScalar scalar={TAG_INT,123};NvmServicesIndirectExecutionReport report=nvm_services_execute_indirect_bytes(grant,bytes,size,&options,&scalar);
 CHECK(report.runtime.status!=NVM_SERVICES_RUNTIME_OK && !report.runtime.acquired && scalar.tag==TAG_INT && scalar.value==123);
 CHECK(nvm_services_host_grant_destroy(&grant)==NVM_SERVICES_HOST_OK && !grant);
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
 ws_runtime_refusal(bytes,size,p,&b);
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
 ws_runtime_refusal(bytes,size,p,&b);nvm_services_indirect_hosted_free(p);free(bytes);nvm_module_free(m);
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
int main(void){
 ws_unused();
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
