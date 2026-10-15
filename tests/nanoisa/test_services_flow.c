/* I reuse the exact mixed transport fixture and independently assemble CODE. */
#define main multi_transport_fixture_main
#include "test_multi_nominal.c"
#undef main
#include "../../src/nanoisa/services_indirect_hosted.h"
#include "../../src/nanoisa/services_cyclic_hosted.h"
#define OK(x) CHECK((x)==NVM_SERVICES_FLOW_OK)
typedef struct {uint8_t data[8192];uint32_t n;} Code;
static void byte(Code *c,uint8_t v){CHECK(c->n<sizeof c->data);c->data[c->n++]=v;}
static void word(Code *c,uint16_t v){byte(c,(uint8_t)v);byte(c,(uint8_t)(v>>8));}
static void dword(Code *c,uint32_t v){for(unsigned i=0;i<4;i++)byte(c,(uint8_t)(v>>(8*i)));}
static void local(Code *c,uint8_t op,uint16_t v){byte(c,op);word(c,v);}
static void number(Code *c){byte(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)byte(c,0);}
static void svc(Code *c,const NvmMultiNominalBindings *b,unsigned instance,unsigned method,uint16_t ref){byte(c,OP_FILE_SERVICE);dword(c,b->instances[instance].imports[method]);word(c,ref);}
static uint32_t jump(Code *c,uint8_t op,uint16_t slot){uint32_t pc=c->n;byte(c,op);if(op==OP_FILE_RESULT_BRANCH)word(c,slot);dword(c,0);return pc;}
static void destination(Code *c,uint32_t pc,uint32_t target){wr32(c->data+pc+(c->data[pc]==OP_FILE_RESULT_BRANCH?3:1),(uint32_t)((int32_t)target-(int32_t)pc));}
static void take(Code *c,uint16_t slot,uint8_t arm){local(c,OP_FILE_RESULT_TAKE,slot);byte(c,arm);}
static void owned_call(Code *c,unsigned f,bool indirect){byte(c,indirect?OP_FUNCREF:OP_CALL);dword(c,f);if(indirect){byte(c,OP_CALL_INDIRECT);word(c,1);word(c,1);}}
static void borrowed_call(Code *c,NvmModule *m,unsigned f,bool indirect){
 number(c);
 if(indirect){const char refs[]={20,0,(char)255,(char)255};uint32_t map=nvm_add_string(m,refs,4);CHECK(map!=UINT32_MAX);
  byte(c,OP_FUNCREF);dword(c,f);byte(c,OP_FILE_CALL_INDIRECT_REFS);word(c,2);word(c,1);dword(c,map);
 }else{byte(c,OP_CALL_REF);dword(c,f);word(c,20);}
}
static void contracts(NvmModule *m,const NvmMultiNominalBindings *b){
 free(m->function_param_types[0]);m->function_param_types[0]=NULL;
 for(unsigned f=0;f<7;f++){
  NvmFunctionEntry fn=m->functions[0];fn.arity=f?(f<4?1:2):0;fn.local_count=f?fn.arity:6;fn.result_count=1;fn.result_tag=f?(f<4?TAG_STRUCT:TAG_UNION):TAG_INT;
  if(f==0)m->functions[0]=fn;else CHECK(nvm_add_function(m,&fn)==f);
  uint8_t params[]={TAG_STRUCT,TAG_INT};CHECK(nvm_set_function_param_types(m,f,params,fn.arity));
 }
 /* 26 layout flags, padded to28; one main + six helper declarations. */
 size_t size=40;for(unsigned f=0;f<7;f++)size+=12+8*m->functions[f].local_count;
 free(m->ownership_data);m->ownership_data=calloc(1,size);CHECK(m->ownership_data);m->ownership_size=(uint32_t)size;uint8_t *o=m->ownership_data;
 wr32(o,1);wr32(o+4,26);for(unsigned i=0;i<3;i++)for(unsigned j=0;j<types(&b->instances[i]);j++)o[8+b->instances[i].layouts[j]]=(j==0||j==3)?3:1;
 wr32(o+36,7);size_t at=40;
 for(unsigned f=0;f<7;f++){
  const NvmFunctionEntry *fn=&m->functions[f];o[at]=(uint8_t)fn->local_count;o[at+2]=(uint8_t)fn->arity;
  unsigned instance=f?(f<4?f-1:f-4):0;
  descriptor(o+at+4,fn->result_tag,0,f?b->instances[instance].layouts[f<4?0:4]:NVM_V2_NO_INDEX);
  for(unsigned j=0;j<fn->local_count;j++){
   uint8_t tag=TAG_STRUCT,mode=0;uint32_t layout;
   if(!f){tag=j<3?TAG_UNION:TAG_STRUCT;layout=b->instances[j%3].layouts[j<3?3:0];}
   else if(f<4)layout=b->instances[instance].layouts[0];
   else if(!j){layout=b->instances[instance].layouts[0];mode=2;}
   else{layout=NVM_V2_NO_INDEX;tag=TAG_INT;}
   descriptor(o+at+12+8*j,tag,mode,layout);
  }
  at+=12+8*fn->local_count;
 }
 CHECK(at==size);
}
static NvmModule *mixed_program(bool indirect,bool loop,bool reverse,unsigned defect,NvmMultiNominalBindings *b){
 NvmModule *m=fixture(reverse,b);contracts(m,b);Code code[7]={0};Code *c=&code[0];
 NvmServicesNominalPlan *nominal=NULL;CHECK(nvm_services_nominal_plan(m,&nominal)==NVM_MULTI_NOMINAL_DESCRIBED);
 for(unsigned i=0;i<3;i++){
  if(i==1){for(unsigned j=0;j<7;j++)number(c);NvmServicesNominalLayout endpoint;CHECK(nvm_services_nominal_type(nominal,17,&endpoint));
   byte(c,OP_AGG_PACK);byte(c,AGG_RECORD);dword(c,endpoint.source_ordinal);word(c,0);word(c,7);}
  svc(c,b,i,0,UINT16_MAX);local(c,OP_OWN_STORE_LOCAL,(uint16_t)(defect==3 && i==1?0:i));
  uint32_t error=jump(c,OP_FILE_RESULT_BRANCH,(uint16_t)i);take(c,(uint16_t)i,0);local(c,OP_OWN_STORE_LOCAL,(uint16_t)(3+i));
  byte(c,OP_REGION_BEGIN);byte(c,OP_BORROW_LOCAL_EXCLUSIVE);word(c,20);word(c,(uint16_t)(3+i));
  borrowed_call(c,m,defect==1 && !i?6:4+i,indirect);byte(c,OP_POP);
  svc(c,b,i,2,20);byte(c,OP_POP);svc(c,b,i,3,20);byte(c,OP_POP);
  local(c,OP_FILE_END_BORROW,20);byte(c,OP_REGION_END);
  local(c,OP_OWN_MOVE_LOCAL,(uint16_t)(3+i));owned_call(c,defect==2 && !i?3:1+i,indirect);
  svc(c,b,defect==4 && !i?2:i,4,UINT16_MAX);byte(c,OP_POP);
  uint32_t join=jump(c,OP_JMP,0);destination(c,error,c->n);take(c,(uint16_t)i,1);byte(c,OP_POP);destination(c,join,c->n);
 }
 if(loop){byte(c,OP_PUSH_BOOL);byte(c,0);uint32_t back=jump(c,OP_JMP_TRUE,0);destination(c,back,0);}
 number(c);byte(c,OP_RET);nvm_services_nominal_plan_free(nominal);
 for(unsigned i=0;i<3;i++){
  c=&code[1+i];local(c,OP_OWN_MOVE_LOCAL,0);byte(c,OP_RET);
  c=&code[4+i];local(c,OP_LOAD_LOCAL,1);svc(c,b,i,1,0);byte(c,OP_RET);
 }
 uint32_t size=0;for(unsigned f=0;f<7;f++)size+=code[f].n;
 free(m->code);m->code=malloc(size);CHECK(m->code);m->code_size=m->code_capacity=size;uint32_t at=0;
 for(unsigned f=0;f<7;f++){m->functions[f].code_offset=at;m->functions[f].code_length=code[f].n;memcpy(m->code+at,code[f].data,code[f].n);at+=code[f].n;}
 return m;
}
static void logical(void){
 NvmMultiNominalBindings b;NvmModule *m=mixed_program(false,false,false,0,&b);NvmServicesFlowDeclarations *d=NULL;OK(nvm_services_flow_declarations(m,&d));
 NvmServicesFlowState *s=NULL;OK(nvm_services_flow_state(d,4,&s)); /* exclusive File instance0 + int */
 OK(nvm_services_flow_load(s,1));NvmServicesFlowState *prior=NULL;OK(nvm_services_flow_clone(s,&prior));
 CHECK(nvm_services_flow_service(s,9,b.instances[2].imports[1],0)==NVM_SERVICES_FLOW_INVALID);
 bool changed=true;OK(nvm_services_flow_join(s,prior,&changed));CHECK(!changed);nvm_services_flow_state_free(prior);
 OK(nvm_services_flow_service(s,9,b.instances[0].imports[1],0));NvmServicesFlowObligation fact;CHECK(nvm_services_flow_obligation(s,0,&fact));
 CHECK(fact.result.global_index==b.instances[0].layouts[4] && (fact.checks&NVM_SERVICES_FLOW_CHECK_BYTE));OK(nvm_services_flow_can_exit(s));nvm_services_flow_state_free(s);
 OK(nvm_services_flow_state(d,1,&s));OK(nvm_services_flow_take(s,0));OK(nvm_services_flow_clone(s,&prior));
 CHECK(nvm_services_flow_service(s,10,b.instances[2].imports[4],UINT16_MAX)==NVM_SERVICES_FLOW_INVALID);
 changed=true;OK(nvm_services_flow_join(s,prior,&changed));CHECK(!changed);nvm_services_flow_state_free(prior);
 OK(nvm_services_flow_service(s,10,b.instances[0].imports[4],UINT16_MAX));CHECK(nvm_services_flow_obligation(s,0,&fact) && fact.owned_inputs==1);nvm_services_flow_state_free(s);
 OK(nvm_services_flow_state(d,0,&s));
 for(unsigned instance=0;instance<3;instance++){
  const NlServicePlanType *error=type(&b.instances[instance],1);
  for(unsigned j=0;j<error->member_count;j++)OK(nvm_services_flow_push_scalar(s,!strcmp(error->members[j].type_id,"nsi:core/bool")?TAG_BOOL:TAG_INT));
  OK(nvm_services_flow_construct(s,instance*9+1,0));NvmServicesFlowValue value;
  CHECK(nvm_services_flow_stack(s,0,&value) && value.type.global_index==b.instances[instance].layouts[1]);
  if(instance==2){OK(nvm_services_flow_clone(s,&prior));CHECK(nvm_services_flow_construct(s,4,1)==NVM_SERVICES_FLOW_INVALID);
   changed=true;OK(nvm_services_flow_join(s,prior,&changed));CHECK(!changed);nvm_services_flow_state_free(prior);}
  OK(nvm_services_flow_construct(s,instance*9+4,1));CHECK(nvm_services_flow_stack(s,0,&value) && value.type.global_index==b.instances[instance].layouts[4]);
  OK(nvm_services_flow_pop(s));
 }
 nvm_services_flow_state_free(s);
 nvm_services_flow_declarations_free(d);nvm_module_free(m);
}
static void facts(NvmServicesIndirectFlow *r,const NvmMultiNominalBindings *b){
 NvmServicesIndirectFlowSummary summary;CHECK(nvm_services_indirect_flow_summary(r,&summary) && !summary.runtime_admitted && summary.targets.functions==7);
 unsigned services=0,endpoint=0,byte_checks=0;
 for(unsigned f=0;f<7;f++){
  NvmServicesCodeFunction fn;CHECK(nvm_services_indirect_flow_function(r,f,&fn));
  for(uint16_t i=0;i<fn.instruction_count;i++){
   NvmServicesCodeInstruction in;CHECK(nvm_services_indirect_flow_instruction(r,f,i,&in));
   if(in.decoded.opcode!=OP_FILE_SERVICE)continue;
   unsigned instance=in.catalog_ordinal/5,method=in.catalog_ordinal%5;CHECK(instance<3);
   uint8_t variants=0;CHECK(nvm_services_indirect_flow_variant_count(r,f,i,&variants) && variants);
   for(uint8_t v=0;v<variants;v++){
    NvmServicesCyclicVariant fact;CHECK(nvm_services_indirect_flow_variant(r,f,i,v,&fact));
    CHECK(fact.body.has_obligation && fact.body.obligation.result.global_index==b->instances[instance].layouts[3+method]);
    CHECK(fact.body.obligation.target==b->instances[instance].imports[method]);
    if(instance==1 && !method){CHECK(fact.body.pending_checks&NVM_SERVICES_FLOW_CHECK_ENDPOINT);endpoint++;}
    if(method==1){CHECK(fact.body.pending_checks&NVM_SERVICES_FLOW_CHECK_BYTE);byte_checks++;}
   }
   services++;
  }
 }
 CHECK(services==15 && endpoint && byte_checks==3);
}
static void complete(void){
 for(unsigned reverse=0;reverse<2;reverse++)for(unsigned indirect=0;indirect<2;indirect++)for(unsigned loop=0;loop<2;loop++){
  NvmMultiNominalBindings b;NvmModule *m=mixed_program(indirect,loop,reverse,0,&b);
  NvmServicesIndirectFlow *r=NULL;NvmServicesFlowStatus status=nvm_services_indirect_flow_analyze(m,&r);
  if(status!=NVM_SERVICES_FLOW_OK)fprintf(stderr,"flow status %u reverse=%u indirect=%u loop=%u\n",status,reverse,indirect,loop);
  OK(status);facts(r,&b);nvm_services_indirect_flow_free(r);
  if(!indirect){
   NvmServicesCyclicReport *cyclic=NULL;OK(nvm_services_cyclic_analyze(m,&cyclic));nvm_services_cyclic_free(cyclic);
   if(!loop){NvmServicesCodePlan *code=NULL;OK(nvm_services_code_prepare(m,&code));nvm_services_code_free(code);
    NvmServicesBodyReport *body=NULL;OK(nvm_services_body_analyze(m,&body));nvm_services_body_free(body);}
  }
  NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);size_t n=0;CHECK(nvm_v2_module_serialize(&v,NULL,0,&n)==NVM_V2_OK);
  uint8_t *wire=malloc(n);CHECK(wire);CHECK(nvm_v2_module_serialize(&v,wire,n,&n)==NVM_V2_OK);
  if(!indirect){NvmServicesCyclicHostedPlan *cyclic=NULL;OK(nvm_services_cyclic_hosted_prepare(wire,n,&cyclic));nvm_services_cyclic_hosted_free(cyclic);
   if(!loop){NvmServicesHostedPlan *hosted=NULL;OK(nvm_services_hosted_prepare(wire,n,&hosted));nvm_services_hosted_free(hosted);}}
  NvmServicesIndirectHostedPlan *p=NULL;OK(nvm_services_indirect_hosted_prepare(wire,n,&p));
  NvmServicesIndirectHostedStartup startup;CHECK(nvm_services_indirect_hosted_startup(p,&startup) && startup.functions==7 && startup.input_bytes==n);
  nvm_v2_module_free(&v);memset(wire,0,n);free(wire);nvm_module_free(m);
  for(unsigned instance=0;instance<3;instance++){NvmServicesNominalLayout row;CHECK(nvm_services_indirect_hosted_type(p,instance*9,&row));CHECK(row.catalog_ordinal==instance*9 && row.global_index==b.instances[instance].layouts[0]);}
  nvm_services_indirect_hosted_free(p);
 }
 for(unsigned indirect=0;indirect<2;indirect++)for(unsigned defect=1;defect<=4;defect++){
  NvmMultiNominalBindings b;NvmModule *m=mixed_program(indirect,true,false,defect,&b);NvmServicesIndirectFlow *r=(void *)&checks;
  CHECK(nvm_services_indirect_flow_analyze(m,&r)!=NVM_SERVICES_FLOW_OK && r==(void *)&checks);nvm_module_free(m);
 }
}
static void expand_instances(NvmModule *m,NvmMultiNominalBindings *b,unsigned count){
 CHECK(count>=3 && count<=64);NvmV2Layouts layouts={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&layouts)==NVM_V2_OK);
 uint32_t total=26+8*(count-3);NvmV2Layout *items=realloc(layouts.items,total*sizeof *items);CHECK(items);layouts.items=items;
 memset(items+26,0,(total-26)*sizeof *items);
 for(unsigned i=3;i<count;i++){
  NvmServiceInstance *v=&b->instances[i];v->catalog=1;v->layouts[8]=UINT32_MAX;
  for(unsigned j=0;j<8;j++)v->layouts[j]=26+8*(i-3)+j;
  for(unsigned j=0;j<5;j++){
   unsigned source=b->instances[0].imports[j];NvmImportEntry old=m->imports[source];
   v->imports[j]=nvm_add_import(m,old.module_name_idx,old.function_name_idx,old.param_count,old.return_type,m->import_param_types[source]);
   CHECK(v->imports[j]!=UINT32_MAX);m->imports[v->imports[j]].kind=NVM_IMPORT_SERVICE;
  }
  for(unsigned j=0;j<8;j++){
   NvmV2Layout *to=&items[v->layouts[j]],*from=&items[b->instances[0].layouts[j]];*to=*from;
   to->fields=calloc(to->field_count,sizeof *to->fields);CHECK(to->fields || !to->field_count);
   for(unsigned k=0;k<to->field_count;k++){
    to->fields[k]=from->fields[k];uint32_t old=to->fields[k].nested_idx;
    if(old!=NVM_V2_NO_INDEX)for(unsigned t=0;t<8;t++)if(old==b->instances[0].layouts[t])to->fields[k].nested_idx=v->layouts[t];
   }
  }
  m->struct_count+=3;m->union_count+=5;
 }
 layouts.count=total;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&layouts);
 size_t flags=(total+3u)&~3u,added=flags-28;uint8_t *ownership=calloc(1,m->ownership_size+added);CHECK(ownership);
 memcpy(ownership,m->ownership_data,8);wr32(ownership+4,total);memcpy(ownership+8,m->ownership_data+8,26);
 for(unsigned i=3;i<count;i++)for(unsigned j=0;j<8;j++)ownership[8+b->instances[i].layouts[j]]=(j==0||j==3)?3:1;
 memcpy(ownership+8+flags,m->ownership_data+36,m->ownership_size-36);free(m->ownership_data);m->ownership_data=ownership;m->ownership_size+=(uint32_t)added;
 b->count=count;size_t size=0;CHECK(nvm_multi_nominal_encode(b,NULL,0,&size)==NVM_SERVICE_OK);free(m->service_data);m->service_data=malloc(size);CHECK(m->service_data);
 CHECK(nvm_multi_nominal_encode(b,m->service_data,size,&size)==NVM_SERVICE_OK);m->service_size=(uint32_t)size;
}
static void wide_hosted(void){
 NvmMultiNominalBindings b;NvmModule *m=mixed_program(true,true,false,0,&b);expand_instances(m,&b,64);
 CHECK(m->import_count==320);NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);size_t n=0;
 CHECK(nvm_v2_module_serialize(&v,NULL,0,&n)==NVM_V2_OK);uint8_t *wire=malloc(n);CHECK(wire);CHECK(nvm_v2_module_serialize(&v,wire,n,&n)==NVM_V2_OK);
 NvmServicesIndirectHostedPlan *p=NULL;OK(nvm_services_indirect_hosted_prepare(wire,n,&p));NvmServicesIndirectHostedStartup info;
 CHECK(nvm_services_indirect_hosted_startup(p,&info) && info.allocation_bound<=NVM_SERVICES_HOSTED_BYTES);
 for(unsigned i=0;i<64;i++){
  NvmServicesNominalLayout row;CHECK(nvm_services_indirect_hosted_type(p,i*9,&row) && row.global_index==b.instances[i].layouts[0]);
  uint32_t import=UINT32_MAX;CHECK(nvm_services_indirect_hosted_import(p,i*5+4,&import) && import==b.instances[i].imports[4]);
 }
 uint8_t *copy=malloc(n);CHECK(copy);CHECK(nvm_services_indirect_hosted_bytes(p,0,copy,n) && !memcmp(copy,wire,n));free(copy);
 nvm_services_indirect_hosted_free(p);p=(void *)&checks;
 CHECK(nvm_services_indirect_hosted_prepare(wire,n-1,&p)!=NVM_SERVICES_FLOW_OK && p==(void *)&checks);
 free(wire);nvm_v2_module_free(&v);nvm_module_free(m);
}

#ifdef SERVICE_ALLOC_TEST
static void flow_allocations(void){
 NvmMultiNominalBindings b;NvmModule *m=mixed_program(true,true,false,0,&b);bool success=false;
 for(int n=0;n<2048;n++){
  NvmServicesIndirectFlow *r=(void *)&checks;budget=n;NvmServicesFlowStatus status=nvm_services_indirect_flow_analyze(m,&r);budget=-1;
  if(status==NVM_SERVICES_FLOW_OK){nvm_services_indirect_flow_free(r);success=true;printf("allocation prefixes: %d\n",n);break;}
  CHECK(r==(void *)&checks);
 }
 CHECK(success);
 NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);size_t bytes=0;
 CHECK(nvm_v2_module_serialize(&v,NULL,0,&bytes)==NVM_V2_OK);uint8_t *wire=malloc(bytes);CHECK(wire);
 CHECK(nvm_v2_module_serialize(&v,wire,bytes,&bytes)==NVM_V2_OK);success=false;
 for(int n=0;n<2048;n++){
  NvmServicesIndirectHostedPlan *p=(void *)&checks;budget=n;NvmServicesFlowStatus status=nvm_services_indirect_hosted_prepare(wire,bytes,&p);budget=-1;
  if(status==NVM_SERVICES_FLOW_OK){nvm_services_indirect_hosted_free(p);success=true;printf("hosted allocation prefixes: %d\n",n);break;}
  CHECK(p==(void *)&checks);
 }
 CHECK(success);free(wire);nvm_v2_module_free(&v);nvm_module_free(m);
}
#endif
#ifndef SERVICES_FLOW_MAIN
#define SERVICES_FLOW_MAIN main
#endif
int SERVICES_FLOW_MAIN(void){logical();complete();wide_hosted();
#ifdef SERVICE_ALLOC_TEST
 flow_allocations();
#endif
 printf("PASS %u mixed flow/CODE/cyclic/indirect/hosted checks; no execution\n",checks);return 0;}
