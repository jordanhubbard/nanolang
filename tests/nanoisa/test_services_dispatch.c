#define SERVICES_FLOW_MAIN services_flow_fixture_main
#include "test_services_flow.c"
#undef OK
#include "../../src/nanovm/services_vm_indirect_private.h"
#include "../../src/nanoisa/nvm2c_services_indirect_private.h"
static void integer(Code *c,int64_t value){number(c);uint64_t bits;memcpy(&bits,&value,8);for(unsigned i=0;i<8;i++)c->data[c->n-8+i]=(uint8_t)(bits>>(8*i));}
static void assertion(Code *c){byte(c,OP_PUSH_BOOL);byte(c,0);byte(c,OP_ASSERT);}
static void runtime_contracts(NvmModule *m,const NvmMultiNominalBindings *b){
 contracts(m,b);m->functions[0].local_count=18;
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
   if(!f){unsigned type=j<3?3:j<6?0:j/3+2;tag=j>=3 && j<6?TAG_STRUCT:TAG_UNION;layout=b->instances[j%3].layouts[type];}
   else if(f<4)layout=b->instances[instance].layouts[0];
   else if(!j){layout=b->instances[instance].layouts[0];mode=2;}
   else{layout=NVM_V2_NO_INDEX;tag=TAG_INT;}
   descriptor(o+at+12+8*j,tag,mode,layout);
  }
  at+=12+8*fn->local_count;
 }
 CHECK(at==size);
}
/* Every scalar Result arm is consumed; TCP readiness/data errors retry under
 * invocation fuel. File errors and any mismatched observed payload assert. */
static void result(Code *c,uint16_t slot,unsigned kind,int64_t expected,uint32_t retry){
 local(c,OP_STORE_LOCAL,slot);uint32_t error=jump(c,OP_FILE_RESULT_BRANCH,slot);take(c,slot,0);
 if(kind){if(kind==2)local(c,OP_AGG_GET,0);integer(c,expected);byte(c,OP_EQ);byte(c,OP_ASSERT);}else byte(c,OP_POP);
 uint32_t done=jump(c,OP_JMP,0);destination(c,error,c->n);take(c,slot,1);byte(c,OP_POP);
 if(retry!=UINT32_MAX){uint32_t back=jump(c,OP_JMP,0);destination(c,back,retry);}else assertion(c);
 destination(c,done,c->n);
}
static NvmModule *program(unsigned mode,unsigned port,bool ipv6){
 bool indirect=mode&1,reverse=mode&2,trap=mode&4;NvmMultiNominalBindings b;
 NvmModule *m=fixture(reverse,&b);runtime_contracts(m,&b);Code code[7]={0};Code *c=&code[0];
 NvmServicesNominalPlan *nominal=NULL;CHECK(nvm_services_nominal_plan(m,&nominal)==NVM_MULTI_NOMINAL_DESCRIBED);
 for(unsigned i=0;i<3;i++){
  if(i==1){int64_t v[]={ipv6?6:4,ipv6?0:0x7f000001,0,0,ipv6?1:0,port,0};for(unsigned j=0;j<7;j++)integer(c,v[j]);
   NvmServicesNominalLayout endpoint;CHECK(nvm_services_nominal_type(nominal,17,&endpoint));
   byte(c,OP_AGG_PACK);byte(c,AGG_RECORD);dword(c,endpoint.source_ordinal);word(c,0);word(c,7);}
  svc(c,&b,i,0,UINT16_MAX);local(c,OP_OWN_STORE_LOCAL,(uint16_t)i);
  uint32_t error=jump(c,OP_FILE_RESULT_BRANCH,(uint16_t)i);take(c,(uint16_t)i,0);local(c,OP_OWN_STORE_LOCAL,(uint16_t)(3+i));
  uint32_t success=jump(c,OP_JMP,0);destination(c,error,c->n);take(c,(uint16_t)i,1);byte(c,OP_POP);
  for(unsigned j=0;j<i;j++)local(c,OP_FILE_DROP_LOCAL,(uint16_t)(3+j));
  assertion(c);integer(c,-1);byte(c,OP_RET);destination(c,success,c->n);
 }
 for(unsigned i=0;i<3;i++){
  byte(c,OP_REGION_BEGIN);byte(c,OP_BORROW_LOCAL_EXCLUSIVE);word(c,20);word(c,(uint16_t)(3+i));
  if(i==1){uint32_t retry=c->n;svc(c,&b,i,2,20);result(c,(uint16_t)(9+i),0,0,retry);if(trap)assertion(c);}
  uint32_t retry=c->n;borrowed_call(c,m,4+i,indirect);result(c,(uint16_t)(6+i),1,1,i==1?retry:UINT32_MAX);
  if(i!=1){svc(c,&b,i,2,20);result(c,(uint16_t)(9+i),0,0,UINT32_MAX);}
  retry=c->n;svc(c,&b,i,3,20);result(c,(uint16_t)(12+i),2,0,i==1?retry:UINT32_MAX);
  local(c,OP_FILE_END_BORROW,20);byte(c,OP_REGION_END);
  local(c,OP_OWN_MOVE_LOCAL,(uint16_t)(3+i));owned_call(c,1+i,indirect);
  svc(c,&b,i,4,UINT16_MAX);result(c,(uint16_t)(15+i),0,0,UINT32_MAX);
 }
 integer(c,42);byte(c,OP_RET);nvm_services_nominal_plan_free(nominal);
 for(unsigned i=0;i<3;i++){
  c=&code[1+i];local(c,OP_OWN_MOVE_LOCAL,0);byte(c,OP_RET);
  c=&code[4+i];local(c,OP_LOAD_LOCAL,1);svc(c,&b,i,1,0);byte(c,OP_RET);
 }
 uint32_t size=0;for(unsigned f=0;f<7;f++)size+=code[f].n;
 free(m->code);m->code=malloc(size);CHECK(m->code);m->code_size=m->code_capacity=size;uint32_t at=0;
 for(unsigned f=0;f<7;f++){m->functions[f].code_offset=at;m->functions[f].code_length=code[f].n;memcpy(m->code+at,code[f].data,code[f].n);at+=code[f].n;}
 return m;
}
int main(int argc,char **argv){
 CHECK(argc==6);unsigned mode=(unsigned)strtoul(argv[1],NULL,10),port=(unsigned)strtoul(argv[2],NULL,10);bool ipv6=atoi(argv[3])!=0;
 NvmModule *m=program(mode,port,ipv6);NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);size_t bytes=0;
 CHECK(nvm_v2_module_serialize(&v,NULL,0,&bytes)==NVM_V2_OK);uint8_t *wire=malloc(bytes);CHECK(wire);
 CHECK(nvm_v2_module_serialize(&v,wire,bytes,&bytes)==NVM_V2_OK);nvm_v2_module_free(&v);nvm_module_free(m);
 NvmServicesIndirectOptions invalid={2,1000};NvmServicesRuntimeView prior={.values={999}};
 NvmServicesIndirectExecutionReport refused=nvm_services_vm_indirect_execute(wire,bytes,&invalid,&prior);
 CHECK(refused.runtime.status==NVM_SERVICES_RUNTIME_INVALID && !refused.runtime.acquired && prior.values[0]==999);
 invalid.revision=1;refused=nvm_services_vm_indirect_execute(wire,bytes-1,&invalid,&prior);
 CHECK(refused.runtime.status!=NVM_SERVICES_RUNTIME_OK && !refused.runtime.acquired && prior.values[0]==999);
 char *preserved=(void *)&checks,why[256];
 CHECK(nvm2c_services_indirect_private_emit(wire,bytes-1,&preserved,why,sizeof why)!=NVM_SERVICES_RUNTIME_OK && preserved==(void *)&checks);
#ifdef SERVICE_ALLOC_TEST
 for(int fault=0;fault<6;fault++) {
  NvmServicesRuntime *runtime=(void *)&checks;budget=fault;
  NvmServicesRuntimeStatus status=nvm_services_runtime_indirect_create(wire,bytes,NVM_SERVICES_RUNTIME_VM,&invalid,&runtime);budget=-1;
  CHECK(status==NVM_SERVICES_RUNTIME_MEMORY && runtime==(void *)&checks);
 }
 NvmServicesRuntime *runtime=NULL;budget=6;
 CHECK(nvm_services_runtime_indirect_create(wire,bytes,NVM_SERVICES_RUNTIME_VM,&invalid,&runtime)==NVM_SERVICES_RUNTIME_OK);budget=-1;
 (void)nvm_services_runtime_indirect_destroy(&runtime,NULL);CHECK(!runtime);
 for(int fault=0;fault<=10;fault++) {
  CHECK(nvm_services_runtime_indirect_create(wire,bytes,NVM_SERVICES_RUNTIME_VM,&invalid,&runtime)==NVM_SERVICES_RUNTIME_OK);
  budget=fault;NvmServicesRuntimeStatus status=nvm_services_runtime_begin(runtime);budget=-1;
  CHECK(status==(fault<10?NVM_SERVICES_RUNTIME_MEMORY:NVM_SERVICES_RUNTIME_OK));
  NvmServicesIndirectExecutionReport failed=nvm_services_runtime_indirect_destroy(&runtime,&prior);
  CHECK(!runtime && prior.values[0]==999 && !failed.runtime.cleanup.cleanup_failures);
  CHECK(failed.runtime.acquired==(fault==10));
 }
#endif
 char *generated=NULL,diagnostic[256];NvmServicesRuntimeStatus emitted=nvm2c_services_indirect_private_emit(wire,bytes,&generated,diagnostic,sizeof diagnostic);
 if(emitted!=NVM_SERVICES_RUNTIME_OK)fprintf(stderr,"emit %u: %s\n",emitted,diagnostic);
 CHECK(emitted==NVM_SERVICES_RUNTIME_OK);FILE *f=fopen(argv[4],"w");CHECK(f);CHECK(fputs(generated,f)>=0 && !fclose(f));free(generated);
 NvmServicesIndirectOptions options={1,(uint64_t)strtoull(argv[5],NULL,10)};NvmServicesRuntimeView out={.values={999}};
 NvmServicesIndirectExecutionReport report=nvm_services_vm_indirect_execute(wire,bytes,&options,&out);free(wire);
 unsigned expected=options.instruction_limit<100?NVM_SERVICES_RUNTIME_LIMIT:mode&4?NVM_SERVICES_RUNTIME_ASSERT:NVM_SERVICES_RUNTIME_OK;
 if(report.runtime.status!=expected)fprintf(stderr,"runtime status=%u core=%u site=%u:%u steps=%llu expected=%u\n",report.runtime.status,report.runtime.core_status,report.runtime.function,report.runtime.instruction,(unsigned long long)report.instructions_started,expected);
 CHECK(report.runtime.status==expected && !report.runtime.cleanup.cleanup_failures);
 CHECK(out.values[0]==(expected==NVM_SERVICES_RUNTIME_OK?42:999));
 CHECK(report.runtime.cleanup.count==3);printf("PASS mixed VM %u checks; status=%u steps=%llu\n",checks,report.runtime.status,(unsigned long long)report.instructions_started);return 0;
}
