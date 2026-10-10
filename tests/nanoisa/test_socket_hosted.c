/* I retain TCP startup and ownership facts; no network operation executes. */
#define FILE_BODY_MAIN prior_socket_body_fixture_main
#include "test_socket_body.c"
#undef FILE_BODY_MAIN
#include "../../src/nanoisa/socket_hosted.h"
#include "../../src/nanoisa/socket_cyclic_hosted.h"
#include "../../src/nanoisa/socket_indirect_hosted.h"
#include "../../src/nanoisa/file_hosted.h"
static uint8_t *serialize_tcp(NvmModule *m,size_t *size){
 NvmV2Module wire={0};CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
 CHECK(nvm_v2_module_serialize(&wire,NULL,0,size)==NVM_V2_OK);uint8_t *bytes=malloc(*size);CHECK(bytes);
 CHECK(nvm_v2_module_serialize(&wire,bytes,*size,size)==NVM_V2_OK);nvm_v2_module_free(&wire);return bytes;
}
static void callable(Body *c,uint32_t target){op(c,OP_FUNCREF);u32(c,target);}
static void indirect_call(Body *c){op(c,OP_CALL_INDIRECT);u16(c,1);u16(c,1);}
static Body loop_code(NvmModule *m,NvmSocketNominalBindings b,bool indirect,bool leak){
 Body c={0};service(&c,b,0,UINT16_MAX);
 if(indirect){callable(&c,4);indirect_call(&c);}
 one(&c,OP_OWN_STORE_LOCAL,1);uint32_t error=branch(&c,OP_FILE_RESULT_BRANCH,1);
 take_result(&c,1,0);
 if(indirect){callable(&c,1);indirect_call(&c);}
 one(&c,OP_OWN_STORE_LOCAL,0);op(&c,OP_REGION_BEGIN);
 op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);
 uint32_t head=c.n;op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 integer(&c);
 if(indirect){
  const char refs[]={20,0,(char)255,(char)255};uint32_t map=nvm_add_string(m,refs,4);CHECK(map!=UINT32_MAX);
  callable(&c,2);op(&c,OP_FILE_CALL_INDIRECT_REFS);u16(&c,2);u16(&c,1);u32(&c,map);
 }else{op(&c,OP_CALL_REF);u32(&c,2);u16(&c,20);}
 op(&c,OP_POP);service(&c,b,2,20);op(&c,OP_POP);service(&c,b,3,20);op(&c,OP_POP);
 uint32_t back=branch(&c,OP_JMP,0);wr32(c.bytes+back+1,(uint32_t)(int32_t)((int64_t)head-back));
 target(&c,leave);op(&c,OP_REGION_END);
 if(!leak){one(&c,OP_OWN_MOVE_LOCAL,0);service(&c,b,4,UINT16_MAX);op(&c,OP_POP);}
 retint(&c);target(&c,error);take_result(&c,1,1);op(&c,OP_POP);retint(&c);return c;
}
static void hosted(unsigned mode,bool permute){
 NvmSocketNominalBindings b;NvmModule *m=bodymodule(&b,permute);
 setbody(m,0,mode?loop_code(m,b,mode==2,false):lifecycle_code(b));size_t size=0;uint8_t *bytes=serialize_tcp(m,&size);
 CHECK(!nvm_verify(m).ok);char error[128];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
 NvmFileHostedPlan *old=(void *)(uintptr_t)1;CHECK(nvm_file_hosted_prepare(bytes,size,&old)!=NVM_FILE_FLOW_OK && old==(void *)(uintptr_t)1);
 if(mode==0){
  NvmSocketHostedPlan *p=NULL;OK(nvm_socket_hosted_prepare(bytes,size,&p));NvmSocketHostedStartup startup;NvmSocketHostedFunction fn;
  CHECK(nvm_socket_hosted_startup(p,&startup) && startup.entry==0 && startup.frames==2);
  CHECK(nvm_socket_hosted_function(p,0,&fn) && fn.operand_peak==7 && fn.locals==13);
  memset(bytes,0,size);nvm_module_free(m);m=NULL;
  CHECK(nvm_socket_hosted_startup(p,&startup) && startup.vm_value_slots>0);nvm_socket_hosted_free(p);
 }else if(mode==1){
  NvmSocketCyclicHostedPlan *p=NULL;OK(nvm_socket_cyclic_hosted_prepare(bytes,size,&p));NvmSocketCyclicHostedStartup startup;NvmSocketCyclicHostedFunction fn;
  CHECK(nvm_socket_cyclic_hosted_startup(p,&startup) && !startup.runtime_admitted && startup.input_bytes==size);
  CHECK(nvm_socket_cyclic_hosted_function(p,0,&fn) && fn.operand_peak==7 && fn.locals==13);
  unsigned pending=0;
  for(uint16_t i=0;i<fn.code.instruction_count;i++){uint8_t count=0;CHECK(nvm_socket_cyclic_hosted_variant_count(p,0,i,&count));
   for(uint8_t v=0;v<count;v++){NvmSocketCyclicVariant fact;CHECK(nvm_socket_cyclic_hosted_variant(p,0,i,v,&fact));
    if(fact.body.has_obligation && fact.body.obligation.target==b.imports[0] && fact.body.obligation.kind==NVM_SOCKET_FLOW_SERVICE){CHECK(fact.body.pending_checks & NVM_SOCKET_FLOW_CHECK_ENDPOINT);pending++;}}
  }CHECK(pending);
  memset(bytes,0,size);nvm_module_free(m);m=NULL;CHECK(nvm_socket_cyclic_hosted_startup(p,&startup));nvm_socket_cyclic_hosted_free(p);
 }else{
  NvmSocketIndirectHostedPlan *p=NULL;OK(nvm_socket_indirect_hosted_prepare(bytes,size,&p));NvmSocketIndirectHostedStartup startup;NvmSocketIndirectHostedFunction fn;
  CHECK(nvm_socket_indirect_hosted_startup(p,&startup) && !startup.runtime_admitted && startup.input_bytes==size);
  CHECK(nvm_socket_indirect_hosted_function(p,0,&fn) && fn.operand_peak==7 && fn.locals==13);
  unsigned calls=0,pending=0;
  for(uint16_t i=0;i<fn.code.instruction_count;i++){NvmSocketCodeInstruction in;CHECK(nvm_socket_indirect_hosted_instruction(p,0,i,&in));uint8_t count=0;CHECK(nvm_socket_indirect_hosted_variant_count(p,0,i,&count));
   for(uint8_t v=0;v<count;v++){NvmSocketCyclicVariant fact;CHECK(nvm_socket_indirect_hosted_variant(p,0,i,v,&fact));
    if(in.decoded.opcode==OP_FILE_SERVICE && in.catalog_ordinal==0){CHECK(fact.body.pending_checks & NVM_SOCKET_FLOW_CHECK_ENDPOINT);pending++;}
    if(isa_is_file_indirect_call(in.decoded.opcode)){NvmSocketIndirectFlowCall call;CHECK(nvm_socket_indirect_hosted_call(p,0,i,v,&call));CHECK(call.checked_candidates==call.candidates && call.candidates);calls++;}
   }
  }CHECK(calls>=3 && pending);
  memset(bytes,0,size);nvm_module_free(m);m=NULL;CHECK(nvm_socket_indirect_hosted_startup(p,&startup));nvm_socket_indirect_hosted_free(p);
 }
 free(bytes);nvm_module_free(m);
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
}
static void refusals_and_allocations(void){
 NvmSocketNominalBindings b;NvmModule *m=bodymodule(&b,false);setbody(m,0,loop_code(m,b,true,true));
 NvmSocketIndirectFlow *out=(void *)(uintptr_t)1;CHECK(nvm_socket_indirect_flow_analyze(m,&out)!=NVM_SOCKET_FLOW_OK && out==(void *)(uintptr_t)1);
 setbody(m,0,loop_code(m,b,true,false));size_t size=0;uint8_t *bytes=serialize_tcp(m,&size);
 NvmSocketIndirectHostedPlan *p=(void *)(uintptr_t)1;
 CHECK(nvm_socket_indirect_hosted_prepare(bytes,size-1,&p)!=NVM_SOCKET_FLOW_OK && p==(void *)(uintptr_t)1);
#ifdef FLOW_INSTRUMENT
 bool success=false;unsigned failed=0;
 for(int n=0;n<2048;n++){
  budget=n;NvmSocketFlowStatus status=nvm_socket_indirect_hosted_prepare(bytes,size,&p);budget=-1;
  if(status==NVM_SOCKET_FLOW_OK){nvm_socket_indirect_hosted_free(p);CHECK(!live);success=true;break;}
  CHECK((status==NVM_SOCKET_FLOW_MEMORY || status==NVM_SOCKET_FLOW_UNRESOLVED) && p==(void *)(uintptr_t)1 && !live);failed++;
 }
 CHECK(success && failed>30);
#endif
 free(bytes);nvm_module_free(m);
}
int main(void){for(unsigned mode=0;mode<3;mode++)for(unsigned p=0;p<2;p++)hosted(mode,p!=0);refusals_and_allocations();printf("PASS %u TCP loop, indirect and hosted preparation checks; no execution\n",checks);return 0;}
