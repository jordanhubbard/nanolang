/* I inspect ownership composition only; no File service executes. */
#define FILE_INDIRECT_TARGET_MAIN prior_indirect_target_fixture_main
#include "test_file_indirect_targets.c"
#undef FILE_INDIRECT_TARGET_MAIN
#include "../../src/nanoisa/file_indirect_flow.h"
static NvmFileIndirectFlow *flow_query(NvmModule *m,NvmFileFlowStatus expected){
 NvmFileIndirectFlow *r=(void *)(uintptr_t)1;NvmFileFlowStatus actual=nvm_file_indirect_flow_analyze(m,&r);
 if(actual!=expected)fprintf(stderr,"I expected ownership %u, received %u\n",expected,actual);
 CHECK(actual==expected);
 if(expected!=NVM_FILE_FLOW_OK){CHECK(r==(void *)(uintptr_t)1);return NULL;}
 NvmFileIndirectFlowSummary s;CHECK(nvm_file_indirect_flow_summary(r,&s));
 CHECK(!s.runtime_admitted && !s.ownership.runtime_admitted && s.storage_bound<=NVM_FILE_INDIRECT_FLOW_BYTES);
 CHECK(s.ownership.storage_peak<=NVM_FILE_CYCLIC_BYTES && s.ownership.variants==s.ownership.transfers);
 CHECK(s.candidate_applications<=NVM_FILE_INDIRECT_FLOW_APPLICATIONS);return r;
}
static void flow_release(NvmFileIndirectFlow *r){nvm_file_indirect_flow_free(r);
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
}
static uint16_t flow_index(NvmFileIndirectFlow *r,uint32_t f,uint32_t pc){
 NvmFileCodeFunction fn;CHECK(nvm_file_indirect_flow_function(r,f,&fn));
 for(uint16_t i=0;i<fn.instruction_count;i++){NvmFileCodeInstruction in;CHECK(nvm_file_indirect_flow_instruction(r,f,i,&in));if(in.byte_offset==pc)return i;}
 CHECK(false);return 0;
}
static void call_facts(NvmFileIndirectFlow *r,uint32_t pc,uint64_t bits,unsigned owners,unsigned tag){
 uint16_t i=flow_index(r,0,pc);uint8_t variants;CHECK(nvm_file_indirect_flow_variant_count(r,0,i,&variants) && variants);
 for(uint8_t v=0;v<variants;v++){
  NvmFileIndirectFlowCall call;CHECK(nvm_file_indirect_flow_call(r,0,i,v,&call));
  CHECK(call.function==0 && call.pc==pc && call.candidates==bits && call.checked_candidates==bits);
  CHECK(call.common.kind==NVM_FILE_FLOW_CALL && call.common.target==NVM_V2_NO_INDEX && call.common.site==pc);
  CHECK(call.common.parameters==1 && call.common.owned_inputs==owners && !call.common.borrowed_inputs);
  CHECK(call.common.result_count==1 && call.common.result.tag==tag);
  NvmFileCyclicVariant variant;CHECK(nvm_file_indirect_flow_variant(r,0,i,v,&variant));
  CHECK(variant.input.stack==2 && variant.body.input_stack==2 && variant.output.stack==1 && variant.body.output_stack==1);
  CHECK(variant.body.has_obligation && variant.body.cleanup==NVM_FILE_BODY_CLEANUP_CALL);
  CHECK(variant.body.discharged_checks==NVM_FILE_FLOW_CHECK_CALLEE && !(variant.body.pending_checks&NVM_FILE_FLOW_CHECK_CALLEE));
  NvmFileFlowValue callable,arg;CHECK(nvm_file_indirect_flow_input_stack(r,0,i,v,1,&callable));
  CHECK(callable.initialized && callable.type.tag==TAG_FUNCTION && !callable.owner && !callable.type.mode && callable.arm==NVM_FILE_FLOW_ARM_NONE);
  CHECK(nvm_file_indirect_flow_input_stack(r,0,i,v,0,&arg));CHECK(arg.initialized && arg.type.tag==tag && (!!arg.owner)==!!owners);
 }
}
static void invalid_getters(NvmFileIndirectFlow *r){
 NvmFileIndirectFlowCall call,old;memset(&call,0x5a,sizeof call);old=call;
 CHECK(!nvm_file_indirect_flow_call(r,99,0,0,&call) && !memcmp(&call,&old,sizeof call));
 NvmFileFlowValue val,prior;memset(&val,0x5a,sizeof val);prior=val;
 CHECK(!nvm_file_indirect_flow_input_stack(r,0,0,99,0,&val) && !memcmp(&val,&prior,sizeof val));
 uint8_t count=231;CHECK(!nvm_file_indirect_flow_variant_count(r,99,0,&count) && count==231);
 uint16_t component=1234;CHECK(!nvm_file_indirect_flow_component(r,99,0,&component) && component==1234);
 uint32_t import=UINT32_MAX;CHECK(!nvm_file_indirect_flow_import(r,99,&import) && import==UINT32_MAX);
 size_t unchanged=0;CHECK(!nvm_file_indirect_flow_summary(NULL,NULL) && !unchanged);
}
static void scalar_composition(void){
 NvmModule *m=target_module();uint32_t pc;setbody(m,0,loop_body(true,&pc));
 NvmFileIndirectFlow *r=flow_query(m,NVM_FILE_FLOW_OK);call_facts(r,pc,18,0,TAG_INT);invalid_getters(r);
 NvmFileIndirectFlowSummary s;CHECK(nvm_file_indirect_flow_summary(r,&s) && s.targets.calls==1 && s.candidate_applications>=2);
 NvmFileCyclicReport *old=(void *)(uintptr_t)1;CHECK(nvm_file_cyclic_analyze(m,&old)!=NVM_FILE_FLOW_OK && old==(void *)(uintptr_t)1);
 CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
 memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);
 call_facts(r,pc,18,0,TAG_INT);flow_release(r);
 m=target_module();setbody(m,0,loop_body(false,&pc));flow_query(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
}
static void choose_pair(Body *c,bool reverse){
 op(c,OP_PUSH_BOOL);op(c,1);uint32_t other=branch(c,OP_JMP_FALSE,0);
 reference(c,reverse?4:1);uint32_t join=branch(c,OP_JMP,0);target(c,other);reference(c,reverse?1:4);target(c,join);one(c,OP_STORE_LOCAL,4);
}
static NvmModule *owner_module(NvmFileNominalBindings *b,bool permutation,bool result){
 NvmModule *m=bodymodule(b,permutation);uint8_t tag=result?TAG_UNION:TAG_STRUCT;uint32_t layout=b->layouts[result?3:0];
 for(unsigned f=1;f<=4;f+=3){uint8_t *p=function_descriptor(m,f);
  m->functions[f].result_tag=tag;CHECK(nvm_set_function_param_types(m,f,&tag,1));desc(p+4,tag,0,layout);desc(p+12,tag,0,layout);
  Body c={0};if(f==4)op(&c,OP_NOP);one(&c,OP_OWN_MOVE_LOCAL,0);op(&c,OP_RET);setbody(m,f,c);
 }
 desc(function_descriptor(m,0)+12+8*4,TAG_FUNCTION,0,NVM_V2_NO_INDEX);return m;
}
static Body owner_body(NvmFileNominalBindings b,bool result,bool reverse,bool held,bool leak,uint32_t *pc){
 Body c={0};choose_pair(&c,reverse);service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);
 uint32_t failed=UINT32_MAX;
 if(!result){failed=branch(&c,OP_FILE_RESULT_BRANCH,1);take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);}
 uint16_t slot=result?1:0;
 if(held){op(&c,OP_REGION_BEGIN);op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);}
 uint32_t header=c.n;one(&c,OP_OWN_MOVE_LOCAL,slot);one(&c,OP_LOAD_LOCAL,4);*pc=c.n;indirect(&c,1,1);one(&c,OP_OWN_STORE_LOCAL,slot);
 op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t leave=branch(&c,OP_JMP_FALSE,0);jump_back(&c,header);target(&c,leave);
 if(!leak)one(&c,OP_FILE_DROP_LOCAL,slot);
 retint(&c);
 if(!result){target(&c,failed);take_result(&c,1,1);op(&c,OP_POP);retint(&c);}
 return c;
}
static void owned_composition(void){
 for(unsigned permutation=0;permutation<2;permutation++)for(unsigned result=0;result<2;result++)for(unsigned reverse=0;reverse<2;reverse++){
  NvmFileNominalBindings b;NvmModule *m=owner_module(&b,permutation!=0,result!=0);uint32_t pc;
  setbody(m,0,owner_body(b,result!=0,reverse!=0,false,false,&pc));NvmFileIndirectFlow *r=flow_query(m,NVM_FILE_FLOW_OK);
  call_facts(r,pc,18,1,result?TAG_UNION:TAG_STRUCT);
  for(unsigned k=0;k<8;k++){NvmFileNominalLayout t;CHECK(nvm_file_indirect_flow_type(r,k,&t) && t.global_index==b.layouts[k]);}
  for(unsigned k=0;k<5;k++){uint32_t import;CHECK(nvm_file_indirect_flow_import(r,k,&import) && import==b.imports[k]);}
  memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);
  call_facts(r,pc,18,1,result?TAG_UNION:TAG_STRUCT);
  for(unsigned k=0;k<8;k++){NvmFileNominalLayout t;CHECK(nvm_file_indirect_flow_type(r,k,&t) && t.global_index==b.layouts[k]);}
  flow_release(r);
 }
}
static void ownership_refusals(void){
 NvmFileNominalBindings b;NvmModule *m=owner_module(&b,false,false);uint32_t pc;
 setbody(m,0,owner_body(b,false,false,true,false,&pc));NvmFileIndirectTargets *targets=query(m,NVM_FILE_INDIRECT_DESCRIBED);release_targets(targets);
 flow_query(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 m=owner_module(&b,false,true);setbody(m,0,owner_body(b,true,false,false,true,&pc));targets=query(m,NVM_FILE_INDIRECT_DESCRIBED);release_targets(targets);
 flow_query(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 /* Only the higher-index candidate holds its File when moving it to return. */
 m=owner_module(&b,false,false);setbody(m,0,owner_body(b,false,false,false,false,&pc));Body c={0};
 op(&c,OP_REGION_BEGIN);op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);one(&c,OP_OWN_MOVE_LOCAL,0);op(&c,OP_RET);setbody(m,4,c);
 targets=query(m,NVM_FILE_INDIRECT_DESCRIBED);release_targets(targets);flow_query(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 /* An uncalled body still has to discharge its owner on return. */
 m=owner_module(&b,false,false);c=(Body){0};retint(&c);setbody(m,0,c);
 m->functions[4].result_tag=TAG_INT;desc(function_descriptor(m,4)+4,TAG_INT,0,NVM_V2_NO_INDEX);setbody(m,4,c);
 targets=query(m,NVM_FILE_INDIRECT_DESCRIBED);release_targets(targets);flow_query(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
}
static void flow_faults_and_bounds(void){
 NvmModule *m=target_module();uint32_t pc;setbody(m,0,loop_body(true,&pc));
 uint32_t saved=m->function_count;m->function_count=65;flow_query(m,NVM_FILE_FLOW_LIMIT);m->function_count=saved;
 CHECK(nvm_file_indirect_flow_analyze(m,NULL)==NVM_FILE_FLOW_INVALID);
#ifdef FLOW_INSTRUMENT
 allocations=0;NvmFileIndirectFlow *r=flow_query(m,NVM_FILE_FLOW_OK);unsigned measured=allocations;flow_release(r);CHECK(measured);
 uint8_t *before=malloc(m->code_size);CHECK(before);memcpy(before,m->code,m->code_size);
 for(unsigned i=0;i<measured;i++){
  budget=(int)i;allocations=0;flow_query(m,NVM_FILE_FLOW_MEMORY);CHECK(!live);budget=-1;
  CHECK(!memcmp(before,m->code,m->code_size));r=flow_query(m,NVM_FILE_FLOW_OK);flow_release(r);
  transient_allocation=i+1;allocations=0;flow_query(m,NVM_FILE_FLOW_MEMORY);CHECK(!live);transient_allocation=0;
  r=flow_query(m,NVM_FILE_FLOW_OK);flow_release(r);
 }
 printf("I checked %u allocating preparation sites in both failure modes.\n",measured);free(before);
 uint32_t instructions;size_t bytes,decl;CHECK(file_cyclic_budget(m,SIZE_MAX,&instructions,&bytes,&decl)==NVM_FILE_FLOW_LIMIT);
 r=flow_query(m,NVM_FILE_FLOW_OK);uint16_t i=flow_index(r,0,pc);NvmFileCodePlan *p=r->ownership->plan;
 NvmFileFlowState *s=state(p->declarations,0);OK(nvm_file_flow_push_scalar(s,TAG_INT));s->stack[s->stack_count++]=initial_value(scalar_type(TAG_FUNCTION),true,0);
 FileIndirectFlowScratch *scratch=calloc(1,sizeof *scratch);CHECK(scratch);scratch->targets=r->targets;scratch->bits[p->starts[0]+i]=18;
 scratch->applications=NVM_FILE_INDIRECT_FLOW_APPLICATIONS;NvmFileBodyReport body={0};body.plan=p;body.checked[1]=body.checked[4]=true;NvmFileBodyInstruction fact={0};
 unsigned old_allocations=allocations;CHECK(fif_transfer(scratch,&body,s,&p->instructions[p->starts[0]+i],&fact)==NVM_FILE_FLOW_LIMIT);
 CHECK(scratch->applications==NVM_FILE_INDIRECT_FLOW_APPLICATIONS && allocations==old_allocations);
 s->stack[s->stack_count++]=initial_value(scalar_type(TAG_FUNCTION),true,0);
 scratch->applications=0;budget=0;old_allocations=allocations;
 CHECK(fif_transfer(scratch,&body,s,&p->instructions[p->starts[0]+i],&fact)==NVM_FILE_FLOW_OK);
 budget=-1;CHECK(scratch->applications==2 && allocations==old_allocations && s->stack_count==1 && s->stack[0].type.tag==TAG_INT);
 CHECK(fact.has_obligation && fact.obligation.target==NVM_V2_NO_INDEX);
 free(scratch);nvm_file_flow_state_free(s);flow_release(r);
#endif
 nvm_module_free(m);
}
int main(void){setvbuf(stdout,NULL,_IONBF,0);
 puts("I begin scalar composition.");scalar_composition();puts("I begin owned composition.");owned_composition();
 puts("I begin ownership refusals.");ownership_refusals();puts("I begin allocation and bound controls.");flow_faults_and_bounds();
 printf("PASS %u private indirect ownership checks; no File execution\n",checks);return 0;
}
