/* I inspect copied target facts; no File service or callable body executes. */
#define FILE_CYCLIC_ALLOC_TEST
#define FILE_BODY_MAIN prior_file_body_fixture_main
#include "test_file_body.c"
#undef FILE_BODY_MAIN
#include "../../src/nanoisa/file_indirect_targets.h"
static uint8_t *function_descriptor(NvmModule *m,unsigned f){
 size_t p=24;for(unsigned i=0;i<f;i++)p+=12+8*m->functions[i].local_count;return m->ownership_data+p;
}
static NvmModule *target_module(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);
 for(unsigned f=1;f<=4;f+=3){
  uint8_t *p=function_descriptor(m,f),tag=TAG_INT;
  m->functions[f].result_tag=TAG_INT;CHECK(nvm_set_function_param_types(m,f,&tag,1));
  desc(p+4,TAG_INT,0,NVM_V2_NO_INDEX);desc(p+12,TAG_INT,0,NVM_V2_NO_INDEX);
  Body c={0};one(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);setbody(m,f,c);
 }
 desc(function_descriptor(m,0)+12+8*4,TAG_FUNCTION,0,NVM_V2_NO_INDEX);return m;
}
static void reference(Body *c,uint32_t f){op(c,OP_FUNCREF);u32(c,f);}
static void indirect(Body *c,uint16_t args,uint16_t results){op(c,OP_CALL_INDIRECT);u16(c,args);u16(c,results);}
static void jump_back(Body *c,uint32_t destination){uint32_t at=branch(c,OP_JMP,0);wr32(c->bytes+at+1,(uint32_t)((int32_t)destination-(int32_t)at));}
static NvmFileIndirectTargets *query(NvmModule *m,NvmFileIndirectStatus expected){
 NvmFileIndirectTargets *r=(void *)(uintptr_t)1;NvmFileIndirectResult status=nvm_file_indirect_targets(m,&r);
 if(status.status!=expected)fprintf(stderr,"I expected %u, received %u at %u:%u: %s\n",expected,status.status,status.function,status.pc,status.message);
 CHECK(status.status==expected);
 if(expected!=NVM_FILE_INDIRECT_DESCRIBED){CHECK(r==(void *)(uintptr_t)1);return NULL;}
 CHECK(r && r!=(void *)(uintptr_t)1);return r;
}
static void release_targets(NvmFileIndirectTargets *r){nvm_file_indirect_targets_free(r);
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
}
static void row(NvmFileIndirectTargets *r,uint32_t pc,uint64_t bits){
 NvmFileIndirectSummary summary;CHECK(nvm_file_indirect_targets_summary(r,&summary));
 CHECK(summary.functions==5 && summary.calls==1 && summary.visits && summary.visits<=NVM_FILE_INDIRECT_VISITS);
 CHECK(summary.storage_bound<=NVM_FILE_INDIRECT_BYTES);
 NvmFileIndirectCall call;CHECK(nvm_file_indirect_targets_call(r,0,&call));
 CHECK(call.function==0 && call.pc==pc && call.candidates==bits && call.parameters==1 && call.result_count==1);
 CHECK(call.result.tag==TAG_INT && !call.result.mode && call.result.global_index==NVM_V2_NO_INDEX);
 NvmFileFlowDeclaration p;CHECK(nvm_file_indirect_targets_parameter(r,0,0,&p));
 CHECK(p.tag==TAG_INT && !p.mode && p.global_index==NVM_V2_NO_INDEX);
 NvmFileIndirectCall sentinel;memset(&sentinel,0x5a,sizeof sentinel);call=sentinel;
 CHECK(!nvm_file_indirect_targets_call(r,1,&call) && !memcmp(&call,&sentinel,sizeof call));
 NvmFileFlowDeclaration prior=p;CHECK(!nvm_file_indirect_targets_parameter(r,0,1,&p) && !memcmp(&p,&prior,sizeof p));
}
static Body loop_body(bool initialized,uint32_t *site){
 Body c={0};if(initialized){reference(&c,1);one(&c,OP_STORE_LOCAL,4);}
 uint32_t header=c.n;integer(&c);one(&c,OP_LOAD_LOCAL,4);*site=c.n;indirect(&c,1,1);op(&c,OP_POP);
 op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 reference(&c,4);one(&c,OP_STORE_LOCAL,4);jump_back(&c,header);target(&c,leave);retint(&c);return c;
}
static void target_sets(void){
 NvmModule *m=target_module();uint32_t site;Body c=loop_body(true,&site);setbody(m,0,c);
 NvmFileIndirectTargets *r=query(m,NVM_FILE_INDIRECT_DESCRIBED);row(r,site,(UINT64_C(1)<<1)|(UINT64_C(1)<<4));
 NvmFileCodePlan *old=(void *)(uintptr_t)1;CHECK(nvm_file_code_prepare(m,&old)!=NVM_FILE_FLOW_OK && old==(void *)(uintptr_t)1);
 CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
 memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);row(r,site,18);release_targets(r);
 m=target_module();c=loop_body(false,&site);setbody(m,0,c);query(m,NVM_FILE_INDIRECT_INVALID);nvm_module_free(m);
 /* I include both branches even though the condition is a constant. */
 m=target_module();c=(Body){0};op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t other=branch(&c,OP_JMP_FALSE,0);
 reference(&c,1);uint32_t join=branch(&c,OP_JMP,0);target(&c,other);reference(&c,4);target(&c,join);
 one(&c,OP_STORE_LOCAL,4);integer(&c);one(&c,OP_LOAD_LOCAL,4);site=c.n;indirect(&c,1,1);op(&c,OP_RET);setbody(m,0,c);
 r=query(m,NVM_FILE_INDIRECT_DESCRIBED);row(r,site,18);release_targets(r);nvm_module_free(m);
}
static void refusals(void){
 NvmModule *m=target_module();Body c={0};retint(&c);integer(&c);reference(&c,1);indirect(&c,1,1);op(&c,OP_RET);setbody(m,0,c);
 query(m,NVM_FILE_INDIRECT_UNRESOLVED);nvm_module_free(m);
 m=target_module();c=(Body){0};integer(&c);reference(&c,1);indirect(&c,0,1);op(&c,OP_RET);setbody(m,0,c);
 query(m,NVM_FILE_INDIRECT_INVALID);nvm_module_free(m);
 m=target_module();c=(Body){0};integer(&c);reference(&c,UINT32_MAX);indirect(&c,1,1);op(&c,OP_RET);setbody(m,0,c);
 query(m,NVM_FILE_INDIRECT_INVALID);nvm_module_free(m);
 /* An unused body's self-edge is still recursion. */
 m=target_module();c=(Body){0};one(&c,OP_LOAD_LOCAL,0);reference(&c,4);indirect(&c,1,1);op(&c,OP_RET);setbody(m,4,c);
 query(m,NVM_FILE_INDIRECT_UNRESOLVED);nvm_module_free(m);
}
static void categories_and_bounds(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);setbody(m,0,lifecycle_code(b));
 NvmFileIndirectTargets *r=query(m,NVM_FILE_INDIRECT_DESCRIBED);NvmFileIndirectSummary summary;
 CHECK(nvm_file_indirect_targets_summary(r,&summary) && !summary.calls);release_targets(r);nvm_module_free(m);
 /* I discover the incompatible candidate only on the backedge. */
 m=target_module();uint32_t site;setbody(m,0,loop_body(true,&site));
 uint8_t tag=TAG_BOOL;CHECK(nvm_set_function_param_types(m,4,&tag,1));
 desc(function_descriptor(m,4)+12,TAG_BOOL,0,NVM_V2_NO_INDEX);
 query(m,NVM_FILE_INDIRECT_UNRESOLVED);nvm_module_free(m);
 /* A direct edge combines with an indirect edge to form recursion. */
 m=target_module();Body c={0};integer(&c);reference(&c,1);indirect(&c,1,1);op(&c,OP_RET);setbody(m,0,c);
 c=(Body){0};op(&c,OP_CALL);u32(&c,0);op(&c,OP_RET);setbody(m,1,c);
 query(m,NVM_FILE_INDIRECT_UNRESOLVED);nvm_module_free(m);
 /* A zero-iteration exit retains the uninitialized alternative. */
 m=target_module();c=(Body){0};op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 reference(&c,1);one(&c,OP_STORE_LOCAL,4);target(&c,leave);integer(&c);one(&c,OP_LOAD_LOCAL,4);indirect(&c,1,1);op(&c,OP_RET);setbody(m,0,c);
 query(m,NVM_FILE_INDIRECT_INVALID);nvm_module_free(m);
 /* Local overwrite removes the previous target; DUP/POP preserve the new one. */
 m=target_module();c=(Body){0};reference(&c,1);one(&c,OP_STORE_LOCAL,4);reference(&c,4);op(&c,OP_DUP);op(&c,OP_POP);one(&c,OP_STORE_LOCAL,4);
 integer(&c);one(&c,OP_LOAD_LOCAL,4);site=c.n;indirect(&c,1,1);op(&c,OP_RET);setbody(m,0,c);
 r=query(m,NVM_FILE_INDIRECT_DESCRIBED);row(r,site,16);release_targets(r);
 uint32_t saved=m->function_count;m->function_count=65;query(m,NVM_FILE_INDIRECT_LIMIT);m->function_count=saved;
 saved=m->code_size;m->code_size=65537;query(m,NVM_FILE_INDIRECT_LIMIT);m->code_size=saved;
 uint16_t locals=m->functions[0].local_count;m->functions[0].local_count=257;query(m,NVM_FILE_INDIRECT_LIMIT);m->functions[0].local_count=locals;
 CHECK(nvm_file_indirect_targets(m,NULL).status==NVM_FILE_INDIRECT_INVALID);nvm_module_free(m);
}
static void allocation_failures(void){
#ifdef FLOW_INSTRUMENT
 NvmModule *m=target_module();uint32_t site;setbody(m,0,loop_body(true,&site));allocations=0;
 NvmFileIndirectTargets *r=query(m,NVM_FILE_INDIRECT_DESCRIBED);unsigned count=allocations;release_targets(r);CHECK(count);
 uint8_t *code=malloc(m->code_size);CHECK(code);memcpy(code,m->code,m->code_size);
 for(unsigned i=0;i<count;i++){
  budget=(int)i;allocations=0;query(m,NVM_FILE_INDIRECT_MEMORY);CHECK(!live);budget=-1;
  CHECK(!memcmp(code,m->code,m->code_size));r=query(m,NVM_FILE_INDIRECT_DESCRIBED);row(r,site,18);release_targets(r);
  transient_allocation=i+1;allocations=0;query(m,NVM_FILE_INDIRECT_MEMORY);CHECK(!live);transient_allocation=0;
  r=query(m,NVM_FILE_INDIRECT_DESCRIBED);release_targets(r);
 }
 free(code);nvm_module_free(m);
#endif
}
int main(void){target_sets();refusals();categories_and_bounds();allocation_failures();printf("PASS %u private indirect target checks; no pending module execution\n",checks);return 0;}
