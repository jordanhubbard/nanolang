/* I analyze synthetic bodies only; no File opcode or host service executes. */
#define main prior_file_flow_fixture_main
#include "test_file_flow.c"
#undef main
#include "../../src/nanoisa/file_body.h"
typedef struct {uint8_t bytes[8192];uint32_t n;} Body;
static void op(Body *c,uint8_t value){CHECK(c->n<sizeof c->bytes);c->bytes[c->n++]=value;}
static void u16(Body *c,uint16_t value){op(c,(uint8_t)value);op(c,(uint8_t)(value>>8));}
static void u32(Body *c,uint32_t value){for(unsigned i=0;i<4;i++)op(c,(uint8_t)(value>>(8*i)));}
static void integer(Body *c){op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)op(c,0);}
static void one(Body *c,uint8_t code,uint16_t x){op(c,code);u16(c,x);}
static void service(Body *c,NvmFileNominalBindings b,unsigned method,uint16_t ref){op(c,OP_FILE_SERVICE);u32(c,b.imports[method]);u16(c,ref);}
static uint32_t branch(Body *c,uint8_t code,uint16_t slot){uint32_t pc=c->n;op(c,code);if(code==OP_FILE_RESULT_BRANCH)u16(c,slot);u32(c,0);return pc;}
static void target(Body *c,uint32_t pc){wr32(c->bytes+pc+(c->bytes[pc]==OP_FILE_RESULT_BRANCH?3:1),c->n-pc);}
static void take_result(Body *c,uint16_t slot,uint8_t arm){one(c,OP_FILE_RESULT_TAKE,slot);op(c,arm);}
static void retint(Body *c){integer(c);op(c,OP_RET);}
static void setbody(NvmModule *m,unsigned which,Body c){
 size_t size=c.n;for(unsigned f=0;f<m->function_count;f++)if(f!=which)size+=m->functions[f].code_length;
 uint8_t *bytes=malloc(size);CHECK(bytes);uint32_t cursor=0;
 for(unsigned f=0;f<m->function_count;f++) {uint32_t n=f==which?c.n:m->functions[f].code_length;
  memcpy(bytes+cursor,f==which?c.bytes:m->code+m->functions[f].code_offset,n);
  m->functions[f].code_offset=cursor;m->functions[f].code_length=n;cursor+=n;}
 free(m->code);m->code=bytes;m->code_size=m->code_capacity=(uint32_t)size;
}
static NvmModule *bodymodule(NvmFileNominalBindings *b,bool permutation){
 NvmModule *m=module(b,permutation);Body c={0};
 /* First replace the overlapping placeholders before setbody rearranges them. */
 free(m->code);m->code=calloc(5,1);CHECK(m->code);m->code_size=m->code_capacity=5;
 for(unsigned f=0;f<5;f++){m->functions[f].code_offset=f;m->functions[f].code_length=1;m->code[f]=OP_RET;}
 retint(&c);setbody(m,0,c);c=(Body){0};one(&c,OP_OWN_MOVE_LOCAL,0);op(&c,OP_RET);setbody(m,1,c);
 c=(Body){0};one(&c,OP_LOAD_LOCAL,1);service(&c,*b,1,0);op(&c,OP_RET);setbody(m,2,c);
 c=(Body){0};retint(&c);setbody(m,3,c);
 c=(Body){0};one(&c,OP_OWN_MOVE_LOCAL,0);op(&c,OP_RET);setbody(m,4,c);return m;
}
static void expect(NvmModule *m,NvmFileFlowStatus expected){
 NvmFileBodyReport *r=(NvmFileBodyReport *)(uintptr_t)1;CHECK(nvm_file_body_analyze(m,&r)==expected);
 if(expected==NVM_FILE_FLOW_OK){CHECK(r && r!=(NvmFileBodyReport *)(uintptr_t)1);nvm_file_body_free(r);}
 else CHECK(r==(NvmFileBodyReport *)(uintptr_t)1);
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
}
static NvmFileBodyInstruction fact_at(NvmFileBodyReport *r,unsigned f,uint32_t pc){
 NvmFileCodeFunction fn;CHECK(nvm_file_body_function(r,f,&fn));
 for(uint16_t i=0;i<fn.instruction_count;i++){NvmFileCodeInstruction in;NvmFileBodyInstruction fact;CHECK(nvm_file_body_instruction(r,f,i,&in,&fact));if(in.byte_offset==pc)return fact;}
 CHECK(false);return (NvmFileBodyInstruction){0};
}
static Body lifecycle_code(NvmFileNominalBindings b){
 Body c={0};service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);
 uint32_t failed=branch(&c,OP_FILE_RESULT_BRANCH,1);
 take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);
 op(&c,OP_REGION_BEGIN);op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);
 integer(&c);service(&c,b,1,20);op(&c,OP_POP);
 service(&c,b,2,20);op(&c,OP_POP);service(&c,b,3,20);op(&c,OP_POP);
 integer(&c);op(&c,OP_CALL_REF);u32(&c,2);u16(&c,20);op(&c,OP_POP);
 one(&c,OP_FILE_END_BORROW,20);op(&c,OP_REGION_END);
 one(&c,OP_OWN_MOVE_LOCAL,0);op(&c,OP_CALL);u32(&c,1);
 service(&c,b,4,UINT16_MAX);one(&c,OP_STORE_LOCAL,3);
 uint32_t closeerror=branch(&c,OP_FILE_RESULT_BRANCH,3);
 take_result(&c,3,0);op(&c,OP_POP);retint(&c);
 target(&c,closeerror);take_result(&c,3,1);one(&c,OP_AGG_GET,5);op(&c,OP_POP);retint(&c);
 target(&c,failed);take_result(&c,1,1);op(&c,OP_POP);retint(&c);return c;
}
static void complete_bodies(void){
 for(unsigned permutation=0;permutation<2;permutation++){
  NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,permutation!=0);Body c=lifecycle_code(b);setbody(m,0,c);
  NvmFileBodyReport *r=NULL;OK(nvm_file_body_analyze(m,&r));CHECK(nvm_file_body_function_count(r)==5);
  NvmFileCodeFunction fn;CHECK(nvm_file_body_function(r,0,&fn));unsigned returns=0,calls=0,services=0,refinements=0;
  for(uint16_t i=0;i<fn.instruction_count;i++){
   NvmFileCodeInstruction in;NvmFileBodyInstruction fact;CHECK(nvm_file_body_instruction(r,0,i,&in,&fact));CHECK(fact.reachable);
   CHECK(fact.pending_checks & NVM_FILE_FLOW_CHECK_CLEANUP);
   if(in.decoded.opcode==OP_RET){CHECK(fact.exit_checked && fact.output_stack==1);returns++;}
   if(fact.refinement)refinements++;
   if(fact.has_obligation){CHECK(fact.obligation.site==in.byte_offset);
    if(fact.obligation.kind==NVM_FILE_FLOW_CALL){calls++;CHECK(fact.discharged_checks==NVM_FILE_FLOW_CHECK_CALLEE);CHECK(!(fact.pending_checks & NVM_FILE_FLOW_CHECK_CALLEE));CHECK(fact.obligation.checks & NVM_FILE_FLOW_CHECK_CALLEE);}
    else {services++;CHECK(fact.discharged_checks==0);CHECK((fact.pending_checks & fact.obligation.checks)==fact.obligation.checks);
     if(fact.obligation.target==b.imports[4])CHECK(fact.obligation.outcomes[0]==NVM_FILE_FLOW_INPUT_CONSUMED && fact.obligation.outcomes[1]==NVM_FILE_FLOW_INPUT_CONSUMED);
    }
   }
  }
  CHECK(returns==3 && calls==2 && services==5 && refinements==2);
  NvmFileCodeInstruction in={0},oldin=in;NvmFileBodyInstruction fact={0},oldfact=fact;
  CHECK(!nvm_file_body_instruction(r,5,0,&in,&fact) && !memcmp(&in,&oldin,sizeof in) && !memcmp(&fact,&oldfact,sizeof fact));
  NvmFileFlowDeclaration decl;CHECK(nvm_file_body_local(r,2,0,&decl) && decl.mode==2);
  CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
  memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);
  CHECK(nvm_file_body_function(r,0,&fn));CHECK(fact_at(r,0,0).has_obligation);nvm_file_body_free(r);
 }
}
static void wrong_bodies(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);Body c={0};retint(&c);setbody(m,1,c);expect(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m); /* uncalled helper still checked */
 m=bodymodule(&b,false);c=(Body){0};op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t other=branch(&c,OP_JMP_TRUE,0);retint(&c);target(&c,other);op(&c,OP_PUSH_BOOL);op(&c,0);op(&c,OP_RET);setbody(m,0,c);expect(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};one(&c,OP_LOAD_LOCAL,4);op(&c,OP_RET);setbody(m,0,c);expect(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);retint(&c);setbody(m,0,c);expect(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};one(&c,OP_FILE_END_BORROW,0);integer(&c);service(&c,b,1,0);op(&c,OP_RET);setbody(m,2,c);expect(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};integer(&c);op(&c,OP_CALL);u32(&c,1);op(&c,OP_POP);retint(&c);setbody(m,0,c);expect(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
}
static void joins_and_cleanup(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);Body c={0};op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t other=branch(&c,OP_JMP_FALSE,0);
 integer(&c);one(&c,OP_STORE_LOCAL,4);uint32_t end=branch(&c,OP_JMP,0);target(&c,other);integer(&c);one(&c,OP_STORE_LOCAL,4);target(&c,end);one(&c,OP_LOAD_LOCAL,4);op(&c,OP_RET);setbody(m,0,c);expect(m,NVM_FILE_FLOW_OK);nvm_module_free(m);
 /* Branch-local fresh owners cannot be renamed to make an ordinary join pass. */
 m=bodymodule(&b,false);c=(Body){0};op(&c,OP_PUSH_BOOL);op(&c,1);other=branch(&c,OP_JMP_TRUE,0);service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);end=branch(&c,OP_JMP,0);target(&c,other);service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);target(&c,end);one(&c,OP_FILE_DROP_LOCAL,1);retint(&c);setbody(m,0,c);expect(m,NVM_FILE_FLOW_UNRESOLVED);nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);uint32_t drop=c.n;one(&c,OP_FILE_DROP_LOCAL,1);service(&c,b,0,UINT16_MAX);uint32_t stackdrop=c.n;op(&c,OP_FILE_DROP_STACK);op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t assertion=c.n;op(&c,OP_ASSERT);retint(&c);setbody(m,0,c);
 NvmFileBodyReport *r=NULL;OK(nvm_file_body_analyze(m,&r));CHECK(fact_at(r,0,drop).cleanup==NVM_FILE_BODY_CLEANUP_DROP_LOCAL && fact_at(r,0,drop).cleanup_local==1);CHECK(fact_at(r,0,stackdrop).cleanup==NVM_FILE_BODY_CLEANUP_DROP_STACK);CHECK(fact_at(r,0,assertion).cleanup==NVM_FILE_BODY_CLEANUP_ASSERT);nvm_file_body_free(r);nvm_module_free(m);
}
static void known_result_and_projection(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);NvmFileNominalPlan *p=NULL;CHECK(nvm_file_nominal_plan(m,&p)==NVM_FILE_NOMINAL_DESCRIBED);NvmFileNominalLayout result;CHECK(nvm_file_nominal_type(p,4,&result));nvm_file_nominal_plan_free(p);
 Body c={0};integer(&c);op(&c,OP_UNION_CONSTRUCT);u32(&c,result.source_ordinal);u16(&c,0);u16(&c,1);one(&c,OP_STORE_LOCAL,6);
 uint32_t error=branch(&c,OP_FILE_RESULT_BRANCH,6);one(&c,OP_LOAD_LOCAL,6);one(&c,OP_UNION_FIELD,0);op(&c,OP_RET);target(&c,error);uint32_t dead=c.n;one(&c,OP_LOAD_LOCAL,4);op(&c,OP_RET);setbody(m,0,c);
 NvmFileBodyReport *r=NULL;OK(nvm_file_body_analyze(m,&r));CHECK(!fact_at(r,0,dead).reachable);nvm_file_body_free(r);
 /* Unknown call result is not silently projected as an exact known arm. */
 c=(Body){0};one(&c,OP_LOAD_LOCAL,1);service(&c,b,1,0);one(&c,OP_UNION_FIELD,0);op(&c,OP_RET);setbody(m,2,c);expect(m,NVM_FILE_FLOW_UNRESOLVED);nvm_module_free(m);
}
static void scalar_and_capacity(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);const uint8_t binary[]={OP_ADD,OP_SUB,OP_MUL,OP_DIV,OP_MOD,OP_I64_ADD,OP_I64_SUB,OP_I64_MUL,OP_I64_DIV_S,OP_I64_REM_S,OP_EQ,OP_NE,OP_LT,OP_LE,OP_GT,OP_GE,OP_I64_EQ,OP_I64_NE,OP_I64_LT_S,OP_I64_LE_S,OP_I64_GT_S,OP_I64_GE_S};
 for(unsigned i=0;i<sizeof binary;i++){Body c={0};integer(&c);integer(&c);op(&c,binary[i]);op(&c,OP_POP);retint(&c);setbody(m,0,c);expect(m,NVM_FILE_FLOW_OK);}
 Body c={0};op(&c,OP_PUSH_BOOL);op(&c,1);op(&c,OP_NOT);op(&c,OP_PUSH_BOOL);op(&c,0);op(&c,OP_AND);op(&c,OP_PUSH_BOOL);op(&c,1);op(&c,OP_OR);op(&c,OP_ASSERT);integer(&c);op(&c,OP_NEG);op(&c,OP_I64_NEG);op(&c,OP_RET);setbody(m,0,c);expect(m,NVM_FILE_FLOW_OK);
 c=(Body){0};op(&c,OP_PUSH_BOOL);op(&c,0);integer(&c);op(&c,OP_EQ);op(&c,OP_POP);retint(&c);setbody(m,0,c);expect(m,NVM_FILE_FLOW_UNRESOLVED);
 c=(Body){0};for(unsigned i=0;i<255;i++)op(&c,OP_NOP);op(&c,OP_RET);setbody(m,0,c);expect(m,NVM_FILE_FLOW_INVALID);op(&c,OP_RET);setbody(m,0,c);expect(m,NVM_FILE_FLOW_LIMIT);
 NvmFileFlowDeclarations *d=NULL;OK(nvm_file_flow_declarations(m,&d));NvmFileFlowState *s=state(d,0);for(unsigned i=0;i<256;i++)OK(nvm_file_flow_push_scalar(s,TAG_INT));CHECK(counts(s).stack==256);CHECK(nvm_file_flow_push_scalar(s,TAG_INT)==NVM_FILE_FLOW_LIMIT && counts(s).stack==256);
#ifdef FLOW_INSTRUMENT
 OK(file_body_scalar(s,TAG_INT,TAG_INT,2,false));CHECK(counts(s).stack==255);
#endif
 nvm_file_flow_state_free(s);nvm_file_flow_declarations_free(d);nvm_module_free(m);
}
static void allocation_recovery(void){
#ifdef FLOW_INSTRUMENT
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);setbody(m,0,lifecycle_code(b));bool success=false;unsigned failures=0;
 for(int limit=0;limit<512;limit++){budget=limit;NvmFileBodyReport *out=(NvmFileBodyReport *)(uintptr_t)1;NvmFileFlowStatus status=nvm_file_body_analyze(m,&out);budget=-1;
  if(status==NVM_FILE_FLOW_OK){nvm_file_body_free(out);CHECK(!live);success=true;break;}
  CHECK(status==NVM_FILE_FLOW_MEMORY && out==(NvmFileBodyReport *)(uintptr_t)1 && !live);failures++;expect(m,NVM_FILE_FLOW_OK);
 }
 CHECK(success && failures>30);
 NvmFileFlowDeclarations *d=NULL;OK(nvm_file_flow_declarations(m,&d));NvmFileFlowState *s=state(d,0);OK(nvm_file_flow_service(s,1,b.imports[0],UINT16_MAX));OK(nvm_file_flow_put(s,1));unsigned before=live;
 NvmFileFlowState *a=(NvmFileFlowState *)(uintptr_t)1,*e=(NvmFileFlowState *)(uintptr_t)2;budget=1;CHECK(nvm_file_flow_refine(s,1,&a,&e)==NVM_FILE_FLOW_MEMORY);budget=-1;
 CHECK(a==(NvmFileFlowState *)(uintptr_t)1 && e==(NvmFileFlowState *)(uintptr_t)2 && live==before && local(s,1).arm==NVM_FILE_FLOW_ARM_UNKNOWN);
 FileBodyWorkspace w={0};w.bytes=NVM_FILE_FLOW_BYTES-sizeof(NvmFileFlowState);CHECK(file_body_room(&w,1) && !file_body_room(&w,2));w.bytes++;CHECK(!file_body_room(&w,1));
 nvm_file_flow_state_free(s);nvm_file_flow_declarations_free(d);nvm_module_free(m);CHECK(!live);
#endif
}
int main(void){complete_bodies();wrong_bodies();joins_and_cleanup();known_result_and_projection();scalar_and_capacity();allocation_recovery();
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
 printf("PASS %u private acyclic File body checks; no hosted stack ABI or execution authority\n",checks);return 0;}
