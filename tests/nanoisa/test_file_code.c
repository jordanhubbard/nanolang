/* I prepare CODE facts only; none of these synthetic bodies execute. */
#define main prior_file_flow_fixture_main
#include "test_file_flow.c"
#undef main
#include "../../src/nanoisa/file_code.h"
static void body(NvmModule *m,uint32_t which,const uint8_t *bytes,uint32_t size) {
 CHECK(which<m->function_count);size_t total=size;
 for(uint32_t f=0;f<m->function_count;f++)if(f!=which)total+=m->functions[f].code_length;
 CHECK(total<=UINT32_MAX);uint8_t *next=malloc(total?total:1);CHECK(next);uint32_t offset=0;
 for(uint32_t f=0;f<m->function_count;f++) {
  uint32_t n=f==which?size:m->functions[f].code_length;
  const uint8_t *from=f==which?bytes:m->code+m->functions[f].code_offset;
  memcpy(next+offset,from,n);m->functions[f].code_offset=offset;m->functions[f].code_length=n;offset+=n;
 }
 free(m->code);m->code=next;m->code_size=m->code_capacity=(uint32_t)total;
}
static NvmModule *plan_module(NvmFileNominalBindings *b,bool permutation) {
 NvmModule *m=module(b,permutation);uint8_t bytes[50];
 for(uint32_t f=0;f<5;f++) {
  memset(bytes+10*f,0,10);bytes[10*f]=OP_PUSH_I64;bytes[10*f+1]=7;bytes[10*f+9]=OP_RET;
  m->functions[f].code_offset=10*f;m->functions[f].code_length=10;
 }
 free(m->code);m->code=malloc(sizeof bytes);CHECK(m->code);memcpy(m->code,bytes,sizeof bytes);m->code_size=m->code_capacity=sizeof bytes;
 return m;
}
static NvmModule *bounded_module(uint32_t functions,uint16_t locals,uint16_t instructions) {
 NvmFileNominalBindings b;NvmModule *m=fixture(false,&b);
 m->functions[0].arity=0;m->functions[0].local_count=locals;CHECK(nvm_set_function_param_types(m,0,NULL,0));
 for(uint32_t f=1;f<functions;f++){NvmFunctionEntry fn=m->functions[0];CHECK(nvm_add_function(m,&fn)==f);}
 size_t size=24+(size_t)functions*(12+8*locals);free(m->ownership_data);m->ownership_data=calloc(1,size);CHECK(m->ownership_data);m->ownership_size=(uint32_t)size;
 uint8_t *o=m->ownership_data;wr32(o,1);wr32(o+4,9);for(unsigned i=0;i<8;i++)o[8+b.layouts[i]]=(i==0||i==3)?3:1;wr32(o+20,functions);
 size_t at=24;for(uint32_t f=0;f<functions;f++) {o[at]=(uint8_t)locals;o[at+1]=(uint8_t)(locals>>8);desc(o+at+4,TAG_INT,0,NVM_V2_NO_INDEX);at+=12;for(uint16_t l=0;l<locals;l++){desc(o+at,TAG_INT,0,NVM_V2_NO_INDEX);at+=8;}}
 free(m->code);m->code_size=m->code_capacity=functions*instructions;m->code=calloc(1,m->code_size);CHECK(m->code);
 for(uint32_t f=0;f<functions;f++){m->functions[f].code_offset=f*instructions;m->functions[f].code_length=instructions;m->code[(f+1)*instructions-1]=OP_RET;}
 return m;
}
static void expect_prepare(NvmModule *m,NvmFileFlowStatus status) {
 NvmFileCodePlan *out=(NvmFileCodePlan *)(uintptr_t)1;
 CHECK(nvm_file_code_prepare(m,&out)==status);
 if(status==NVM_FILE_FLOW_OK){CHECK(out && out!=(NvmFileCodePlan *)(uintptr_t)1);nvm_file_code_free(out);}
 else CHECK(out==(NvmFileCodePlan *)(uintptr_t)1);
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
}
static void preparation_facts(void) {
 for(unsigned permutation=0;permutation<2;permutation++) {
  NvmFileNominalBindings b;NvmModule *m=plan_module(&b,permutation!=0);NvmFileCodePlan *p=NULL;OK(nvm_file_code_prepare(m,&p));CHECK(nvm_file_code_function_count(p)==5);
  NvmFileCodeFunction fn;CHECK(nvm_file_code_function(p,2,&fn) && fn.instruction_count==2 && fn.declaration.parameters==2 && fn.declaration.result.catalog_ordinal==4);
  NvmFileFlowDeclaration local;CHECK(nvm_file_code_local(p,2,0,&local) && local.mode==2 && local.global_index==b.layouts[0]);
  NvmFileCodeInstruction in;CHECK(nvm_file_code_instruction(p,0,0,&in) && in.byte_offset==0 && in.decoded.opcode==OP_PUSH_I64 && in.decoded.operands[0].i64==7 && in.successor_count==1 && in.successors[0]==1);
  uint16_t rank=99;CHECK(nvm_file_code_instruction_order(p,0,0,&rank) && rank==0);uint32_t function=99;CHECK(nvm_file_code_function_order(p,0,&function) && function==0);
  NvmFileCodeInstruction old=in;CHECK(!nvm_file_code_instruction(p,0,2,&in) && !memcmp(&old,&in,sizeof in));CHECK(!nvm_file_code_instruction_order(p,5,0,&rank) && rank==0);CHECK(!nvm_file_code_function_order(p,5,&function) && function==0);
  CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
  /* A prepared owner-return helper deliberately still has an INT body: this
   * object cannot stand in for dependent transfer/body certification. */
  memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);
  CHECK(nvm_file_code_instruction(p,0,0,&in) && in.decoded.operands[0].i64==7);CHECK(nvm_file_code_local(p,2,0,&local) && local.mode==2);nvm_file_code_free(p);
  m=plan_module(&b,permutation!=0);uint32_t offset=m->functions[0].code_offset;m->functions[0].code_offset=m->functions[2].code_offset;m->functions[2].code_offset=offset;expect_prepare(m,NVM_FILE_FLOW_OK);
  m->header.flags&=~NVM_FLAG_HAS_MAIN;expect_prepare(m,NVM_FILE_FLOW_OK);m->functions[1].name_idx=string(m,"__init__");expect_prepare(m,NVM_FILE_FLOW_OK);nvm_module_free(m);
 }
}
static void malformed_and_graphs(void) {
 NvmFileNominalBindings b;NvmModule *m=plan_module(&b,false);
 uint32_t save=m->functions[0].name_idx;m->functions[0].name_idx=m->string_count;expect_prepare(m,NVM_FILE_FLOW_INVALID);m->functions[0].name_idx=save;
 m->functions[0].result_tag=TAG_VOID;expect_prepare(m,NVM_FILE_FLOW_INVALID);m->functions[0].result_tag=TAG_INT;
 save=m->functions[1].code_offset;m->functions[1].code_offset=1;expect_prepare(m,NVM_FILE_FLOW_INVALID);m->functions[1].code_offset=save;
 m->functions[1].code_offset=UINT32_MAX;expect_prepare(m,NVM_FILE_FLOW_INVALID);m->functions[1].code_offset=save;
 m->functions[1].code_length=0;expect_prepare(m,NVM_FILE_FLOW_INVALID);m->functions[1].code_length=10;
 uint8_t *more=realloc(m->code,51);CHECK(more);m->code=more;m->code[50]=OP_RET;m->code_size=m->code_capacity=51;expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);m->code_size=50;
 const uint8_t truncated[]={OP_FILE_SERVICE,0};body(m,0,truncated,sizeof truncated);expect_prepare(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 const uint8_t unsupported[]={OP_CALL_INDIRECT,0,0,0,0,OP_RET};m=plan_module(&b,false);body(m,4,unsupported,sizeof unsupported);expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);nvm_module_free(m);
 uint8_t unreachable[]={OP_JMP,8,0,0,0,OP_LOAD_LOCAL,255,255,OP_RET};m=plan_module(&b,false);body(m,4,unreachable,sizeof unreachable);expect_prepare(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 uint8_t branch[]={OP_PUSH_BOOL,1,OP_JMP_TRUE,6,0,0,0,OP_NOP,OP_RET};m=plan_module(&b,false);body(m,0,branch,sizeof branch);expect_prepare(m,NVM_FILE_FLOW_OK);
 const uint32_t bad[]={1,UINT32_MAX,UINT32_C(0x80000000),UINT32_C(0x7fffffff),7};for(unsigned i=0;i<sizeof bad/sizeof bad[0];i++){wr32(m->code+3,bad[i]);expect_prepare(m,NVM_FILE_FLOW_INVALID);}nvm_module_free(m);
 const uint8_t loop[]={OP_JMP,0,0,0,0};m=plan_module(&b,false);body(m,0,loop,sizeof loop);expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);nvm_module_free(m);
 const uint8_t dead_loop[]={OP_RET,OP_JMP,0,0,0,0};m=plan_module(&b,false);body(m,4,dead_loop,sizeof dead_loop);expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);nvm_module_free(m);
 uint8_t call[]={OP_CALL,1,0,0,0,OP_RET};m=plan_module(&b,false);body(m,0,call,sizeof call);NvmFileCodePlan *p=NULL;OK(nvm_file_code_prepare(m,&p));uint32_t order[5];for(unsigned i=0;i<5;i++)CHECK(nvm_file_code_function_order(p,i,&order[i]));unsigned caller=0,callee=0;for(unsigned i=0;i<5;i++){if(order[i]==0)caller=i;if(order[i]==1)callee=i;}CHECK(callee<caller);nvm_file_code_free(p);
 wr32(call+1,0);body(m,1,call,sizeof call);expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);nvm_module_free(m);
 m=plan_module(&b,false);body(m,0,call,sizeof call);expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);wr32(call+1,5);body(m,0,call,sizeof call);expect_prepare(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
 uint8_t reference[]={OP_CALL_REF,2,0,0,0,0,0,OP_RET};m=plan_module(&b,false);body(m,0,reference,sizeof reference);expect_prepare(m,NVM_FILE_FLOW_OK);reference[1]=3;body(m,0,reference,sizeof reference);expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);reference[1]=2;reference[6]=1;body(m,0,reference,sizeof reference);expect_prepare(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
}
static void exact_operands(void) {
 for(unsigned permutation=0;permutation<2;permutation++) {
  NvmFileNominalBindings b;NvmModule *m=plan_module(&b,permutation!=0);
  uint8_t service[]={OP_FILE_SERVICE,0,0,0,0,255,255,OP_RET};wr32(service+1,b.imports[0]);body(m,0,service,sizeof service);
  NvmFileCodePlan *p=NULL;OK(nvm_file_code_prepare(m,&p));NvmFileCodeInstruction in;CHECK(nvm_file_code_instruction(p,0,0,&in) && in.catalog_ordinal==0);nvm_file_code_free(p);
  service[5]=0;service[6]=0;body(m,0,service,sizeof service);expect_prepare(m,NVM_FILE_FLOW_INVALID);
  wr32(service+1,b.imports[1]);body(m,0,service,sizeof service);expect_prepare(m,NVM_FILE_FLOW_OK);service[5]=255;service[6]=255;body(m,0,service,sizeof service);expect_prepare(m,NVM_FILE_FLOW_INVALID);wr32(service+1,5);body(m,0,service,sizeof service);expect_prepare(m,NVM_FILE_FLOW_INVALID);
  NvmFileNominalPlan *nominal=NULL;CHECK(nvm_file_nominal_plan(m,&nominal)==NVM_FILE_NOMINAL_DESCRIBED);
  NvmFileNominalLayout layouts[8];for(unsigned ordinal=0;ordinal<8;ordinal++)CHECK(nvm_file_nominal_type(nominal,ordinal,&layouts[ordinal]));nvm_file_nominal_plan_free(nominal);
  for(unsigned ordinal=0;ordinal<8;ordinal++) {
   NvmFileNominalLayout layout=layouts[ordinal];const NlFilePlanType *type=nl_file_catalog_type(ordinal);
   uint8_t construct[]={OP_AGG_PACK,ordinal<3?AGG_RECORD:AGG_VARIANT,0,0,0,0,0,0,0,0,OP_RET};wr32(construct+2,layout.source_ordinal);construct[8]=(uint8_t)(ordinal<3?type->member_count:1);body(m,0,construct,sizeof construct);
   if(ordinal==0 || ordinal==3)expect_prepare(m,NVM_FILE_FLOW_UNRESOLVED);
   else {p=NULL;OK(nvm_file_code_prepare(m,&p));CHECK(nvm_file_code_instruction(p,0,0,&in) && in.catalog_ordinal==ordinal);nvm_file_code_free(p);construct[8]++;body(m,0,construct,sizeof construct);expect_prepare(m,NVM_FILE_FLOW_INVALID);}
  }
  nvm_module_free(m);
 }
}
static void local_operands(void) {
 NvmFileNominalBindings b;NvmModule *m=plan_module(&b,false);
 uint8_t result[]={OP_FILE_RESULT_BRANCH,1,0,8,0,0,0,OP_RET,OP_RET};body(m,0,result,sizeof result);expect_prepare(m,NVM_FILE_FLOW_OK);result[1]=12;body(m,0,result,sizeof result);expect_prepare(m,NVM_FILE_FLOW_INVALID);
 uint8_t take[]={OP_FILE_RESULT_TAKE,1,0,0,OP_RET};body(m,0,take,sizeof take);expect_prepare(m,NVM_FILE_FLOW_OK);take[3]=2;body(m,0,take,sizeof take);expect_prepare(m,NVM_FILE_FLOW_INVALID);
 const uint8_t locals[]={OP_LOAD_LOCAL,OP_STORE_LOCAL,OP_OWN_MOVE_LOCAL,OP_OWN_STORE_LOCAL,OP_FILE_DROP_LOCAL};
 for(unsigned i=0;i<sizeof locals;i++){uint8_t bytes[]={locals[i],12,0,OP_RET};body(m,0,bytes,sizeof bytes);expect_prepare(m,NVM_FILE_FLOW_INVALID);}
 uint8_t borrow[]={OP_BORROW_LOCAL_EXCLUSIVE,0,0,0,0,OP_RET};body(m,0,borrow,sizeof borrow);expect_prepare(m,NVM_FILE_FLOW_OK);borrow[2]=1;body(m,0,borrow,sizeof borrow);expect_prepare(m,NVM_FILE_FLOW_INVALID);borrow[2]=0;borrow[3]=12;body(m,0,borrow,sizeof borrow);expect_prepare(m,NVM_FILE_FLOW_INVALID);
 const uint8_t end[]={OP_FILE_END_BORROW,0,1,OP_RET};body(m,0,end,sizeof end);expect_prepare(m,NVM_FILE_FLOW_INVALID);nvm_module_free(m);
}
static void prepare_limits_and_failures(void) {
 NvmModule *m=bounded_module(64,256,1);expect_prepare(m,NVM_FILE_FLOW_OK);NvmFunctionEntry fn=m->functions[0];CHECK(nvm_add_function(m,&fn)==64);expect_prepare(m,NVM_FILE_FLOW_LIMIT);nvm_module_free(m);
 m=bounded_module(16,0,256);expect_prepare(m,NVM_FILE_FLOW_OK);m->functions[0].code_length=257;m->functions[1].code_offset++;m->functions[1].code_length--;expect_prepare(m,NVM_FILE_FLOW_LIMIT);nvm_module_free(m);
 m=bounded_module(17,0,256);expect_prepare(m,NVM_FILE_FLOW_LIMIT);nvm_module_free(m);
 m=bounded_module(1,257,1);expect_prepare(m,NVM_FILE_FLOW_LIMIT);nvm_module_free(m);
 m=bounded_module(1,0,1);uint8_t *code=realloc(m->code,NVM_FILE_CODE_BYTES+1);CHECK(code);m->code=code;m->code_size=m->code_capacity=NVM_FILE_CODE_BYTES+1;expect_prepare(m,NVM_FILE_FLOW_LIMIT);nvm_module_free(m);
#ifdef FLOW_INSTRUMENT
 NvmFileNominalBindings b;m=plan_module(&b,false);bool success=false;unsigned failed=0;
 for(int limit=0;limit<16;limit++) {budget=limit;NvmFileCodePlan *out=(NvmFileCodePlan *)(uintptr_t)1;NvmFileFlowStatus result=nvm_file_code_prepare(m,&out);budget=-1;
  if(result==NVM_FILE_FLOW_OK){nvm_file_code_free(out);success=true;CHECK(!live);break;}
  CHECK(result==NVM_FILE_FLOW_MEMORY && out==(NvmFileCodePlan *)(uintptr_t)1 && !live);failed++;expect_prepare(m,NVM_FILE_FLOW_OK);
 }
 CHECK(success && failed==5);nvm_module_free(m);
#endif
}
int main(void) {
 preparation_facts();malformed_and_graphs();exact_operands();local_operands();prepare_limits_and_failures();
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
 printf("PASS %u private File CODE preparation checks; no transfer/body certificate or execution\n",checks);return 0;
}
