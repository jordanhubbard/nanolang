/* I retain the complete manual carrier/frame controls, then execute fresh CODE
 * through the private adapter. Instrumented-only mutations are explicitly
 * identified; no public VM handler is enabled. */
#define FILE_FRAMES_MAIN prior_file_frames_main
#include "test_file_runtime_frames.c"
#undef FILE_FRAMES_MAIN
#include "../../src/nanovm/file_vm_private.h"
#ifdef HOSTED_INSTRUMENT
static bool vm_no_alloc,vm_scratch_failure;
static int vm_generation=-1;
static uint32_t vm_fault_function,vm_fault_site;
static NvmFileRuntimeStatus vm_begin_hook(NvmFileRuntime *c){
 NvmFileRuntimeStatus status=nvm_file_runtime_begin(c);
 if(status==NVM_FILE_RUNTIME_OK && vm_no_alloc)allocation_budget=0;
 return status;
}
static NvmFileRuntimeStatus vm_call_hook(NvmFileRuntime *c){
 if(vm_generation>=0 && vm_generation<4){
  NvmFileRuntimeFrameView f=fv(c);vm_fault_function=f.function;vm_fault_site=f.byte_offset;
  near_limit(c,fs(c,(unsigned)(vm_generation&1)),vm_generation<2?UINT64_MAX:UINT64_MAX-1);
 }
 return nvm_file_runtime_frame_call(c);
}
static NvmFileRuntimeStatus vm_return_hook(NvmFileRuntime *c){
 NvmFileRuntimeFrameView f=fv(c);
 if(vm_generation>=4 && f.depth==2){vm_fault_function=f.function;vm_fault_site=f.byte_offset;
  near_limit(c,fs(c,0),vm_generation==4?UINT64_MAX:UINT64_MAX-1);}
 return nvm_file_runtime_frame_return(c);
}
static NvmFileRuntimeStatus vm_service_hook(NvmFileRuntime *c,uint32_t import,uint32_t reference,uint32_t input,uint32_t output){
 NvmFileRuntimeStatus status=nvm_file_runtime_service(c,import,reference,input,output);
 if(status==NVM_FILE_RUNTIME_OK && vm_scratch_failure && view(c,output).owning){
  NvmFileRuntimeFrameView f=fv(c);vm_fault_function=f.function;vm_fault_site=f.byte_offset;near_limit(c,output,UINT64_MAX);
 }
 return status;
}
#define nvm_file_runtime_begin vm_begin_hook
#define nvm_file_runtime_frame_call vm_call_hook
#define nvm_file_runtime_frame_return vm_return_hook
#define nvm_file_runtime_service vm_service_hook
#include "../../src/nanovm/file_vm_private.c"
#undef nvm_file_runtime_begin
#undef nvm_file_runtime_frame_call
#undef nvm_file_runtime_frame_return
#undef nvm_file_runtime_service
#endif
static void vb(Body *p,bool b){op(p,OP_PUSH_BOOL);op(p,b);}
static void veq(Body *p,int64_t value){fi(p,value);op(p,OP_I64_EQ);op(p,OP_ASSERT);}
static void varm(Body *p,unsigned expected){op(p,OP_DUP);op(p,OP_UNION_TAG);veq(p,expected);op(p,OP_POP);}
static NvmFileRuntimeReport vrbytes(uint8_t *bytes,size_t n,NvmFileRuntimeStatus expected,uint8_t tag,int64_t value){
 NvmFileRuntimeView out;memset(&out,0xa5,sizeof out);NvmFileRuntimeView saved=out;
 unsigned loads=loader_attempts,forks=fork_attempts;
 NvmFileRuntimeReport report=nvm_file_vm_execute(bytes,n,&out);
 CHECK(report.status==expected && loads==loader_attempts && forks==fork_attempts);
 if(expected==NVM_FILE_RUNTIME_OK){
  if(out.type.tag!=tag || out.values[0]!=value)fprintf(stderr,"private VM output: expected tag=%u value=%lld; actual tag=%u value=%lld\n",(unsigned)tag,(long long)value,(unsigned)out.type.tag,(long long)out.values[0]);
  CHECK(report.acquired && !report.cleanup.cleanup_failures && out.initialized &&
   !out.owning && !out.formal && out.type.tag==tag && out.values[0]==value);
 }
 else CHECK(!memcmp(&out,&saved,sizeof out));
 empty_host();return report;
}
static NvmFileRuntimeReport vr(NvmModule *m,NvmFileRuntimeStatus expected,uint8_t tag,int64_t value){
 size_t n;uint8_t *bytes=serialize(m,&n);nvm_module_free(m);
 NvmFileRuntimeReport report=vrbytes(bytes,n,expected,tag,value);free(bytes);return report;
}
static void vm_numeric(void){
 struct {uint8_t op;int64_t a,b,want;bool unary,boolean;} cases[]={
  {OP_ADD,INT64_MAX,1,INT64_MIN,0,0},{OP_I64_ADD,INT64_MAX,1,INT64_MIN,0,0},
  {OP_SUB,INT64_MIN,1,INT64_MAX,0,0},{OP_I64_SUB,INT64_MIN,1,INT64_MAX,0,0},
  {OP_MUL,INT64_MIN,-1,INT64_MIN,0,0},{OP_I64_MUL,INT64_MIN,-1,INT64_MIN,0,0},
  {OP_DIV,INT64_MIN,-1,INT64_MIN,0,0},{OP_I64_DIV_S,INT64_MIN,-1,INT64_MIN,0,0},
  {OP_DIV,19,0,0,0,0},{OP_I64_DIV_S,-19,4,-4,0,0},
  {OP_MOD,INT64_MIN,-1,0,0,0},{OP_I64_REM_S,-19,4,-3,0,0},
  {OP_MOD,19,0,0,0,0},{OP_I64_REM_S,19,0,0,0,0},
  {OP_NEG,INT64_MIN,0,INT64_MIN,1,0},{OP_I64_NEG,INT64_MIN,0,INT64_MIN,1,0},
  {OP_EQ,7,7,1,0,1},{OP_NE,7,8,1,0,1},{OP_LT,-1,0,1,0,1},{OP_LE,0,0,1,0,1},
  {OP_GT,1,0,1,0,1},{OP_GE,0,0,1,0,1},{OP_I64_EQ,7,7,1,0,1},{OP_I64_NE,7,8,1,0,1},
  {OP_I64_LT_S,-1,0,1,0,1},{OP_I64_LE_S,0,0,1,0,1},{OP_I64_GT_S,1,0,1,0,1},{OP_I64_GE_S,0,0,1,0,1}
 };
 for(unsigned i=0;i<sizeof cases/sizeof *cases;i++){
  FrameSpec s={.result=cases[i].boolean?-2:-1};NvmFileNominalBindings b;
  fi(&s.code,cases[i].a);if(!cases[i].unary)fi(&s.code,cases[i].b);op(&s.code,cases[i].op);op(&s.code,OP_RET);
  vr(frame_module(&s,1,&b,i&1,-1),NVM_FILE_RUNTIME_OK,cases[i].boolean?TAG_BOOL:TAG_INT,cases[i].want);
 }
 const uint8_t ops[]={OP_AND,OP_OR,OP_NOT,OP_EQ,OP_NE};
 for(unsigned i=0;i<5;i++)for(unsigned a=0;a<2;a++)for(unsigned b=0;b<2;b++){
  FrameSpec s={.result=-2};NvmFileNominalBindings binding;vb(&s.code,a);if(ops[i]!=OP_NOT)vb(&s.code,b);
  op(&s.code,ops[i]);op(&s.code,OP_RET);bool want=i==0?(a&&b):i==1?(a||b):i==2?!a:i==3?a==b:a!=b;
  vr(frame_module(&s,1,&binding,false,-1),NVM_FILE_RUNTIME_OK,TAG_BOOL,want);
 }
}
static void vm_control(void){
 for(unsigned which=0;which<2;which++)for(unsigned pred=0;pred<2;pred++){
  FrameSpec s={.locals=1,.types={-1},.result=-1};NvmFileNominalBindings b;Body *p=&s.code;
  op(p,OP_NOP);op(p,OP_PUSH_VOID);op(p,OP_POP);vb(p,pred);uint32_t yes=branch(p,which?OP_JMP_TRUE:OP_JMP_FALSE,0);
  fi(p,11);one(p,OP_STORE_LOCAL,0);uint32_t end=branch(p,OP_JMP,0);target(p,yes);fi(p,22);one(p,OP_STORE_LOCAL,0);
  target(p,end);one(p,OP_LOAD_LOCAL,0);op(p,OP_DUP);op(p,OP_POP);op(p,OP_RET);
  vr(frame_module(&s,1,&b,false,-1),NVM_FILE_RUNTIME_OK,TAG_INT,(pred!=0)==(which!=0)?22:11);
 }
 for(unsigned pred=0;pred<2;pred++){
  FrameSpec s={.result=-1};NvmFileNominalBindings b;vb(&s.code,pred);uint32_t site=s.code.n;op(&s.code,OP_ASSERT);fi(&s.code,71);op(&s.code,OP_RET);
  NvmFileRuntimeReport r=vr(frame_module(&s,1,&b,false,-1),pred?NVM_FILE_RUNTIME_OK:NVM_FILE_RUNTIME_ASSERT,TAG_INT,71);
  if(!pred)CHECK(r.function==0 && r.instruction==site);
 }
}
static void vm_passive(void){
 for(unsigned perm=0;perm<2;perm++)for(unsigned spelling=0;spelling<2;spelling++){
  NvmFileNominalBindings b;NvmModule *seed=fixture(perm,&b);NvmFileNominalPlan *plan=NULL;
  CHECK(nvm_file_nominal_plan(seed,&plan)==NVM_FILE_NOMINAL_DESCRIBED);NvmFileNominalLayout read,write;
  CHECK(nvm_file_nominal_type(plan,2,&read) && nvm_file_nominal_type(plan,4,&write));nvm_file_nominal_plan_free(plan);nvm_module_free(seed);
  FrameSpec s={.locals=1,.types={4},.result=-1};Body *p=&s.code;
  fi(p,251);vb(p,false);op(p,OP_AGG_PACK);op(p,AGG_RECORD);u32(p,read.source_ordinal);u16(p,0);u16(p,2);one(p,OP_AGG_GET,0);veq(p,251);
  fi(p,19);op(p,spelling?OP_AGG_PACK:OP_UNION_CONSTRUCT);if(spelling)op(p,AGG_VARIANT);
  u32(p,write.source_ordinal);u16(p,0);u16(p,1);one(p,OP_STORE_LOCAL,0);
  uint32_t error=branch(p,OP_FILE_RESULT_BRANCH,0);one(p,OP_LOAD_LOCAL,0);op(p,spelling?OP_AGG_TAG:OP_UNION_TAG);veq(p,0);
  one(p,OP_LOAD_LOCAL,0);one(p,spelling?OP_AGG_GET:OP_UNION_FIELD,0);op(p,OP_RET);
  target(p,error);fi(p,-1);op(p,OP_RET);
  vr(frame_module(&s,1,&b,perm,-1),NVM_FILE_RUNTIME_OK,TAG_INT,19);
 }
}
static NvmModule *vm_io_module(NvmFileNominalBindings *b,bool perm,unsigned fault){
 NvmModule *seed=fixture(perm,b);nvm_module_free(seed);FrameSpec s={.locals=3,.types={3,0,6},.result=-1};Body *p=&s.code;
 service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,0);uint32_t openerror=branch(p,OP_FILE_RESULT_BRANCH,0);
 take_result(p,0,0);one(p,OP_OWN_STORE_LOCAL,1);op(p,OP_REGION_BEGIN);op(p,OP_BORROW_LOCAL_EXCLUSIVE);u16(p,20);u16(p,1);
 fi(p,fault==4?256:251);service(p,*b,1,20);varm(p,fault==1||fault==4);
 service(p,*b,2,20);varm(p,fault==3);
 if(fault!=3 && fault!=4){
  service(p,*b,3,20);one(p,OP_STORE_LOCAL,2);uint32_t readerror=branch(p,OP_FILE_RESULT_BRANCH,2);
  take_result(p,2,0);op(p,OP_DUP);one(p,OP_AGG_GET,0);veq(p,251);one(p,OP_AGG_GET,1);op(p,OP_NOT);op(p,OP_ASSERT);
  service(p,*b,3,20);one(p,OP_STORE_LOCAL,2);uint32_t eoferror=branch(p,OP_FILE_RESULT_BRANCH,2);
  take_result(p,2,0);op(p,OP_DUP);one(p,OP_AGG_GET,0);veq(p,0);one(p,OP_AGG_GET,1);op(p,OP_ASSERT);
  uint32_t done=branch(p,OP_JMP,0);
  target(p,readerror);take_result(p,2,1);one(p,OP_AGG_GET,3);veq(p,1);uint32_t errorexit=branch(p,OP_JMP,0);
  target(p,eoferror);take_result(p,2,1);op(p,OP_POP);vb(p,false);op(p,OP_ASSERT);
  target(p,done);target(p,errorexit);
 }
 one(p,OP_FILE_END_BORROW,20);op(p,OP_REGION_END);one(p,OP_OWN_MOVE_LOCAL,1);service(p,*b,4,UINT16_MAX);op(p,OP_POP);fi(p,251);op(p,OP_RET);
 target(p,openerror);take_result(p,0,1);one(p,OP_AGG_GET,1);veq(p,EACCES);fi(p,-1);op(p,OP_RET);
 return frame_module(&s,1,b,perm,-1);
}
static void vm_lifetimes(void){
 for(unsigned perm=0;perm<2;perm++){
  NvmFileNominalBindings b;vr(vm_io_module(&b,perm,0),NVM_FILE_RUNTIME_OK,TAG_INT,251);
  vr(overlap_module(&b,perm),NVM_FILE_RUNTIME_OK,TAG_INT,909);
  vr(owner_module(&b,perm),NVM_FILE_RUNTIME_OK,TAG_INT,37);
  NvmModule *m=bodymodule(&b,perm);setbody(m,0,lifecycle_code(b));vr(m,NVM_FILE_RUNTIME_OK,TAG_INT,0);
  vr(init_frame_module(&b),NVM_FILE_RUNTIME_OK,TAG_INT,52);
 }
 NvmFileNominalBindings b;FrameSpec s={.locals=1,.types={3},.result=-1};NvmModule *seed=fixture(false,&b);nvm_module_free(seed);
 service(&s.code,b,0,UINT16_MAX);one(&s.code,OP_OWN_STORE_LOCAL,0);one(&s.code,OP_FILE_DROP_LOCAL,0);
 service(&s.code,b,0,UINT16_MAX);op(&s.code,OP_FILE_DROP_STACK);fi(&s.code,81);op(&s.code,OP_RET);
 vr(frame_module(&s,1,&b,false,-1),NVM_FILE_RUNTIME_OK,TAG_INT,81);
}
#ifdef HOSTED_INSTRUMENT
static void vm_faults(void){
 NvmFileNominalBindings b;deny_open=true;vr(vm_io_module(&b,false,0),NVM_FILE_RUNTIME_OK,TAG_INT,-1);deny_open=false;
 model_write_error=true;vr(vm_io_module(&b,false,1),NVM_FILE_RUNTIME_OK,TAG_INT,251);model_write_error=false;modeled_error_stream=NULL;
 model_read_error=true;vr(vm_io_module(&b,false,2),NVM_FILE_RUNTIME_OK,TAG_INT,251);model_read_error=false;modeled_error_stream=NULL;
 deny_seek=true;vr(vm_io_module(&b,false,3),NVM_FILE_RUNTIME_OK,TAG_INT,251);deny_seek=false;
 vr(vm_io_module(&b,false,4),NVM_FILE_RUNTIME_OK,TAG_INT,251);
 close_index=0;close_error[0]=EIO;unsigned attempts=open_attempts;
 NvmModule *init=init_frame_module(&b);Body entry={0};service(&entry,b,0,UINT16_MAX);op(&entry,OP_FILE_DROP_STACK);fi(&entry,52);op(&entry,OP_RET);setbody(init,0,entry);
 NvmFileRuntimeReport r=vr(init,NVM_FILE_RUNTIME_CLEANUP,TAG_INT,0);
 CHECK(r.function==1 && r.cleanup.cleanup_failures==1 && open_attempts==attempts+1);
 memset(close_error,0,sizeof close_error);close_index=0;
 for(int failure=0;failure<6;failure++){
  vm_generation=failure;unsigned before=closed;close_index=0;close_error[0]=EIO;
  r=vr(owner_module(&b,true),NVM_FILE_RUNTIME_LIMIT,TAG_INT,0);
  CHECK(r.function==vm_fault_function && r.instruction==vm_fault_site && closed==before+2 && r.cleanup.cleanup_failures==1);
  vm_generation=-1;memset(close_error,0,sizeof close_error);close_index=0;
 }
 vm_scratch_failure=true;unsigned before=closed;r=vr(vm_io_module(&b,true,0),NVM_FILE_RUNTIME_LIMIT,TAG_INT,0);
 CHECK(closed==before+1 && r.function==vm_fault_function && r.instruction==vm_fault_site);vm_scratch_failure=false;
 vm_no_alloc=true;size_t failed=failed_calls;vr(vm_io_module(&b,false,0),NVM_FILE_RUNTIME_OK,TAG_INT,251);
 CHECK(failed_calls==failed && allocation_budget==0);allocation_budget=-1;vm_no_alloc=false;
}
static void vm_allocation_and_masks(void){
 NvmFileNominalBindings b;NvmModule *m=vm_io_module(&b,false,0);size_t n;uint8_t *bytes=serialize(m,&n);nvm_module_free(m);
 size_t baseline=tracked_bytes,live=tracked_live;unsigned refusals=0,recovered=0;
 for(unsigned transient=0;transient<2;transient++){
  bool finished=false;
  for(unsigned prefix=0;prefix<2048;prefix++){
   NvmFileRuntimeView out;memset(&out,0xa5,sizeof out);NvmFileRuntimeView old=out;unsigned opens=open_attempts;
   size_t failures=failed_calls;allocation_budget=(int)prefix;single_failure=transient!=0;
   NvmFileRuntimeReport r=nvm_file_vm_execute(bytes,n,&out);allocation_budget=-1;single_failure=false;
   CHECK(tracked_bytes==baseline && tracked_live==live);empty_host();
   if(r.status==NVM_FILE_RUNTIME_OK){CHECK(out.type.tag==TAG_INT && out.values[0]==251);if(failed_calls>failures)recovered++;else{finished=true;break;}}
   else {CHECK(r.status==NVM_FILE_RUNTIME_MEMORY || r.status==NVM_FILE_RUNTIME_UNRESOLVED);CHECK(!memcmp(&out,&old,sizeof out) && open_attempts==opens);refusals++;}
  }
  CHECK(finished);
 }
 CHECK(refusals>0);printf("private VM allocation refusals=%u recovered=%u\n",refusals,recovered);
 NvmFileRuntime *c=NULL;ROK(nvm_file_runtime_create(bytes,n,NVM_FILE_RUNTIME_VM,&c));const NvmFileHostedPlan *plan=nvm_file_runtime_plan(c);
 CHECK(fvm_coverage(plan));NvmFileCodeInstruction in;NvmFileBodyInstruction fact;CHECK(nvm_file_hosted_instruction(plan,0,0,&in,&fact));
 CHECK(fvm_fact(plan,&in,&fact));fact.pending_checks|=UINT32_C(1)<<31;CHECK(!fvm_fact(plan,&in,&fact));
 fact=(NvmFileBodyInstruction){0};in.decoded.opcode=OP_PUSH_F64;CHECK(!fvm_fact(plan,&in,&fact));
 /* This unit check mutates copied facts, not the immutable API input. */
 unsigned opens=open_attempts;(void)nvm_file_runtime_destroy(&c,NULL);CHECK(open_attempts==opens);free(bytes);
}
#endif
static void vm_refusals(void){
 NvmFileNominalBindings b;FrameSpec s={.result=-1};fi(&s.code,3);op(&s.code,OP_RET);
 NvmModule *m=frame_module(&s,1,&b,false,-1);size_t n;uint8_t *bytes=serialize(m,&n);unsigned opens=open_attempts;
 NvmFileRuntimeView out;memset(&out,0xa5,sizeof out);NvmFileRuntimeView old=out;
 CHECK(nvm_file_vm_execute(bytes,n,NULL).status==NVM_FILE_RUNTIME_INVALID);
 CHECK(nvm_file_vm_execute(bytes,1,&out).status!=NVM_FILE_RUNTIME_OK && !memcmp(&out,&old,sizeof out));
 CHECK(nvm_file_vm_execute(NULL,n,&out).status==NVM_FILE_RUNTIME_INVALID && !memcmp(&out,&old,sizeof out));
 CHECK(open_attempts==opens);free(bytes);nvm_module_free(m);
 /* A dead unsupported instruction still refuses before acquisition. */
 s.code=(Body){0};fi(&s.code,3);op(&s.code,OP_RET);op(&s.code,OP_PUSH_F64);for(unsigned i=0;i<8;i++)op(&s.code,0);op(&s.code,OP_POP);fi(&s.code,4);op(&s.code,OP_RET);
 m=frame_module(&s,1,&b,false,-1);bytes=serialize(m,&n);NvmFileRuntimeReport r=nvm_file_vm_execute(bytes,n,&out);
 CHECK(r.status==NVM_FILE_RUNTIME_UNRESOLVED && !r.acquired && open_attempts==opens && !memcmp(&out,&old,sizeof out));free(bytes);nvm_module_free(m);
}
int main(void){
 CHECK(prior_file_frames_main()==0);unsigned before=checks;
 vm_numeric();vm_control();vm_passive();vm_lifetimes();vm_refusals();
#ifdef HOSTED_INSTRUMENT
 vm_faults();vm_allocation_and_masks();CHECK(!tracked_live && !tracked_bytes);
#endif
 empty_host();printf("PASS %u actual private File VM dispatch checks; public File routes remain refused\n",checks-before);return 0;
}
