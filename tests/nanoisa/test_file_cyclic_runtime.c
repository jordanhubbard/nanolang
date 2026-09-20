/* I manually exercise the private carrier protocol, not a VM/native dispatcher. */
#define FILE_FRAMES_MAIN prior_file_frames_main
#include "test_file_runtime_frames.c"
#undef FILE_FRAMES_MAIN
#include "../../src/nanoisa/file_cyclic_runtime.h"
#ifdef HOSTED_INSTRUMENT
void file_cyclic_epoch(NlFileValues *,NlFileValueBorrow *,uint64_t);
void file_cyclic_retire_empty(NlFileValues *);
#endif
static NvmFileCodeInstruction cent(NvmFileRuntime *c,uint8_t op_expected){
 NvmFileCyclicFrameView v;CHECK(nvm_file_runtime_cyclic_frame_view(c,&v));
 NvmFileCodeInstruction in;CHECK(nvm_file_cyclic_hosted_instruction(nvm_file_runtime_cyclic_plan(c),v.frame.function,(uint16_t)v.frame.instruction,&in));
 CHECK(in.decoded.opcode==op_expected && !v.instruction_open);ROK(nvm_file_runtime_cyclic_enter(c));return in;
}
static void cfuel_limit(NvmFileRuntime *c){
 NvmFileCyclicFrameView v;NvmFileCodeInstruction in;CHECK(nvm_file_runtime_cyclic_frame_view(c,&v));
 CHECK(nvm_file_cyclic_hosted_instruction(nvm_file_runtime_cyclic_plan(c),v.frame.function,(uint16_t)v.frame.instruction,&in));
 CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_LIMIT);NvmFileCyclicExecutionReport r;
 CHECK(nvm_file_runtime_cyclic_report(c,&r) && r.fuel_exhausted && r.runtime.function==v.frame.function && r.runtime.instruction==in.byte_offset);
}
static NvmFileRuntime *ccreate(NvmModule *m,NvmFileRuntimeMode mode,uint64_t fuel,bool begin){
 size_t n;uint8_t *wire=serialize(m,&n);NvmFileRuntime *c=NULL;NvmFileCyclicOptions options={1,fuel};
 unsigned before=open_attempts;ROK(nvm_file_runtime_cyclic_create(wire,n,mode,&options,&c));
 CHECK(!nvm_file_runtime_plan(c) && nvm_file_runtime_cyclic_plan(c));
 memset(wire,0,n);free(wire);nvm_module_free(m);CHECK(open_attempts==before);
 NvmFileRuntimeStorage storage;NvmFileCyclicHostedStartup startup;
 CHECK(nvm_file_runtime_storage(c,&storage) && nvm_file_cyclic_hosted_startup(nvm_file_runtime_cyclic_plan(c),&startup));
 CHECK(storage.allocation_bound<=NVM_FILE_RUNTIME_BYTES && !startup.runtime_admitted);
#ifdef HOSTED_INSTRUMENT
 CHECK(tracked_bytes<=storage.allocation_bound);
#endif
 if(begin){ROK(nvm_file_runtime_begin(c));ROK(nvm_file_runtime_frame_start(c));}return c;
}
static NvmFileCyclicExecutionReport cfinish(NvmFileRuntime **c,NvmFileRuntimeStatus status,int64_t value){
#ifdef HOSTED_INSTRUMENT
 if(status==NVM_FILE_RUNTIME_OK){uint64_t owners=UINT64_MAX,borrows=UINT64_MAX;
  CHECK((*c)->complete && !(*c)->frame_count && !(*c)->region_count);
  CHECK(nl_file_values_live_slots((*c)->files,&owners,&borrows)==NL_FILE_VALUE_OK && !owners && !borrows);
 }
#endif
 NvmFileRuntimeView out;memset(&out,0xa5,sizeof out);NvmFileRuntimeView before=out;
 NvmFileCyclicExecutionReport first=nvm_file_runtime_cyclic_finish(*c,&out);
 CHECK(first.revision==1 && first.runtime.status==status);
 if(status==NVM_FILE_RUNTIME_OK)CHECK(out.type.tag==TAG_INT && out.values[0]==value && !first.runtime.cleanup.cleanup_failures);
 else CHECK(!memcmp(&out,&before,sizeof out));
 NvmFileCyclicExecutionReport again=nvm_file_runtime_cyclic_destroy(c,&out);
 CHECK(!*c && again.runtime.status==first.runtime.status && again.runtime.function==first.runtime.function &&
  again.runtime.instruction==first.runtime.instruction && again.instructions_started==first.instructions_started &&
  again.fuel_exhausted==first.fuel_exhausted && again.runtime.cleanup.cleanup_failures==first.runtime.cleanup.cleanup_failures);
 empty_host();return first;
}
static void cnext(NvmFileRuntime *c){ROK(nvm_file_runtime_frame_next(c,0));}
static void cpush(NvmFileRuntime *c,int64_t value){NvmFileCodeInstruction in=cent(c,OP_PUSH_I64);CHECK(in.decoded.operands[0].i64==value);scalar(c,fout(c,fv(c).stack_count),value);cnext(c);}
static void cstore(NvmFileRuntime *c,bool own){cent(c,own?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL);ROK(nvm_file_runtime_frame_store(c));cnext(c);}
static void cload(NvmFileRuntime *c,bool own){NvmFileCodeInstruction in=cent(c,own?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL);uint32_t dst=fout(c,fv(c).stack_count),src=fl(c,in.decoded.operands[0].u16);
 if(own)ROK(nvm_file_runtime_move(c,src,dst));else ROK(nvm_file_runtime_copy(c,src,dst));cnext(c);}
static void cpop(NvmFileRuntime *c){cent(c,OP_POP);ROK(nvm_file_runtime_drop(c,fs(c,fv(c).stack_count-1)));cnext(c);}
static void cservice(NvmFileRuntime *c,unsigned ordinal){NvmFileCodeInstruction in=cent(c,OP_FILE_SERVICE);uint32_t import;
 CHECK(nvm_file_cyclic_hosted_import(nvm_file_runtime_cyclic_plan(c),ordinal,&import) && import==in.decoded.operands[0].u32);
 NvmFileRuntimeFrameView f=fv(c);bool input=ordinal==1||ordinal==4;uint32_t scratch=f.staging_base+f.staging_slots-1;
 ROK(nvm_file_runtime_service(c,import,ordinal>=1&&ordinal<=3?fr(c,in.decoded.operands[1].u16):NS,input?fs(c,f.stack_count-1):NS,scratch));
 ROK(nvm_file_runtime_move(c,scratch,fout(c,f.stack_count-(input?1:0))));cnext(c);
}
static void cbranch(NvmFileRuntime *c,bool error){NvmFileCodeInstruction in=cent(c,OP_FILE_RESULT_BRANCH);arm(c,fl(c,in.decoded.operands[0].u16),error?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK);ROK(nvm_file_runtime_frame_next(c,error?1:0));}
static void ctake(NvmFileRuntime *c){NvmFileCodeInstruction in=cent(c,OP_FILE_RESULT_TAKE);ROK(nvm_file_runtime_take(c,fl(c,in.decoded.operands[0].u16),in.decoded.operands[1].u8?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK,fout(c,fv(c).stack_count)));cnext(c);}
static void cret(NvmFileRuntime *c){cent(c,OP_RET);ROK(nvm_file_runtime_frame_return(c));}
static void ccall(NvmFileRuntime *c,bool ref){cent(c,ref?OP_CALL_REF:OP_CALL);ROK(nvm_file_runtime_frame_call(c));}
static void cregion(NvmFileRuntime *c,bool begin){cent(c,begin?OP_REGION_BEGIN:OP_REGION_END);if(begin)ROK(nvm_file_runtime_frame_region_begin(c));else ROK(nvm_file_runtime_frame_region_end(c));cnext(c);}
static void cborrow(NvmFileRuntime *c){cent(c,OP_BORROW_LOCAL_EXCLUSIVE);ROK(nvm_file_runtime_frame_borrow(c));cnext(c);}
static void cend(NvmFileRuntime *c){cent(c,OP_FILE_END_BORROW);ROK(nvm_file_runtime_frame_end_reference(c));cnext(c);}
static void cbin(NvmFileRuntime *c,bool compare){cent(c,compare?OP_I64_GT_S:OP_I64_SUB);uint32_t a=fs(c,0),b=fs(c,1);int64_t x=view(c,a).values[0],y=view(c,b).values[0];
 ROK(nvm_file_runtime_drop(c,b));ROK(nvm_file_runtime_drop(c,a));ROK(nvm_file_runtime_scalar(c,a,compare?TAG_BOOL:TAG_INT,compare?x>y:x-y));cnext(c);}
static NvmModule *cloop(NvmFileNominalBindings *b,bool permuted,int64_t iterations){
 NvmModule *seed=fixture(permuted,b);nvm_module_free(seed);FrameSpec s={.locals=3,.types={-1,3,0},.result=-1};Body *p=&s.code;
 fi(p,iterations);one(p,OP_STORE_LOCAL,0);uint32_t head=p->n;one(p,OP_LOAD_LOCAL,0);fi(p,0);op(p,OP_I64_GT_S);uint32_t done=branch(p,OP_JMP_FALSE,0);
 service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,1);uint32_t error=branch(p,OP_FILE_RESULT_BRANCH,1);
 take_result(p,1,0);one(p,OP_OWN_STORE_LOCAL,2);op(p,OP_REGION_BEGIN);op(p,OP_BORROW_LOCAL_EXCLUSIVE);u16(p,0);u16(p,2);
 fi(p,255);service(p,*b,1,0);op(p,OP_POP);one(p,OP_FILE_END_BORROW,0);op(p,OP_REGION_END);
 one(p,OP_OWN_MOVE_LOCAL,2);service(p,*b,4,UINT16_MAX);op(p,OP_POP);one(p,OP_LOAD_LOCAL,0);fi(p,1);op(p,OP_I64_SUB);one(p,OP_STORE_LOCAL,0);
 uint32_t back=branch(p,OP_JMP,0);wr32(p->bytes+back+1,(uint32_t)((int32_t)head-(int32_t)back));target(p,done);fi(p,73);op(p,OP_RET);
 target(p,error);take_result(p,1,1);op(p,OP_POP);fi(p,0);op(p,OP_RET);
 return frame_module(&s,1,b,permuted,-1);
}
static void cheader(NvmFileRuntime *c,bool done){cload(c,false);cpush(c,0);cbin(c,true);cent(c,OP_JMP_FALSE);CHECK((view(c,fs(c,0)).values[0]==0)==done);ROK(nvm_file_runtime_drop(c,fs(c,0)));ROK(nvm_file_runtime_frame_next(c,done?1:0));}
static void citeration(NvmFileRuntime *c){
 cservice(c,0);cstore(c,true);cbranch(c,false);ctake(c);cstore(c,true);cregion(c,true);cborrow(c);cpush(c,255);cservice(c,1);cpop(c);cend(c);cregion(c,false);
 cload(c,true);cservice(c,4);cpop(c);cload(c,false);cpush(c,1);cbin(c,false);cstore(c,false);cent(c,OP_JMP);cnext(c);
}
static void cyclic_owner_calls(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned perm=0;perm<2;perm++){
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(owner_module(&b,perm!=0),(NvmFileRuntimeMode)mode,9,true);
  cservice(c,0);cservice(c,0);NvmFileRuntimeFrameView parent=fv(c);ccall(c,false);
  CHECK(view(c,fl(c,0)).owning && view(c,fl(c,1)).owning);
  cent(c,OP_FILE_DROP_LOCAL);ROK(nvm_file_runtime_drop(c,fl(c,1)));cnext(c);cload(c,true);cret(c);
  CHECK(view(c,fs(c,0)).owning && !view(c,parent.staging_base+parent.staging_slots-1).initialized);
  cent(c,OP_FILE_DROP_STACK);ROK(nvm_file_runtime_drop(c,fs(c,0)));cnext(c);cpush(c,37);cret(c);
  CHECK(cfinish(&c,NVM_FILE_RUNTIME_OK,37).instructions_started==9);
 }
}
static NvmModule *cswap(NvmFileNominalBindings *b){
 NvmModule *seed=fixture(false,b);nvm_module_free(seed);FrameSpec s={.locals=3,.types={3,3,3},.result=-1};Body *p=&s.code;
 service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,0);service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,1);
 one(p,OP_OWN_MOVE_LOCAL,0);one(p,OP_OWN_STORE_LOCAL,2);one(p,OP_OWN_MOVE_LOCAL,1);one(p,OP_OWN_STORE_LOCAL,0);one(p,OP_OWN_MOVE_LOCAL,2);one(p,OP_OWN_STORE_LOCAL,1);
 one(p,OP_FILE_DROP_LOCAL,0);one(p,OP_FILE_DROP_LOCAL,1);fi(p,31);op(p,OP_RET);return frame_module(&s,1,b,false,-1);
}
static void swapped_roots(void){
 for(unsigned mode=0;mode<2;mode++){
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cswap(&b),(NvmFileRuntimeMode)mode,14,true);
  cservice(c,0);cstore(c,true);cservice(c,0);cstore(c,true);
#ifdef HOSTED_INSTRUMENT
  uint32_t a=c->values[fl(c,0)].owner.slot,z=c->values[fl(c,1)].owner.slot;CHECK(a!=z);
#endif
  cload(c,true);cstore(c,true);cload(c,true);cstore(c,true);cload(c,true);cstore(c,true);
#ifdef HOSTED_INSTRUMENT
  CHECK(c->values[fl(c,0)].owner.slot==z && c->values[fl(c,1)].owner.slot==a);
#endif
  CHECK(!view(c,fl(c,2)).initialized);
  for(unsigned i=0;i<2;i++){NvmFileCodeInstruction in=cent(c,OP_FILE_DROP_LOCAL);CHECK(in.decoded.operands[0].u16==i);ROK(nvm_file_runtime_drop(c,fl(c,i)));cnext(c);}
  cpush(c,31);cret(c);CHECK(cfinish(&c,NVM_FILE_RUNTIME_OK,31).instructions_started==14);
 }
}
static void loops(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned permutation=0;permutation<2;permutation++)for(unsigned zero=0;zero<2;zero++){
  unsigned acquired=carrier_opened;NvmFileNominalBindings b;int64_t n=zero?0:258;
  NvmFileRuntime *c=ccreate(cloop(&b,permutation!=0,n),(NvmFileRuntimeMode)mode,100000,true);
#ifdef HOSTED_INSTRUMENT
  size_t execution_live=tracked_live,execution_bytes=tracked_bytes;
  allocation_budget=0;single_failure=false;failed_calls=0;
#endif
  cpush(c,n);cstore(c,false);for(int64_t i=0;i<n;i++){cheader(c,false);citeration(c);}cheader(c,true);cpush(c,73);cret(c);
#ifdef HOSTED_INSTRUMENT
  CHECK(!failed_calls && tracked_live==execution_live && tracked_bytes==execution_bytes);
  allocation_budget=-1;
#endif
  NvmFileCyclicExecutionReport r=cfinish(&c,NVM_FILE_RUNTIME_OK,73);CHECK(!r.fuel_exhausted && r.instructions_started==8+(uint64_t)n*24);
#ifdef HOSTED_INSTRUMENT
  CHECK(carrier_opened-acquired==(unsigned)n);
#else
  (void)acquired;
#endif
 }
}
static NvmModule *csimple(NvmFileNominalBindings *b,bool init){FrameSpec s[2]={{.result=-1},{.result=-3}};fi(&s[0].code,17);op(&s[0].code,OP_RET);op(&s[1].code,OP_RET);return frame_module(s,init?2:1,b,false,init?1:-1);}
static void fuel_and_lifecycle(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned init=0;init<2;init++)for(uint64_t fuel=0;fuel<5;fuel++){
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(csimple(&b,init),(NvmFileRuntimeMode)mode,fuel,true);uint64_t total=2+init,started=0;
  if(init){if(fuel){cent(c,OP_RET);ROK(nvm_file_runtime_complete_root(c,NS));started++;ROK(nvm_file_runtime_frame_start(c));}else cfuel_limit(c);}
  if(!init || fuel){if(started==fuel)cfuel_limit(c);else{cpush(c,17);started++;if(started==fuel)cfuel_limit(c);else{cent(c,OP_RET);ROK(nvm_file_runtime_complete_root(c,fs(c,0)));started++;}}}
  NvmFileCyclicExecutionReport r=cfinish(&c,fuel<total?NVM_FILE_RUNTIME_LIMIT:NVM_FILE_RUNTIME_OK,17);
  CHECK(r.instructions_started==(fuel<total?fuel:total) && r.fuel_exhausted==(fuel<total) && r.instruction_limit==fuel);
 }
 NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(csimple(&b,false),NVM_FILE_RUNTIME_VM,4,true);
 NvmFileRuntime *saved=c;CHECK(nvm_file_runtime_destroy(&saved,NULL).status==NVM_FILE_RUNTIME_STATE && saved==c);
 CHECK(nvm_file_runtime_finish(c,NULL).status==NVM_FILE_RUNTIME_STATE);cent(c,OP_PUSH_I64);
 CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_STATE);CHECK(cfinish(&c,NVM_FILE_RUNTIME_STATE,0).instructions_started==1);
 c=ccreate(csimple(&b,false),NVM_FILE_RUNTIME_VM,4,true);CHECK(nvm_file_runtime_scalar(c,fout(c,0),TAG_INT,17)==NVM_FILE_RUNTIME_STATE);cfinish(&c,NVM_FILE_RUNTIME_STATE,0);
 c=ccreate(csimple(&b,false),NVM_FILE_RUNTIME_VM,4,true);cent(c,OP_PUSH_I64);uint32_t root=fout(c,0);scalar(c,root,17);CHECK(nvm_file_runtime_complete_root(c,root)==NVM_FILE_RUNTIME_STATE);cfinish(&c,NVM_FILE_RUNTIME_STATE,0);
 c=ccreate(csimple(&b,false),NVM_FILE_RUNTIME_VM,4,true);cent(c,OP_PUSH_I64);CHECK(nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_ASSERT)==NVM_FILE_RUNTIME_ASSERT);CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_ASSERT);CHECK(!cfinish(&c,NVM_FILE_RUNTIME_ASSERT,0).fuel_exhausted);
}
static void reverse_kind_guard(void){
 NvmFileNominalBindings b;NvmFileRuntime *c=frame_context(csimple(&b,false),NVM_FILE_RUNTIME_VM),*saved=c;
 NvmFileCyclicFrameView frame;memset(&frame,0xa5,sizeof frame);NvmFileCyclicFrameView prior=frame;
 CHECK(!nvm_file_runtime_cyclic_frame_view(c,&frame) && !memcmp(&frame,&prior,sizeof frame));
 CHECK(!nvm_file_runtime_cyclic_plan(c));
 CHECK(nvm_file_runtime_cyclic_finish(c,NULL).runtime.status==NVM_FILE_RUNTIME_INVALID);
 CHECK(nvm_file_runtime_cyclic_destroy(&saved,NULL).runtime.status==NVM_FILE_RUNTIME_INVALID && saved==c);
 fpush(c,17);fret(c);frame_finish(&c,17);
}
static void invalid_options(void){
 NvmFileNominalBindings b;NvmModule *m=csimple(&b,false);size_t n;uint8_t *wire=serialize(m,&n);nvm_module_free(m);
 NvmFileCyclicOptions cases[]={{0,2},{2,2},{1,1000001}};
 for(unsigned i=0;i<3;i++){NvmFileRuntime *c=(NvmFileRuntime *)(uintptr_t)1;CHECK(nvm_file_runtime_cyclic_create(wire,n,NVM_FILE_RUNTIME_VM,&cases[i],&c)==NVM_FILE_RUNTIME_INVALID && c==(NvmFileRuntime *)(uintptr_t)1);}
 NvmFileRuntime *c=(NvmFileRuntime *)(uintptr_t)1;CHECK(nvm_file_runtime_cyclic_create(wire,n,NVM_FILE_RUNTIME_VM,NULL,&c)==NVM_FILE_RUNTIME_INVALID && c==(NvmFileRuntime *)(uintptr_t)1);
 CHECK(nvm_file_runtime_cyclic_abi(1,sizeof(NvmFileCyclicOptions),sizeof(NvmFileCyclicExecutionReport),sizeof(NvmFileRuntimeView),sizeof(NvmFileCyclicFrameView)));
 CHECK(!nvm_file_runtime_cyclic_abi(2,sizeof(NvmFileCyclicOptions),sizeof(NvmFileCyclicExecutionReport),sizeof(NvmFileRuntimeView),sizeof(NvmFileCyclicFrameView)));free(wire);
}
static NvmModule *cnested(NvmFileNominalBindings *b,bool permuted){
 NvmModule *seed=fixture(permuted,b);nvm_module_free(seed);FrameSpec s[3]={{.locals=2,.types={3,0},.result=-1},{.parameters=1,.locals=1,.types={0},.result=-1,.borrowed=1},{.parameters=1,.locals=1,.types={0},.result=-1,.borrowed=1}};
 Body *p=&s[0].code;service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,0);uint32_t error=branch(p,OP_FILE_RESULT_BRANCH,0);
 take_result(p,0,0);one(p,OP_OWN_STORE_LOCAL,1);op(p,OP_REGION_BEGIN);op(p,OP_BORROW_LOCAL_EXCLUSIVE);u16(p,0);u16(p,1);
 fcr(p,1,0);op(p,OP_POP);one(p,OP_FILE_END_BORROW,0);op(p,OP_REGION_END);one(p,OP_OWN_MOVE_LOCAL,1);service(p,*b,4,UINT16_MAX);op(p,OP_POP);fi(p,42);op(p,OP_RET);
 target(p,error);take_result(p,0,1);op(p,OP_POP);fi(p,0);op(p,OP_RET);
 fcr(&s[1].code,2,0);op(&s[1].code,OP_RET);fi(&s[2].code,7);op(&s[2].code,OP_RET);
 return frame_module(s,3,b,permuted,-1);
}
static void nested_borrows(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned permutation=0;permutation<2;permutation++){
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cnested(&b,permutation!=0),(NvmFileRuntimeMode)mode,20,true);
  cservice(c,0);cstore(c,true);cbranch(c,false);ctake(c);cstore(c,true);cregion(c,true);cborrow(c);ccall(c,true);CHECK(fv(c).depth==2);ccall(c,true);CHECK(fv(c).depth==3);
  cpush(c,7);cret(c);CHECK(fv(c).depth==2);cret(c);CHECK(fv(c).depth==1);cpop(c);cend(c);cregion(c,false);cload(c,true);cservice(c,4);cpop(c);cpush(c,42);cret(c);
  NvmFileCyclicExecutionReport r=cfinish(&c,NVM_FILE_RUNTIME_OK,42);CHECK(r.instructions_started==20 && !r.fuel_exhausted);
 }
 for(unsigned limit=8;limit<10;limit++){
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cnested(&b,false),NVM_FILE_RUNTIME_VM,limit,true);
  cservice(c,0);cstore(c,true);cbranch(c,false);ctake(c);cstore(c,true);cregion(c,true);cborrow(c);ccall(c,true);if(limit==9)ccall(c,true);
  cfuel_limit(c);NvmFileCyclicExecutionReport r=cfinish(&c,NVM_FILE_RUNTIME_LIMIT,0);CHECK(r.instructions_started==limit && r.fuel_exhausted);
 }
}
static void held_owner_refusal(void){
 for(unsigned mode=0;mode<2;mode++){
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cnested(&b,false),(NvmFileRuntimeMode)mode,100,true);
  cservice(c,0);cstore(c,true);cbranch(c,false);ctake(c);cstore(c,true);cregion(c,true);cborrow(c);
  cent(c,OP_CALL_REF);uint32_t dst=fv(c).staging_base,src=fl(c,1);
  CHECK(nvm_file_runtime_move(c,src,dst)==NVM_FILE_RUNTIME_BORROWED);
  CHECK(view(c,src).owning && !view(c,dst).initialized);
  CHECK(!cfinish(&c,NVM_FILE_RUNTIME_BORROWED,0).fuel_exhausted);
 }
}
static void service_fuel_and_error(void){
 for(unsigned which=0;which<4;which++){
  uint64_t limit=which==0?6:which==1?7:which==2?8:13;unsigned before=open_attempts;
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cloop(&b,false,1),NVM_FILE_RUNTIME_VM,limit,true);cpush(c,1);cstore(c,false);cheader(c,false);
  if(which){cservice(c,0);if(which>1){cstore(c,true);if(which>2){cbranch(c,false);ctake(c);cstore(c,true);cregion(c,true);cborrow(c);}}}
  cfuel_limit(c);
#ifdef HOSTED_INSTRUMENT
  CHECK(open_attempts-before==(which?1u:0u));
#else
  (void)before;
#endif
  CHECK(cfinish(&c,NVM_FILE_RUNTIME_LIMIT,0).instructions_started==limit);
 }
#ifdef HOSTED_INSTRUMENT
 NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cloop(&b,false,1),NVM_FILE_RUNTIME_VM,100,true);cpush(c,1);cstore(c,false);cheader(c,false);deny_open=true;cservice(c,0);deny_open=false;cstore(c,true);cbranch(c,true);ctake(c);cpop(c);cpush(c,0);cret(c);cfinish(&c,NVM_FILE_RUNTIME_OK,0);
 c=ccreate(cloop(&b,false,1),NVM_FILE_RUNTIME_VM,7,true);cpush(c,1);cstore(c,false);cheader(c,false);cservice(c,0);close_error[0]=EIO;close_index=0;cfuel_limit(c);
 NvmFileCyclicExecutionReport r=cfinish(&c,NVM_FILE_RUNTIME_LIMIT,0);CHECK(r.runtime.cleanup.cleanup_failures==1 && r.fuel_exhausted);memset(close_error,0,sizeof close_error);close_index=0;
#endif
}
#ifdef HOSTED_INSTRUMENT
static void forged_boundaries(void){
 for(unsigned which=0;which<6;which++){
  NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cloop(&b,false,1),NVM_FILE_RUNTIME_VM,1000,true);cpush(c,1);cstore(c,false);cheader(c,false);
  if(which==0){uint8_t old=c->frames[0].variant;c->frames[0].variant=255;CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_STATE);c->frames[0].variant=old;cfinish(&c,NVM_FILE_RUNTIME_STATE,0);continue;}
  if(which==1){uint32_t root=c->frames[0].staging_base;c->values[root].view=fr_view(fr_scalar_type(TAG_INT));c->values[root].view.fields=1;CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_STATE);fr_clear(c,root);cfinish(&c,NVM_FILE_RUNTIME_STATE,0);continue;}
  if(which==2){NlFileValue orphan={0};CHECK(nl_file_values_temp(c->files,&orphan)==NL_FILE_VALUE_OK);CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_STATE);CHECK(nl_file_value_drop(c->files,&orphan)==NL_FILE_VALUE_OK);cfinish(&c,NVM_FILE_RUNTIME_STATE,0);continue;}
  cservice(c,0);cstore(c,true);
  if(which==3){cent(c,OP_FILE_RESULT_BRANCH);CHECK(nvm_file_runtime_frame_next(c,1)==NVM_FILE_RUNTIME_STATE);cfinish(&c,NVM_FILE_RUNTIME_STATE,0);continue;}
  if(which==4){uint32_t root=fl(c,1);uint64_t generation=c->values[root].owner.generation;c->values[root].owner.generation++;CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_STALE);c->values[root].owner.generation=generation;cfinish(&c,NVM_FILE_RUNTIME_STALE,0);continue;}
  cbranch(c,false);ctake(c);cstore(c,true);cregion(c,true);cborrow(c);uint32_t ref=fr(c,0);uint64_t epoch=c->references[ref].borrow.epoch;c->references[ref].borrow.epoch++;
  CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_STATE);c->references[ref].borrow.epoch=epoch;cfinish(&c,NVM_FILE_RUNTIME_STATE,0);
 }
}
static void counter_boundaries(void){
 NvmFileNominalBindings b;NvmFileRuntime *c=ccreate(cloop(&b,false,1),NVM_FILE_RUNTIME_VM,100,true);cpush(c,1);cstore(c,false);cheader(c,false);cservice(c,0);cstore(c,true);near_limit(c,fl(c,1),UINT64_MAX);cbranch(c,false);cent(c,OP_FILE_RESULT_TAKE);
 uint32_t root=fl(c,1);CHECK(nvm_file_runtime_take(c,root,NVM_FILE_FLOW_ARM_OK,fout(c,0))==NVM_FILE_RUNTIME_LIMIT);CHECK(view(c,root).owning);CHECK(!cfinish(&c,NVM_FILE_RUNTIME_LIMIT,0).fuel_exhausted);
 c=ccreate(cloop(&b,false,1),NVM_FILE_RUNTIME_VM,100,true);cpush(c,1);cstore(c,false);cheader(c,false);cservice(c,0);cstore(c,true);cbranch(c,false);ctake(c);cstore(c,true);c->next_region=UINT64_MAX;cent(c,OP_REGION_BEGIN);
 CHECK(nvm_file_runtime_frame_region_begin(c)==NVM_FILE_RUNTIME_LIMIT && !c->region_count);CHECK(!cfinish(&c,NVM_FILE_RUNTIME_LIMIT,0).fuel_exhausted);
 NlFileValues *core=NULL;CHECK(nl_file_values_create(&core)==NL_FILE_VALUE_OK);NlFileValue open={0},file={0};CHECK(nl_file_values_temp(core,&open)==NL_FILE_VALUE_OK);CHECK(nl_file_open_take_ok(core,&open,&file)==NL_FILE_VALUE_OK);
 NlFileValueBorrow borrow={0};CHECK(nl_file_value_borrow(core,&file,&borrow)==NL_FILE_VALUE_OK);file_cyclic_epoch(core,&borrow,UINT64_MAX);CHECK(nl_file_value_borrow_validate(core,&borrow)==NL_FILE_VALUE_OK);CHECK(nl_file_value_end_borrow(core,&borrow)==NL_FILE_VALUE_OK);
 CHECK(nl_file_value_borrow(core,&file,&borrow)==NL_FILE_VALUE_LIMIT && !borrow.epoch);CHECK(nl_file_value_drop(core,&file)==NL_FILE_VALUE_OK);file_cyclic_retire_empty(core);unsigned attempts=open_attempts;
 CHECK(nl_file_values_temp(core,&open)==NL_FILE_VALUE_LIMIT && !open.invocation && open_attempts==attempts);CHECK(nl_file_values_destroy(core,NL_FILE_VALUE_OK).cleanup_failures==0);empty_host();
}
static void allocation_failures(void){
 NvmFileNominalBindings b;NvmModule *m=cloop(&b,false,0);size_t n;uint8_t *wire=serialize(m,&n);nvm_module_free(m);size_t baseline=tracked_live,bytes=tracked_bytes;
 unsigned create_failures=0,begin_failures=0;
 for(unsigned transient=0;transient<2;transient++){
  bool reached=false;NvmFileCyclicOptions measuring={1,100};NvmFileRuntime *measure=NULL;
  allocation_budget=1000000;ROK(nvm_file_runtime_cyclic_create(wire,n,NVM_FILE_RUNTIME_VM,&measuring,&measure));ROK(nvm_file_runtime_begin(measure));
  int calls=1000000-allocation_budget;allocation_budget=-1;CHECK(calls>0 && calls<2048);
  printf("I measure %d create/begin allocation attempts for %s failure prefixes\n",calls,transient?"single":"persistent");
  CHECK(nvm_file_runtime_cyclic_destroy(&measure,NULL).runtime.status==NVM_FILE_RUNTIME_STATE);
  for(int budget=0;budget<=calls;budget++){
   NvmFileRuntime *c=(NvmFileRuntime *)(uintptr_t)1;NvmFileCyclicOptions options={1,100};allocation_budget=budget;single_failure=transient!=0;failed_calls=0;
   NvmFileRuntimeStatus s=nvm_file_runtime_cyclic_create(wire,n,NVM_FILE_RUNTIME_VM,&options,&c);
   if(s!=NVM_FILE_RUNTIME_OK){CHECK(c==(NvmFileRuntime *)(uintptr_t)1);create_failures++;}
   else {s=nvm_file_runtime_begin(c);if(s!=NVM_FILE_RUNTIME_OK)begin_failures++;allocation_budget=-1;single_failure=false;
    NvmFileCyclicExecutionReport r=nvm_file_runtime_cyclic_destroy(&c,NULL);CHECK(!c && (s==NVM_FILE_RUNTIME_OK?r.runtime.status==NVM_FILE_RUNTIME_STATE:r.runtime.status==s));if(s==NVM_FILE_RUNTIME_OK)reached=true;}
   CHECK(budget<calls?failed_calls>0:failed_calls==0);
   allocation_budget=-1;single_failure=false;CHECK(tracked_live==baseline && tracked_bytes==bytes);empty_host();
   NvmFileRuntime *fresh=NULL;ROK(nvm_file_runtime_cyclic_create(wire,n,NVM_FILE_RUNTIME_VM,&options,&fresh));ROK(nvm_file_runtime_begin(fresh));CHECK(nvm_file_runtime_cyclic_destroy(&fresh,NULL).runtime.status==NVM_FILE_RUNTIME_STATE && !fresh);CHECK(tracked_live==baseline && tracked_bytes==bytes);
   if(budget==calls)CHECK(reached);
  }
  CHECK(reached);
 }
 CHECK(create_failures && begin_failures);printf("PASS %u create and %u begin allocation refusals with fresh recovery\n",create_failures,begin_failures);free(wire);
}
#endif
#define CYCLIC_CASE(name) do { printf("I begin cyclic carrier case %s\n",#name);name(); } while(0)
#ifndef FILE_CYCLIC_RUNTIME_MAIN
#define FILE_CYCLIC_RUNTIME_MAIN main
#endif
int FILE_CYCLIC_RUNTIME_MAIN(void){
 CHECK(setvbuf(stdout,NULL,_IONBF,0)==0);
 printf("I begin retained acyclic carrier/frame corpus\n");CHECK(prior_file_frames_main()==0);unsigned before=checks;
 CYCLIC_CASE(invalid_options);CYCLIC_CASE(reverse_kind_guard);CYCLIC_CASE(fuel_and_lifecycle);
 CYCLIC_CASE(nested_borrows);CYCLIC_CASE(cyclic_owner_calls);CYCLIC_CASE(swapped_roots);
 CYCLIC_CASE(held_owner_refusal);CYCLIC_CASE(service_fuel_and_error);CYCLIC_CASE(loops);
#ifdef HOSTED_INSTRUMENT
 CYCLIC_CASE(forged_boundaries);CYCLIC_CASE(counter_boundaries);CYCLIC_CASE(allocation_failures);CHECK(!tracked_live && !tracked_bytes);
#endif
 empty_host();printf("PASS %u manual cyclic carrier/fuel checks; no cyclic opcode dispatcher\n",checks-before);return 0;
}
