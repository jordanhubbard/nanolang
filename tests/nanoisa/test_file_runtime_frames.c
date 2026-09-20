/* I manually drive checked frame transfers at explicit prepared instruction
 * sites. I do not install an opcode loop, VM route or generated-native target. */
#define FILE_RUNTIME_MAIN prior_file_runtime_main
#include "test_file_runtime.c"
#undef FILE_RUNTIME_MAIN
#include "../../src/nanoisa/file_runtime_frames.h"
#ifdef HOSTED_INSTRUMENT
void file_frame_set_generation(NlFileValues *,NlFileValue *,uint64_t);
/* Included fixture builders and hooked providers own tracked allocations.
 * Keep every new helper allocation/free in that same instrumented domain. */
#define malloc file_test_malloc
#define calloc file_test_calloc
#define realloc file_test_realloc
#define free file_test_free
#endif
/* Negative types are scalar: -1 INT, -2 BOOL, -3 VOID. Others are exact catalog
 * ordinals. Borrow bits are formal parameter modes, not inferred tag authority. */
typedef struct {uint16_t parameters,locals;int types[8],result;uint8_t borrowed;Body code;} FrameSpec;
static void fi(Body *c,int64_t v){op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)op(c,(uint8_t)((uint64_t)v>>(8*i)));}
static void fc(Body *c,uint32_t fn){op(c,OP_CALL);u32(c,fn);}
static void fcr(Body *c,uint32_t fn,uint16_t ref){op(c,OP_CALL_REF);u32(c,fn);u16(c,ref);}
static uint8_t ftag(int t){return t==-1?TAG_INT:t==-2?TAG_BOOL:t==-3?TAG_VOID:t<3?TAG_STRUCT:TAG_UNION;}
static NvmModule *frame_module(FrameSpec *spec,unsigned count,NvmFileNominalBindings *b,bool permute,int init){
 NvmModule *m=fixture(permute,b);CHECK(count && count<=64);
 for(unsigned f=1;f<count;f++){NvmFunctionEntry copy=m->functions[0];CHECK(nvm_add_function(m,&copy)==f);}
 size_t size=24;for(unsigned f=0;f<count;f++){CHECK(spec[f].locals<=8 && spec[f].parameters<=spec[f].locals);size+=12+8*spec[f].locals;}
 free(m->ownership_data);m->ownership_data=calloc(1,size);CHECK(m->ownership_data);m->ownership_size=(uint32_t)size;
 uint8_t *o=m->ownership_data;wr32(o,1);wr32(o+4,9);for(unsigned i=0;i<8;i++)o[8+b->layouts[i]]=(i==0||i==3)?3:1;wr32(o+20,count);
 free(m->code);m->code=malloc(count);CHECK(m->code);m->code_size=m->code_capacity=count;size_t at=24;
 for(unsigned f=0;f<count;f++){
  FrameSpec *s=&spec[f];NvmFunctionEntry *fn=&m->functions[f];fn->arity=s->parameters;fn->local_count=s->locals;
  fn->result_count=s->result!=-3;fn->result_tag=ftag(s->result);
  fn->name_idx=string(m,(int)f==init?"__init__":f?"helper":"main");fn->code_offset=f;fn->code_length=1;m->code[f]=OP_RET;
  o[at]=(uint8_t)s->locals;o[at+1]=(uint8_t)(s->locals>>8);o[at+2]=(uint8_t)s->parameters;o[at+3]=(uint8_t)(s->parameters>>8);
  desc(o+at+4,ftag(s->result),0,s->result<0?NVM_V2_NO_INDEX:b->layouts[s->result]);at+=12;uint8_t params[8];
  for(unsigned i=0;i<s->locals;i++){int t=s->types[i];params[i]=ftag(t);desc(o+at,ftag(t),(s->borrowed&(1u<<i))?2:0,t<0?NVM_V2_NO_INDEX:b->layouts[t]);at+=8;}
  CHECK(nvm_set_function_param_types(m,f,params,s->parameters));
 }
 CHECK(at==size);for(unsigned f=0;f<count;f++)setbody(m,f,spec[f].code);return m;
}
static NvmFileRuntime *frame_context(NvmModule *m,NvmFileRuntimeMode mode){
 size_t size;uint8_t *bytes=serialize(m,&size);NvmFileRuntime *c=NULL;unsigned before=open_attempts;
 ROK(nvm_file_runtime_create(bytes,size,mode,&c));memset(bytes,0,size);free(bytes);nvm_module_free(m);
 NvmFileRuntimeStorage storage;NvmFileHostedStartup startup;
 CHECK(nvm_file_runtime_storage(c,&storage) && nvm_file_hosted_startup(nvm_file_runtime_plan(c),&startup));
 CHECK(storage.values==(mode==NVM_FILE_RUNTIME_VM?startup.vm_value_slots:startup.native_value_slots));
#ifdef HOSTED_INSTRUMENT
 size_t core;CHECK(nl_file_values_storage_bound(&core));
 CHECK(storage.allocation_bound==startup.allocation_bound+sizeof(*c)+storage.values*sizeof(*c->values)+
  storage.references*sizeof(*c->references)+storage.regions*sizeof(*c->regions)+storage.frames*sizeof(*c->frames)+core);
#endif
 ROK(nvm_file_runtime_begin(c));ROK(nvm_file_runtime_frame_start(c));CHECK(open_attempts==before);return c;
}
static NvmFileRuntimeFrameView fv(NvmFileRuntime *c){NvmFileRuntimeFrameView v;CHECK(nvm_file_runtime_frame_view(c,&v));return v;}
static NvmFileCodeInstruction fat(NvmFileRuntime *c,uint8_t opcode){NvmFileRuntimeFrameView v=fv(c);NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
 CHECK(nvm_file_hosted_instruction(nvm_file_runtime_plan(c),v.function,(uint16_t)v.instruction,&in,&fact));CHECK(in.decoded.opcode==opcode && fact.reachable);return in;}
static uint32_t fl(NvmFileRuntime *c,unsigned i){uint32_t n=NS;CHECK(nvm_file_runtime_frame_local(c,(uint16_t)i,&n));return n;}
static uint32_t fs(NvmFileRuntime *c,unsigned i){uint32_t n=NS;CHECK(nvm_file_runtime_frame_operand(c,(uint16_t)i,&n));return n;}
static uint32_t fr(NvmFileRuntime *c,unsigned i){uint32_t n=NS;CHECK(nvm_file_runtime_frame_reference(c,(uint16_t)i,&n));return n;}
static uint32_t fout(NvmFileRuntime *c,unsigned i){uint32_t n=NS;CHECK(nvm_file_runtime_frame_reserve(c,(uint16_t)i,&n));return n;}
static void fnxt(NvmFileRuntime *c){ROK(nvm_file_runtime_frame_next(c,0));}
static void fpush(NvmFileRuntime *c,int64_t value){NvmFileCodeInstruction in=fat(c,OP_PUSH_I64);CHECK(in.decoded.operands[0].i64==value);NvmFileRuntimeFrameView f=fv(c);scalar(c,fout(c,f.stack_count),value);fnxt(c);}
static void fpop(NvmFileRuntime *c,bool owner){fat(c,owner?OP_FILE_DROP_STACK:OP_POP);NvmFileRuntimeFrameView f=fv(c);CHECK(f.stack_count);ROK(nvm_file_runtime_drop(c,fs(c,f.stack_count-1)));fnxt(c);}
static void fstore(NvmFileRuntime *c,bool owner){fat(c,owner?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL);ROK(nvm_file_runtime_frame_store(c));fnxt(c);}
static void fload(NvmFileRuntime *c,bool owner){NvmFileCodeInstruction in=fat(c,owner?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL);NvmFileRuntimeFrameView f=fv(c);
 uint32_t src=fl(c,in.decoded.operands[0].u16),dst=fout(c,f.stack_count);if(owner)ROK(nvm_file_runtime_move(c,src,dst));else ROK(nvm_file_runtime_copy(c,src,dst));fnxt(c);}
static void fservice(NvmFileRuntime *c,unsigned ordinal){NvmFileCodeInstruction in=fat(c,OP_FILE_SERVICE);NvmFileRuntimeFrameView f=fv(c);uint32_t actual;
 CHECK(nvm_file_hosted_import(nvm_file_runtime_plan(c),in.decoded.operands[0].u32,&actual) && actual==ordinal);
 bool input=ordinal==1||ordinal==4;uint32_t src=input?fs(c,f.stack_count-1):NS;
 /* A service result uses reserved staging, then the consumed stack position.
  * I keep this helper explicit and do not claim a production opcode handler. */
 uint32_t dst=f.staging_base+f.staging_slots-1;CHECK(!view(c,dst).initialized);
 ROK(nvm_file_runtime_service(c,in.decoded.operands[0].u32,ordinal>=1&&ordinal<=3?fr(c,in.decoded.operands[1].u16):NS,src,dst));
 uint32_t output=fout(c,f.stack_count-(input?1:0));ROK(nvm_file_runtime_move(c,dst,output));fnxt(c);
}
static void fbranch_ok(NvmFileRuntime *c){NvmFileCodeInstruction in=fat(c,OP_FILE_RESULT_BRANCH);arm(c,fl(c,in.decoded.operands[0].u16),NVM_FILE_FLOW_ARM_OK);fnxt(c);}
static void ftake(NvmFileRuntime *c){NvmFileCodeInstruction in=fat(c,OP_FILE_RESULT_TAKE);NvmFileRuntimeFrameView f=fv(c);
 ROK(nvm_file_runtime_take(c,fl(c,in.decoded.operands[0].u16),in.decoded.operands[1].u8?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK,fout(c,f.stack_count)));fnxt(c);}
static void fret(NvmFileRuntime *c){fat(c,OP_RET);ROK(nvm_file_runtime_frame_return(c));}
static void frame_finish(NvmFileRuntime **c,int64_t expected){NvmFileRuntimeView out={0};NvmFileRuntimeReport r=nvm_file_runtime_destroy(c,&out);
 CHECK(r.status==NVM_FILE_RUNTIME_OK && r.acquired && !r.cleanup.cleanup_failures && !*c && out.type.tag==TAG_INT && out.values[0]==expected);empty_host();}
static void fb_begin(NvmFileRuntime *c){fat(c,OP_REGION_BEGIN);ROK(nvm_file_runtime_frame_region_begin(c));fnxt(c);}
static void fb_end(NvmFileRuntime *c){fat(c,OP_REGION_END);ROK(nvm_file_runtime_frame_region_end(c));fnxt(c);}
static void fb_borrow(NvmFileRuntime *c){fat(c,OP_BORROW_LOCAL_EXCLUSIVE);ROK(nvm_file_runtime_frame_borrow(c));fnxt(c);}
static void fb_release(NvmFileRuntime *c){fat(c,OP_FILE_END_BORROW);ROK(nvm_file_runtime_frame_end_reference(c));fnxt(c);}
static void frame_call_checked(NvmFileRuntime *c){
#ifdef HOSTED_INSTRUMENT
 size_t bytes=tracked_bytes,allocs=failed_calls;allocation_budget=0;
#endif
 ROK(nvm_file_runtime_frame_call(c));
#ifdef HOSTED_INSTRUMENT
 CHECK(!allocation_budget && failed_calls==allocs && tracked_bytes==bytes);allocation_budget=-1;
#endif
}
static void frame_return_checked(NvmFileRuntime *c){
#ifdef HOSTED_INSTRUMENT
 size_t bytes=tracked_bytes,allocs=failed_calls;allocation_budget=0;
#endif
 fret(c);
#ifdef HOSTED_INSTRUMENT
 CHECK(!allocation_budget && failed_calls==allocs && tracked_bytes==bytes);allocation_budget=-1;
#endif
}
static NvmModule *overlap_module(NvmFileNominalBindings *b,bool permuted){
 /* I obtain catalog indices before emitting the service operands. */
 NvmModule *seed=fixture(permuted,b);nvm_module_free(seed);FrameSpec s[3]={0};
 s[0]=(FrameSpec){.locals=3,.types={0,3,-1},.result=-1};s[1]=(FrameSpec){.parameters=3,.locals=3,.types={0,-1,-1},.result=-1,.borrowed=1};
 s[2]=(FrameSpec){.parameters=1,.locals=1,.types={0},.result=-3,.borrowed=1};
 Body *p=&s[0].code;service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,1);uint32_t err=branch(p,OP_FILE_RESULT_BRANCH,1);
 take_result(p,1,0);one(p,OP_OWN_STORE_LOCAL,0);fi(p,901);one(p,OP_STORE_LOCAL,2);op(p,OP_REGION_BEGIN);op(p,OP_BORROW_LOCAL_EXCLUSIVE);u16(p,20);u16(p,0);
 fi(p,702);for(unsigned i=0;i<2;i++){fi(p,11+i);fi(p,22+i);fcr(p,1,20);op(p,OP_POP);}op(p,OP_POP);
 one(p,OP_FILE_END_BORROW,20);op(p,OP_REGION_END);one(p,OP_OWN_MOVE_LOCAL,0);service(p,*b,4,UINT16_MAX);op(p,OP_POP);fi(p,909);op(p,OP_RET);
 target(p,err);take_result(p,1,1);op(p,OP_POP);fi(p,0);op(p,OP_RET);
 fcr(&s[1].code,2,0);one(&s[1].code,OP_LOAD_LOCAL,2);op(&s[1].code,OP_RET);
 op(&s[2].code,OP_REGION_BEGIN);op(&s[2].code,OP_REGION_END);op(&s[2].code,OP_RET);
 return frame_module(s,3,b,permuted,-1);
}
static void overlap_and_aliases(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned perm=0;perm<2;perm++){
  NvmFileNominalBindings b;NvmFileRuntime *c=frame_context(overlap_module(&b,perm!=0),(NvmFileRuntimeMode)mode);
  CHECK(fv(c).mode==(NvmFileRuntimeMode)mode);fservice(c,0);fstore(c,true);fbranch_ok(c);ftake(c);fstore(c,true);
  fpush(c,901);fstore(c,false);fb_begin(c);fb_borrow(c);fpush(c,702);
  for(unsigned repeat=0;repeat<2;repeat++){
   fpush(c,11+repeat);fpush(c,22+repeat);NvmFileRuntimeFrameView parent=fv(c);uint32_t origin=fr(c,20);
   uint32_t old_later=fs(c,2);frame_call_checked(c);NvmFileRuntimeFrameView child=fv(c);
   CHECK(child.depth==2 && child.function==1 && child.stack_count==0 && child.reference_base==256);
   CHECK(child.locals_base==parent.stack_base+(mode?parent.operand_peak:1));
   if(!mode)CHECK(child.locals_base+1==old_later); /* local1 overwrites the later argument only AFTER staging */
   CHECK(view(c,fl(c,0)).formal && !view(c,fl(c,0)).owning && view(c,fl(c,1)).values[0]==11+repeat && view(c,fl(c,2)).values[0]==22+repeat);
   CHECK(view(c,parent.locals_base+2).values[0]==901 && view(c,parent.stack_base).values[0]==702);
   frame_call_checked(c);NvmFileRuntimeFrameView nested=fv(c);CHECK(nested.depth==3 && nested.reference_base==512 && view(c,fl(c,0)).formal);
   fb_begin(c);fb_end(c);frame_return_checked(c);CHECK(fv(c).depth==2 && view(c,fl(c,0)).formal);
   fload(c,false);frame_return_checked(c);CHECK(fv(c).depth==1 && view(c,fs(c,1)).values[0]==22+repeat);
   if(!mode)CHECK(fs(c,1)==child.locals_base); /* caller result replaces cleared callee first local */
   CHECK(view(c,parent.locals_base).owning);ROK(nvm_file_runtime_service(c,b.imports[2],origin,NS,parent.staging_base));
   ROK(nvm_file_runtime_drop(c,parent.staging_base));fpop(c,false);
  }
  fpop(c,false);fb_release(c);fb_end(c);fload(c,true);fservice(c,4);fpop(c,false);fpush(c,909);fret(c);frame_finish(&c,909);
 }
}
static NvmModule *owner_module(NvmFileNominalBindings *b,bool permute){
 NvmModule *seed=fixture(permute,b);nvm_module_free(seed);FrameSpec s[2]={0};
 s[0].result=-1;s[1]=(FrameSpec){.parameters=2,.locals=2,.types={3,3},.result=3};
 service(&s[0].code,*b,0,UINT16_MAX);service(&s[0].code,*b,0,UINT16_MAX);fc(&s[0].code,1);op(&s[0].code,OP_FILE_DROP_STACK);fi(&s[0].code,37);op(&s[0].code,OP_RET);
 one(&s[1].code,OP_FILE_DROP_LOCAL,1);one(&s[1].code,OP_OWN_MOVE_LOCAL,0);op(&s[1].code,OP_RET);
 return frame_module(s,2,b,permute,-1);
}
static void owner_returns(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned perm=0;perm<2;perm++){
  NvmFileNominalBindings b;NvmFileRuntime *c=frame_context(owner_module(&b,perm!=0),(NvmFileRuntimeMode)mode);
  fservice(c,0);fservice(c,0);NvmFileRuntimeFrameView parent=fv(c);frame_call_checked(c);NvmFileRuntimeFrameView child=fv(c);
  CHECK(view(c,fl(c,0)).owning && view(c,fl(c,1)).owning);fat(c,OP_FILE_DROP_LOCAL);ROK(nvm_file_runtime_drop(c,fl(c,1)));fnxt(c);
  fload(c,true);frame_return_checked(c);CHECK(view(c,fs(c,0)).owning && !view(c,parent.staging_base+parent.staging_slots-1).initialized);
  if(!mode){CHECK(fs(c,0)==child.locals_base);}fpop(c,true);fpush(c,37);fret(c);frame_finish(&c,37);
 }
}
static void file_and_passive_result_returns(void){
 for(unsigned mode=0;mode<2;mode++){
  NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,mode!=0);setbody(m,0,lifecycle_code(b));
  NvmFileRuntime *c=frame_context(m,(NvmFileRuntimeMode)mode);
  fservice(c,0);fstore(c,true);fbranch_ok(c);ftake(c);fstore(c,true);fb_begin(c);fb_borrow(c);
  fpush(c,0);fservice(c,1);fpop(c,false);fservice(c,2);fpop(c,false);fservice(c,3);fpop(c,false);
  fpush(c,0);frame_call_checked(c);CHECK(fv(c).function==2 && view(c,fl(c,0)).formal);
  fload(c,false);fservice(c,1);CHECK(view(c,fs(c,0)).type.global_index==b.layouts[4]);
  frame_return_checked(c);CHECK(view(c,fs(c,0)).type.global_index==b.layouts[4]);fpop(c,false);
  fb_release(c);fb_end(c);fload(c,true);frame_call_checked(c);CHECK(fv(c).function==1);
  fload(c,true);frame_return_checked(c);CHECK(view(c,fs(c,0)).owning && view(c,fs(c,0)).type.global_index==b.layouts[0]);
  fservice(c,4);fstore(c,false);fbranch_ok(c);ftake(c);fpop(c,false);fpush(c,0);fret(c);frame_finish(&c,0);
 }
}
static void equal_modes_depth_and_getters(void){
 for(unsigned mode=0;mode<2;mode++){
  NvmFileNominalBindings b;FrameSpec single={.result=-1};fi(&single.code,19);op(&single.code,OP_RET);
  NvmFileRuntime *c=frame_context(frame_module(&single,1,&b,false,-1),(NvmFileRuntimeMode)mode);
  NvmFileHostedStartup startup;CHECK(nvm_file_hosted_startup(nvm_file_runtime_plan(c),&startup));
  CHECK(startup.vm_value_slots==startup.native_value_slots && fv(c).mode==(NvmFileRuntimeMode)mode);
  uint32_t slot=123456;CHECK(!nvm_file_runtime_frame_local(c,0,&slot) && slot==123456);
  CHECK(!nvm_file_runtime_frame_operand(c,0,&slot) && slot==123456);
  CHECK(!nvm_file_runtime_frame_reserve(c,fv(c).operand_peak,&slot) && slot==123456);
  CHECK(!nvm_file_runtime_frame_reference(c,256,&slot) && slot==123456);
  NvmFileRuntimeFrameView sentinel;memset(&sentinel,0xa5,sizeof sentinel);NvmFileRuntimeFrameView old=sentinel;
  CHECK(!nvm_file_runtime_frame_view(NULL,&sentinel) && !memcmp(&sentinel,&old,sizeof old));
  fpush(c,19);fret(c);CHECK(!nvm_file_runtime_frame_view(c,&sentinel) && !memcmp(&sentinel,&old,sizeof old));frame_finish(&c,19);
  FrameSpec *chain=calloc(64,sizeof *chain);CHECK(chain);
  for(unsigned i=0;i<64;i++){chain[i].result=-1;if(i<63)fc(&chain[i].code,i+1);else fi(&chain[i].code,64);op(&chain[i].code,OP_RET);}
  NvmModule *m=frame_module(chain,64,&b,false,-1);free(chain);c=frame_context(m,(NvmFileRuntimeMode)mode);
  CHECK(nvm_file_hosted_startup(nvm_file_runtime_plan(c),&startup) && startup.frames==64);
  for(unsigned depth=1;depth<64;depth++){CHECK(fv(c).depth==depth);frame_call_checked(c);}
  CHECK(fv(c).depth==64 && fv(c).reference_base==63*256);fpush(c,64);
  for(unsigned depth=64;depth;depth--){CHECK(fv(c).depth==depth);frame_return_checked(c);}
  frame_finish(&c,64);
 }
}
static void frame_refusals(void){
 for(unsigned which=0;which<7;which++){
  NvmFileNominalBindings b;NvmFileRuntime *c=frame_context(overlap_module(&b,false),NVM_FILE_RUNTIME_VM);
  NvmFileRuntimeStatus expected=NVM_FILE_RUNTIME_STATE;
  if(which==0){CHECK(nvm_file_runtime_frame_call(c)==expected);finish_bad(&c,expected);continue;}
  if(which==1){CHECK(nvm_file_runtime_frame_return(c)==expected);finish_bad(&c,expected);continue;}
  fservice(c,0);fstore(c,true);fbranch_ok(c);ftake(c);fstore(c,true);fpush(c,901);fstore(c,false);fb_begin(c);fb_borrow(c);fpush(c,702);fpush(c,11);fpush(c,22);
  if(which==2){
   uint32_t bad=fs(c,2);ROK(nvm_file_runtime_drop(c,bad));ROK(nvm_file_runtime_scalar(c,bad,TAG_BOOL,1));
   NvmFileRuntimeFrameView parent=fv(c);expected=NVM_FILE_RUNTIME_TYPE;CHECK(nvm_file_runtime_frame_call(c)==expected);
   CHECK(view(c,parent.stack_base+1).values[0]==11 && view(c,bad).type.tag==TAG_BOOL && !view(c,parent.staging_base).initialized);
  }else if(which==3){
   uint32_t missing=fs(c,2),retained=fs(c,1);ROK(nvm_file_runtime_drop(c,missing));CHECK(nvm_file_runtime_frame_call(c)==expected);
   CHECK(view(c,retained).values[0]==11);
  }else{
   frame_call_checked(c);
   if(which==4){expected=NVM_FILE_RUNTIME_BORROWED;CHECK(nvm_file_runtime_frame_end_reference(c)==expected);}
   if(which==5){frame_call_checked(c);fb_begin(c);ROK(nvm_file_runtime_region_end(c));CHECK(nvm_file_runtime_frame_region_end(c)==expected);}
   if(which==6){CHECK(nvm_file_runtime_frame_next(c,0)==expected);}
  }
  finish_bad(&c,expected);
 }
 /* Exact store refuses a scalar in the OpenResult local without consuming it. */
 NvmFileNominalBindings b;FrameSpec s={.result=-1};fi(&s.code,10);fi(&s.code,20);op(&s.code,OP_POP);op(&s.code,OP_RET);
 NvmFileRuntime *c=frame_context(frame_module(&s,1,&b,false,-1),NVM_FILE_RUNTIME_VM);
 fpush(c,10);fpush(c,20);uint32_t hidden=fs(c,1),prefix=fs(c,0);ROK(nvm_file_runtime_drop(c,hidden));
 ROK(nvm_file_runtime_service(c,b.imports[0],NS,NS,hidden));
 CHECK(nvm_file_runtime_frame_next(c,0)==NVM_FILE_RUNTIME_STATE);CHECK(view(c,hidden).owning && view(c,prefix).values[0]==10);finish_bad(&c,NVM_FILE_RUNTIME_STATE);
 c=frame_context(overlap_module(&b,false),NVM_FILE_RUNTIME_VM);fservice(c,0);uint32_t src=fs(c,0),dst=fl(c,1);
 ROK(nvm_file_runtime_drop(c,src));scalar(c,src,71);CHECK(nvm_file_runtime_frame_store(c)==NVM_FILE_RUNTIME_TYPE);
 CHECK(view(c,src).values[0]==71 && !view(c,dst).initialized);finish_bad(&c,NVM_FILE_RUNTIME_TYPE);
}
#ifdef HOSTED_INSTRUMENT
static unsigned owning_roots(NvmFileRuntime *c){unsigned count=0;for(uint32_t i=0;i<c->storage.values;i++)count+=c->values[i].view.initialized&&c->values[i].view.owning;return count;}
static void near_limit(NvmFileRuntime *c,uint32_t root,uint64_t generation){CHECK(view(c,root).owning);file_frame_set_generation(c->files,&c->values[root].owner,generation);}
static void partial_generation_failures(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned failure=0;failure<6;failure++){
  NvmFileNominalBindings b;NvmFileRuntime *c=frame_context(owner_module(&b,true),(NvmFileRuntimeMode)mode);
  fservice(c,0);fservice(c,0);NvmFileRuntimeFrameView parent=fv(c);uint32_t a=fs(c,0),z=fs(c,1);
  unsigned before=closed;
  if(failure<4){
   near_limit(c,(failure&1)?z:a,failure<2?UINT64_MAX:UINT64_MAX-1);
   CHECK(nvm_file_runtime_frame_call(c)==NVM_FILE_RUNTIME_LIMIT && owning_roots(c)==2);
   uint32_t base=parent.stack_base+(mode?parent.operand_peak:0);
   CHECK(view(c,failure==0?a:failure==3?base:parent.staging_base).owning);
   CHECK(view(c,failure<2?z:parent.staging_base+1).owning);
   CHECK(c->frame_count==1 && c->report.function==0 && c->report.instruction==parent.byte_offset);
  }else{
   frame_call_checked(c);fat(c,OP_FILE_DROP_LOCAL);ROK(nvm_file_runtime_drop(c,fl(c,1)));fnxt(c);fload(c,true);
   NvmFileRuntimeFrameView child=fv(c);uint32_t result=fs(c,0);near_limit(c,result,failure==4?UINT64_MAX:UINT64_MAX-1);
   CHECK(nvm_file_runtime_frame_return(c)==NVM_FILE_RUNTIME_LIMIT && owning_roots(c)==1);
   CHECK(view(c,failure==4?result:parent.staging_base+parent.staging_slots-1).owning);
   CHECK(c->report.function==1 && c->report.instruction==child.byte_offset);
  }
  reenter=c;close_index=0;close_error[0]=EIO;NvmFileRuntimeReport r=finish_bad(&c,NVM_FILE_RUNTIME_LIMIT);reenter=NULL;
  CHECK(r.cleanup.cleanup_failures==1 && r.cleanup.first_cleanup.host_errno==EIO);
  CHECK(closed==before+2);memset(close_error,0,sizeof close_error);close_index=0;
 }
}
#endif
static NvmModule *init_frame_module(NvmFileNominalBindings *b){
 NvmModule *seed=fixture(false,b);nvm_module_free(seed);FrameSpec s[2]={0};s[0].result=-1;fi(&s[0].code,52);op(&s[0].code,OP_RET);
 s[1]=(FrameSpec){.locals=2,.types={3,0},.result=-3};Body *p=&s[1].code;
 service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,0);uint32_t error=branch(p,OP_FILE_RESULT_BRANCH,0);
 take_result(p,0,0);one(p,OP_OWN_STORE_LOCAL,1);one(p,OP_OWN_MOVE_LOCAL,1);service(p,*b,4,UINT16_MAX);op(p,OP_POP);op(p,OP_RET);
 target(p,error);take_result(p,0,1);op(p,OP_POP);op(p,OP_RET);return frame_module(s,2,b,false,1);
}
static void initializer_frames(void){
 for(unsigned mode=0;mode<2;mode++){
  unsigned cases=1;
#ifdef HOSTED_INSTRUMENT
  cases=2;
#endif
  for(unsigned fail=0;fail<cases;fail++){
   NvmFileNominalBindings b;NvmFileRuntime *c=frame_context(init_frame_module(&b),(NvmFileRuntimeMode)mode);CHECK(fv(c).function==1);
   fservice(c,0);fstore(c,true);fbranch_ok(c);ftake(c);fstore(c,true);fload(c,true);
   if(fail){close_index=0;close_error[0]=EIO;}
   fservice(c,4);if(fail){arm(c,fs(c,0),NVM_FILE_FLOW_ARM_ERROR);CHECK(view(c,fs(c,0)).values[5]==1);}
   fpop(c,false);unsigned before=open_attempts;
   if(fail){
    CHECK(nvm_file_runtime_frame_return(c)==NVM_FILE_RUNTIME_CLEANUP);
    CHECK(nvm_file_runtime_frame_start(c)==NVM_FILE_RUNTIME_CLEANUP && open_attempts==before);
    uint32_t root;CHECK(nvm_file_runtime_current_root(c,&root) && root==1);
    NvmFileRuntimeReport report=finish_bad(&c,NVM_FILE_RUNTIME_CLEANUP);CHECK(report.cleanup.cleanup_failures==1);
    memset(close_error,0,sizeof close_error);close_index=0;
   }else{fret(c);ROK(nvm_file_runtime_frame_start(c));CHECK(fv(c).function==0 && fv(c).depth==1 && open_attempts==before);fpush(c,52);fret(c);frame_finish(&c,52);}
  }
 }
}
static void formal_end_and_duplicate_origin(void){
 NvmFileNominalBindings b;NvmModule *seed=fixture(false,&b);nvm_module_free(seed);FrameSpec s={.locals=2,.types={3,0},.result=-1};Body *p=&s.code;
 service(p,b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,0);uint32_t error=branch(p,OP_FILE_RESULT_BRANCH,0);
 take_result(p,0,0);one(p,OP_OWN_STORE_LOCAL,1);op(p,OP_REGION_BEGIN);op(p,OP_BORROW_LOCAL_EXCLUSIVE);u16(p,20);u16(p,1);
 one(p,OP_FILE_END_BORROW,20);op(p,OP_REGION_END);one(p,OP_OWN_MOVE_LOCAL,1);service(p,b,4,UINT16_MAX);op(p,OP_POP);fi(p,3);op(p,OP_RET);
 target(p,error);take_result(p,0,1);op(p,OP_POP);fi(p,0);op(p,OP_RET);
 NvmFileRuntime *c=frame_context(frame_module(&s,1,&b,false,-1),NVM_FILE_RUNTIME_VM);
 fservice(c,0);fstore(c,true);fbranch_ok(c);ftake(c);fstore(c,true);fb_begin(c);fb_borrow(c);fat(c,OP_FILE_END_BORROW);
 /* I deliberately replace the named origin with a real formal alias through
  * private primitives. The checked END opcode must not release that alias. */
 uint32_t original=fr(c,20),origin=fr(c,21),staged=fv(c).staging_base;
 ROK(nvm_file_runtime_end_reference(c,original));ROK(nvm_file_runtime_borrow(c,fl(c,1),origin));
 ROK(nvm_file_runtime_bind_formal(c,origin,staged,original));CHECK(nvm_file_runtime_frame_end_reference(c)==NVM_FILE_RUNTIME_BORROWED);
 CHECK(view(c,staged).formal);finish_bad(&c,NVM_FILE_RUNTIME_BORROWED);
 /* CODE preparation admits exactly one borrowed formal for CALL_REF. It
  * refuses this shape as UNRESOLVED before flow's duplicate-origin check. */
 NvmModule *m=overlap_module(&b,false);size_t at=ownership_function_offset(m,1);
 desc(m->ownership_data+at+12+8,TAG_STRUCT,2,b.layouts[0]);uint8_t params[]={TAG_STRUCT,TAG_STRUCT,TAG_INT};
 CHECK(nvm_set_function_param_types(m,1,params,3));size_t n;uint8_t *bytes=serialize(m,&n);nvm_module_free(m);
 unsigned attempts=open_attempts;c=(NvmFileRuntime *)(uintptr_t)1;
 CHECK(nvm_file_runtime_create(bytes,n,NVM_FILE_RUNTIME_VM,&c)==NVM_FILE_RUNTIME_UNRESOLVED);
 CHECK(c==(NvmFileRuntime *)(uintptr_t)1 && open_attempts==attempts);free(bytes);
}
static void return_refusals(void){
 for(unsigned which=0;which<2;which++){
  NvmFileNominalBindings b;NvmFileRuntime *c=frame_context(owner_module(&b,false),NVM_FILE_RUNTIME_VM);
  fservice(c,0);fservice(c,0);frame_call_checked(c);uint32_t retained=fl(c,1);
  if(which)ROK(nvm_file_runtime_drop(c,retained));
  /* Deliberately omit a required local drop in case0; frame transfer is not an
   * opcode evaluator, but RET must still reject the remaining real root. */
  fnxt(c);fload(c,true);uint32_t result=fs(c,0);NvmFileRuntimeStatus expected=NVM_FILE_RUNTIME_STATE;
  if(which){ROK(nvm_file_runtime_drop(c,result));ROK(nvm_file_runtime_scalar(c,result,TAG_BOOL,1));expected=NVM_FILE_RUNTIME_TYPE;}
  CHECK(nvm_file_runtime_frame_return(c)==expected);
  CHECK(which?view(c,result).type.tag==TAG_BOOL:(view(c,result).owning&&view(c,retained).owning));finish_bad(&c,expected);
 }
}
int main(void){
 CHECK(prior_file_runtime_main()==0);unsigned before=checks;
 overlap_and_aliases();owner_returns();file_and_passive_result_returns();equal_modes_depth_and_getters();frame_refusals();formal_end_and_duplicate_origin();return_refusals();initializer_frames();
#ifdef HOSTED_INSTRUMENT
 partial_generation_failures();CHECK(!tracked_live && !tracked_bytes);
#endif
 empty_host();printf("PASS %u manual private File frame transfer checks; no opcode dispatcher\n",checks-before);return 0;
}
