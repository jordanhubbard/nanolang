/* I manually drive carrier operations. This fixture is not a shipped dispatcher. */
#define FILE_CYCLIC_RUNTIME_MAIN prior_cyclic_runtime_main
#include "test_file_cyclic_runtime.c"
#undef FILE_CYCLIC_RUNTIME_MAIN
#include "../../src/nanoisa/file_indirect_runtime.h"
static void iref(Body *b,unsigned target){op(b,OP_FUNCREF);u32(b,target);}
static void icall(Body *b){op(b,OP_CALL_INDIRECT);u16(b,1);u16(b,1);}
static NvmModule *imodule(bool other,unsigned owner,bool init,bool permute){
 FrameSpec spec[4]={0};NvmFileNominalBindings bindings;
 spec[0]=(FrameSpec){.locals=2,.types={-4,3},.result=-1};
 for(unsigned f=1;f<3;f++)spec[f]=(FrameSpec){.parameters=1,.locals=1,.types={owner?(owner==2?0:3):-1},.result=owner?(owner==2?0:3):-1};
 spec[3]=(FrameSpec){.result=-3};op(&spec[3].code,OP_RET);
 /* I collect both candidates but execute either branch. */
 Body *b=&spec[0].code;op(b,OP_PUSH_BOOL);op(b,!other);uint32_t alternate=branch(b,OP_JMP_FALSE,0);
 iref(b,1);uint32_t joined=branch(b,OP_JMP,0);target(b,alternate);iref(b,2);target(b,joined);one(b,OP_STORE_LOCAL,0);
 if(!owner)fi(b,41);
 /* I fill the real import after creating the nominal catalog. */
 if(owner){op(b,OP_FILE_SERVICE);u32(b,0);u16(b,UINT16_MAX);}
 uint32_t failed=UINT32_MAX;
 if(owner==2){one(b,OP_OWN_STORE_LOCAL,1);failed=branch(b,OP_FILE_RESULT_BRANCH,1);take_result(b,1,0);}
 one(b,OP_LOAD_LOCAL,0);icall(b);
 if(owner){op(b,OP_FILE_DROP_STACK);fi(b,9);}op(b,OP_RET);
 if(owner==2){target(b,failed);take_result(b,1,1);op(b,OP_POP);fi(b,9);op(b,OP_RET);}
 for(unsigned f=1;f<3;f++){
  if(owner || f==1)one(&spec[f].code,owner?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,0);
  else fi(&spec[f].code,42);
  op(&spec[f].code,OP_RET);
 }
 NvmModule *m=frame_module(spec,init?4:3,&bindings,permute,init?3:-1);
 if(owner){
  for(unsigned pc=m->functions[0].code_offset;pc<m->functions[0].code_offset+m->functions[0].code_length;){
   DecodedInstruction d;CHECK(isa_decode(m->code+pc,m->code_size-pc,&d));
   if(d.opcode==OP_FILE_SERVICE)wr32(m->code+pc+1,bindings.imports[0]);
   pc+=d.byte_length;
  }
 }
 return m;
}
static NvmFileRuntime *icreate(NvmModule *m,unsigned mode,uint64_t fuel){
 size_t size;uint8_t *bytes=serialize(m,&size);NvmFileRuntime *c=NULL;NvmFileIndirectOptions options={1,fuel};
 unsigned opened=open_attempts;ROK(nvm_file_runtime_indirect_create(bytes,size,(NvmFileRuntimeMode)mode,&options,&c));
 CHECK(open_attempts==opened && nvm_file_runtime_indirect_plan(c));
 CHECK(!nvm_file_runtime_plan(c) && !nvm_file_runtime_cyclic_plan(c));
 CHECK(nvm_file_runtime_cyclic_enter(c)==NVM_FILE_RUNTIME_INVALID);
 CHECK(nvm_file_runtime_cyclic_destroy(&c,NULL).runtime.status==NVM_FILE_RUNTIME_INVALID && c);
 CHECK(nvm_file_runtime_destroy(&c,NULL).status==NVM_FILE_RUNTIME_STATE && c);
 memset(bytes,0,size);free(bytes);nvm_module_free(m);
 ROK(nvm_file_runtime_begin(c));ROK(nvm_file_runtime_frame_start(c));return c;
}
static NvmFileRuntimeStatus istep(NvmFileRuntime *c,NvmFileIndirectFrameView f,NvmFileCodeInstruction in){
 uint32_t src,dst;uint8_t edge=0;NvmFileRuntimeStatus s=NVM_FILE_RUNTIME_OK;
 switch(in.decoded.opcode){
 case OP_PUSH_I64:case OP_PUSH_BOOL:
  dst=fout(c,f.frame.stack_count);s=nvm_file_runtime_scalar(c,dst,in.decoded.opcode==OP_PUSH_I64?TAG_INT:TAG_BOOL,
   in.decoded.opcode==OP_PUSH_I64?in.decoded.operands[0].i64:in.decoded.operands[0].u8);break;
 case OP_FUNCREF:s=nvm_file_runtime_indirect_funcref(c,fout(c,f.frame.stack_count));break;
 case OP_STORE_LOCAL:case OP_OWN_STORE_LOCAL:s=nvm_file_runtime_frame_store(c);break;
 case OP_LOAD_LOCAL:case OP_OWN_MOVE_LOCAL:
  src=fl(c,in.decoded.operands[0].u16);dst=fout(c,f.frame.stack_count);
  s=in.decoded.opcode==OP_LOAD_LOCAL?nvm_file_runtime_copy(c,src,dst):nvm_file_runtime_move(c,src,dst);break;
 case OP_JMP:break;
 case OP_JMP_FALSE:
  src=fs(c,f.frame.stack_count-1);edge=!view(c,src).values[0];s=nvm_file_runtime_drop(c,src);break;
 case OP_CALL_INDIRECT:return nvm_file_runtime_frame_call(c);
 case OP_RET:return nvm_file_runtime_frame_return(c);
 case OP_FILE_SERVICE:
  CHECK(in.catalog_ordinal==0);CHECK(nvm_file_runtime_frame_scratch(c,&dst));
  s=nvm_file_runtime_service(c,in.decoded.operands[0].u32,NS,NS,dst);
  if(s==NVM_FILE_RUNTIME_OK)s=nvm_file_runtime_move(c,dst,fout(c,f.frame.stack_count));
  break;
 case OP_FILE_RESULT_BRANCH:{
  NvmFileFlowArm arm; s=nvm_file_runtime_result_arm(c,fl(c,in.decoded.operands[0].u16),&arm);
  edge=arm==NVM_FILE_FLOW_ARM_ERROR;break;
 }
 case OP_FILE_RESULT_TAKE:
  s=nvm_file_runtime_take(c,fl(c,in.decoded.operands[0].u16),in.decoded.operands[1].u8?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK,fout(c,f.frame.stack_count));break;
 case OP_POP:case OP_FILE_DROP_STACK:s=nvm_file_runtime_drop(c,fs(c,f.frame.stack_count-1));break;
 default:CHECK(false);
 }
 return s==NVM_FILE_RUNTIME_OK?nvm_file_runtime_frame_next(c,edge):s;
}
static NvmFileIndirectExecutionReport irun(unsigned mode,bool other,unsigned owner,bool init,bool permute,uint64_t fuel){
 NvmFileRuntime *c=icreate(imodule(other,owner,init,permute),mode,fuel);NvmFileRuntimeStatus status=NVM_FILE_RUNTIME_OK;
#ifdef HOSTED_INSTRUMENT
 size_t failed_before=failed_calls;allocation_budget=0;
#endif
 while(status==NVM_FILE_RUNTIME_OK){
  NvmFileIndirectFrameView f;NvmFileCodeInstruction in;
  CHECK(nvm_file_runtime_indirect_frame_view(c,&f));
  CHECK(nvm_file_indirect_hosted_instruction(nvm_file_runtime_indirect_plan(c),f.frame.function,(uint16_t)f.frame.instruction,&in));
  status=nvm_file_runtime_indirect_enter(c);if(status!=NVM_FILE_RUNTIME_OK)break;
  status=istep(c,f,in);
  if(status==NVM_FILE_RUNTIME_OK && in.decoded.opcode==OP_CALL_INDIRECT)CHECK(fv(c).function==(other?2u:1u));
  if(status==NVM_FILE_RUNTIME_OK && in.decoded.opcode==OP_RET && f.frame.depth==1){
   uint32_t root;if(!nvm_file_runtime_current_root(c,&root))break;status=nvm_file_runtime_frame_start(c);
  }
 }
#ifdef HOSTED_INSTRUMENT
 CHECK(failed_calls==failed_before);allocation_budget=-1;
#endif
 NvmFileRuntimeView out,old;memset(&out,0xa5,sizeof out);old=out;
 NvmFileIndirectExecutionReport report=nvm_file_runtime_indirect_finish(c,&out);
 CHECK(report.runtime.status==status);
 if(status==NVM_FILE_RUNTIME_OK)CHECK(out.type.tag==TAG_INT && out.values[0]==(owner?9:other?42:41));
 else CHECK(status==NVM_FILE_RUNTIME_LIMIT && report.fuel_exhausted && !memcmp(&out,&old,sizeof out));
 NvmFileIndirectExecutionReport again=nvm_file_runtime_indirect_destroy(&c,&out);
 CHECK(!c && again.runtime.status==report.runtime.status && again.instructions_started==report.instructions_started && !report.runtime.cleanup.cleanup_failures);
 empty_host();return report;
}
static void indirect_options(void){
 NvmModule *m=imodule(false,0,false,false);size_t size;uint8_t *wire=serialize(m,&size);nvm_module_free(m);
 NvmFileIndirectOptions options={1,100};NvmFileRuntime *c=(void *)(uintptr_t)1;unsigned attempts=open_attempts;
 CHECK(nvm_file_runtime_indirect_create(wire,size,NVM_FILE_RUNTIME_VM,NULL,&c)==NVM_FILE_RUNTIME_INVALID && c==(void *)(uintptr_t)1);
 options.revision=0;CHECK(nvm_file_runtime_indirect_create(wire,size,NVM_FILE_RUNTIME_VM,&options,&c)==NVM_FILE_RUNTIME_INVALID && c==(void *)(uintptr_t)1);
 options.revision=1;options.instruction_limit=NVM_FILE_INDIRECT_FUEL_MAX+1;
 CHECK(nvm_file_runtime_indirect_create(wire,size,NVM_FILE_RUNTIME_VM,&options,&c)==NVM_FILE_RUNTIME_INVALID && c==(void *)(uintptr_t)1);
 options.instruction_limit=100;
 CHECK(nvm_file_runtime_indirect_create(wire,size,(NvmFileRuntimeMode)2,&options,&c)==NVM_FILE_RUNTIME_INVALID && c==(void *)(uintptr_t)1);
 CHECK(nvm_file_runtime_indirect_create(wire,size,NVM_FILE_RUNTIME_VM,&options,NULL)==NVM_FILE_RUNTIME_INVALID);
 CHECK(attempts==open_attempts);free(wire);
 NvmFileNominalBindings b;NvmModule *old=bodymodule(&b,false);setbody(old,0,lifecycle_code(b));
 c=ccreate(old,NVM_FILE_RUNTIME_VM,100,false);NvmFileRuntime *same=c;
 CHECK(!nvm_file_runtime_indirect_plan(c) && nvm_file_runtime_indirect_enter(c)==NVM_FILE_RUNTIME_INVALID);
 CHECK(nvm_file_runtime_indirect_destroy(&c,NULL).runtime.status==NVM_FILE_RUNTIME_INVALID && c==same);
 CHECK(nvm_file_runtime_cyclic_destroy(&c,NULL).runtime.status==NVM_FILE_RUNTIME_STATE && !c);
}
static void indirect_matrix(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned other=0;other<2;other++)for(unsigned owner=0;owner<3;owner++)for(unsigned init=0;init<2;init++)for(unsigned permute=0;permute<2;permute++){
  NvmFileIndirectExecutionReport r=irun(mode,other,owner,init,permute,100);
  CHECK(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started>0 && r.instructions_started<100);
  CHECK(irun(mode,other,owner,init,permute,r.instructions_started).runtime.status==NVM_FILE_RUNTIME_OK);
  for(uint64_t fuel=0;fuel<r.instructions_started;fuel++){
   NvmFileIndirectExecutionReport limited=irun(mode,other,owner,init,permute,fuel);
   CHECK(limited.runtime.status==NVM_FILE_RUNTIME_LIMIT && limited.instructions_started==fuel);
  }
 }
}
#ifdef HOSTED_INSTRUMENT
static void indirect_forged(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned fault=0;fault<4;fault++){
  NvmFileRuntime *c=icreate(imodule(false,true,false,false),mode,100);
  for(;;){
   NvmFileIndirectFrameView f;NvmFileCodeInstruction in;CHECK(nvm_file_runtime_indirect_frame_view(c,&f));
   CHECK(nvm_file_indirect_hosted_instruction(c->indirect_plan,f.frame.function,(uint16_t)f.frame.instruction,&in));
   ROK(nvm_file_runtime_indirect_enter(c));
   if(in.decoded.opcode==OP_CALL_INDIRECT){
    uint32_t callable=fs(c,1),argument=fs(c,0);NlFileValue owner=c->values[argument].owner;
    if(fault==0)c->values[callable].callable_target=0;
    if(fault==1)c->values[callable].callable_plan=NULL;
    if(fault==2)c->values[callable].callable_target=64;
    if(fault<3){
     CHECK(nvm_file_runtime_frame_call(c)==NVM_FILE_RUNTIME_STATE);
     CHECK(c->frame_count==1 && c->values[argument].view.owning && frc_identity(owner,c->values[argument].owner));
    }else{
     ROK(nvm_file_runtime_frame_call(c));c->frames[0].selected_target=2;
     CHECK(nvm_file_runtime_indirect_enter(c)==NVM_FILE_RUNTIME_STATE);
    }
    break;
   }
   ROK(istep(c,f,in));
  }
  if(fault==2){close_index=0;close_error[0]=EIO;}
  NvmFileIndirectExecutionReport r=nvm_file_runtime_indirect_destroy(&c,NULL);
  CHECK(!c && r.runtime.status==NVM_FILE_RUNTIME_STATE && r.runtime.cleanup.cleanup_failures==(fault==2?1u:0u));
  memset(close_error,0,sizeof close_error);close_index=0;empty_host();
 }
}
static void indirect_allocations(void){
 NvmModule *m=imodule(false,true,true,false);size_t n;uint8_t *wire=serialize(m,&n);nvm_module_free(m);
 size_t baseline=tracked_live,bytes=tracked_bytes;NvmFileIndirectOptions options={1,100};
 for(unsigned transient=0;transient<2;transient++){
  NvmFileRuntime *measure=NULL;allocation_budget=1000000;
  ROK(nvm_file_runtime_indirect_create(wire,n,NVM_FILE_RUNTIME_VM,&options,&measure));ROK(nvm_file_runtime_begin(measure));
  int calls=1000000-allocation_budget;allocation_budget=-1;CHECK(calls>0 && calls<2048);
  CHECK(nvm_file_runtime_indirect_destroy(&measure,NULL).runtime.status==NVM_FILE_RUNTIME_STATE);
  for(int budget=0;budget<=calls;budget++){
   NvmFileRuntime *c=(void *)(uintptr_t)1;allocation_budget=budget;single_failure=transient!=0;failed_calls=0;
   NvmFileRuntimeStatus status=nvm_file_runtime_indirect_create(wire,n,NVM_FILE_RUNTIME_VM,&options,&c);
   if(status!=NVM_FILE_RUNTIME_OK)CHECK(c==(void *)(uintptr_t)1);
   else{
    status=nvm_file_runtime_begin(c);allocation_budget=-1;single_failure=false;
    NvmFileIndirectExecutionReport r=nvm_file_runtime_indirect_destroy(&c,NULL);
    CHECK(!c && r.runtime.status==(status==NVM_FILE_RUNTIME_OK?NVM_FILE_RUNTIME_STATE:status));
   }
   CHECK(budget<calls?failed_calls>0:failed_calls==0);
   allocation_budget=-1;single_failure=false;CHECK(tracked_live==baseline && tracked_bytes==bytes);empty_host();
   NvmFileRuntime *fresh=NULL;ROK(nvm_file_runtime_indirect_create(wire,n,NVM_FILE_RUNTIME_VM,&options,&fresh));
   ROK(nvm_file_runtime_begin(fresh));CHECK(nvm_file_runtime_indirect_destroy(&fresh,NULL).runtime.status==NVM_FILE_RUNTIME_STATE && !fresh);
   CHECK(tracked_live==baseline && tracked_bytes==bytes);
  }
  printf("I checked %d indirect create/begin allocation sites in %s failure mode\n",calls,transient?"single":"persistent");
 }
 free(wire);
}

#endif
#ifndef FILE_INDIRECT_RUNTIME_MAIN
#define FILE_INDIRECT_RUNTIME_MAIN main
#endif
int FILE_INDIRECT_RUNTIME_MAIN(void){
 CHECK(prior_cyclic_runtime_main()==0);unsigned before=checks;
 indirect_options();indirect_matrix();
#ifdef HOSTED_INSTRUMENT
 indirect_forged();indirect_allocations();CHECK(!tracked_live && !tracked_bytes);
#endif
 empty_host();printf("PASS %u private indirect carrier checks; no shipped opcode dispatcher\n",checks-before);return 0;
}
