/* One exact serialized corpus drives actual private VM capture and separately
 * compiled native replay. I do not execute a pending public File module. */
#define FILE_CYCLIC_RUNTIME_MAIN retained_cyclic_carrier_main
#define file_runtime_tmpfile dispatch_base_tmpfile
#define file_runtime_fclose dispatch_base_fclose
#define file_runtime_fread dispatch_base_fread
#define file_runtime_fwrite dispatch_base_fwrite
#define file_runtime_fseek dispatch_base_fseek
#include "test_file_cyclic_runtime.c"
#undef FILE_CYCLIC_RUNTIME_MAIN
#undef file_runtime_tmpfile
#undef file_runtime_fclose
#undef file_runtime_fread
#undef file_runtime_fwrite
#undef file_runtime_fseek
#undef malloc
#undef calloc
#undef realloc
#undef free
#include "../../src/nanovm/file_vm_cyclic_private.h"
#include "../../src/nanoisa/nvm2c_file_cyclic_private.h"
#include "file_cyclic_dispatch_hooks.h"
#include <stdarg.h>
static char events[65536];static size_t event_bytes;
static bool no_alloc=true;static int staging_fault=-1;static size_t execution_failures;
static unsigned invocations;
#ifdef HOSTED_INSTRUMENT
extern unsigned dispatch_core_drains;
#endif
static void event(const char *format,...){va_list ap;va_start(ap,format);int n=vsnprintf(events+event_bytes,sizeof events-event_bytes,format,ap);va_end(ap);CHECK(n>=0 && (size_t)n<sizeof events-event_bytes);event_bytes+=(size_t)n;}
FILE *file_runtime_tmpfile(void){FILE *f=dispatch_base_tmpfile();event("O%u;",f!=NULL);return f;}
int file_runtime_fclose(FILE *f){int n=dispatch_base_fclose(f);event("C%d:%d;",n,n?errno:0);return n;}
size_t file_runtime_fread(void *p,size_t n,size_t count,FILE *f){size_t got=dispatch_base_fread(p,n,count,f);event("R%zu:%zu:%zu:%u;",n,count,got,got?*(unsigned char *)p:0);return got;}
size_t file_runtime_fwrite(const void *p,size_t n,size_t count,FILE *f){size_t got=dispatch_base_fwrite(p,n,count,f);event("W%zu:%zu:%zu:%u;",n,count,got,n&&count?*(const unsigned char *)p:0);return got;}
int file_runtime_fseek(FILE *f,long off,int whence){int n=dispatch_base_fseek(f,off,whence);event("S%ld:%d:%d;",off,whence,n);return n;}
static void release_wire(void *p){
#ifdef HOSTED_INSTRUMENT
 file_test_free(p);
#else
 free(p);
#endif
}
NvmFileRuntimeStatus dispatch_begin(NvmFileRuntime *c){
 NvmFileRuntimeStatus status=nvm_file_runtime_begin(c);
#ifdef HOSTED_INSTRUMENT
 if(status==NVM_FILE_RUNTIME_OK && no_alloc){execution_failures=failed_calls;allocation_budget=0;}
#else
 (void)no_alloc;(void)execution_failures;
#endif
 return status;
}
NvmFileRuntimeStatus dispatch_call(NvmFileRuntime *c){
#ifdef HOSTED_INSTRUMENT
 if(staging_fault==0)near_limit(c,fs(c,0),UINT64_MAX);
#endif
 return nvm_file_runtime_frame_call(c);
}
NvmFileRuntimeStatus dispatch_return(NvmFileRuntime *c){
#ifdef HOSTED_INSTRUMENT
 if(staging_fault==1 && fv(c).depth>1)near_limit(c,fs(c,0),UINT64_MAX);
#endif
 return nvm_file_runtime_frame_return(c);
}
NvmFileRuntimeStatus dispatch_service(NvmFileRuntime *c,uint32_t i,uint32_t r,uint32_t a,uint32_t o){
 NvmFileRuntimeStatus status=nvm_file_runtime_service(c,i,r,a,o);
#ifdef HOSTED_INSTRUMENT
 if(status==NVM_FILE_RUNTIME_OK && staging_fault==2 && view(c,o).owning)near_limit(c,o,UINT64_MAX);
#endif
 return status;
}
NvmFileCyclicExecutionReport dispatch_destroy(NvmFileRuntime **address,NvmFileRuntimeView *out){
#ifdef HOSTED_INSTRUMENT
 if(address && *address && (*address)->acquired){
  if(no_alloc){CHECK(failed_calls==execution_failures);allocation_budget=-1;}
  if((*address)->report.status==NVM_FILE_RUNTIME_OK && (*address)->complete){
   uint64_t owners=UINT64_MAX,borrows=UINT64_MAX;
   CHECK(!(*address)->frame_count && !(*address)->region_count);
   CHECK(nl_file_values_live_slots((*address)->files,&owners,&borrows)==NL_FILE_VALUE_OK && !owners && !borrows);
  }
 }
#endif
 return nvm_file_runtime_cyclic_destroy(address,out);
}
#ifdef HOSTED_INSTRUMENT
#define nvm_file_runtime_begin dispatch_begin
#define nvm_file_runtime_frame_call dispatch_call
#define nvm_file_runtime_frame_return dispatch_return
#define nvm_file_runtime_service dispatch_service
#define nvm_file_runtime_cyclic_destroy dispatch_destroy
#include "../../src/nanovm/file_vm_cyclic_private.c"
#undef nvm_file_runtime_begin
#undef nvm_file_runtime_frame_call
#undef nvm_file_runtime_frame_return
#undef nvm_file_runtime_service
#undef nvm_file_runtime_cyclic_destroy
#endif
#ifndef FILE_CYCLIC_VM_EXECUTE
#define FILE_CYCLIC_VM_EXECUTE nvm_file_vm_cyclic_execute
#endif
#ifndef FILE_CYCLIC_EMIT
#define FILE_CYCLIC_EMIT nvm2c_file_cyclic_private_emit
#endif
#ifndef FILE_CYCLIC_DISPATCH_MAIN
#define FILE_CYCLIC_DISPATCH_MAIN main
#endif
#ifdef FILE_CYCLIC_CAPTURE
static struct {uint8_t *bytes;size_t size;} captured[128];static unsigned captured_count;
static NvmFileCyclicExecutionReport execute(const uint8_t *bytes,size_t size,const NvmFileCyclicOptions *options,NvmFileRuntimeView *out){
 if(bytes && size){unsigned i=0;for(;i<captured_count;i++)if(captured[i].size==size && !memcmp(captured[i].bytes,bytes,size))break;
  if(i==captured_count){CHECK(i<128);captured[i].bytes=malloc(size);CHECK(captured[i].bytes);memcpy(captured[i].bytes,bytes,size);captured[i].size=size;captured_count++;}}
 return FILE_CYCLIC_VM_EXECUTE(bytes,size,options,out);
}
#else
NvmFileCyclicExecutionReport file_cyclic_registered(const uint8_t *,size_t,const NvmFileCyclicOptions *,NvmFileRuntimeView *);
static NvmFileCyclicExecutionReport execute(const uint8_t *bytes,size_t size,const NvmFileCyclicOptions *options,NvmFileRuntimeView *out){return file_cyclic_registered(bytes,size,options,out);}
#endif
static void print_detail(NlFileResult r){printf("%u,%d,%d,%zu,%u,%u,%u",r.status,r.host_errno,r.cleanup_errno,r.bytes,r.eof,r.consumed,r.cleanup_failed);}
static NvmFileCyclicExecutionReport run_wire(const uint8_t *wire,size_t size,uint64_t fuel,NvmFileRuntimeStatus expected,int64_t value,uint64_t instructions,bool trace){
 NvmFileRuntimeView out,old;memset(&out,0xa5,sizeof out);old=out;NvmFileCyclicOptions options={1,fuel};
 printf("I begin dispatch case%u fuel%llu expected%u\n",invocations,(unsigned long long)fuel,expected);
 event_bytes=0;events[0]=0;unsigned opens=open_attempts,closed_before=closed;
#ifdef HOSTED_INSTRUMENT
 size_t live=tracked_live,bytes=tracked_bytes;unsigned drains=dispatch_core_drains;
#endif
 NvmFileCyclicExecutionReport r=execute(wire,size,&options,&out);
 if(r.runtime.status!=expected)fprintf(stderr,"I expected status%u but got%u at%u:%u after%llu fuel\n",expected,r.runtime.status,r.runtime.function,r.runtime.instruction,(unsigned long long)r.instructions_started);
 CHECK(r.revision==1 && r.runtime.status==expected && r.instruction_limit==fuel);
 if(instructions!=UINT64_MAX)CHECK(r.instructions_started==instructions);
 CHECK(r.fuel_exhausted==(expected==NVM_FILE_RUNTIME_LIMIT && staging_fault<0));
 if(expected==NVM_FILE_RUNTIME_OK)CHECK(out.initialized && out.type.tag==TAG_INT && out.values[0]==value);
 else CHECK(!memcmp(&out,&old,sizeof out));
 empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(tracked_live==live && tracked_bytes==bytes);CHECK(dispatch_core_drains==drains+(r.runtime.acquired?1u:0u));
#endif
 if(trace){printf("TRACE %u %u %u %u %u %llu %llu %u %u %u %u %llu ",invocations++,r.revision,r.runtime.status,r.runtime.acquired,r.runtime.core_status,(unsigned long long)r.instruction_limit,(unsigned long long)r.instructions_started,r.fuel_exhausted,r.runtime.function,r.runtime.instruction,r.runtime.cleanup.execution,(unsigned long long)r.runtime.cleanup.cleanup_failures);
  print_detail(r.runtime.cleanup.first_cleanup);putchar(' ');print_detail(r.runtime.cleanup.next_cleanup);
  printf(" OUT%lld HOST%u/%u %s\n",expected==NVM_FILE_RUNTIME_OK?(long long)out.values[0]:-999LL,open_attempts-opens,closed-closed_before,events);
 }
 return r;
}
static void run_module(NvmModule *m,uint64_t fuel,NvmFileRuntimeStatus expected,int64_t value,uint64_t instructions){
 size_t size;uint8_t *wire=serialize(m,&size);nvm_module_free(m);run_wire(wire,size,fuel,expected,value,instructions,true);release_wire(wire);
}
static NvmModule *init_helper(NvmFileNominalBindings *b){
 FrameSpec s[3]={{.result=-1},{.result=-1},{.result=-3}};
 fi(&s[0].code,11);op(&s[0].code,OP_RET);fc(&s[1].code,0);op(&s[1].code,OP_RET);fc(&s[2].code,0);op(&s[2].code,OP_POP);op(&s[2].code,OP_RET);
 NvmModule *m=frame_module(s,3,b,false,2);m->header.entry_point=1;return m;
}
static void initializer_fuel(void){
 NvmFileNominalBindings b;NvmModule *m=init_helper(&b);size_t n;uint8_t *wire=serialize(m,&n);nvm_module_free(m);
 NvmFileCyclicExecutionReport r=run_wire(wire,n,9,NVM_FILE_RUNTIME_OK,11,9,true);
 CHECK(r.runtime.function==1 && r.runtime.instruction==5);
 r=run_wire(wire,n,8,NVM_FILE_RUNTIME_LIMIT,0,8,true);CHECK(r.runtime.function==1 && r.runtime.instruction==5);
 r=run_wire(wire,n,0,NVM_FILE_RUNTIME_LIMIT,0,0,true);CHECK(r.runtime.function==2 && r.runtime.instruction==0);
 release_wire(wire);
}
static NvmModule *held_loop(NvmFileNominalBindings *b,bool fail,bool endless){
 NvmModule *seed=fixture(false,b);nvm_module_free(seed);FrameSpec s={.locals=3,.types={3,0,-1},.result=-1};Body *p=&s.code;
 service(p,*b,0,UINT16_MAX);one(p,OP_OWN_STORE_LOCAL,0);uint32_t error=branch(p,OP_FILE_RESULT_BRANCH,0);
 take_result(p,0,0);one(p,OP_OWN_STORE_LOCAL,1);op(p,OP_REGION_BEGIN);op(p,OP_BORROW_LOCAL_EXCLUSIVE);u16(p,0);u16(p,1);
 fi(p,3);one(p,OP_STORE_LOCAL,2);uint32_t header=p->n;
 if(fail){op(p,OP_PUSH_BOOL);op(p,0);op(p,OP_ASSERT);}
 one(p,OP_LOAD_LOCAL,2);fi(p,0);op(p,OP_I64_GT_S);uint32_t done=branch(p,OP_JMP_FALSE,0);
 if(!endless){one(p,OP_LOAD_LOCAL,2);fi(p,1);op(p,OP_I64_SUB);one(p,OP_STORE_LOCAL,2);}
 uint32_t back=branch(p,OP_JMP,0);wr32(p->bytes+back+1,(uint32_t)((int32_t)header-(int32_t)back));target(p,done);
 one(p,OP_FILE_END_BORROW,0);op(p,OP_REGION_END);one(p,OP_FILE_DROP_LOCAL,1);fi(p,29);op(p,OP_RET);
 target(p,error);take_result(p,0,1);op(p,OP_POP);fi(p,0);op(p,OP_RET);
 return frame_module(&s,1,b,false,-1);
}
static NvmModule *dead_label(NvmFileNominalBindings *b){NvmModule *m=csimple(b,false);Body c={0};fi(&c,17);op(&c,OP_RET);op(&c,OP_NOP);op(&c,OP_RET);setbody(m,0,c);return m;}
static NvmModule *multiple_variants(NvmFileNominalBindings *b){
 FrameSpec s={.locals=1,.types={-1},.result=-1};Body *p=&s.code;
 op(p,OP_PUSH_BOOL);op(p,1);uint32_t skip=branch(p,OP_JMP_FALSE,0);
 fi(p,7);one(p,OP_STORE_LOCAL,0);target(p,skip);fi(p,17);op(p,OP_RET);
 return frame_module(&s,1,b,false,-1);
}
static void back_to(Body *p,uint32_t where){uint32_t j=branch(p,OP_JMP,0);wr32(p->bytes+j+1,(uint32_t)((int32_t)where-(int32_t)j));}
static NvmModule *nested_iterations(NvmFileNominalBindings *b,bool early){
 FrameSpec s={.locals=3,.types={-1,-1,-1},.result=-1};Body *p=&s.code;
 fi(p,0);one(p,OP_STORE_LOCAL,2);fi(p,2);one(p,OP_STORE_LOCAL,0);uint32_t outer=p->n;
 one(p,OP_LOAD_LOCAL,0);fi(p,0);op(p,OP_I64_GT_S);uint32_t done=branch(p,OP_JMP_FALSE,0);
 fi(p,2);one(p,OP_STORE_LOCAL,1);uint32_t inner=p->n;
 one(p,OP_LOAD_LOCAL,1);fi(p,0);op(p,OP_I64_GT_S);uint32_t leave=branch(p,OP_JMP_FALSE,0);
 op(p,OP_PUSH_BOOL);op(p,early);uint32_t no_return=branch(p,OP_JMP_FALSE,0);fi(p,99);op(p,OP_RET);target(p,no_return);
 one(p,OP_LOAD_LOCAL,2);fi(p,1);op(p,OP_I64_ADD);one(p,OP_STORE_LOCAL,2);
 one(p,OP_LOAD_LOCAL,1);fi(p,1);op(p,OP_I64_SUB);one(p,OP_STORE_LOCAL,1);
 one(p,OP_LOAD_LOCAL,1);fi(p,1);op(p,OP_I64_EQ);uint32_t onward=branch(p,OP_JMP_TRUE,0);op(p,OP_NOP);target(p,onward);back_to(p,inner);
 target(p,leave);one(p,OP_LOAD_LOCAL,0);fi(p,1);op(p,OP_I64_SUB);one(p,OP_STORE_LOCAL,0);back_to(p,outer);
 target(p,done);one(p,OP_LOAD_LOCAL,2);op(p,OP_RET);return frame_module(&s,1,b,false,-1);
}
static void corpus(void){
 NvmFileNominalBindings b;
 run_module(dead_label(&b),2,NVM_FILE_RUNTIME_OK,17,2);
 run_module(multiple_variants(&b),100,NVM_FILE_RUNTIME_OK,17,6);
 run_module(nested_iterations(&b,false),1000,NVM_FILE_RUNTIME_OK,4,UINT64_MAX);
 run_module(nested_iterations(&b,true),1000,NVM_FILE_RUNTIME_OK,99,UINT64_MAX);
 initializer_fuel();
 for(unsigned perm=0;perm<2;perm++)for(unsigned k=0;k<4;k++){int64_t count=k==3?258:k;uint64_t n=8+(uint64_t)count*24;
  run_module(cloop(&b,perm!=0,count),n,NVM_FILE_RUNTIME_OK,73,n);
  run_module(cloop(&b,perm!=0,count),n-1,NVM_FILE_RUNTIME_LIMIT,0,n-1);
 }
 run_module(cswap(&b),14,NVM_FILE_RUNTIME_OK,31,14);
 run_module(owner_module(&b,false),9,NVM_FILE_RUNTIME_OK,37,9);
 run_module(cnested(&b,false),20,NVM_FILE_RUNTIME_OK,42,20);
 run_module(cnested(&b,true),9,NVM_FILE_RUNTIME_LIMIT,0,9);
 run_module(held_loop(&b,false,false),45,NVM_FILE_RUNTIME_OK,29,45);
 run_module(held_loop(&b,false,true),43,NVM_FILE_RUNTIME_LIMIT,0,43);
 run_module(held_loop(&b,true,false),100,NVM_FILE_RUNTIME_ASSERT,0,11);
 deny_open=true;run_module(cloop(&b,false,3),100,NVM_FILE_RUNTIME_OK,0,13);deny_open=false;
 close_index=0;close_error[0]=EIO;close_error[1]=ENOSPC;
 run_module(cswap(&b),14,NVM_FILE_RUNTIME_CLEANUP,0,14);memset(close_error,0,sizeof close_error);close_index=0;
#ifdef HOSTED_INSTRUMENT
 for(staging_fault=0;staging_fault<3;staging_fault++)run_module(owner_module(&b,false),100,NVM_FILE_RUNTIME_LIMIT,0,UINT64_MAX);
 staging_fault=-1;
#endif
}
static void refusal_controls(void){
 NvmFileNominalBindings b;NvmModule *m=csimple(&b,false);size_t size;uint8_t *wire=serialize(m,&size);nvm_module_free(m);
 for(unsigned kind=0;kind<3;kind++){
  NvmFileCyclicOptions options={kind==0?2:1,kind==1?NVM_FILE_CYCLIC_FUEL_MAX+1:37};
  NvmFileRuntimeView out,old;memset(&out,0x5a,sizeof out);old=out;unsigned opens=open_attempts;
  NvmFileCyclicExecutionReport report=execute(wire,size,kind==2?NULL:&options,&out);
  CHECK(report.runtime.status==NVM_FILE_RUNTIME_INVALID && !report.runtime.acquired && !report.instructions_started && !report.fuel_exhausted);
  CHECK(report.instruction_limit==(kind==2?0:options.instruction_limit) && !memcmp(&out,&old,sizeof out) && open_attempts==opens);
 }
 release_wire(wire);
 m=dead_label(&b);m->code[m->code_size-2]=OP_PRINT;
 run_module(m,100,NVM_FILE_RUNTIME_UNRESOLVED,0,0);
 m=held_loop(&b,false,false);CHECK(!nvm_verify(m).ok);char error[256];CHECK(!nvm2c_emit(m,error,sizeof error));
 nvm_module_free(m);
 CHECK(nvm_file_runtime_cyclic_native_abi(1,sizeof(NvmFileCyclicOptions),sizeof(NvmFileCyclicExecutionReport),sizeof(NvmFileCyclicFrameView),sizeof(NvmFileRuntimeFrameView),sizeof(NvmFileRuntimeView)));
 CHECK(!nvm_file_runtime_cyclic_native_abi(2,sizeof(NvmFileCyclicOptions),sizeof(NvmFileCyclicExecutionReport),sizeof(NvmFileCyclicFrameView),sizeof(NvmFileRuntimeFrameView),sizeof(NvmFileRuntimeView)));
 CHECK(!nvm_file_runtime_cyclic_native_abi(1,sizeof(NvmFileCyclicOptions),sizeof(NvmFileCyclicExecutionReport),sizeof(NvmFileCyclicFrameView)+1,sizeof(NvmFileRuntimeFrameView),sizeof(NvmFileRuntimeView)));
}
static void prepare_faults(void){
#ifdef HOSTED_INSTRUMENT
 NvmFileNominalBindings b;NvmModule *m=cloop(&b,false,1);size_t size;uint8_t *wire=serialize(m,&size);nvm_module_free(m);
 no_alloc=false;size_t live=tracked_live,bytes=tracked_bytes;unsigned refusals=0;
 NvmFileCyclicOptions options={1,32};
 for(unsigned transient=0;transient<2;transient++){
  bool complete=false;
  for(unsigned prefix=0;prefix<4096;prefix++){
   printf("I begin allocation prefix%u transient%u\n",prefix,transient);
   NvmFileRuntimeView out,old;memset(&out,0xa5,sizeof out);old=out;
   allocation_budget=(int)prefix;single_failure=transient!=0;failed_calls=0;
   NvmFileCyclicExecutionReport report=execute(wire,size,&options,&out);
   allocation_budget=-1;single_failure=false;
   if(report.runtime.status==NVM_FILE_RUNTIME_OK){CHECK(out.values[0]==73);if(!failed_calls)complete=true;}
   else {CHECK(failed_calls && !memcmp(&out,&old,sizeof out));CHECK(report.runtime.status==NVM_FILE_RUNTIME_MEMORY || report.runtime.status==NVM_FILE_RUNTIME_UNRESOLVED);CHECK(!report.fuel_exhausted);refusals++;}
   CHECK(tracked_live==live && tracked_bytes==bytes);empty_host();
   NvmFileCyclicExecutionReport recovery=execute(wire,size,&options,&out);CHECK(recovery.runtime.status==NVM_FILE_RUNTIME_OK && out.values[0]==73);
   CHECK(tracked_live==live && tracked_bytes==bytes);empty_host();if(complete)break;
  }
  CHECK(complete);
 }
 CHECK(refusals);printf("PASS cyclic dispatch preparation allocation refusals=%u\n",refusals);no_alloc=true;release_wire(wire);
#endif
}
#ifdef FILE_CYCLIC_CAPTURE
int file_cyclic_buffer_checks(void);
static void emit_cases(const char *directory){
 CHECK(file_cyclic_buffer_checks()==0);
#ifdef HOSTED_INSTRUMENT
 size_t live=tracked_live,bytes=tracked_bytes;unsigned refusals=0;
 for(unsigned transient=0;transient<2;transient++){
  bool complete=false;
  for(unsigned prefix=0;prefix<4096;prefix++){
   printf("I begin allocation prefix%u transient%u\n",prefix,transient);
   char *out=(char *)(uintptr_t)1;char error[256];unsigned opens=open_attempts;
   allocation_budget=(int)prefix;single_failure=transient!=0;failed_calls=0;
   NvmFileRuntimeStatus status=FILE_CYCLIC_EMIT(captured[0].bytes,captured[0].size,&out,error,sizeof error);
   allocation_budget=-1;single_failure=false;CHECK(open_attempts==opens);
   if(status==NVM_FILE_RUNTIME_OK){CHECK(out!=(char *)(uintptr_t)1);file_test_free(out);if(!failed_calls)complete=true;}
   else {CHECK(failed_calls && out==(char *)(uintptr_t)1);CHECK(status==NVM_FILE_RUNTIME_MEMORY || status==NVM_FILE_RUNTIME_UNRESOLVED);refusals++;}
   CHECK(tracked_live==live && tracked_bytes==bytes);
   char *recovered=(char *)(uintptr_t)1;size_t failures=failed_calls;
   ROK(FILE_CYCLIC_EMIT(captured[0].bytes,captured[0].size,&recovered,error,sizeof error));
   CHECK(recovered && recovered!=(char *)(uintptr_t)1 && recovered[0] && !error[0]);
   CHECK(open_attempts==opens && failed_calls==failures);file_test_free(recovered);
   CHECK(tracked_live==live && tracked_bytes==bytes);if(complete)break;
  }
  CHECK(complete);
 }
 CHECK(refusals);printf("PASS cyclic emitter allocation refusals=%u\n",refusals);
#endif
 char path[4096];CHECK(snprintf(path,sizeof path,"%s/cases.tsv",directory)>0);FILE *manifest=fopen(path,"w");CHECK(manifest);
 for(unsigned i=0;i<captured_count;i++){
  CHECK(snprintf(path,sizeof path,"%s/case-%03u.nvm",directory,i)>0);FILE *file=fopen(path,"wb");CHECK(file);
  CHECK(fwrite(captured[i].bytes,1,captured[i].size,file)==captured[i].size && !fclose(file));
  char *text=(char *)(uintptr_t)1;char error[256];unsigned opens=open_attempts;
  NvmFileRuntimeStatus status=FILE_CYCLIC_EMIT(captured[i].bytes,captured[i].size,&text,error,sizeof error);CHECK(open_attempts==opens);
  if(status==NVM_FILE_RUNTIME_OK){CHECK(snprintf(path,sizeof path,"%s/case-%03u.c",directory,i)>0);file=fopen(path,"w");CHECK(file);size_t n=strlen(text);CHECK(fwrite(text,1,n,file)==n && !fclose(file));release_wire(text);}
  else CHECK(text==(char *)(uintptr_t)1 && status==NVM_FILE_RUNTIME_UNRESOLVED);
  CHECK(fprintf(manifest,"%u\t%u\t%zu\n",i,status,captured[i].size)>0);free(captured[i].bytes);
 }
 CHECK(!fclose(manifest));CHECK(captured_count>=15);
 printf("PASS cyclic VM capture: %u exact modules\n",captured_count);
}
int FILE_CYCLIC_DISPATCH_MAIN(int argc,char **argv){CHECK(argc==2);CHECK(!setvbuf(stdout,NULL,_IONBF,0));
 (void)retained_cyclic_carrier_main;corpus();refusal_controls();prepare_faults();emit_cases(argv[1]);empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
 return 0;
}
#else
int FILE_CYCLIC_DISPATCH_MAIN(void){CHECK(!setvbuf(stdout,NULL,_IONBF,0));(void)retained_cyclic_carrier_main;
 corpus();refusal_controls();prepare_faults();empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
 printf("PASS cyclic native replay: %u matching report/host traces\n",invocations);return 0;
}
#endif
