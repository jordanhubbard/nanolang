/* My original VM assertions are reused without changing their expected values.
 * Capture executes that VM corpus; replay selects exact serialized cases and
 * calls separately compiled generated C, never a generic instruction handler. */
#include "../../src/nanoisa/nvm2c_file_private.h"
static NvmFileRuntimeReport native_dispatch(const uint8_t *,size_t,NvmFileRuntimeView *);
#define FILE_VM_EXECUTE native_dispatch
#define FILE_VM_MAIN original_vm_corpus_main
#ifndef FILE_NATIVE_CAPTURE
#define FILE_VM_TARGET_LABEL "actual generated-native File dispatch"
#endif
#include "test_file_private_vm.c"
#undef FILE_VM_EXECUTE
#undef FILE_VM_MAIN
/* Capture bookkeeping belongs to libc, outside the project allocation ledger. */
#undef malloc
#undef calloc
#undef realloc
#undef free
#include "file_native_hooks.h"
#ifdef HOSTED_INSTRUMENT
NvmFileRuntimeStatus file_native_begin(NvmFileRuntime *c){return vm_begin_hook(c);}
NvmFileRuntimeStatus file_native_call(NvmFileRuntime *c){return vm_call_hook(c);}
NvmFileRuntimeStatus file_native_return(NvmFileRuntime *c){return vm_return_hook(c);}
NvmFileRuntimeStatus file_native_service(NvmFileRuntime *c,uint32_t i,uint32_t r,uint32_t a,uint32_t o){return vm_service_hook(c,i,r,a,o);}
#else
NvmFileRuntimeStatus file_native_begin(NvmFileRuntime *c){return nvm_file_runtime_begin(c);}
NvmFileRuntimeStatus file_native_call(NvmFileRuntime *c){return nvm_file_runtime_frame_call(c);}
NvmFileRuntimeStatus file_native_return(NvmFileRuntime *c){return nvm_file_runtime_frame_return(c);}
NvmFileRuntimeStatus file_native_service(NvmFileRuntime *c,uint32_t i,uint32_t r,uint32_t a,uint32_t o){return nvm_file_runtime_service(c,i,r,a,o);}
#endif
#ifndef FILE_NATIVE_MAIN
#define FILE_NATIVE_MAIN main
#endif
#ifndef FILE_NATIVE_VM_EXECUTE
#define FILE_NATIVE_VM_EXECUTE nvm_file_vm_execute
#endif
#ifndef FILE_NATIVE_EMIT
#define FILE_NATIVE_EMIT nvm2c_file_private_emit
#endif
#ifdef FILE_NATIVE_CAPTURE
static struct {uint8_t *bytes;size_t size;} recorded[512];
static unsigned record_count;
static NvmFileRuntimeReport native_dispatch(const uint8_t *bytes,size_t size,NvmFileRuntimeView *out){
 if(bytes && size){
  unsigned i=0;for(;i<record_count;i++)if(recorded[i].size==size && !memcmp(recorded[i].bytes,bytes,size))break;
  if(i==record_count){CHECK(i<512);recorded[i].bytes=malloc(size);CHECK(recorded[i].bytes);memcpy(recorded[i].bytes,bytes,size);recorded[i].size=size;record_count++;}
 }
 return FILE_NATIVE_VM_EXECUTE(bytes,size,out);
}
int file_native_buffer_checks(void);
static void emission_faults(void){
 CHECK(file_native_buffer_checks()==0);
#ifdef HOSTED_INSTRUMENT
 CHECK(record_count);size_t baseline=tracked_bytes,live=tracked_live;unsigned refusals=0;
 for(unsigned transient=0;transient<2;transient++){
  bool finished=false;
  for(unsigned prefix=0;prefix<4096;prefix++){
   char *out=(char *)(uintptr_t)1;char error[256];unsigned opens=open_attempts;size_t failures=failed_calls;
   allocation_budget=(int)prefix;single_failure=transient!=0;
   NvmFileRuntimeStatus status=FILE_NATIVE_EMIT(recorded[0].bytes,recorded[0].size,&out,error,sizeof error);
   allocation_budget=-1;single_failure=false;CHECK(open_attempts==opens);
   if(status==NVM_FILE_RUNTIME_OK){CHECK(out!=(char *)(uintptr_t)1);file_test_free(out);if(failed_calls==failures)finished=true;}
   else {CHECK(out==(char *)(uintptr_t)1);CHECK(status==NVM_FILE_RUNTIME_MEMORY || status==NVM_FILE_RUNTIME_UNRESOLVED);refusals++;}
   CHECK(tracked_bytes==baseline && tracked_live==live);if(finished)break;
  }
  CHECK(finished);
 }
 CHECK(refusals>0);printf("PASS native emission allocation refusals=%u\n",refusals);
#endif
 CHECK(!nvm_file_runtime_native_abi(NVM_FILE_NATIVE_ABI+1,sizeof(NvmFileRuntimeView),sizeof(NvmFileRuntimeFrameView),sizeof(NvmFileRuntimeReport)));
 CHECK(!nvm_file_runtime_native_abi(NVM_FILE_NATIVE_ABI,sizeof(NvmFileRuntimeView)+1,sizeof(NvmFileRuntimeFrameView),sizeof(NvmFileRuntimeReport)));
 CHECK(nvm_file_runtime_native_abi(NVM_FILE_NATIVE_ABI,sizeof(NvmFileRuntimeView),sizeof(NvmFileRuntimeFrameView),sizeof(NvmFileRuntimeReport)));
}
static void emit_cases(const char *directory){
 emission_faults();
 char path[4096];CHECK(snprintf(path,sizeof path,"%s/cases.tsv",directory)>0);FILE *manifest=fopen(path,"w");CHECK(manifest);
 unsigned positive=0,negative=0;
 for(unsigned i=0;i<record_count;i++){
  CHECK(snprintf(path,sizeof path,"%s/case-%03u.nvm",directory,i)>0);FILE *wire=fopen(path,"wb");CHECK(wire);
  CHECK(fwrite(recorded[i].bytes,1,recorded[i].size,wire)==recorded[i].size);CHECK(!fclose(wire));
  char *text=(char *)(uintptr_t)1;char error[256];unsigned opens=open_attempts;
  NvmFileRuntimeStatus status=FILE_NATIVE_EMIT(recorded[i].bytes,recorded[i].size,&text,error,sizeof error);
  CHECK(open_attempts==opens);
  if(status==NVM_FILE_RUNTIME_OK){
   CHECK(text && text!=(char *)(uintptr_t)1);CHECK(snprintf(path,sizeof path,"%s/case-%03u.c",directory,i)>0);
   FILE *source=fopen(path,"w");CHECK(source);size_t n=strlen(text);CHECK(fwrite(text,1,n,source)==n);CHECK(!fclose(source));
#ifdef HOSTED_INSTRUMENT
   file_test_free(text);
#else
   free(text);
#endif
   positive++;
  }else{CHECK(text==(char *)(uintptr_t)1 && (status==NVM_FILE_RUNTIME_INVALID || status==NVM_FILE_RUNTIME_UNRESOLVED));negative++;}
  CHECK(fprintf(manifest,"%u\t%u\t%zu\n",i,(unsigned)status,recorded[i].size)>0);
  free(recorded[i].bytes);recorded[i].bytes=NULL;
 }
 CHECK(!fclose(manifest));CHECK(positive>30 && negative>=2);empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
 printf("PASS native capture: %u exact modules, %u emitted, %u refused before host\n",record_count,positive,negative);
}
int FILE_NATIVE_MAIN(int argc,char **argv){CHECK(argc==2);CHECK(original_vm_corpus_main()==0);emit_cases(argv[1]);return 0;}
#else
NvmFileRuntimeReport file_native_registered(const uint8_t *,size_t,NvmFileRuntimeView *);
static unsigned native_calls;
static NvmFileRuntimeReport native_dispatch(const uint8_t *bytes,size_t size,NvmFileRuntimeView *out){
 native_calls++;return file_native_registered(bytes,size,out);
}
int FILE_NATIVE_MAIN(void){
 CHECK(original_vm_corpus_main()==0);CHECK(native_calls>30);empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
 printf("PASS private generated-native replay: %u calls, unchanged original corpus assertions\n",native_calls);return 0;
}
#endif
