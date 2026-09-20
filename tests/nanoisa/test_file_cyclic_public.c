/* I retain the complete private corpus and inspect the actual public scalar
 * before rebuilding the passive view expected by its unchanged assertions. */
#include "../../src/nanoisa/file_cyclic_public.h"
static NvmFileHostGrant *cyclic_grant;
static NvmFileCyclicExecutionReport cyclic_public_bridge(const uint8_t *,size_t,const NvmFileCyclicOptions *,NvmFileRuntimeView *);
static NvmFileRuntimeStatus cyclic_public_emit(const uint8_t *,size_t,char **,char *,size_t);
static void cyclic_nested(void);
#define FILE_RUNTIME_NESTED_EXTRA cyclic_nested
#define FILE_CYCLIC_VM_EXECUTE cyclic_public_bridge
#define FILE_CYCLIC_EMIT cyclic_public_emit
#define FILE_CYCLIC_DISPATCH_MAIN cyclic_prior_main
#include "test_file_cyclic_dispatch.c"
#undef FILE_CYCLIC_DISPATCH_MAIN
#undef FILE_RUNTIME_NESTED_EXTRA
#include "../../src/nanoisa/file_cyclic_public_internal.h"
#include "../../src/nanoisa/file_cyclic_native_public.h"
static bool cyclic_reentry;
static unsigned cyclic_nested_count,cyclic_public_calls;
NvmFileHostGrant *file_cyclic_public_test_grant(void){return cyclic_grant;}
static NvmFileCyclicExecutionReport cyclic_public_bridge(const uint8_t *bytes,size_t n,
 const NvmFileCyclicOptions *options,NvmFileRuntimeView *out){
 NvmFileScalar value,before;memset(&value,0xa5,sizeof value);before=value;
 NvmFileCyclicExecutionReport r=nvm_file_execute_cyclic_bytes(cyclic_grant,bytes,n,options,out?&value:NULL);
 cyclic_public_calls++;
 if(r.runtime.status==NVM_FILE_RUNTIME_OK){
  CHECK(out && (value.tag==TAG_INT || value.tag==TAG_BOOL));
  NvmFileRuntimeView view={0};view.initialized=true;view.fields=1;
  view.type.tag=value.tag;view.type.category=NVM_FILE_CATEGORY_UNKNOWN;
  view.type.global_index=view.type.catalog_ordinal=NVM_V2_NO_INDEX;view.values[0]=value.value;*out=view;
 }else CHECK(!memcmp(&value,&before,sizeof value));
 return r;
}
static NvmFileRuntimeStatus cyclic_public_emit(const uint8_t *bytes,size_t n,char **out,char *error,size_t cap){
 return nvm2c_emit_file_cyclic_bytes(bytes,n,"case",out,error,cap);
}
#ifndef FILE_CYCLIC_CAPTURE
void file_cyclic_public_native_busy(void);
#endif
static void cyclic_busy(void){
 const void *invalid=(const void *)(uintptr_t)1;
 NvmFileScalar out,before;memset(&out,0xa5,sizeof out);before=out;
 NvmFileCyclicExecutionReport r=nvm_file_execute_cyclic_bytes((NvmFileHostGrant *)invalid,
  invalid,SIZE_MAX,invalid,&out);
 CHECK(r.revision==1 && r.runtime.status==NVM_FILE_RUNTIME_BUSY && !r.runtime.acquired &&
  r.runtime.core_status==0 && r.runtime.function==UINT32_MAX && r.runtime.instruction==UINT32_MAX &&
  !r.instruction_limit && !r.instructions_started && !r.fuel_exhausted &&
  !r.runtime.cleanup.cleanup_failures && !memcmp(&out,&before,sizeof out));
 /* Both APIs also leave invalid output addresses unread while the gate is held. */
 CHECK(nvm_file_execute_cyclic_bytes((NvmFileHostGrant *)invalid,invalid,SIZE_MAX,invalid,(NvmFileScalar *)invalid).runtime.status==NVM_FILE_RUNTIME_BUSY);
 CHECK(nvm_file_execute_bytes((NvmFileHostGrant *)invalid,invalid,SIZE_MAX,(NvmFileScalar *)invalid).status==NVM_FILE_RUNTIME_BUSY);
 char *text=(char *)(uintptr_t)1;char diagnostic[8]="held";
 CHECK(nvm2c_emit_file_cyclic_bytes(invalid,SIZE_MAX,invalid,&text,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_BUSY);
 CHECK(text==(char *)(uintptr_t)1 && !strcmp(diagnostic,"held"));
 CHECK(nvm2c_emit_file_cyclic_bytes(invalid,SIZE_MAX,invalid,(char **)invalid,(char *)invalid,SIZE_MAX)==NVM_FILE_RUNTIME_BUSY);
 CHECK(nvm2c_emit_file_bytes(invalid,SIZE_MAX,invalid,(char **)invalid,(char *)invalid,SIZE_MAX)==NVM_FILE_RUNTIME_BUSY);
 CHECK(nvm_file_host_grant_revoke(cyclic_grant)==NVM_FILE_HOST_BUSY);
 NvmFileHostGrant *copy=cyclic_grant;CHECK(nvm_file_host_grant_destroy(&copy)==NVM_FILE_HOST_BUSY && copy==cyclic_grant);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);
#ifndef FILE_CYCLIC_CAPTURE
 file_cyclic_public_native_busy();
#endif
}
static void cyclic_nested(void){if(cyclic_reentry){cyclic_busy();cyclic_nested_count++;}}
static void public_early(NvmFileCyclicExecutionReport r,NvmFileRuntimeStatus expected,uint64_t fuel){
 CHECK(r.revision==1 && r.runtime.status==expected && !r.runtime.acquired &&
  r.runtime.function==UINT32_MAX && r.runtime.instruction==UINT32_MAX &&
  r.instruction_limit==fuel && !r.instructions_started && !r.fuel_exhausted &&
  !r.runtime.cleanup.cleanup_failures);
}
static void public_boundaries(const char *directory){
 NvmFileNominalBindings b;NvmModule *m=cloop(&b,false,1);size_t n;
 uint8_t *wire=serialize(m,&n);nvm_module_free(m);unsigned opens=open_attempts;
 NvmFileScalar out,before;memset(&out,0xa5,sizeof out);before=out;
 NvmFileCyclicOptions options={1,32};
 public_early(nvm_file_execute_cyclic_bytes(NULL,wire,n,&options,&out),NVM_FILE_RUNTIME_INVALID,32);
 CHECK(!memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);cyclic_busy();nvm_file_host_leave();
 CHECK(open_attempts==opens);
 public_early(nvm_file_execute_cyclic_bytes(cyclic_grant,wire,n,&options,NULL),NVM_FILE_RUNTIME_INVALID,32);
 public_early(nvm_file_execute_cyclic_bytes(cyclic_grant,NULL,0,&options,&out),NVM_FILE_RUNTIME_INVALID,32);
 CHECK(!memcmp(&out,&before,sizeof out));
 options.instruction_limit=0;NvmFileCyclicExecutionReport r=nvm_file_execute_cyclic_bytes(cyclic_grant,wire,n,&options,&out);
 CHECK(r.runtime.status==NVM_FILE_RUNTIME_LIMIT && r.fuel_exhausted && !r.instructions_started && !r.instruction_limit && open_attempts==opens && !memcmp(&out,&before,sizeof out));
 options.instruction_limit=NVM_FILE_CYCLIC_FUEL_MAX;
 cyclic_reentry=true;r=nvm_file_execute_cyclic_bytes(cyclic_grant,wire,n,&options,&out);cyclic_reentry=false;
 CHECK(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started==32 && out.value==73 && cyclic_nested_count);
 /* The old public selector keeps refusing this same positive cyclic wire. */
 out=before;CHECK(nvm_file_execute_bytes(cyclic_grant,wire,n,&out).status==NVM_FILE_RUNTIME_UNRESOLVED);
 CHECK(!memcmp(&out,&before,sizeof out));char *text=(char *)(uintptr_t)1;char error[256];
 CHECK(nvm2c_emit_file_bytes(wire,n,"old",&text,error,sizeof error)==NVM_FILE_RUNTIME_UNRESOLVED && text==(char *)(uintptr_t)1);
 CHECK(nvm_file_host_grant_revoke(cyclic_grant)==NVM_FILE_HOST_OK);
 public_early(nvm_file_execute_cyclic_bytes(cyclic_grant,wire,n,&options,&out),NVM_FILE_RUNTIME_STATE,NVM_FILE_CYCLIC_FUEL_MAX);
 CHECK(!memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_host_grant_destroy(&cyclic_grant)==NVM_FILE_HOST_OK);
 CHECK(nvm_file_host_grant_create_temporary_files(&cyclic_grant)==NVM_FILE_HOST_OK);
 r=nvm_file_execute_cyclic_bytes(cyclic_grant,wire,n,&options,&out);CHECK(r.runtime.status==NVM_FILE_RUNTIME_OK && out.value==73);
 const char *badnames[]={NULL,"","_bad","9bad","bad-name","bad name"};
 for(unsigned i=0;i<sizeof badnames/sizeof *badnames;i++){
  text=(char *)(uintptr_t)1;CHECK(nvm2c_emit_file_cyclic_bytes(wire,n,badnames[i],&text,error,sizeof error)==NVM_FILE_RUNTIME_INVALID && text==(char *)(uintptr_t)1);
 }
 char name[65];memset(name,'a',64);name[64]=0;text=(char *)(uintptr_t)1;
 CHECK(nvm2c_emit_file_cyclic_bytes(wire,n,name,&text,error,sizeof error)==NVM_FILE_RUNTIME_INVALID && text==(char *)(uintptr_t)1);
 name[63]=0;ROK(nvm2c_emit_file_cyclic_bytes(wire,n,name,&text,error,sizeof error));release_wire(text);
 for(unsigned i=0;i<5;i++)CHECK(nvm_file_runtime_cyclic_public_abi(i?1:2,
  sizeof(NvmFileCyclicOptions)+(i==1),sizeof(NvmFileCyclicExecutionReport)+(i==2),sizeof(NvmFileScalar)+(i==3))==(i==4));
 if(directory){
  const char *names[]={"loop","initializer","bool-false","bool-true","negative-int"};
  for(unsigned i=0;i<5;i++){
   NvmModule *item;
   if(i==0)item=cloop(&b,false,1);else if(i==1)item=init_helper(&b);
   else {FrameSpec spec={.result=i<4?-2:-1};
    if(i<4){op(&spec.code,OP_PUSH_BOOL);op(&spec.code,i==3);}else fi(&spec.code,-257);
    op(&spec.code,OP_RET);item=frame_module(&spec,1,&b,false,-1);}
   size_t length;uint8_t *data=serialize(item,&length);nvm_module_free(item);char path[4096];
   int count=snprintf(path,sizeof path,"%s/%s.nvm",directory,names[i]);CHECK(count>0 && (size_t)count<sizeof path);
   FILE *f=fopen(path,"wb");CHECK(f && fwrite(data,1,length,f)==length && !fclose(f));release_wire(data);
  }
 }
 /* Unsupported instructions refuse before begin even at zero fuel. */
 m=dead_label(&b);m->code[m->code_size-2]=OP_PRINT;size_t badn;uint8_t *bad=serialize(m,&badn);nvm_module_free(m);
 options.instruction_limit=0;out=before;opens=open_attempts;
 public_early(nvm_file_execute_cyclic_bytes(cyclic_grant,bad,badn,&options,&out),NVM_FILE_RUNTIME_UNRESOLVED,0);
 CHECK(open_attempts==opens && !memcmp(&out,&before,sizeof out));release_wire(bad);
 /* A successful old acyclic invocation also holds the shared gate against cyclic reentry. */
 m=owner_module(&b,false);size_t oldn;uint8_t *oldwire=serialize(m,&oldn);nvm_module_free(m);
 unsigned nested=cyclic_nested_count;cyclic_reentry=true;
 NvmFileRuntimeReport oldreport=nvm_file_execute_bytes(cyclic_grant,oldwire,oldn,&out);cyclic_reentry=false;
 CHECK(oldreport.status==NVM_FILE_RUNTIME_OK && out.value==37 && cyclic_nested_count>nested);
 release_wire(oldwire);release_wire(wire);empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
}
#ifdef FILE_CYCLIC_CAPTURE
int main(int argc,char **argv){CHECK(argc==2);CHECK(nvm_file_host_grant_create_temporary_files(&cyclic_grant)==NVM_FILE_HOST_OK);
 CHECK(cyclic_prior_main(argc,argv)==0);CHECK(cyclic_public_calls>30);public_boundaries(argv[1]);
 CHECK(nvm_file_host_grant_destroy(&cyclic_grant)==NVM_FILE_HOST_OK);
 puts("PASS public cyclic VM full corpus and grant boundary");return 0;}
#else
int main(void){(void)cyclic_public_bridge;(void)cyclic_public_emit;
 CHECK(nvm_file_host_grant_create_temporary_files(&cyclic_grant)==NVM_FILE_HOST_OK);
 CHECK(cyclic_prior_main()==0);CHECK(cyclic_public_calls==0);public_boundaries(NULL);
 CHECK(nvm_file_host_grant_destroy(&cyclic_grant)==NVM_FILE_HOST_OK);
 puts("PASS public cyclic native full corpus and grant boundary");return 0;}
#endif
