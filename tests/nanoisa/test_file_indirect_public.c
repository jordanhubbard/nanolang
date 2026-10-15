/* I retain the complete private corpus and inspect the actual public scalar
 * before rebuilding the passive view expected by its unchanged assertions. */
#include "../../src/nanoisa/file_indirect_public.h"
#include "../../src/nanoisa/file_cyclic_public.h"
static NvmFileHostGrant *indirect_grant;
static NvmFileIndirectExecutionReport indirect_public_bridge(const uint8_t *,size_t,const NvmFileIndirectOptions *,NvmFileRuntimeView *);
static NvmFileRuntimeStatus indirect_public_emit(const uint8_t *,size_t,char **,char *,size_t);
static void indirect_nested(void);
#define FILE_RUNTIME_NESTED_EXTRA indirect_nested
#define FILE_INDIRECT_VM_EXECUTE indirect_public_bridge
#define FILE_INDIRECT_EMIT indirect_public_emit
#define FILE_INDIRECT_DISPATCH_MAIN indirect_prior_main
#include "test_file_indirect_dispatch.c"
#undef FILE_INDIRECT_DISPATCH_MAIN
#undef FILE_RUNTIME_NESTED_EXTRA
#include "../../src/nanoisa/file_indirect_public_internal.h"
#include "../../src/nanoisa/file_indirect_native_public.h"
static bool indirect_reentry;
static unsigned indirect_nested_count,indirect_public_calls;
NvmFileHostGrant *file_indirect_public_test_grant(void){return indirect_grant;}
static NvmFileIndirectExecutionReport indirect_public_bridge(const uint8_t *bytes,size_t n,
 const NvmFileIndirectOptions *options,NvmFileRuntimeView *out){
 NvmFileScalar value,before;memset(&value,0xa5,sizeof value);before=value;
 NvmFileIndirectExecutionReport r=nvm_file_execute_indirect_bytes(indirect_grant,bytes,n,options,out?&value:NULL);
 indirect_public_calls++;
 if(r.runtime.status==NVM_FILE_RUNTIME_OK){
  CHECK(out && (value.tag==TAG_INT || value.tag==TAG_BOOL));
  NvmFileRuntimeView view={0};view.initialized=true;view.fields=1;
  view.type.tag=value.tag;view.type.category=NVM_FILE_CATEGORY_UNKNOWN;
  view.type.global_index=view.type.catalog_ordinal=NVM_V2_NO_INDEX;view.values[0]=value.value;*out=view;
 }else CHECK(!memcmp(&value,&before,sizeof value));
 return r;
}
static NvmFileRuntimeStatus indirect_public_emit(const uint8_t *bytes,size_t n,char **out,char *error,size_t cap){
 return nvm2c_emit_file_indirect_bytes(bytes,n,"case",out,error,cap);
}
#ifndef FILE_INDIRECT_CAPTURE
void file_indirect_public_native_busy(void);
#endif
static void indirect_busy(void){
 const void *invalid=(const void *)(uintptr_t)1;
 NvmFileScalar out,before;memset(&out,0xa5,sizeof out);before=out;
 NvmFileIndirectExecutionReport r=nvm_file_execute_indirect_bytes((NvmFileHostGrant *)invalid,
  invalid,SIZE_MAX,invalid,&out);
 CHECK(r.revision==1 && r.runtime.status==NVM_FILE_RUNTIME_BUSY && !r.runtime.acquired &&
  r.runtime.core_status==0 && r.runtime.function==UINT32_MAX && r.runtime.instruction==UINT32_MAX &&
  !r.instruction_limit && !r.instructions_started && !r.fuel_exhausted &&
  !r.runtime.cleanup.cleanup_failures && !memcmp(&out,&before,sizeof out));
 /* Both APIs also leave invalid output addresses unread while the gate is held. */
 CHECK(nvm_file_execute_indirect_bytes((NvmFileHostGrant *)invalid,invalid,SIZE_MAX,invalid,(NvmFileScalar *)invalid).runtime.status==NVM_FILE_RUNTIME_BUSY);
 CHECK(nvm_file_execute_bytes((NvmFileHostGrant *)invalid,invalid,SIZE_MAX,(NvmFileScalar *)invalid).status==NVM_FILE_RUNTIME_BUSY);
 char *text=(char *)(uintptr_t)1;char diagnostic[8]="held";
 CHECK(nvm2c_emit_file_indirect_bytes(invalid,SIZE_MAX,invalid,&text,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_BUSY);
 CHECK(text==(char *)(uintptr_t)1 && !strcmp(diagnostic,"held"));
 CHECK(nvm2c_emit_file_indirect_bytes(invalid,SIZE_MAX,invalid,(char **)invalid,(char *)invalid,SIZE_MAX)==NVM_FILE_RUNTIME_BUSY);
 CHECK(nvm2c_emit_file_bytes(invalid,SIZE_MAX,invalid,(char **)invalid,(char *)invalid,SIZE_MAX)==NVM_FILE_RUNTIME_BUSY);
 CHECK(nvm_file_host_grant_revoke(indirect_grant)==NVM_FILE_HOST_BUSY);
 NvmFileHostGrant *copy=indirect_grant;CHECK(nvm_file_host_grant_destroy(&copy)==NVM_FILE_HOST_BUSY && copy==indirect_grant);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);
#ifndef FILE_INDIRECT_CAPTURE
 file_indirect_public_native_busy();
#endif
}
static void indirect_nested(void){if(indirect_reentry){indirect_busy();indirect_nested_count++;}}
static void public_early(NvmFileIndirectExecutionReport r,NvmFileRuntimeStatus expected,uint64_t fuel){
 CHECK(r.revision==1 && r.runtime.status==expected && !r.runtime.acquired &&
  r.runtime.function==UINT32_MAX && r.runtime.instruction==UINT32_MAX &&
  r.instruction_limit==fuel && !r.instructions_started && !r.fuel_exhausted &&
  !r.runtime.cleanup.cleanup_failures);
}
static void public_boundaries(const char *directory){
 NvmFileNominalBindings b;NvmModule *m=cloop(&b,false,1);size_t n;
 uint8_t *wire=serialize(m,&n);nvm_module_free(m);unsigned opens=open_attempts;
 NvmFileScalar out,before;memset(&out,0xa5,sizeof out);before=out;
 NvmFileIndirectOptions options={1,32};
 public_early(nvm_file_execute_indirect_bytes(NULL,wire,n,&options,&out),NVM_FILE_RUNTIME_INVALID,32);
 CHECK(!memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);indirect_busy();nvm_file_host_leave();
 CHECK(open_attempts==opens);
 public_early(nvm_file_execute_indirect_bytes(indirect_grant,wire,n,&options,NULL),NVM_FILE_RUNTIME_INVALID,32);
 public_early(nvm_file_execute_indirect_bytes(indirect_grant,NULL,0,&options,&out),NVM_FILE_RUNTIME_INVALID,32);
 CHECK(!memcmp(&out,&before,sizeof out));
 options.instruction_limit=0;NvmFileIndirectExecutionReport r=nvm_file_execute_indirect_bytes(indirect_grant,wire,n,&options,&out);
 CHECK(r.runtime.status==NVM_FILE_RUNTIME_LIMIT && r.fuel_exhausted && !r.instructions_started && !r.instruction_limit && open_attempts==opens && !memcmp(&out,&before,sizeof out));
 options.instruction_limit=NVM_FILE_INDIRECT_FUEL_MAX;
 indirect_reentry=true;r=nvm_file_execute_indirect_bytes(indirect_grant,wire,n,&options,&out);indirect_reentry=false;
 CHECK(r.runtime.status==NVM_FILE_RUNTIME_OK && r.instructions_started==32 && out.value==73 && indirect_nested_count);
 /* The old public selector keeps refusing this same positive indirect wire. */
 out=before;CHECK(nvm_file_execute_bytes(indirect_grant,wire,n,&out).status==NVM_FILE_RUNTIME_UNRESOLVED);
 CHECK(!memcmp(&out,&before,sizeof out));char *text=(char *)(uintptr_t)1;char error[256];
 CHECK(nvm2c_emit_file_bytes(wire,n,"old",&text,error,sizeof error)==NVM_FILE_RUNTIME_UNRESOLVED && text==(char *)(uintptr_t)1);
 CHECK(nvm_file_host_grant_revoke(indirect_grant)==NVM_FILE_HOST_OK);
 public_early(nvm_file_execute_indirect_bytes(indirect_grant,wire,n,&options,&out),NVM_FILE_RUNTIME_STATE,NVM_FILE_INDIRECT_FUEL_MAX);
 CHECK(!memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_host_grant_destroy(&indirect_grant)==NVM_FILE_HOST_OK);
 CHECK(nvm_file_host_grant_create_temporary_files(&indirect_grant)==NVM_FILE_HOST_OK);
 r=nvm_file_execute_indirect_bytes(indirect_grant,wire,n,&options,&out);CHECK(r.runtime.status==NVM_FILE_RUNTIME_OK && out.value==73);
 NvmModule *indirect_module=borrowed_indirect(0,0,0);size_t indirect_size;
 uint8_t *indirect_wire=serialize(indirect_module,&indirect_size);nvm_module_free(indirect_module);
 NvmFileCyclicOptions legacy_options={1,1000000};out=before;
 CHECK(nvm_file_execute_cyclic_bytes(indirect_grant,indirect_wire,indirect_size,&legacy_options,&out).runtime.status==NVM_FILE_RUNTIME_UNRESOLVED);
 CHECK(!memcmp(&out,&before,sizeof out));text=(char *)(uintptr_t)1;
 CHECK(nvm2c_emit_file_cyclic_bytes(indirect_wire,indirect_size,"old",&text,error,sizeof error)==NVM_FILE_RUNTIME_UNRESOLVED && text==(char *)(uintptr_t)1);
 release_wire(indirect_wire);
 const char *badnames[]={NULL,"","_bad","9bad","bad-name","bad name"};
 for(unsigned i=0;i<sizeof badnames/sizeof *badnames;i++){
  text=(char *)(uintptr_t)1;CHECK(nvm2c_emit_file_indirect_bytes(wire,n,badnames[i],&text,error,sizeof error)==NVM_FILE_RUNTIME_INVALID && text==(char *)(uintptr_t)1);
 }
 char name[65];memset(name,'a',64);name[64]=0;text=(char *)(uintptr_t)1;
 CHECK(nvm2c_emit_file_indirect_bytes(wire,n,name,&text,error,sizeof error)==NVM_FILE_RUNTIME_INVALID && text==(char *)(uintptr_t)1);
 name[63]=0;ROK(nvm2c_emit_file_indirect_bytes(wire,n,name,&text,error,sizeof error));release_wire(text);
 for(unsigned i=0;i<5;i++)CHECK(nvm_file_runtime_indirect_public_abi(i?1:2,
  sizeof(NvmFileIndirectOptions)+(i==1),sizeof(NvmFileIndirectExecutionReport)+(i==2),sizeof(NvmFileScalar)+(i==3))==(i==4));
 if(directory){
  const char *names[]={"loop","initializer","bool-false","bool-true","negative-int","borrowed-maximum"};
  for(unsigned i=0;i<6;i++){
   NvmModule *item;
   if(i==0)item=cloop(&b,false,1);else if(i==1)item=init_helper(&b);else if(i==5)item=borrowed_maximum();
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
 public_early(nvm_file_execute_indirect_bytes(indirect_grant,bad,badn,&options,&out),NVM_FILE_RUNTIME_UNRESOLVED,0);
 CHECK(open_attempts==opens && !memcmp(&out,&before,sizeof out));release_wire(bad);
 /* A successful old acyclic invocation also holds the shared gate against indirect reentry. */
 m=owner_module(&b,false);size_t oldn;uint8_t *oldwire=serialize(m,&oldn);nvm_module_free(m);
 unsigned nested=indirect_nested_count;indirect_reentry=true;
 NvmFileRuntimeReport oldreport=nvm_file_execute_bytes(indirect_grant,oldwire,oldn,&out);indirect_reentry=false;
 CHECK(oldreport.status==NVM_FILE_RUNTIME_OK && out.value==37 && indirect_nested_count>nested);
 release_wire(oldwire);release_wire(wire);empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
}
#ifdef FILE_INDIRECT_CAPTURE
int main(int argc,char **argv){CHECK(argc==2);CHECK(nvm_file_host_grant_create_temporary_files(&indirect_grant)==NVM_FILE_HOST_OK);
 CHECK(indirect_prior_main(argc,argv)==0);CHECK(indirect_public_calls>30);public_boundaries(argv[1]);
 CHECK(nvm_file_host_grant_destroy(&indirect_grant)==NVM_FILE_HOST_OK);
 puts("PASS public indirect VM full corpus and grant boundary");return 0;}
#else
int main(void){(void)indirect_public_bridge;(void)indirect_public_emit;
 CHECK(nvm_file_host_grant_create_temporary_files(&indirect_grant)==NVM_FILE_HOST_OK);
 CHECK(indirect_prior_main()==0);CHECK(indirect_public_calls==0);public_boundaries(NULL);
 CHECK(nvm_file_host_grant_destroy(&indirect_grant)==NVM_FILE_HOST_OK);
 puts("PASS public indirect native full corpus and grant boundary");return 0;}
#endif
