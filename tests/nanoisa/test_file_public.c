/* I retain every private corpus assertion, routing actual execution through the
 * public grant/scalar boundary. Manual primitives remain separately labeled. */
#include "../../src/nanoisa/file_public.h"
#include <pthread.h>
static NvmFileHostGrant *public_grant;
static NvmFileRuntimeReport public_vm_bridge(const uint8_t *,size_t,NvmFileRuntimeView *);
static NvmFileRuntimeStatus public_emit_bridge(const uint8_t *,size_t,char **,char *,size_t);
static void public_nested(void);
#define FILE_RUNTIME_NESTED_EXTRA public_nested
#define FILE_VM_BOUNDARY_LABEL "legacy grant-less File routes remain refused"
#define FILE_NATIVE_VM_EXECUTE public_vm_bridge
#define FILE_NATIVE_EMIT public_emit_bridge
#define FILE_NATIVE_MAIN public_prior_main
#include "test_file_private_native.c"
#undef FILE_NATIVE_MAIN
#undef FILE_RUNTIME_NESTED_EXTRA
#include "../../src/nanoisa/file_public_internal.h"

static bool public_reentry;
static unsigned public_reentries;
static const uint8_t *public_wire;
static size_t public_size;
static unsigned public_calls;
#if defined(FILE_NATIVE_CAPTURE) && !defined(HOSTED_INSTRUMENT)
static const char *public_parity_directory;static unsigned public_parity_count;
NvmFileRuntimeStatus file_reference_emit(const uint8_t *,size_t,char **,char *,size_t);
#endif
NvmFileHostGrant *file_public_test_grant(void){return public_grant;}
/* The bridge checks the actual public sentinel before reconstructing the old
 * passive view expected by unchanged corpus assertions. */
static NvmFileRuntimeReport public_vm_bridge(const uint8_t *bytes,size_t size,NvmFileRuntimeView *out){
 NvmFileScalar scalar;memset(&scalar,0xa5,sizeof scalar);NvmFileScalar before=scalar;
 NvmFileRuntimeReport report=nvm_file_execute_bytes(public_grant,bytes,size,out?&scalar:NULL);
 public_calls++;
 if(report.status==NVM_FILE_RUNTIME_OK){
  CHECK(out && (scalar.tag==TAG_INT || scalar.tag==TAG_BOOL));
  NvmFileRuntimeView v={0};v.initialized=true;v.fields=1;v.type.tag=scalar.tag;
  v.type.category=NVM_FILE_CATEGORY_UNKNOWN;v.type.global_index=v.type.catalog_ordinal=NVM_V2_NO_INDEX;
  v.values[0]=scalar.value;*out=v;
 }else CHECK(!memcmp(&scalar,&before,sizeof scalar));
 return report;
}
static NvmFileRuntimeStatus public_emit_bridge(const uint8_t *bytes,size_t size,char **out,char *error,size_t n){
 NvmFileRuntimeStatus status=nvm2c_emit_file_bytes(bytes,size,"case",out,error,n);
#if defined(FILE_NATIVE_CAPTURE) && !defined(HOSTED_INSTRUMENT)
 if(status==NVM_FILE_RUNTIME_OK && public_parity_directory){
  char *current=NULL,*reference=NULL;char diagnostic[256],path[4096];
  CHECK(nvm2c_file_private_emit(bytes,size,&current,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_OK);
  CHECK(file_reference_emit(bytes,size,&reference,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_OK);
  CHECK(!strcmp(current,reference));
  const char *text[2]={current,reference};const char *label[2]={"current","reference"};
  for(unsigned i=0;i<2;i++){
   int len=snprintf(path,sizeof path,"%s/parity-%03u-%s.c",public_parity_directory,public_parity_count,label[i]);
   CHECK(len>0 && (size_t)len<sizeof path);FILE *f=fopen(path,"wb");CHECK(f);
   size_t length=strlen(text[i]);CHECK(fwrite(text[i],1,length,f)==length && !fclose(f));
  }
  public_parity_count++;free(current);free(reference);
 }
#endif
 return status;
}
#ifndef FILE_NATIVE_CAPTURE
void file_public_native_busy_check(void);
#endif
static void public_busy_checks(void){
#ifndef FILE_NATIVE_CAPTURE
 file_public_native_busy_check();
#endif
 NvmFileScalar out;memset(&out,0xa5,sizeof out);NvmFileScalar before=out;
 NvmFileRuntimeReport r=nvm_file_execute_bytes(public_grant,public_wire,public_size,&out);
 CHECK(r.status==NVM_FILE_RUNTIME_BUSY && !r.acquired && !memcmp(&out,&before,sizeof out));
 char *text=(char *)(uintptr_t)1;char error[64];memset(error,'x',sizeof error);
 CHECK(nvm2c_emit_file_bytes(public_wire,public_size,"busy",&text,error,sizeof error)==NVM_FILE_RUNTIME_BUSY);
 CHECK(text==(char *)(uintptr_t)1 && error[0]=='x');
 CHECK(nvm_file_host_grant_revoke(public_grant)==NVM_FILE_HOST_BUSY);
 NvmFileHostGrant *copy=public_grant;CHECK(nvm_file_host_grant_destroy(&copy)==NVM_FILE_HOST_BUSY && copy==public_grant);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);
}
static void public_nested(void){if(public_reentry){public_busy_checks();public_reentries++;}}
/* Worker checks use independent outputs; only the owning main thread reads
 * instrumentation counters after joining. CHECK's shared counter is avoided. */
static void *public_contender(void *unused){
 (void)unused;NvmFileScalar out={TAG_INT,913},before=out;
 NvmFileRuntimeReport r=nvm_file_execute_bytes(public_grant,public_wire,public_size,&out);
 char *text=(char *)(uintptr_t)1;char error[8]="held";
 if(r.status!=NVM_FILE_RUNTIME_BUSY || r.acquired || memcmp(&out,&before,sizeof out) ||
    nvm2c_emit_file_bytes(public_wire,public_size,"thread",&text,error,sizeof error)!=NVM_FILE_RUNTIME_BUSY ||
    text!=(char *)(uintptr_t)1 || strcmp(error,"held"))return (void *)(uintptr_t)1;
 return NULL;
}
static void public_free(void *p){
#ifdef HOSTED_INSTRUMENT
 file_test_free(p);
#else
 free(p);
#endif
}
static void public_reject(const uint8_t *bytes,size_t n,NvmFileRuntimeStatus expected){
 unsigned opens=open_attempts,loads=loader_attempts,forks=fork_attempts;
 NvmFileScalar out;memset(&out,0xa5,sizeof out);NvmFileScalar old=out;
 NvmFileRuntimeReport r=nvm_file_execute_bytes(public_grant,bytes,n,&out);
 CHECK(r.status==expected && !r.acquired && !memcmp(&out,&old,sizeof out));
 char *text=(char *)(uintptr_t)1;char diagnostic[256];
 CHECK(nvm2c_emit_file_bytes(bytes,n,"rejected",&text,diagnostic,sizeof diagnostic)==expected);
 CHECK(text==(char *)(uintptr_t)1 && open_attempts==opens && loader_attempts==loads && fork_attempts==forks);
}
static void public_wire_refusals(const uint8_t *wire,size_t n){
 uint8_t *bytes=malloc(n);CHECK(bytes);memcpy(bytes,wire,n);
 NvmV2Header h;CHECK(nvm_v2_read_header(bytes,n,&h)==NVM_V2_OK);uint32_t features=h.feature_bits;
 const uint32_t required[]={NVM_V2_FEATURE_FFI,NVM_V2_FEATURE_RETAINED_LAYOUTS,NVM_V2_FEATURE_OWNERSHIP,NVM_V2_FEATURE_SERVICE_BINDINGS};
 for(unsigned i=0;i<4;i++){h.feature_bits=features & ~required[i];nvm_v2_write_header(bytes,&h);public_reject(bytes,n,NVM_FILE_RUNTIME_INVALID);}
 h.feature_bits=features|UINT32_C(0x400);nvm_v2_write_header(bytes,&h);public_reject(bytes,n,NVM_FILE_RUNTIME_INVALID);
 h.feature_bits=features|NVM_V2_FEATURE_CALLBACKS;nvm_v2_write_header(bytes,&h);public_reject(bytes,n,NVM_FILE_RUNTIME_UNRESOLVED);
 memcpy(bytes,wire,n);NvmV2SectionEntry svc=section(bytes,n,NVM_V2_SECTION_SERVICE_BINDINGS);
 bytes[svc.offset]=1;rehash(bytes,n);public_reject(bytes,n,NVM_FILE_RUNTIME_INVALID);
 memcpy(bytes,wire,n);bytes[svc.offset+2]^=1;rehash(bytes,n);public_reject(bytes,n,NVM_FILE_RUNTIME_INVALID);
 memcpy(bytes,wire,n);bytes[3]=1;public_reject(bytes,n,NVM_FILE_RUNTIME_INVALID);
 public_reject(wire,n-1,NVM_FILE_RUNTIME_INVALID);public_reject(NULL,0,NVM_FILE_RUNTIME_INVALID);
 uint8_t dummy=0;public_reject(&dummy,(size_t)NVM_FILE_HOSTED_INPUT_BYTES+1,NVM_FILE_RUNTIME_LIMIT);free(bytes);
 NvmFileNominalBindings b;size_t size;
 for(unsigned kind=0;kind<5;kind++){
  NvmModule *m=bodymodule(&b,false);
  if(kind==2)m->header.entry_point=1;
  else {
   make_initializer(m,3);
   if(kind!=3){
    Body code={0};size_t at=ownership_function_offset(m,3);bool owner=kind==1 || kind==4;
    m->functions[3].result_count=1;m->functions[3].result_tag=owner?TAG_UNION:TAG_INT;
    desc(m->ownership_data+at+4,owner?TAG_UNION:TAG_INT,0,owner?b.layouts[3]:NVM_V2_NO_INDEX);
    if(owner)service(&code,b,0,UINT16_MAX);else fi(&code,3);op(&code,OP_RET);setbody(m,3,code);
   }
   if(kind>=3){m->functions[3].name_idx=string(m,"invalid_entry");m->header.entry_point=3;}
  }
  uint8_t *bad=serialize(m,&size);nvm_module_free(m);public_reject(bad,size,NVM_FILE_RUNTIME_UNRESOLVED);public_free(bad);
 }
}
static void public_boundaries(void){
 NvmFileNominalBindings b;NvmModule *m=vm_io_module(&b,false,0);size_t n;uint8_t *bytes=serialize(m,&n);nvm_module_free(m);
#if defined(FILE_NATIVE_CAPTURE) && !defined(HOSTED_INSTRUMENT)
 char path[4096];int length=snprintf(path,sizeof path,"%s/public-lifecycle.nvm",public_parity_directory);
 CHECK(length>0 && (size_t)length<sizeof path);FILE *wire=fopen(path,"wb");CHECK(wire);
 CHECK(fwrite(bytes,1,n,wire)==n && !fclose(wire));
 const char *names[]={"bool-false","bool-true","negative-int"};
 for(unsigned i=0;i<3;i++){
  FrameSpec spec={.result=i<2?-2:-1};NvmFileNominalBindings bindings;
  int64_t expected=i<2?(int64_t)i:INT64_C(-257);
  if(i<2)vb(&spec.code,i!=0);else fi(&spec.code,expected);op(&spec.code,OP_RET);
  NvmModule *scalar_module=frame_module(&spec,1,&bindings,false,-1);size_t scalar_size;
  uint8_t *scalar_wire=serialize(scalar_module,&scalar_size);nvm_module_free(scalar_module);
  NvmFileScalar scalar_out={0};NvmFileRuntimeReport scalar_report=nvm_file_execute_bytes(public_grant,scalar_wire,scalar_size,&scalar_out);
  CHECK(scalar_report.status==NVM_FILE_RUNTIME_OK && scalar_out.tag==(i<2?TAG_BOOL:TAG_INT) && scalar_out.value==expected);
  length=snprintf(path,sizeof path,"%s/%s.nvm",public_parity_directory,names[i]);CHECK(length>0 && (size_t)length<sizeof path);
  wire=fopen(path,"wb");CHECK(wire);CHECK(fwrite(scalar_wire,1,scalar_size,wire)==scalar_size && !fclose(wire));public_free(scalar_wire);
 }
#endif
 public_wire_refusals(bytes,n);
 public_wire=bytes;public_size=n;unsigned opens=open_attempts,loads=loader_attempts,forks=fork_attempts;
 NvmFileScalar out;memset(&out,0xa5,sizeof out);NvmFileScalar before=out;
 NvmFileRuntimeReport r=nvm_file_execute_bytes(NULL,bytes,n,&out);
 CHECK(r.status==NVM_FILE_RUNTIME_INVALID && !r.acquired && !memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_host_enter(public_grant,NVM_FILE_HOST_ABI+1,NVM_FILE_HOST_CATALOG)==NVM_FILE_HOST_UNRESOLVED);
 CHECK(nvm_file_host_enter(public_grant,NVM_FILE_HOST_ABI,NVM_FILE_HOST_CATALOG+1)==NVM_FILE_HOST_UNRESOLVED);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);public_busy_checks();
 pthread_t threads[16];for(unsigned i=0;i<16;i++)CHECK(!pthread_create(&threads[i],NULL,public_contender,NULL));
 for(unsigned i=0;i<16;i++){void *result=NULL;CHECK(!pthread_join(threads[i],&result) && !result);}
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);nvm_file_host_leave();
 CHECK(open_attempts==opens && loader_attempts==loads && fork_attempts==forks);
 const char *invalid[]={NULL,"","_bad","0bad","a-b","a b","a;bad","a\"bad"};
 for(unsigned i=0;i<sizeof invalid/sizeof *invalid;i++){
  char *text=(char *)(uintptr_t)1;char error[128];
  CHECK(nvm2c_emit_file_bytes(bytes,n,invalid[i],&text,error,sizeof error)==NVM_FILE_RUNTIME_INVALID && text==(char *)(uintptr_t)1);
 }
 char name[65];memset(name,'a',64);name[64]='\0';char *text=(char *)(uintptr_t)1;char error[128];
 CHECK(nvm2c_emit_file_bytes(bytes,n,name,&text,error,sizeof error)==NVM_FILE_RUNTIME_INVALID && text==(char *)(uintptr_t)1);
 name[63]='\0';CHECK(nvm2c_emit_file_bytes(bytes,n,name,&text,error,sizeof error)==NVM_FILE_RUNTIME_OK);public_free(text);
 CHECK(nvm2c_emit_file_bytes(bytes,n,"a",&text,error,sizeof error)==NVM_FILE_RUNTIME_OK);public_free(text);
 CHECK(open_attempts==opens);
 public_reentry=true;r=nvm_file_execute_bytes(public_grant,bytes,n,&out);public_reentry=false;
 CHECK(r.status==NVM_FILE_RUNTIME_OK && out.tag==TAG_INT && out.value==251);
#ifdef HOSTED_INSTRUMENT
 CHECK(public_reentries>0);
#else
 CHECK(public_reentries==0);
#endif
 CHECK(nvm_file_host_grant_revoke(public_grant)==NVM_FILE_HOST_OK);out=before;
 r=nvm_file_execute_bytes(public_grant,bytes,n,&out);
 CHECK(r.status==NVM_FILE_RUNTIME_STATE && !r.acquired && !memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_host_grant_destroy(&public_grant)==NVM_FILE_HOST_OK && !public_grant);
 CHECK(nvm_file_host_grant_create_temporary_files(&public_grant)==NVM_FILE_HOST_OK);
 r=nvm_file_execute_bytes(public_grant,bytes,n,&out);CHECK(r.status==NVM_FILE_RUNTIME_OK && out.value==251);
 public_wire=NULL;public_size=0;public_free(bytes);empty_host();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
}
#ifdef FILE_NATIVE_CAPTURE
int main(int argc,char **argv){
 CHECK(argc==2);
#ifndef HOSTED_INSTRUMENT
 public_parity_directory=argv[1];
#endif
 CHECK(nvm_file_host_grant_create_temporary_files(&public_grant)==NVM_FILE_HOST_OK);
 CHECK(public_prior_main(argc,argv)==0);CHECK(public_calls>30);public_boundaries();
 CHECK(nvm_file_host_grant_destroy(&public_grant)==NVM_FILE_HOST_OK);
 puts("PASS public VM corpus, shared grant and scalar boundary");return 0;
}
#else
int main(void){
 (void)public_vm_bridge;(void)public_emit_bridge;
 CHECK(nvm_file_host_grant_create_temporary_files(&public_grant)==NVM_FILE_HOST_OK);
 CHECK(public_prior_main()==0);CHECK(public_calls==0);public_boundaries();
 CHECK(nvm_file_host_grant_destroy(&public_grant)==NVM_FILE_HOST_OK);
 puts("PASS public generated-native corpus and shared gate");return 0;
}
#endif
