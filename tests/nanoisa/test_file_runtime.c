/* I call private carrier primitives manually. I execute no File bytecode and
 * qualify no frame/call dispatcher or public File selection here. */
#define FILE_HOSTED_MAIN prior_file_hosted_fixture_main
#include "test_file_hosted.c"
#undef FILE_HOSTED_MAIN
#include "../../src/nanoisa/file_runtime.h"
#include "../../src/nanovm/vm.h"
#include "../../src/nanovm/vm_ffi.h"
#include "file_runtime_hooks.h"
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef HOSTED_INSTRUMENT
#define calloc file_test_calloc
#define free file_test_free
#include "../../src/nanoisa/file_runtime.c"
#undef calloc
#undef free
#endif
int g_argc;char **g_argv;
#define ROK(x) CHECK((x)==NVM_FILE_RUNTIME_OK)
#define NS NVM_FILE_RUNTIME_NO_SLOT
static unsigned open_attempts,opened,closed,io_attempts,loader_attempts,fork_attempts;
static FILE *streams[128];
static bool deny_open,deny_seek,model_read_error,model_write_error;
static FILE *modeled_error_stream;
static int close_error[4];static unsigned close_index;
static NvmFileRuntime *reenter;
static unsigned reentries;
static void nested_entry(void){
 if(!reenter)return;
 NvmFileRuntime *saved=reenter;NvmFileRuntimeView out;memset(&out,0xa5,sizeof out);NvmFileRuntimeView before=out;
 CHECK(nvm_file_runtime_begin(saved)==NVM_FILE_RUNTIME_BUSY);
 NvmFileRuntimeReport report=nvm_file_runtime_finish(saved,&out);
 CHECK(report.status==NVM_FILE_RUNTIME_BUSY && !report.acquired && !memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_runtime_destroy(&saved,&out).status==NVM_FILE_RUNTIME_BUSY && saved==reenter);
 CHECK(nvm_file_runtime_service(saved,0,NS,NS,0)==NVM_FILE_RUNTIME_BUSY);
 reentries++;
}
FILE *file_runtime_tmpfile(void){
 open_attempts++;nested_entry();if(deny_open){errno=EACCES;return NULL;}
 FILE *f=tmpfile();if(!f)return NULL;
 unsigned i=0;while(i<128 && streams[i])i++;CHECK(i<128);streams[i]=f;opened++;return f;
}
int file_runtime_fclose(FILE *f){
 nested_entry();unsigned i=0;while(i<128 && streams[i]!=f)i++;CHECK(i<128);streams[i]=NULL;closed++;
 int fd=fileno(f);CHECK(fd>=0);int rc=fclose(f),saved=errno;
 CHECK(fcntl(fd,F_GETFD)==-1 && errno==EBADF);int injected=close_index<4?close_error[close_index++]:0;
 /* I really close the stream, then model failure reporting. */
 if(injected){errno=injected;return EOF;}errno=saved;return rc;
}
size_t file_runtime_fread(void *p,size_t n,size_t count,FILE *f){
 io_attempts++;nested_entry();size_t done=fread(p,n,count,f);
 if(model_read_error){CHECK(n==1 && count==1 && done==1);modeled_error_stream=f;errno=EIO;}
 return done;
}
size_t file_runtime_fwrite(const void *p,size_t n,size_t count,FILE *f){
 io_attempts++;nested_entry();size_t done=fwrite(p,n,count,f);
 if(model_write_error){CHECK(n==1 && count==1 && done==1);modeled_error_stream=f;errno=ENOSPC;}
 return done;
}
int file_runtime_ferror(FILE *f){return f==modeled_error_stream?1:ferror(f);}
int file_runtime_fseek(FILE *f,long off,int whence){io_attempts++;nested_entry();if(deny_seek){errno=ESPIPE;return -1;}return fseek(f,off,whence);}
bool file_runtime_loader_init(bool verbose){(void)verbose;loader_attempts++;return false;}
bool file_runtime_loader_open(const char *name,const char *path){(void)name;(void)path;loader_attempts++;return false;}
pid_t file_runtime_fork(void){fork_attempts++;errno=EACCES;return -1;}
static void empty_host(void){CHECK(opened==closed);for(unsigned i=0;i<128;i++)CHECK(!streams[i]);}
static NvmFileRuntimeView view(NvmFileRuntime *c,uint32_t slot){NvmFileRuntimeView v;CHECK(nvm_file_runtime_view(c,slot,&v));return v;}
static void scalar(NvmFileRuntime *c,uint32_t slot,int64_t v){ROK(nvm_file_runtime_scalar(c,slot,TAG_INT,v));}
static void arm(NvmFileRuntime *c,uint32_t slot,NvmFileFlowArm expected){NvmFileFlowArm a=NVM_FILE_FLOW_ARM_UNKNOWN;ROK(nvm_file_runtime_result_arm(c,slot,&a));CHECK(a==expected);}
static uint8_t *runtime_wire(NvmFileNominalBindings *b,size_t *size,bool permuted,bool init){
 NvmModule *m=bodymodule(b,permuted);setbody(m,0,lifecycle_code(*b));if(init)make_initializer(m,3);
 uint8_t *bytes=serialize(m,size);nvm_module_free(m);return bytes;
}
static NvmFileRuntime *context(NvmFileNominalBindings *b,NvmFileRuntimeMode mode,bool permuted,bool init,bool begin){
 size_t size;uint8_t *bytes=runtime_wire(b,&size,permuted,init);NvmFileRuntime *c=NULL;
 unsigned attempts=open_attempts;ROK(nvm_file_runtime_create(bytes,size,mode,&c));CHECK(open_attempts==attempts);
 memset(bytes,0,size);free(bytes);NvmFileRuntimeStorage storage;CHECK(nvm_file_runtime_storage(c,&storage));
 NvmFileHostedStartup startup;CHECK(nvm_file_hosted_startup(nvm_file_runtime_plan(c),&startup));
 CHECK(storage.values==(mode==NVM_FILE_RUNTIME_VM?startup.vm_value_slots:startup.native_value_slots));
 CHECK(storage.references==startup.reference_slots && storage.regions==startup.region_slots && storage.frames==startup.frames);
 CHECK(storage.allocation_bound<=NVM_FILE_RUNTIME_BYTES);
#ifdef HOSTED_INSTRUMENT
 size_t core;CHECK(nl_file_values_storage_bound(&core));
 CHECK(storage.allocation_bound==startup.allocation_bound+sizeof(*c)+storage.values*sizeof(*c->values)+
       storage.references*sizeof(*c->references)+storage.regions*sizeof(*c->regions)+storage.frames*sizeof(*c->frames)+core);
 CHECK(tracked_bytes<=storage.allocation_bound);
#endif
 if(begin){
#ifdef HOSTED_INSTRUMENT
 size_t before_core=tracked_bytes;
#endif
 ROK(nvm_file_runtime_begin(c));CHECK(open_attempts==attempts);
#ifdef HOSTED_INSTRUMENT
 CHECK(tracked_bytes-before_core==core);
#endif
 }
 return c;
}
static void file(NvmFileRuntime *c,NvmFileNominalBindings b,uint32_t dst){
 ROK(nvm_file_runtime_service(c,b.imports[0],NS,NS,0));arm(c,0,NVM_FILE_FLOW_ARM_OK);
 ROK(nvm_file_runtime_take(c,0,NVM_FILE_FLOW_ARM_OK,dst));CHECK(!view(c,0).initialized && view(c,dst).owning);
}
static void finish_ok(NvmFileRuntime **c,int64_t result){
 scalar(*c,0,result);ROK(nvm_file_runtime_complete_root(*c,0));NvmFileRuntimeView out={0};
 NvmFileRuntimeReport r=nvm_file_runtime_finish(*c,&out);CHECK(r.status==NVM_FILE_RUNTIME_OK && r.acquired && !r.cleanup.cleanup_failures);
 CHECK(out.initialized && out.type.tag==TAG_INT && out.values[0]==result);
 unsigned closes=closed;NvmFileRuntimeReport again=nvm_file_runtime_finish(*c,&out);
 CHECK(again.status==r.status && closed==closes);CHECK(nvm_file_runtime_destroy(c,&out).status==NVM_FILE_RUNTIME_OK && !*c);empty_host();
}
static NvmFileRuntimeReport finish_bad(NvmFileRuntime **c,NvmFileRuntimeStatus status){
 NvmFileRuntimeView out;memset(&out,0xa5,sizeof out);NvmFileRuntimeView before=out;
 NvmFileRuntimeReport r=nvm_file_runtime_finish(*c,&out);CHECK(r.status==status && !memcmp(&out,&before,sizeof out));
 unsigned closes=closed;NvmFileRuntimeReport again=nvm_file_runtime_destroy(c,&out);
 CHECK(again.status==r.status && again.cleanup.cleanup_failures==r.cleanup.cleanup_failures && !*c && closed==closes);
 CHECK(!memcmp(&out,&before,sizeof out));empty_host();return r;
}
static void carrier_lifecycle(void){
 for(unsigned mode=0;mode<2;mode++)for(unsigned perm=0;perm<2;perm++){
  NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,(NvmFileRuntimeMode)mode,perm!=0,false,true);
  ROK(nvm_file_runtime_site(c,0,0));file(c,b,1);ROK(nvm_file_runtime_move(c,1,2));CHECK(!view(c,1).initialized);
  ROK(nvm_file_runtime_region_begin(c));ROK(nvm_file_runtime_borrow(c,2,0));
  ROK(nvm_file_runtime_bind_formal(c,0,3,1));ROK(nvm_file_runtime_bind_formal(c,1,4,2));
  scalar(c,5,255);ROK(nvm_file_runtime_service(c,b.imports[1],2,5,6));arm(c,6,NVM_FILE_FLOW_ARM_OK);
  CHECK(!view(c,5).initialized);ROK(nvm_file_runtime_take(c,6,NVM_FILE_FLOW_ARM_OK,5));CHECK(view(c,5).values[0]==1);ROK(nvm_file_runtime_drop(c,5));
  ROK(nvm_file_runtime_service(c,b.imports[2],2,NS,6));arm(c,6,NVM_FILE_FLOW_ARM_OK);ROK(nvm_file_runtime_drop(c,6));
  ROK(nvm_file_runtime_service(c,b.imports[3],2,NS,6));ROK(nvm_file_runtime_take(c,6,NVM_FILE_FLOW_ARM_OK,5));
  CHECK(view(c,5).type.global_index==b.layouts[2] && view(c,5).values[0]==255 && !view(c,5).values[1]);
  ROK(nvm_file_runtime_project(c,5,0,5));CHECK(view(c,5).values[0]==255);ROK(nvm_file_runtime_drop(c,5));
  ROK(nvm_file_runtime_service(c,b.imports[3],2,NS,6));ROK(nvm_file_runtime_take(c,6,NVM_FILE_FLOW_ARM_OK,5));CHECK(view(c,5).values[0]==0 && view(c,5).values[1]);ROK(nvm_file_runtime_drop(c,5));
  ROK(nvm_file_runtime_end_reference(c,1));CHECK(!view(c,3).initialized && view(c,4).formal);
  ROK(nvm_file_runtime_end_reference(c,2));ROK(nvm_file_runtime_end_reference(c,0));ROK(nvm_file_runtime_region_end(c));
  ROK(nvm_file_runtime_service(c,b.imports[4],NS,2,6));CHECK(!view(c,2).initialized);arm(c,6,NVM_FILE_FLOW_ARM_OK);ROK(nvm_file_runtime_drop(c,6));finish_ok(&c,255);
 }
}
static void passive(void){
 NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,NVM_FILE_RUNTIME_VM,true,false,true);uint32_t inputs[7];
 for(unsigned i=0;i<7;i++){inputs[i]=i;ROK(nvm_file_runtime_scalar(c,i,i<4?TAG_INT:TAG_BOOL,i<4?(int64_t)(100+i):1));}
 ROK(nvm_file_runtime_construct(c,1,0,inputs,7,0));CHECK(view(c,0).fields==7 && view(c,0).type.global_index==b.layouts[1]);
 ROK(nvm_file_runtime_copy(c,0,1));ROK(nvm_file_runtime_project(c,1,3,1));CHECK(view(c,1).values[0]==103);ROK(nvm_file_runtime_drop(c,1));
 uint32_t input=0;ROK(nvm_file_runtime_construct(c,7,1,&input,1,0));arm(c,0,NVM_FILE_FLOW_ARM_ERROR);
 ROK(nvm_file_runtime_project(c,0,0,0));CHECK(view(c,0).type.global_index==b.layouts[1]);ROK(nvm_file_runtime_drop(c,0));
 ROK(nvm_file_runtime_scalar(c,0,TAG_VOID,0));ROK(nvm_file_runtime_construct(c,5,0,&input,1,0));
 ROK(nvm_file_runtime_take(c,0,NVM_FILE_FLOW_ARM_OK,1));CHECK(view(c,1).type.tag==TAG_VOID && !view(c,1).fields);ROK(nvm_file_runtime_drop(c,1));finish_ok(&c,17);
}
static void invalid_and_partial(void){
 for(unsigned failure=0;failure<9;failure++){
  NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);file(c,b,1);
  NvmFileRuntimeView owner=view(c,1);NvmFileRuntimeStatus expected=NVM_FILE_RUNTIME_TYPE;
  if(failure==0)CHECK(nvm_file_runtime_copy(c,1,2)==expected);
  if(failure==1){scalar(c,2,7);CHECK(nvm_file_runtime_move(c,1,2)==expected);CHECK(view(c,2).values[0]==7);}
  if(failure>=2 && failure<=6){ROK(nvm_file_runtime_region_begin(c));ROK(nvm_file_runtime_borrow(c,1,0));}
  if(failure==2){expected=NVM_FILE_RUNTIME_BORROWED;CHECK(nvm_file_runtime_move(c,1,2)==expected);CHECK(!view(c,2).initialized);}
  if(failure==3){expected=NVM_FILE_RUNTIME_BORROWED;ROK(nvm_file_runtime_bind_formal(c,0,2,1));CHECK(nvm_file_runtime_end_reference(c,0)==expected);}
  if(failure==4){expected=NVM_FILE_RUNTIME_BORROWED;ROK(nvm_file_runtime_bind_formal(c,0,2,1));CHECK(nvm_file_runtime_region_end(c)==expected);}
  if(failure==5){ROK(nvm_file_runtime_bind_formal(c,0,2,1));CHECK(nvm_file_runtime_move(c,2,3)==expected);}
  if(failure==6){expected=NVM_FILE_RUNTIME_BORROWED;CHECK(nvm_file_runtime_service(c,b.imports[4],NS,1,2)==expected);CHECK(!view(c,2).initialized);}
  if(failure==7){expected=NVM_FILE_RUNTIME_INVALID;CHECK(nvm_file_runtime_service(c,UINT32_MAX,NS,NS,2)==expected);}
  if(failure==8)CHECK(nvm_file_runtime_complete_root(c,0)==expected);
  NvmFileRuntimeView after=view(c,1);CHECK(!memcmp(&owner,&after,sizeof owner));finish_bad(&c,expected);
 }
 /* Every staged OpenResult/File root is visited by terminal cleanup. */
 NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,NVM_FILE_RUNTIME_NATIVE,false,false,true);NvmFileRuntimeStorage s;CHECK(nvm_file_runtime_storage(c,&s));
 for(uint32_t i=0;i<s.values;i++)ROK(nvm_file_runtime_service(c,b.imports[0],NS,NS,i));
 CHECK(nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_ASSERT)==NVM_FILE_RUNTIME_ASSERT);finish_bad(&c,NVM_FILE_RUNTIME_ASSERT);
 c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);ROK(nvm_file_runtime_service(c,b.imports[0],NS,NS,0));
 CHECK(nvm_file_runtime_take(c,0,NVM_FILE_FLOW_ARM_ERROR,1)==NVM_FILE_RUNTIME_TYPE);CHECK(view(c,0).owning && !view(c,1).initialized);finish_bad(&c,NVM_FILE_RUNTIME_TYPE);
}
static void invalid_passive(void){
 for(unsigned which=0;which<4;which++){
  NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);
  scalar(c,0,44);ROK(nvm_file_runtime_scalar(c,1,TAG_BOOL,1));uint32_t inputs[]={0,1};
  if(which==0){inputs[1]=0;CHECK(nvm_file_runtime_construct(c,2,0,inputs,2,2)==NVM_FILE_RUNTIME_INVALID);}
  if(which==1){CHECK(nvm_file_runtime_construct(c,1,0,inputs,2,2)==NVM_FILE_RUNTIME_TYPE);}
  if(which==2){CHECK(nvm_file_runtime_construct(c,0,0,NULL,0,2)==NVM_FILE_RUNTIME_TYPE);}
  if(which==3){CHECK(nvm_file_runtime_construct(c,4,1,inputs,1,2)==NVM_FILE_RUNTIME_TYPE);}
  CHECK(view(c,0).values[0]==44 && view(c,1).values[0]==1 && !view(c,2).initialized);
  finish_bad(&c,which==0?NVM_FILE_RUNTIME_INVALID:NVM_FILE_RUNTIME_TYPE);
 }
}
static void scalar_and_limits(void){
 NvmFileNominalBindings b;size_t n;NvmModule *m=bodymodule(&b,false);Body code={0};
 m->functions[0].result_tag=TAG_BOOL;desc(m->ownership_data+28,TAG_BOOL,0,NVM_V2_NO_INDEX);
 op(&code,OP_PUSH_BOOL);op(&code,1);op(&code,OP_RET);setbody(m,0,code);uint8_t *bytes=serialize(m,&n);
 NvmFileRuntime *c=NULL;ROK(nvm_file_runtime_create(bytes,n,NVM_FILE_RUNTIME_VM,&c));free(bytes);nvm_module_free(m);
 ROK(nvm_file_runtime_begin(c));ROK(nvm_file_runtime_scalar(c,0,TAG_BOOL,1));ROK(nvm_file_runtime_complete_root(c,0));
 NvmFileRuntimeView result={0};CHECK(nvm_file_runtime_destroy(&c,&result).status==NVM_FILE_RUNTIME_OK && result.type.tag==TAG_BOOL && result.values[0]==1);
 c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);NvmFileRuntimeStorage storage;CHECK(nvm_file_runtime_storage(c,&storage));
 NvmFileRuntimeView out;memset(&out,0xa5,sizeof out);NvmFileRuntimeView old=out;
 CHECK(!nvm_file_runtime_view(c,storage.values,&out) && !memcmp(&out,&old,sizeof out));
 for(uint32_t i=0;i<storage.regions;i++)ROK(nvm_file_runtime_region_begin(c));
 CHECK(nvm_file_runtime_region_begin(c)==NVM_FILE_RUNTIME_LIMIT);finish_bad(&c,NVM_FILE_RUNTIME_LIMIT);
 c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);CHECK(nvm_file_runtime_scalar(c,0,TAG_BOOL,2)==NVM_FILE_RUNTIME_TYPE);CHECK(!view(c,0).initialized);finish_bad(&c,NVM_FILE_RUNTIME_TYPE);
 c=context(&b,NVM_FILE_RUNTIME_NATIVE,false,false,true);CHECK(nvm_file_runtime_storage(c,&storage));
 for(uint32_t i=1;i<storage.values;i++)file(c,b,i);
 ROK(nvm_file_runtime_service(c,b.imports[0],NS,NS,0));CHECK(nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_ASSERT)==NVM_FILE_RUNTIME_ASSERT);finish_bad(&c,NVM_FILE_RUNTIME_ASSERT);
#ifdef HOSTED_INSTRUMENT
 size_t amount=NVM_FILE_RUNTIME_BYTES-1;CHECK(fr_add(&amount,1,1) && amount==NVM_FILE_RUNTIME_BYTES);
 CHECK(!fr_add(&amount,1,1) && amount==NVM_FILE_RUNTIME_BYTES);amount=7;
 CHECK(!fr_add(&amount,SIZE_MAX,2) && amount==7);
 size_t sentinel=17;CHECK(!nl_file_values_storage_bound(NULL) && !nl_file_service_storage_bound(NULL));
 CHECK(nl_file_values_storage_bound(&sentinel) && sentinel>17);
#endif
}
static void initializer(void){
 NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,NVM_FILE_RUNTIME_VM,false,true,true);uint32_t root=99;
 CHECK(nvm_file_runtime_current_root(c,&root) && root==3);file(c,b,1);ROK(nvm_file_runtime_service(c,b.imports[4],NS,1,2));
 ROK(nvm_file_runtime_complete_root(c,NS));CHECK(nvm_file_runtime_current_root(c,&root) && root==0);finish_ok(&c,81);
 c=context(&b,NVM_FILE_RUNTIME_VM,false,false,false);unsigned attempts=open_attempts;
 CHECK(nvm_file_runtime_service(c,b.imports[0],NS,NS,0)==NVM_FILE_RUNTIME_STATE && open_attempts==attempts);
 CHECK(!nvm_file_runtime_current_root(c,&root));finish_bad(&c,NVM_FILE_RUNTIME_STATE);
}
static void public_refusal(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);setbody(m,0,lifecycle_code(b));
 unsigned opens=open_attempts,loads=loader_attempts,forks=fork_attempts;char error[256];
 CHECK(!nvm_verify(m).ok && !nvm_verify_owned_module(m).ok && !nvm_verify_function(m,0).ok);
 CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
 VmState vm;vm_init(&vm,m);NanoValue out=val_int(812);
 CHECK(vm_invoke(&vm,0,NULL,0,&out)!=VM_OK && out.tag==TAG_INT && out.as.i64==812);
 CHECK(vm_execute(&vm)!=VM_OK && !vm.stack_size && !vm.frame_count);
 CHECK(!vm_ffi_load_import(m,b.imports[0]));CHECK(!vm_ffi_call(m,b.imports[0],NULL,0,&out,&vm.heap,error,sizeof error));
 CHECK(!vm_ffi_call_vm(&vm,m,b.imports[0],NULL,0,&out,error,sizeof error));CHECK(!vm_ffi_cop_start(&vm,m));
 CHECK(!vm_ffi_call_cop(&vm,m,b.imports[0],NULL,0,&out,&vm.heap,error,sizeof error));
 CHECK(out.tag==TAG_INT && out.as.i64==812 && vm.cop_pid<=0);
 CHECK(opens==open_attempts && loads==loader_attempts && forks==fork_attempts);vm_destroy(&vm);nvm_module_free(m);
 /* Hook sanity is separate from the zero-attempt assertion; no real load/fork. */
 CHECK(!file_runtime_loader_init(false));CHECK(!file_runtime_loader_open("sentinel","sentinel"));CHECK(file_runtime_fork()==-1);
 CHECK(loader_attempts==loads+2 && fork_attempts==forks+1);
}
#ifdef HOSTED_INSTRUMENT
static void progress_error_payload(NvmFileRuntime *c,NvmFileNominalBindings b,uint32_t result,int error){
 arm(c,result,NVM_FILE_FLOW_ARM_ERROR);NvmFileRuntimeView v=view(c,result);
 const int64_t expected[]={NL_FILE_IO,error,0,1,0,0,0};
 CHECK(v.fields==7);for(unsigned i=0;i<7;i++)CHECK(v.values[i]==expected[i]);
 ROK(nvm_file_runtime_take(c,result,NVM_FILE_FLOW_ARM_ERROR,4));v=view(c,4);
 CHECK(v.type.global_index==b.layouts[1] && v.type.catalog_ordinal==1 && v.fields==7 && !v.owning);
 for(unsigned i=0;i<7;i++)CHECK(v.values[i]==expected[i]);
 ROK(nvm_file_runtime_drop(c,4));
 CHECK(view(c,1).initialized && view(c,1).owning && view(c,1).type.global_index==b.layouts[0]);
}
static void modeled_progress_errors(void){
 /* I perform the complete one-byte request, then model an error report with
  * progress. This does not claim an actual short transfer or device failure. */
 for(unsigned mode=0;mode<2;mode++){
  NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,(NvmFileRuntimeMode)mode,mode!=0,false,true);
  file(c,b,1);ROK(nvm_file_runtime_region_begin(c));ROK(nvm_file_runtime_borrow(c,1,0));
  scalar(c,2,173);model_write_error=true;
  ROK(nvm_file_runtime_service(c,b.imports[1],0,2,3));model_write_error=false;modeled_error_stream=NULL;
  CHECK(!view(c,2).initialized);progress_error_payload(c,b,3,ENOSPC);
  ROK(nvm_file_runtime_service(c,b.imports[2],0,NS,3));arm(c,3,NVM_FILE_FLOW_ARM_OK);ROK(nvm_file_runtime_drop(c,3));
  model_read_error=true;ROK(nvm_file_runtime_service(c,b.imports[3],0,NS,3));model_read_error=false;modeled_error_stream=NULL;
  progress_error_payload(c,b,3,EIO);
  ROK(nvm_file_runtime_service(c,b.imports[2],0,NS,3));arm(c,3,NVM_FILE_FLOW_ARM_OK);ROK(nvm_file_runtime_drop(c,3));
  scalar(c,2,251);ROK(nvm_file_runtime_service(c,b.imports[1],0,2,3));arm(c,3,NVM_FILE_FLOW_ARM_OK);ROK(nvm_file_runtime_drop(c,3));
  ROK(nvm_file_runtime_service(c,b.imports[2],0,NS,3));ROK(nvm_file_runtime_drop(c,3));
  ROK(nvm_file_runtime_service(c,b.imports[3],0,NS,3));ROK(nvm_file_runtime_take(c,3,NVM_FILE_FLOW_ARM_OK,4));
  CHECK(view(c,4).values[0]==251 && !view(c,4).values[1]);ROK(nvm_file_runtime_drop(c,4));
  ROK(nvm_file_runtime_end_reference(c,0));ROK(nvm_file_runtime_region_end(c));
  ROK(nvm_file_runtime_service(c,b.imports[4],NS,1,3));arm(c,3,NVM_FILE_FLOW_ARM_OK);ROK(nvm_file_runtime_drop(c,3));finish_ok(&c,251);
 }
}
static void faults(void){
 NvmFileNominalBindings b;NvmFileRuntime *c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);
 deny_open=true;unsigned before=open_attempts,success=opened;ROK(nvm_file_runtime_service(c,b.imports[0],NS,NS,0));deny_open=false;
 CHECK(open_attempts==before+1 && opened==success);arm(c,0,NVM_FILE_FLOW_ARM_ERROR);
 ROK(nvm_file_runtime_move(c,0,1));ROK(nvm_file_runtime_take(c,1,NVM_FILE_FLOW_ARM_ERROR,0));
 CHECK(view(c,0).values[0]==NL_FILE_IO && view(c,0).values[1]==EACCES);ROK(nvm_file_runtime_drop(c,0));finish_ok(&c,0);
 c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);reenter=c;file(c,b,1);ROK(nvm_file_runtime_region_begin(c));ROK(nvm_file_runtime_borrow(c,1,0));
 before=io_attempts;scalar(c,2,256);ROK(nvm_file_runtime_service(c,b.imports[1],0,2,3));arm(c,3,NVM_FILE_FLOW_ARM_ERROR);
 CHECK(io_attempts==before && view(c,3).values[0]==NL_FILE_ARGUMENT);ROK(nvm_file_runtime_drop(c,3));
 scalar(c,2,42);ROK(nvm_file_runtime_service(c,b.imports[1],0,2,3));ROK(nvm_file_runtime_drop(c,3));
 deny_seek=true;ROK(nvm_file_runtime_service(c,b.imports[2],0,NS,3));deny_seek=false;
 CHECK(view(c,3).values[1]==ESPIPE);ROK(nvm_file_runtime_drop(c,3));before=io_attempts;
 ROK(nvm_file_runtime_service(c,b.imports[3],0,NS,3));CHECK(io_attempts==before && view(c,3).values[0]==NL_FILE_DIRECTION);ROK(nvm_file_runtime_drop(c,3));
 ROK(nvm_file_runtime_service(c,b.imports[2],0,NS,3));ROK(nvm_file_runtime_drop(c,3));
 ROK(nvm_file_runtime_service(c,b.imports[3],0,NS,3));ROK(nvm_file_runtime_drop(c,3));before=io_attempts;
 scalar(c,2,23);ROK(nvm_file_runtime_service(c,b.imports[1],0,2,3));CHECK(io_attempts==before && view(c,3).values[0]==NL_FILE_DIRECTION);ROK(nvm_file_runtime_drop(c,3));
 ROK(nvm_file_runtime_end_reference(c,0));ROK(nvm_file_runtime_region_end(c));ROK(nvm_file_runtime_service(c,b.imports[4],NS,1,3));ROK(nvm_file_runtime_drop(c,3));
 CHECK(reentries>=5);reenter=NULL;finish_ok(&c,42);
 /* Failure of each of the three core allocations has no host acquisition. */
 for(int limit=0;limit<3;limit++){
  c=context(&b,NVM_FILE_RUNTIME_VM,false,false,false);before=open_attempts;size_t live=tracked_live;
  allocation_budget=limit;CHECK(nvm_file_runtime_begin(c)==NVM_FILE_RUNTIME_MEMORY);allocation_budget=-1;
  CHECK(tracked_live==live && open_attempts==before);NvmFileRuntimeReport r=finish_bad(&c,NVM_FILE_RUNTIME_MEMORY);CHECK(!r.acquired);
 }
 /* A cleanup error in the initializer suppresses entry, even if its Result
  * was explicitly consumed. I do not diagnose a stream that refused close. */
 c=context(&b,NVM_FILE_RUNTIME_VM,false,true,true);file(c,b,1);close_index=0;close_error[0]=EIO;
 ROK(nvm_file_runtime_service(c,b.imports[4],NS,1,2));CHECK(!view(c,1).initialized && view(c,2).values[5]==1);ROK(nvm_file_runtime_drop(c,2));
 CHECK(nvm_file_runtime_complete_root(c,NS)==NVM_FILE_RUNTIME_CLEANUP);uint32_t root=99;
 CHECK(nvm_file_runtime_current_root(c,&root) && root==3);before=open_attempts;
 CHECK(nvm_file_runtime_service(c,b.imports[0],NS,NS,0)==NVM_FILE_RUNTIME_CLEANUP && open_attempts==before);
 NvmFileRuntimeReport r=finish_bad(&c,NVM_FILE_RUNTIME_CLEANUP);CHECK(r.cleanup.cleanup_failures==1 && r.cleanup.first_cleanup.host_errno==EIO);
 memset(close_error,0,sizeof close_error);close_index=0;
 c=context(&b,NVM_FILE_RUNTIME_NATIVE,false,false,true);file(c,b,1);file(c,b,2);file(c,b,3);
 ROK(nvm_file_runtime_region_begin(c));ROK(nvm_file_runtime_borrow(c,3,0));ROK(nvm_file_runtime_bind_formal(c,0,4,1));
 ROK(nvm_file_runtime_site(c,0,0));CHECK(nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_ASSERT)==NVM_FILE_RUNTIME_ASSERT);
 close_index=0;close_error[0]=EIO;close_error[1]=ENOSPC;close_error[2]=EPIPE;reenter=c;
 r=finish_bad(&c,NVM_FILE_RUNTIME_ASSERT);reenter=NULL;
 CHECK(r.acquired && r.function==0 && r.instruction==0 && r.cleanup.cleanup_failures==3);
 CHECK(r.cleanup.first_cleanup.host_errno==EIO && r.cleanup.next_cleanup.host_errno==ENOSPC);
 memset(close_error,0,sizeof close_error);close_index=0;
 /* A clean entry cannot publish its staged scalar after earlier close error. */
 c=context(&b,NVM_FILE_RUNTIME_VM,false,false,true);file(c,b,1);close_error[0]=EIO;
 ROK(nvm_file_runtime_service(c,b.imports[4],NS,1,2));ROK(nvm_file_runtime_drop(c,2));scalar(c,0,91);ROK(nvm_file_runtime_complete_root(c,0));
 r=finish_bad(&c,NVM_FILE_RUNTIME_CLEANUP);CHECK(r.cleanup.cleanup_failures==1);memset(close_error,0,sizeof close_error);close_index=0;
}
static void allocation_controls(void){
 NvmFileNominalBindings b;size_t size;uint8_t *bytes=runtime_wire(&b,&size,false,false);
 size_t baseline=tracked_live,basebytes=tracked_bytes;unsigned failures=0,recovered=0;
 for(unsigned transient=0;transient<2;transient++){
  bool reached=false;single_failure=transient!=0;
  for(int limit=0;limit<4096;limit++){
   allocation_budget=limit;failed_calls=0;tracked_peak=tracked_bytes;unsigned opens=open_attempts;
   NvmFileRuntime *c=(NvmFileRuntime *)(uintptr_t)1;
   NvmFileRuntimeStatus status=nvm_file_runtime_create(bytes,size,NVM_FILE_RUNTIME_NATIVE,&c);allocation_budget=-1;
   CHECK(open_attempts==opens);
   if(status==NVM_FILE_RUNTIME_OK){
    CHECK(c && c!=(NvmFileRuntime *)(uintptr_t)1);NvmFileRuntimeStorage storage;CHECK(nvm_file_runtime_storage(c,&storage));
    CHECK(tracked_peak-basebytes<=storage.allocation_bound);ROK(nvm_file_runtime_begin(c));
    CHECK(tracked_bytes-basebytes<=storage.allocation_bound);finish_ok(&c,3);if(failed_calls)recovered++;
   }else{
    CHECK(c==(NvmFileRuntime *)(uintptr_t)1);CHECK(status==NVM_FILE_RUNTIME_MEMORY || status==NVM_FILE_RUNTIME_UNRESOLVED);failures++;
   }
   CHECK(tracked_live==baseline && tracked_bytes==basebytes);if(!failed_calls){reached=true;break;}
  }
  CHECK(reached);
 }
 single_failure=false;free(bytes);CHECK(failures && !tracked_live && !tracked_bytes);
 printf("I retain %u allocation refusals and %u recovered complete carriers\n",failures,recovered);
}
#endif
int main(void){
 FILE *sentinel=tmpfile();CHECK(sentinel);int sentinel_fd=fileno(sentinel);CHECK(sentinel_fd>=0);
 carrier_lifecycle();passive();invalid_and_partial();invalid_passive();scalar_and_limits();initializer();public_refusal();
#ifdef HOSTED_INSTRUMENT
 modeled_progress_errors();faults();allocation_controls();CHECK(!tracked_live && !tracked_bytes);
#endif
 empty_host();CHECK(fcntl(sentinel_fd,F_GETFD)>=0);CHECK(fclose(sentinel)==0);printf("PASS %u manual private File carrier checks; %u acquisition attempts, %u real opens, %u real closes; no File CODE/frame dispatch\n",checks,open_attempts,opened,closed);return 0;
}
