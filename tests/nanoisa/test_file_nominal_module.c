/* I reuse the exact pinned catalog/module fixture, not a second shape oracle. */
#define main private_nominal_controls_main
#include "test_file_nominal.c"
#undef main
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanoisa/nvm2llvm.h"
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanovm/vm_ffi.h"
#include "../../src/runtime/ffi_loader.h"
#include "../../src/nsi.h"
int g_argc;char **g_argv;
#ifdef SERVICE_ALLOC_TEST
static int budget=-1;static unsigned attempts;
void *service_test_malloc(size_t n){attempts++;if(budget==0)return NULL;if(budget>0)budget--;return malloc(n);}
void *service_test_calloc(size_t n,size_t z){attempts++;if(budget==0)return NULL;if(budget>0)budget--;return calloc(n,z);}
void *service_test_realloc(void *p,size_t n){attempts++;if(budget==0)return NULL;if(budget>0)budget--;return realloc(p,n);}
#endif
static void all_consumers(NvmModule *m){
 CHECK(!nvm_verify(m).ok && !nvm_verify_owned_module(m).ok && !nvm_verify_function(m,0).ok);
 uint16_t depth=99;CHECK(!nvm_verify_function_max_stack(m,0,&depth).ok && depth==99);
 for(unsigned p=0;p<=NVM_PROFILE_CLOSED_MANAGED_STRINGS;p++)CHECK(!nvm_verify_profile(m,(NvmVerifyProfile)p).ok);
 char err[512];CHECK(nvm2c_emit(m,err,sizeof err)==NULL);CHECK(disasm_module(m)==NULL);CHECK(nanoisa_print(m)==NULL);CHECK(nanoisa_pretty_print(m)==NULL);
 FILE *f=tmpfile();CHECK(f);CHECK(fputs("keep",f)>=0);long pos=ftell(f);disasm_module_to_file(m,f);disasm_function(m->code,m->code_size,m,f);CHECK(ftell(f)==pos);
 CHECK(!nvm2llvm_emit_target(m,f,err,sizeof err,"main",NVM_LLVM_NATIVE) && ftell(f)==pos);
 CHECK(!nvm2llvm_emit_target(m,f,err,sizeof err,"main",NVM_LLVM_WASM32) && ftell(f)==pos);CHECK(fclose(f)==0);
 uint32_t bytes=123;CHECK(nvm_serialize(m,&bytes)==NULL && bytes==0);
 VmState vm;vm_init(&vm,m);NanoValue output=val_int(123);
 CHECK(vm_invoke(&vm,0,NULL,0,&output)!=VM_OK && output.tag==TAG_INT && output.as.i64==123);
 CHECK(vm_call_function(&vm,0,NULL,0)!=VM_OK);CHECK(vm_execute(&vm)!=VM_OK);CHECK(vm_core_execute(&vm).type==TRAP_ERROR);
 CHECK(vm_invoke_callable(&vm,val_function(0),NULL,0,&output)!=VM_OK);CHECK(vm.stack_size==0 && vm.frame_count==0);
 bool loader=ffi_loader_is_initialized();CHECK(!vm_ffi_load_import(m,0));
 CHECK(!vm_ffi_call(m,0,NULL,0,&output,&vm.heap,err,sizeof err));CHECK(!vm_ffi_call_vm(&vm,m,0,NULL,0,&output,err,sizeof err));
 CHECK(!vm_ffi_cop_start(&vm,m));CHECK(!vm_ffi_call_cop(&vm,m,0,NULL,0,&output,&vm.heap,err,sizeof err));CHECK(!vm_ffi_call_cop_batch(&vm,m,NULL,0,&output,&vm.heap,err,sizeof err));
 CHECK(output.tag==TAG_INT && output.as.i64==123 && vm.cop_pid<=0 && ffi_loader_is_initialized()==loader);vm_destroy(&vm);
}
static void same_facts(NvmModule *a,NvmModule *b){
 CHECK(a->service_size==b->service_size && !memcmp(a->service_data,b->service_data,a->service_size));
 CHECK(a->ownership_size==b->ownership_size && !memcmp(a->ownership_data,b->ownership_data,a->ownership_size));
 CHECK(a->layout_size==b->layout_size && !memcmp(a->layout_data,b->layout_data,a->layout_size));
 NvmFileNominalPlan *x=NULL,*y=NULL;CHECK(nvm_file_nominal_plan(a,&x)==NVM_FILE_NOMINAL_DESCRIBED);CHECK(nvm_file_nominal_plan(b,&y)==NVM_FILE_NOMINAL_DESCRIBED);
 for(unsigned i=0;i<8;i++){NvmFileNominalLayout p,q;CHECK(nvm_file_nominal_type(x,i,&p) && nvm_file_nominal_type(y,i,&q));CHECK(p.global_index==q.global_index && p.source_ordinal==q.source_ordinal && p.ownership_flags==q.ownership_flags && p.category==q.category);}
 nvm_file_nominal_plan_free(x);nvm_file_nominal_plan_free(y);
}
static void wire_refused(uint8_t *wire,size_t bytes){
 NvmV2Header h;CHECK(nvm_v2_read_header(wire,bytes,&h)==NVM_V2_OK);h.checksum=nvm_crc32(wire+NVM_V2_HEADER_SIZE,(uint32_t)(bytes-NVM_V2_HEADER_SIZE));nvm_v2_write_header(wire,&h);
 NvmV2Module out={0};CHECK(nvm_v2_module_deserialize(wire,bytes,&out)!=NVM_V2_OK);nvm_v2_module_free(&out);
}
static void transport(const NlFilePlan *catalog,bool permute){
 NvmFileNominalBindings b;NvmModule *m=fixture(permute,&b);uint8_t *original=m->service_data;uint8_t exact[120];memcpy(exact,original,120);
 CHECK(nvm_file_nominal_attach(m,catalog,&b)==NVM_V2_OK && m->service_data==original);
 NvmFileNominalBindings bad=b;bad.layouts[0]=bad.layouts[1];CHECK(nvm_file_nominal_attach(m,catalog,&bad)!=NVM_V2_OK && m->service_data==original && !memcmp(exact,original,120));
 m->service_data=NULL;m->service_size=0;CHECK(nvm_file_nominal_attach(m,catalog,&b)==NVM_V2_OK);free(original);all_consumers(m);
 NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);CHECK(v.service_data==m->service_data && v.ownership_data==m->ownership_data);CHECK(v.functions.items[0].max_stack==0);
 CHECK(nvm_v2_service_bindings_validate(&v)==NVM_V2_OK);
 uint32_t old_count=v.layouts.count;v.layouts.count=0;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.layouts.count=old_count;
 const uint8_t *old_owner=v.ownership_data;v.ownership_data=NULL;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.ownership_data=old_owner;
 uint32_t old_service_size=v.service_size;v.service_size=0;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.service_size=old_service_size;
 uint32_t sig=v.functions.items[0].signature_idx;v.functions.items[0].signature_idx=v.signatures.count;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.functions.items[0].signature_idx=sig;
 v.functions.items[0].local_count++;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.functions.items[0].local_count--;
 const uint8_t *old_params=v.signatures.items[sig].param_tags;v.signatures.items[sig].param_tags=NULL;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.signatures.items[sig].param_tags=old_params;
 uint8_t old_kind=v.imports.items[0].kind;v.imports.items[0].kind=NVM_V2_IMPORT_FFI;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.imports.items[0].kind=old_kind;
 v.links.count=1;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.links.count=0;
 NvmV2LayoutField *old_fields=v.layouts.items[b.layouts[1]].fields;v.layouts.items[b.layouts[1]].fields=NULL;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.layouts.items[b.layouts[1]].fields=old_fields;
 CHECK(nvm_v2_service_bindings_validate(&v)==NVM_V2_OK);
 size_t bytes=0;CHECK(nvm_v2_module_serialize(&v,NULL,0,&bytes)==NVM_V2_OK);uint8_t *wire=malloc(bytes),*changed=malloc(bytes);CHECK(wire && changed);
 CHECK(nvm_v2_module_serialize(&v,wire,bytes,&bytes)==NVM_V2_OK);
 const char *path=getenv("NOMINAL_MODULE_WIRE");if(path){FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(wire,1,bytes,f)==bytes);CHECK(fclose(f)==0);}
 NvmV2Header h;CHECK(nvm_v2_read_header(wire,bytes,&h)==NVM_V2_OK);
 const uint32_t required=NVM_V2_FEATURE_FFI|NVM_V2_FEATURE_RETAINED_LAYOUTS|NVM_V2_FEATURE_OWNERSHIP|NVM_V2_FEATURE_SERVICE_BINDINGS;
 CHECK((h.feature_bits&required)==required);
 for(unsigned bit=0;bit<32;bit++)if(required&(UINT32_C(1)<<bit)){memcpy(changed,wire,bytes);NvmV2Header badh=h;badh.feature_bits&=~(UINT32_C(1)<<bit);nvm_v2_write_header(changed,&badh);wire_refused(changed,bytes);}
 for(uint32_t i=0;i<h.section_count;i++){
  NvmV2SectionEntry section;CHECK(nvm_v2_read_section(wire,bytes,&h,i,&section)==NVM_V2_OK);
  if(section.type==NVM_V2_SECTION_SERVICE_BINDINGS || section.type==NVM_V2_SECTION_OWNERSHIP){
   memcpy(changed,wire,bytes);changed[section.offset]^=0x40;wire_refused(changed,bytes);
  }
 }
 NvmV2Module decoded={0};CHECK(nvm_v2_module_deserialize(wire,bytes,&decoded)==NVM_V2_OK);NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);same_facts(m,copy);all_consumers(copy);
 CHECK(copy->service_data!=m->service_data && copy->ownership_data!=m->ownership_data);
 nvm_v2_module_free(&decoded);memset(wire,0,bytes);free(wire);free(changed);nvm_v2_module_free(&v);nvm_module_free(m);
 NvmFileNominalPlan *proof=NULL;CHECK(nvm_file_nominal_plan(copy,&proof)==NVM_FILE_NOMINAL_DESCRIBED);nvm_file_nominal_plan_free(proof);all_consumers(copy);nvm_module_free(copy);
}
#ifdef SERVICE_ALLOC_TEST
static void allocation(const NlFilePlan *catalog){
 NvmFileNominalBindings b;NvmModule *m=fixture(false,&b);free(m->service_data);m->service_data=NULL;m->service_size=0;
 bool success=false;
 for(int limit=0;limit<128;limit++){
  budget=limit;attempts=0;NvmV2Result r=nvm_file_nominal_attach(m,catalog,&b);budget=-1;
  if(r==NVM_V2_OK){success=true;CHECK(m->service_data && m->service_size==120);break;}
  CHECK(m->service_data==NULL && m->service_size==0 && attempts>0);
 }CHECK(success);
 success=false;
 for(int limit=0;limit<256;limit++){
  NvmV2Module v={0};budget=limit;attempts=0;NvmV2Result r=nvm_v2_from_nvm_module(m,&v);budget=-1;
  if(r==NVM_V2_OK){success=true;nvm_v2_module_free(&v);break;}
  CHECK(attempts>0);nvm_v2_module_free(&v);CHECK(nvm_service_bindings_validate(m)==NVM_V2_OK);
 }CHECK(success);
 NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);success=false;
 for(int limit=0;limit<256;limit++){
  budget=limit;attempts=0;NvmV2Result r=nvm_v2_service_bindings_validate(&v);budget=-1;
  if(r==NVM_V2_OK){success=true;break;}CHECK(attempts>0);
 }CHECK(success);success=false;
 for(int limit=0;limit<512;limit++){
  NvmModule *out=m;budget=limit;attempts=0;NvmV2Result r=nvm_v2_to_nvm_module(&v,&out);budget=-1;
  if(r==NVM_V2_OK){success=true;same_facts(m,out);nvm_module_free(out);break;}
  CHECK(out==NULL && attempts>0);
 }CHECK(success);nvm_v2_module_free(&v);nvm_module_free(m);
}
#endif
int main(int argc,char **argv){g_argc=argc;g_argv=argv;CHECK(argc==2);
 NlNsi *doc=nl_nsi_load_path(argv[1]);CHECK(doc);NlFilePlan *catalog=NULL;CHECK(nl_file_plan_build(doc,&catalog)==NL_FILE_PLAN_OK);nl_nsi_free(doc);
 transport(catalog,false);transport(catalog,true);
#ifdef SERVICE_ALLOC_TEST
 allocation(catalog);
#endif
 nl_file_plan_free(catalog);printf("nominal service transport: %u checks passed; no host execution\n",checks);return 0;}
