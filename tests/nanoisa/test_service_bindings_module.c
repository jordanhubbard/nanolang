#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "service_bindings_module.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "nvm2c.h"
#include "nvm2llvm.h"
#include "nanoisa.h"
#include "../../src/nsi.h"
#include "../../src/nanovm/vm_ffi.h"

static unsigned checks;
#define CHECK(x) do { checks++;assert(x); } while(0)
#ifdef SERVICE_ALLOC_TEST
static int budget=-1;static unsigned attempts;
void *service_test_malloc(size_t n){attempts++;if(budget==0)return NULL;if(budget>0)budget--;return malloc(n);}
void *service_test_calloc(size_t n,size_t z){attempts++;if(budget==0)return NULL;if(budget>0)budget--;return calloc(n,z);}
void *service_test_realloc(void *p,size_t n){attempts++;if(budget==0)return NULL;if(budget>0)budget--;return realloc(p,n);}
#endif
static const char *const ids[]={"nsi:nanolang/filesystem#temp","nsi:nanolang/filesystem#write_byte",
 "nsi:nanolang/filesystem#rewind","nsi:nanolang/filesystem#read_byte","nsi:nanolang/filesystem#close"};
static NvmModule *base(void) {
    AsmResult result={0};
    NvmModule *m=asm_assemble(".function main 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n.entry main\n",&result);
    CHECK(m);return m;
}
static NvmModule *imports(bool reverse,NvmServiceBindings *binding) {
    NvmModule *m=base();uint32_t module=nvm_add_string(m,"nsi:nanolang/filesystem",(uint32_t)strlen("nsi:nanolang/filesystem"));
    CHECK(module!=UINT32_MAX);
    for(unsigned n=0;n<5;n++) {
        unsigned op=reverse?4-n:n;uint32_t name=nvm_add_string(m,ids[op],(uint32_t)strlen(ids[op]));
        uint8_t tags[]={TAG_STRUCT,TAG_INT};unsigned count=op==0?0:op==1?2:1;
        uint32_t i=nvm_add_import(m,module,name,(uint16_t)count,TAG_UNION,tags);
        CHECK(i==n);m->imports[i].kind=NVM_IMPORT_SERVICE;binding->imports[op]=i;
    }
    return m;
}
static void consumers(NvmModule *m) {
    CHECK(nvm_service_bindings_present(m));
    CHECK(!nvm_verify(m).ok);CHECK(!nvm_verify_owned_module(m).ok);
    CHECK(!nvm_verify_function(m,0).ok);uint16_t depth=99;
    CHECK(!nvm_verify_function_max_stack(m,0,&depth).ok);CHECK(depth==99);
    for(unsigned p=0;p<=NVM_PROFILE_CLOSED_MANAGED_STRINGS;p++) CHECK(!nvm_verify_profile(m,(NvmVerifyProfile)p).ok);
    NvmModule *ordinary=base();const NvmModule *linked[]={m};
    CHECK(!nvm_verify_linked(ordinary,linked,1).ok);nvm_module_free(ordinary);
    char error[512];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
    CHECK(disasm_module(m)==NULL);CHECK(nanoisa_print(m)==NULL);CHECK(nanoisa_pretty_print(m)==NULL);
    FILE *f=tmpfile();CHECK(f);fputs("keep",f);long before=ftell(f);
    disasm_module_to_file(m,f);disasm_function(m->code,m->code_size,m,f);CHECK(ftell(f)==before);
    CHECK(!nvm2llvm_emit_target(m,f,error,sizeof error,"main",NVM_LLVM_NATIVE));CHECK(ftell(f)==before);
    CHECK(!nvm2llvm_emit_target(m,f,error,sizeof error,"main",NVM_LLVM_WASM32));CHECK(ftell(f)==before);fclose(f);
    uint32_t size=123;CHECK(nvm_serialize(m,&size)==NULL);CHECK(size==0);
    VmState vm;vm_init(&vm,m);NanoValue output=val_int(123);
    CHECK(vm_invoke(&vm,0,NULL,0,&output)!=VM_OK);CHECK(output.tag==TAG_INT && output.as.i64==123);
    CHECK(vm_call_function(&vm,0,NULL,0)!=VM_OK);CHECK(vm_execute(&vm)!=VM_OK);
    CHECK(vm_core_execute(&vm).type==TRAP_ERROR);CHECK(vm.stack_size==0 && vm.frame_count==0);
    CHECK(!vm_ffi_load_import(m,0));
    CHECK(!vm_ffi_call(m,0,NULL,0,&output,&vm.heap,error,sizeof error));
    CHECK(!vm_ffi_call_vm(&vm,m,0,NULL,0,&output,error,sizeof error));
    CHECK(!vm_ffi_cop_start(&vm,m));
    CHECK(!vm_ffi_call_cop(&vm,m,0,NULL,0,&output,&vm.heap,error,sizeof error));
    CHECK(!vm_ffi_call_cop_batch(&vm,m,NULL,0,&output,&vm.heap,error,sizeof error));
    CHECK(output.tag==TAG_INT && output.as.i64==123);CHECK(vm.cop_pid<=0);vm_destroy(&vm);
}
static void invalid_wire(uint8_t *wire,size_t bytes) {
    NvmV2Header h;CHECK(nvm_v2_read_header(wire,bytes,&h)==NVM_V2_OK);
    h.checksum=nvm_crc32(wire+NVM_V2_HEADER_SIZE,(uint32_t)(bytes-NVM_V2_HEADER_SIZE));nvm_v2_write_header(wire,&h);
    NvmV2Module out={0};CHECK(nvm_v2_module_deserialize(wire,bytes,&out)!=NVM_V2_OK);nvm_v2_module_free(&out);
}
static void roundtrip(const NlFilePlan *plan,bool reverse) {
    NvmServiceBindings binding;NvmModule *m=imports(reverse,&binding);
    CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);consumers(m);
    CHECK(nvm_service_bindings_attach(m,plan,&binding)==NVM_V2_OK);
    uint8_t exact[56];memcpy(exact,m->service_data,56);uint8_t *original=m->service_data;
    CHECK(nvm_service_bindings_attach(m,plan,&binding)==NVM_V2_OK && m->service_data==original);
    NvmServiceBindings changed=binding;uint32_t x=changed.imports[0];changed.imports[0]=changed.imports[1];changed.imports[1]=x;
    CHECK(nvm_service_bindings_attach(m,plan,&changed)!=NVM_V2_OK);CHECK(m->service_data==original && !memcmp(exact,original,56));
    consumers(m);
    NvmV2Module v2={0};CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(v2.service_data==m->service_data);size_t bytes=0;CHECK(nvm_v2_module_serialize(&v2,NULL,0,&bytes)==NVM_V2_OK);
    uint8_t *wire=malloc(bytes),*mut=malloc(bytes);CHECK(wire && mut);
    CHECK(nvm_v2_module_serialize(&v2,wire,bytes,&bytes)==NVM_V2_OK);
    NvmV2Header header;CHECK(nvm_v2_read_header(wire,bytes,&header)==NVM_V2_OK);
    CHECK((header.feature_bits&(NVM_V2_FEATURE_FFI|NVM_V2_FEATURE_SERVICE_BINDINGS))==(NVM_V2_FEATURE_FFI|NVM_V2_FEATURE_SERVICE_BINDINGS));
    NvmV2SectionEntry service={0},im={0};
    for(uint32_t i=0;i<header.section_count;i++){NvmV2SectionEntry e;CHECK(nvm_v2_read_section(wire,bytes,&header,i,&e)==NVM_V2_OK);if(e.type==NVM_V2_SECTION_SERVICE_BINDINGS)service=e;if(e.type==NVM_V2_SECTION_IMPORTS)im=e;}
    CHECK(service.size==56 && !memcmp(wire+service.offset,exact,56));CHECK(im.size==84);
    NvmV2Module decoded={0};CHECK(nvm_v2_module_deserialize(wire,bytes,&decoded)==NVM_V2_OK);
    CHECK(decoded.service_data==wire+service.offset);NvmModule *copy=NULL;
    CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);CHECK(copy->service_data!=decoded.service_data);
    nvm_v2_module_free(&decoded);memset(wire+service.offset,0,56);
    CHECK(!memcmp(copy->service_data,exact,56));CHECK(nvm_service_bindings_validate(copy)==NVM_V2_OK);consumers(copy);
    memcpy(wire+service.offset,exact,56);
    for(unsigned bit=0;bit<2;bit++) {memcpy(mut,wire,bytes);NvmV2Header h=header;h.feature_bits&=~(bit?NVM_V2_FEATURE_FFI:NVM_V2_FEATURE_SERVICE_BINDINGS);nvm_v2_write_header(mut,&h);invalid_wire(mut,bytes);}
    for(unsigned i=0;i<5;i++){memcpy(mut,wire,bytes);mut[im.offset+4+16*i+12]=NVM_V2_IMPORT_FFI;invalid_wire(mut,bytes);}
    memcpy(mut,wire,bytes);mut[service.offset+4]=0;invalid_wire(mut,bytes);
    memcpy(mut,wire,bytes);mut[service.offset+20]=5;invalid_wire(mut,bytes);
    for(unsigned op=0;op<5;op++) {
        NvmImportEntry *entry=&m->imports[binding.imports[op]];uint8_t old=entry->return_type;entry->return_type=TAG_INT;
        CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);entry->return_type=old;
        uint32_t oldname=entry->function_name_idx;entry->function_name_idx=entry->module_name_idx;
        CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);entry->function_name_idx=oldname;
        if(entry->param_count){uint8_t *tag=m->import_param_types[binding.imports[op]];old=*tag;*tag=TAG_INT;CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);*tag=old;}
    }
    m->module_ref_count=1;CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);m->module_ref_count=0;
    m->callback_contract_count=1;CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);m->callback_contract_count=0;
    char *name=m->strings[m->imports[0].function_name_idx];char saved=name[3];name[3]=0;
    CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);name[3]=saved;
    uint32_t length=m->string_lengths[m->imports[0].module_name_idx];
    m->string_lengths[m->imports[0].module_name_idx]=length-1;
    CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);m->string_lengths[m->imports[0].module_name_idx]=length;
    NvmV2Signature *sig=&v2.signatures.items[v2.imports.items[0].signature_idx];uint16_t result_count=sig->result_count;
    sig->result_count=2;CHECK(nvm_v2_service_bindings_validate(&v2)!=NVM_V2_OK);sig->result_count=result_count;
    NvmModule partial=*m;partial.service_data=NULL;consumers(&partial);
    partial=*m;partial.service_size=0;consumers(&partial);
    partial=*m;partial.import_count=0;consumers(&partial);
    uint8_t output[8];memset(output,0xa5,sizeof output);size_t sentinel=777;
    NvmV2Module malformed=v2;malformed.service_size=0;
    CHECK(nvm_v2_module_serialize(&malformed,output,sizeof output,&sentinel)!=NVM_V2_OK);
    CHECK(sentinel==777 && output[0]==0xa5);NvmModule *absent=(void *)1;
    CHECK(nvm_v2_to_nvm_module(&malformed,&absent)!=NVM_V2_OK && absent==NULL);
    free(wire);free(mut);nvm_v2_module_free(&v2);nvm_module_free(m);nvm_module_free(copy);
}
#ifdef SERVICE_ALLOC_TEST
static void allocation(const NlFilePlan *plan) {
    NvmServiceBindings binding;NvmModule *m=imports(false,&binding);
    budget=0;CHECK(nvm_service_bindings_attach(m,plan,&binding)==NVM_V2_ERR_TRUNCATED);budget=-1;
    CHECK(!m->service_data && !m->service_size);CHECK(nvm_service_bindings_attach(m,plan,&binding)==NVM_V2_OK);
    NvmV2Module v2={0};attempts=0;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);unsigned count=attempts;nvm_v2_module_free(&v2);CHECK(count>0);
    for(unsigned i=0;i<count;i++){budget=(int)i;NvmV2Result r=nvm_v2_from_nvm_module(m,&v2);budget=-1;CHECK(r!=NVM_V2_OK);nvm_v2_module_free(&v2);CHECK(nvm_service_bindings_validate(m)==NVM_V2_OK);}
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);NvmModule *copy=NULL;attempts=0;
    CHECK(nvm_v2_to_nvm_module(&v2,&copy)==NVM_V2_OK);count=attempts;nvm_module_free(copy);CHECK(count>0);
    for(unsigned i=0;i<count;i++){copy=(void *)1;budget=(int)i;NvmV2Result r=nvm_v2_to_nvm_module(&v2,&copy);budget=-1;CHECK(r!=NVM_V2_OK && copy==NULL);CHECK(nvm_v2_service_bindings_validate(&v2)==NVM_V2_OK);}
    CHECK(nvm_v2_to_nvm_module(&v2,&copy)==NVM_V2_OK);nvm_module_free(copy);nvm_v2_module_free(&v2);nvm_module_free(m);
}
#endif
int main(int argc,char **argv) {
    CHECK(argc==2);NlNsi *doc=nl_nsi_load_path(argv[1]);CHECK(doc);NlFilePlan *plan=NULL;
    CHECK(nl_file_plan_build(doc,&plan)==NL_FILE_PLAN_OK);nl_nsi_free(doc);
    roundtrip(plan,false);roundtrip(plan,true);
#ifdef SERVICE_ALLOC_TEST
    allocation(plan);
#endif
    NvmModule *ordinary=base();CHECK(nvm_verify(ordinary).ok);CHECK(!nvm_service_bindings_present(ordinary));
    NanoisaErr error;uint32_t bytes=0;uint8_t *wire=nanoisa_save_bytes(ordinary,&bytes,&error);CHECK(wire);
    NvmModule *again=nanoisa_load_bytes(wire,bytes,&error);CHECK(again && nvm_verify(again).ok);free(wire);nvm_module_free(again);nvm_module_free(ordinary);
    ordinary=base();VmState vm;vm_init(&vm,ordinary);NanoValue result=val_void();
    CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK && result.tag==TAG_INT && result.as.i64==7);vm_destroy(&vm);
    char diagnostic[256];char *c=nvm2c_emit(ordinary,diagnostic,sizeof diagnostic);CHECK(c);free(c);
    char *text=disasm_module(ordinary);CHECK(text);free(text);FILE *ir=tmpfile();CHECK(ir);
    CHECK(nvm2llvm_emit(ordinary,ir,diagnostic,sizeof diagnostic));CHECK(ftell(ir)>0);fclose(ir);nvm_module_free(ordinary);
    nl_file_plan_free(plan);printf("service transport: %u checks passed\n",checks);return 0;
}
