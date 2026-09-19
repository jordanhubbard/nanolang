/* I decode the new family and reject execution; no File handler runs. */
#define main file_opcode_nominal_fixture_main
#include "test_file_nominal.c"
#undef main
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanoisa/nvm2llvm.h"
#include "../../src/nanoisa/mixed_samples_internal.h"
#include "../../src/nanovm/vm_decode.h"
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanovm/vm_ffi.h"
#include "../../src/runtime/ffi_loader.h"
#include "../../src/nanovirt/wrapper_gen.h"
int g_argc;char **g_argv;
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
static NvmModule *ordinary(void){AsmResult r={0};NvmModule *m=asm_assemble(".function main 0 2 0 int 1\nPUSH_I64 7\nRET\n.end\n.entry main\n",&r);CHECK(m);return m;}
static void code(NvmModule *m,const uint8_t *bytes,uint32_t n){free(m->code);m->code=malloc(n?n:1);CHECK(m->code);memcpy(m->code,bytes,n);m->code_size=m->code_capacity=n;m->functions[0].code_offset=0;m->functions[0].code_length=n;}
static void decode_refusal(NvmModule *m){VmDecodedFunction d={0};char error[VM_DECODE_ERROR_SIZE];CHECK(!vm_decode_function(m,0,&d,error));vm_decoded_function_free(&d);CHECK(!nvm_verify(m).ok);}
static void opcode_raw(void){
 static const uint8_t golden[6][7]={{0x91,4,3,2,1,255,255},{0x92,255,255,0,0,0,128},{0x93,255,255,1},{0x94,255,255},{0x95},{0x96,255,255}};
 static const unsigned sizes[]={7,7,4,3,1,3};
 static const char *const names[]={"FILE_SERVICE","FILE_RESULT_BRANCH","FILE_RESULT_TAKE","FILE_DROP_LOCAL","FILE_DROP_STACK","FILE_END_BORROW"};
 for(unsigned i=0;i<6;i++){
  uint8_t opcode=(uint8_t)(0x91+i);CHECK(isa_is_file_opcode(opcode));CHECK(isa_opcode_by_name(names[i])==opcode);CHECK(!strcmp(isa_get_info(opcode)->name,names[i]));
  DecodedInstruction d;CHECK(isa_decode(golden[i],sizes[i],&d)==sizes[i]);CHECK(d.opcode==opcode && d.byte_length==sizes[i]);
  uint8_t out[16];memset(out,0xa5,sizeof out);CHECK(isa_encode(&d,out,sizeof out)==sizes[i]);CHECK(!memcmp(out,golden[i],sizes[i]));for(unsigned k=sizes[i];k<16;k++)CHECK(out[k]==0xa5);
  for(unsigned n=0;n<sizes[i];n++){CHECK(isa_decode(golden[i],n,&d)==0);memset(out,0xa5,sizeof out);d.opcode=opcode;CHECK(isa_encode(&d,out,n)==0);for(unsigned k=0;k<16;k++)CHECK(out[k]==0xa5);CHECK(isa_code_has_file_instructions(golden[i],n)==(n!=0));}
  NvmModule *m=ordinary();code(m,golden[i],sizes[i]);CHECK(!nvm_service_bindings_present(m));CHECK(nvm_file_instructions_present(m) && nvm_service_execution_pending(m));
  CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)!=NVM_V2_OK);nvm_v2_module_free(&v);all_consumers(m);
  NvmMixedSamplesPlan *sentinel=(NvmMixedSamplesPlan *)(uintptr_t)1;CHECK(!nvm_mixed_samples_candidate(m));CHECK(nvm_mixed_samples_prepare(m,&sentinel).status==NVM_MIXED_SHAPE_UNRESOLVED && sentinel==(NvmMixedSamplesPlan *)(uintptr_t)1);CHECK(nvm_mixed_samples_admit(m,&sentinel).status==NVM_MIXED_SHAPE_UNRESOLVED && sentinel==(NvmMixedSamplesPlan *)(uintptr_t)1);
  if(sizes[i]>1){code(m,golden[i],1);CHECK(nvm_file_instructions_present(m));decode_refusal(m);all_consumers(m);}
  nvm_module_free(m);
 }
 CHECK(!isa_is_file_opcode(0x90) && !isa_is_file_opcode(0x97));CHECK(isa_get_info(0x97)==NULL);DecodedInstruction d;uint8_t unknown[]={0x97};CHECK(isa_decode(unknown,1,&d)==0);
 /* All six bytes remain data inside ordinary I64 and F64 immediates. */
 uint8_t integer[]={OP_PUSH_I64,0x91,0x92,0x93,0x94,0x95,0x96,0,0,OP_RET};
 NvmModule *m=ordinary();code(m,integer,sizeof integer);CHECK(!nvm_file_instructions_present(m) && !nvm_service_execution_pending(m));CHECK(nvm_verify(m).ok);
 uint32_t bytes=0;uint8_t *wire=nvm_serialize(m,&bytes);CHECK(wire && bytes);free(wire);char error[256];char *native=nvm2c_emit(m,error,sizeof error);CHECK(native);free(native);
 VmState vm;vm_init(&vm,m);NanoValue out=val_int(0);CHECK(vm_invoke(&vm,0,NULL,0,&out)==VM_OK && out.tag==TAG_INT && out.as.i64==INT64_C(0x969594939291));vm_destroy(&vm);nvm_module_free(m);
 integer[0]=OP_PUSH_F64;CHECK(!isa_code_has_file_instructions(integer,sizeof integer));
 /* Undecodable prior bytes/ranges are not invented service claims. I ask the
  * structural decoder/verifier to refuse, without running these bytecodes. */
 uint8_t malformed[]={0x97,OP_FILE_DROP_STACK};m=ordinary();code(m,malformed,sizeof malformed);CHECK(!nvm_file_instructions_present(m));decode_refusal(m);
 m->functions[0].code_offset=m->code_size+1;CHECK(!nvm_file_instructions_present(m));decode_refusal(m);nvm_module_free(m);
}
static void branches(void){
 uint8_t bytes[]={OP_FILE_RESULT_BRANCH,1,0,9,0,0,0,OP_PUSH_BOOL,1,OP_RET};NvmModule *m=ordinary();code(m,bytes,sizeof bytes);
 VmDecodedFunction d={0};char error[VM_DECODE_ERROR_SIZE];CHECK(vm_decode_function(m,0,&d,error));CHECK(d.instructions[0].resolved_target==9 && d.instructions[0].next_byte_offset==7);vm_decoded_function_free(&d);
 const uint32_t bad[]={1,8,10,UINT32_MAX,UINT32_C(0x80000000),UINT32_C(0x7fffffff)};
 for(unsigned i=0;i<sizeof bad/sizeof bad[0];i++){wr32(m->code+3,bad[i]);decode_refusal(m);}wr32(m->code+3,9);
 m->functions[0].code_length=7;decode_refusal(m);m->functions[0].code_length=sizeof bytes;nvm_module_free(m);
 /* Labels roundtrip only through the explicit raw, module-free formatter. */
 const char *source=".function main 0 2 0 int 1\nFILE_RESULT_BRANCH 1 error\nFILE_RESULT_TAKE 1 0\nRET\nerror:\nFILE_RESULT_TAKE 1 1\nRET\n.end\n.entry main\n";
 AsmResult result={0};m=asm_assemble_unverified(source,&result);CHECK(m);CHECK(!asm_assemble(source,&result));
 FILE *f=tmpfile();CHECK(f);disasm_function_styled(m->code,m->code_size,NULL,f,DISASM_STYLE_CANONICAL);CHECK(fflush(f)==0);long length=ftell(f);CHECK(length>0 && length<4096);rewind(f);char text[4096];CHECK(fread(text,1,(size_t)length,f)==(size_t)length);text[length]=0;CHECK(fclose(f)==0);CHECK(strstr(text,"FILE_RESULT_BRANCH 1 L") && strstr(text,"FILE_RESULT_TAKE"));
 char assembly[4608];int n=snprintf(assembly,sizeof assembly,".function main 0 2 0 int 1\n%s.end\n.entry main\n",text);CHECK(n>0 && (size_t)n<sizeof assembly);NvmModule *copy=asm_assemble_unverified(assembly,&result);CHECK(copy && copy->code_size==m->code_size && !memcmp(copy->code,m->code,m->code_size));all_consumers(m);nvm_module_free(copy);nvm_module_free(m);
 /* Repaired-only widened formatting of extreme targets, no execution. */
 bytes[0]=OP_FILE_RESULT_BRANCH;wr32(bytes+3,UINT32_C(0x7fffffff));f=tmpfile();CHECK(f);disasm_function_styled(bytes,sizeof bytes,NULL,f,DISASM_STYLE_CANONICAL);CHECK(ftell(f)>0);CHECK(fclose(f)==0);
}
static void wrapper_refusal(NvmModule *claimed,const uint8_t *wire,size_t size){
 const char *path=getenv("FILE_OPCODE_WRAPPER_OUTPUT");CHECK(path && size<=UINT32_MAX);
 FILE *f=fopen(path,"wb");CHECK(f);CHECK(fputs("preserved",f)>=0);CHECK(fclose(f)==0);
 NvmModule *clean=ordinary();NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(clean,&v)==NVM_V2_OK);size_t n=0;CHECK(nvm_v2_module_serialize(&v,NULL,0,&n)==NVM_V2_OK);CHECK(n<=UINT32_MAX);uint8_t *bytes=malloc(n);CHECK(bytes);CHECK(nvm_v2_module_serialize(&v,bytes,n,&n)==NVM_V2_OK);
 CHECK(!wrapper_generate(clean,wire,(uint32_t)size,path,NULL,NULL,false));
 CHECK(!wrapper_generate(claimed,bytes,(uint32_t)n,path,NULL,NULL,false));
 CHECK(!wrapper_generate_daemon(wire,(uint32_t)size,path,false));
 CHECK(!wrapper_generate(clean,wire,1,path,NULL,NULL,false));
 CHECK(!wrapper_generate_daemon(wire,1,path,false));
 f=fopen(path,"rb");CHECK(f);char out[16]={0};CHECK(fread(out,1,sizeof out,f)==9 && !memcmp(out,"preserved",9));CHECK(fclose(f)==0);
 free(bytes);nvm_v2_module_free(&v);nvm_module_free(clean);
}
static void transport(void){
 for(unsigned permutation=0;permutation<2;permutation++){
  NvmFileNominalBindings b;NvmModule *m=fixture(permutation!=0,&b);uint8_t bytes[]={OP_FILE_SERVICE,0,0,0,0,255,255,OP_RET};wr32(bytes+1,b.imports[0]);code(m,bytes,sizeof bytes);
  CHECK(nvm_service_bindings_validate(m)==NVM_V2_OK);all_consumers(m);
  NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);CHECK(nvm_v2_file_instructions_present(&v));CHECK(nvm_v2_service_bindings_validate(&v)==NVM_V2_OK);
  size_t n=0;CHECK(nvm_v2_module_serialize(&v,NULL,0,&n)==NVM_V2_OK);uint8_t *wire=malloc(n);CHECK(wire);CHECK(nvm_v2_module_serialize(&v,wire,n,&n)==NVM_V2_OK);
  wrapper_refusal(m,wire,n);
  NvmV2Header h;CHECK(nvm_v2_read_header(wire,n,&h)==NVM_V2_OK);uint32_t required=NVM_V2_FEATURE_FFI|NVM_V2_FEATURE_OWNERSHIP|NVM_V2_FEATURE_RETAINED_LAYOUTS|NVM_V2_FEATURE_SERVICE_BINDINGS;CHECK((h.feature_bits&required)==required);
  const char *path=getenv("FILE_OPCODE_WIRE");if(path){FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(wire,1,n,f)==n);CHECK(fclose(f)==0);}
  NvmV2Module decoded={0};CHECK(nvm_v2_module_deserialize(wire,n,&decoded)==NVM_V2_OK);NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);CHECK(copy->code_size==sizeof bytes && !memcmp(copy->code,bytes,sizeof bytes));all_consumers(copy);nvm_module_free(copy);nvm_v2_module_free(&decoded);free(wire);
  const uint8_t *saved=v.service_data;uint32_t saved_size=v.service_size;v.service_data=NULL;v.service_size=0;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);copy=(NvmModule *)(uintptr_t)1;CHECK(nvm_v2_to_nvm_module(&v,&copy)!=NVM_V2_OK && copy==NULL);
  NvmServiceBindings old;memcpy(old.imports,b.imports,sizeof old.imports);uint8_t v1[56];size_t used=0;CHECK(nvm_service_bindings_encode(&old,v1,sizeof v1,&used)==NVM_SERVICE_OK);v.service_data=v1;v.service_size=56;CHECK(nvm_v2_service_bindings_validate(&v)!=NVM_V2_OK);v.service_data=saved;v.service_size=saved_size;nvm_v2_module_free(&v);
  uint8_t *owned=m->service_data;uint32_t owned_size=m->service_size;m->service_data=v1;m->service_size=56;CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);m->service_data=owned;m->service_size=owned_size;nvm_module_free(m);
 }
}
int main(void){opcode_raw();branches();transport();printf("PASS %u File encoding/refusal checks; no service handler or CFG certificate\n",checks);return 0;}
