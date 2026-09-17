#define CALLER_RUNTIME_TEST
#include "test_caller_reference_analysis.c"
#include "nvm2c.h"
#include "../../src/nanovm/vm.h"
#include "../../src/runtime/callback_runtime.h"
int g_argc=0;char **g_argv=NULL;
static void write_artifact(NvmModule *m,const char *path) {
    NvmV2Module v2;size_t size;
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(bytes,1,size,f)==size);CHECK(!fclose(f));
    free(bytes);nvm_v2_module_free(&v2);
}
static void execute_module(NvmModule *m,int64_t expected,const char *dir,unsigned number,uint8_t tag) {
    if (tag!=TAG_INT) {
        NvmV2Layouts layouts={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&layouts)==NVM_V2_OK);
        layouts.items[0].fields[0].type_tag=tag;layouts.items[1].fields[0].type_tag=tag;
        CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&layouts);
        m->functions[0].result_tag=tag;m->ownership_data[20]=tag;
    }
    NvmVerifyResult valid=nvm_verify(m);
    if(!valid.ok)fprintf(stderr,"%s\n",valid.error_msg);
    CHECK(valid.ok);CHECK(nvm_verify_owned_module(m).ok);
    uint16_t max;CHECK(nvm_verify_function_max_stack(m,0,&max).ok && max<=256);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    for(unsigned iteration=0;iteration<20;iteration++) {
        NanoValue result=val_void();CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);
        CHECK(result.tag==tag);
        CHECK((tag==TAG_BOOL?(int64_t)result.as.boolean:tag==TAG_U8?(int64_t)result.as.u8:result.as.i64)==expected);
        CHECK(vm.stack_size==0 && vm.frame_count==0);
        CHECK(!vm.references.active && !vm.callee_references.active);
        CHECK(vm.heap.stats.num_objects==baseline);
    }
    vm_destroy(&vm);
    char error[256];char *source=nvm2c_emit(m,error,sizeof(error));
    if(!source)fprintf(stderr,"%s\n",error);
    CHECK(source);
    char path[1024];snprintf(path,sizeof(path),"%s/case%u.c",dir,number);
    FILE *f=fopen(path,"w");CHECK(f);CHECK(fputs(source,f)>=0);CHECK(!fclose(f));free(source);
    snprintf(path,sizeof(path),"%s/case%u.nvm",dir,number);write_artifact(m,path);
    printf("case %u %lld\n",number,(long long)expected);nvm_module_free(m);
}
static void refused(NvmModule *m,const char *dir,unsigned number) {
    CHECK(!nvm_verify_owned_module(m).ok);CHECK(!nvm_verify(m).ok);
    VmState vm;vm_init(&vm,m);NanoValue result=val_void();
    CHECK(vm_invoke(&vm,0,NULL,0,&result)!=VM_OK);vm_destroy(&vm);
    char error[256];CHECK(nvm2c_emit(m,error,sizeof(error))==NULL);
    char path[1024];snprintf(path,sizeof(path),"%s/refused%u.nvm",dir,number);write_artifact(m,path);
    nvm_module_free(m);
}
static void resume_caller(void) {
    char helper[8192];strcpy(helper,"REGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\n");
    for(unsigned i=0;i<1200;i++)strcat(helper,"NOP\n");
    strcat(helper,"PUSH_I64 42\nREF_SET 1 0\nREGION_END\nREF_GET 0 0\nRET");
    NvmModule *m=call_fixture("CALL_REF 1 0",helper);CHECK(nvm_verify(m).ok);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    NanoValue argument=val_void();
    CHECK(vm_call_function(&vm,1,&argument,1)!=VM_OK);
    NanoValue output=val_int(123);
    CHECK(vm_invoke(&vm,1,&argument,1,&output)!=VM_OK);
    CHECK(vm_invoke_callable(&vm,val_function(1),&argument,1,&output)!=VM_OK);
    CHECK(output.tag==TAG_INT && output.as.i64==123);
    CHECK(vm.frame_count==0 && !vm.references.active);
    vm.callbacks=nano_callback_runtime_create();CHECK(vm.callbacks);
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0]=(VmCallFrame){.fn_idx=0,.local_count=m->functions[0].local_count,.module=m};
    vm.stack_size=m->functions[0].local_count;
    VmTrap trap=vm_core_execute(&vm);CHECK(trap.type==TRAP_YIELD);
    CHECK(vm.frame_count==2 && vm.references.active && vm.callee_references.active);
    CHECK(vm.callee_references.slots[1].live && vm.callee_references.slots[1].origin_frame==0);
    CHECK(vm.callee_references.slots[1].origin_generation==vm.references.generation);
    CHECK(vm.callee_references.slots[1].root==5 && vm.callee_references.slots[1].path==0);
    uint64_t generation=vm.callee_references.generation;
    CHECK(vm_call_function(&vm,0,NULL,0)!=VM_OK);
    CHECK(vm_invoke(&vm,0,NULL,0,&output)!=VM_OK);
    CHECK(vm_invoke_callable(&vm,val_function(0),NULL,0,&output)!=VM_OK);
    CHECK(vm_execute(&vm)!=VM_OK);
    CHECK(vm.callee_references.generation==generation && vm.callee_references.slots[1].live);
    NanoValue *next=calloc(vm.stack_capacity*2,sizeof(*next));CHECK(next);
    memcpy(next,vm.stack,vm.stack_size*sizeof(*next));free(vm.stack);vm.stack=next;vm.stack_capacity*=2;
    do {trap=vm_core_execute(&vm);} while(trap.type==TRAP_YIELD);
    CHECK(trap.type==TRAP_NONE && !vm.references.active && !vm.callee_references.active);
    CHECK(vm.stack_size==1 && vm.stack[0].tag==TAG_INT && vm.stack[0].as.i64==74);
    CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);nvm_module_free(m);
}
int caller_runtime_cases(int argc,char **argv) {
    CHECK(argc==2);
    execute_module(call_fixture("CALL_REF 1 0","REF_GET 0 0\nRET"),42,argv[1],0,TAG_INT);
    execute_module(call_fixture("CALL_REF 1 0","PUSH_I64 42\nREF_SET 0 0\nREF_GET 0 0\nRET"),74,argv[1],1,TAG_INT);
    execute_module(call_fixture("CALL_REF 1 0\nPOP\nCALL_REF 1 0","REF_GET 0 0\nPUSH_I64 1\nADD\nREF_SET 0 0\nREF_GET 0 0\nRET"),44,argv[1],2,TAG_INT);
    execute_module(call_fixture("CALL_REF 1 0","REGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nPUSH_I64 50\nREF_SET 1 0\nREGION_END\nREF_GET 0 0\nRET"),82,argv[1],3,TAG_INT);
    execute_module(call_fixture_impl("CALL_REF 1 0","PUSH_I64 42\nREF_SET 0 0\nREF_GET 0 0\nRET",false),74,argv[1],4,TAG_INT);
    NvmModule *shared=call_fixture("CALL_REF 1 0","REGION_BEGIN\nREBORROW_SHARED 1 0\nREF_GET 1 0\nREGION_END\nRET");
    shared->ownership_data[105]=NVM_REFERENCE_SHARED;
    execute_module(shared,42,argv[1],5,TAG_INT);
    execute_module(call_fixture("CALL_REF 1 0\nPOP\nPUSH_I64 5\nREF_SET 0 0\nCALL_REF 1 0","REF_GET 0 0\nRET"),37,argv[1],6,TAG_INT);
    NvmModule *scalar_result=call_fixture("CALL_REF 1 0","PUSH_BOOL 1\nRET");
    scalar_result->functions[1].result_tag=TAG_BOOL;
    slot(scalar_result->ownership_data+96,TAG_BOOL,0,NVM_V2_NO_INDEX);
    execute_module(scalar_result,42,argv[1],7,TAG_INT);
    scalar_result=call_fixture("CALL_REF 1 0","PUSH_U8 255\nRET");
    scalar_result->functions[1].result_tag=TAG_U8;
    slot(scalar_result->ownership_data+96,TAG_U8,0,NVM_V2_NO_INDEX);
    execute_module(scalar_result,42,argv[1],8,TAG_INT);
    const char *invalid[]={"LOAD_LOCAL 0\nAGG_GET 0\nRET","OWN_MOVE_LOCAL 0\nRET","REGION_BEGIN\nREF_GET 0 0\nRET","CALL_REF 1 0\nRET"};
    for(unsigned i=0;i<4;i++)refused(call_fixture("CALL_REF 1 0",invalid[i]),argv[1],i);
    NvmModule *bad=call_fixture("CALL_REF 1 0","PUSH_I64 42\nREF_SET 0 0\nREF_GET 0 0\nRET");
    bad->ownership_data[105]=NVM_REFERENCE_SHARED;refused(bad,argv[1],4);
    refused(call_fixture("REGION_BEGIN\nREBORROW_SHARED 1 0\nCALL_REF 1 0\nREGION_END","REF_GET 0 0\nRET"),argv[1],5);
    refused(call_fixture("CALL_REF 0 0","REF_GET 0 0\nRET"),argv[1],6);
    refused(call_fixture("CALL_REF 1 1","REF_GET 0 0\nRET"),argv[1],7);
    bad=call_fixture("CALL_REF 1 0","REF_GET 0 0\nRET");
    slot(bad->ownership_data+104,TAG_STRUCT,2,2);refused(bad,argv[1],8);
    refused(call_fixture("CALL_REF 1 0","REGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nREF_GET 0 0\nREGION_END\nRET"),argv[1],9);
    refused(call_fixture("CALL_REF 1 0","PUSH_I64 1\nSTORE_LOCAL 0\nPUSH_I64 0\nRET"),argv[1],10);
    bad=call_fixture("CALL_REF 1 0","REF_GET 0 0\nRET");
    for(uint32_t offset=0;offset<bad->functions[0].code_length;) {
        DecodedInstruction instruction;uint32_t length=isa_decode(bad->code+offset,bad->code_size-offset,&instruction);CHECK(length);
        if(instruction.opcode==OP_BORROW_PATH_EXCLUSIVE){bad->code[offset]=OP_BORROW_PATH_SHARED;break;}
        offset+=length;
    }
    refused(bad,argv[1],11);
    resume_caller();
    printf("%u caller reference checks passed\n",checks);return 0;
}

#ifndef CALLER_ALLOC_TEST
int main(int argc,char **argv){return caller_runtime_cases(argc,argv);}
#endif
