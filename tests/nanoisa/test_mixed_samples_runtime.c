/* I execute only modules accepted by the complete public conjunction. */
#define main previous_shape_fixture_main
#include "test_mixed_float_proof.c"
#undef main
#include "nvm2c.h"
#include "disassembler.h"
#include "../../src/nanovm/vm.h"
int g_argc=0;char **g_argv=NULL;
static const char *consume=".function close 1 1 0 int 1\nOWN_UNPACK_LOCAL 0\nRET\n.end\n.parameters 1 struct\n";
static NvmModule *runtime_fixture(unsigned index) {
    const char *operations[]={
        "LOAD_LOCAL 1\nPUSH_F64 2.5\nARR_PUSH\nSTORE_LOCAL 2\n"
        "LOAD_LOCAL 2\nPUSH_I64 0\nPUSH_F64 9.5\nARR_SET\nPOP\n"
        "LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 2\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nPUSH_F64 9.5\nF64_EQ\nASSERT\n"
        "ARR_NEW 3\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 1\nPUSH_I64 1\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nASSERT\n"
        "LOAD_LOCAL 1\nPUSH_I64 -1\nARR_GET\nPUSH_VOID\nEQ\nASSERT\n",
        "LOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nPUSH_F64 1.0\nF64_ADD\nPOP\n",
        "PUSH_F64 1.0\nLOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nF64_SUB\nPOP\n",
        "LOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nLOAD_LOCAL 1\nPUSH_I64 -1\nARR_GET\nF64_EQ\nPOP\n",
        "PUSH_BOOL 0\nASSERT\n",
        "LOAD_LOCAL 1\nPUSH_I64 99\nARR_GET\nPUSH_F64 0.0\nNE\nASSERT\n"
        "PUSH_F64 -0.0\nPUSH_F64 0.0\nEQ\nASSERT\n"
    };
    CHECK(index<sizeof operations/sizeof *operations);
    char body[4096];snprintf(body,sizeof body,
        "PUSH_F64 1.5\nARR_LITERAL 3 1\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nSTORE_LOCAL 2\n"
        "PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 3\n%s"
        "OWN_MOVE_LOCAL 3\nCALL 1\nPUSH_I64 7\nEQ\nASSERT\nPUSH_I64 0\nRET\n",operations[index]);
    Type locals[]={{TAG_STRUCT,1},SCALAR(TAG_ARRAY),SCALAR(TAG_ARRAY),{TAG_STRUCT,0}};
    return build(body,locals,4,consume,false);
}
static void runtime_artifacts(NvmModule *m,const char *dir,unsigned index) {
    NvmV2Module v2;size_t size;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    char path[1024];snprintf(path,sizeof path,"%s/case%u.nvm",dir,index);
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(bytes,1,size,f)==size);CHECK(!fclose(f));free(bytes);nvm_v2_module_free(&v2);
    char error[256];char *source=nvm2c_emit(m,error,sizeof error);if(!source)fprintf(stderr,"native: %s\n",error);CHECK(source);
    snprintf(path,sizeof path,"%s/case%u.c",dir,index);f=fopen(path,"w");CHECK(f);CHECK(fputs(source,f)>=0);CHECK(!fclose(f));free(source);
}
static VmResult runtime_api(VmState *vm,unsigned api,NanoValue *out) {
    return api==0?vm_invoke(vm,0,NULL,0,out):api==1?vm_execute(vm):api==2?vm_call_function(vm,0,NULL,0):vm_invoke_callable(vm,val_function(0),NULL,0,out);
}
static void runtime_clean(VmState *vm,size_t baseline) {
    CHECK(!vm->stack_size && !vm->frame_count);
    CHECK(!vm->references.active && !vm->callee_references.active);
    for(unsigned f=0;f<NVM_OWNED_MAX_FUNCTIONS-2;f++)CHECK(!vm->value_references[f].active);
    CHECK(vm->heap.stats.num_objects==baseline);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    for(unsigned index=0;index<6;index++) {
        NvmModule *m=runtime_fixture(index);NvmVerifyResult verified=nvm_verify(m);
        if(!verified.ok)fprintf(stderr,"case%u: %s\n",index,verified.error_msg);
        CHECK(verified.ok);
        CHECK(!nvm_verify_owned_module(m).ok);CHECK(nvm_verify_linked(m,NULL,0).ok);
        runtime_artifacts(m,argv[1],index);
        char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);AsmResult error;
        NvmModule *copy=asm_assemble(text,&error);CHECK(copy);CHECK(nvm_verify(copy).ok);
        CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
        CHECK(copy->layout_size==m->layout_size && !memcmp(copy->layout_data,m->layout_data,m->layout_size));
        free(text);nvm_module_free(copy);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        VmResult wanted=index==4?VM_ERR_ASSERT_FAILED:index>=1&&index<=3?VM_ERR_TYPE_ERROR:VM_OK;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            NanoValue out=val_int(-91);VmResult result=runtime_api(&vm,api,&out);
            if(result!=wanted)fprintf(stderr,"case%u api%u repeat%u wanted%d got%d: %s\n",index,api,repeat,wanted,result,vm.error_msg);
            CHECK(result==wanted);
            if(result==VM_OK){if(api==1||api==2){CHECK(vm.stack_size==1);out=vm.stack[--vm.stack_size];}CHECK(out.tag==TAG_INT && !out.as.i64);vm_release(&vm.heap,out);}
            runtime_clean(&vm,baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u 0\n",index,index==4?2:index>=1&&index<=3?3:0);
    }
    printf("%u mixed runtime lifecycle checks passed\n",checks);return 0;
}
