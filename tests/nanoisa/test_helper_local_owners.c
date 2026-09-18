#define CALLER_RUNTIME_TEST
#include "test_caller_reference_analysis.c"
#include "nvm2c.h"
#include "../../src/nanovm/vm.h"
#include "../../src/runtime/callback_runtime.h"
int g_argc=0;char **g_argv=NULL;

static void artifacts(NvmModule *m,const char *dir,unsigned index) {
    NvmV2Module v2;size_t size;
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    char path[1024];snprintf(path,sizeof(path),"%s/case%u.nvm",dir,index);
    FILE *file=fopen(path,"wb");CHECK(file);CHECK(fwrite(bytes,1,size,file)==size);CHECK(!fclose(file));
    free(bytes);nvm_v2_module_free(&v2);
    char error[256];char *source=nvm2c_emit(m,error,sizeof(error));
    if(!source)fprintf(stderr,"%s\n",error);
    CHECK(source);snprintf(path,sizeof(path),"%s/case%u.c",dir,index);
    file=fopen(path,"w");CHECK(file);CHECK(fputs(source,file)>=0);CHECK(!fclose(file));free(source);
}


static NvmModule *local_fixture(bool shared,bool failure) {
    const char *body =
        "PUSH_I64 70\nOWN_PACK 0\nOWN_STORE_LOCAL 2\n"
        "PUSH_I64 90\nOWN_PACK 0\nOWN_STORE_LOCAL 3\n"
        "OWN_MOVE_LOCAL 2\nOWN_MOVE_LOCAL 3\nOWN_PACK 1\nOWN_STORE_LOCAL 5\n"
        "REGION_BEGIN\nBORROW_PATH_EXCLUSIVE 1 5 0\n"
        "PUSH_I64 77\nREF_SET 1 0\n"
        "REF_GET 1 0\nPUSH_I64 77\nEQ\nASSERT\n"
        "REGION_BEGIN\nREBORROW_SHARED 4 1\nREBORROW_SHARED 6 0\n"
        "REF_GET 4 0\nPUSH_I64 77\nEQ\nASSERT\nREF_GET 6 0\nPOP\nREGION_END\n";
    char helper[4096];snprintf(helper,sizeof(helper),"%s%s%s%s",body,
        shared?"REF_GET 0 0\nPUSH_I64 10\nEQ\nASSERT\n":
            "PUSH_I64 -32\nREF_SET 0 0\nREF_GET 1 0\nPUSH_I64 77\nEQ\nASSERT\n",
        failure?"PUSH_BOOL 0\nASSERT\n":"PUSH_BOOL 1\nASSERT\n",
        "REGION_END\nOWN_UNPACK_LOCAL 5\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 2\n"
        "REGION_BEGIN\nBORROW_LOCAL_SHARED 1 2\nREF_GET 1 0\nPUSH_I64 77\nEQ\nASSERT\nREGION_END\n"
        "OWN_UNPACK_LOCAL 2\nPOP\nOWN_UNPACK_LOCAL 3\nPOP\nPUSH_I64 0\nRET");
    NvmModule *m=call_fixture("CALL_REF 1 0\nPOP\nCALL_REF 1 0",helper);
    m->functions[1].local_count=8;
    uint8_t *data=calloc(180,1);CHECK(data);memcpy(data,m->ownership_data,112);
    free(m->ownership_data);m->ownership_data=data;m->ownership_size=180;
    data[92]=8;
    if(shared)data[105]=1;
    for(unsigned i=1;i<8;i++)slot(data+104+8*i,
        i==2||i==3||i==5?TAG_STRUCT:TAG_INT,0,
        i==2||i==3?0:i==5?1:NVM_V2_NO_INDEX);
    word(data+168,1);data[172]=1;
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    NvmVerifyResult verified=nvm_verify(m);
    if(!verified.ok)fprintf(stderr,"%s\n",verified.error_msg);
    CHECK(verified.ok);verified=nvm_verify_owned_module(m);
    if(!verified.ok)fprintf(stderr,"%s\n",verified.error_msg);
    CHECK(verified.ok);return m;
}

#ifndef HELPER_OWNER_ALLOC_TEST
int main(int argc,char **argv) {
    CHECK(argc==2);
    for(unsigned index=0;index<4;index++) {
        bool shared=index&1,failure=index&2;
        NvmModule *m=local_fixture(shared,failure);artifacts(m,argv[1],index);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<16;repeat++) {
            NanoValue result=val_void();
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):
                api==1?vm_execute(&vm):api==2?vm_call_function(&vm,0,NULL,0):
                vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            CHECK(status==(failure?VM_ERR_ASSERT_FAILED:VM_OK));
            if(!failure) {
                if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
                CHECK(result.tag==TAG_INT && result.as.i64==(shared?42:0));
                vm_release(&vm.heap,result);
            }
            CHECK(vm.stack_size==0 && vm.frame_count==0);
            CHECK(!vm.references.active && !vm.callee_references.active);
            CHECK(vm.heap.stats.num_objects==baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);
        printf("case %u %u %u\n",index,failure?2:0,shared?42:0);
    }
    printf("%u helper-local owner checks passed\n",checks);return 0;
}
#endif
