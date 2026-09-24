#define main affine_suite_unused_main
#include "test_affine_bytecode.c"
#undef main
#include "../../src/nanovm/vm.h"
int g_argc=0;char **g_argv=NULL;
static void indirect_artifacts(NvmModule *m,const char *directory,unsigned index,unsigned apply,unsigned other) {
    NvmV2Module wire={0};size_t length=0;
    CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&wire,NULL,0,&length)==NVM_V2_OK);
    uint8_t *bytes=malloc(length);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&wire,bytes,length,NULL)==NVM_V2_OK);
    char path[1024];snprintf(path,sizeof(path),"%s/case%u.nvm",directory,index);
    FILE *file=fopen(path,"wb");CHECK(file);CHECK(fwrite(bytes,1,length,file)==length);CHECK(!fclose(file));
    free(bytes);nvm_v2_module_free(&wire);
    char error[256];char *source=nvm2c_emit(m,error,sizeof(error));
    if(!source)fprintf(stderr,"%s\n",error);CHECK(source);
    snprintf(path,sizeof(path),"%s/case%u.c",directory,index);
    file=fopen(path,"w");CHECK(file);CHECK(fputs(source,file)>=0);CHECK(!fclose(file));free(source);
    snprintf(path,sizeof(path),"%s/case%u.guard.c",directory,index);
    file=fopen(path,"w");CHECK(file);
    CHECK(fprintf(file,
        " for(unsigned wrong=0;wrong<2;wrong++){\n"
        "  nown_value arguments[1]={{.scalar=%u,.tag=wrong?%u:%u}},result={0};uint64_t generation=2;\n"
        "  assert(nown_function_%u(arguments,&generation,2,&result)==3);\n"
        "  assert(live==0 && !arguments[0].tag && !result.tag);\n }\n",
        other,TAG_INT,TAG_FUNCTION,apply)>0);CHECK(!fclose(file));
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    const unsigned orders[][6]={{0,1,2,3,4,5},{0,5,4,3,2,1},{0,3,1,5,2,4},
        {0,2,4,1,5,3},{0,4,3,2,1,5},{0,5,2,4,3,1}};
    const unsigned kinds[]={0,1,2,8,9,10,11,14,15,16};unsigned index=0;
    for(unsigned order=0;order<6;order++)for(unsigned test=0;test<(order?10u:16u);test++) {
        unsigned kind=test<10?kinds[test]:(test&1?14:15)+32*((test-10)/2+1);
        bool failure=kind>=32;int64_t expected=kind==1?41:42;
        NvmModule *m=targets_fixture(kind,orders[order]);
        NvmVerifyResult verified=nvm_verify_owned_module(m);
        if(!verified.ok)fprintf(stderr,"case %u: %s\n",index,verified.error_msg);CHECK(verified.ok);
        CHECK(nvm_verify(m).ok);
        unsigned id[6];for(unsigned f=0;f<6;f++)id[orders[order][f]]=f;
        indirect_artifacts(m,argv[1],index,id[3],id[2]);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            NanoValue result=val_void();uint64_t generation=vm.reference_generation;
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            if(status!=(failure?VM_ERR_ASSERT_FAILED:VM_OK))fprintf(stderr,"case %u api %u: %s\n",index,api,vm.error_msg);
            CHECK(status==(failure?VM_ERR_ASSERT_FAILED:VM_OK));CHECK(vm.reference_generation>generation);
            if(!failure) {
                if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
                CHECK(result.tag==TAG_INT&&result.as.i64==expected);vm_release(&vm.heap,result);
            }
            CHECK(!vm.stack_size&&!vm.frame_count);CHECK(!vm.references.active&&!vm.callee_references.active);
            for(unsigned frame=0;frame<NVM_OWNED_MAX_FUNCTIONS-2;frame++)CHECK(!vm.value_references[frame].active);
            CHECK(vm.heap.stats.num_objects==baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u %lld\n",index++,failure?2:0,(long long)expected);
    }
    CHECK(index==66);printf("%u owned indirect call checks passed\n",checks);return 0;
}
