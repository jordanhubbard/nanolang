#define HELPER_OWNER_ALLOC_TEST
#include "test_helper_local_owners.c"

static NvmModule *consuming_fixture(unsigned index,const char *call_override,const char *helper_override) {
    (void)local_fixture;
    bool nested=index&1, failure=index==2||index==3;
    uint8_t result_tag=index==5?TAG_BOOL:index==6?TAG_U8:TAG_INT;
    const char *result_name=index==5?"bool":index==6?"u8":"int";
    const char *construct=nested?
        "PUSH_I64 42\nOWN_PACK 0\nPUSH_I64 11\nOWN_PACK 0\nOWN_PACK 1\n":
        "PUSH_I64 42\nOWN_PACK 0\n";
    const char *read=nested?"BORROW_PATH_SHARED 0 0 0\n":"BORROW_LOCAL_SHARED 0 0\n";
    const char *dispose=nested?
        "OWN_UNPACK_LOCAL 0\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 2\n"
        "OWN_UNPACK_LOCAL 2\nPOP\nOWN_UNPACK_LOCAL 3\nPOP\n":"OWN_UNPACK_LOCAL 0\nPOP\n";
    char helper[4096],source[16384];
    snprintf(helper,sizeof(helper),
        "REGION_BEGIN\n%sREF_GET 0 0\nSTORE_LOCAL 1\nREGION_END\n"
        "PUSH_I64 8\nOWN_PACK 0\nOWN_STORE_LOCAL 4\n"
        "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 4\nREF_GET 0 0\nPUSH_I64 8\nEQ\nASSERT\nREGION_END\n"
        "%sOWN_UNPACK_LOCAL 4\nPOP\n%s%sRET\n",
        read,failure?"PUSH_BOOL 0\nASSERT\n":"PUSH_BOOL 1\nASSERT\n",dispose,
        index==5?"PUSH_BOOL 1\n":index==6?"PUSH_U8 7\n":"LOAD_LOCAL 1\n");
    const char *call=call_override?call_override:"OWN_MOVE_LOCAL 0\nCALL 1\nSTORE_LOCAL 1\n";
    snprintf(source,sizeof(source),
        ".types 3 0 0\n.entry 0\n.function main 0 8 0 %s 1\n"
        "PUSH_I64 9\nOWN_PACK 0\nOWN_STORE_LOCAL 2\n"
        "%sOWN_STORE_LOCAL 0\n%s%sOWN_STORE_LOCAL 0\n%s"
        "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 2\nREF_GET 0 0\nPUSH_I64 9\nEQ\nASSERT\nREGION_END\n"
        "%sOWN_UNPACK_LOCAL 2\nPOP\nLOAD_LOCAL 1\nRET\n.end\n"
        ".function inspect 1 8 0 %s 1\n%s.end\n.parameters 1 struct\n",
        result_name,construct,call,construct,call,index==4?"PUSH_BOOL 0\nASSERT\n":"",
        result_name,helper_override?helper_override:helper);
    NvmModule *m=fixture();AsmResult result;
    NvmModule *code=asm_assemble_unverified(source,&result);
    if(!code)fprintf(stderr,"%s\n",result.message);
    CHECK(code);
    free(m->code);m->code=code->code;code->code=NULL;m->code_size=code->code_size;
    memcpy(m->functions,code->functions,2*sizeof(*m->functions));nvm_module_free(code);
    free(m->ownership_data);m->ownership_size=180;m->ownership_data=calloc(180,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,2);word(p+4,3);p[8]=p[9]=p[10]=3;word(p+12,2);
    for(unsigned fn=0;fn<2;fn++){
        unsigned header=fn?92:16,base=header+12;
        p[header]=8;p[header+2]=fn;slot(p+header+4,result_tag,0,NVM_V2_NO_INDEX);
        for(unsigned i=0;i<8;i++)slot(p+base+8*i,
            i==0||i==2||i==3||i==4?TAG_STRUCT:(!fn&&i==1?result_tag:TAG_INT),0,
            i==0?(nested?1:0):i==2||i==3||i==4?0:NVM_V2_NO_INDEX);
    }
    word(p+168,1);p[172]=1;
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
    return m;
}
static void consuming_verified(NvmModule *m) {
    NvmVerifyResult result=nvm_verify(m);if(!result.ok)fprintf(stderr,"%s\n",result.error_msg);
    CHECK(result.ok);
    result=nvm_verify_owned_module(m);if(!result.ok)fprintf(stderr,"%s\n",result.error_msg);
    CHECK(result.ok);
}
static void consuming_refusals(void) {
    const char *calls[]={
        "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 0\nOWN_MOVE_LOCAL 0\nCALL 1\nSTORE_LOCAL 1\nREGION_END\n",
        "OWN_MOVE_LOCAL 0\nCALL 1\nPOP\nOWN_MOVE_LOCAL 0\nCALL 1\nSTORE_LOCAL 1\n",
        "LOAD_LOCAL 0\nCALL 1\nSTORE_LOCAL 1\n"};
    for(unsigned i=0;i<3;i++){
        NvmModule *m=consuming_fixture(0,calls[i],NULL);
        CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    }
    NvmModule *m=consuming_fixture(0,NULL,NULL);
    slot(m->ownership_data+104,TAG_STRUCT,0,2); /* Same shape, distinct nominal identity. */
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=consuming_fixture(0,NULL,"PUSH_I64 42\nRET\n");
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=consuming_fixture(0,NULL,"OWN_MOVE_LOCAL 0\nCALL 1\nRET\n");
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=consuming_fixture(0,NULL,NULL);m->ownership_data[105]=1;
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
}
#ifndef CONSUMING_ALLOC_TEST
int main(int argc,char **argv) {
    CHECK(argc==2);consuming_refusals();
    for(unsigned index=0;index<7;index++) {
        bool failure=index>=2&&index<=4;
        int expected=index==5?1:index==6?7:42;
        uint8_t tag=index==5?TAG_BOOL:index==6?TAG_U8:TAG_INT;NvmModule *m=consuming_fixture(index,NULL,NULL);
        consuming_verified(m);artifacts(m,argv[1],index);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<16;repeat++) {
            NanoValue result=val_void();uint64_t generation=vm.reference_generation;
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):
                api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            CHECK(status==(failure?VM_ERR_ASSERT_FAILED:VM_OK));
            CHECK(vm.reference_generation>generation);
            if(!failure){if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
                CHECK(result.tag==tag);
                CHECK((tag==TAG_BOOL?result.as.boolean:tag==TAG_U8?result.as.u8:result.as.i64)==expected);vm_release(&vm.heap,result);}
            CHECK(vm.stack_size==0&&vm.frame_count==0);
            CHECK(!vm.references.active&&!vm.callee_references.active);
            CHECK(vm.heap.stats.num_objects==baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u %d\n",index,failure?2:0,expected);
    }
    printf("%u consuming-call checks passed\n",checks);return 0;
}
#endif
