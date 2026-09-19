/* I intersect STRING initialization without weakening exact declarations. */
#define OWNED_RESULT_ALLOC_TEST
#include "test_owned_value_results.c"
static NvmModule *join_fixture(bool alternate,bool missing) {
    char source[4096];snprintf(source,sizeof(source),
        ".string left \"left\"\n.string right \"right\"\n.types 1 0 0\n.entry 0\n"
        ".function main 0 3 0 int 1\nPUSH_I64 99\nOWN_PACK 0\nOWN_STORE_LOCAL 0\n"
        "PUSH_BOOL %u\nJMP_FALSE other\nPUSH_STR left\nSTORE_LOCAL 1\nJMP joined\n"
        "other:\n%sjoined:\nLOAD_LOCAL 1\nDUP\nPOP\nPUSH_STR %s\nEQ\nASSERT\n"
        "OWN_UNPACK_LOCAL 0\nPOP\nPUSH_I64 42\nRET\n.end\n",
        alternate?0:1,missing?"":"PUSH_STR right\nSTORE_LOCAL 1\n",alternate?"right":"left");
    AsmResult error;NvmModule *m=asm_assemble_unverified(source,&error);CHECK(m);
    NvmV2LayoutField scalar={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout row={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar};
    NvmV2Layouts rows={&row,1};CHECK(nvm_retain_layouts(m,&rows)==NVM_V2_OK);
    m->ownership_size=56;m->ownership_data=calloc(56,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,2);word(p+4,1);p[8]=3;word(p+12,1);p[16]=3;
    slot(p+20,TAG_INT,0,NVM_V2_NO_INDEX);slot(p+28,TAG_STRUCT,0,0);
    slot(p+36,TAG_STRING,0,NVM_V2_NO_INDEX);slot(p+44,TAG_INT,0,NVM_V2_NO_INDEX);
    return m;
}
static void meet_queries(NvmModule *m) {
    NvmAffineState *base=nvm_affine_state_create(m,0,0);CHECK(base);
    for(unsigned left_init=0;left_init<2;left_init++)for(unsigned right_init=0;right_init<2;right_init++) {
        NvmAffineState *left=nvm_affine_state_clone(base),*right=nvm_affine_state_clone(base);CHECK(left&&right);
        if(left_init)CHECK(nvm_affine_string_define(left,1));
        if(right_init)CHECK(nvm_affine_string_define(right,1));
        bool changed=false;CHECK(nvm_affine_state_meet_initialization(left,right,&changed));
        CHECK(changed==(left_init&&!right_init));uint8_t tag=TAG_FLOAT,mode=77;
        CHECK(nvm_affine_local_info(left,1,&tag,&mode)==(left_init&&right_init));
        if(left_init&&right_init)CHECK(tag==TAG_STRING&&mode==0);
        else CHECK(tag==TAG_FLOAT&&mode==77);
        changed=true;CHECK(nvm_affine_state_meet_initialization(left,right,&changed));CHECK(!changed);
        nvm_affine_state_free(left);nvm_affine_state_free(right);
    }
    nvm_affine_state_free(base);
}
int main(int argc,char **argv) {
    (void)result_fixture;CHECK(argc==2);
    for(unsigned arm=0;arm<2;arm++) {
        NvmModule *refused=join_fixture(arm!=0,true);bool needs=false;
        CHECK(nvm_ownership_contracts_validate(refused,&needs)==NVM_V2_OK&&needs);
        CHECK(!nvm_verify_owned_module(refused).ok);CHECK(!nvm_verify(refused).ok);
        char error[256];CHECK(!nvm2c_emit(refused,error,sizeof(error)));
        nvm_module_free(refused); /* I never execute either uninitialized module. */
        NvmModule *m=join_fixture(arm!=0,false);consuming_verified(m);meet_queries(m);artifacts(m,argv[1],arm);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<3;repeat++) {
            NanoValue value=val_int(-91);CHECK(result_api(&vm,api,&value)==VM_OK);
            if(api==1||api==2)value=vm.stack[--vm.stack_size];
            CHECK(value.tag==TAG_INT&&value.as.i64==42);vm_release(&vm.heap,value);result_clean(&vm,baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u 0 42\n",arm);
    }
    printf("%u STRING initialization meet/refusal/runtime checks passed\n",checks);return 0;
}
