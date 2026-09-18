/* I establish fresh proofs at each public entry and preserve advisory fallback. */
#define OWNED_GRAPH_ALLOC_TEST
#include "test_owned_value_graphs.c"
static NvmModule *advisory_fixture(bool wrong) {
    const char *valid=".types 3 0 0\n.entry 0\n.function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n";
    const char *invalid=".types 3 0 0\n.entry 0\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nPUSH_I64 1\nI64_ADD\nRET\n.end\n";
    AsmResult result;NvmModule *m=asm_assemble_unverified(wrong?invalid:valid,&result);CHECK(m);
    NvmModule *layouts=fixture();m->layout_data=layouts->layout_data;layouts->layout_data=NULL;
    m->layout_size=layouts->layout_size;nvm_module_free(layouts);
    m->ownership_size=32;m->ownership_data=calloc(32,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,3);data[8]=data[9]=data[10]=NVM_LAYOUT_COMPLETE;
    word(data+12,1);slot(data+20,TAG_INT,0,NVM_V2_NO_INDEX);
    bool needs=true;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&!needs);return m;
}
int main(void) {
    (void)ordinary_chain;(void)graph_refusals;(void)artifacts;
    NvmModule *m=graph_fixture(0,4);consuming_verified(m);
    CHECK(nvm_verify_linked(m,NULL,0).ok);
    uint16_t depth=7;CHECK(nvm_verify_function(m,2).ok);
    CHECK(nvm_verify_function_max_stack(m,2,&depth).ok&&depth==NVM_AFFINE_MAX_STACK);
    depth=7;CHECK(!nvm_verify_function_max_stack(m,4,&depth).ok&&depth==7);
    const NvmModule *linked[]={m};CHECK(!nvm_verify_linked(m,linked,1).ok);
    CHECK(nvm_verify_affine_function(m,2).ok);
    /* A subsequent public call must inspect changed nominal metadata again. */
    slot(m->ownership_data+16+3*140+12,TAG_STRUCT,0,2);
    CHECK(!nvm_verify(m).ok);CHECK(!nvm_verify_function(m,0).ok);CHECK(!nvm_verify_linked(m,NULL,0).ok);
    slot(m->ownership_data+16+3*140+12,TAG_STRUCT,0,0);CHECK(nvm_verify(m).ok);nvm_module_free(m);
    for(unsigned wrong=0;wrong<2;wrong++) {
        m=advisory_fixture(wrong);CHECK(!nvm_verify_owned_module(m).ok);
        CHECK(nvm_verify(m).ok==!wrong);CHECK(nvm_verify_function(m,0).ok==!wrong);
        CHECK(nvm_verify_linked(m,NULL,0).ok==!wrong);
        depth=0;CHECK(nvm_verify_function_max_stack(m,0,&depth).ok==!wrong);
        if(!wrong)CHECK(depth==1);
        nvm_module_free(m);
    }
    printf("%u owned verification reuse checks passed\n",checks);return 0;
}
