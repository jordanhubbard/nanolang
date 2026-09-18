/* I prove that each handled PRINT trap invalidates the invocation proof. */
#define OWNED_STRING_ALLOC_TEST
#include "test_owned_string_print.c"
static unsigned admissions;
static NvmVerifyResult counted_owned_admission(const NvmModule *m) {
    admissions++;return nvm_verify_owned_module(m);
}
#define nvm_verify_owned_module counted_owned_admission
#include "../../src/nanovm/vm.c"
#undef nvm_verify_owned_module

static NvmModule *proof_sequence(void) {
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(
        ".string value \"x\"\n.types 3 0 0\n.entry 0\n"
        ".function main 0 8 0 int 1\n"
        "PUSH_I64 99\nOWN_PACK 0\nOWN_STORE_LOCAL 0\n"
        "PUSH_STR value\nPRINT\nPUSH_STR value\nPRINTLN\n"
        "PUSH_STR value\nPRINT\nPUSH_STR value\nPRINTLN\n"
        "OWN_UNPACK_LOCAL 0\nPOP\nPUSH_I64 42\nRET\n.end\n",
        &assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);NvmModule *layouts=fixture();
    m->layout_data=layouts->layout_data;layouts->layout_data=NULL;
    m->layout_size=layouts->layout_size;nvm_module_free(layouts);
    m->ownership_size=16+76+4;m->ownership_data=calloc(m->ownership_size,1);
    CHECK(m->ownership_data);uint8_t *data=m->ownership_data;
    word(data,2);word(data+4,3);data[8]=data[9]=data[10]=3;word(data+12,1);
    data[16]=8;slot(data+20,TAG_INT,0,NVM_V2_NO_INDEX);
    for(unsigned local=0;local<8;local++)
        slot(data+28+8*local,local?TAG_INT:TAG_STRUCT,0,
             local?NVM_V2_NO_INDEX:0);
    bool needs=false;
    CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
    CHECK(nvm_verify_owned_module(m).ok);return m;
}

int main(void) {
    (void)artifacts;(void)result_fixture;(void)result_api;(void)roundtrip;
    (void)refusals;(void)missing_instantiated_literal;
    NvmModule *m=proof_sequence();consuming_verified(m);VmState vm;vm_init(&vm,m);
    size_t baseline=vm.heap.stats.num_objects;VmOwnedInvocationProof proof={0};
    admissions=0;CHECK(vm_ownership_admit(&vm,&proof));CHECK(admissions==1);
    CHECK(vm_owned_proof_matches(&vm,&proof));
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0]=(VmCallFrame){.fn_idx=0,.local_count=m->functions[0].local_count,.module=m};
    vm.stack_size=m->functions[0].local_count;
    VmTrap trap=vm_core_execute_scoped(&vm,&proof);
    CHECK(trap.type==TRAP_PRINT&&admissions==1);
    for(unsigned boundary=0;boundary<4;boundary++) {
        CHECK(trap.type==TRAP_PRINT);vm_release(&vm.heap,trap.data.print.value);
        proof.module=NULL;CHECK(!vm_owned_proof_matches(&vm,&proof));
        unsigned before=admissions;trap=vm_core_execute_scoped(&vm,&proof);
        CHECK(admissions==before+3);
        CHECK(trap.type==(boundary==3?TRAP_NONE:TRAP_PRINT));
    }
    CHECK(admissions==13&&vm.frame_count==0&&vm.stack_size==1);
    NanoValue result=vm.stack[--vm.stack_size];
    CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);
    CHECK(vm.heap.stats.num_objects==baseline&&!vm.references.active);
    vm_destroy(&vm);nvm_module_free(m);

    /* The public harness has no helper activations that could mask a boundary. */
    m=proof_sequence();vm_init(&vm,m);baseline=vm.heap.stats.num_objects;
    FILE *output=tmpfile();CHECK(output);vm.output=output;result=val_void();admissions=0;
    CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);
    CHECK(admissions==13);CHECK(result.tag==TAG_INT&&result.as.i64==42);
    vm_release(&vm.heap,result);result_clean(&vm,baseline);exact_stream(output,"xx\nxx\n");
    vm.output=NULL;CHECK(!fclose(output));vm_destroy(&vm);nvm_module_free(m);
    printf("%u owned string proof checks passed; %u admissions\n",checks,admissions);
    return 0;
}
