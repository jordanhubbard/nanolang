/* I count actual runtime admissions without changing verifier results. */
#define OWNED_GRAPH_ALLOC_TEST
#include "test_owned_value_graphs.c"
static unsigned admissions;
static NvmVerifyResult counted_owned_admission(const NvmModule *m) {
    admissions++;
    return nvm_verify_owned_module(m);
}
#define nvm_verify_owned_module counted_owned_admission
#include "../../src/nanovm/vm.c"
#undef nvm_verify_owned_module
static VmResult invoke_api(VmState *vm,unsigned api,NanoValue *result) {
    return api==0?vm_invoke(vm,0,NULL,0,result):api==1?vm_execute(vm):
        api==2?vm_call_function(vm,0,NULL,0):
        vm_invoke_callable(vm,val_function(0),NULL,0,result);
}
int main(void) {
    (void)ordinary_chain;(void)graph_refusals;(void)artifacts;
    for(unsigned failure=0;failure<2;failure++)for(unsigned api=0;api<4;api++) {
        NvmModule *m=graph_fixture(failure,4);consuming_verified(m);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned repeat=0;repeat<2;repeat++) {
            uint64_t generation=vm.reference_generation;
            admissions=0;NanoValue result=val_void();
            VmResult status=invoke_api(&vm,api,&result);
            CHECK(admissions==1);CHECK(status==(failure?VM_ERR_ASSERT_FAILED:VM_OK));
            if(!failure) {
                if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
                CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);
            }
            CHECK(vm.reference_generation>generation);
            CHECK(!vm.stack_size&&!vm.frame_count);
            CHECK(!vm.references.active&&!vm.callee_references.active);
            for(unsigned f=0;f<NVM_OWNED_MAX_FUNCTIONS-2;f++)CHECK(!vm.value_references[f].active);
            CHECK(vm.heap.stats.num_objects==baseline);
        }
        /* A previous successful proof does not authorize changed declarations.
         * I require refusal before execution and then restore valid metadata. */
        uint8_t tag=m->functions[1].result_tag;m->functions[1].result_tag=TAG_FLOAT;
        uint64_t generation=vm.reference_generation;admissions=0;NanoValue result=val_void();
        CHECK(invoke_api(&vm,api,&result)==VM_ERR_TYPE_ERROR);CHECK(admissions<=1);
        CHECK(vm.reference_generation==generation);CHECK(!vm.stack_size&&!vm.frame_count);
        m->functions[1].result_tag=tag;
        vm_destroy(&vm);nvm_module_free(m);
    }
    /* Eligibility alone never creates a proof; tracing disables this reuse. */
    NvmModule *m=graph_fixture(0,4);VmState vm;vm_init(&vm,m);
    VmOwnedInvocationProof proof={0};CHECK(!vm_owned_proof_matches(&vm,&proof));
    CHECK(vm_ownership_admit(&vm,&proof));CHECK(vm_owned_proof_matches(&vm,&proof));
    vm.opcode_trace=true;CHECK(!vm_owned_proof_matches(&vm,&proof));vm.opcode_trace=false;
    vm.root_module=NULL;CHECK(!vm_owned_proof_matches(&vm,&proof));vm.root_module=m;
    vm.linked_module_count=1;CHECK(!vm_owned_proof_matches(&vm,&proof));vm.linked_module_count=0;
    proof.module=NULL;CHECK(!vm_owned_proof_matches(&vm,&proof));
    vm_destroy(&vm);nvm_module_free(m);
    printf("%u invocation proof checks passed\n",checks);return 0;
}
