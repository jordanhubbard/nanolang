/* I inject only the real VM's frame reallocation or helper-contract result. */
#define MULTIPLE_CONSUMING_ALLOC_TEST
#include "test_multiple_consuming_calls.c"
static bool fail_frame,fail_contract,fail_last_parameter;
static unsigned frame_attempts,contract_attempts;
static void *consuming_realloc(void *p,size_t n) {
    frame_attempts++;
    return fail_frame?NULL:realloc(p,n);
}
static NvmAffineState *consuming_contract(const NvmModule *m,uint32_t function,uint16_t references) {
    if(function==1){contract_attempts++;if(fail_contract)return NULL;}
    return nvm_affine_state_create(m,function,references);
}
static bool consuming_parameters(const NvmAffineState *state,NvmAffineType *types,uint16_t capacity,uint16_t *count) {
    bool valid=nvm_affine_consuming_parameters(state,types,capacity,count);
    /* I inject a late preflight mismatch into a valid contract result only.
     * My fixture remains verified; no mismatched helper executes. */
    if(valid && fail_last_parameter) types[*count-1].layout=0;
    return valid;
}
#define nvm_affine_consuming_parameters consuming_parameters
#define realloc consuming_realloc
#define nvm_affine_state_create consuming_contract
#include "../../src/nanovm/vm.c"
#undef nvm_affine_state_create
#undef nvm_affine_consuming_parameters
#undef realloc
int main(void) {
    (void)artifacts;(void)multiple_refusals;
    NvmModule *m=multiple_fixture(2,NULL,NULL);consuming_verified(m);
    for(unsigned failure=0;failure<3;failure++)for(unsigned api=0;api<4;api++) {
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        vm.stack_capacity=25; /* 16 entry locals + 8 arguments fit; helper frame needs32. */
        fail_frame=failure==0;fail_contract=failure==1;fail_last_parameter=failure==2;
        frame_attempts=contract_attempts=0;NanoValue result=val_void();
        VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):
            api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
        fail_frame=fail_contract=fail_last_parameter=false;
        CHECK(status==(failure==2?VM_ERR_TYPE_ERROR:VM_ERR_MEMORY));CHECK(contract_attempts==1);
        CHECK(frame_attempts==(failure==0?1u:0u));
        CHECK(vm.reference_generation==1); /* No helper activation before reservation. */
        CHECK(vm.stack_size==0&&vm.frame_count==0);
        CHECK(!vm.references.active&&!vm.callee_references.active);
        CHECK(vm.heap.stats.num_objects==baseline);
        CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);
        CHECK(result.tag==TAG_INT&&result.as.i64==42);
        CHECK(vm.heap.stats.num_objects==baseline);
        vm_destroy(&vm);
    }
    nvm_module_free(m);printf("%u multiple consuming-call preflight checks passed\n",checks);return 0;
}
