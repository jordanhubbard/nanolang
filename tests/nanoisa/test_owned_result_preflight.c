/* I fail result facts before detaching the live returned owner. */
#define OWNED_RESULT_ALLOC_TEST
#include "test_owned_value_results.c"
static VmState *active_vm;
static bool fail_return;
static unsigned return_failures;
static NvmAffineState *result_contract(const NvmModule *m,uint32_t function,uint32_t references) {
    if(fail_return&&active_vm&&active_vm->frame_count&&active_vm->current_fn==function&&function==1){return_failures++;return NULL;}
    return nvm_affine_state_create(m,function,references);
}
#define nvm_affine_state_create result_contract
#include "../../src/nanovm/vm.c"
#undef nvm_affine_state_create
int main(void) {
    (void)artifacts;NvmModule *m=result_fixture(0,0);consuming_verified(m);
    for(unsigned api=0;api<4;api++) {
        VmState vm;vm_init(&vm,m);active_vm=&vm;size_t baseline=vm.heap.stats.num_objects;
        for(unsigned retry=0;retry<2;retry++) {
            fail_return=!retry;return_failures=0;NanoValue result=val_void();
            VmResult status=result_api(&vm,api,&result);
            CHECK(status==(retry?VM_OK:VM_ERR_MEMORY));CHECK(return_failures==(!retry));
            if(retry){if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);}
            result_clean(&vm,baseline);
        }
        active_vm=NULL;vm_destroy(&vm);
    }
    nvm_module_free(m);printf("%u return preflight checks passed\n",checks);return 0;
}
