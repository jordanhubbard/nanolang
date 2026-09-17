#define CALLER_ALLOC_TEST
#include "test_caller_references.c"
static VmState *observed_vm;
static unsigned attempts,fail_at;
static bool refuse_allocation(void) {
    return observed_vm && observed_vm->references.active &&
       !observed_vm->callee_references.active && observed_vm->current_fn==0 &&
       ++attempts==fail_at;
}
void *caller_state_calloc(size_t n,size_t size) {
    return refuse_allocation()?NULL:calloc(n,size);
}
void *caller_state_malloc(size_t size) {
    return refuse_allocation()?NULL:malloc(size);
}
int main(void) {
    NvmModule *m=call_fixture("CALL_REF 1 0","PUSH_I64 42\nREF_SET 0 0\nREF_GET 0 0\nRET");
    CHECK(nvm_verify(m).ok);
    for(unsigned failure=1;;failure++) {
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        observed_vm=&vm;attempts=0;fail_at=failure;NanoValue result=val_void();
        VmResult status=vm_invoke(&vm,0,NULL,0,&result);observed_vm=NULL;
        CHECK(vm.frame_count==0 && vm.stack_size==0);
        CHECK(!vm.references.active && !vm.callee_references.active);
        vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==baseline);
        if(status==VM_OK){CHECK(result.tag==TAG_INT && result.as.i64==74);vm_destroy(&vm);break;}
        CHECK(status==VM_ERR_MEMORY);CHECK(failure<12);
        CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK && result.as.i64==74);
        CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);
    }
    nvm_module_free(m);printf("%u caller parameter allocation checks passed\n",checks);return 0;
}
