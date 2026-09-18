#define HELPER_OWNER_ALLOC_TEST
#include "test_helper_local_owners.c"
static unsigned attempts,fail_at;
void *helper_heap_malloc(size_t n){return ++attempts==fail_at?NULL:malloc(n);}
void *helper_heap_calloc(size_t n,size_t z){return ++attempts==fail_at?NULL:calloc(n,z);}
void *helper_heap_realloc(void *p,size_t n){return ++attempts==fail_at?NULL:realloc(p,n);}
int main(void) {
    (void)artifacts;
    NvmModule *m=local_fixture(false,false);
    for(unsigned failure=1;;failure++) {
        fail_at=0;VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        attempts=0;fail_at=failure;NanoValue result=val_void();
        VmResult status=vm_invoke(&vm,0,NULL,0,&result);fail_at=0;
        CHECK(vm.stack_size==0 && vm.frame_count==0);
        CHECK(!vm.references.active && !vm.callee_references.active);
        vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);
        if(status==VM_OK){CHECK(result.tag==TAG_INT && result.as.i64==0);break;}
        CHECK(status==VM_ERR_MEMORY);CHECK(failure<128);
    }
    nvm_module_free(m);printf("%u helper-local allocation checks passed\n",checks);return 0;
}
