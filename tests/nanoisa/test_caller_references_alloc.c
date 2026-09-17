#define CALLER_ALLOC_TEST
#include "test_caller_references.c"
static unsigned heap_attempt,heap_failure;
void *owned_heap_malloc(size_t n){if(++heap_attempt==heap_failure)return NULL;return malloc(n);}
void *owned_heap_calloc(size_t n,size_t z){if(++heap_attempt==heap_failure)return NULL;return calloc(n,z);}
void *owned_heap_realloc(void *p,size_t n){if(++heap_attempt==heap_failure)return NULL;return realloc(p,n);}
int main(void) {
    NvmModule *m=call_fixture("CALL_REF 1 0","PUSH_I64 42\nREF_SET 0 0\nREF_GET 0 0\nRET");
    CHECK(nvm_verify(m).ok);
    for(unsigned failure=1;;failure++) {
        VmState vm;heap_failure=0;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        heap_attempt=0;heap_failure=failure;NanoValue result=val_void();
        VmResult status=vm_invoke(&vm,0,NULL,0,&result);heap_failure=0;
        CHECK(vm.frame_count==0 && vm.stack_size==0);
        CHECK(!vm.references.active && !vm.callee_references.active);
        vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);
        if(status==VM_OK){CHECK(result.tag==TAG_INT && result.as.i64==74);break;}
        CHECK(status==VM_ERR_MEMORY);CHECK(failure<32);
    }
    nvm_module_free(m);printf("%u caller allocation checks passed\n",checks);return 0;
}
