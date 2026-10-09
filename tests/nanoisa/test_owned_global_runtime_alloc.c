#define OWNED_GLOBAL_NO_MAIN
#include "test_owned_global_runtime.c"
static unsigned heap_attempt,heap_failure;
static VmHeap *observed_heap;
static size_t peak;
static bool fail_allocation(void) {
    if(observed_heap && observed_heap->stats.num_objects>peak)peak=observed_heap->stats.num_objects;
    return ++heap_attempt==heap_failure;
}
void *owned_heap_malloc(size_t n){if(fail_allocation())return NULL;return malloc(n);}
void *owned_heap_calloc(size_t n,size_t s){if(fail_allocation())return NULL;return calloc(n,s);}
void *owned_heap_realloc(void *p,size_t n){if(fail_allocation())return NULL;return realloc(p,n);}
int main(void) {
    global_public_routes();
    NvmModule *m=global_union_module(false);CHECK(nvm_verify(m).ok);
    for(unsigned failure=1;;failure++) {
        VmState vm;heap_failure=0;observed_heap=NULL;vm_init(&vm,m);
        size_t baseline=vm.heap.stats.num_objects;peak=baseline;observed_heap=&vm.heap;
        heap_attempt=0;heap_failure=failure;NanoValue result=val_void();
        VmResult status=vm_invoke(&vm,0,NULL,0,&result);heap_failure=0;
        CHECK(!vm.global_count && !vm.stack_size && !vm.frame_count);
        CHECK(vm.heap.stats.num_objects==baseline && peak<=baseline+2);
        unsigned attempts=heap_attempt;
        CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK && result.tag==TAG_INT && result.as.i64==42);
        CHECK(!vm.global_count && vm.heap.stats.num_objects==baseline);
        observed_heap=NULL;vm_destroy(&vm);
        if(status==VM_OK) {if(attempts<failure)break;}
        else CHECK(status==VM_ERR_MEMORY);
        CHECK(failure<256);
    }
    nvm_module_free(m);printf("%u typed global VM allocation checks passed\n",checks);return 0;
}
