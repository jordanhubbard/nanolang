#define main runtime_fixture_main
#include "test_owned_runtime.c"
#undef main
#include "nested_reference_fixture.h"
static unsigned heap_attempt, heap_failure;
void *owned_heap_malloc(size_t n){if(++heap_attempt==heap_failure)return NULL;return malloc(n);}
void *owned_heap_calloc(size_t n,size_t s){if(++heap_attempt==heap_failure)return NULL;return calloc(n,s);}
void *owned_heap_realloc(void *p,size_t n){if(++heap_attempt==heap_failure)return NULL;return realloc(p,n);}
int main(void){
 const char *body=NESTED_ALLOCATION_BODY;
 NvmModule *m=nested_fixture(body);CHECK(nvm_verify(m).ok);
 for(unsigned failure=1;;failure++){
  VmState vm;heap_failure=0;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
  heap_attempt=0;heap_failure=failure;NanoValue result=val_void();
  VmResult status=vm_invoke(&vm,0,NULL,0,&result);heap_failure=0;
  CHECK(vm.stack_size==0 && vm.frame_count==0);
  CHECK(!vm.references.active && !vm.references.region && !vm.references.slots[0].region);
  vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==baseline);
  vm_destroy(&vm);
  if(status==VM_OK){CHECK(result.tag==TAG_INT && result.as.i64==42);CHECK(heap_attempt<failure);break;}
  CHECK(status==VM_ERR_MEMORY);CHECK(failure<32);
 }
 nvm_module_free(m);printf("%u nested reference VM allocation checks passed\n",checks);return 0;
}
