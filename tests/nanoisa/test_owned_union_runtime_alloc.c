#define main union_runtime_fixture_main
#include "test_owned_union_runtime.c"
#undef main
static unsigned heap_attempt,heap_failure;
void *owned_heap_malloc(size_t n){if(++heap_attempt==heap_failure)return NULL;return malloc(n);}
void *owned_heap_calloc(size_t n,size_t s){if(++heap_attempt==heap_failure)return NULL;return calloc(n,s);}
void *owned_heap_realloc(void *p,size_t n){if(++heap_attempt==heap_failure)return NULL;return realloc(p,n);}
int main(void) {
 const char *construct="PUSH_I64 42\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nAGG_PACK 1 1 0 1\nOWN_STORE_LOCAL 2\n";
 const char *consume="LOAD_LOCAL 2\nMATCH_TAG 0 outer\nPOP\nHALT\nouter:\nPOP\nOWN_UNPACK_VARIANT 2 0 1\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nMATCH_TAG 0 inner\nPOP\nHALT\ninner:\nPOP\nOWN_UNPACK_VARIANT 0 0 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n";
 for(unsigned trap=0;trap<2;trap++) {
  char body[2048];snprintf(body,sizeof(body),"%s%s%s",construct,trap?"PUSH_BOOL 0\nASSERT\n":"",consume);
  NvmModule *m=nested_entry(body);CHECK(nvm_verify(m).ok);
  for(unsigned failure=1;;failure++) {
   VmState vm;heap_failure=0;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
   heap_attempt=0;heap_failure=failure;NanoValue result=val_void();
   VmResult status=vm_invoke(&vm,0,NULL,0,&result);heap_failure=0;
   CHECK(vm.stack_size==0 && vm.frame_count==0);
   vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==baseline);
   vm_destroy(&vm);
   if(heap_attempt<failure) {
    CHECK(status==(trap?VM_ERR_ASSERT_FAILED:VM_OK));
    if(!trap)CHECK(result.tag==TAG_INT && result.as.i64==42);
    break;
   }
   CHECK(status==VM_ERR_MEMORY);CHECK(failure<64);
  }
  nvm_module_free(m);
 }
 printf("%u owned union VM allocation checks passed\n",checks);return 0;
}
