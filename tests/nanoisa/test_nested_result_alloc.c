#define NESTED_RESULT_ALLOC_TEST
#include "test_nested_owned_results.c"
static unsigned attempts,fail_at,failures;
void *result_heap_malloc(size_t n){if(++attempts==fail_at){failures++;return NULL;}return malloc(n);}
void *result_heap_calloc(size_t n,size_t s){
    /* I count only required capacity; an empty record permits NULL fields. */
    if(n && s && ++attempts==fail_at){failures++;return NULL;}
    return calloc(n,s);
}
void *result_heap_realloc(void *p,size_t n){if(++attempts==fail_at){failures++;return NULL;}return realloc(p,n);}
int main(void) {
    (void)result_fixture;(void)artifacts;(void)nested_roundtrip;unsigned injected=0,budgets=0;
    for(unsigned index=0;index<6;index++) {
        NvmModule *m=nested_fixture(index);consuming_verified(m);
        for(unsigned api=0;api<4;api++) {
            bool terminal=false;
            for(unsigned fault=1;fault<256;fault++) {
                fail_at=0;VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
                fprintf(stderr,"nested allocation case=%u api=%u fault=%u\n",index,api,fault);
                attempts=failures=0;fail_at=fault;NanoValue value=val_void();VmResult status=result_api(&vm,api,&value);
                fail_at=0;budgets++;injected+=failures;
                if(status==VM_OK){if(api==1||api==2)value=vm.stack[--vm.stack_size];CHECK(value.tag==TAG_INT&&value.as.i64==42);vm_release(&vm.heap,value);}
                result_clean(&vm,baseline);vm_destroy(&vm);
                if(status!=VM_ERR_MEMORY){CHECK(!failures);CHECK(status==(index>=4?VM_ERR_ASSERT_FAILED:VM_OK));terminal=true;break;}
                CHECK(failures==1);
            }
            CHECK(terminal);
        }
        nvm_module_free(m);
    }
    printf("%u nested result allocation checks; %u budgets, %u actual failures\n",checks,budgets,injected);return 0;
}
