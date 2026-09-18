/* I count actual heap allocation failures while returned owners remain live. */
#define OWNED_RESULT_ALLOC_TEST
#include "test_owned_value_results.c"
static unsigned attempts,fail_at,failures;
void *result_heap_malloc(size_t n){if(++attempts==fail_at){failures++;return NULL;}return malloc(n);}
void *result_heap_calloc(size_t n,size_t s){if(++attempts==fail_at){failures++;return NULL;}return calloc(n,s);}
void *result_heap_realloc(void *p,size_t n){if(++attempts==fail_at){failures++;return NULL;}return realloc(p,n);}
int main(void) {
    (void)artifacts;unsigned total_failures=0,budgets=0;
    for(unsigned index=0;index<10;index++) {
        NvmModule *m=result_fixture(index,0);consuming_verified(m);
        for(unsigned api=0;api<4;api++) {
            bool terminal=false;
            for(unsigned fault=1;fault<128;fault++) {
                fail_at=0;VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
                attempts=failures=0;fail_at=fault;NanoValue result=val_void();
                VmResult status=result_api(&vm,api,&result);fail_at=0;budgets++;total_failures+=failures;
                if(status==VM_OK){if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);}
                result_clean(&vm,baseline);vm_destroy(&vm);
                if(status!=VM_ERR_MEMORY){CHECK(!failures);CHECK(status==(index>=6?VM_ERR_ASSERT_FAILED:VM_OK));terminal=true;break;}
                CHECK(failures==1);
            }
            CHECK(terminal);
        }
        nvm_module_free(m);
    }
    printf("%u result allocation checks passed; %u budgets, %u actual injected failures\n",checks,budgets,total_failures);return 0;
}
