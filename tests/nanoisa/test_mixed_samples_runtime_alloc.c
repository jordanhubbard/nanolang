#define MIXED_RUNTIME_ALLOC_TEST
#include "test_mixed_samples_runtime.c"
static unsigned attempts,fail_at,failures,total_failures,total_budgets;
void *mixed_heap_malloc(size_t n){if(n&&++attempts==fail_at){failures++;return NULL;}return malloc(n);}
void *mixed_heap_calloc(size_t n,size_t size){if(n&&size&&++attempts==fail_at){failures++;return NULL;}return calloc(n,size);}
void *mixed_heap_realloc(void *p,size_t n){if(n&&++attempts==fail_at){failures++;return NULL;}return realloc(p,n);}
static void allocation_invoke(VmState *vm,unsigned api,VmResult wanted,size_t baseline,bool inject) {
    NanoValue result=val_int(-91);VmResult status=runtime_api(vm,api,&result);fail_at=0;
    if(inject&&failures) {
        CHECK(failures==1 && status==VM_ERR_MEMORY);
        if(api==1||api==2)CHECK(result.tag==TAG_INT && result.as.i64==-91);
        else CHECK(result.tag==TAG_VOID);
    } else {
        CHECK(status==wanted);
        if(status==VM_OK){if(api==1||api==2)result=vm->stack[--vm->stack_size];CHECK(result.tag==TAG_INT && !result.as.i64);vm_release(&vm->heap,result);}
    }
    runtime_clean(vm,baseline);
}
int main(void) {
    (void)runtime_artifacts;(void)runtime_core;
    for(unsigned index=0;index<12;index++) {
        NvmModule *m=runtime_fixture(index);CHECK(nvm_verify(m).ok);
        VmResult wanted=(index==4||index==8)?VM_ERR_ASSERT_FAILED:index>=1&&index<=3?VM_ERR_TYPE_ERROR:VM_OK;
        for(unsigned api=0;api<4;api++) {
            VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);size_t baseline=vm.heap.stats.num_objects;bool done=false;
            for(unsigned fault=1;fault<128;fault++) {
                fprintf(stderr,"mixed VM heap case=%u api=%u fault=%u\n",index,api,fault);
                attempts=failures=0;fail_at=fault;allocation_invoke(&vm,api,wanted,baseline,true);
                unsigned hit=failures;total_budgets++;total_failures+=hit;
                if(!hit){done=true;break;}
                allocation_invoke(&vm,api,wanted,baseline,false);
            }
            CHECK(done);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
        }
        nvm_module_free(m);
    }
    printf("%u mixed VM heap checks; %u budgets, %u failures\n",checks,total_budgets,total_failures);return 0;
}
