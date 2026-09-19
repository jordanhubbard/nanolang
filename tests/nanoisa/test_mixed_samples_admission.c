#define MIXED_RUNTIME_ALLOC_TEST
#include "test_mixed_samples_runtime.c"
static bool fail_admission;
static unsigned admission_failures;
void *mixed_admit_malloc(size_t n){if(fail_admission){fail_admission=false;admission_failures++;return NULL;}return malloc(n);}
void *mixed_admit_calloc(size_t n,size_t size){if(fail_admission){fail_admission=false;admission_failures++;return NULL;}return calloc(n,size);}
int main(void) {
    (void)runtime_artifacts;(void)runtime_core;
    NvmModule *m=runtime_fixture(0);CHECK(nvm_verify(m).ok);
    for(unsigned api=0;api<4;api++) {
        VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);size_t baseline=vm.heap.stats.num_objects;
        NanoValue result=val_int(-91);uint64_t generation=vm.reference_generation;
        admission_failures=0;fail_admission=true;VmResult status=runtime_api(&vm,api,&result);fail_admission=false;
        CHECK(admission_failures==1 && status==VM_ERR_MEMORY);
        CHECK(result.tag==TAG_INT && result.as.i64==-91);CHECK(vm.reference_generation==generation);
        runtime_clean(&vm,baseline);
        CHECK(runtime_api(&vm,api,&result)==VM_OK);
        if(api==1||api==2)result=vm.stack[--vm.stack_size];
        CHECK(result.tag==TAG_INT && !result.as.i64);vm_release(&vm.heap,result);
        runtime_clean(&vm,baseline);vm_destroy(&vm);
    }
    uint16_t maximum=123;admission_failures=0;fail_admission=true;
    CHECK(!nvm_verify_function_max_stack(m,0,&maximum).ok);fail_admission=false;
    CHECK(admission_failures==1 && maximum==123);
    CHECK(nvm_verify_function_max_stack(m,0,&maximum).ok && maximum!=123);
    char error[256];admission_failures=0;fail_admission=true;
    char *source=nvm2c_emit(m,error,sizeof error);fail_admission=false;
    CHECK(!source && admission_failures==1);
    source=nvm2c_emit(m,error,sizeof error);CHECK(source);free(source);
    nvm_module_free(m);printf("%u mixed admission allocation/output checks passed\n",checks);return 0;
}
