#define OWNED_FIELD_ALLOC_TEST
#include "test_owned_string_fields.c"
static unsigned attempts,fail_at,failures,budgets,injected;
void *result_heap_malloc(size_t n){if(n&&++attempts==fail_at){failures++;return NULL;}return malloc(n);}
void *result_heap_calloc(size_t n,size_t s){if(n&&s&&++attempts==fail_at){failures++;return NULL;}return calloc(n,s);}
void *result_heap_realloc(void *p,size_t n){if(n&&++attempts==fail_at){failures++;return NULL;}return realloc(p,n);}
static void invoke(VmState *vm,unsigned api,unsigned index,size_t baseline,bool fault) {
    NanoValue value=val_int(-91);VmResult status=result_api(vm,api,&value);fail_at=0;
    if(fault&&failures) {
        CHECK(failures==1&&status==VM_ERR_MEMORY);
        CHECK(value.tag==(api==0||api==3?TAG_VOID:TAG_INT));
        if(api==1||api==2)CHECK(value.as.i64==-91);
    } else {
        CHECK(status==(index>=2?VM_ERR_ASSERT_FAILED:VM_OK));
        if(status==VM_OK){if(api==1||api==2)value=vm->stack[--vm->stack_size];CHECK(value.tag==TAG_INT&&value.as.i64==42);vm_release(&vm->heap,value);}
    }
    result_clean(vm,baseline);
}
int main(void) {
    (void)result_fixture;(void)string_fixture;(void)roundtrip;(void)refusals;
    (void)missing_instantiated_literal;(void)field_refusals;(void)artifacts;(void)exact_stream;
    for(unsigned index=0;index<4;index++) {
        NvmModule *m=field_fixture(index);consuming_verified(m);bool done=false;
        for(unsigned fault=1;fault<256;fault++) {
            fprintf(stderr,"string allocation phase=setup case=%u fault=%u\n",index,fault);
            attempts=failures=0;fail_at=fault;VmState vm;vm_init(&vm,m);fail_at=0;
            unsigned hit=failures;budgets++;injected+=hit;
            if(vm.last_error==VM_OK)invoke(&vm,0,index,vm.heap.stats.num_objects,false);
            else CHECK(hit==1&&vm.last_error==VM_ERR_MEMORY);
            vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
            if(!hit){done=true;break;}
            VmState recovered;vm_init(&recovered,m);CHECK(recovered.last_error==VM_OK);
            invoke(&recovered,0,index,recovered.heap.stats.num_objects,false);vm_destroy(&recovered);
        }
        CHECK(done);
        for(unsigned api=0;api<4;api++) {
            VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);size_t baseline=vm.heap.stats.num_objects;done=false;
            for(unsigned fault=1;fault<256;fault++) {
                fprintf(stderr,"string allocation phase=invoke case=%u api=%u fault=%u\n",index,api,fault);
                attempts=failures=0;fail_at=fault;invoke(&vm,api,index,baseline,true);
                unsigned hit=failures;budgets++;injected+=hit;
                if(!hit){done=true;break;}
                invoke(&vm,api,index,baseline,false);
            }
            CHECK(done);vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
        }
        nvm_module_free(m);
    }
    printf("%u string field allocation checks; %u budgets, %u failures\n",checks,budgets,injected);return 0;
}
