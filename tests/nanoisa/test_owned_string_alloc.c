/* I inject reached VmHeap setup and invocation sites, not call-frame allocation. */
#define OWNED_STRING_ALLOC_TEST
#include "test_owned_string_print.c"
static unsigned attempts,fail_at,failures,setup_attempts,invoke_attempts;
void *result_heap_malloc(size_t n){if(++attempts==fail_at){failures++;return NULL;}return malloc(n);}
void *result_heap_calloc(size_t n,size_t s){if(++attempts==fail_at){failures++;return NULL;}return calloc(n,s);}
void *result_heap_realloc(void *p,size_t n){if(++attempts==fail_at){failures++;return NULL;}return realloc(p,n);}

static void invoke_success(VmState *vm,size_t baseline) {
    FILE *output=tmpfile();CHECK(output);vm->output=output;NanoValue result=val_void();
    CHECK(vm_invoke(vm,0,NULL,0,&result)==VM_OK);
    CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm->heap,result);
    result_clean(vm,baseline);exact_stream(output,expected_output);
    vm->output=NULL;CHECK(!fclose(output));
}

static void setup_faults(NvmModule *m) {
    bool terminal=false;
    for(unsigned fault=1;fault<128;fault++) {
        attempts=failures=0;fail_at=fault;VmState vm;vm_init(&vm,m);fail_at=0;
        if(vm.last_error==VM_OK)invoke_success(&vm,vm.heap.stats.num_objects);
        else CHECK(vm.last_error==VM_ERR_MEMORY&&failures==1);
        bool injected=failures!=0;
        vm_destroy(&vm);CHECK(vm.heap.stats.num_objects==0);
        if(injected) {
            VmState recovered;attempts=failures=0;vm_init(&recovered,m);
            CHECK(recovered.last_error==VM_OK);
            invoke_success(&recovered,recovered.heap.stats.num_objects);
            vm_destroy(&recovered);CHECK(recovered.heap.stats.num_objects==0);
        } else {setup_attempts=attempts;terminal=true;}
        if(terminal)break;
    }
    CHECK(terminal&&setup_attempts>0);
}

static void invoke_faults(NvmModule *m,unsigned api) {
    VmState vm;attempts=failures=0;fail_at=0;vm_init(&vm,m);
    CHECK(vm.last_error==VM_OK);size_t baseline=vm.heap.stats.num_objects;bool terminal=false;
    for(unsigned fault=1;fault<128;fault++) {
        FILE *failed=tmpfile();CHECK(failed);vm.output=failed;
        attempts=failures=0;fail_at=fault;NanoValue result=val_int(-91);
        VmResult status=result_api(&vm,api,&result);fail_at=0;
        if(status==VM_ERR_MEMORY) {
            CHECK(failures==1);
            if(api==0||api==3)CHECK(result.tag==TAG_VOID);
            else CHECK(result.tag==TAG_INT&&result.as.i64==-91);
            result_clean(&vm,baseline);exact_stream(failed,"");
            vm.output=NULL;CHECK(!fclose(failed));
            FILE *recovered=tmpfile();CHECK(recovered);vm.output=recovered;
            result=val_int(-91);CHECK(result_api(&vm,api,&result)==VM_OK);
            if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
            CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);
            result_clean(&vm,baseline);exact_stream(recovered,expected_output);
            vm.output=NULL;CHECK(!fclose(recovered));continue;
        }
        CHECK(!failures&&status==VM_OK);terminal=true;
        if(invoke_attempts)CHECK(attempts==invoke_attempts);else invoke_attempts=attempts;
        if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
        CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);
        result_clean(&vm,baseline);exact_stream(failed,expected_output);
        vm.output=NULL;CHECK(!fclose(failed));break;
    }
    CHECK(terminal);vm_destroy(&vm);
}

int main(void) {
    (void)artifacts;(void)result_fixture;(void)roundtrip;
    (void)refusals;(void)missing_instantiated_literal;
    NvmModule *m=string_fixture(false);consuming_verified(m);setup_faults(m);
    for(unsigned api=0;api<4;api++)invoke_faults(m,api);
    nvm_module_free(m);
    printf("%u owned string allocation checks passed; %u setup and %u invocation attempts\n",
           checks,setup_attempts,invoke_attempts);
    return 0;
}
