/* I prove that each handled PRINT trap invalidates the invocation proof. */
#define OWNED_STRING_ALLOC_TEST
#include "test_owned_string_print.c"
static unsigned admissions;
static NvmVerifyResult counted_owned_admission(const NvmModule *m) {
    admissions++;return nvm_verify_owned_module(m);
}
#define nvm_verify_owned_module counted_owned_admission
#include "../../src/nanovm/vm.c"
#undef nvm_verify_owned_module

int main(void) {
    (void)artifacts;(void)result_fixture;(void)result_api;(void)roundtrip;
    (void)refusals;(void)missing_instantiated_literal;
    NvmModule *m=string_fixture(false);consuming_verified(m);VmState vm;vm_init(&vm,m);
    size_t baseline=vm.heap.stats.num_objects;FILE *output=tmpfile();CHECK(output);vm.output=output;
    NanoValue result=val_void();admissions=0;
    CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);
    CHECK(admissions>1);CHECK(result.tag==TAG_INT&&result.as.i64==42);
    vm_release(&vm.heap,result);result_clean(&vm,baseline);exact_stream(output,expected_output);
    vm.output=NULL;CHECK(!fclose(output));vm_destroy(&vm);nvm_module_free(m);
    printf("%u owned string proof checks passed; %u admissions\n",checks,admissions);
    return 0;
}
