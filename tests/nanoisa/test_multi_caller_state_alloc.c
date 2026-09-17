#define MULTI_CALLER_ALLOC_TEST
#include "test_multi_caller_references.c"
static unsigned attempts,fail_at;
static bool direct_failure;
static VmState *observed_vm;
static bool allocation_fails(void) {
    return (direct_failure || (observed_vm && observed_vm->references.active &&
        !observed_vm->callee_references.active && observed_vm->current_fn==0)) && ++attempts==fail_at;
}
void *multi_state_malloc(size_t n){return allocation_fails()?NULL:malloc(n);}
void *multi_state_calloc(size_t n,size_t z){return allocation_fails()?NULL:calloc(n,z);}
static void binding_atomicity(void) {
    NvmModule *m=multi_fixture("PUSH_I64 0\nRET","PUSH_I64 0\nRET",2,shared);
    NvmAffineState *caller=nvm_affine_state_create(m,0,16),*empty=nvm_affine_state_create(m,1,10);
    CHECK(caller && empty);uint16_t scalar=0,pair[2]={2,3},left=0,right=1;
    CHECK(nvm_affine_scalar_define(caller,0));
    CHECK(nvm_affine_pack(caller,2,&scalar,1));CHECK(nvm_affine_pack(caller,3,&scalar,1));
    CHECK(nvm_affine_pack(caller,10,pair,2));CHECK(nvm_affine_region_begin(caller));
    CHECK(nvm_affine_borrow(caller,0,10,&left,1,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_borrow(caller,1,10,&right,1,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_borrow(caller,2,10,&right,1,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_borrow(caller,3,10,&left,1,NVM_REFERENCE_SHARED));
    NvmAffineState *before=nvm_affine_state_clone(caller);CHECK(before);
    for(unsigned failure=1;failure<=3;failure++) {
        NvmAffineState *trial=nvm_affine_state_clone(empty);CHECK(trial);
        attempts=0;fail_at=failure;direct_failure=true;
        bool bound=nvm_affine_bind_caller(trial,caller,0);direct_failure=false;
        CHECK(nvm_affine_state_equal(caller,before));
        CHECK(bound==(failure==3));
        if(!bound)CHECK(nvm_affine_state_equal(trial,empty));
        else {
            CHECK(nvm_affine_region_begin(trial));
            CHECK(nvm_affine_reborrow(trial,2,0,NVM_REFERENCE_SHARED));
            CHECK(nvm_affine_reborrow(trial,3,1,NVM_REFERENCE_SHARED));
            CHECK(nvm_affine_reference_access(trial,2,0,false));
            CHECK(nvm_affine_reference_access(trial,3,0,false));
        }
        nvm_affine_state_free(trial);
    }
    NvmAffineState *ordered=nvm_affine_state_clone(empty),*reordered=nvm_affine_state_clone(empty);CHECK(ordered && reordered);
    CHECK(nvm_affine_bind_caller(ordered,caller,0));CHECK(nvm_affine_bind_caller(reordered,caller,2));
    CHECK(!nvm_affine_state_equal(ordered,reordered));
    NO_CHANGE(empty,nvm_affine_bind_caller(empty,caller,UINT32_MAX));
    NO_CHANGE(empty,nvm_affine_bind_caller(empty,caller,15));
    CHECK(nvm_affine_state_equal(caller,before));
    nvm_affine_state_free(ordered);nvm_affine_state_free(reordered);nvm_affine_state_free(empty);
    nvm_affine_state_free(before);nvm_affine_state_free(caller);nvm_module_free(m);
}
int main(void) {
    binding_atomicity();NvmModule *m=eight_arguments();CHECK(nvm_verify(m).ok);
    for(unsigned failure=1;;failure++) {
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        attempts=0;fail_at=failure;observed_vm=&vm;NanoValue result=val_void();
        VmResult status=vm_invoke(&vm,0,NULL,0,&result);observed_vm=NULL;
        CHECK(vm.stack_size==0 && vm.frame_count==0);
        CHECK(!vm.references.active && !vm.callee_references.active);
        vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==baseline);
        if(status==VM_OK){CHECK(result.tag==TAG_INT && result.as.i64==828);vm_destroy(&vm);break;}
        CHECK(status==VM_ERR_MEMORY);CHECK(failure<12);
        CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK && result.as.i64==828);
        CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);
    }
    nvm_module_free(m);printf("%u multi-caller atomic binding/allocation checks passed\n",checks);return 0;
}
