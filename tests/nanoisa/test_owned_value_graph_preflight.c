/* I fail a deeper frame preflight while all callers still retain their roots. */
#define OWNED_GRAPH_ALLOC_TEST
#include "test_owned_value_graphs.c"
static unsigned failure,requested,attempts;
static NvmAffineState *graph_contract(const NvmModule *m,uint32_t function,uint16_t refs) {
    requested=function;
    if(function==4){attempts++;if(failure==1)return NULL;}
    return nvm_affine_state_create(m,function,refs);
}
static bool graph_parameters(const NvmAffineState *s,NvmAffineType *types,uint16_t capacity,uint16_t *count) {
    bool valid=nvm_affine_value_parameters(s,types,capacity,count);
    if(valid&&requested==4&&failure==2&&*count)types[*count-1].layout=0;
    return valid;
}
static void *graph_realloc(void *p,size_t n) {
    return failure==3&&requested==4?NULL:realloc(p,n);
}
#define nvm_affine_state_create graph_contract
#define nvm_affine_value_parameters graph_parameters
#define realloc graph_realloc
#include "../../src/nanovm/vm.c"
#undef realloc
#undef nvm_affine_value_parameters
#undef nvm_affine_state_create
int main(void) {
    (void)ordinary_chain;(void)graph_refusals;(void)artifacts;
    NvmModule *m=graph_fixture(0,8);consuming_verified(m);
    for(unsigned kind=1;kind<=3;kind++)for(unsigned api=0;api<4;api++) {
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        vm.stack_capacity=66; /* Four frames plus two prepared arguments fit; the fifth frame does not. */failure=kind;requested=attempts=0;
        for(unsigned retry=0;retry<2;retry++) {
            NanoValue result=val_void();
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            if(!retry){CHECK(status==(kind==2?VM_ERR_TYPE_ERROR:VM_ERR_MEMORY));CHECK(attempts==1);CHECK(vm.reference_generation==4);}
            else {CHECK(status==VM_OK);if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);}
            CHECK(!vm.stack_size&&!vm.frame_count);CHECK(!vm.references.active&&!vm.callee_references.active);
            for(unsigned f=0;f<NVM_OWNED_MAX_FUNCTIONS-2;f++)CHECK(!vm.value_references[f].active);
            CHECK(vm.heap.stats.num_objects==baseline);failure=0;
        }
        vm_destroy(&vm);
    }
    nvm_module_free(m);printf("%u graph preflight checks passed\n",checks);return 0;
}
