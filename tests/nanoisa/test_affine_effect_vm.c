/* I release suspended owners and handler roots on success and failure. */
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanovm/vm.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int g_argc=0;char **g_argv=NULL;
static size_t attempt,fail_at;
void *effect_vm_malloc(size_t n){if(++attempt==fail_at)return NULL;return malloc(n);}
void *effect_vm_calloc(size_t n,size_t s){if(++attempt==fail_at)return NULL;return calloc(n,s);}
void *effect_vm_realloc(void *p,size_t n){if(++attempt==fail_at)return NULL;return realloc(p,n);}
static size_t execute(NvmModule *module,size_t failure,int expected) {
    VmState *vm=malloc(sizeof(*vm));assert(vm);fail_at=0;vm_init(vm,module);
    size_t baseline=vm->heap.stats.num_objects;
    size_t count=0;
    for(unsigned round=0;round<2;round++) {
        attempt=0;fail_at=round?0:failure;NanoValue result=val_void();
        VmResult status=vm_invoke(vm,0,NULL,0,&result);fail_at=0;
        if(!round && failure && failure<=attempt)assert(status==VM_ERR_MEMORY);
        else if(!expected){assert(status==VM_OK);assert(result.tag==TAG_INT && result.as.i64==7);}
        else if(expected==2)assert(status==VM_ERR_ASSERT_FAILED);
        else assert(status==VM_ERR_TYPE_ERROR || status==VM_ERR_CALL_DEPTH);
        assert(vm->stack_size==0 && vm->frame_count==0 && vm->handler_count==0);
        assert(!vm->references.active && !vm->callee_references.active);
        for(unsigned i=0;i<NVM_OWNED_MAX_FUNCTIONS-2;i++)assert(!vm->value_references[i].active);
        for(unsigned i=0;i<VM_MAX_FRAMES-NVM_OWNED_MAX_FUNCTIONS;i++)assert(!vm->effect_references[i]);
        vm_gc_collect_cycles(&vm->heap);assert(vm->heap.stats.num_objects==baseline);
        if(!round)count=attempt;
    }
    vm_destroy(vm);free(vm);return count;
}
int main(int argc,char **argv) {
    assert(argc==3);NanoisaErr error;NvmModule *module=nanoisa_load_file(argv[1],&error);assert(module);
    int expected=atoi(argv[2]);size_t count=execute(module,0,expected);
    for(size_t failure=1;failure<=count;failure++)execute(module,failure,expected);
    execute(module,0,expected);nvm_module_free(module);
    printf("I check %zu VM allocation positions and reuse after every outcome.\n",count);return 0;
}
