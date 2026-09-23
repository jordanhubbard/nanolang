#define main scalar_global_runtime_fixture_main
#include "test_owned_scalar_global_runtime.c"
#undef main
static unsigned allocation_attempt,allocation_failure;
static bool sustained;
static bool allocation_refused(void) {
    ++allocation_attempt;
    return allocation_failure && (sustained ? allocation_attempt>=allocation_failure : allocation_attempt==allocation_failure);
}
void *owned_global_malloc(size_t n){if(allocation_refused())return NULL;return malloc(n);}
void *owned_global_calloc(size_t n,size_t s){if(allocation_refused())return NULL;return calloc(n,s);}
void *owned_global_realloc(void *p,size_t n){if(allocation_refused())return NULL;return realloc(p,n);}
int main(void) {
    for(unsigned persistent=0;persistent<2;persistent++)
    for(unsigned trap=0;trap<2;trap++)for(unsigned dispatch=0;dispatch<2;dispatch++) {
        const char *consume=strstr(body,"LOAD_LOCAL 0");CHECK(consume);
        char code[4096];snprintf(code,sizeof(code),"PUSH_STR 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nPUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 0\n%s%s",trap?"PUSH_BOOL 0\nASSERT\n":"",consume);
        NvmModule *m=owned_union_fixture(code,0,TAG_INT);
        helper(m,"PUSH_STR 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n");
        uint8_t tag=TAG_STRING;attach(m,&tag,1);CHECK(nvm_verify(m).ok);
        for(unsigned failure=1;;failure++) {
            VmState vm;sustained=persistent;allocation_attempt=0;allocation_failure=failure;
            vm_init(&vm,m);NanoValue result=val_void();VmResult status=vm.last_error;
            if(!persistent && failure==1)CHECK(vm.stack==NULL);
            if(status==VM_OK) {
                size_t baseline=vm.heap.stats.num_objects;
                vm_set_dispatch_profile(&vm,dispatch?vm_dispatch_profile_all():vm_dispatch_profile_none());
                status=vm_invoke(&vm,0,NULL,0,&result);
                vm_gc_collect_cycles(&vm.heap);CHECK(vm.heap.stats.num_objects==baseline);
            }
            allocation_failure=0;
            CHECK(vm.stack_size==0 && vm.frame_count==0);
            if(allocation_attempt>=failure && status!=VM_ERR_MEMORY && (persistent || failure>2))fprintf(stderr,"allocation %u/%u trap=%u dispatch=%u status=%d: %s\n",failure,allocation_attempt,trap,dispatch,(int)status,vm.error_msg);
            vm_destroy(&vm);
            if(allocation_attempt<failure) {
                CHECK(status==(trap?VM_ERR_ASSERT_FAILED:VM_OK));
                if(!trap)CHECK(result.tag==TAG_INT && result.as.i64==7);
                break;
            }
            /* Initial stack reservation and intern-table setup have bounded retry paths.
             * Every subsequent allocation in this fixture is required. */
            if(!persistent && failure<=2) {
                CHECK(status==(trap?VM_ERR_ASSERT_FAILED:VM_OK));
                if(!trap)CHECK(result.tag==TAG_INT && result.as.i64==7);
            } else CHECK(status==VM_ERR_MEMORY);
            CHECK(failure<4096);
        }
        nvm_module_free(m);
    }
    printf("I passed %u scalar-global VM allocation checks.\n",checks);return 0;
}
