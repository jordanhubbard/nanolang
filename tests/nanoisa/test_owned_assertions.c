#define CALLER_RUNTIME_TEST
#include "test_caller_reference_analysis.c"
#include "nvm2c.h"
#include "../../src/nanovm/vm.h"
#include "../../src/runtime/callback_runtime.h"
int g_argc=0;char **g_argv=NULL;

static void artifacts(NvmModule *m,const char *dir,unsigned index) {
    NvmV2Module v2;size_t size;
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    char path[1024];snprintf(path,sizeof(path),"%s/case%u.nvm",dir,index);
    FILE *file=fopen(path,"wb");CHECK(file);CHECK(fwrite(bytes,1,size,file)==size);CHECK(!fclose(file));
    free(bytes);nvm_v2_module_free(&v2);
    char error[256];char *source=nvm2c_emit(m,error,sizeof(error));
    if(!source)fprintf(stderr,"%s\n",error);
    CHECK(source);snprintf(path,sizeof(path),"%s/case%u.c",dir,index);
    file=fopen(path,"w");CHECK(file);CHECK(fputs(source,file)>=0);CHECK(!fclose(file));free(source);
}

static void run_case(const char *caller,const char *helper,bool succeeds,const char *dir,unsigned index) {
    NvmModule *m=call_fixture(caller,helper);
    NvmVerifyResult verified=nvm_verify(m);
    if(!verified.ok)fprintf(stderr,"%s\n",verified.error_msg);
    CHECK(verified.ok);CHECK(nvm_verify_owned_module(m).ok);
    artifacts(m,dir,index);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    for(unsigned api=0;api<3;api++)for(unsigned repetition=0;repetition<8;repetition++) {
        NanoValue result=val_void();
        VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):
            api==1?vm_execute(&vm):vm_call_function(&vm,0,NULL,0);
        CHECK(status==(succeeds?VM_OK:VM_ERR_ASSERT_FAILED));
        if(succeeds) {
            if(api) {CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
            CHECK(result.tag==TAG_INT && result.as.i64==0);
            vm_release(&vm.heap,result);
        } else CHECK(strstr(vm.error_msg,"Assertion failed")!=NULL);
        CHECK(vm.stack_size==0 && vm.frame_count==0);
        CHECK(!vm.references.active && !vm.callee_references.active);
        CHECK(vm.heap.stats.num_objects==baseline);
    }
    vm_destroy(&vm);nvm_module_free(m);
    printf("case %u %u\n",index,succeeds?1:0);
}

static void resumed_assertion(void) {
    char helper[8192];strcpy(helper,"REGION_BEGIN\nREBORROW_SHARED 1 0\n");
    for(unsigned i=0;i<1200;i++)strcat(helper,"NOP\n");
    strcat(helper,"PUSH_BOOL 1\nASSERT\nREF_GET 1 0\nREGION_END\nRET");
    NvmModule *m=call_fixture("CALL_REF 1 0",helper);CHECK(nvm_verify(m).ok);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    vm.callbacks=nano_callback_runtime_create();CHECK(vm.callbacks);
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0]=(VmCallFrame){.fn_idx=0,.local_count=m->functions[0].local_count,.module=m};
    vm.stack_size=m->functions[0].local_count;
    VmTrap trap=vm_core_execute(&vm);CHECK(trap.type==TRAP_YIELD);
    CHECK(vm.references.active && vm.callee_references.active && vm.frame_count==2);
    NanoValue *moved=calloc(vm.stack_capacity*2,sizeof(*moved));CHECK(moved);
    memcpy(moved,vm.stack,vm.stack_size*sizeof(*moved));free(vm.stack);
    vm.stack=moved;vm.stack_capacity*=2;
    unsigned assertions=0;
    do {
        trap=vm_core_execute(&vm);
        if(trap.type==TRAP_ASSERT) {
            CHECK(trap.data.assert_check.condition.tag==TAG_BOOL);
            CHECK(val_truthy(trap.data.assert_check.condition));
            CHECK(vm.references.active && vm.callee_references.active);
            vm_release(&vm.heap,trap.data.assert_check.condition);assertions++;
        }
    } while(trap.type==TRAP_YIELD || trap.type==TRAP_ASSERT);
    CHECK(trap.type==TRAP_NONE && assertions==1);
    CHECK(!vm.references.active && !vm.callee_references.active);
    CHECK(vm.stack_size==1 && vm.stack[0].tag==TAG_INT && vm.stack[0].as.i64==42);
    CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);nvm_module_free(m);
}

int main(int argc,char **argv) {
    CHECK(argc==2);
    run_case("PUSH_BOOL 1\nASSERT\nCALL_REF 1 0","PUSH_I64 -32\nREF_SET 0 0\nREF_GET 0 0\nRET",true,argv[1],0);
    run_case("PUSH_BOOL 0\nASSERT\nCALL_REF 1 0","REF_GET 0 0\nRET",false,argv[1],1);
    run_case("CALL_REF 1 0","PUSH_BOOL 1\nASSERT\nPUSH_I64 -32\nREF_SET 0 0\nREF_GET 0 0\nRET",true,argv[1],2);
    run_case("CALL_REF 1 0","REGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nPUSH_BOOL 0\nASSERT\nREGION_END\nREF_GET 0 0\nRET",false,argv[1],3);
    resumed_assertion();
    printf("%u owned assertion lifecycle checks passed\n",checks);return 0;
}
