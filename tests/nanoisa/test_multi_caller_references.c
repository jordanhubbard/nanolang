#define CALLER_ALLOC_TEST
#include "test_caller_references.c"
#include "multi_caller_fixture.h"
static const uint8_t exclusive[9]={2,2,2,2,2,2,2,2,2};
static const uint8_t shared[2]={1,1},mixed[2]={1,2};
static NvmModule *eight_arguments(void) {
    char body[8192]="",helper[8192]="",line[128];
    for(unsigned i=0;i<8;i++){snprintf(line,sizeof(line),"PUSH_I64 %u\nOWN_PACK 0\nOWN_STORE_LOCAL %u\n",10+i,2+i);strcat(body,line);}
    strcat(body,"REGION_BEGIN\n");
    for(unsigned i=0;i<8;i++){snprintf(line,sizeof(line),"BORROW_LOCAL_EXCLUSIVE %u %u\n",i,2+i);strcat(body,line);}
    strcat(body,"CALL_REF 1 0\nPOP\nREGION_END\nPUSH_I64 0\n");
    for(unsigned i=0;i<8;i++){snprintf(line,sizeof(line),"OWN_UNPACK_LOCAL %u\nADD\n",2+i);strcat(body,line);}
    strcat(body,"RET");
    for(unsigned i=0;i<8;i++){snprintf(line,sizeof(line),"PUSH_I64 %u\nREF_SET %u 0\n",100+i,i);strcat(helper,line);}
    strcat(helper,"PUSH_I64 0\n");
    for(unsigned i=0;i<8;i++){snprintf(line,sizeof(line),"REF_GET %u 0\nADD\n",i);strcat(helper,line);}
    strcat(helper,"RET");return multi_fixture(body,helper,8,exclusive);
}
static void resume_multiple(void) {
    char helper[8192]="REGION_BEGIN\nREBORROW_EXCLUSIVE 2 0\nREBORROW_EXCLUSIVE 3 1\n";
    for(unsigned i=0;i<1200;i++)strcat(helper,"NOP\n");
    strcat(helper,"PUSH_I64 50\nREF_SET 2 0\nPUSH_I64 70\nREF_SET 3 0\nREGION_END\n" READ_BOTH);
    NvmModule *m=multi_fixture(PAIR_ROOT "REGION_BEGIN\nBORROW_PATH_EXCLUSIVE 0 10 0\nBORROW_PATH_EXCLUSIVE 1 10 1\nCALL_REF 1 0\nPOP\n" PAIR_SUM,helper,2,exclusive);
    CHECK(nvm_verify(m).ok);VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    vm.callbacks=nano_callback_runtime_create();CHECK(vm.callbacks);
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0]=(VmCallFrame){.fn_idx=0,.local_count=16,.module=m};vm.stack_size=16;
    VmTrap trap=vm_core_execute(&vm);CHECK(trap.type==TRAP_YIELD);
    CHECK(vm.frame_count==2 && vm.callee_references.slots[2].live && vm.callee_references.slots[3].live);
    CHECK(vm.callee_references.slots[2].root==10 && vm.callee_references.slots[3].root==10);
    CHECK(vm.callee_references.slots[2].path==0 && vm.callee_references.slots[3].path==1);
    NanoValue *next=calloc(vm.stack_capacity*2,sizeof(*next));CHECK(next);
    memcpy(next,vm.stack,vm.stack_size*sizeof(*next));free(vm.stack);vm.stack=next;vm.stack_capacity*=2;
    do{trap=vm_core_execute(&vm);}while(trap.type==TRAP_YIELD);
    CHECK(trap.type==TRAP_NONE && !vm.references.active && !vm.callee_references.active);
    CHECK(vm.stack_size==1 && vm.stack[0].as.i64==120);
    CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);nvm_module_free(m);
}
int multi_runtime_cases(int argc,char **argv) {
    CHECK(argc==2);
    execute_module(multi_fixture("PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 2\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 2\nBORROW_LOCAL_SHARED 1 2\nCALL_REF 1 0\nSTORE_LOCAL 0\nREGION_END\nOWN_UNPACK_LOCAL 2\nPOP\nLOAD_LOCAL 0\nRET",READ_BOTH,2,shared),20,argv[1],0,TAG_INT);
    execute_module(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\n" TWO_SUM,WRITE_BOTH,2,exclusive),49,argv[1],1,TAG_INT);
    execute_module(multi_fixture(TWO_ROOTS "REGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 3\nBORROW_LOCAL_EXCLUSIVE 1 2\nCALL_REF 1 0\nPOP\nREGION_END\nOWN_UNPACK_LOCAL 2\nPUSH_I64 100\nMUL\nOWN_UNPACK_LOCAL 3\nADD\nRET",WRITE_BOTH,2,exclusive),742,argv[1],2,TAG_INT);
    execute_module(multi_fixture(PAIR_ROOT "REGION_BEGIN\nBORROW_PATH_EXCLUSIVE 0 10 0\nBORROW_PATH_EXCLUSIVE 1 10 1\nCALL_REF 1 0\nPOP\n" PAIR_SUM,WRITE_BOTH,2,exclusive),49,argv[1],3,TAG_INT);
    execute_module(multi_fixture(PAIR_ROOT "REGION_BEGIN\nBORROW_PATH_SHARED 0 10 0\nBORROW_PATH_SHARED 1 10 2\nCALL_REF 1 0\nSTORE_LOCAL 0\nREGION_END\nOWN_UNPACK_LOCAL 10\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nPOP\nOWN_UNPACK_LOCAL 3\nPOP\nLOAD_LOCAL 0\nRET",READ_BOTH,2,shared),20,argv[1],4,TAG_INT);
    execute_module(multi_fixture(TWO_ROOTS "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 2\nBORROW_LOCAL_EXCLUSIVE 1 3\nCALL_REF 1 0\nPOP\n" TWO_SUM,"REF_GET 0 0\nPUSH_I64 1\nADD\nREF_SET 1 0\nREF_GET 1 0\nRET",2,mixed),21,argv[1],5,TAG_INT);
    execute_module(eight_arguments(),828,argv[1],6,TAG_INT);
    execute_module(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\nCALL_REF 1 0\nPOP\n" TWO_SUM,"REF_GET 0 0\nPUSH_I64 1\nADD\nREF_SET 0 0\nREF_GET 1 0\nPUSH_I64 1\nADD\nREF_SET 1 0\n" READ_BOTH,2,exclusive),46,argv[1],7,TAG_INT);
    execute_module(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\n" TWO_SUM,"REGION_BEGIN\nREBORROW_EXCLUSIVE 2 0\nREBORROW_EXCLUSIVE 3 1\nPUSH_I64 50\nREF_SET 2 0\nPUSH_I64 70\nREF_SET 3 0\nREGION_END\n" READ_BOTH,2,exclusive),120,argv[1],8,TAG_INT);
    execute_module(multi_fixture("PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 2\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 2\nREGION_BEGIN\nREBORROW_SHARED 1 0\nCALL_REF 1 0\nPOP\nREGION_END\nPUSH_I64 42\nREF_SET 0 0\nREGION_END\nOWN_UNPACK_LOCAL 2\nRET",READ_BOTH,2,shared),42,argv[1],9,TAG_INT);
    execute_module(multi_fixture(TWO_ROOTS "REGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 3 2\nBORROW_LOCAL_EXCLUSIVE 4 3\nCALL_REF 1 3\nPOP\n" TWO_SUM,WRITE_BOTH,2,exclusive),49,argv[1],10,TAG_INT);
    NvmModule *different=multi_fixture("PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 2\nPUSH_I64 32\nOWN_PACK 2\nOWN_STORE_LOCAL 11\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 2\nBORROW_LOCAL_EXCLUSIVE 1 11\nCALL_REF 1 0\nPOP\nREGION_END\nOWN_UNPACK_LOCAL 2\nOWN_UNPACK_LOCAL 11\nADD\nRET",WRITE_BOTH,2,exclusive);
    slot(different->ownership_data+176,TAG_STRUCT,2,2);
    execute_module(different,49,argv[1],11,TAG_INT);
    const char *badhelpers[]={"LOAD_LOCAL 1\nAGG_GET 0\nRET","OWN_MOVE_LOCAL 1\nRET","PUSH_I64 1\nSTORE_LOCAL 1\nPUSH_I64 0\nRET","REGION_BEGIN\n" READ_BOTH};
    for(unsigned i=0;i<4;i++)refused(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\n" TWO_SUM,badhelpers[i],2,exclusive),argv[1],i);
    refused(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 65535\nPOP\n" TWO_SUM,READ_BOTH,2,exclusive),argv[1],4);
    refused(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 15\nPOP\n" TWO_SUM,READ_BOTH,2,exclusive),argv[1],5);
    refused(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 1\nPOP\n" TWO_SUM,READ_BOTH,2,exclusive),argv[1],6);
    refused(multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\n" TWO_SUM,READ_BOTH,9,exclusive),argv[1],7);
    NvmModule *bad=multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\n" TWO_SUM,READ_BOTH,2,exclusive);
    slot(bad->ownership_data+176,TAG_STRUCT,2,2);refused(bad,argv[1],8);
    refused(multi_fixture(TWO_ROOTS TWO_BORROWS "REGION_BEGIN\nREBORROW_EXCLUSIVE 3 0\nCALL_REF 1 0\nPOP\nREGION_END\n" TWO_SUM,READ_BOTH,2,exclusive),argv[1],9);
    refused(multi_fixture("PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 2\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 2\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nCALL_REF 1 0\nPOP\nREGION_END\nREGION_END\nOWN_UNPACK_LOCAL 2\nRET",READ_BOTH,2,exclusive),argv[1],10);
    refused(multi_fixture("PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 2\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 2\nREGION_BEGIN\nREBORROW_SHARED 1 0\nCALL_REF 1 0\nPOP\nPUSH_I64 42\nREF_SET 0 0\nREGION_END\nREGION_END\nOWN_UNPACK_LOCAL 2\nRET",READ_BOTH,2,shared),argv[1],11);
    refused(multi_fixture(TWO_ROOTS "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 2\nOWN_MOVE_LOCAL 2\nPOP\nBORROW_LOCAL_SHARED 1 3\nCALL_REF 1 0\nPOP\n" TWO_SUM,READ_BOTH,2,shared),argv[1],12);
    resume_multiple();printf("%u multi-caller checks passed\n",checks);return 0;
}
#ifndef MULTI_CALLER_ALLOC_TEST
int main(int argc,char **argv){return multi_runtime_cases(argc,argv);}
#endif
