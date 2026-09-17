#define main owned_fixture_main
#include "test_owned_runtime.c"
#undef main
#include "../../src/runtime/callback_runtime.h"
#include "disassembler.h"
#define START "PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nREGION_BEGIN\n"
#define FINISH "REGION_END\nOWN_UNPACK_LOCAL 0\nRET\n"
static const char *loop = START "BORROW_LOCAL_EXCLUSIVE 0 0\nloop:\nREF_GET 0 0\nPUSH_I64 1\nADD\nREF_SET 0 0\nREF_GET 0 0\nPUSH_I64 1000\nLT\nJMP_TRUE loop\n" FINISH;
static void transport(void) {
    const char *names[]={"REGION_BEGIN","REGION_END","BORROW_LOCAL_SHARED","BORROW_LOCAL_EXCLUSIVE","REF_GET","REF_SET"};
    for (unsigned i=0;i<6;i++) {
        uint8_t op=isa_opcode_by_name(names[i]);CHECK(op==0x16+i);
        DecodedInstruction in={0},out={0};in.opcode=op;
        in.operands[0].u16=0x1234;in.operands[1].u16=0x5678;
        uint8_t bytes[ISA_MAX_INSTRUCTION_SIZE];unsigned size=isa_encode(&in,bytes,sizeof(bytes));
        CHECK(size==(i<2?1:5));CHECK(isa_decode(bytes,size,&out)==size && out.opcode==op);
        if(i>=2) CHECK(bytes[1]==0x34 && bytes[2]==0x12 && bytes[3]==0x78 && bytes[4]==0x56 &&
                       out.operands[0].u16==0x1234 && out.operands[1].u16==0x5678);
    }
    NvmModule *m=fixture(START "BORROW_LOCAL_SHARED 0 0\nREF_GET 0 0\nPOP\nREGION_END\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_I64 42\nREF_SET 0 0\n" FINISH,false,false);
    uint32_t size;CHECK(nvm_serialize(m,&size)==NULL);
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
    for(unsigned i=0;i<6;i++)CHECK(strstr(text,names[i]));
    AsmResult error;NvmModule *copy=asm_assemble(text,&error);CHECK(copy);
    CHECK(copy->code_size==m->code_size && !memcmp(copy->code,m->code,m->code_size));
    CHECK(nvm_verify(copy).ok);nvm_module_free(copy);free(text);
    free(m->ownership_data);m->ownership_data=NULL;m->ownership_size=0;
    CHECK(!nvm_verify(m).ok);char message[256];CHECK(!nvm2c_emit(m,message,sizeof(message)));
    VmState vm;vm_init(&vm,m);NanoValue result=val_void();CHECK(vm_invoke(&vm,0,NULL,0,&result)!=VM_OK);vm_destroy(&vm);nvm_module_free(m);
}
static void resume_and_relocate(void) {
    NvmModule *m=fixture(loop,false,false);CHECK(nvm_verify(m).ok);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    vm.callbacks=nano_callback_runtime_create();CHECK(vm.callbacks);
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0]=(VmCallFrame){.fn_idx=0,.local_count=m->functions[0].local_count,.module=m};
    vm.stack_size=m->functions[0].local_count;
    VmTrap trap=vm_core_execute(&vm);CHECK(trap.type==TRAP_YIELD);
    CHECK(vm.references.active && vm.references.region==1 && vm.references.slots[0].region==1);
    CHECK(vm_call_function(&vm,0,NULL,0)==VM_ERR_TYPE_ERROR);
    CHECK(vm.references.active && vm.references.slots[0].region==1);
    /* A suspended owner remains the same value after the host relocates its
     * value storage. References carry indices, never addresses into it. */
    NanoValue *next=calloc(vm.stack_capacity,sizeof(*next));CHECK(next);
    memcpy(next,vm.stack,vm.stack_size*sizeof(*next));free(vm.stack);vm.stack=next;
    unsigned yields=1;
    do {trap=vm_core_execute(&vm);if(trap.type==TRAP_YIELD)yields++;} while(trap.type==TRAP_YIELD);
    CHECK(trap.type==TRAP_NONE);CHECK(yields>1);
    CHECK(!vm.references.active && !vm.references.region && !vm.references.slots[0].region);
    CHECK(vm.frame_count==0 && vm.stack_size==1 && vm.stack[0].as.i64==1000);
    CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);nvm_module_free(m);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    execute(START "BORROW_LOCAL_SHARED 0 0\nBORROW_LOCAL_SHARED 1 0\nREF_GET 0 0\nREF_GET 1 0\nADD\nSTORE_LOCAL 4\nREGION_END\nOWN_UNPACK_LOCAL 0\nLOAD_LOCAL 4\nADD\nRET\n",30,argv[1],0,TAG_INT);
    execute(START "BORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_I64 42\nREF_SET 0 0\n" FINISH,42,argv[1],1,TAG_INT);
    execute(START "BORROW_LOCAL_SHARED 0 0\nREF_GET 0 0\nPOP\nREGION_END\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_I64 71\nREF_SET 0 0\n" FINISH,71,argv[1],2,TAG_INT);
    execute(START "BORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 42\nREF_SET 0 0\nJMP join\nother:\nPUSH_I64 8\nREF_SET 0 0\njoin:\n" FINISH,42,argv[1],3,TAG_INT);
    execute(loop,1000,argv[1],4,TAG_INT);
    execute("PUSH_I64 10\nOWN_PACK 0\nPUSH_I64 32\nOWN_PACK 0\nOWN_PACK 2\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 0\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 0\nBORROW_LOCAL_EXCLUSIVE 1 3\nREF_GET 0 0\nREF_GET 1 0\nADD\nREF_SET 1 0\nREGION_END\nOWN_UNPACK_LOCAL 0\nPOP\nOWN_UNPACK_LOCAL 3\nRET\n",42,argv[1],5,TAG_INT);
    execute("PUSH_BOOL 0\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_BOOL 1\nREF_SET 0 0\n" FINISH,1,argv[1],6,TAG_BOOL);
    execute("PUSH_U8 0\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nREGION_BEGIN\nBORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_U8 255\nREF_SET 0 0\n" FINISH,255,argv[1],7,TAG_U8);
    execute(START "BORROW_LOCAL_SHARED 0 0\nREGION_BEGIN\nBORROW_LOCAL_SHARED 1 0\nREF_GET 1 0\nPOP\nREGION_END\nREF_GET 0 0\nPOP\n" FINISH,10,argv[1],8,TAG_INT);
    const char *bad[]={
        START "BORROW_LOCAL_SHARED 0 0\nBORROW_LOCAL_EXCLUSIVE 1 0\n" FINISH,
        START "BORROW_LOCAL_EXCLUSIVE 0 0\nBORROW_LOCAL_SHARED 1 0\n" FINISH,
        START "BORROW_LOCAL_SHARED 0 0\nPUSH_I64 1\nREF_SET 0 0\n" FINISH,
        START "BORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_BOOL 1\nREF_SET 0 0\n" FINISH,
        START "BORROW_LOCAL_SHARED 0 0\nREGION_END\nREF_GET 0 0\nPOP\nOWN_UNPACK_LOCAL 0\nRET\n",
        START "BORROW_LOCAL_SHARED 0 0\nOWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 3\nREGION_END\nOWN_UNPACK_LOCAL 3\nRET\n",
        START "BORROW_LOCAL_SHARED 0 0\nOWN_UNPACK_LOCAL 0\nREGION_END\nRET\n",
        START "BORROW_LOCAL_EXCLUSIVE 0 0\nPUSH_I64 0\nRET\n",
        START "PUSH_BOOL 1\nJMP_FALSE other\nBORROW_LOCAL_SHARED 0 0\nJMP join\nother:\nBORROW_LOCAL_SHARED 1 0\njoin:\n" FINISH,
        START "BORROW_LOCAL_SHARED 5 0\n" FINISH,
        START "BORROW_LOCAL_SHARED 0 0\nREF_GET 0 1\nPOP\n" FINISH,
        START "BORROW_LOCAL_EXCLUSIVE 0 0\nLOAD_LOCAL 0\nAGG_GET 0\nPOP\n" FINISH,
        START "LOAD_LOCAL 0\nBORROW_LOCAL_EXCLUSIVE 0 0\nAGG_GET 0\nPOP\n" FINISH,
        "PUSH_I64 10\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nBORROW_LOCAL_SHARED 0 0\nOWN_UNPACK_LOCAL 0\nRET\n",
        START "BORROW_LOCAL_SHARED 0 0\nBORROW_LOCAL_SHARED 0 0\n" FINISH,
        "PUSH_I64 1\nOWN_PACK 0\nPUSH_I64 2\nOWN_PACK 0\nOWN_PACK 2\nOWN_STORE_LOCAL 2\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 2\nREGION_END\nOWN_UNPACK_LOCAL 2\nOWN_STORE_LOCAL 0\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 0\nOWN_UNPACK_LOCAL 3\nADD\nRET\n"
    };
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++)refused(fixture(bad[i],false,false),argv[1],i);
    transport();
    resume_and_relocate();
    printf("%u same-frame reference checks passed\n",checks);return 0;
}
