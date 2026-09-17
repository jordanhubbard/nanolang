#define main owned_fixture_main
#include "test_owned_runtime.c"
#undef main
#include "nested_reference_fixture.h"
#include "../../src/runtime/callback_runtime.h"
#include "disassembler.h"
static void nested_execute(const char *body,int64_t result,const char *dir,unsigned n) {
    execute_module(nested_fixture(body),result,dir,n,TAG_INT);
}
static void resume_nested(void) {
    char body[8192];strcpy(body,NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\n");
    for(unsigned i=0;i<1200;i++)strcat(body,"NOP\n");
    strcat(body,"PUSH_I64 42\nREF_SET 1 0\nREGION_END\nREF_GET 0 0\nPOP\n" NESTED_FINISH);
    NvmModule *m=nested_fixture(body);CHECK(nvm_verify(m).ok);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    vm.callbacks=nano_callback_runtime_create();CHECK(vm.callbacks);
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0]=(VmCallFrame){.fn_idx=0,.local_count=m->functions[0].local_count,.module=m};
    vm.stack_size=m->functions[0].local_count;
    VmTrap trap=vm_core_execute(&vm);CHECK(trap.type==TRAP_YIELD);
    CHECK(vm.references.region==2 && vm.references.slots[1].region==2 &&
          vm.references.slots[1].parent==0 && vm.references.slots[1].path==0);
    NanoValue *next=calloc(vm.stack_capacity,sizeof(*next));CHECK(next);
    memcpy(next,vm.stack,vm.stack_size*sizeof(*next));free(vm.stack);vm.stack=next;
    do {trap=vm_core_execute(&vm);} while(trap.type==TRAP_YIELD);
    CHECK(trap.type==TRAP_NONE && !vm.references.active);
    CHECK(vm.stack_size==1 && vm.stack[0].tag==TAG_INT && vm.stack[0].as.i64==74);
    CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);nvm_module_free(m);
}
static void transport_nested(void) {
    const char *names[]={"BORROW_PATH_SHARED","BORROW_PATH_EXCLUSIVE","REBORROW_SHARED","REBORROW_EXCLUSIVE"};
    for(unsigned i=0;i<4;i++) {
        uint8_t op=isa_opcode_by_name(names[i]);CHECK(op==0x1c+i);
        DecodedInstruction in={0},out={0};in.opcode=op;in.operands[0].u16=1;in.operands[1].u16=2;in.operands[2].u32=3;
        uint8_t bytes[ISA_MAX_INSTRUCTION_SIZE];unsigned n=isa_encode(&in,bytes,sizeof(bytes));
        CHECK(n==(i<2?9:5));CHECK(isa_decode(bytes,n,&out)==n && out.opcode==op);
        CHECK(out.operands[0].u16==1 && out.operands[1].u16==2);
        if(i<2)CHECK(out.operands[2].u32==3);
    }
    NvmModule *m=nested_fixture(NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nREF_GET 1 0\nPOP\nREGION_END\n" NESTED_FINISH);
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult error;NvmModule *copy=asm_assemble(text,&error);CHECK(copy);
    CHECK(copy->code_size==m->code_size && !memcmp(copy->code,m->code,m->code_size));
    CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
    nvm_module_free(copy);free(text);nvm_module_free(m);
}
static NvmModule *maximum_depth_fixture(void) {
    char body[8192]="PUSH_I64 10\n",line[128];
    for(unsigned i=0;i<=32;i++){snprintf(line,sizeof(line),"OWN_PACK %u\n",i);strcat(body,line);}
    strcat(body,"OWN_STORE_LOCAL 32\nREGION_BEGIN\nBORROW_PATH_EXCLUSIVE 0 32 0\nPUSH_I64 42\nREF_SET 0 0\nREGION_END\n");
    for(unsigned i=32;i>0;i--){snprintf(line,sizeof(line),"OWN_UNPACK_LOCAL %u\nOWN_STORE_LOCAL %u\n",i,i-1);strcat(body,line);}
    strcat(body,"OWN_UNPACK_LOCAL 0\nRET\n");
    NvmModule *m=fixture(body,false,false);m->struct_count=33;m->functions[0].local_count=33;
    NvmV2Layout items[33];NvmV2LayoutField fields[33];
    for(unsigned i=0;i<33;i++) {
        fields[i]=(NvmV2LayoutField){i?TAG_STRUCT:TAG_INT,i?i-1:NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
        items[i]=(NvmV2Layout){NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&fields[i]};
    }
    NvmV2Layouts layouts={items,33};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    free(m->ownership_data);m->ownership_size=396;m->ownership_data=calloc(396,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,33);memset(data+8,3,33);
    word(data+44,1);data[48]=33;slot(data+52,TAG_INT,NVM_V2_NO_INDEX);
    for(unsigned i=0;i<33;i++)slot(data+60+i*8,TAG_STRUCT,i);
    word(data+324,1);data[328]=32;
    return m;
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    nested_execute(NESTED_START "BORROW_PATH_SHARED 0 2 0\nBORROW_PATH_SHARED 1 2 1\nREF_GET 0 0\nREF_GET 1 0\nADD\nPOP\n" NESTED_FINISH,42,argv[1],0);
    nested_execute(NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nBORROW_PATH_EXCLUSIVE 1 2 1\nPUSH_I64 41\nREF_SET 0 0\nPUSH_I64 1\nREF_SET 1 0\n" NESTED_FINISH,42,argv[1],1);
    nested_execute(NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nPUSH_I64 42\nREF_SET 1 0\nREGION_END\nREF_GET 0 0\nPOP\n" NESTED_FINISH,74,argv[1],2);
    nested_execute(NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nREF_GET 1 0\nREF_GET 0 0\nADD\nPOP\nREGION_END\nPUSH_I64 11\nREF_SET 0 0\n" NESTED_FINISH,43,argv[1],3);
    nested_execute(NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nREGION_BEGIN\nREBORROW_EXCLUSIVE 2 1\nPUSH_I64 40\nREF_SET 2 0\nREGION_END\nREF_GET 1 0\nPUSH_I64 1\nADD\nREF_SET 1 0\nREGION_END\nREF_GET 0 0\nPUSH_I64 1\nADD\nREF_SET 0 0\n" NESTED_FINISH,74,argv[1],4);
    nested_execute(NESTED_START "BORROW_PATH_SHARED 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nREF_GET 1 0\nPOP\nREGION_END\nREF_GET 0 0\nPOP\n" NESTED_FINISH,42,argv[1],5);
    nested_execute(NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nREF_GET 1 0\nPOP\nREGION_END\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nPUSH_I64 7\nREF_SET 1 0\nREGION_END\n" NESTED_FINISH,39,argv[1],6);
    nested_execute(NESTED_START "PUSH_BOOL 1\nJMP_FALSE other\nBORROW_PATH_SHARED 0 2 0\nJMP join\nother:\nBORROW_PATH_SHARED 0 2 2\njoin:\nREF_GET 0 0\nPOP\n" NESTED_FINISH,42,argv[1],7);
    nested_execute(NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nloop:\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nREF_GET 1 0\nPUSH_I64 1\nADD\nREF_SET 1 0\nREGION_END\nREF_GET 0 0\nPUSH_I64 1000\nLT\nJMP_TRUE loop\n" NESTED_FINISH,1032,argv[1],8);
    nested_execute(NESTED_ALLOCATION_BODY,42,argv[1],9);
    execute_module(maximum_depth_fixture(),42,argv[1],10,TAG_INT);
    const char *bad[]={
        NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nBORROW_PATH_SHARED 1 2 2\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 0\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nREGION_END\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREBORROW_SHARED 1 0\n" NESTED_FINISH,
        NESTED_START "REGION_BEGIN\nREBORROW_SHARED 1 0\nREGION_END\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_EXCLUSIVE 1 0\nREF_GET 0 0\nPOP\nREGION_END\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nPUSH_I64 5\nREF_SET 0 0\nREGION_END\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nREGION_END\nREF_GET 1 0\nPOP\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 3\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 5\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 0\nOWN_MOVE_LOCAL 2\nOWN_STORE_LOCAL 2\n" NESTED_FINISH,
        NESTED_START "PUSH_BOOL 1\nJMP_FALSE other\nBORROW_PATH_SHARED 0 2 0\nJMP join\nother:\nBORROW_PATH_SHARED 0 2 1\njoin:\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 0\nPUSH_I64 0\nRET\n",
        NESTED_START "BORROW_PATH_EXCLUSIVE 0 2 0\nREGION_BEGIN\nREBORROW_SHARED 1 0\nREBORROW_EXCLUSIVE 3 0\nREGION_END\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 4\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 0\nREF_GET 0 1\nPOP\n" NESTED_FINISH,
        NESTED_START "BORROW_PATH_SHARED 0 2 0\nBORROW_PATH_SHARED 2 2 0\nREGION_BEGIN\nPUSH_BOOL 1\nJMP_FALSE other\nREBORROW_SHARED 1 0\nJMP join\nother:\nREBORROW_SHARED 1 2\njoin:\nREGION_END\n" NESTED_FINISH
    };
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++)refused(nested_fixture(bad[i]),argv[1],i);
    transport_nested();resume_nested();printf("%u nested reference checks passed\n",checks);return 0;
}
