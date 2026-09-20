/* I check exact byte conversion and owned argument recovery in verified modules. */
#include "nanoisa/assembler.h"
#include "nanoisa/disassembler.h"
#include "nanovm/vm.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;
char **g_argv;
static unsigned checks;
#define CHECK(x) do { checks++; assert(x); } while (0)
static const char prefix[] = ".entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n";
static NvmModule *assemble_body(const char *body, unsigned arity, unsigned locals) {
    char source[2048];
    int n=snprintf(source,sizeof source,"%s.function convert %u %u 0 u8 1\n%s\nCAST_U8\nRET\n.end\n",prefix,arity,locals,body);
    CHECK(n>0 && (size_t)n<sizeof source);
    AsmResult result; NvmModule *m=asm_assemble(source,&result);
    if(!m) fprintf(stderr,"I could not verify my fixture: %s\n",result.message);
    CHECK(m && result.error==ASM_OK); return m;
}
static void literal(const char *instruction, unsigned expected) {
    NvmModule *m=assemble_body(instruction,0,0);
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);
    CHECK(text && strstr(text,"CAST_U8"));
    AsmResult result; NvmModule *copy=asm_assemble(text,&result);
    CHECK(copy && result.error==ASM_OK);
    for(unsigned pass=0;pass<2;pass++) {
        VmState vm;vm_init(&vm,pass?copy:m);NanoValue out=val_void();
        CHECK(vm_invoke(&vm,1,NULL,0,&out)==VM_OK);
        CHECK(out.tag==TAG_U8 && out.as.u8==expected);
        CHECK(vm.stack_size==0 && vm.frame_count==0);
        vm_destroy(&vm);
    }
    free(text);nvm_module_free(copy);nvm_module_free(m);
}
static void reject_known(const char *instruction) {
    char source[2048];int n=snprintf(source,sizeof source,"%s.function convert 0 0 0 u8 1\n%s\nCAST_U8\nRET\n.end\n",prefix,instruction);
    CHECK(n>0 && (size_t)n<sizeof source);
    AsmResult result;NvmModule *m=asm_assemble(source,&result);
    CHECK(!m && result.error==ASM_ERR_VERIFY);
}
int main(void) {
    const InstructionInfo *info=isa_get_info(OP_CAST_U8);
    CHECK(OP_CAST_U8==0x8f && info && !strcmp(info->name,"CAST_U8"));
    CHECK(info->operand_count==0 && info->pop_count==1 && info->push_count==1);
    DecodedInstruction input={.opcode=OP_CAST_U8},output={0};uint8_t wire[2]={0,0xa5};
    CHECK(isa_encode(&input,wire,sizeof wire)==1 && wire[0]==0x8f && wire[1]==0xa5);
    CHECK(isa_decode(wire,1,&output)==1 && output.opcode==OP_CAST_U8 && output.operand_count==0);
    static const struct {int64_t input;unsigned expected;} cases[]={
        {0,0},{1,1},{127,127},{128,128},{200,200},{201,201},{255,255},
        {256,0},{257,1},{-1,255},{-256,0},{INT64_MIN,0},{INT64_MAX,255}};
    char instruction[96];
    for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++) {
        snprintf(instruction,sizeof instruction,"PUSH_I64 %" PRId64,cases[i].input);
        literal(instruction,cases[i].expected);
    }
    for(unsigned i=0;i<256;i++) {
        snprintf(instruction,sizeof instruction,"PUSH_U8 %u",i);literal(instruction,i);
    }
    reject_known("PUSH_F64 1.5");reject_known("PUSH_BOOL 1");
    /* Formal argument tags are unknown: I require the actual runtime guard. */
    NvmModule *m=assemble_body("LOAD_LOCAL 0",1,1);VmState vm;vm_init(&vm,m);
    size_t baseline=vm.heap.stats.num_objects;
    VmString *string=vm_string_new(&vm.heap,"a\0b",3);CHECK(string);
    NanoValue argument=val_string(string),out=val_void();
    for(unsigned i=0;i<8;i++) {
        CHECK(vm_invoke(&vm,1,&argument,1,&out)==VM_ERR_TYPE_ERROR);
        CHECK(out.tag==TAG_VOID && vm.stack_size==0 && vm.frame_count==0);
        CHECK(string->header.ref_count==1 && vm.heap.stats.num_objects==baseline+1);
        CHECK(string->length==3 && !memcmp(string->data,"a\0b",3));
        NanoValue valid=val_int(257);
        CHECK(vm_invoke(&vm,1,&valid,1,&out)==VM_OK && out.tag==TAG_U8 && out.as.u8==1);
    }
    vm_release(&vm.heap,argument);CHECK(vm.heap.stats.num_objects==baseline);
    vm_destroy(&vm);nvm_module_free(m);
    printf("I checked %u raw byte conversion assertions.\n",checks);return 0;
}
