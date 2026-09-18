#include "affine_bytecode.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "isa.h"
#include "nvm2c.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(c) do { checks++;assert(c); } while(0)
#ifdef OWN_TRANSFER_ALLOCATION_TEST
static unsigned allocation_attempts,fail_at;
void *affine_bytecode_test_malloc(size_t size) {
    if(++allocation_attempts==fail_at)return NULL;
    return malloc(size);
}
void *affine_bytecode_test_calloc(size_t count,size_t size) {
    if(++allocation_attempts==fail_at)return NULL;
    return calloc(count,size);
}
void *affine_bytecode_test_realloc(void *old,size_t size) {
    if(++allocation_attempts==fail_at)return NULL;
    return realloc(old,size);
}
#endif
#include "owned_fixture.h"
static void decision(const char *body,bool parameter,bool record_result,bool expected,const char *reason) {
    NvmModule *m=fixture(body,parameter,record_result);
    NvmVerifyResult result=nvm_verify_affine_function(m,0);
    if(result.ok!=expected)fprintf(stderr,"Unexpected: %s\n%s",result.error_msg,body);
    CHECK(result.ok==expected);
    if(reason)CHECK(strstr(result.error_msg,reason));
    NvmVerifyResult admission=nvm_verify(m);
    CHECK(admission.ok==(expected && !parameter && !record_result));
    if (!admission.ok) {
        /* Failed affine analysis precedes runtime signature admission. */
        const char *guard=expected
            ? "scalar entry and exact bounded value-result helper signatures"
            : "ownership instruction dataflow";
        if (!strstr(admission.error_msg,guard)) fprintf(stderr,"Admission reason: %s; expected: %s\n%s",admission.error_msg,guard,body);
        CHECK(strstr(admission.error_msg,guard));
    }
    nvm_module_free(m);
}
static void roundtrip(const char *body,const char *path) {
    NvmModule *m=fixture(body,false,false);CHECK(nvm_verify_affine_function(m,0).ok);
    uint32_t legacy_size;CHECK(nvm_serialize(m,&legacy_size)==NULL);
    NvmV2Module v2;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    size_t size;CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    if(path) {FILE *file=fopen(path,"wb");CHECK(file);CHECK(fwrite(bytes,1,size,file)==size);CHECK(fclose(file)==0);}
    nvm_v2_module_free(&v2);CHECK(nvm_v2_module_deserialize(bytes,size,&v2)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&v2,&copy)==NVM_V2_OK);nvm_v2_module_free(&v2);free(bytes);
    CHECK(copy->code_size==m->code_size && !memcmp(copy->code,m->code,m->code_size));
    CHECK(nvm_verify_affine_function(copy,0).ok);
    char *text=disasm_module_styled(copy,DISASM_STYLE_CANONICAL);CHECK(text);
    CHECK(strstr(text,"OWN_MOVE_LOCAL") && strstr(text,"OWN_UNPACK_LOCAL") && strstr(text,".ownership"));
    AsmResult error;NvmModule *rebuilt=asm_assemble_unverified(text,&error);CHECK(rebuilt);
    CHECK(rebuilt->code_size==m->code_size && !memcmp(rebuilt->code,m->code,m->code_size));
    CHECK(nvm_verify_affine_function(rebuilt,0).ok);
    NvmModule *verified=asm_assemble(text,&error);CHECK(verified);nvm_module_free(verified);
    nvm_module_free(rebuilt);free(text);nvm_module_free(copy);nvm_module_free(m);
}
int main(int argc,char **argv) {
    const uint8_t opcodes[4]={OP_OWN_MOVE_LOCAL,OP_OWN_STORE_LOCAL,OP_OWN_PACK,OP_OWN_UNPACK_LOCAL};
    const char *names[4]={"OWN_MOVE_LOCAL","OWN_STORE_LOCAL","OWN_PACK","OWN_UNPACK_LOCAL"};
    for(unsigned i=0;i<4;i++) {
        CHECK(opcodes[i]==0x0b+i);CHECK(isa_opcode_by_name(names[i])==opcodes[i]);
        const InstructionInfo *info=isa_get_info(opcodes[i]);CHECK(info && !strcmp(info->name,names[i]));
        DecodedInstruction in={0},out={0};in.opcode=opcodes[i];in.operands[0].u32=0x1234;
        uint8_t bytes[ISA_MAX_INSTRUCTION_SIZE];uint32_t size=isa_encode(&in,bytes,sizeof(bytes));
        CHECK(size==(i==2?5:3));CHECK(bytes[0]==opcodes[i] && bytes[1]==0x34 && bytes[2]==0x12);
        CHECK(isa_decode(bytes,size,&out)==size && out.opcode==opcodes[i]);
        CHECK((i==2?out.operands[0].u32:out.operands[0].u16)==0x1234);
    }
    const char *move="PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nRET\n";
    decision(move,false,false,true,NULL);
    decision("OWN_UNPACK_LOCAL 0\nRET\n",true,false,true,NULL);
    decision("OWN_MOVE_LOCAL 0\nRET\n",true,true,true,NULL);
    decision("PUSH_I64 42\nOWN_PACK 0\nRET\n",false,true,true,NULL);
    decision("PUSH_I64 10\nOWN_PACK 0\nPUSH_I64 32\nOWN_PACK 0\nOWN_PACK 2\nOWN_STORE_LOCAL 2\n"
        "OWN_UNPACK_LOCAL 2\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nOWN_UNPACK_LOCAL 3\nADD\nRET\n",false,false,true,NULL);
    decision("OWN_MOVE_LOCAL 0\nOWN_MOVE_LOCAL 0\nRET\n",true,true,false,"live unheld owner");
    decision("OWN_MOVE_LOCAL 0\nDUP\nRET\n",true,true,false,"duplicate");
    decision("OWN_MOVE_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",true,false,false,"discard");
    decision("OWN_MOVE_LOCAL 0\nAGG_GET 0\nRET\n",true,false,false,"field observation");
    decision("OWN_MOVE_LOCAL 0\nSTORE_LOCAL 3\nPUSH_I64 0\nRET\n",true,false,false,"escape");
    decision("LOAD_LOCAL 0\nOWN_STORE_LOCAL 3\nPUSH_I64 0\nRET\n",true,false,false,"owned token");
    decision("OWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 1\nPUSH_I64 0\nRET\n",true,false,false,"exact available owner");
    decision("PUSH_I64 8\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_I64 0\nRET\n",false,false,false,"owned obligations");
    decision("PUSH_I64 8\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_I64 9\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nRET\n",false,false,false,"available owner");
    decision("PUSH_BOOL 1\nOWN_PACK 0\nRET\n",false,true,false,"exact scalar or owned fields");
    decision("PUSH_I64 1\nOWN_PACK 1\nPUSH_I64 2\nOWN_PACK 0\nOWN_PACK 2\nRET\n",false,true,false,"declaration order");
    decision("OWN_UNPACK_LOCAL 0\nOWN_UNPACK_LOCAL 0\nRET\n",true,false,false,"intact owner");
    decision("PUSH_BOOL 1\nJMP_FALSE other\nOWN_MOVE_LOCAL 0\nJMP join\nother:\nLOAD_LOCAL 0\njoin:\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nRET\n",true,false,false,NULL);
    decision("PUSH_BOOL 1\nJMP_FALSE other\nOWN_MOVE_LOCAL 0\nJMP join\nother:\nOWN_MOVE_LOCAL 0\njoin:\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nRET\n",true,false,true,NULL);
    decision("loop:\nOWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 0\nPUSH_BOOL 0\nJMP_TRUE loop\nOWN_UNPACK_LOCAL 0\nRET\n",true,false,true,NULL);
    NvmModule *m=fixture(move,false,false);free(m->ownership_data);m->ownership_data=NULL;m->ownership_size=0;
    CHECK(!nvm_verify_affine_function(m,0).ok);CHECK(!nvm_verify(m).ok);
    char native_error[256];char *native=nvm2c_emit(m,native_error,sizeof(native_error));
    CHECK(native==NULL);CHECK(strstr(native_error,"ownership instruction"));nvm_module_free(m);
#ifdef OWN_TRANSFER_ALLOCATION_TEST
    m=fixture(move,false,false);
    for(unsigned failure=1;;failure++) {
        allocation_attempts=0;fail_at=failure;
        NvmVerifyResult got=nvm_verify_affine_function(m,0);
        fail_at=0;
        if(got.ok){CHECK(allocation_attempts<failure);break;}
        CHECK(failure<200);CHECK(got.error_msg[0]);
        CHECK(nvm_verify_affine_function(m,0).ok);
    }
    nvm_module_free(m);
#endif
    roundtrip(move,argc>1?argv[1]:NULL);
    printf("%u owned transfer checks passed\n",checks);return 0;
}
