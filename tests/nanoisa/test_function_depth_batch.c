/* I compare scoped hints to scalar checking and independently known depths. */
#include "verifier.h"
#include "nvm_format.h"
#include "nvm_v2_sections.h"
#include "isa.h"
#include "assembler.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "../../src/nanovm/vm.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) assert(x)
#include "owned_fixture.h"

static uint32_t encode(uint8_t *out, NanoOpcode op) {
    DecodedInstruction in = {0};
    in.opcode = op; in.operand_count = isa_get_info(op)->operand_count;
    if (op == OP_PUSH_I64) in.operands[0].i64 = 7;
    uint32_t size = isa_encode(&in, out, 128);
    assert(size); return size;
}
static NvmModule *ordinary(uint32_t count) {
    NvmModule *m = nvm_module_new(); assert(m);
    for (uint32_t i=0;i<count;++i) {
        char name[40]; snprintf(name,sizeof name,"function_%u",i);
        uint8_t code[128]; uint32_t size=0, depth=i%7+1;
        for(uint32_t j=0;j<depth;++j)size+=encode(code+size,OP_PUSH_I64);
        for(uint32_t j=1;j<depth;++j)size+=encode(code+size,OP_POP);
        size+=encode(code+size,OP_RET);
        NvmFunctionEntry f={0};f.name_idx=nvm_add_string(m,name,(uint32_t)strlen(name));
        f.code_offset=nvm_append_code(m,code,size);f.code_length=size;
        f.result_count=1;f.result_tag=TAG_INT;
        assert(nvm_add_function(m,&f)==i);
    }
    return m;
}
static unsigned cases;
static void paired(NvmModule *m, const char *label, bool publish, bool known) {
    const size_t stride=9, offset=3;
    size_t size=(size_t)m->function_count*stride;
    unsigned char *records=malloc(size?size:1); assert(records);
    memset(records,0xa5,size);
    uint16_t *expected=calloc(m->function_count?m->function_count:1,sizeof *expected);assert(expected);
    bool all=true;char first_error[NVM_VERIFY_ERROR_SIZE]="";
    for(uint32_t i=0;i<m->function_count;++i) {
        uint16_t depth=0;NvmVerifyResult r=nvm_verify_function_max_stack(m,i,&depth);
        if(r.ok)expected[i]=depth;else {if(all)snprintf(first_error,sizeof first_error,"%s",r.error_msg);all=false;}
        if(known)assert(r.ok && depth==i%7+1);
    }
    NvmVerifyResult result=nvm_infer_function_max_stacks(m,records,m->function_count,stride,offset);
    assert(result.ok==all);
    if(!all)assert(!strcmp(result.error_msg,first_error));
    if(!strcmp(label,"bad-body"))assert(expected[0]==1 && expected[1]==2 && expected[2]==0 && expected[3]==4);
    for(uint32_t i=0;i<m->function_count;++i) {
        uint16_t got;memcpy(&got,records+(size_t)i*stride+offset,sizeof got);assert(got==expected[i]);
        for(size_t j=0;j<stride;++j)if(j<offset || j>=offset+sizeof got)assert(records[(size_t)i*stride+j]==0xa5);
    }
    if(publish) {
        NvmV2Module wire={0};assert(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
        assert(wire.functions.count==m->function_count);
        for(uint32_t i=0;i<m->function_count;++i)assert(wire.functions.items[i].max_stack==expected[i]);
        size_t capacity=65536+(size_t)m->code_size*2+(size_t)m->function_count*256;
        uint8_t *first=malloc(capacity),*second=malloc(capacity);assert(first && second);
        size_t a=0,b=0;assert(nvm_v2_module_serialize(&wire,first,capacity,&a)==NVM_V2_OK);
        /* I restore the independent scalar oracle before a second encoding. */
        for(uint32_t i=0;i<m->function_count;++i)wire.functions.items[i].max_stack=expected[i];
        assert(nvm_v2_module_serialize(&wire,second,capacity,&b)==NVM_V2_OK);
        assert(a==b && !memcmp(first,second,a));
        free(first);free(second);nvm_v2_module_free(&wire);
    }
    printf("{\"case\":\"%s\",\"functions\":%u,\"all_inferred\":%s,\"wire_checked\":%s}\n",label,m->function_count,all?"true":"false",publish?"true":"false");
    ++cases;free(expected);free(records);
}
static void invalid_outputs(void) {
    NvmModule *m=ordinary(2);unsigned char bytes[32],copy[32];memset(bytes,0xa5,sizeof bytes);memcpy(copy,bytes,sizeof bytes);
    assert(!nvm_infer_function_max_stacks(NULL,bytes,2,8,1).ok);
    assert(!nvm_infer_function_max_stacks(m,NULL,2,8,1).ok);
    assert(!nvm_infer_function_max_stacks(m,bytes,1,8,1).ok);
    assert(!nvm_infer_function_max_stacks(m,bytes,2,0,0).ok);
    assert(!nvm_infer_function_max_stacks(m,bytes,2,8,7).ok);
    assert(!nvm_infer_function_max_stacks(m,bytes,2,SIZE_MAX,0).ok);
    assert(!memcmp(bytes,copy,sizeof bytes));
    nvm_module_free(m);m=ordinary(0);
    assert(nvm_infer_function_max_stacks(m,NULL,0,0,0).ok);nvm_module_free(m);
    ++cases;
}
static void mutations(void) {
    NvmModule *m=ordinary(8);paired(m,"before-mutation",true,true);
    uint32_t at=m->functions[2].code_offset;uint8_t saved=m->code[at];m->code[at]=OP_POP;
    paired(m,"bad-body",true,false);m->code[at]=saved;paired(m,"restored-body",true,true);
    uint32_t start=m->functions[7].code_offset;m->functions[7].code_offset=UINT32_MAX;
    paired(m,"bad-module-range",false,false);m->functions[7].code_offset=start;
    paired(m,"restored-module-range",true,true);
    uint32_t name=m->functions[1].name_idx;m->functions[1].name_idx=m->string_count;
    paired(m,"bad-module-name",false,false);m->functions[1].name_idx=name;
    paired(m,"restored-module-name",true,true);nvm_module_free(m);
}
static void ownership(void) {
    NvmModule *m=fixture("PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nRET\n",false,false);
    assert(nvm_verify_function_max_stack(m,0,NULL).ok);
    paired(m,"owned-record",true,false);
    uint8_t saved=m->ownership_data[8];m->ownership_data[8]=255;
    paired(m,"invalid-ownership",false,false);m->ownership_data[8]=saved;
    paired(m,"restored-ownership",true,false);nvm_module_free(m);
}
static void supplied(const char *path) {
    FILE *f=fopen(path,"rb");assert(f);assert(fseek(f,0,SEEK_END)==0);
    long length=ftell(f);assert(length>0 && (uint64_t)length<=UINT32_MAX);
    assert(fseek(f,0,SEEK_SET)==0);uint8_t *data=malloc((size_t)length);assert(data);
    assert(fread(data,1,(size_t)length,f)==(size_t)length);assert(fclose(f)==0);
    NvmModule *m=nvm_deserialize(data,(uint32_t)length);assert(m);free(data);
    paired(m,path,false,false);nvm_module_free(m);
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IOLBF,0);const uint32_t counts[]={0,1,8,64,256};
    for(size_t i=0;i<sizeof counts/sizeof counts[0];++i){NvmModule *m=ordinary(counts[i]);paired(m,"ordinary",true,true);nvm_module_free(m);}
    NvmModule *empty=ordinary(1);empty->functions[0].code_length=0;
    empty->functions[0].result_count=0;empty->functions[0].result_tag=TAG_VOID;
    paired(empty,"empty-body",true,false);nvm_module_free(empty);
    invalid_outputs();mutations();ownership();
    for(int i=1;i<argc;++i)supplied(argv[i]);
    printf("I passed %u paired depth cases and output boundaries.\n",cases);return 0;
}
