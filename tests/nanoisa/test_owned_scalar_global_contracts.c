/* I transport exact scalar globals without granting instruction execution. */
#include "ownership_contracts.h"
#include "affine_bytecode.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "verifier.h"
#include "nvm2c.h"
#include "isa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(c) do {checks++;assert(c);} while(0)
#include "union_fixture.h"
static const char *body="PUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\n"
    "OWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nMATCH_TAG 0 selected\nPOP\nPUSH_BOOL 0\nASSERT\nHALT\n"
    "selected:\nPOP\nOWN_UNPACK_VARIANT 0 0 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n";
static size_t attach(NvmModule *m,const uint8_t *tags,uint32_t count) {
    size_t at=m->ownership_size;uint32_t bytes=4+4*count;
    uint8_t *data=realloc(m->ownership_data,at+8+bytes);CHECK(data);
    m->ownership_data=data;m->ownership_size=(uint32_t)(at+8+bytes);
    memset(data+at,0,8+bytes);
    /* My entry has three locals; each optional zero-local helper adds 12 bytes. */
    word(data+60+(m->function_count-1)*12,2);data[at]=NVM_OWNERSHIP_EXTENSION_SCALAR_GLOBALS;
    data[at+2]=NVM_OWNERSHIP_EXTENSION_REVISION_1;word(data+at+4,bytes);word(data+at+8,count);
    for(uint32_t i=0;i<count;i++)data[at+12+4*i]=tags[i];
    return at;
}
static void query(NvmModule *m,const uint8_t *tags,uint32_t count) {
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    uint8_t out[256];memset(out,0xa5,sizeof(out));uint32_t n=999;
    CHECK(nvm_ownership_scalar_globals(m,out,sizeof(out),&n)==NVM_V2_OK);
    CHECK(n==count && !memcmp(out,tags,count));
    for(uint32_t i=count;i<sizeof(out);i++)CHECK(out[i]==0xa5);
}
static void refused_query(NvmModule *m) {
    uint8_t out[256];memset(out,0xa5,sizeof(out));uint32_t n=999;
    CHECK(nvm_ownership_scalar_globals(m,out,sizeof(out),&n)!=NVM_V2_OK);
    CHECK(n==999);
    for(unsigned i=0;i<sizeof(out);i++)CHECK(out[i]==0xa5);
}
static void roundtrip(NvmModule *m,const uint8_t *tags,uint32_t count) {
    NvmV2Module v2,decoded;size_t size=0;
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
    CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
    query(copy,tags,count);nvm_module_free(copy);nvm_v2_module_free(&decoded);free(bytes);nvm_v2_module_free(&v2);
}
static void helper(NvmModule *m,const char *code) {
    char source[1024];snprintf(source,sizeof(source),".function change 0 0 0 int 1\n%s.end\n",code);
    AsmResult assembled;NvmModule *h=asm_assemble_unverified(source,&assembled);CHECK(h);
    NvmFunctionEntry entry=h->functions[0];entry.name_idx=nvm_add_string(m,"change",6);
    entry.code_offset=nvm_append_code(m,h->code,h->code_size);CHECK(entry.code_offset!=UINT32_MAX);
    CHECK(nvm_add_function(m,&entry)==1);nvm_module_free(h);
    uint32_t size=m->ownership_size;uint8_t *data=realloc(m->ownership_data,size+12);CHECK(data);
    m->ownership_data=data;m->ownership_size=size+12;
    memmove(data+64,data+52,size-52);memset(data+52,0,12);slot(data+56,TAG_INT,0);word(data+12,2);
}
static void analysis_many(const char *prefix,const uint8_t *tags,uint32_t count,const char *callee,bool accepted) {
    char code[4096];snprintf(code,sizeof(code),"%s%s",prefix,body);
    NvmModule *m=owned_union_fixture(code,0,TAG_INT);
    if(callee)helper(m,callee);
    attach(m,tags,count);
    NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);
    if(result.ok!=accepted)fprintf(stderr,"global flow: %s\n%s\n",result.message,prefix);
    CHECK(result.ok==accepted);
    CHECK(!nvm_verify(m).ok); /* My public runtime remains gated at this checkpoint. */
    nvm_module_free(m);
}
static void analysis_case(const char *prefix,uint8_t tag,const char *callee,bool accepted) {
    analysis_many(prefix,&tag,1,callee,accepted);
}
static void analysis_cases(void) {
    analysis_case("PUSH_I64 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\n",TAG_INT,NULL,true);
    analysis_case("PUSH_U8 7\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPOP\n",TAG_U8,NULL,true);
    analysis_case("PUSH_BOOL 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPOP\n",TAG_BOOL,NULL,true);
    analysis_case("PUSH_F64 2.5\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPOP\n",TAG_FLOAT,NULL,true);
    analysis_case("PUSH_STR 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPOP\n",TAG_STRING,NULL,true);
    analysis_case("PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 1\nSTORE_GLOBAL 0\nJMP join\nother:\nPUSH_I64 2\nSTORE_GLOBAL 0\njoin:\nLOAD_GLOBAL 0\nPOP\n",TAG_INT,NULL,true);
    analysis_case("PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 1\nSTORE_GLOBAL 0\nJMP join\nother:\nNOP\njoin:\nLOAD_GLOBAL 0\nPOP\n",TAG_INT,NULL,false);
    analysis_case("PUSH_BOOL 0\nJMP_FALSE after\nloop:\nPUSH_I64 1\nSTORE_GLOBAL 0\nPUSH_BOOL 0\nJMP_TRUE loop\nafter:\nLOAD_GLOBAL 0\nPOP\n",TAG_INT,NULL,false);
    analysis_case("LOAD_GLOBAL 0\nPOP\n",TAG_INT,NULL,false);
    analysis_case("PUSH_I64 0\nSTORE_GLOBAL 1\n",TAG_INT,NULL,false);
    analysis_case("PUSH_I64 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 1\nPOP\n",TAG_INT,NULL,false);
    analysis_case("PUSH_BOOL 1\nSTORE_GLOBAL 0\n",TAG_INT,NULL,false);
    analysis_case("PUSH_I64 7\nOWN_PACK 0\nSTORE_GLOBAL 0\n",TAG_INT,NULL,false);
    analysis_case("PUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_GLOBAL 0\n",TAG_INT,NULL,false);
    analysis_case("",TAG_INT,NULL,false);
    const char *increment="LOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n";
    const uint8_t pair[]={TAG_INT,TAG_BOOL};
    analysis_many("PUSH_I64 0\nSTORE_GLOBAL 0\nPUSH_BOOL 1\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\nPOP\n",pair,2,NULL,true);
    analysis_many("PUSH_I64 0\nSTORE_GLOBAL 0\n",pair,2,NULL,false);
    analysis_many("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nPUSH_BOOL 1\nSTORE_GLOBAL 1\n",pair,2,increment,false);
    analysis_case("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nCALL 1\nPOP\n",TAG_INT,increment,true);
    analysis_case("CALL 1\nPOP\nPUSH_I64 0\nSTORE_GLOBAL 0\n",TAG_INT,increment,false);
    analysis_case("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\n",TAG_INT,"PUSH_BOOL 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",false);
}
#ifdef GLOBAL_FLOW_ALLOCATION_TEST
static size_t allocation_calls,allocation_failure;
void *global_flow_test_malloc(size_t size) {
    if(++allocation_calls==allocation_failure)return NULL;
    return malloc(size);
}
void *global_flow_test_calloc(size_t count,size_t size) {
    if(++allocation_calls==allocation_failure)return NULL;
    return calloc(count,size);
}
void *global_flow_test_realloc(void *old,size_t size) {
    if(++allocation_calls==allocation_failure)return NULL;
    return realloc(old,size);
}
static void allocation_cases(void) {
    char code[4096];snprintf(code,sizeof(code),"PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\n%s",body);
    NvmModule *m=owned_union_fixture(code,0,TAG_INT);
    helper(m,"LOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n");
    uint8_t tag=TAG_INT;attach(m,&tag,1);
    allocation_calls=0;CHECK(nvm_affine_analyze_function(m,0).ok);
    size_t total=allocation_calls;CHECK(total>0);
    for(size_t failure=1;failure<=total;failure++) {
        allocation_calls=0;allocation_failure=failure;
        CHECK(!nvm_affine_analyze_function(m,0).ok);
        allocation_failure=0;query(m,&tag,1);
    }
    CHECK(nvm_affine_analyze_function(m,0).ok);nvm_module_free(m);
}
#endif
int main(void) {
    uint8_t tags[]={TAG_INT,TAG_U8,TAG_BOOL,TAG_FLOAT,TAG_STRING};
    NvmModule *m=owned_union_fixture(body,0,TAG_INT);
    CHECK(nvm_verify(m).ok);query(m,tags,0);
    size_t at=attach(m,tags,sizeof(tags));query(m,tags,sizeof(tags));roundtrip(m,tags,sizeof(tags));
    /* The transport checkpoint retains a closed execution gate. */
    CHECK(!nvm_verify(m).ok);char error[256];CHECK(!nvm2c_emit(m,error,sizeof(error)));
    uint8_t output[5];memset(output,0xa5,sizeof(output));uint32_t count=999;
    CHECK(nvm_ownership_scalar_globals(m,output,4,&count)!=NVM_V2_OK && count==999);
    for(unsigned i=0;i<sizeof(output);i++)CHECK(output[i]==0xa5);
    CHECK(nvm_ownership_scalar_globals(m,NULL,256,&count)!=NVM_V2_OK && count==999);
    CHECK(nvm_ownership_scalar_globals(m,output,5,NULL)!=NVM_V2_OK);
    for(unsigned tag=0;tag<256;tag++) {
        bool scalar=false;for(unsigned i=0;i<sizeof(tags);i++)scalar|=tag==tags[i];
        if(scalar)continue;
        m->ownership_data[at+12]=(uint8_t)tag;refused_query(m);
    }
    m->ownership_data[at+12]=TAG_INT;
    for(unsigned reserved=1;reserved<4;reserved++) {
        m->ownership_data[at+12+reserved]=1;refused_query(m);m->ownership_data[at+12+reserved]=0;
    }
    m->ownership_data[at+2]=2;refused_query(m);m->ownership_data[at+2]=1;
    m->ownership_data[at]=4;refused_query(m);m->ownership_data[at]=1;refused_query(m);m->ownership_data[at]=3;
    word(m->ownership_data+at+8,0);refused_query(m);word(m->ownership_data+at+8,257);refused_query(m);
    word(m->ownership_data+at+8,5);
    uint32_t full=m->ownership_size;
    for(uint32_t size=0;size<full;size++){m->ownership_size=size;refused_query(m);}
    m->ownership_size=full;query(m,tags,5);
    uint8_t *extended=realloc(m->ownership_data,full+4);CHECK(extended);m->ownership_data=extended;
    memset(extended+full,0,4);m->ownership_size=full+4;refused_query(m);
    word(extended+at+4,28);refused_query(m);nvm_module_free(m);
    m=union_fixture("PUSH_I64 0\nRET\n",0,3,TAG_INT);attach(m,tags,5);refused_query(m);nvm_module_free(m);
    uint8_t many[256];memset(many,TAG_INT,sizeof(many));
    m=owned_union_fixture(body,0,TAG_INT);attach(m,many,256);query(m,many,256);roundtrip(m,many,256);nvm_module_free(m);
    NvmModule absent={0};count=999;CHECK(nvm_ownership_scalar_globals(&absent,NULL,0,&count)==NVM_V2_OK && !count);
    count=999;CHECK(nvm_ownership_scalar_globals(NULL,NULL,0,&count)!=NVM_V2_OK && count==999);
    analysis_cases();
#ifdef GLOBAL_FLOW_ALLOCATION_TEST
    allocation_cases();
#endif
    printf("I passed %u scalar-global declaration and flow checks.\n",checks);return 0;
}
