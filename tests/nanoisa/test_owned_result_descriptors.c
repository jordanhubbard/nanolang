/* I qualify declarations/transport separately from executable admission. */
#include "affine_state.h"
#include "assembler.h"
#include "disassembler.h"
#include "retained_layouts.h"
#include "verifier.h"
#include "isa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(c) do {checks++;assert(c);} while(0)
static void word(uint8_t *p,uint32_t n){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(n>>(i*8));}
static NvmModule *fixture(uint8_t tag,uint32_t layout,unsigned version) {
    const char *name=tag==TAG_VOID?"void":tag==TAG_INT?"int":tag==TAG_BOOL?"bool":tag==TAG_U8?"u8":tag==TAG_FLOAT?"float":"struct";
    char body[128];
    if(tag==TAG_STRUCT)snprintf(body,sizeof(body),"PUSH_I64 42\n%sOWN_PACK %u\nRET\n",layout==2?"OWN_PACK 0\n":"",layout);
    else snprintf(body,sizeof(body),"%sRET\n",tag==TAG_VOID?"":tag==TAG_INT?"PUSH_I64 42\n":tag==TAG_BOOL?"PUSH_BOOL 1\n":tag==TAG_U8?"PUSH_U8 42\n":"PUSH_F64 42.0\n");
    char source[512];snprintf(source,sizeof(source),".types 3 0 0\n.entry 0\n.function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.function helper 0 0 0 %s %u\n%s.end\n",name,tag!=TAG_VOID,body);
    AsmResult result;NvmModule *m=asm_assemble_unverified(source,&result);CHECK(m);
    NvmV2LayoutField scalar={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField nested={TAG_STRUCT,0,NVM_V2_NO_INDEX};
    NvmV2Layout rows[3]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&nested}};
    NvmV2Layouts all={rows,3};CHECK(nvm_retain_layouts(m,&all)==NVM_V2_OK);
    m->ownership_size=version==1?40:44;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,version);word(p+4,3);p[8]=p[9]=p[10]=3;word(p+12,2);
    p[20]=TAG_INT;word(p+24,NVM_V2_NO_INDEX);p[32]=tag;word(p+36,layout);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
    return m;
}
static void query(NvmModule *m,uint8_t tag,uint32_t layout,bool accepted) {
    NvmAffineState *s=nvm_affine_state_create(m,1,0);CHECK(s);
    NvmAffineType type={TAG_FLOAT,123};uint16_t fields=456;
    CHECK(nvm_affine_value_result(s,&type,&fields)==accepted);
    if(accepted){CHECK(type.tag==tag&&type.layout==layout);CHECK(fields==(tag==TAG_STRUCT?1:0));}
    else {CHECK(type.tag==TAG_FLOAT&&type.layout==123);CHECK(fields==456);}
    nvm_affine_state_free(s);
}
static void same_transport(const NvmModule *a,const NvmModule *b) {
    CHECK(a->ownership_size==b->ownership_size&&!memcmp(a->ownership_data,b->ownership_data,a->ownership_size));
    CHECK(a->layout_size==b->layout_size&&!memcmp(a->layout_data,b->layout_data,a->layout_size));
    CHECK(a->function_count==b->function_count);
    for(uint32_t f=0;f<a->function_count;f++) {
        CHECK(a->functions[f].result_count==b->functions[f].result_count);
        CHECK(a->functions[f].result_tag==b->functions[f].result_tag);
    }
}
static void roundtrip(NvmModule *m,uint8_t tag,uint32_t layout) {
    NvmV2Module encoded={0},decoded={0};size_t size=0;
    CHECK(nvm_v2_from_nvm_module(m,&encoded)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&encoded,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&encoded,bytes,size,NULL)==NVM_V2_OK);
    CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
    same_transport(m,copy);query(copy,tag,layout,true);
    char *text=disasm_module_styled(copy,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult result;NvmModule *text_copy=asm_assemble_unverified(text,&result);CHECK(text_copy);
    same_transport(m,text_copy);query(text_copy,tag,layout,true);
    char *again=disasm_module_styled(text_copy,DISASM_STYLE_CANONICAL);CHECK(again&&!strcmp(text,again));
    /* These transport-only fixtures are never executed. */
    free(again);free(text);nvm_module_free(text_copy);nvm_module_free(copy);
    nvm_v2_module_free(&decoded);free(bytes);nvm_v2_module_free(&encoded);
}
int main(void) {
    const uint8_t tags[]={TAG_VOID,TAG_INT,TAG_BOOL,TAG_U8,TAG_STRUCT,TAG_STRUCT};
    for(unsigned version=1;version<=2;version++)for(unsigned i=0;i<sizeof(tags);i++) {
        uint32_t layout=i<4?NVM_V2_NO_INDEX:i-4;NvmModule *m=fixture(tags[i],layout,version);
        query(m,tags[i],layout,true);roundtrip(m,tags[i],layout);nvm_module_free(m);
    }
    NvmModule *m=fixture(TAG_STRUCT,2,2);query(m,TAG_STRUCT,2,true);roundtrip(m,TAG_STRUCT,2);nvm_module_free(m);
    m=fixture(TAG_FLOAT,NVM_V2_NO_INDEX,2);query(m,TAG_FLOAT,NVM_V2_NO_INDEX,false);nvm_module_free(m);
    m=fixture(TAG_STRUCT,0,2);m->ownership_data[8]=NVM_LAYOUT_COMPLETE;
    query(m,TAG_STRUCT,0,false);nvm_module_free(m);
    m=fixture(TAG_VOID,NVM_V2_NO_INDEX,2);m->functions[1].result_tag=TAG_INT;
    query(m,TAG_VOID,NVM_V2_NO_INDEX,false);nvm_module_free(m);
    NvmAffineType type={TAG_BOOL,17};uint16_t fields=19;
    CHECK(!nvm_affine_value_result(NULL,&type,&fields));CHECK(type.tag==TAG_BOOL&&type.layout==17&&fields==19);
    printf("%u owned result descriptor/roundtrip checks passed\n",checks);return 0;
}
