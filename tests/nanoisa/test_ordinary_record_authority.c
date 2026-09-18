#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static long budget=-1;
static void *authority_calloc(size_t count,size_t size) {
    if(!budget)return NULL;
    if(budget>0)budget--;
    return calloc(count,size);
}
#define calloc authority_calloc
#include "../../src/nanoisa/ownership_contracts.c"
#undef calloc
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
static unsigned checks;
#define CHECK(x) do { checks++; assert(x); } while(0)
static void word(uint8_t *p,uint32_t n) {for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(n>>(8*i));}
static void check_authority(NvmModule *m,uint32_t index,NvmLayoutAuthority expected) {
    NvmLayoutAuthority found=NVM_LAYOUT_AUTHORITY_RESOURCE;
    CHECK(nvm_ownership_layout_authority(m,index,&found)==NVM_V2_OK && found==expected);
}
static void invalid(NvmModule *m) {
    bool needs=true;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK && !needs);
    NvmLayoutAuthority found=NVM_LAYOUT_AUTHORITY_RESOURCE;
    CHECK(nvm_ownership_layout_authority(m,0,&found)!=NVM_V2_OK && found==NVM_LAYOUT_AUTHORITY_RESOURCE);
}
int main(int argc,char **argv) {
    AsmResult error;
    NvmModule *m=asm_assemble(".types 4 0 0\n.string text \"abc\"\n.entry main\n.function main 0 0 0 int 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\nAGG_GET 0\nAGG_GET 0\nSTR_LEN\nPUSH_I64 3\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",&error);
    CHECK(m);
    NvmV2LayoutField leaf={TAG_STRING,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField nested={TAG_STRUCT,0,NVM_V2_NO_INDEX};
    NvmV2Layout entries[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&leaf},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&nested},
        {NVM_V2_LAYOUT_STRUCT,0,NVM_V2_NO_INDEX,NULL},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&leaf}};
    NvmV2Layouts layouts={entries,4};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    check_authority(m,0,NVM_LAYOUT_AUTHORITY_UNKNOWN);
    m->ownership_size=28;m->ownership_data=calloc(28,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,1);word(data+4,4);
    for(unsigned i=0;i<4;i++)data[8+i]=NVM_LAYOUT_COMPLETE;
    word(data+12,1);data[20]=TAG_INT;word(data+24,NVM_V2_NO_INDEX);
    bool needs=true;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && !needs);
    CHECK(nvm_verify(m).ok);
    for(unsigned i=0;i<4;i++)check_authority(m,i,NVM_LAYOUT_AUTHORITY_ORDINARY);
    char *before=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(before);
    NvmModule *copy=asm_assemble(before,&error);CHECK(copy);
    CHECK(copy->layout_size==m->layout_size && !memcmp(copy->layout_data,m->layout_data,m->layout_size));
    CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,data,m->ownership_size));
    check_authority(copy,1,NVM_LAYOUT_AUTHORITY_ORDINARY);nvm_module_free(copy);
    NvmLayoutAuthority found=NVM_LAYOUT_AUTHORITY_RESOURCE;
    budget=0;CHECK(nvm_ownership_layout_authority(m,1,&found)==NVM_V2_ERR_TRUNCATED);budget=-1;
    CHECK(found==NVM_LAYOUT_AUTHORITY_RESOURCE);
    char *after=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(after && !strcmp(before,after));free(after);
    budget=1;CHECK(nvm_ownership_layout_authority(m,1,&found)==NVM_V2_OK);budget=-1;
    CHECK(found==NVM_LAYOUT_AUTHORITY_ORDINARY);
    found=NVM_LAYOUT_AUTHORITY_RESOURCE;
    CHECK(nvm_ownership_layout_authority(m,4,&found)==NVM_V2_ERR_INDEX_RANGE && found==NVM_LAYOUT_AUTHORITY_RESOURCE);
    CHECK(nvm_ownership_layout_authority(m,0,NULL)==NVM_V2_ERR_INDEX_RANGE);
    data[8]=3;invalid(m);data[8]=1; /* Direct resource string stays unsupported. */
    data[9]=3;invalid(m);data[9]=1; /* So does a transitive string child. */
    data[8]=0;invalid(m);data[8]=1; /* Unknown child cannot certify its parent. */
    data[11]=0;check_authority(m,3,NVM_LAYOUT_AUTHORITY_UNKNOWN);data[11]=1;
    leaf.type_tag=TAG_INT;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    data[9]=3;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    check_authority(m,0,NVM_LAYOUT_AUTHORITY_ORDINARY);check_authority(m,1,NVM_LAYOUT_AUTHORITY_RESOURCE);
    data[8]=3;data[9]=1;invalid(m);data[9]=3;
    CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    data[8]=data[9]=1;
    leaf.type_tag=TAG_ARRAY;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);invalid(m);
    leaf.type_tag=TAG_STRING;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    /* Existing v2 empty-path transport has the same ordinary declarations. */
    uint8_t *paths=calloc(32,1);CHECK(paths);memcpy(paths,data,28);word(paths,2);
    m->ownership_data=paths;m->ownership_size=32;check_authority(m,1,NVM_LAYOUT_AUTHORITY_ORDINARY);
    free(paths);m->ownership_data=data;m->ownership_size=28;
    CHECK(nvm_verify(m).ok);
    if(argc==2) {
        NvmV2Module v2;CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);size_t length;
        CHECK(nvm_v2_module_serialize(&v2,NULL,0,&length)==NVM_V2_OK);uint8_t *bytes=malloc(length);CHECK(bytes);
        CHECK(nvm_v2_module_serialize(&v2,bytes,length,NULL)==NVM_V2_OK);
        FILE *f=fopen(argv[1],"wb");CHECK(f && fwrite(bytes,1,length,f)==length && fclose(f)==0);
        free(bytes);nvm_v2_module_free(&v2);
    }
    free(before);nvm_module_free(m);
    printf("I passed %u ordinary authority checks.\n",checks);return 0;
}
