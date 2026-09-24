#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ownership_contracts.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "isa.h"
#include "nvm2c.h"

static unsigned checks;
#define CHECK(c) do { checks++; assert(c); } while (0)
static void word(uint8_t *data, unsigned offset, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) data[offset+i] = (uint8_t)(value >> (8*i));
}
static void half(uint8_t *data, unsigned offset, uint16_t value) {
    data[offset] = (uint8_t)value; data[offset+1] = (uint8_t)(value >> 8);
}
static void slot(uint8_t *data, unsigned offset, uint8_t tag, uint8_t mode, uint32_t layout) {
    data[offset] = tag; data[offset+1] = mode; word(data, offset+4, layout);
}
static uint8_t *wire(const NvmModule *module, size_t *size, const char *path) {
    NvmV2Module out;
    CHECK(nvm_v2_from_nvm_module(module, &out) == NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&out, NULL, 0, size) == NVM_V2_OK);
    uint8_t *bytes = malloc(*size); CHECK(bytes != NULL);
    CHECK(nvm_v2_module_serialize(&out, bytes, *size, NULL) == NVM_V2_OK);
    if (path) {
        FILE *file = fopen(path, "wb"); CHECK(file != NULL);
        CHECK(fwrite(bytes, 1, *size, file) == *size); CHECK(fclose(file) == 0);
    }
    nvm_v2_module_free(&out); return bytes;
}
static void check_status(NvmModule *module, bool valid, bool needs) {
    bool found = false;
    NvmV2Result result = nvm_ownership_contracts_validate(module, &found);
    CHECK((result == NVM_V2_OK) == valid);
    if (valid) CHECK(found == needs);
    else {
        CHECK(!nvm_verify(module).ok);
        char error[256];CHECK(nvm2c_emit(module,error,sizeof(error))==NULL);
    }
}

static void check_path_transport(NvmModule *module) {
    uint8_t *original=module->ownership_data;uint32_t old_size=module->ownership_size;
    uint16_t fields[32]={99},count=99;
    CHECK(nvm_ownership_path(module,0,fields,32,&count)==NVM_V2_ERR_FORMAT_VERSION);
    CHECK(fields[0]==99 && count==99);
    module->ownership_size=old_size+20;module->ownership_data=calloc(module->ownership_size,1);
    CHECK(module->ownership_data);memcpy(module->ownership_data,original,old_size);
    uint8_t *data=module->ownership_data;
    word(data,0,NVM_OWNERSHIP_PATH_VERSION);word(data,old_size,2);
    data[old_size+4]=1; /* path0: field0, followed by alignment padding */
    data[old_size+12]=2;data[old_size+18]=1; /* path1: fields0,1 */
    check_status(module,true,false);
    CHECK(nvm_ownership_path(module,0,fields,32,&count)==NVM_V2_OK && count==1 && fields[0]==0);
    CHECK(nvm_ownership_path(module,1,fields,32,&count)==NVM_V2_OK && count==2 && fields[0]==0 && fields[1]==1);
    fields[0]=99;count=99;
    CHECK(nvm_ownership_path(module,2,fields,32,&count)==NVM_V2_ERR_INDEX_RANGE);
    CHECK(fields[0]==99 && count==99);
    CHECK(nvm_ownership_path(module,1,fields,1,&count)==NVM_V2_ERR_INDEX_RANGE);
    CHECK(fields[0]==99 && count==99);
    size_t size;uint8_t *bytes=wire(module,&size,NULL);NvmV2Module decoded;
    CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
    CHECK(copy->ownership_size==module->ownership_size && !memcmp(copy->ownership_data,data,module->ownership_size));
    nvm_module_free(copy);nvm_v2_module_free(&decoded);free(bytes);
    char *text=disasm_module_styled(module,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult error;copy=asm_assemble(text,&error);CHECK(copy);
    CHECK(copy->ownership_size==module->ownership_size && !memcmp(copy->ownership_data,data,module->ownership_size));
    nvm_module_free(copy);free(text);
    data[old_size+6]=1;check_status(module,false,false);data[old_size+6]=0;
    data[old_size+4]=33;check_status(module,false,false);data[old_size+4]=1;
    data[old_size+10]=1;check_status(module,false,false);data[old_size+10]=0;
    word(data,old_size,257);check_status(module,false,false);word(data,old_size,2);
    word(data,0,3);check_status(module,false,false);word(data,0,2);
    module->ownership_size=old_size+4;word(data,old_size,0);check_status(module,true,false);
    size_t cap_size=old_size+4+256*8;
    data=realloc(data,cap_size);CHECK(data);module->ownership_data=data;module->ownership_size=cap_size;
    memset(data+old_size,0,cap_size-old_size);word(data,old_size,256);
    for(unsigned i=0;i<256;i++)data[old_size+4+i*8]=1;
    check_status(module,true,false);
    CHECK(nvm_ownership_path(module,255,fields,32,&count)==NVM_V2_OK && count==1 && fields[0]==0);
    module->ownership_size=old_size+72;memset(data+old_size,0,72);word(data,old_size,1);data[old_size+4]=32;
    check_status(module,true,false);
    CHECK(nvm_ownership_path(module,0,fields,32,&count)==NVM_V2_OK && count==32 && fields[31]==0);
    free(data);module->ownership_data=original;module->ownership_size=old_size;
    check_status(module,true,false);CHECK(nvm_verify(module).ok);
}

static void check_union_transport(void) {
    NvmModule *module=nvm_module_new();CHECK(module!=NULL);
    uint32_t identity=nvm_add_string(module,"Choice<int,string>",18);
    uint32_t int_variant=nvm_add_string(module,"IntValue",8);
    uint32_t pair_variant=nvm_add_string(module,"TextPair",8);
    uint32_t empty_variant=nvm_add_string(module,"Empty",5);
    uint32_t value_name=nvm_add_string(module,"value",5);
    uint32_t left_name=nvm_add_string(module,"left",4);
    uint32_t right_name=nvm_add_string(module,"right",5);
    CHECK(identity!=UINT32_MAX && int_variant!=UINT32_MAX &&
          pair_variant!=UINT32_MAX && empty_variant!=UINT32_MAX &&
          value_name!=UINT32_MAX && left_name!=UINT32_MAX && right_name!=UINT32_MAX);
    NvmV2LayoutField fields[]={{TAG_INT,NVM_V2_NO_INDEX,value_name},
        {TAG_STRING,NVM_V2_NO_INDEX,left_name},{TAG_BOOL,NVM_V2_NO_INDEX,right_name}};
    NvmV2Layout layout={NVM_V2_LAYOUT_UNION,3,identity,fields};
    NvmV2Layouts layouts={&layout,1};module->union_count=1;
    CHECK(nvm_retain_layouts(module,&layouts)==NVM_V2_OK);
    module->ownership_size=72;module->ownership_data=calloc(72,1);CHECK(module->ownership_data);
    uint8_t *data=module->ownership_data;
    word(data,0,NVM_OWNERSHIP_EXTENSION_VERSION);word(data,4,1);
    word(data,12,0); /* no functions */
    word(data,16,4);word(data,20,0); /* bounded empty v2 path suffix */
    word(data,24,1);half(data,28,NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS);
    half(data,30,NVM_OWNERSHIP_EXTENSION_REVISION_1);word(data,32,36);
    word(data,36,1);word(data,40,0);half(data,44,3);
    word(data,48,int_variant);half(data,52,0);half(data,54,1);
    word(data,56,pair_variant);half(data,60,1);half(data,62,2);
    word(data,64,empty_variant);half(data,68,3);half(data,70,0);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_OK && needs);
    NvmUnionVariantFact fact={99,99,99,99};
    CHECK(nvm_ownership_union_variant(module,0,0,&fact)==NVM_V2_OK &&
          fact.layout==0 && fact.name_idx==int_variant &&
          fact.field_offset==0 && fact.field_count==1);
    CHECK(nvm_ownership_union_variant(module,0,1,&fact)==NVM_V2_OK &&
          fact.name_idx==pair_variant && fact.field_offset==1 && fact.field_count==2);
    CHECK(nvm_ownership_union_variant(module,0,2,&fact)==NVM_V2_OK &&
          fact.name_idx==empty_variant && fact.field_offset==3 && fact.field_count==0);
    fact=(NvmUnionVariantFact){99,99,99,99};
    CHECK(nvm_ownership_union_variant(module,0,3,&fact)==NVM_V2_ERR_INDEX_RANGE &&
          fact.layout==99 && fact.name_idx==99 && fact.field_offset==99 && fact.field_count==99);
    data=realloc(data,80);CHECK(data);module->ownership_data=data;module->ownership_size=80;
    memmove(data+32,data+24,48);word(data,16,12);word(data,20,1);
    half(data,24,1);half(data,26,0);half(data,28,7);half(data,30,0);
    CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_OK);
    uint16_t path[2]={99,99},path_count=99;
    CHECK(nvm_ownership_path(module,0,path,2,&path_count)==NVM_V2_OK &&
          path_count==1 && path[0]==7);
    half(data,38,2);path[0]=99;path_count=99;
    CHECK(nvm_ownership_path(module,0,path,2,&path_count)==NVM_V2_ERR_FORMAT_VERSION &&
          path[0]==99 && path_count==99);
    half(data,38,NVM_OWNERSHIP_EXTENSION_REVISION_1);
    memmove(data+24,data+32,48);module->ownership_size=72;word(data,16,4);word(data,20,0);
    half(data,60,0);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_INDEX_RANGE);half(data,60,1);
    half(data,62,3);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_INDEX_RANGE);half(data,62,2);
    word(data,56,int_variant);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);word(data,56,pair_variant);
    half(data,46,1);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);half(data,46,0);
    word(data,40,1);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);word(data,40,0);
    word(data,36,2);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_INDEX_RANGE);word(data,36,1);
    word(data,16,3);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_RANGE);word(data,16,4);
    word(data,24,0);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_INDEX_RANGE);word(data,24,1);
    /* I distinguish an unknown extension from a known kind replacing required union facts. */
    half(data,28,UINT16_MAX);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_FORMAT_VERSION);
    half(data,28,NVM_OWNERSHIP_EXTENSION_SCALAR_GLOBALS);
    CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);
    half(data,28,NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS);
    half(data,30,2);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_FORMAT_VERSION);half(data,30,1);
    word(data,32,35);data[71]=1;CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_RANGE);
    data[71]=0;CHECK(nvm_ownership_contracts_validate(module,&needs)!=NVM_V2_OK);word(data,32,36);
    module->ownership_size=71;CHECK(nvm_ownership_contracts_validate(module,&needs)!=NVM_V2_OK);
    module->ownership_size=72;CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_OK);
    data=realloc(data,80);CHECK(data);module->ownership_data=data;module->ownership_size=80;
    word(data,24,2);half(data,72,NVM_OWNERSHIP_EXTENSION_ARRAY_FIELDS);
    half(data,74,NVM_OWNERSHIP_EXTENSION_REVISION_1);word(data,76,0);
    CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_FORMAT_VERSION);
    half(data,72,NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS);
    CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);
    half(data,72,NVM_OWNERSHIP_EXTENSION_SCALAR_GLOBALS);
    CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_FORMAT_VERSION);
    half(data,72,UINT16_MAX);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_FORMAT_VERSION);
    module->ownership_size=72;word(data,24,1);
    word(data,0,NVM_OWNERSHIP_PATH_VERSION);
    fact=(NvmUnionVariantFact){99,99,99,99};
    NvmV2Result query=nvm_ownership_union_variant(module,0,0,&fact);
    CHECK(query!=NVM_V2_OK);
    word(data,0,NVM_OWNERSHIP_EXTENSION_VERSION);
    nvm_module_free(module);
}

static void check_concrete_union_instances(void) {
    NvmModule *module=nvm_module_new();CHECK(module!=NULL);
    uint32_t first=nvm_add_string(module,"Choice<int,string>",18);
    uint32_t second=nvm_add_string(module,"Choice<float,bool>",18);
    uint32_t left=nvm_add_string(module,"IntValue",8);
    uint32_t right=nvm_add_string(module,"FloatValue",10);
    uint32_t value=nvm_add_string(module,"value",5);
    CHECK(first!=UINT32_MAX && second!=UINT32_MAX && left!=UINT32_MAX &&
          right!=UINT32_MAX && value!=UINT32_MAX);
    NvmV2LayoutField fields[]={{TAG_INT,NVM_V2_NO_INDEX,value},
                               {TAG_FLOAT,NVM_V2_NO_INDEX,value}};
    NvmV2Layout items[]={{NVM_V2_LAYOUT_UNION,1,first,&fields[0]},
                         {NVM_V2_LAYOUT_UNION,1,second,&fields[1]}};
    NvmV2Layouts layouts={items,2};module->union_count=2;
    CHECK(nvm_retain_layouts(module,&layouts)==NVM_V2_OK);
    module->ownership_size=72;module->ownership_data=calloc(72,1);
    CHECK(module->ownership_data!=NULL);uint8_t *data=module->ownership_data;
    word(data,0,NVM_OWNERSHIP_EXTENSION_VERSION);word(data,4,2);
    word(data,12,0);word(data,16,4);word(data,20,0);word(data,24,1);
    half(data,28,NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS);
    half(data,30,NVM_OWNERSHIP_EXTENSION_REVISION_1);word(data,32,36);
    word(data,36,2);word(data,40,0);half(data,44,1);word(data,48,left);half(data,54,1);
    word(data,56,1);half(data,60,1);word(data,64,right);half(data,70,1);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_OK && needs);
    NvmUnionVariantFact fact={0};
    CHECK(nvm_ownership_union_variant(module,0,0,&fact)==NVM_V2_OK &&
          fact.layout==0 && fact.name_idx==left && fact.field_count==1);
    CHECK(nvm_ownership_union_variant(module,1,0,&fact)==NVM_V2_OK &&
          fact.layout==1 && fact.name_idx==right && fact.field_count==1);
    word(data,56,0);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);
    word(data,56,1);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_OK);
    nvm_module_free(module);
}

static void check_owned_union_transport(const char *path) {
    AsmResult error;
    NvmModule *m=asm_assemble(".types 1 0 2\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n",&error);
    CHECK(m);
    uint32_t handle=nvm_add_string(m,"Handle",6), result=nvm_add_string(m,"Result<Handle,string>",21);
    uint32_t box=nvm_add_string(m,"Box<Result<Handle,string>>",26);
    uint32_t value=nvm_add_string(m,"value",5), ok=nvm_add_string(m,"Ok",2);
    uint32_t err=nvm_add_string(m,"Err",3), empty=nvm_add_string(m,"Empty",5);
    NvmV2LayoutField fd={TAG_INT,NVM_V2_NO_INDEX,value};
    NvmV2LayoutField payload[]={{TAG_STRUCT,0,value},{TAG_STRING,NVM_V2_NO_INDEX,value}};
    NvmV2LayoutField nested={TAG_UNION,1,value};
    NvmV2Layout items[]={{NVM_V2_LAYOUT_STRUCT,1,handle,&fd},
        {NVM_V2_LAYOUT_UNION,2,result,payload},{NVM_V2_LAYOUT_UNION,1,box,&nested}};
    NvmV2Layouts layouts={items,3};
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=108;m->ownership_data=calloc(108,1);CHECK(m->ownership_data);
    uint8_t *b=m->ownership_data;
    word(b,0,NVM_OWNERSHIP_UNION_GRAPH_VERSION);word(b,4,3);
    b[8]=b[9]=b[10]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;
    word(b,12,1);slot(b,20,TAG_INT,0,NVM_V2_NO_INDEX);
    word(b,28,4);word(b,32,0);word(b,36,1);
    half(b,40,NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS);
    half(b,42,NVM_OWNERSHIP_EXTENSION_REVISION_1);word(b,44,60);
    word(b,48,2);word(b,52,1);half(b,56,3);
    word(b,60,ok);half(b,64,0);half(b,66,1);
    word(b,68,err);half(b,72,1);half(b,74,1);
    word(b,76,empty);half(b,80,2);half(b,82,0);
    word(b,84,2);half(b,88,2);
    word(b,92,ok);half(b,96,0);half(b,98,1);
    word(b,100,empty);half(b,104,1);half(b,106,0);
    bool needs=false;
    CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    NvmUnionVariantFact fact={0};
    CHECK(nvm_ownership_union_variant(m,0,1,&fact)==NVM_V2_OK &&
          fact.layout==1 && fact.field_offset==1 && fact.field_count==1);
    CHECK(nvm_ownership_union_variant(m,1,1,&fact)==NVM_V2_OK &&
          fact.layout==2 && fact.field_offset==1 && fact.field_count==0);
    NvmLayoutAuthority authority[3];
    CHECK(nvm_ownership_layout_authorities(m,3,authority)==NVM_V2_OK &&
          authority[0]==NVM_LAYOUT_AUTHORITY_RESOURCE && authority[1]==authority[0] && authority[2]==authority[0]);
    CHECK(!nvm_verify(m).ok);
    char diagnostic[256];char *c=nvm2c_emit(m,diagnostic,sizeof diagnostic);CHECK(!c);free(c);
    /* Exact descriptors alone do not grant execution without a checked transfer. */
    m->functions[0].result_tag=TAG_UNION;slot(b,20,TAG_UNION,0,2);
    CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    word(b,24,NVM_V2_NO_INDEX);CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
    word(b,24,0);CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
    m->functions[0].result_tag=TAG_INT;slot(b,20,TAG_INT,0,NVM_V2_NO_INDEX);
    /* I retain wire transport; textual assembly still requires execution admission. */
    size_t size;uint8_t *bytes=wire(m,&size,path);NvmV2Module decoded;
    CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
    CHECK(copy->ownership_size==108 && !memcmp(copy->ownership_data,b,108));
    CHECK(nvm_ownership_union_variant(copy,1,0,&fact)==NVM_V2_OK && fact.layout==2);
    CHECK(!nvm_verify(copy).ok);
    nvm_module_free(copy);nvm_v2_module_free(&decoded);free(bytes);
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
    copy=asm_assemble(text,&error);CHECK(!copy);
    CHECK(strstr(error.message,"explicit owned entry execution")!=NULL);
    copy=asm_assemble_unverified(text,&error);CHECK(copy);
    CHECK(copy->ownership_size==108 && !memcmp(copy->ownership_data,b,108));
    CHECK(nvm_ownership_contracts_validate(copy,&needs)==NVM_V2_OK && needs);
    nvm_module_free(copy);free(text);
    /* I cannot erase a stored child's ownership or invent an empty owner. */
    for(unsigned i=8;i<11;i++) {
        uint8_t old=b[i];b[i]=0;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);b[i]=old;
        b[i]=NVM_LAYOUT_COMPLETE;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);b[i]=old;
    }
    b[8]=b[9]=b[10]=NVM_LAYOUT_COMPLETE;
    CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    CHECK(nvm_ownership_layout_authorities(m,3,authority)==NVM_V2_OK && authority[2]==NVM_LAYOUT_AUTHORITY_ORDINARY);
    b[9]|=NVM_LAYOUT_RESOURCE;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
    b[8]=b[9]=b[10]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;
    word(b,0,NVM_OWNERSHIP_EXTENSION_VERSION);CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
    word(b,0,5);CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_ERR_FORMAT_VERSION);
    word(b,0,NVM_OWNERSHIP_UNION_GRAPH_VERSION);
    /* I validate aggregate tag/child-kind agreement, not just numeric indices. */
    nested.type_tag=TAG_STRUCT;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
    nested.type_tag=TAG_UNION;nested.nested_idx=0;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
    nested.nested_idx=1;CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    /* I reject forward/cyclic and out-of-range children before any query writes. */
    for(uint32_t child=1;child<=3;child++) {
        word(m->layout_data,36,child);
        CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);
        authority[0]=authority[1]=authority[2]=NVM_LAYOUT_AUTHORITY_UNKNOWN;
        CHECK(nvm_ownership_layout_authorities(m,3,authority)!=NVM_V2_OK);
        CHECK(authority[0]==NVM_LAYOUT_AUTHORITY_UNKNOWN &&
              authority[1]==NVM_LAYOUT_AUTHORITY_UNKNOWN && authority[2]==NVM_LAYOUT_AUTHORITY_UNKNOWN);
    }
    word(m->layout_data,36,0);
    uint16_t path_fields[1]={99},path_count=99;
    CHECK(nvm_ownership_path(m,0,path_fields,1,&path_count)==NVM_V2_ERR_INDEX_RANGE &&
          path_fields[0]==99 && path_count==99);
    for(uint32_t n=0;n<108;n++) {
        m->ownership_size=n;fact=(NvmUnionVariantFact){99,99,99,99};
        CHECK(nvm_ownership_union_variant(m,1,0,&fact)!=NVM_V2_OK);
        CHECK(fact.layout==99 && fact.name_idx==99 && fact.field_offset==99 && fact.field_count==99);
    }
    m->ownership_size=108;
    half(b,98,2);CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);half(b,98,1);
    CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK);
    nvm_module_free(m);
}

int main(int argc, char **argv) {
    check_owned_union_transport(argc>3?argv[3]:NULL);
    check_union_transport();
    check_concrete_union_instances();
    AsmResult result;
    NvmModule *module = asm_assemble(
        ".types 1 0 0\n.entry 1\n.function read 1 1 0 int 1\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n.parameters 0 struct\n"
        ".function main 0 1 0 int 1\nPUSH_I64 42\nAGG_PACK 0 0 0 1\n"
        "STORE_LOCAL 0\nLOAD_LOCAL 0\nCALL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n", &result);
    if (!module) fprintf(stderr, "%s\n", result.message);
    CHECK(module != NULL);
    NvmV2LayoutField field = {TAG_INT, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX};
    NvmV2Layout item = {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &field};
    NvmV2Layouts layouts = {&item, 1};
    CHECK(nvm_retain_layouts(module, &layouts) == NVM_V2_OK);
    module->ownership_size = 56;
    module->ownership_data = calloc(module->ownership_size, 1);
    CHECK(module->ownership_data != NULL);
    uint8_t *data = module->ownership_data;
    word(data, 0, 1); word(data, 4, 1); data[8] = NVM_LAYOUT_COMPLETE;
    word(data, 12, 2);
    data[16] = 1; data[18] = 1; /* read: one local, one parameter */
    slot(data, 20, TAG_INT, 0, NVM_V2_NO_INDEX);
    slot(data, 28, TAG_STRUCT, 0, 0);
    data[36] = 1; /* main: one local, no parameters */
    slot(data, 40, TAG_INT, 0, NVM_V2_NO_INDEX);
    slot(data, 48, TAG_STRUCT, 0, 0);
    check_status(module, true, false);
    CHECK(nvm_verify(module).ok);
    size_t size;
    uint8_t *bytes = wire(module, &size, argc > 1 ? argv[1] : NULL);
    NvmV2Header header;
    CHECK(nvm_v2_read_header(bytes, size, &header) == NVM_V2_OK);
    CHECK(header.feature_bits & NVM_V2_FEATURE_OWNERSHIP);
    free(bytes);
    char *text = disasm_module_styled(module, DISASM_STYLE_CANONICAL);
    CHECK(text && strstr(text, ".ownership \""));
    NvmModule *copy = asm_assemble(text, &result);
    CHECK(copy != NULL);
    CHECK(copy->ownership_size == module->ownership_size);
    CHECK(!memcmp(copy->ownership_data, data, module->ownership_size));
    nvm_module_free(copy); free(text);

    check_path_transport(module);

    /* Shared/exclusive declarations remain transportable but cannot execute
     * before genuine references and instruction lifetimes are implemented. */
    data[8] |= NVM_LAYOUT_RESOURCE;
    data[29] = 1;
    check_status(module, true, true);
    CHECK(!nvm_verify(module).ok);
    bytes = wire(module, &size, argc > 2 ? argv[2] : NULL);
    NvmV2Module decoded;
    CHECK(nvm_v2_module_deserialize(bytes, size, &decoded) == NVM_V2_OK);
    CHECK(nvm_v2_to_nvm_module(&decoded, &copy) == NVM_V2_OK);
    nvm_v2_module_free(&decoded); free(bytes);
    check_status(copy, true, true);
    CHECK(copy->ownership_data[29] == 1);
    CHECK(copy->ownership_data != module->ownership_data);
    text = disasm_module_styled(copy, DISASM_STYLE_CANONICAL);
    CHECK(text != NULL);
    CHECK(asm_assemble(text, &result) == NULL);
    CHECK(result.error == ASM_ERR_VERIFY);
    NvmModule *retained = asm_assemble_unverified(text, &result);
    CHECK(retained != NULL);
    check_status(retained, true, true);
    CHECK(!memcmp(retained->ownership_data, data, module->ownership_size));
    nvm_module_free(retained); nvm_module_free(copy); free(text);
    data[29] = 2; check_status(module, true, true);

    /* Normal declaration mistakes do not become unknown reference facts. */
    data[29] = 3; check_status(module, false, false); data[29] = 1;
    data[8] = NVM_LAYOUT_COMPLETE; check_status(module, false, false);
    data[8] = NVM_LAYOUT_RESOURCE; check_status(module, false, false);
    data[8] = NVM_LAYOUT_COMPLETE | NVM_LAYOUT_RESOURCE;
    data[49] = 1; check_status(module, false, false); data[49] = 0;
    data[21] = 1; check_status(module, false, false); data[21] = 0;
    data[28] = TAG_INT; check_status(module, false, false); data[28] = TAG_STRUCT;
    word(data, 32, NVM_V2_NO_INDEX); check_status(module, false, false); word(data, 32, 0);
    word(data, 52, 1); check_status(module, false, false); word(data, 52, 0);
    data[18] = 0; check_status(module, false, false); data[18] = 1;
    data[36] = 2; check_status(module, false, false); data[36] = 1;
    data[20] = TAG_BOOL; check_status(module, false, false); data[20] = TAG_INT;
    data[30] = 1; check_status(module, false, false); data[30] = 0;
    data[9] = 1; check_status(module, false, false); data[9] = 0;
    word(data, 0, 2); check_status(module, false, false); word(data, 0, 1);
    word(data, 12, 1); check_status(module, false, false); word(data, 12, 2);
    check_status(module, true, true);
    /* A complete outer record must propagate a nested resource obligation. */
    NvmV2LayoutField nested = {TAG_STRUCT, 0, NVM_V2_NO_INDEX};
    NvmV2Layout pair[] = {item, {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &nested}};
    NvmV2Layouts tree = {pair, 2};
    module->struct_count = 2;
    CHECK(nvm_retain_layouts(module, &tree) == NVM_V2_OK);
    word(data, 4, 2); data[9] = NVM_LAYOUT_COMPLETE;
    check_status(module, false, false);
    data[9] |= NVM_LAYOUT_RESOURCE;
    check_status(module, true, true);
    /* A borrowed leaf cannot silently become an aggregate referent. */
    word(data, 32, 1);
    check_status(module, false, false);
    word(data, 32, 0);
    check_status(module, true, true);
    nvm_module_free(module);
    printf("I passed %u ownership-contract checks.\n", checks);
    return 0;
}
