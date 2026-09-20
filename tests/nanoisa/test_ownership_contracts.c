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
    module->ownership_size=56;module->ownership_data=calloc(56,1);CHECK(module->ownership_data);
    uint8_t *data=module->ownership_data;
    word(data,0,NVM_OWNERSHIP_UNION_VERSION);word(data,4,1);
    word(data,12,0); /* no functions */
    word(data,16,0); /* no reference paths */
    word(data,20,1);word(data,24,0);half(data,28,3);
    word(data,32,int_variant);half(data,36,0);half(data,38,1);
    word(data,40,pair_variant);half(data,44,1);half(data,46,2);
    word(data,48,empty_variant);half(data,52,3);half(data,54,0);
    bool needs=true;CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_OK && !needs);
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
    half(data,44,0);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_INDEX_RANGE);half(data,44,1);
    half(data,46,3);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_INDEX_RANGE);half(data,46,2);
    word(data,40,int_variant);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);word(data,40,pair_variant);
    half(data,30,1);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);half(data,30,0);
    word(data,24,1);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_SECTION_TYPE);word(data,24,0);
    word(data,20,2);CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_ERR_INDEX_RANGE);word(data,20,1);
    module->ownership_size=55;CHECK(nvm_ownership_contracts_validate(module,&needs)!=NVM_V2_OK);
    module->ownership_size=56;CHECK(nvm_ownership_contracts_validate(module,&needs)==NVM_V2_OK);
    word(data,0,NVM_OWNERSHIP_PATH_VERSION);
    fact=(NvmUnionVariantFact){99,99,99,99};
    NvmV2Result query=nvm_ownership_union_variant(module,0,0,&fact);
    CHECK(query==NVM_V2_ERR_SECTION_RANGE || query==NVM_V2_ERR_FORMAT_VERSION);
    word(data,0,NVM_OWNERSHIP_UNION_VERSION);
    nvm_module_free(module);
}

int main(int argc, char **argv) {
    check_union_transport();
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
