#include "affine_bytecode.h"
#include "ownership_contracts.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "verifier.h"
#include "nvm2c.h"
#include "isa.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static unsigned checks;
static const char *native_output,*module_output;
#define CHECK(c) do {checks++;assert(c);} while(0)
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
static unsigned allocation_attempts, fail_at;
static uint32_t forced_visit_limit;
uint32_t affine_bytecode_test_visit_limit(uint32_t limit) {
    return forced_visit_limit && forced_visit_limit<limit?forced_visit_limit:limit;
}
void *affine_bytecode_test_malloc(size_t size) {
    if (++allocation_attempts==fail_at) return NULL;
    return malloc(size);
}
void *affine_bytecode_test_calloc(size_t count,size_t size) {
    if (++allocation_attempts==fail_at) return NULL;
    return calloc(count,size);
}
void *affine_bytecode_test_realloc(void *old,size_t size) {
    if (++allocation_attempts==fail_at) return NULL;
    return realloc(old,size);
}
#endif
static void word(uint8_t *p,uint32_t v) {for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));}
static void slot(uint8_t *p,uint8_t tag,uint8_t mode) {
    p[0]=tag;p[1]=mode;word(p+4,(tag==TAG_STRUCT || tag==TAG_UNION)?0:NVM_V2_NO_INDEX);
}
static const char *tag_name(uint8_t tag) {
    switch(tag) {
    case TAG_STRUCT:return "struct";case TAG_UNION:return "union";case TAG_INT:return "int";
    case TAG_BOOL:return "bool";case TAG_FLOAT:return "float";
    default:return "void";
    }
}
static NvmModule *union_fixture(const char *body,uint16_t params,uint16_t locals,
                                uint8_t result) {
    char parameters[256]={0};size_t parameter_bytes=0;
    if (params) {
        int wrote=snprintf(parameters,sizeof(parameters),".parameters 0");
        CHECK(wrote>0 && (size_t)wrote<sizeof(parameters));parameter_bytes=(size_t)wrote;
        for (uint16_t i=0;i<params;i++) {
            wrote=snprintf(parameters+parameter_bytes,sizeof(parameters)-parameter_bytes," union");
            CHECK(wrote>0 && (size_t)wrote<sizeof(parameters)-parameter_bytes);
            parameter_bytes+=(size_t)wrote;
        }
        CHECK(parameter_bytes+1<sizeof(parameters));parameters[parameter_bytes++]='\n';
        parameters[parameter_bytes]='\0';
    }
    char source[8192];int used=snprintf(source,sizeof(source),
        ".types 0 0 1\n.entry 0\n.string \"text\"\n.function inspect %u %u 0 %s %u\n%s.end\n%s",
        params,locals,tag_name(result),result!=TAG_VOID,body,parameters);
    CHECK(used>0 && (size_t)used<sizeof(source));
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m) {
        fprintf(stderr,"%s\n",assembled.message);
    }
    CHECK(m);
    uint32_t identity=nvm_add_string(m,"Choice<int,string>",18);
    uint32_t v0=nvm_add_string(m,"IntValue",8),v1=nvm_add_string(m,"TextPair",8);
    uint32_t v2=nvm_add_string(m,"Empty",5),f0=nvm_add_string(m,"value",5);
    uint32_t f1=nvm_add_string(m,"left",4),f2=nvm_add_string(m,"right",5);
    CHECK(identity!=UINT32_MAX && v0!=UINT32_MAX && v1!=UINT32_MAX &&
          v2!=UINT32_MAX && f0!=UINT32_MAX && f1!=UINT32_MAX && f2!=UINT32_MAX);
    NvmV2LayoutField fields[]={{TAG_INT,NVM_V2_NO_INDEX,f0},
        {TAG_STRING,NVM_V2_NO_INDEX,f1},{TAG_BOOL,NVM_V2_NO_INDEX,f2}};
    NvmV2Layout layout={NVM_V2_LAYOUT_UNION,3,identity,fields};
    NvmV2Layouts layouts={&layout,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    uint32_t at=28+8u*locals;
    m->ownership_size=at+56;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,NVM_OWNERSHIP_EXTENSION_VERSION);word(p+4,1);
    word(p+12,1);p[16]=(uint8_t)locals;p[17]=(uint8_t)(locals>>8);
    p[18]=(uint8_t)params;p[19]=(uint8_t)(params>>8);slot(p+20,result,0);
    for(uint16_t i=0;i<locals;i++)slot(p+28+8*i,TAG_UNION,0);
    word(p+at,4);word(p+at+4,0);word(p+at+8,1);
    p[at+12]=NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS;
    p[at+14]=NVM_OWNERSHIP_EXTENSION_REVISION_1;word(p+at+16,36);
    word(p+at+20,1);word(p+at+24,0);p[at+28]=3;
    word(p+at+32,v0);p[at+38]=1;
    word(p+at+40,v1);p[at+44]=1;p[at+46]=2;
    word(p+at+48,v2);p[at+52]=3;
    bool needs=false;NvmV2Result contract=nvm_ownership_contracts_validate(m,&needs);
    if(contract!=NVM_V2_OK)fprintf(stderr,"union ownership contract: %d\n",contract);
    CHECK(contract==NVM_V2_OK && needs);
    return m;
}
static void analyze_union(const char *body,uint16_t params,uint16_t locals,uint8_t result,
                          bool accepted,const char *reason) {
    NvmModule *m=union_fixture(body,params,locals,result);
    NvmAffineAnalysis got=nvm_affine_analyze_function(m,0);
    if(got.ok!=accepted)fprintf(stderr,"Unexpected union analysis: %s at%u\n%s",got.message,got.byte_offset,body);
    CHECK(got.ok==accepted);if(reason)CHECK(strstr(got.message,reason));
    if(accepted && !params) {
        NvmVerifyResult verified=nvm_verify_owned_module(m);
        if(!verified.ok)fprintf(stderr,"union verification: %s\n",verified.error_msg);
        CHECK(verified.ok);
        CHECK(nvm_verify(m).ok);
        char error[256]={0};char *native=nvm2c_emit(m,error,sizeof(error));
        if(!native)fprintf(stderr,"union native emission: %s\n",error);
        CHECK(native!=NULL);
        if(native_output) {
            FILE *file=fopen(native_output,"wb");CHECK(file!=NULL);
            size_t length=strlen(native);CHECK(fwrite(native,1,length,file)==length);
            CHECK(fclose(file)==0);
            NvmV2Module wire={0};size_t bytes=0;
            CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
            CHECK(nvm_v2_module_serialize(&wire,NULL,0,&bytes)==NVM_V2_OK);
            uint8_t *serialized=malloc(bytes);CHECK(serialized!=NULL);
            CHECK(nvm_v2_module_serialize(&wire,serialized,bytes,NULL)==NVM_V2_OK);
            file=fopen(module_output,"wb");CHECK(file!=NULL);
            CHECK(fwrite(serialized,1,bytes,file)==bytes);CHECK(fclose(file)==0);
            free(serialized);nvm_v2_module_free(&wire);
            native_output=NULL;module_output=NULL;
        }
        free(native);
    }
    nvm_module_free(m);
}
static NvmModule *fixture(const char *body,uint16_t params,uint16_t locals,
                           const uint8_t *tags,const uint8_t *modes,uint8_t result,bool resource) {
    char source[32768];int used=snprintf(source,sizeof(source),
        ".types 1 0 0\n.entry 0\n.function inspect %u %u 0 %s %u\n%s.end\n",
        params,locals,tag_name(result),result!=TAG_VOID,body);
    CHECK(used>0 && (size_t)used<sizeof(source));
    if (params) {
        used+=snprintf(source+used,sizeof(source)-used,".parameters 0");
        for(uint16_t i=0;i<params;i++)used+=snprintf(source+used,sizeof(source)-used," %s",tag_name(tags[i]));
        snprintf(source+used,sizeof(source)-used,"\n");
    }
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout layout={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field};
    NvmV2Layouts layouts={&layout,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=28+8*locals;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,1);word(p+4,1);p[8]=1|(resource?2:0);word(p+12,1);
    p[16]=(uint8_t)locals;p[17]=(uint8_t)(locals>>8);p[18]=(uint8_t)params;p[19]=(uint8_t)(params>>8);
    slot(p+20,result,0);
    for(uint16_t i=0;i<locals;i++)slot(p+28+8*i,tags[i],modes?modes[i]:0);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK);
    if(needs)CHECK(!nvm_verify(m).ok);
    return m;
}
static void analyze(const char *body,uint16_t params,uint16_t locals,const uint8_t *tags,
                      const uint8_t *modes,uint8_t result,bool resource,bool accepted,const char *reason) {
    NvmModule *m=fixture(body,params,locals,tags,modes,result,resource);
    uint8_t *before=malloc(m->ownership_size);CHECK(before);memcpy(before,m->ownership_data,m->ownership_size);
    NvmAffineAnalysis got=nvm_affine_analyze_function(m,0);
    if(got.ok!=accepted)fprintf(stderr,"Unexpected analysis: %s at%u\n%s",got.message,got.byte_offset,body);
    CHECK(got.ok==accepted);
    CHECK(got.visits<=NVM_AFFINE_MAX_INSTRUCTIONS*((uint32_t)locals+1u));
    CHECK(got.reachable<=NVM_AFFINE_MAX_INSTRUCTIONS);
    if(reason)CHECK(strstr(got.message,reason));
    CHECK(!memcmp(before,m->ownership_data,m->ownership_size));
    if(resource)CHECK(!nvm_verify(m).ok); /* Analysis never admits runtime. */
    free(before);nvm_module_free(m);
}
static NvmModule *owned_union_fixture(const char *body) {
    NvmModule *m=union_fixture(body,1,3,TAG_VOID);
    NvmV2Layouts old={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&old)==NVM_V2_OK);
    uint32_t name=nvm_add_string(m,"Handle",6);
    NvmV2LayoutField fd={TAG_INT,NVM_V2_NO_INDEX,old.items[0].fields[0].name_idx};
    old.items[0].fields[0].type_tag=TAG_STRUCT;old.items[0].fields[0].nested_idx=0;
    old.items[0].fields[1].type_tag=TAG_INT;
    NvmV2Layout items[]={{NVM_V2_LAYOUT_STRUCT,1,name,&fd},old.items[0]};
    NvmV2Layouts layouts={items,2};m->struct_count=1;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&old);
    uint8_t *p=m->ownership_data;word(p,NVM_OWNERSHIP_UNION_GRAPH_VERSION);word(p+4,2);
    p[8]=p[9]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;
    slot(p+28,TAG_UNION,0);word(p+32,1);
    slot(p+36,TAG_STRUCT,0);slot(p+44,TAG_UNION,0);word(p+48,1);
    /* The union extension follows the three local descriptors and names layout 1. */
    word(p+76,1);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}
static void owned_union_case(const char *body,bool accepted,const char *reason) {
    NvmModule *m=owned_union_fixture(body);
    NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);
    if(result.ok!=accepted)fprintf(stderr,"Owned union analysis: %s at %u\n%s",result.message,result.byte_offset,body);
    CHECK(result.ok==accepted);
    if(reason)CHECK(strstr(result.message,reason));
    CHECK(!nvm_verify(m).ok);
    nvm_module_free(m);
}
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
static void owned_union_allocations(void) {
    NvmModule *m=owned_union_fixture("LOAD_LOCAL 0\nMATCH_TAG 0 owner\nPOP\nHALT\n"
        "owner:\nPOP\nOWN_UNPACK_VARIANT 0 0 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nPOP\nRET\n");
    allocation_attempts=0;fail_at=0;
    CHECK(nvm_affine_analyze_function(m,0).ok);
    unsigned attempts=allocation_attempts;CHECK(attempts>0);
    for(unsigned failure=1;failure<=attempts;failure++) {
        allocation_attempts=0;fail_at=failure;
        NvmAffineAnalysis result=nvm_affine_analyze_function(m,0);
        fail_at=0;CHECK(!result.ok);
    }
    nvm_module_free(m);
}
#endif
static void owned_union_flow(void) {
    DecodedInstruction instruction={0},decoded={0};uint8_t bytes[7];
    instruction.opcode=OP_OWN_UNPACK_VARIANT;
    instruction.operands[0].u16=0x1234;instruction.operands[1].u16=0x5678;instruction.operands[2].u16=0x9abc;
    CHECK(isa_encode(&instruction,bytes,sizeof bytes)==7);
    const uint8_t expected[]={0x97,0x34,0x12,0x78,0x56,0xbc,0x9a};
    CHECK(!memcmp(bytes,expected,7));CHECK(isa_decode(bytes,7,&decoded)==7);
    CHECK(decoded.operands[0].u16==0x1234 && decoded.operands[1].u16==0x5678 && decoded.operands[2].u16==0x9abc);
    for(unsigned n=0;n<7;n++)CHECK(!isa_decode(bytes,n,&decoded));
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 0 owner\nMATCH_TAG 1 ordinary\nMATCH_TAG 2 empty\nPOP\nHALT\n"
        "owner:\nPOP\nOWN_UNPACK_VARIANT 0 0 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nPOP\nRET\n"
        "ordinary:\nPOP\nOWN_UNPACK_VARIANT 0 1 2\nPOP\nPOP\nRET\n"
        "empty:\nPOP\nOWN_UNPACK_VARIANT 0 2 0\nRET\n",true,NULL);
    owned_union_case("OWN_UNPACK_VARIANT 0 0 1\nRET\n",false,"selected union");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 2 arm\nPOP\nHALT\narm:\nPOP\n"
        "PUSH_BOOL 1\nJMP_FALSE keep\nOWN_UNPACK_VARIANT 0 2 0\nJMP join\n"
        "keep:\nNOP\njoin:\nRET\n",false,NULL);
    owned_union_case("PUSH_I64 7\nAGG_PACK 1 0 0 1\nRET\n",false,"fields in declaration order");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 0 arm\nPOP\nHALT\narm:\n"
        "OWN_UNPACK_VARIANT 0 0 1\nRET\n",false,"unobserved");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 0 arm\nPOP\nHALT\narm:\nPOP\n"
        "OWN_UNPACK_VARIANT 0 0 2\nRET\n",false,"payload count");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 0 arm\nPOP\nHALT\narm:\nPOP\n"
        "OWN_UNPACK_VARIANT 0 1 2\nRET\n",false,"selected union");
    owned_union_case("LOAD_LOCAL 0\nOWN_MOVE_LOCAL 0\nRET\n",false,"observed owner");
    owned_union_case("LOAD_LOCAL 0\nDUP\nRET\n",false,"duplicate reference authority");
    owned_union_case("OWN_MOVE_LOCAL 0\nDUP\nRET\n",false,"duplicate reference authority");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 2 arm\nPOP\nHALT\narm:\nPOP\n"
        "OWN_UNPACK_VARIANT 0 2 0\nOWN_UNPACK_VARIANT 0 2 0\nRET\n",false,"selected union");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 2 arm\nPOP\nHALT\narm:\nPOP\n"
        "OWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 2\nOWN_MOVE_LOCAL 2\nOWN_STORE_LOCAL 0\n"
        "OWN_UNPACK_VARIANT 0 2 0\nRET\n",false,"selected union");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 2 arm\nPOP\nHALT\narm:\nPOP\n"
        "AGG_PACK 1 0 2 0\nOWN_STORE_LOCAL 0\nRET\n",false,"owner destination");
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 0 arm\nPOP\nHALT\narm:\nPOP\n"
        "OWN_UNPACK_VARIANT 0 0 1\nPOP\nRET\n",false,"scalar discard");
    owned_union_case("OWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 2\nLOAD_LOCAL 2\nMATCH_TAG 2 arm\nPOP\nHALT\n"
        "arm:\nPOP\nOWN_UNPACK_VARIANT 2 2 0\nRET\n",true,NULL);
    owned_union_case("LOAD_LOCAL 0\nMATCH_TAG 2 arm\nPOP\nHALT\narm:\nPOP\n"
        "OWN_UNPACK_VARIANT 0 2 0\nPUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 2\n"
        "LOAD_LOCAL 2\nMATCH_TAG 0 made\nPOP\nHALT\nmade:\nPOP\n"
        "OWN_UNPACK_VARIANT 2 0 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nPOP\nRET\n",true,NULL);
}

int main(int argc,char **argv) {
    owned_union_flow();
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
    owned_union_allocations();
#endif
    if(argc==3){native_output=argv[1];module_output=argv[2];}
    else CHECK(argc==1);
    uint8_t tags[4]={TAG_STRUCT,TAG_BOOL,TAG_INT,TAG_INT};
    uint8_t shared[4]={1,0,0,0},exclusive[4]={2,0,0,0};
    analyze("LOAD_LOCAL 0\nAGG_GET 0\nRET\n",1,1,tags,shared,TAG_INT,true,true,NULL);
    analyze_union("PUSH_STR 0\nPUSH_BOOL 1\nAGG_PACK 1 0 1 2\nSTORE_LOCAL 0\n"
                  "LOAD_LOCAL 0\nMATCH_TAG 1 matched\nPOP\nPUSH_BOOL 0\nRET\n"
                  "matched:\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nAGG_GET 1\nRET\n",
                  0,2,TAG_BOOL,true,NULL);
    if(argc==3)CHECK(native_output==NULL && module_output==NULL);
    analyze_union("PUSH_STR 0\nPUSH_BOOL 1\nAGG_PACK 1 0 1 2\nMATCH_TAG 0 matched\n"
                  "POP\nPUSH_BOOL 0\nRET\nmatched:\nAGG_GET 9\nRET\n",
                  0,0,TAG_BOOL,true,NULL);
    analyze_union("PUSH_STR 0\nPUSH_BOOL 1\nAGG_PACK 1 0 1 2\nMATCH_TAG 1 matched\n"
                  "AGG_GET 9\nRET\nmatched:\nAGG_GET 1\nRET\n",
                  0,0,TAG_BOOL,true,NULL);
    analyze_union("LOAD_LOCAL 0\nMATCH_TAG 0 matched\nPOP\nPUSH_I64 0\nRET\n"
                  "matched:\nAGG_GET 0\nRET\n",1,1,TAG_INT,true,NULL);
    analyze_union("LOAD_LOCAL 0\nMATCH_TAG 0 matched\nPOP\nPUSH_I64 0\nRET\n"
                  "matched:\nPOP\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n",1,1,TAG_INT,false,
                  "proven scalar-union variant");
    analyze_union("LOAD_LOCAL 0\nMATCH_TAG 0 matched\nPOP\nPUSH_I64 0\nRET\n"
                  "matched:\nPOP\nLOAD_LOCAL 1\nAGG_GET 0\nRET\n",2,2,TAG_INT,false,
                  "proven scalar-union variant");
    analyze_union("LOAD_LOCAL 0\nMATCH_TAG 0 matched\nJMP joined\n"
                  "matched:\nNOP\njoined:\nAGG_GET 0\nRET\n",1,1,TAG_INT,false,
                  "proven scalar-union variant");
    analyze_union("LOAD_LOCAL 0\nAGG_GET 0\nRET\n",1,1,TAG_INT,false,
                  "proven scalar-union variant");
    analyze_union("LOAD_LOCAL 0\nMATCH_TAG 0 matched\nPOP\nPUSH_I64 0\nRET\n"
                  "matched:\nAGG_GET 1\nRET\n",1,1,TAG_INT,false,
                  "payload field");
    analyze_union("PUSH_STR 0\nAGG_PACK 1 0 1 1\nPOP\nPUSH_BOOL 0\nRET\n",
                  0,0,TAG_BOOL,false,"constructor shape");
    analyze("LOAD_LOCAL 0\nSTRUCT_GET 0\nRET\n",1,1,tags,exclusive,TAG_INT,true,true,NULL);
    analyze("LOAD_LOCAL 1\nJMP_FALSE otherwise\nLOAD_LOCAL 0\nAGG_GET 0\nJMP joined\n"
            "otherwise:\nPUSH_I64 42\njoined:\nRET\n",2,2,tags,shared,TAG_INT,true,true,NULL);
    analyze("LOAD_LOCAL 1\nJMP_FALSE otherwise\nLOAD_LOCAL 0\nJMP joined\n"
            "otherwise:\nLOAD_LOCAL 0\njoined:\nAGG_GET 0\nRET\n",2,2,tags,shared,TAG_INT,true,true,NULL);
    analyze("LOAD_LOCAL 0\nDUP\nAGG_GET 0\nRET\n",1,1,tags,shared,TAG_INT,true,false,"duplicate");
    analyze("LOAD_LOCAL 0\nRET\n",1,1,tags,shared,TAG_STRUCT,true,false,"escape");
    analyze("LOAD_LOCAL 0\nPOP\nRET\n",1,1,tags,shared,TAG_VOID,true,false,"discard");
    analyze("LOAD_LOCAL 0\nSTORE_LOCAL 2\nPUSH_I64 0\nRET\n",1,3,tags,shared,TAG_INT,true,false,"escape");
    analyze("LOAD_LOCAL 0\nAGG_GET 1\nRET\n",1,1,tags,shared,TAG_INT,true,false,"field observation");
    analyze("LOAD_LOCAL 0\nAGG_GET 0\nRET\n",1,1,tags,NULL,TAG_INT,true,false,"owned obligations");
    analyze("LOAD_LOCAL 0\nAGG_GET 0\nRET\n",1,1,tags,NULL,TAG_INT,false,true,NULL);
    analyze("LOAD_LOCAL 0\nPUSH_I64 8\nAGG_SET 0\nRET\n",1,1,tags,exclusive,TAG_STRUCT,true,false,"instruction contract");
    analyze("LOAD_LOCAL 0\nCALL 0\nRET\n",1,1,tags,shared,TAG_INT,true,false,"checked acyclic owned value call");
    analyze("PUSH_I64 1\nRET\nLOAD_LOCAL 0\nTAIL_CALL 0\nRET\n",1,1,tags,shared,TAG_INT,true,false,"instruction contract");
    uint8_t aliases[3]={TAG_STRUCT,TAG_STRUCT,TAG_BOOL},modes[3]={1,1,0};
    analyze("LOAD_LOCAL 2\nJMP_FALSE other\nLOAD_LOCAL 0\nJMP joined\n"
            "other:\nLOAD_LOCAL 1\njoined:\nAGG_GET 0\nRET\n",3,3,aliases,modes,TAG_INT,true,false,"every join");
    uint8_t scalar_tags[3]={TAG_INT,TAG_INT,TAG_BOOL};
    analyze("PUSH_I64 0\nSTORE_LOCAL 1\nloop:\nLOAD_LOCAL 1\nPUSH_I64 4\nLT\n"
            "JMP_FALSE done\nLOAD_LOCAL 1\nPUSH_I64 1\nADD\nSTORE_LOCAL 1\nJMP loop\n"
            "done:\nLOAD_LOCAL 1\nRET\n",0,2,scalar_tags,NULL,TAG_INT,false,true,NULL);
    analyze("loop:\nPUSH_I64 1\nSTORE_LOCAL 1\nJMP loop\n",0,2,scalar_tags,NULL,TAG_VOID,false,true,NULL);
    analyze("LOAD_LOCAL 0\nRET\n",0,1,scalar_tags,NULL,TAG_INT,false,false,"live checked local");
    analyze("PUSH_BOOL 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nRET\n",0,1,scalar_tags,NULL,TAG_INT,false,false,"exact scalar local");
    analyze("PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 1\nJMP joined\n"
            "other:\nPUSH_BOOL 1\njoined:\nPOP\nPUSH_I64 0\nRET\n",0,0,NULL,NULL,TAG_INT,false,false,"every join");
    analyze("PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 1\nSTORE_LOCAL 0\nJMP joined\n"
            "other:\nNOP\njoined:\nPUSH_I64 0\nRET\n",0,1,scalar_tags,NULL,TAG_INT,false,true,NULL);
    /* I retain unused branch-local initialization and require every incoming
     * path for any later load, independently of predecessor visitation order. */
    analyze("PUSH_BOOL 1\nJMP_FALSE other\nNOP\nJMP joined\n"
            "other:\nPUSH_I64 1\nSTORE_LOCAL 0\njoined:\nPUSH_I64 0\nRET\n",
            0,1,scalar_tags,NULL,TAG_INT,false,true,NULL);
    analyze("PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 1\nSTORE_LOCAL 0\nJMP joined\n"
            "other:\nPUSH_I64 2\nSTORE_LOCAL 0\njoined:\nLOAD_LOCAL 0\nRET\n",
            0,1,scalar_tags,NULL,TAG_INT,false,true,NULL);
    analyze("loop:\nPUSH_BOOL 0\nJMP_FALSE done\nPUSH_I64 3\nSTORE_LOCAL 0\nJMP loop\n"
            "done:\nPUSH_I64 0\nRET\n",0,1,scalar_tags,NULL,TAG_INT,false,true,NULL);
    analyze("loop:\nPUSH_BOOL 0\nJMP_FALSE done\nPUSH_I64 3\nSTORE_LOCAL 0\nJMP loop\n"
            "done:\nLOAD_LOCAL 0\nRET\n",0,1,scalar_tags,NULL,TAG_INT,false,false,"live checked local");
    for (unsigned initialized_target=0;initialized_target<2;initialized_target++) {
        char body[2048];
        const char *initialized="PUSH_I64 1\nSTORE_LOCAL 0\n";
        const char *delayed="NOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\n";
        snprintf(body,sizeof(body),"PUSH_BOOL 1\nJMP_FALSE other\n%sJMP joined\nother:\n%s"
            "joined:\nNOP\nNOP\nLOAD_LOCAL 0\nRET\n",
            initialized_target?delayed:initialized,initialized_target?initialized:delayed);
        analyze(body,0,1,scalar_tags,NULL,TAG_INT,false,false,"live checked local");
        /* The same normal diamond without a read converges by revisiting its
         * already processed descendants after the delayed predecessor. */
        char *load=strstr(body,"LOAD_LOCAL 0");CHECK(load);strcpy(load,"PUSH_I64 0\nRET\n");
        NvmModule *m=fixture(body,0,1,scalar_tags,NULL,TAG_INT,false);
        NvmAffineAnalysis got=nvm_affine_analyze_function(m,0);
        CHECK(got.ok);CHECK(got.visits>got.reachable);
        CHECK(got.visits<=got.reachable*2u);
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
        forced_visit_limit=got.visits-1;
        NvmAffineAnalysis limited=nvm_affine_analyze_function(m,0);
        CHECK(!limited.ok && strstr(limited.message,"visit count"));
        CHECK(limited.visits==forced_visit_limit);forced_visit_limit=0;
        CHECK(nvm_affine_analyze_function(m,0).ok);
        if (!initialized_target) for (unsigned failure=1;;failure++) {
            allocation_attempts=0;fail_at=failure;
            NvmAffineAnalysis trial=nvm_affine_analyze_function(m,0);
            fail_at=0;
            if (trial.ok) {CHECK(allocation_attempts<failure);break;}
            CHECK(failure<1000);CHECK(trial.message[0]);
            CHECK(nvm_affine_analyze_function(m,0).ok);
        }
#endif
        nvm_module_free(m);
    }
    analyze("PUSH_I64 1\nPUSH_I64 2\nRET\n",0,0,NULL,NULL,TAG_INT,false,false,"extra return");
    analyze("PUSH_I64 1\nNOP\n",0,0,NULL,NULL,TAG_INT,false,false,"fallthrough");
    analyze("PUSH_I64 1\nJMP_TRUE end\nend:\nRET\n",0,0,NULL,NULL,TAG_INT,false,false,"Boolean branch");
    analyze("PUSH_F64 2.0\nPUSH_F64 3.0\nF64_ADD\nRET\n",0,0,NULL,NULL,TAG_FLOAT,false,true,NULL);
    analyze("PUSH_I64 1\nPUSH_F64 2.0\nF64_ADD\nRET\n",0,0,NULL,NULL,TAG_FLOAT,false,false,"float arithmetic");
    analyze("PUSH_BOOL 1\nDUP\nAND\nNOT\nRET\n",0,0,NULL,NULL,TAG_BOOL,false,true,NULL);
    char deep[5000]={0};for(unsigned i=0;i<257;i++)strcat(deep,"PUSH_I64 1\n");strcat(deep,"RET\n");
    analyze(deep,0,0,NULL,NULL,TAG_INT,false,false,"analysis stack");
#ifdef AFFINE_BYTECODE_ALLOCATION_TEST
    NvmModule *allocation=fixture("LOAD_LOCAL 1\nJMP_FALSE otherwise\nLOAD_LOCAL 0\nAGG_GET 0\nJMP joined\n"
        "otherwise:\nPUSH_I64 42\njoined:\nRET\n",2,2,tags,shared,TAG_INT,true);
    for(unsigned failure=1;;failure++) {
        allocation_attempts=0;fail_at=failure;
        NvmAffineAnalysis got=nvm_affine_analyze_function(allocation,0);
        fail_at=0;
        if(got.ok) {CHECK(allocation_attempts<failure);break;}
        CHECK(failure<200);CHECK(got.message[0]);
        CHECK(nvm_affine_analyze_function(allocation,0).ok);
        CHECK(!nvm_verify(allocation).ok);
    }
    nvm_module_free(allocation);
#endif
    char *many=calloc((NVM_AFFINE_MAX_INSTRUCTIONS+1)*4+1,1);CHECK(many);
    for(unsigned i=0;i<NVM_AFFINE_MAX_INSTRUCTIONS;i++)memcpy(many+i*4,"NOP\n",4);
    memcpy(many+(NVM_AFFINE_MAX_INSTRUCTIONS-1)*4,"RET\n",4);
    analyze(many,0,0,NULL,NULL,TAG_VOID,false,true,NULL);
    memcpy(many+(NVM_AFFINE_MAX_INSTRUCTIONS-1)*4,"NOP\nRET\n",8);
    analyze(many,0,0,NULL,NULL,TAG_VOID,false,false,"instruction count");free(many);
    uint8_t many_locals[NVM_AFFINE_MAX_LOCALS+1];memset(many_locals,TAG_INT,sizeof(many_locals));
    analyze("RET\n",0,NVM_AFFINE_MAX_LOCALS+1,many_locals,NULL,TAG_VOID,false,false,"analysis size");
    CHECK(!nvm_affine_analyze_function(NULL,0).ok);
    printf("%u affine bytecode checks passed\n",checks);return 0;
}
