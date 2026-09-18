#include "affine_bytecode.h"
#include "ownership_contracts.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "verifier.h"
#include "isa.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static unsigned checks;
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
    p[0]=tag;p[1]=mode;word(p+4,tag==TAG_STRUCT?0:NVM_V2_NO_INDEX);
}
static const char *tag_name(uint8_t tag) {
    switch(tag) {
    case TAG_STRUCT:return "struct";case TAG_INT:return "int";
    case TAG_BOOL:return "bool";case TAG_FLOAT:return "float";
    default:return "void";
    }
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
int main(void) {
    uint8_t tags[4]={TAG_STRUCT,TAG_BOOL,TAG_INT,TAG_INT};
    uint8_t shared[4]={1,0,0,0},exclusive[4]={2,0,0,0};
    analyze("LOAD_LOCAL 0\nAGG_GET 0\nRET\n",1,1,tags,shared,TAG_INT,true,true,NULL);
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
    analyze("LOAD_LOCAL 0\nCALL 0\nRET\n",1,1,tags,shared,TAG_INT,true,false,"entry-to-helper consuming call");
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
