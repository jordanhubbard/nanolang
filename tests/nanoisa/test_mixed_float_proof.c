/* I query pending bytecode only. No VM or generated program is executed. */
#include "mixed_float_proof.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "assembler.h"
#include "verifier.h"
#include "isa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
static long budget=-1;
#define CHECK(x) do {checks++;assert(x);} while(0)
void *mfp_test_malloc(size_t n) {if(!budget)return NULL;if(budget>0)budget--;return malloc(n);}
void *mfp_test_calloc(size_t n,size_t s) {if(!budget)return NULL;if(budget>0)budget--;return calloc(n,s);}
static void word(uint8_t *p,uint32_t x) {for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(x>>(8*i));}
static void half(uint8_t *p,uint16_t x) {p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);}
typedef struct {uint8_t tag;uint32_t layout;} Type;
#define SCALAR(t) {t,NVM_V2_NO_INDEX}
static NvmModule *build_with_ordinary_owner(const char *body,const Type *locals,unsigned count,const char *helpers,bool other_int,bool ordinary_owner) {
    size_t size=strlen(body)+(helpers?strlen(helpers):0)+256;
    char *text=malloc(size);CHECK(text);
    unsigned layout_count=ordinary_owner?6:5;
    snprintf(text,size,".types %u 0 0\n.entry main\n.function main 0 %u 0 int 1\n%s.end\n%s",layout_count,count,body,helpers?helpers:"");
    AsmResult err;NvmModule *m=asm_assemble_unverified(text,&err);free(text);
    if(!m)fprintf(stderr,"assembly: %s\n",err.message);
    CHECK(m);
    NvmV2LayoutField integer={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField array={TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField child={TAG_STRUCT,1,NVM_V2_NO_INDEX};
    NvmV2LayoutField owned_child={TAG_STRUCT,0,NVM_V2_NO_INDEX};
    NvmV2Layout rows[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&integer},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&array},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,other_int?&integer:&array},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&child},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&owned_child},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&child}};
    NvmV2Layouts layouts={rows,layout_count};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    uint32_t bytes=20;
    for(uint32_t f=0;f<m->function_count;f++)bytes+=4+8*(m->functions[f].local_count+1);
    m->ownership_data=calloc(bytes,1);m->ownership_size=bytes;CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,1);word(p+4,layout_count);p[8]=3;p[9]=p[10]=p[11]=1;p[12]=3;if(ordinary_owner)p[13]=3;word(p+16,m->function_count);p+=20;
    for(uint32_t f=0;f<m->function_count;f++) {
        NvmFunctionEntry *fn=&m->functions[f];half(p,fn->local_count);half(p+2,fn->arity);p+=4;
        p[0]=fn->result_count?fn->result_tag:TAG_VOID;word(p+4,NVM_V2_NO_INDEX);p+=8;
        for(uint16_t j=0;j<fn->local_count;j++) {
            Type t=f?(Type){TAG_STRUCT,0}:locals[j];p[0]=t.tag;word(p+4,t.layout);p+=8;
        }
    }
    return m;
}
static NvmModule *build(const char *body,const Type *locals,unsigned count,const char *helpers,bool other_int) {
    return build_with_ordinary_owner(body,locals,count,helpers,other_int,false);
}
static NvmMixedFloatProof *expect(NvmModule *m,NvmMixedShapeStatus wanted) {
    NvmMixedFloatProof sentinel={0},*p=&sentinel;
    NvmMixedShapeResult r=nvm_analyze_mixed_float_origins(m,&p);
    if(r.status!=wanted)fprintf(stderr,"wanted %d got %d f%u pc%u: %s\n",wanted,r.status,r.function,r.pc,r.message);
    CHECK(r.status==wanted);
    if(wanted==NVM_MIXED_SHAPE_PROVED){CHECK(p!=&sentinel && p->requires_affine_verification);return p;}
    CHECK(p==&sentinel);return NULL;
}
static void positive_and_faults(void) {
    const char *body="PUSH_I64 7\nOWN_PACK 0\nCALL 1\nPUSH_I64 7\nEQ\nASSERT\n"
        "PUSH_F64 1.5\nPUSH_F64 2.5\nARR_LITERAL 3 2\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 1\nARR_GET\n"
        "PUSH_F64 2.5\nF64_EQ\nASSERT\nPUSH_I64 0\nRET\n";
    const char *helper=".function close 1 1 0 int 1\nOWN_UNPACK_LOCAL 0\nRET\n.end\n.parameters 1 struct\n";
    Type locals[]={{TAG_STRUCT,1},SCALAR(TAG_ARRAY)};
    NvmModule *m=build(body,locals,2,helper,false);
    bool needs=false;NvmV2Result old=nvm_ownership_contracts_validate(m,&needs);CHECK(old!=NVM_V2_OK);
    CHECK(nvm_verify(m).ok);
    NvmMixedFloatProof *p=expect(m,NVM_MIXED_SHAPE_PROVED);
    CHECK(p->origin_count==2 && p->fields[0].tags==(1u<<TAG_ARRAY));
    CHECK(p->fields[0].origins==1 && p->view->classes[1]==NVM_MIXED_PENDING_ARRAY_PROOF);
    CHECK(p->view->global_to_managed[1]==NVM_V2_NO_INDEX);
    unsigned reads=0,obligations=0;
    for(uint32_t i=0;i<p->obligation_count;i++) {
        NvmMixedScalarObligation *o=&p->obligations[i];
        if(o->read_tags){CHECK(o->read_tags==((1u<<TAG_FLOAT)|(1u<<TAG_VOID)));reads++;}
        if(o->required_tags==(1u<<TAG_FLOAT) && (o->actual_tags&(1u<<TAG_VOID)))obligations++;
    }
    CHECK(reads==1 && obligations>=1);nvm_mixed_float_proof_free(p);
    uint8_t *code=malloc(m->code_size),*owned=malloc(m->ownership_size);CHECK(code&&owned);
    memcpy(code,m->code,m->code_size);memcpy(owned,m->ownership_data,m->ownership_size);
    unsigned failures=0,successes=0;
    for(long n=0;n<80;n++) {
        NvmMixedFloatProof sentinel={0};p=&sentinel;budget=n;
        NvmMixedShapeResult r=nvm_analyze_mixed_float_origins(m,&p);budget=-1;
        if(r.status==NVM_MIXED_SHAPE_MEMORY){CHECK(p==&sentinel);failures++;}
        else {CHECK(r.status==NVM_MIXED_SHAPE_PROVED);nvm_mixed_float_proof_free(p);successes++;}
        CHECK(!memcmp(code,m->code,m->code_size) && !memcmp(owned,m->ownership_data,m->ownership_size));
    }
    CHECK(failures>20 && successes>0);
    CHECK(nvm_ownership_contracts_validate(m,&needs)==old && nvm_verify(m).ok);
    free(code);free(owned);nvm_module_free(m);
}
static void loop_unions(void) {
    Type locals[]={SCALAR(TAG_ARRAY),SCALAR(TAG_ARRAY),SCALAR(TAG_BOOL),{TAG_STRUCT,1}};
    for(unsigned order=0;order<2;order++) {
        char body[1800];snprintf(body,sizeof body,
            "ARR_NEW 3\nSTORE_LOCAL 0\nPUSH_F64 2.5\nARR_LITERAL 3 1\nSTORE_LOCAL 1\n"
            "PUSH_BOOL 1\nSTORE_LOCAL 2\nloop:\nLOAD_LOCAL 2\nJMP_FALSE second\nLOAD_LOCAL %u\nJMP pack\n"
            "second:\nLOAD_LOCAL %u\npack:\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 3\n"
            "PUSH_BOOL 0\nSTORE_LOCAL 2\nLOAD_LOCAL 2\nJMP_TRUE loop\n"
            "LOAD_LOCAL 3\nAGG_GET 0\nDUP\nPUSH_F64 1.0\nARR_PUSH\nPOP\n"
            "DUP\nPUSH_I64 0\nPUSH_F64 3.0\nARR_SET\nPOP\nPUSH_I64 -1\nARR_GET\nPOP\nPUSH_I64 0\nRET\n",order,1-order);
        NvmModule *m=build(body,locals,4,NULL,false);NvmMixedFloatProof *p=expect(m,NVM_MIXED_SHAPE_PROVED);
        CHECK(p->origin_count==3 && p->fields[0].origins==3 && p->checked_writes==2);
        nvm_mixed_float_proof_free(p);nvm_module_free(m);
    }
    for(unsigned order=0;order<2;order++) {
        char body[1200];snprintf(body,sizeof body,
            "PUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 3\nAGG_PACK 0 %u 0 1\nJMP read\n"
            "other:\nARR_NEW 3\nAGG_PACK 0 %u 0 1\nread:\nAGG_GET 0\nPUSH_I64 9\nARR_GET\nPOP\nPUSH_I64 0\nRET\n",1+order,2-order);
        NvmModule *m=build(body,NULL,0,NULL,false);NvmMixedFloatProof *p=expect(m,NVM_MIXED_SHAPE_PROVED);
        CHECK(p->origin_count==4);nvm_mixed_float_proof_free(p);nvm_module_free(m);
    }
}
static void nested_owner_tokens(void) {
    Type locals[]={{TAG_STRUCT,4},{TAG_STRUCT,0}};
    NvmModule *m=build("PUSH_I64 7\nOWN_PACK 0\nOWN_PACK 4\nOWN_STORE_LOCAL 0\n"
        "OWN_UNPACK_LOCAL 0\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n",locals,2,NULL,false);
    NvmMixedFloatProof *p=expect(m,NVM_MIXED_SHAPE_PROVED);
    nvm_mixed_float_proof_free(p);nvm_module_free(m);
    m=build_with_ordinary_owner("ARR_NEW 3\nAGG_PACK 0 1 0 1\nOWN_PACK 5\nPOP\nPUSH_I64 0\nRET\n",NULL,0,NULL,false,true);
    expect(m,NVM_MIXED_SHAPE_INVALID);nvm_module_free(m);
}
static void refusal_cases(void) {
    struct {const char *body;NvmMixedShapeStatus status;} cases[]={
        {"LOAD_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"PUSH_BOOL 0\nJMP_FALSE done\nARR_NEW 3\nSTORE_LOCAL 0\ndone:\nLOAD_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"ARR_NEW 3\nPUSH_I64 1\nARR_PUSH\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"ARR_NEW 1\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"ARR_NEW 3\nAGG_PACK 0 2 0 1\nAGG_PACK 0 3 0 1\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"PUSH_I64 1\nOWN_PACK 0\nDUP\nPOP\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"PUSH_I64 1\nOWN_PACK 0\nAGG_PACK 0 1 0 1\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"PUSH_I64 0\nRET\nLOAD_GLOBAL 0\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"PUSH_I64 0\nRET\nNOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_UNRESOLVED},
        {"ARR_NEW 3\nAGG_PACK 0 1 0 0\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_INVALID},
        {"LOAD_LOCAL 99\nPOP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_INVALID},
        {"POP\nPUSH_I64 0\nRET\n",NVM_MIXED_SHAPE_INVALID},
    };
    Type array[]={SCALAR(TAG_ARRAY)};
    for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++) {
        NvmModule *m=build(cases[i].body,array,1,NULL,false);expect(m,cases[i].status);nvm_module_free(m);
    }
    const char *both="PUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 3\nAGG_PACK 0 1 0 1\nJMP read\n"
        "other:\nPUSH_I64 7\nAGG_PACK 0 2 0 1\nread:\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nPOP\nPUSH_I64 0\nRET\n";
    NvmModule *m=build(both,NULL,0,NULL,true);expect(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
    m=build("PUSH_I64 0\nRET\n",NULL,0,".function unused 0 0 0 int 1\nCALL 1\nRET\n.end\n",false);
    expect(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
    m=build("PUSH_I64 0\nRET\n",NULL,0,".function unused 0 0 0 int 1\nPOP\nPUSH_I64 0\nRET\n.end\n",false);
    expect(m,NVM_MIXED_SHAPE_INVALID);nvm_module_free(m);
    m=build("JMP target\ntarget:\nPUSH_I64 0\nRET\n",NULL,0,NULL,false);
    word(m->code+1,1);expect(m,NVM_MIXED_SHAPE_INVALID);nvm_module_free(m);
    m=build("PUSH_I64 0\nRET\n",NULL,0,NULL,false);
    m->ownership_data[11]=4;expect(m,NVM_MIXED_SHAPE_INVALID);nvm_module_free(m);
}
static void budgets_and_calls(void) {
    char *body=calloc(50000,1);CHECK(body);
    for(unsigned i=0;i<65;i++)strcat(body,"ARR_NEW 3\nPOP\n");
    strcat(body,"PUSH_I64 0\nRET\n");
    NvmModule *m=build(body,NULL,0,NULL,false);expect(m,NVM_MIXED_SHAPE_LIMIT);nvm_module_free(m);
    body[0]=0;for(unsigned i=0;i<4097;i++)strcat(body,"NOP\n");strcat(body,"PUSH_I64 0\nRET\n");
    m=build(body,NULL,0,NULL,false);expect(m,NVM_MIXED_SHAPE_LIMIT);nvm_module_free(m);
    body[0]=0;for(unsigned i=0;i<2200;i++)strcat(body,"NOP\n");strcat(body,"PUSH_I64 0\nRET\n");
    Type locals[257];for(unsigned i=0;i<257;i++)locals[i]=(Type)SCALAR(TAG_INT);
    m=build(body,locals,256,NULL,false);expect(m,NVM_MIXED_SHAPE_LIMIT);nvm_module_free(m);
    m=build("PUSH_I64 0\nRET\n",locals,257,NULL,false);expect(m,NVM_MIXED_SHAPE_LIMIT);nvm_module_free(m);free(body);
    m=build("CALL 2\nRET\n",NULL,0,
        ".function leaf 0 0 0 int 1\nARR_NEW 3\nPUSH_I64 0\nARR_GET\nPOP\nPUSH_I64 0\nRET\n.end\n"
        ".function middle 0 0 0 int 1\nCALL 1\nRET\n.end\n",false);
    NvmMixedFloatProof *p=expect(m,NVM_MIXED_SHAPE_PROVED);CHECK(p->origin_count==1);
    nvm_module_free(m);CHECK(p->origins[0].function==1);nvm_mixed_float_proof_free(p);
}
int main(void) {
    positive_and_faults();loop_unions();nested_owner_tokens();refusal_cases();budgets_and_calls();
    printf("%u mixed FLOAT proof checks passed; no pending module execution\n",checks);return 0;
}
