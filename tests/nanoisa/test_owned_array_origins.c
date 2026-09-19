/* I query fresh pending modules only; no VM or generated program executes. */
#include "owned_array_origins.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "assembler.h"
#include "isa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
static long budget=-1;
#define CHECK(x) do {checks++;assert(x);} while(0)
void *owner_origin_test_malloc(size_t n) {if(!budget)return NULL;if(budget>0)budget--;return malloc(n);}
void *owner_origin_test_calloc(size_t n,size_t s) {if(!budget)return NULL;if(budget>0)budget--;return calloc(n,s);}
static void word(uint8_t *p,uint32_t x) {for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(x>>(8*i));}
static void half(uint8_t *p,uint16_t x) {p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);}
typedef struct {uint8_t tag;uint32_t layout;} Type;
#define T(t) {t,NVM_V2_NO_INDEX}
#define OWNER(l) {TAG_STRUCT,l}
typedef struct {const char *body;uint16_t arity,count;Type result,locals[8];} Function;
static NvmModule *build(const Function *functions,unsigned count) {
    char text[32768];size_t used=(size_t)snprintf(text,sizeof text,".string value \"retained\"\n.types 5 0 0\n.entry 0\n");
    for(unsigned f=0;f<count;f++) {
        const Function *fn=&functions[f];
        int n=snprintf(text+used,sizeof(text)-used,".function f%u %u %u 0 %s %u\n%s.end\n",f,fn->arity,fn->count,
                       isa_tag_name(fn->result.tag),fn->result.tag!=TAG_VOID,fn->body);
        CHECK(n>0 && (size_t)n<sizeof(text)-used);used+=(size_t)n;
        if(fn->arity) {
            n=snprintf(text+used,sizeof(text)-used,".parameters %u",f);CHECK(n>0);used+=(size_t)n;
            for(unsigned p=0;p<fn->arity;p++) {n=snprintf(text+used,sizeof(text)-used," %s",isa_tag_name(fn->locals[p].tag));CHECK(n>0);used+=(size_t)n;}
            CHECK(used+2<sizeof text);text[used++]='\n';text[used]=0;
        }
    }
    AsmResult error;NvmModule *m=asm_assemble_unverified(text,&error);
    if(!m)fprintf(stderr,"assembly: %s\n",error.message);
    CHECK(m);
    NvmV2LayoutField integer={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField bundle[]={{TAG_STRUCT,0,NVM_V2_NO_INDEX},{TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX},{TAG_STRING,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2LayoutField outer[]={{TAG_STRUCT,1,NVM_V2_NO_INDEX},{TAG_STRUCT,1,NVM_V2_NO_INDEX}};
    NvmV2LayoutField pair[]={{TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX},{TAG_ARRAY,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2Layout rows[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&integer},
        {NVM_V2_LAYOUT_STRUCT,3,NVM_V2_NO_INDEX,bundle},{NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,outer},
        {NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,pair},{NVM_V2_LAYOUT_STRUCT,0,NVM_V2_NO_INDEX,NULL}};
    NvmV2Layouts layouts={rows,5};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    uint32_t bytes=20;for(unsigned f=0;f<count;f++)bytes+=4+8*(functions[f].count+1);
    m->ownership_data=calloc(bytes,1);CHECK(m->ownership_data);m->ownership_size=bytes;
    uint8_t *p=m->ownership_data;word(p,1);word(p+4,5);memset(p+8,3,5);word(p+16,count);p+=20;
    for(unsigned f=0;f<count;f++) {
        const Function *fn=&functions[f];half(p,fn->count);half(p+2,fn->arity);p+=4;
        p[0]=fn->result.tag;word(p+4,fn->result.layout);p+=8;
        for(unsigned n=0;n<fn->count;n++) {p[0]=fn->locals[n].tag;word(p+4,fn->locals[n].layout);p+=8;}
    }
    return m;
}
static NvmOwnedArrayOrigins *expect(NvmModule *m,NvmOwnerOriginStatus status) {
    unsigned char sentinel;NvmOwnedArrayOrigins *p=(void *)&sentinel;
    NvmOwnerOriginResult r=nvm_analyze_owned_array_origins(m,&p);
    if(r.status!=status)fprintf(stderr,"wanted%d got%d f%u pc%u: %s\n",status,r.status,r.function,r.pc,r.message);
    CHECK(r.status==status);
    if(status==NVM_OWNER_ORIGIN_PROVED) {CHECK(p!=(void *)&sentinel);return p;}
    CHECK(p==(void *)&sentinel);return NULL;
}
static void locations(NvmModule *m,NvmOwnedArrayOrigins *p) {
    NvmOwnerOriginCounts counts;CHECK(nvm_owned_array_origin_counts(p,&counts));uint32_t index=0,returns=0;
    for(uint32_t f=0;f<m->function_count;f++)for(uint32_t pc=0;pc<m->functions[f].code_length;) {
        DecodedInstruction in;uint32_t width=isa_decode(m->code+m->functions[f].code_offset+pc,m->functions[f].code_length-pc,&in);CHECK(width);
        NvmOwnerOriginObligation row;CHECK(nvm_owned_array_origin_obligation(p,index++,&row));
        CHECK(row.function==f && row.pc==pc);if(in.opcode==OP_RET)returns++;
        pc+=width;
    }
    CHECK(index==counts.obligations && returns>=m->function_count);
}
static void positive_and_faults(void) {
    Function f[]={
        {"ARR_NEW 3\nSTORE_LOCAL 0\nPUSH_I64 7\nOWN_PACK 0\nLOAD_LOCAL 0\nPUSH_STR value\nOWN_PACK 1\n"
         "PUSH_I64 8\nOWN_PACK 0\nARR_NEW 3\nPUSH_STR value\nOWN_PACK 1\nOWN_PACK 2\nCALL 1\nOWN_STORE_LOCAL 1\n"
         "LOAD_LOCAL 1\nAGG_GET 1\nAGG_GET 1\nARR_LEN\nPOP\nOWN_MOVE_LOCAL 1\nCALL 3\nPOP\n"
         "CALL 5\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nPOP\nPOP\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPOP\nPUSH_I64 0\nRET\n",0,4,T(TAG_INT),{T(TAG_ARRAY),OWNER(2),OWNER(1),OWNER(0)}},
        {"OWN_MOVE_LOCAL 0\nCALL 2\nRET\n",1,1,OWNER(2),{OWNER(2)}},
        {"OWN_MOVE_LOCAL 0\nRET\n",1,1,OWNER(2),{OWNER(2)}},
        {"OWN_UNPACK_LOCAL 0\nOWN_STORE_LOCAL 1\nOWN_STORE_LOCAL 2\n"
         "OWN_UNPACK_LOCAL 1\nPOP\nARR_LEN\nPOP\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPOP\n"
         "OWN_UNPACK_LOCAL 2\nPOP\nPOP\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nRET\n",1,4,T(TAG_INT),{OWNER(2),OWNER(1),OWNER(1),OWNER(0)}},
        {"OWN_MOVE_LOCAL 0\nRET\n",1,1,OWNER(3),{OWNER(3)}},
        {"PUSH_I64 9\nOWN_PACK 0\nARR_NEW 3\nPUSH_STR value\nOWN_PACK 1\nRET\n",0,0,OWNER(1),{{0}}}
    };
    NvmModule *m=build(f,6);NvmOwnedArrayOrigins *p=expect(m,NVM_OWNER_ORIGIN_PROVED);locations(m,p);
    NvmOwnerOriginCounts counts;CHECK(nvm_owned_array_origin_counts(p,&counts));CHECK(counts.functions==6 && counts.sites==3);
    NvmOwnerOriginSummary summary;
    CHECK(nvm_owned_array_origin_summary(p,0,&summary));CHECK(summary.reachable && !summary.conditional && !summary.formal_count && summary.required.words[0]==7);
    for(unsigned w=1;w<5;w++)CHECK(!summary.required.words[w]);
    for(unsigned helper=1;helper<=2;helper++) {
        CHECK(nvm_owned_array_origin_summary(p,helper,&summary));CHECK(summary.reachable && summary.conditional && summary.formal_count==2 && summary.result_count==2);
        for(unsigned n=0;n<2;n++) {
            NvmOwnerOriginPath path;CHECK(nvm_owned_array_origin_result(p,helper,n,&path));
            CHECK(path.path.root_layout==2 && path.path.length==2 && path.path.fields[0]==n && path.path.fields[1]==1);
            CHECK(path.origins.words[0]==0 && path.origins.words[1]==(UINT64_C(1)<<n));
        }
    }
    CHECK(nvm_owned_array_origin_summary(p,4,&summary));CHECK(!summary.reachable && summary.conditional && summary.formal_count==2 && summary.required.words[1]==3);
    NvmOwnerOriginPath path;CHECK(nvm_owned_array_origin_input(p,4,1,&path));CHECK(path.ordinal==0 && path.path.root_layout==3 && path.path.length==1 && path.path.fields[0]==1);
    CHECK(nvm_owned_array_origin_result(p,5,0,&path));CHECK(path.path.root_layout==1 && path.origins.words[0]==4 && !path.origins.words[1]);
    NvmOwnerOriginSite site;CHECK(nvm_owned_array_origin_site(p,2,&site));CHECK(site.function==5 && site.opcode==OP_ARR_NEW);
    NvmOwnerOriginSite saved_site=site;CHECK(!nvm_owned_array_origin_site(p,3,&site) && !memcmp(&site,&saved_site,sizeof site));
    NvmOwnerOriginPath saved_path=path;CHECK(!nvm_owned_array_origin_result(p,5,1,&path) && !memcmp(&path,&saved_path,sizeof path));
    CHECK(!nvm_owned_array_origin_input(p,6,0,&path) && !memcmp(&path,&saved_path,sizeof path));
    NvmOwnerOriginSummary saved_summary=summary;CHECK(!nvm_owned_array_origin_summary(p,6,&summary) && !memcmp(&summary,&saved_summary,sizeof summary));
    NvmOwnerOriginCounts saved_counts=counts;CHECK(!nvm_owned_array_origin_counts(NULL,&counts) && !memcmp(&counts,&saved_counts,sizeof counts));
    NvmOwnerOriginObligation row={9,9,9,9,9},saved_row=row;CHECK(!nvm_owned_array_origin_obligation(p,counts.obligations,&row) && !memcmp(&row,&saved_row,sizeof row));
    nvm_owned_array_origins_free(p);
    uint8_t *code=malloc(m->code_size),*owned=malloc(m->ownership_size),*layouts=malloc(m->layout_size);CHECK(code && owned && layouts);
    memcpy(code,m->code,m->code_size);memcpy(owned,m->ownership_data,m->ownership_size);memcpy(layouts,m->layout_data,m->layout_size);
    unsigned failed=0;bool succeeded=false;
    for(long n=0;n<512;n++) {
        unsigned char sentinel;p=(void *)&sentinel;budget=n;NvmOwnerOriginResult r=nvm_analyze_owned_array_origins(m,&p);budget=-1;
        if(r.status==NVM_OWNER_ORIGIN_MEMORY) {CHECK(p==(void *)&sentinel);failed++;}
        else {if(r.status!=NVM_OWNER_ORIGIN_PROVED)fprintf(stderr,"fault%ld status%d %s\n",n,r.status,r.message);CHECK(r.status==NVM_OWNER_ORIGIN_PROVED);nvm_owned_array_origins_free(p);succeeded=true;}
        CHECK(!memcmp(code,m->code,m->code_size) && !memcmp(owned,m->ownership_data,m->ownership_size) && !memcmp(layouts,m->layout_data,m->layout_size));
        if(succeeded)break;
    }
    CHECK(failed>30 && succeeded);printf("I preserve outputs through %u allocation failures.\n",failed);
    p=expect(m,NVM_OWNER_ORIGIN_PROVED);nvm_module_free(m);CHECK(nvm_owned_array_origin_summary(p,4,&summary) && !summary.reachable);nvm_owned_array_origins_free(p);
    free(code);free(owned);free(layouts);
}
static void joins_aliases_and_loops(void) {
    const char *bodies[]={
        "PUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 3\nSTORE_LOCAL 0\nJMP joined\nother:\nARR_NEW 3\nSTORE_LOCAL 0\njoined:\n"
        "LOAD_LOCAL 0\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 2\nLOAD_LOCAL 2\nAGG_GET 0\nPUSH_F64 2.0\nARR_PUSH\nPOP\n"
        "OWN_UNPACK_LOCAL 2\nPUSH_I64 0\nPUSH_F64 4.0\nARR_SET\nPOP\nPUSH_I64 -1\nARR_GET\nPUSH_F64 0.0\nEQ\nPOP\nPUSH_I64 0\nRET\n",
        "ARR_NEW 3\nSTORE_LOCAL 0\nPUSH_BOOL 1\nSTORE_LOCAL 1\nloop:\nLOAD_LOCAL 1\nJMP_FALSE done\n"
        "ARR_NEW 3\nSTORE_LOCAL 0\nPUSH_BOOL 0\nSTORE_LOCAL 1\nJMP loop\ndone:\nLOAD_LOCAL 0\nARR_LEN\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_F64 2.0\nARR_LITERAL 3 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nLOAD_LOCAL 0\nOWN_PACK 3\nOWN_STORE_LOCAL 2\n"
        "PUSH_BOOL 0\nJMP_FALSE right\nOWN_MOVE_LOCAL 2\nJMP joined\nright:\nOWN_MOVE_LOCAL 2\njoined:\n"
        "OWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nPOP\nPOP\nPUSH_I64 0\nRET\n",
        "OWN_PACK 4\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPUSH_STR value\nSTORE_LOCAL 4\nLOAD_LOCAL 4\nDUP\nPOP\nPOP\nPUSH_I64 0\nRET\n"
    };
    for(unsigned n=0;n<sizeof bodies/sizeof bodies[0];n++) {
        Function f={bodies[n],0,5,T(TAG_INT),{T(TAG_ARRAY),T(TAG_BOOL),OWNER(3),OWNER(4),T(TAG_STRING)}};
        NvmModule *m=build(&f,1);NvmOwnedArrayOrigins *p=expect(m,NVM_OWNER_ORIGIN_PROVED);locations(m,p);
        NvmOwnerOriginSummary s;CHECK(nvm_owned_array_origin_summary(p,0,&s));
        CHECK(s.required.words[0]==(n<2?3:n==2?1:0));
        if(n==0) {
            NvmOwnerOriginCounts c;CHECK(nvm_owned_array_origin_counts(p,&c));unsigned reads=0;
            for(uint32_t i=0;i<c.obligations;i++) {NvmOwnerOriginObligation o;CHECK(nvm_owned_array_origin_obligation(p,i,&o));if(o.read_tags){CHECK(o.read_tags==((1u<<TAG_FLOAT)|(1u<<TAG_VOID)));reads++;}}
            CHECK(reads==1);
        }
        nvm_owned_array_origins_free(p);nvm_module_free(m);
    }
}
static void refusals(void) {
    struct {const char *body;NvmOwnerOriginStatus status;} cases[]={
        {"LOAD_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"PUSH_BOOL 1\nJMP_FALSE done\nARR_NEW 3\nSTORE_LOCAL 0\ndone:\nLOAD_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"ARR_NEW 3\nPUSH_I64 1\nARR_PUSH\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"ARR_NEW 1\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"PUSH_I64 1\nARR_LITERAL 3 1\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"ARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 1\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"ARR_NEW 3\nAGG_PACK 0 0 0 1\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"ARR_NEW 3\nDUP\nOWN_PACK 3\nAGG_GET 0\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"LOAD_LOCAL 99\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_INVALID},
        {"POP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_INVALID},
        {"PUSH_I64 0\nRET\nNOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"PUSH_I64 1\nPUSH_I64 2\nLT\nPOP\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"PUSH_I64 1\nJMP_TRUE done\ndone:\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED},
        {"ARR_NEW 3\nDUP\nOWN_PACK 3\nOWN_STORE_LOCAL 2\nLOAD_LOCAL 2\nOWN_MOVE_LOCAL 2\nPUSH_I64 0\nRET\n",NVM_OWNER_ORIGIN_UNRESOLVED}
    };
    for(unsigned n=0;n<sizeof cases/sizeof cases[0];n++) {
        Function f={cases[n].body,0,3,T(TAG_INT),{T(TAG_ARRAY),OWNER(1),OWNER(3)}};
        NvmModule *m=build(&f,1);expect(m,cases[n].status);nvm_module_free(m);
    }
    Function f[]={
        {"PUSH_I64 0\nRET\n",0,0,T(TAG_INT),{{0}}},
        {"OWN_MOVE_LOCAL 0\nCALL 1\nRET\n",1,1,OWNER(3),{OWNER(3)}}
    };
    NvmModule *m=build(f,2);expect(m,NVM_OWNER_ORIGIN_UNRESOLVED);nvm_module_free(m);
    f[1].body="OWN_MOVE_LOCAL 0\nRET\n";f[1].result=(Type)OWNER(1);
    m=build(f,2);expect(m,NVM_OWNER_ORIGIN_UNRESOLVED);nvm_module_free(m);
    f[1].result=(Type)OWNER(1);f[1].locals[0]=(Type)OWNER(1);
    m=build(f,2);m->ownership_data[20+12+4+8+1]=1;
    expect(m,NVM_OWNER_ORIGIN_UNRESOLVED);nvm_module_free(m);
    for(unsigned mode=0;mode<7;mode++) {
        m=build(f,1);
        if(mode==0)m->header.entry_point=1;
        if(mode==1)m->header.flags=0;
        if(mode==2)m->header.magic[0]=0;
        if(mode==3)m->header.format_version++;
        if(mode==4)m->header.flags|=NVM_FLAG_NEEDS_EXTERN;
        if(mode==5)m->service_size=1;
        if(mode==6)m->header.flags|=0x80;
        expect(m,mode==2||mode==3?NVM_OWNER_ORIGIN_INVALID:NVM_OWNER_ORIGIN_UNRESOLVED);
        m->service_size=0;nvm_module_free(m);
    }
    char body[20000];size_t used=0;
    for(unsigned n=0;n<65;n++)used+=(size_t)snprintf(body+used,sizeof body-used,"ARR_NEW 3\nPOP\n");
    snprintf(body+used,sizeof body-used,"PUSH_I64 0\nRET\n");f[0].body=body;
    m=build(f,1);expect(m,NVM_OWNER_ORIGIN_LIMIT);nvm_module_free(m);
    used=0;for(unsigned n=0;n<4096;n++)used+=(size_t)snprintf(body+used,sizeof body-used,"NOP\n");
    snprintf(body+used,sizeof body-used,"PUSH_I64 0\nRET\n");
    m=build(f,1);expect(m,NVM_OWNER_ORIGIN_LIMIT);nvm_module_free(m);
    unsigned char sentinel;NvmOwnedArrayOrigins *p=(void *)&sentinel;
    CHECK(nvm_analyze_owned_array_origins(NULL,&p).status==NVM_OWNER_ORIGIN_INVALID && p==(void *)&sentinel);
    nvm_owned_array_origins_free(NULL);
}
int main(void) {
    positive_and_faults();joins_aliases_and_loops();refusals();
    printf("%u owner ARRAY origin checks passed; no pending module execution\n",checks);return 0;
}
