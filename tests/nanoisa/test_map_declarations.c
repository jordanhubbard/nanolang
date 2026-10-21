/* I exercise declarations independently of executable map admission. */
#include <stdint.h>
#ifdef OAA_INSTRUMENT
static uint64_t map_test_bytes=2u*1024u*1024u,map_test_work=4u*1024u*1024u;
#define NVM_MAP_DECL_BYTES map_test_bytes
#define NVM_MAP_DECL_WORK map_test_work
#endif
#define main old_array_controls_main
#include "test_ordinary_array_authority.c"
#undef main
#include "../../src/nanoisa/ownership_map_declarations.h"
#include "../../src/nanoisa/ownership_declaration_projection.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanoisa/nvm2c.h"
static void map_make(Case *c,bool forward) {
    make_case(c,TAG_INT,forward);
    c->layouts[forward?32:12]=TAG_HASHMAP;
    c->o=c->extension;
    u16(c->ownership,&c->o,5);u16(c->ownership,&c->o,1);u32(c->ownership,&c->o,48);
    u32(c->ownership,&c->o,2);c->type=c->o;
    const uint8_t tags[]={TAG_INT,TAG_STRING};
    for(unsigned i=0;i<2;i++){u8(c->ownership,&c->o,tags[i]);u8(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u32(c->ownership,&c->o,UINT32_MAX);}
    u32(c->ownership,&c->o,1);u32(c->ownership,&c->o,1);u32(c->ownership,&c->o,0);
    u32(c->ownership,&c->o,1);c->binding=c->o;
    u32(c->ownership,&c->o,forward?1:0);u16(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u32(c->ownership,&c->o,0);
    c->module.ownership_size=(uint32_t)c->o;
}
static void map_refuse(Case *c,NvmMapDeclarationStatus status) {
    NvmMapDeclarationPlan *p=(void *)(uintptr_t)1;
    NvmMapDeclarationResult r=nvm_prepare_map_declarations(&c->module,&p);
    if(r.status!=status)fprintf(stderr,"I expected map status%u got%u: %s\n",status,r.status,r.message);
    CHECK(r.status==status);CHECK(p==(void *)(uintptr_t)1);
}
static void map_facts(bool forward) {
    Case c;map_make(&c,forward);NvmMapDeclarationPlan *p=NULL;
    CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_DESCRIBED);
    NvmMapDeclarationCounts n;CHECK(nvm_map_declarations_counts(p,&n));CHECK(n.scalars==2&&n.pairs==1&&n.bindings==1);
    uint8_t tag=0;CHECK(nvm_map_declarations_scalar(p,0,&tag)&&tag==TAG_INT);
    CHECK(nvm_map_declarations_scalar(p,1,&tag)&&tag==TAG_STRING);
    NvmMapDeclarationPair pair;CHECK(nvm_map_declarations_pair(p,0,&pair));CHECK(pair.key_type==1&&pair.value_type==0);
    NvmMapDeclarationBinding b;CHECK(nvm_map_declarations_binding(p,0,&b));CHECK(b.layout==(forward?1u:0u)&&b.field==0&&b.pair==0);
    NvmMapDeclarationPair saved=pair;CHECK(nvm_map_declarations_field(p,b.layout,0,&pair));CHECK(!memcmp(&saved,&pair,sizeof pair));
    CHECK(!nvm_map_declarations_field(p,b.layout,1,&pair));CHECK(!memcmp(&saved,&pair,sizeof pair));
    CHECK(!nvm_map_declarations_pair(p,1,&pair));CHECK(!memcmp(&saved,&pair,sizeof pair));
    CHECK(!nvm_map_declarations_scalar(p,2,&tag)&&tag==TAG_STRING);
    bool needs=true;CHECK(nvm_ownership_contracts_validate(&c.module,&needs)!=NVM_V2_OK);
    NvmOwnershipDeclarationPlan *base=(void *)(uintptr_t)1;
    CHECK(nvm_prepare_ownership_declarations(&c.module,&base).status!=NVM_DECL_PREPARED);CHECK(base==(void *)(uintptr_t)1);
    memset(c.layouts,0,sizeof c.layouts);memset(c.ownership,0,sizeof c.ownership);
    CHECK(nvm_map_declarations_field(p,b.layout,0,&pair));CHECK(pair.key_type==1&&pair.value_type==0);
    nvm_map_declarations_free(p);
}
static void map_bad(void) {
    Case c;map_make(&c,false);
    size_t offsets[]={c.extension+2,c.type+1,c.type+2,c.type+4,c.type+8,c.binding,c.binding+4,c.binding+6,c.binding+8};
    for(size_t i=0;i<sizeof offsets/sizeof *offsets;i++) {
        map_make(&c,false);c.ownership[offsets[i]]^=128;map_refuse(&c,NVM_MAP_INVALID);
    }
    map_make(&c,false);patch(c.ownership,c.extension+8,6);map_refuse(&c,NVM_MAP_LIMIT);
    map_make(&c,false);patch(c.ownership,c.type+16,26);map_refuse(&c,NVM_MAP_LIMIT);
    map_make(&c,false);patch(c.ownership,c.binding-4,65537);map_refuse(&c,NVM_MAP_LIMIT);
    map_make(&c,false);size_t size=c.o;
    for(size_t i=0;i<size;i++){c.module.ownership_size=(uint32_t)i;map_refuse(&c,NVM_MAP_INVALID);}
    map_make(&c,false);c.ownership[8]=0;map_refuse(&c,NVM_MAP_INVALID);
    map_make(&c,false);c.layouts[12]=TAG_INT;map_refuse(&c,NVM_MAP_INVALID);
    map_make(&c,false);c.ownership[c.extension]=4;map_refuse(&c,NVM_MAP_INVALID);
    NvmModule empty={0};NvmMapDeclarationPlan *p=(void *)(uintptr_t)1;
    CHECK(nvm_prepare_map_declarations(&empty,&p).status==NVM_MAP_ABSENT);CHECK(p==(void *)(uintptr_t)1);
    make_case(&c,TAG_INT,false);CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_ABSENT);CHECK(p==(void *)(uintptr_t)1);
    c.ownership[c.extension]=99;map_refuse(&c,NVM_MAP_INVALID);
    /* A structurally exact unsupported pair is a declaration, never execution. */
    map_make(&c,false);patch(c.ownership,c.type+20,0);patch(c.ownership,c.type+24,1);
    p=NULL;CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_DESCRIBED);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(&c.module,&needs)!=NVM_V2_OK);nvm_map_declarations_free(p);
}
static void map_mixed_and_duplicates(void) {
    Case c;map_make(&c,false);
    c.layouts[6]=2;
    u8(c.layouts,&c.l,TAG_ARRAY);u8(c.layouts,&c.l,0);u16(c.layouts,&c.l,0);u32(c.layouts,&c.l,UINT32_MAX);u32(c.layouts,&c.l,UINT32_MAX);
    memmove(c.ownership+c.extension+36,c.ownership+c.extension,c.o-c.extension);
    size_t end=c.o+36;c.o=c.extension;
    u16(c.ownership,&c.o,2);u16(c.ownership,&c.o,1);u32(c.ownership,&c.o,28);
    u32(c.ownership,&c.o,1);u8(c.ownership,&c.o,TAG_INT);u8(c.ownership,&c.o,0);u16(c.ownership,&c.o,0);u32(c.ownership,&c.o,UINT32_MAX);
    u32(c.ownership,&c.o,1);u32(c.ownership,&c.o,0);u16(c.ownership,&c.o,1);u16(c.ownership,&c.o,0);u32(c.ownership,&c.o,0);
    patch(c.ownership,c.extension-4,2);c.o=end;c.module.ownership_size=(uint32_t)end;c.module.layout_size=(uint32_t)c.l;
    NvmMapDeclarationPlan *p=NULL;CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_DESCRIBED);
    nvm_map_declarations_free(p);
    c.ownership[c.extension+36]=2;map_refuse(&c,NVM_MAP_INVALID); /* duplicate extension */
    map_make(&c,false);patch(c.ownership,c.type+24,1);map_refuse(&c,NVM_MAP_INVALID); /* unused INT scalar */
    map_make(&c,false);patch(c.ownership,c.type+20,2);map_refuse(&c,NVM_MAP_INVALID); /* wrong pair reference */
    map_make(&c,false);memcpy(c.ownership+c.o,c.ownership+c.binding,12);c.o+=12;
    patch(c.ownership,c.binding-4,2);patch(c.ownership,c.extension+4,60);c.module.ownership_size=(uint32_t)c.o;
    map_refuse(&c,NVM_MAP_INVALID); /* duplicate field even with identical pair */
    map_make(&c,false);size_t at=c.type+28;
    memmove(c.ownership+at+8,c.ownership+at,c.o-at);memcpy(c.ownership+at,c.ownership+c.type+20,8);c.o+=8;
    patch(c.ownership,c.type+16,2);patch(c.ownership,c.extension+4,56);c.module.ownership_size=(uint32_t)c.o;
    map_refuse(&c,NVM_MAP_INVALID); /* duplicate pair */
    map_make(&c,false);patch(c.ownership,c.extension-4,5);map_refuse(&c,NVM_MAP_LIMIT);
#ifdef OAA_INSTRUMENT
    /* Four framed kinds fit the policy, not necessarily complete semantics. */
    map_make(&c,false);memmove(c.ownership+c.extension+24,c.ownership+c.extension,c.o-c.extension);
    for(unsigned i=0;i<3;i++){size_t at=c.extension+i*8;patch(c.ownership,at,1+i);c.ownership[at+2]=1;patch(c.ownership,at+4,0);}
    c.o+=24;c.module.ownership_size=(uint32_t)c.o;patch(c.ownership,c.extension-4,4);
    uint64_t work=0;CHECK(map_route(&c.module,&work)==MAP_ROUTE_PRESENT);
    map_refuse(&c,NVM_MAP_INVALID);
#endif
}
static void map_signature_boundaries(void) {
    for(unsigned mode=0;mode<3;mode++) {
        Case c;map_make(&c,false);NvmFunctionEntry fn={0};uint8_t param=TAG_HASHMAP,*params[]={&param};
        fn.local_count=1;fn.arity=mode==1?1:0;fn.result_count=1;fn.result_tag=mode==2?TAG_HASHMAP:TAG_INT;
        c.module.functions=&fn;c.module.function_count=1;c.module.function_param_types=params;
        memmove(c.ownership+36,c.ownership+16,c.o-16);c.o+=20;patch(c.ownership,12,1);
        memset(c.ownership+16,0,20);c.ownership[16]=1;c.ownership[18]=(uint8_t)fn.arity;
        c.ownership[20]=fn.result_tag;patch(c.ownership,24,UINT32_MAX);
        c.ownership[28]=TAG_HASHMAP;patch(c.ownership,32,UINT32_MAX);c.module.ownership_size=(uint32_t)c.o;
        if(mode)map_refuse(&c,NVM_MAP_INVALID);
        else { /* A local coarse tag is not a parameter/result pair claim. */
            NvmMapDeclarationPlan *p=NULL;CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_DESCRIBED);
            nvm_map_declarations_free(p);
        }
    }
    Case c;map_make(&c,false);
    memmove(c.ownership+c.extension+20,c.ownership+c.extension,c.o-c.extension);size_t end=c.o+20;c.o=c.extension;
    u16(c.ownership,&c.o,3);u16(c.ownership,&c.o,1);u32(c.ownership,&c.o,12);
    u32(c.ownership,&c.o,1);u8(c.ownership,&c.o,TAG_HASHMAP);u8(c.ownership,&c.o,0);u16(c.ownership,&c.o,0);u32(c.ownership,&c.o,UINT32_MAX);
    patch(c.ownership,c.extension-4,2);c.module.ownership_size=(uint32_t)end;
    map_refuse(&c,NVM_MAP_INVALID);
}
static void map_public_refusal(void) {
    AsmResult error;NvmModule *m=asm_assemble(".entry 0\n.function main 0 0 0 int 1\nHM_NEW 5 1\nPOP\nPUSH_I64 0\nRET\n.end\n",&error);
    CHECK(m);CHECK(nvm_verify(m).ok);
    char diagnostic[256];char *native=nvm2c_emit(m,diagnostic,sizeof diagnostic);CHECK(native);free(native);
    Case c;map_make(&c,false);
    memmove(c.ownership+28,c.ownership+16,c.o-16);c.o+=12;
    patch(c.ownership,12,1);memset(c.ownership+16,0,12);c.ownership[20]=TAG_INT;patch(c.ownership,24,UINT32_MAX);
    m->layout_data=malloc(c.l);m->ownership_data=malloc(c.o);CHECK(m->layout_data&&m->ownership_data);
    memcpy(m->layout_data,c.layouts,c.l);memcpy(m->ownership_data,c.ownership,c.o);
    m->layout_size=(uint32_t)c.l;m->ownership_size=(uint32_t)c.o;m->struct_count=1;
    NvmMapDeclarationPlan *p=NULL;CHECK(nvm_prepare_map_declarations(m,&p).status==NVM_MAP_DESCRIBED);nvm_map_declarations_free(p);
    CHECK(!nvm_verify(m).ok);CHECK(nvm2c_emit(m,diagnostic,sizeof diagnostic)==NULL);
    NvmV2Module encoded;CHECK(nvm_v2_from_nvm_module(m,&encoded)!=NVM_V2_OK);
    m->ownership_data[28+12]=99; /* unknown extension kind; no ARRAY shortcut */
    p=(void *)(uintptr_t)1;CHECK(nvm_prepare_map_declarations(m,&p).status==NVM_MAP_INVALID);CHECK(p==(void *)(uintptr_t)1);
    CHECK(!nvm_verify(m).ok);CHECK(nvm2c_emit(m,diagnostic,sizeof diagnostic)==NULL);
    nvm_module_free(m);
}
static void map_maximum(void) {
    size_t lc=4+256*8+65536*12,oc=2048+65536*12,l=0,o=0;
    uint8_t *lb=calloc(lc,1),*ob=calloc(oc,1);CHECK(lb&&ob);
    u32(lb,&l,256);
    for(unsigned i=0;i<256;i++) {
        u8(lb,&l,NVM_V2_LAYOUT_STRUCT);u8(lb,&l,0);u16(lb,&l,(uint16_t)(i==0?65535:i==1?1:0));u32(lb,&l,UINT32_MAX);
        for(unsigned j=0;j<(i==0?65535u:i==1?1u:0u);j++){u8(lb,&l,TAG_HASHMAP);u8(lb,&l,0);u16(lb,&l,0);u32(lb,&l,UINT32_MAX);u32(lb,&l,UINT32_MAX);}
    }
    u32(ob,&o,3);u32(ob,&o,256);for(unsigned i=0;i<256;i++)u8(ob,&o,NVM_LAYOUT_COMPLETE);
    u32(ob,&o,0);u32(ob,&o,4);u32(ob,&o,0);u32(ob,&o,1);
    u16(ob,&o,5);u16(ob,&o,1);u32(ob,&o,786684);u32(ob,&o,5);
    for(uint8_t i=TAG_INT;i<=TAG_STRING;i++){u8(ob,&o,i);u8(ob,&o,0);u16(ob,&o,0);u32(ob,&o,UINT32_MAX);}
    u32(ob,&o,25);for(unsigned i=0;i<5;i++)for(unsigned j=0;j<5;j++){u32(ob,&o,i);u32(ob,&o,j);}
    u32(ob,&o,65536);
    for(unsigned i=0;i<2;i++)for(unsigned j=0;j<(i?1u:65535u);j++){u32(ob,&o,i);u16(ob,&o,(uint16_t)j);u16(ob,&o,0);u32(ob,&o,(i*65535+j)%25);}
    CHECK(l==lc&&o<=oc);
    NvmModule m={0};m.layout_data=lb;m.layout_size=(uint32_t)l;m.ownership_data=ob;m.ownership_size=(uint32_t)o;m.struct_count=256;
    NvmMapDeclarationPlan *p=NULL;CHECK(nvm_prepare_map_declarations(&m,&p).status==NVM_MAP_DESCRIBED);
    NvmMapDeclarationCounts n;CHECK(nvm_map_declarations_counts(p,&n)&&n.bindings==65536&&n.pairs==25&&n.scalars==5);
    for(unsigned i=0;i<2;i++)for(unsigned j=0;j<(i?1u:65535u);j++) {
        NvmMapDeclarationPair pair;CHECK(nvm_map_declarations_field(p,i,(uint16_t)j,&pair));
        unsigned index=(i*65535+j)%25;CHECK(pair.key_type==index/5&&pair.value_type==index%5);
    }
    NvmMapDeclarationPair old={99,99},pair=old;
    CHECK(!nvm_map_declarations_field(p,0,65535,&pair)&&!memcmp(&pair,&old,sizeof pair));
#ifdef OAA_INSTRUMENT
    printf("I retained map maximum bytes%llu work%llu scratch%zu.\n",(unsigned long long)p->bytes,(unsigned long long)p->work,sizeof(MapDeclarationScratch));
#endif
    nvm_map_declarations_free(p);free(lb);free(ob);
}
#ifdef OAA_INSTRUMENT
static void map_allocations(void) {
    Case c;map_make(&c,true);calls=0;NvmMapDeclarationPlan *p=NULL;
    CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_DESCRIBED);
    size_t count=calls;uint64_t bytes=p->bytes,work=p->work;
    nvm_map_declarations_free(p);CHECK(live==0);
    map_test_bytes=bytes;map_test_work=work;p=NULL;
    CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_DESCRIBED);
    nvm_map_declarations_free(p);CHECK(live==0);
    map_test_bytes=bytes-1;map_refuse(&c,NVM_MAP_LIMIT);CHECK(live==0);
    map_test_bytes=bytes;map_test_work=work-1;map_refuse(&c,NVM_MAP_LIMIT);CHECK(live==0);
    map_test_bytes=2u*1024u*1024u;map_test_work=4u*1024u*1024u;
    printf("I retained exact map byte/work boundaries %llu/%llu and one-below refusal.\n",(unsigned long long)bytes,(unsigned long long)work);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<count;i++) {
        calls=0;fail_at=i;persistent=mode!=0;
        map_refuse(&c,NVM_MAP_MEMORY);CHECK(live==0);
        fail_at=SIZE_MAX;persistent=false;p=NULL;
        CHECK(nvm_prepare_map_declarations(&c.module,&p).status==NVM_MAP_DESCRIBED);
        nvm_map_declarations_free(p);CHECK(live==0);
    }
    printf("I retained %zu map allocation positions and independent retries.\n",count);
}
#endif
int main(void) {
    CHECK(old_array_controls_main()==0);map_facts(false);map_facts(true);map_bad();map_mixed_and_duplicates();map_signature_boundaries();map_public_refusal();map_maximum();
#ifdef OAA_INSTRUMENT
    map_allocations();
#endif
    printf("I passed %u exact map declaration checks; public admission stays refused.\n",checks);return 0;
}
