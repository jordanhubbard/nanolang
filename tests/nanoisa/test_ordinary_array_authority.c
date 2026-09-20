#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../../src/nanoisa/ordinary_array_authority.h"
#include "../../src/nanoisa/ownership_contracts.h"
#include "../../src/nanoisa/isa.h"
#include "../../src/nanoisa/nvm_v2_sections.h"
static unsigned checks;
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"I failed line%u: %s\n",__LINE__,#x);exit(1);} } while(0)
#ifdef OAA_INSTRUMENT
static void *allocations[4096];static size_t live,calls,fail_at=SIZE_MAX;static bool persistent;
static void *oaa_test_calloc(size_t n,size_t size){
    size_t index=calls++;if(index==fail_at || (persistent&&index>fail_at))return NULL;
    void *p=calloc(n,size);if(p){CHECK(live<4096);allocations[live++]=p;}return p;
}
static void oaa_test_free(void *p){
    if(!p)return;
    size_t i=0;while(i<live&&allocations[i]!=p)i++;
    CHECK(i<live);allocations[i]=allocations[--live];free(p);
}
#define calloc oaa_test_calloc
#define free oaa_test_free
#include "../../src/nanoisa/nvm_v2_layouts.c"
#include "../../src/nanoisa/ownership_contracts.c"
#undef calloc
#undef free
#endif

typedef struct {NvmModule module;uint8_t layouts[1024],ownership[2048];size_t l,o,extension,type,binding;} Case;
static void u8(uint8_t *b,size_t *n,uint8_t v){b[(*n)++]=v;}
static void u16(uint8_t *b,size_t *n,uint16_t v){u8(b,n,(uint8_t)v);u8(b,n,(uint8_t)(v>>8));}
static void u32(uint8_t *b,size_t *n,uint32_t v){for(unsigned i=0;i<4;i++)u8(b,n,(uint8_t)(v>>(8*i)));}
static void patch(uint8_t *b,size_t at,uint32_t v){for(unsigned i=0;i<4;i++)b[at+i]=(uint8_t)(v>>(8*i));}
static void make_case(Case *c,uint8_t element,bool forward){
    memset(c,0,sizeof *c);unsigned layouts=forward?2:1;
    u32(c->layouts,&c->l,layouts);
    for(unsigned i=0;i<layouts;i++){
        u8(c->layouts,&c->l,NVM_V2_LAYOUT_STRUCT);u8(c->layouts,&c->l,0);u16(c->layouts,&c->l,1);u32(c->layouts,&c->l,UINT32_MAX);
        u8(c->layouts,&c->l,forward&&i==0?TAG_STRUCT:TAG_ARRAY);u8(c->layouts,&c->l,0);u16(c->layouts,&c->l,0);
        u32(c->layouts,&c->l,forward&&i==0?1:UINT32_MAX);u32(c->layouts,&c->l,UINT32_MAX);
    }
    u32(c->ownership,&c->o,3);u32(c->ownership,&c->o,layouts);
    for(unsigned i=0;i<layouts;i++)u8(c->ownership,&c->o,NVM_LAYOUT_COMPLETE);
    while(c->o%4)u8(c->ownership,&c->o,0);
    u32(c->ownership,&c->o,0); /* functions */
    u32(c->ownership,&c->o,4);u32(c->ownership,&c->o,0); /* paths */
    u32(c->ownership,&c->o,1);c->extension=c->o;
    u16(c->ownership,&c->o,2);u16(c->ownership,&c->o,1);u32(c->ownership,&c->o,28);
    u32(c->ownership,&c->o,1);c->type=c->o;
    u8(c->ownership,&c->o,element);u8(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u32(c->ownership,&c->o,UINT32_MAX);
    u32(c->ownership,&c->o,1);c->binding=c->o;
    u32(c->ownership,&c->o,layouts-1);u16(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u32(c->ownership,&c->o,0);
    c->module.layout_data=c->layouts;c->module.layout_size=(uint32_t)c->l;
    c->module.ownership_data=c->ownership;c->module.ownership_size=(uint32_t)c->o;c->module.struct_count=layouts;
}
static void chain(Case *c,unsigned count,bool reverse,bool cycle){
    make_case(c,TAG_INT,false);c->o=c->extension+8;
    patch(c->ownership,c->extension+4,20u+8u*count);
    u32(c->ownership,&c->o,count);c->type=c->o;
    for(unsigned i=0;i<count;i++){
        bool terminal=reverse?i==0:i+1==count;
        u8(c->ownership,&c->o,terminal&&!cycle?TAG_INT:TAG_ARRAY);u8(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);
        u32(c->ownership,&c->o,terminal?(cycle?(reverse?count-1:0):UINT32_MAX):(reverse?i-1:i+1));
    }
    u32(c->ownership,&c->o,1);c->binding=c->o;
    u32(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u32(c->ownership,&c->o,reverse?count-1:0);
    c->module.ownership_size=(uint32_t)c->o;
}
static void refusal(Case *c,NvmOrdinaryArrayStatus expected){
    NvmOrdinaryArrayAuthority *p=(void *)(uintptr_t)1;
    NvmOrdinaryArrayResult r=nvm_describe_ordinary_array_authority(&c->module,&p);
    CHECK(r.status==expected);CHECK(p==(void *)(uintptr_t)1);
}
static void described(Case *c){
    NvmOrdinaryArrayAuthority *p=NULL;CHECK(nvm_describe_ordinary_array_authority(&c->module,&p).status==NVM_OAA_DESCRIBED);CHECK(p);
    NvmOrdinaryArrayCounts counts;CHECK(nvm_ordinary_array_authority_counts(p,&counts));CHECK(counts.layouts==c->module.struct_count&&counts.types==1&&counts.bindings==1);
    NvmOrdinaryArrayType t;CHECK(nvm_ordinary_array_authority_type(p,0,&t));CHECK(t.tag==c->ownership[c->type]&&t.referent==UINT32_MAX);
    NvmOrdinaryArrayBinding b;CHECK(nvm_ordinary_array_authority_binding(p,0,&b));CHECK(b.layout==counts.layouts-1&&!b.field&&!b.element_type);
    memset(&b,0xa5,sizeof b);NvmOrdinaryArrayBinding old=b;
    CHECK(!nvm_ordinary_array_authority_binding(p,1,&b)&&!memcmp(&b,&old,sizeof b));
    CHECK(!nvm_ordinary_array_authority_binding(NULL,0,&b));CHECK(!nvm_ordinary_array_authority_type(p,0,NULL));CHECK(!nvm_ordinary_array_authority_counts(p,NULL));
    bool needs=true;CHECK(nvm_ownership_contracts_validate(&c->module,&needs)!=NVM_V2_OK);CHECK(!needs);
    memset(c->layouts,0,sizeof c->layouts);memset(c->ownership,0,sizeof c->ownership);
    CHECK(nvm_ordinary_array_authority_type(p,0,&t));CHECK(nvm_ordinary_array_authority_binding(p,0,&b));
    nvm_ordinary_array_authority_free(p);
}
int main(void){
    Case c;const uint8_t tags[]={TAG_INT,TAG_U8,TAG_FLOAT,TAG_BOOL,TAG_STRING};
    for(unsigned i=0;i<5;i++)for(unsigned forward=0;forward<2;forward++){make_case(&c,tags[i],forward);described(&c);}
    make_case(&c,TAG_INT,false);c.module.service_data=(void *)(uintptr_t)1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.module.passive_data=(void *)(uintptr_t)1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.module.import_count=1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.module.module_ref_count=1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.ownership[c.type+1]=1;refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);patch(c.ownership,c.type+4,0);refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_ARRAY,false);patch(c.ownership,c.type+4,0);refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_STRUCT,false);patch(c.ownership,c.type+4,0);refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.ownership[c.binding+6]=1;refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);patch(c.ownership,c.binding+8,1);refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);patch(c.ownership,c.binding,1);refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);c.ownership[c.binding+4]=1;refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);c.ownership[c.extension]=3;refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);c.ownership[c.extension+2]=2;refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);patch(c.ownership,c.type-4,4097);refusal(&c,NVM_OAA_LIMIT);
    make_case(&c,TAG_INT,false);patch(c.layouts,0,257);refusal(&c,NVM_OAA_LIMIT);
    make_case(&c,TAG_INT,false);c.module.layout_size=16777217;refusal(&c,NVM_OAA_LIMIT);
    make_case(&c,TAG_INT,false);c.module.ownership_size++;refusal(&c,NVM_OAA_INVALID);
    for(uint32_t n=1;n<c.o;n++){make_case(&c,TAG_INT,false);c.module.ownership_size=n;refusal(&c,NVM_OAA_INVALID);}
    for(unsigned reverse=0;reverse<2;reverse++){
        chain(&c,2,reverse,false);refusal(&c,NVM_OAA_UNKNOWN);
        chain(&c,64,reverse,false);refusal(&c,NVM_OAA_UNKNOWN);
        chain(&c,65,reverse,false);refusal(&c,NVM_OAA_LIMIT);
        chain(&c,4,reverse,true);refusal(&c,NVM_OAA_INVALID);
    }
    for(unsigned version=1;version<=2;version++){
        make_case(&c,TAG_INT,false);c.layouts[12]=TAG_INT;patch(c.ownership,0,version);
        c.module.ownership_size=version==1?16:20;patch(c.ownership,16,0);
        NvmOrdinaryArrayAuthority *p=NULL;bool needs=true;
        CHECK(nvm_ownership_contracts_validate(&c.module,&needs)==NVM_V2_OK&&!needs);
        CHECK(nvm_describe_ordinary_array_authority(&c.module,&p).status==NVM_OAA_DESCRIBED);
        NvmOrdinaryArrayCounts counts;CHECK(nvm_ordinary_array_authority_counts(p,&counts));CHECK(!counts.types&&!counts.bindings);nvm_ordinary_array_authority_free(p);
    }
    NvmOrdinaryArrayAuthority *sentinel=(void *)(uintptr_t)1;
    CHECK(nvm_describe_ordinary_array_authority(NULL,&sentinel).status==NVM_OAA_INVALID);CHECK(sentinel==(void *)(uintptr_t)1);
    CHECK(nvm_describe_ordinary_array_authority(&c.module,NULL).status==NVM_OAA_INVALID);nvm_ordinary_array_authority_free(NULL);
#ifdef OAA_INSTRUMENT
    make_case(&c,TAG_INT,true);calls=0;NvmOrdinaryArrayAuthority *p=NULL;
    CHECK(nvm_describe_ordinary_array_authority(&c.module,&p).status==NVM_OAA_DESCRIBED);size_t measured=calls;nvm_ordinary_array_authority_free(p);CHECK(!live&&measured>=7);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<measured;i++){
        make_case(&c,TAG_INT,true);calls=0;fail_at=i;p=(void *)(uintptr_t)1;persistent=mode!=0;
        NvmOrdinaryArrayResult r=nvm_describe_ordinary_array_authority(&c.module,&p);
        CHECK(r.status==NVM_OAA_MEMORY||r.status==NVM_OAA_UNKNOWN);CHECK(p==(void *)(uintptr_t)1);CHECK(!live);
        fail_at=SIZE_MAX;persistent=false;calls=0;described(&c);CHECK(!live);
    }
    printf("I covered %zu allocation positions in both failure modes\n",measured);
#endif
    printf("PASS %u ordinary array declaration checks; no execution authority\n",checks);return 0;
}
