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
    NvmOrdinaryArrayType saved_type=t;NvmOrdinaryArrayBinding saved_binding=b;
    memset(&b,0xa5,sizeof b);NvmOrdinaryArrayBinding old=b;
    CHECK(!nvm_ordinary_array_authority_binding(p,1,&b)&&!memcmp(&b,&old,sizeof b));
    memset(&t,0xa5,sizeof t);NvmOrdinaryArrayType old_type=t;
    CHECK(!nvm_ordinary_array_authority_type(p,1,&t)&&!memcmp(&t,&old_type,sizeof t));
    NvmOrdinaryArrayCounts old_counts=counts;
    CHECK(!nvm_ordinary_array_authority_counts(NULL,&counts)&&!memcmp(&counts,&old_counts,sizeof counts));
    CHECK(!nvm_ordinary_array_authority_binding(NULL,0,&b));CHECK(!nvm_ordinary_array_authority_type(p,0,NULL));CHECK(!nvm_ordinary_array_authority_counts(p,NULL));
    bool needs=true;CHECK(nvm_ownership_contracts_validate(&c->module,&needs)!=NVM_V2_OK);CHECK(!needs);
    memset(c->layouts,0,sizeof c->layouts);memset(c->ownership,0,sizeof c->ownership);
    CHECK(nvm_ordinary_array_authority_type(p,0,&t));CHECK(t.tag==saved_type.tag&&t.referent==saved_type.referent);
    CHECK(nvm_ordinary_array_authority_binding(p,0,&b));CHECK(b.layout==saved_binding.layout&&b.field==saved_binding.field&&b.element_type==saved_binding.element_type);
    CHECK(nvm_ordinary_array_authority_counts(p,&counts));CHECK(counts.layouts==old_counts.layouts&&counts.types==old_counts.types&&counts.bindings==old_counts.bindings);
    nvm_ordinary_array_authority_free(p);
}
static void mixed_union(void){
    Case c;make_case(&c,TAG_INT,false);
    patch(c.layouts,0,2);c.l=c.module.layout_size;
    u8(c.layouts,&c.l,NVM_V2_LAYOUT_UNION);u8(c.layouts,&c.l,0);u16(c.layouts,&c.l,1);u32(c.layouts,&c.l,0);
    u8(c.layouts,&c.l,TAG_INT);u8(c.layouts,&c.l,0);u16(c.layouts,&c.l,0);u32(c.layouts,&c.l,UINT32_MAX);u32(c.layouts,&c.l,0);
    c.module.layout_size=(uint32_t)c.l;c.module.union_count=1;c.module.string_count=2;
    patch(c.ownership,4,2);c.ownership[9]=0;patch(c.ownership,24,2);
    uint8_t array[36];memcpy(array,c.ownership+c.extension,36);c.o=c.extension;
    u16(c.ownership,&c.o,1);u16(c.ownership,&c.o,1);u32(c.ownership,&c.o,20);
    u32(c.ownership,&c.o,1);u32(c.ownership,&c.o,1);u16(c.ownership,&c.o,1);u16(c.ownership,&c.o,0);
    size_t variant=c.o;u32(c.ownership,&c.o,1);u16(c.ownership,&c.o,0);u16(c.ownership,&c.o,1);
    memcpy(c.ownership+c.o,array,36);c.o+=36;c.module.ownership_size=(uint32_t)c.o;
    refusal(&c,NVM_OAA_UNKNOWN);
    c.ownership[variant+6]=2;refusal(&c,NVM_OAA_INVALID);c.ownership[variant+6]=1;
    c.ownership[c.o-28+1]=1;refusal(&c,NVM_OAA_INVALID);c.ownership[c.o-28+1]=0;
    c.layouts[12]=TAG_INT;patch(c.ownership,24,1);c.module.ownership_size=(uint32_t)(c.extension+28);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(&c.module,&needs)==NVM_V2_OK&&needs);refusal(&c,NVM_OAA_UNKNOWN);
}
static void binding_order(void){
    Case c;make_case(&c,TAG_INT,false);c.layouts[6]=2;
    memcpy(c.layouts+c.l,c.layouts+12,12);c.l+=12;c.module.layout_size=(uint32_t)c.l;
    refusal(&c,NVM_OAA_INVALID); /* second ARRAY has no binding */
    patch(c.ownership,c.binding-4,2);patch(c.ownership,c.extension+4,40);
    memcpy(c.ownership+c.o,c.ownership+c.binding,12);c.ownership[c.o+4]=1;c.o+=12;c.module.ownership_size=(uint32_t)c.o;
    NvmOrdinaryArrayAuthority *p=NULL;CHECK(nvm_describe_ordinary_array_authority(&c.module,&p).status==NVM_OAA_DESCRIBED);
    NvmOrdinaryArrayBinding b;CHECK(nvm_ordinary_array_authority_binding(p,1,&b)&&b.layout==0&&b.field==1&&b.element_type==0);nvm_ordinary_array_authority_free(p);
    c.ownership[c.binding+12+4]=0;refusal(&c,NVM_OAA_INVALID); /* duplicate */
    c.ownership[c.binding+4]=1;refusal(&c,NVM_OAA_INVALID); /* reversed */
}
static void nominal_identity(void){
    Case c;make_case(&c,TAG_INT,true);c.layouts[12]=TAG_ARRAY;patch(c.layouts,16,UINT32_MAX);
    refusal(&c,NVM_OAA_INVALID); /* equal-shaped layout0 is still unbound */
    patch(c.ownership,c.binding-4,2);patch(c.ownership,c.extension+4,40);
    memcpy(c.ownership+c.o,c.ownership+c.binding,12);patch(c.ownership,c.binding,0);c.o+=12;c.module.ownership_size=(uint32_t)c.o;
    NvmOrdinaryArrayAuthority *p=NULL;CHECK(nvm_describe_ordinary_array_authority(&c.module,&p).status==NVM_OAA_DESCRIBED);
    NvmOrdinaryArrayBinding a,b;CHECK(nvm_ordinary_array_authority_binding(p,0,&a));CHECK(nvm_ordinary_array_authority_binding(p,1,&b));CHECK(a.layout==0&&b.layout==1&&!a.field&&!b.field);nvm_ordinary_array_authority_free(p);
    patch(c.ownership,c.binding+12,0);refusal(&c,NVM_OAA_INVALID);
}
static void maximum_tables(void){
    const uint32_t layouts=256,fields=65536,types=4096;
    size_t layout_bytes=4u+8u*layouts+12u*fields;
    size_t ownership_bytes=8u+layouts+4u+8u+4u+8u+4u+8u*types+4u+12u*fields;
    uint8_t *l=calloc(layout_bytes+12,1),*o=calloc(ownership_bytes,1);CHECK(l&&o);
    size_t n=0;u32(l,&n,layouts);
    for(uint32_t i=0;i<layouts;i++){
        unsigned count=i==0?65535:i==1?1:0;
        u8(l,&n,NVM_V2_LAYOUT_STRUCT);u8(l,&n,0);u16(l,&n,(uint16_t)count);u32(l,&n,UINT32_MAX);
        for(unsigned j=0;j<count;j++){u8(l,&n,TAG_ARRAY);u8(l,&n,0);u16(l,&n,0);u32(l,&n,UINT32_MAX);u32(l,&n,UINT32_MAX);}
    }
    CHECK(n==layout_bytes);n=0;u32(o,&n,3);u32(o,&n,layouts);
    for(unsigned i=0;i<layouts;i++)u8(o,&n,NVM_LAYOUT_COMPLETE);
    u32(o,&n,0);u32(o,&n,4);u32(o,&n,0);u32(o,&n,1);
    u16(o,&n,2);u16(o,&n,1);u32(o,&n,8u+8u*types+12u*fields);
    size_t type_count=n;u32(o,&n,types);
    for(unsigned i=0;i<types;i++){u8(o,&n,TAG_INT);u8(o,&n,0);u16(o,&n,0);u32(o,&n,UINT32_MAX);}
    size_t binding_count=n;u32(o,&n,fields);
    for(unsigned i=0;i<fields;i++){u32(o,&n,i==65535?1:0);u16(o,&n,i==65535?0:(uint16_t)i);u16(o,&n,0);u32(o,&n,types-1);}
    CHECK(n==ownership_bytes);
    NvmModule m={0};m.layout_data=l;m.layout_size=(uint32_t)layout_bytes;m.ownership_data=o;m.ownership_size=(uint32_t)ownership_bytes;m.struct_count=layouts;
    NvmOrdinaryArrayAuthority *p=NULL;CHECK(nvm_describe_ordinary_array_authority(&m,&p).status==NVM_OAA_DESCRIBED);
    NvmOrdinaryArrayCounts counts;CHECK(nvm_ordinary_array_authority_counts(p,&counts));CHECK(counts.layouts==layouts&&counts.types==types&&counts.bindings==fields);
    NvmOrdinaryArrayBinding b;CHECK(nvm_ordinary_array_authority_binding(p,fields-1,&b)&&b.layout==1&&!b.field&&b.element_type==types-1);
    NvmOrdinaryArrayAuthority *sentinel=(void *)(uintptr_t)1;
    patch(o,type_count,types+1);CHECK(nvm_describe_ordinary_array_authority(&m,&sentinel).status==NVM_OAA_LIMIT);CHECK(sentinel==(void *)(uintptr_t)1);patch(o,type_count,types);
    patch(o,binding_count,fields+1);CHECK(nvm_describe_ordinary_array_authority(&m,&sentinel).status==NVM_OAA_LIMIT);patch(o,binding_count,fields);
    patch(l,0,layouts+1);CHECK(nvm_describe_ordinary_array_authority(&m,&sentinel).status==NVM_OAA_LIMIT);patch(l,0,layouts);
    /* Third record gains one field: the aggregate field ceiling refuses before decode. */
    size_t third=4u+8u*2u+12u*fields;l[third+2]=1;m.layout_size+=12;
    CHECK(nvm_describe_ordinary_array_authority(&m,&sentinel).status==NVM_OAA_LIMIT);CHECK(sentinel==(void *)(uintptr_t)1);
    free(l);free(o);CHECK(nvm_ordinary_array_authority_counts(p,&counts));CHECK(counts.layouts==layouts&&counts.types==types&&counts.bindings==fields);
    CHECK(nvm_ordinary_array_authority_binding(p,fields-1,&b)&&b.layout==1&&!b.field&&b.element_type==types-1);
    NvmOrdinaryArrayType t;CHECK(nvm_ordinary_array_authority_type(p,types-1,&t)&&t.tag==TAG_INT&&t.referent==UINT32_MAX);nvm_ordinary_array_authority_free(p);
}
int main(void){
    mixed_union();binding_order();nominal_identity();maximum_tables();
    Case c;const uint8_t tags[]={TAG_INT,TAG_U8,TAG_FLOAT,TAG_BOOL,TAG_STRING};
    for(unsigned i=0;i<5;i++)for(unsigned forward=0;forward<2;forward++){make_case(&c,tags[i],forward);described(&c);}
    make_case(&c,TAG_INT,false);c.module.service_data=(void *)(uintptr_t)1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.module.passive_data=(void *)(uintptr_t)1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.module.service_size=1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);c.module.passive_size=1;refusal(&c,NVM_OAA_UNKNOWN);
    make_case(&c,TAG_INT,false);patch(c.ownership,c.binding-4,65537);refusal(&c,NVM_OAA_LIMIT);
    make_case(&c,TAG_INT,false);c.layouts[12]=TAG_INT;refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_INT,false);c.ownership[8]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;refusal(&c,NVM_OAA_INVALID);
    make_case(&c,TAG_STRUCT,false);patch(c.ownership,c.type+4,0);c.ownership[8]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;refusal(&c,NVM_OAA_INVALID);
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
    NvmOrdinaryArrayAuthority budget={0};budget.steps=OAA_STEPS-1;CHECK(oaa_charge(&budget,1));CHECK(!oaa_charge(&budget,1)&&budget.failure==NVM_OAA_LIMIT&&budget.steps==OAA_STEPS);
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
