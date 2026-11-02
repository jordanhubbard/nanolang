#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../../src/nanoisa/ownership_union_fields_private.h"
#include "../../src/nanoisa/isa.h"
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"I failed line%u: %s\n",__LINE__,#x);exit(1);}}while(0)
#ifdef OAA_INSTRUMENT
static void *allocations[64];static size_t sizes[64],live,calls,fail_at=SIZE_MAX,payload_live,payload_peak;
static bool persistent;
static void *ufd_test_calloc(size_t n,size_t z) {
    size_t index=calls++;
    if(index==fail_at||(persistent&&index>fail_at))return NULL;
    CHECK(!n||z<=SIZE_MAX/n);
    void *p=calloc(n,z);
    if(p){CHECK(live<64);allocations[live]=p;sizes[live++]=n*z;payload_live+=n*z;if(payload_live>payload_peak)payload_peak=payload_live;}
    return p;
}
static void ufd_test_free(void *p) {
    if(!p)return;
    size_t i=0;while(i<live&&allocations[i]!=p)i++;
    CHECK(i<live);payload_live-=sizes[i];live--;allocations[i]=allocations[live];sizes[i]=sizes[live];free(p);
}
#define calloc ufd_test_calloc
#define free ufd_test_free
#include "../../src/nanoisa/nvm_v2_layouts.c"
#include "../../src/nanoisa/ownership_contracts.c"
#undef calloc
#undef free
#endif
typedef struct {NvmModule module;uint8_t layouts[1024],ownership[2048];size_t l,o;} Case;
static void u8(uint8_t *b,size_t *n,uint8_t v){b[(*n)++]=v;}
static void u16(uint8_t *b,size_t *n,uint16_t v){u8(b,n,(uint8_t)v);u8(b,n,(uint8_t)(v>>8));}
static void u32(uint8_t *b,size_t *n,uint32_t v){for(unsigned i=0;i<4;i++)u8(b,n,(uint8_t)(v>>(8*i)));}
static void patch(uint8_t *b,size_t at,uint32_t v){for(unsigned i=0;i<4;i++)b[at+i]=(uint8_t)(v>>(8*i));}
typedef struct {
    Case c;NvmFunctionEntry function;size_t result,local,variant;
} UnionCase;
static void union_case(UnionCase *v) {
    memset(v,0,sizeof *v);Case *c=&v->c;
    u32(c->layouts,&c->l,2);
    for(unsigned i=0;i<2;i++) {
        u8(c->layouts,&c->l,i?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION);
        u8(c->layouts,&c->l,0);u16(c->layouts,&c->l,1);u32(c->layouts,&c->l,i);
        u8(c->layouts,&c->l,i?TAG_UNION:TAG_INT);u8(c->layouts,&c->l,0);u16(c->layouts,&c->l,0);
        u32(c->layouts,&c->l,i?0:UINT32_MAX);u32(c->layouts,&c->l,2+i);
    }
    u32(c->ownership,&c->o,3);u32(c->ownership,&c->o,2);
    u8(c->ownership,&c->o,1);u8(c->ownership,&c->o,1);u16(c->ownership,&c->o,0);
    u32(c->ownership,&c->o,1);u16(c->ownership,&c->o,1);u16(c->ownership,&c->o,0);
    v->result=c->o;u8(c->ownership,&c->o,TAG_INT);u8(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u32(c->ownership,&c->o,UINT32_MAX);
    v->local=c->o;u8(c->ownership,&c->o,TAG_UNION);u8(c->ownership,&c->o,0);u16(c->ownership,&c->o,0);u32(c->ownership,&c->o,0);
    u32(c->ownership,&c->o,4);u32(c->ownership,&c->o,0);u32(c->ownership,&c->o,1);
    u16(c->ownership,&c->o,1);u16(c->ownership,&c->o,1);u32(c->ownership,&c->o,28);
    u32(c->ownership,&c->o,1);u32(c->ownership,&c->o,0);u16(c->ownership,&c->o,2);u16(c->ownership,&c->o,0);
    v->variant=c->o;
    u32(c->ownership,&c->o,4);u16(c->ownership,&c->o,0);u16(c->ownership,&c->o,1);
    u32(c->ownership,&c->o,5);u16(c->ownership,&c->o,1);u16(c->ownership,&c->o,0);
    v->function.local_count=1;v->function.result_count=1;v->function.result_tag=TAG_INT;
    c->module.functions=&v->function;c->module.function_count=1;
    c->module.layout_data=c->layouts;c->module.layout_size=(uint32_t)c->l;
    c->module.ownership_data=c->ownership;c->module.ownership_size=(uint32_t)c->o;
    c->module.struct_count=1;c->module.union_count=1;c->module.string_count=6;
}
static void union_facts(const NvmUnionFieldDeclarations *p) {
    NvmDeclarationLayout l;NvmV2LayoutField f;NvmUnionVariantFact v;uint32_t nominal=99;
    CHECK(nvm_union_field_declarations_nominal(p,TAG_STRUCT,0,&nominal)&&nominal==1);
    CHECK(nvm_union_field_declarations_nominal(p,TAG_UNION,0,&nominal)&&nominal==0);
    CHECK(nvm_union_field_declarations_layout(p,1,&l)&&l.kind==NVM_V2_LAYOUT_STRUCT&&l.flags==1);
    CHECK(nvm_union_field_declarations_field(p,1,0,&f)&&f.type_tag==TAG_UNION&&f.nested_idx==0);
    CHECK(nvm_union_field_declarations_variant(p,0,0,&v)&&v.layout==0&&v.field_count==1&&v.field_offset==0);
    CHECK(nvm_union_field_declarations_variant(p,0,1,&v)&&v.field_count==0&&v.field_offset==1);
    nominal=99;CHECK(!nvm_union_field_declarations_nominal(p,TAG_STRUCT,1,&nominal)&&nominal==99);
    NvmUnionVariantFact old=v;CHECK(!nvm_union_field_declarations_variant(p,1,0,&v)&&!memcmp(&old,&v,sizeof v));
}
static void union_refuse(UnionCase *v,NvmUnionFieldDeclarationStatus status) {
    NvmUnionFieldDeclarations *p=(void *)(uintptr_t)1;
    CHECK(nvm_union_field_declarations_prepare(&v->c.module,&p).status==status);
    CHECK(p==(void *)(uintptr_t)1);
}
int main(void) {
    UnionCase v;union_case(&v);bool needs=false;
    /* The common whole-metadata validator independently precedes new owner. */
    NvmV2Result common=nvm_ownership_contracts_validate(&v.c.module,&needs);
    if(common!=NVM_V2_OK||!needs){fprintf(stderr,"common fixture refused status=%u needs=%u\n",(unsigned)common,(unsigned)needs);return 1;}
    CHECK(common==NVM_V2_OK&&needs);
    NvmUnionFieldDeclarations *p=NULL;
    NvmUnionFieldDeclarationResult r=nvm_union_field_declarations_prepare(&v.c.module,&p);
    if(r.status!=NVM_UFD_PREPARED){fprintf(stderr,"first prepare status=%u: %s\n",(unsigned)r.status,r.message);return 1;}
    CHECK(r.status==NVM_UFD_PREPARED);union_facts(p);
    memset(v.c.layouts,0,sizeof v.c.layouts);memset(v.c.ownership,0,sizeof v.c.ownership);
    union_facts(p);nvm_union_field_declarations_free(p);nvm_union_field_declarations_free(NULL);
    union_case(&v);v.c.module.layout_size++;union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);v.c.module.ownership_size++;union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);v.c.ownership[v.local+2]=1;union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);patch(v.c.ownership,v.variant+8,4);union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);patch(v.c.ownership,v.variant,6);union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);v.c.ownership[v.variant+6]=0;union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);v.c.ownership[v.local+1]=1;union_refuse(&v,NVM_UFD_UNKNOWN);
    union_case(&v);patch(v.c.layouts,8,UINT32_MAX);union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);v.c.layouts[13]=1;union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);patch(v.c.layouts,36,1);union_refuse(&v,NVM_UFD_UNKNOWN);
    union_case(&v);patch(v.c.ownership,v.local+4,1);union_refuse(&v,NVM_UFD_INVALID);
    union_case(&v);v.c.ownership[v.variant+12]=0;union_refuse(&v,NVM_UFD_INVALID);
    /* A second function has an independently malformed late descriptor. */
    union_case(&v);
    NvmFunctionEntry functions[2]={v.function,v.function};
    size_t suffix=v.local+8;
    memmove(v.c.ownership+suffix+20,v.c.ownership+suffix,v.c.o-suffix);
    memcpy(v.c.ownership+suffix,v.c.ownership+16,20);
    v.c.o+=20;v.c.module.ownership_size=(uint32_t)v.c.o;
    patch(v.c.ownership,12,2);v.c.module.functions=functions;v.c.module.function_count=2;
    v.c.ownership[suffix+4+2]=1;union_refuse(&v,NVM_UFD_INVALID);
    /* Empty-first ordering is structural, not a Some/None name allowlist. */
    union_case(&v);v.c.ownership[v.variant+6]=0;v.c.ownership[v.variant+12]=0;v.c.ownership[v.variant+14]=1;
    p=NULL;r=nvm_union_field_declarations_prepare(&v.c.module,&p);CHECK(r.status==NVM_UFD_PREPARED);
    NvmUnionVariantFact reversed;CHECK(nvm_union_field_declarations_variant(p,0,0,&reversed)&&reversed.field_count==0);
    CHECK(nvm_union_field_declarations_variant(p,0,1,&reversed)&&reversed.field_count==1&&reversed.field_offset==0);
    nvm_union_field_declarations_free(p);
    union_case(&v);p=NULL;
    CHECK(nvm_union_field_declarations_prepare(&v.c.module,&p).status==NVM_UFD_PREPARED);
    NvmUnionFieldDeclarations *prior=p;
    v.c.ownership[v.local+2]=1;
    CHECK(nvm_union_field_declarations_prepare(&v.c.module,&p).status==NVM_UFD_INVALID&&p==prior);
    union_facts(prior);v.c.ownership[v.local+2]=0;
    nvm_union_field_declarations_free(prior);

#ifdef OAA_INSTRUMENT
    union_case(&v);calls=0;payload_peak=0;p=NULL;
    r=nvm_union_field_declarations_prepare(&v.c.module,&p);CHECK(r.status==NVM_UFD_PREPARED);
    size_t measured=calls;CHECK(measured==4);CHECK(payload_live==r.bytes_reserved&&payload_peak==r.bytes_reserved);nvm_union_field_declarations_free(p);CHECK(!live&&!payload_live);
    for(unsigned mode=0;mode<2;mode++)for(size_t at=0;at<measured;at++) {
        p=(void *)(uintptr_t)1;calls=0;fail_at=at;persistent=mode!=0;
        CHECK(nvm_union_field_declarations_prepare(&v.c.module,&p).status==NVM_UFD_MEMORY);
        CHECK(p==(void *)(uintptr_t)1&&!live);fail_at=SIZE_MAX;persistent=false;
        CHECK(nvm_union_field_declarations_prepare(&v.c.module,&p).status==NVM_UFD_PREPARED);
        union_facts(p);nvm_union_field_declarations_free(p);CHECK(!live&&!payload_live);
    }
    p=NULL;CHECK(ufd_prepare_reserved(&v.c.module,&p,OAA_BYTES-r.bytes_reserved,OAA_STEPS-r.work_reserved).status==NVM_UFD_PREPARED);
    nvm_union_field_declarations_free(p);p=(void *)(uintptr_t)1;
    CHECK(ufd_prepare_reserved(&v.c.module,&p,OAA_BYTES-r.bytes_reserved+1,0).status==NVM_UFD_LIMIT);
    CHECK(p==(void *)(uintptr_t)1&&!live);
    CHECK(ufd_prepare_reserved(&v.c.module,&p,0,OAA_STEPS-r.work_reserved+1).status==NVM_UFD_LIMIT);
    CHECK(p==(void *)(uintptr_t)1&&!live);
    CHECK(ufd_prepare_reserved(&v.c.module,&p,UINT64_MAX,0).status==NVM_UFD_LIMIT);
    CHECK(p==(void *)(uintptr_t)1&&!live);
    printf("owner=%zu layout=%zu field=%zu reserved=%llu work=%llu allocations=%zu\n",sizeof(NvmUnionFieldDeclarations),sizeof(NvmV2Layout),sizeof(NvmV2LayoutField),(unsigned long long)r.bytes_reserved,(unsigned long long)r.work_reserved,measured);
#endif
    printf("PASS %u union declaration checks; no execution authority\n",checks);return 0;
}
