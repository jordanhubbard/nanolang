#include "affine_state.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "verifier.h"
#include "isa.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static unsigned checks;
#define CHECK(c) do { checks++; assert(c); } while(0)
#define NO_CHANGE(s, expression) do { NvmAffineState *before=nvm_affine_state_clone(s); CHECK(before); CHECK(!(expression)); CHECK(nvm_affine_state_equal(s,before)); nvm_affine_state_free(before); } while(0)
#ifdef AFFINE_ALLOCATION_TEST
static unsigned allocation_attempts, fail_at;
void *affine_test_malloc(size_t size) {
    if (++allocation_attempts==fail_at) return NULL;
    return malloc(size);
}
void *affine_test_calloc(size_t count,size_t size) {
    if (++allocation_attempts==fail_at) return NULL;
    return calloc(count,size);
}
#endif
static void word(uint8_t *p,uint32_t v) { for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i)); }
static void slot(uint8_t *p,uint8_t tag,uint8_t mode,uint32_t layout) {
    p[0]=tag;p[1]=mode;word(p+4,layout);
}
/* 0/1 scalar locals, 2/3/4 Handle, 5/6 Pair<Handle,Handle>, 7 same-shaped
 * distinct OtherHandle. Main has no parameters. Inspect borrows its parameter. */
static NvmModule *fixture(void) {
    AsmResult result;
    NvmModule *m=asm_assemble(".types 3 0 0\n.entry 0\n"
        ".function main 0 8 0 void 0\nRET\n.end\n"
        ".function inspect 1 1 0 void 0\nRET\n.end\n.parameters 1 struct\n",&result);
    CHECK(m);
    NvmV2LayoutField scalar={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField pair[2]={{TAG_STRUCT,0,NVM_V2_NO_INDEX},{TAG_STRUCT,0,NVM_V2_NO_INDEX}};
    NvmV2Layout layouts[3]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar},
        {NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,pair},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar}};
    NvmV2Layouts all={layouts,3}; CHECK(nvm_retain_layouts(m,&all)==NVM_V2_OK);
    m->ownership_size=112;m->ownership_data=calloc(112,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,1);word(p+4,3);p[8]=p[9]=p[10]=3;word(p+12,2);
    p[16]=8;slot(p+20,TAG_VOID,0,NVM_V2_NO_INDEX);
    for(unsigned i=0;i<8;i++)slot(p+28+8*i,i<2?TAG_INT:TAG_STRUCT,0,
        i<2?NVM_V2_NO_INDEX:i<5?0:i<7?1:2);
    p[92]=1;p[94]=1;slot(p+96,TAG_VOID,0,NVM_V2_NO_INDEX);slot(p+104,TAG_STRUCT,2,0);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    CHECK(!nvm_verify(m).ok); return m;
}
static void make_pair(NvmAffineState *s) {
    uint16_t fd=0,two[2]={2,3};
    CHECK(nvm_affine_scalar_define(s,0));
    CHECK(nvm_affine_pack(s,2,&fd,1));CHECK(nvm_affine_pack(s,3,&fd,1));
    CHECK(nvm_affine_pack(s,5,two,2));
}
static void consume_pair(NvmAffineState *s) {
    uint16_t two[2]={2,3},fd=0;
    CHECK(nvm_affine_unpack(s,5,two,2));
    CHECK(nvm_affine_unpack(s,2,&fd,1));CHECK(nvm_affine_unpack(s,3,&fd,1));
    CHECK(nvm_affine_can_exit(s,UINT16_MAX));
}
static void scalar_initialization_meets(void) {
    const uint8_t scalars[]={TAG_INT,TAG_BOOL,TAG_U8,TAG_FLOAT};
    for (unsigned i=0;i<sizeof(scalars);i++) {
        NvmModule *m=fixture();slot(m->ownership_data+28,scalars[i],0,NVM_V2_NO_INDEX);
        NvmAffineState *empty=nvm_affine_state_create(m,0,8);CHECK(empty);
        NvmAffineState *defined=nvm_affine_state_clone(empty);CHECK(defined);
        CHECK(nvm_affine_scalar_define(defined,0));CHECK(!nvm_affine_state_equal(empty,defined));
        bool changed=true;uint8_t tag,mode;
        CHECK(nvm_affine_state_meet_initialization(empty,defined,&changed) && !changed);
        CHECK(!nvm_affine_local_info(empty,0,&tag,&mode));
        CHECK(nvm_affine_state_meet_initialization(defined,empty,&changed) && changed);
        CHECK(nvm_affine_state_equal(empty,defined));
        CHECK(nvm_affine_scalar_define(empty,0));CHECK(nvm_affine_scalar_define(defined,0));
        CHECK(nvm_affine_state_meet_initialization(defined,empty,&changed) && !changed);
        CHECK(nvm_affine_local_info(defined,0,&tag,&mode) && tag==scalars[i] && !mode);
        nvm_affine_state_free(empty);nvm_affine_state_free(defined);nvm_module_free(m);
    }
    NvmModule *m=fixture();NvmAffineState *a=nvm_affine_state_create(m,0,8);CHECK(a);make_pair(a);
    NvmAffineState *b=nvm_affine_state_clone(a);CHECK(b);CHECK(nvm_affine_scalar_define(a,1));
    bool changed=true;
    CHECK(nvm_affine_region_begin(b));
    NO_CHANGE(a,nvm_affine_state_meet_initialization(a,b,&changed));CHECK(!changed);
    CHECK(nvm_affine_region_end(b));
    NvmAffineType moved;CHECK(nvm_affine_take_local(b,5,&moved));
    NO_CHANGE(a,nvm_affine_state_meet_initialization(a,b,&changed));CHECK(!changed);
    CHECK(nvm_affine_put_local(b,5,moved));
    CHECK(nvm_affine_region_begin(a));CHECK(nvm_affine_region_begin(b));
    uint16_t left=0,right=1;
    CHECK(nvm_affine_borrow(a,0,5,&left,1,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_borrow(b,0,5,&right,1,NVM_REFERENCE_SHARED));
    NO_CHANGE(a,nvm_affine_state_meet_initialization(a,b,&changed));CHECK(!changed);
    nvm_affine_state_free(a);nvm_affine_state_free(b);nvm_module_free(m);
}
static void caller_binding_checks(void) {
    NvmModule *m=fixture();
    NvmAffineState *caller=nvm_affine_state_create(m,0,8);CHECK(caller);
    make_pair(caller);CHECK(nvm_affine_region_begin(caller));
    uint16_t left=0,right=1;
    CHECK(nvm_affine_borrow(caller,0,5,&left,1,NVM_REFERENCE_EXCLUSIVE));
    CHECK(nvm_affine_borrow(caller,1,5,&right,1,NVM_REFERENCE_EXCLUSIVE));
    NvmAffineState *before=nvm_affine_state_clone(caller);CHECK(before);
    NvmAffineState *callee=nvm_affine_state_create(m,1,4);CHECK(callee);
    CHECK(nvm_affine_bind_caller(callee,caller,0));
    CHECK(nvm_affine_state_equal(caller,before));
    uint8_t tag,mode;
    CHECK(!nvm_affine_local_info(callee,0,&tag,&mode));
    CHECK(!nvm_affine_take_local(callee,0,&(NvmAffineType){0}));
    CHECK(nvm_affine_reference_field(callee,0,0,true,&tag) && tag==TAG_INT);
    CHECK(nvm_affine_region_begin(callee));
    CHECK(nvm_affine_reborrow(callee,1,0,NVM_REFERENCE_EXCLUSIVE));
    CHECK(!nvm_affine_reference_access(callee,0,0,false));
    CHECK(nvm_affine_reference_access(callee,1,0,true));
    CHECK(!nvm_affine_can_exit(callee,UINT16_MAX));
    CHECK(nvm_affine_region_end(callee));
    CHECK(nvm_affine_reference_access(callee,0,0,true));
    CHECK(nvm_affine_can_exit(callee,UINT16_MAX));
    NO_CHANGE(callee,nvm_affine_bind_caller(callee,caller,1));
    nvm_affine_state_free(callee);
    /* An active child suspends the caller parent; a disjoint sibling does not. */
    CHECK(nvm_affine_region_begin(caller));
    CHECK(nvm_affine_reborrow(caller,2,0,NVM_REFERENCE_SHARED));
    callee=nvm_affine_state_create(m,1,4);CHECK(callee);
    NO_CHANGE(callee,nvm_affine_bind_caller(callee,caller,0));
    CHECK(nvm_affine_bind_caller(callee,caller,1));
    nvm_affine_state_free(callee);
    /* Shared downgrade coexists with a shared child; no upgrade is inferred. */
    m->ownership_data[105]=NVM_REFERENCE_SHARED;
    callee=nvm_affine_state_create(m,1,4);CHECK(callee);
    CHECK(nvm_affine_bind_caller(callee,caller,0));
    CHECK(nvm_affine_reference_access(callee,0,0,false));
    CHECK(!nvm_affine_reference_access(callee,0,0,true));
    nvm_affine_state_free(callee);
    m->ownership_data[105]=NVM_REFERENCE_EXCLUSIVE;
    callee=nvm_affine_state_create(m,1,4);CHECK(callee);
    NO_CHANGE(callee,nvm_affine_bind_caller(callee,caller,2));
    nvm_affine_state_free(callee);
    /* A same-shaped different nominal declaration cannot bind. */
    slot(m->ownership_data+104,TAG_STRUCT,2,2);
    callee=nvm_affine_state_create(m,1,4);CHECK(callee);
    NO_CHANGE(callee,nvm_affine_bind_caller(callee,caller,1));
    nvm_affine_state_free(callee);
    slot(m->ownership_data+104,TAG_STRUCT,2,0);
#ifdef AFFINE_ALLOCATION_TEST
    callee=nvm_affine_state_create(m,1,4);CHECK(callee);
    NvmAffineState *unbound=nvm_affine_state_clone(callee);CHECK(unbound);
    allocation_attempts=0;fail_at=1;
    CHECK(!nvm_affine_bind_caller(callee,caller,1));
    fail_at=0;CHECK(nvm_affine_state_equal(callee,unbound));
    nvm_affine_state_free(unbound);nvm_affine_state_free(callee);
#endif
    CHECK(nvm_affine_region_end(caller));
    CHECK(nvm_affine_state_equal(caller,before));
    CHECK(nvm_affine_region_end(caller));consume_pair(caller);
    nvm_affine_state_free(before);nvm_affine_state_free(caller);nvm_module_free(m);
}
/* I retain a resource-bearing Choice with two-owner, ordinary and empty arms,
 * a nested Box, and a distinct same-shaped union. */
static NvmModule *union_fixture(void) {
    AsmResult error;
    NvmModule *m=asm_assemble(".types 1 0 3\n.function main 0 8 0 void 0\nRET\n.end\n",&error);
    CHECK(m);
    const char *names[]={"Handle","Choice","Box","Other","Pair","Number","Empty","value"};
    uint32_t names_at[8];
    for(unsigned i=0;i<8;i++)names_at[i]=nvm_add_string(m,names[i],strlen(names[i]));
    NvmV2LayoutField fd={TAG_INT,NVM_V2_NO_INDEX,names_at[7]};
    NvmV2LayoutField fields[]={{TAG_STRUCT,0,names_at[7]},{TAG_STRUCT,0,names_at[7]},
        {TAG_INT,NVM_V2_NO_INDEX,names_at[7]}};
    NvmV2LayoutField child={TAG_UNION,1,names_at[7]};
    NvmV2Layout items[]={{NVM_V2_LAYOUT_STRUCT,1,names_at[0],&fd},
        {NVM_V2_LAYOUT_UNION,3,names_at[1],fields},
        {NVM_V2_LAYOUT_UNION,1,names_at[2],&child},
        {NVM_V2_LAYOUT_UNION,3,names_at[3],fields}};
    NvmV2Layouts layouts={items,4};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=204;m->ownership_data=calloc(204,1);CHECK(m->ownership_data);
    uint8_t *b=m->ownership_data;
    word(b,NVM_OWNERSHIP_UNION_GRAPH_VERSION);word(b+4,4);
    b[8]=b[9]=b[10]=b[11]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;
    word(b+12,1);b[16]=8;slot(b+20,TAG_VOID,0,NVM_V2_NO_INDEX);
    const uint8_t tags[]={TAG_INT,TAG_STRUCT,TAG_STRUCT,TAG_UNION,TAG_UNION,TAG_UNION,TAG_UNION,TAG_INT};
    const uint32_t ids[]={NVM_V2_NO_INDEX,0,0,1,1,2,3,NVM_V2_NO_INDEX};
    for(unsigned i=0;i<8;i++)slot(b+28+8*i,tags[i],0,ids[i]);
    word(b+92,4);word(b+96,0);word(b+100,1);
    b[104]=NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS;b[106]=NVM_OWNERSHIP_EXTENSION_REVISION_1;
    word(b+108,92);word(b+112,3);
    unsigned pos=116;
    for(unsigned i=1;i<4;i++) {
        word(b+pos,i);b[pos+4]=i==2?2:3;pos+=8;
        unsigned offset=0;
        for(unsigned v=0;v<(i==2?2u:3u);v++) {
            unsigned count=i==2?(v==0?1:0):(v==0?2:v==1?1:0);
            word(b+pos,names_at[4+v]);b[pos+4]=offset;b[pos+6]=count;
            offset+=count;pos+=8;
        }
    }
    CHECK(pos==204);
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    CHECK(!nvm_verify(m).ok);return m;
}
static void owned_union_transitions(void) {
    NvmModule *m=union_fixture();NvmAffineState *s=nvm_affine_state_create(m,0,8);CHECK(s);
    uint16_t fd=0,pair[]={1,2},duplicate[]={1,1},number=7,child=3;
    uint16_t variant=99;uint8_t tag=99;
    CHECK(nvm_affine_scalar_define(s,0));
    CHECK(nvm_affine_pack(s,1,&fd,1));CHECK(nvm_affine_pack(s,2,&fd,1));
    NO_CHANGE(s,nvm_affine_union_define(s,3,1,0));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,0,duplicate,2));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,0,pair,1));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,1,pair,2));
    CHECK(nvm_affine_union_pack(s,3,0,pair,2));
    CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,2,NULL,0));
    NO_CHANGE(s,nvm_affine_pack(s,3,pair,2));
    NO_CHANGE(s,nvm_affine_unpack(s,3,pair,2));
    CHECK(!nvm_affine_scalar_field(s,3,2,&tag) && tag==99);
    NO_CHANGE(s,nvm_affine_move(s,3,6));
    CHECK(nvm_affine_move(s,3,4));
    CHECK(!nvm_affine_union_variant(s,3,&variant) && variant==99);
    CHECK(nvm_affine_union_variant(s,4,&variant) && variant==0);
    NO_CHANGE(s,nvm_affine_union_refine(s,4,1));
    CHECK(nvm_affine_pack(s,1,&fd,1));
    NO_CHANGE(s,nvm_affine_union_unpack(s,4,0,pair,2));
    CHECK(nvm_affine_unpack(s,1,&fd,1));
    NvmAffineType out[2]={{99,99},{99,99}};uint16_t count=99;
    NO_CHANGE(s,nvm_affine_take_union_payload(s,4,0,out,1,&count));
    CHECK(count==99 && out[0].tag==99 && out[0].layout==99);
    CHECK(nvm_affine_union_unpack(s,4,0,pair,2));
    NO_CHANGE(s,nvm_affine_union_unpack(s,4,0,pair,2));
    CHECK(nvm_affine_unpack(s,1,&fd,1));CHECK(nvm_affine_unpack(s,2,&fd,1));
    CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    /* I do not inherit an inner variant from the outer selection. */
    CHECK(nvm_affine_union_pack(s,3,1,&fd,1));
    CHECK(nvm_affine_union_pack(s,5,0,&child,1));
    CHECK(nvm_affine_union_unpack(s,5,0,&child,1));
    NO_CHANGE(s,nvm_affine_union_unpack(s,3,1,&number,1));
    CHECK(nvm_affine_union_refine(s,3,1));
    CHECK(nvm_affine_union_unpack(s,3,1,&number,1));
    CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    /* Empty arms still carry the whole-union obligation until extraction. */
    CHECK(nvm_affine_union_pack(s,3,2,NULL,0));
    CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    NvmAffineState *live=nvm_affine_state_clone(s);CHECK(live);
    CHECK(nvm_affine_take_union_payload(s,3,2,NULL,0,&count) && count==0);
    bool changed=true;
    NO_CHANGE(s,nvm_affine_state_meet_initialization(s,live,&changed));CHECK(!changed);
    NO_CHANGE(live,nvm_affine_state_meet_initialization(live,s,&changed));CHECK(!changed);
    nvm_affine_state_free(live);
    NO_CHANGE(s,nvm_affine_take_union_payload(s,3,2,NULL,0,&count));
    CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    /* Equal owner liveness can join different variants, losing only selection. */
    CHECK(nvm_affine_union_pack(s,3,2,NULL,0));
    NvmAffineState *other=nvm_affine_state_clone(s);CHECK(other);
    CHECK(nvm_affine_union_unpack(other,3,2,NULL,0));
    CHECK(nvm_affine_union_pack(other,3,1,&fd,1));
    CHECK(nvm_affine_state_meet_initialization(s,other,&changed) && changed);
    variant=99;CHECK(!nvm_affine_union_variant(s,3,&variant) && variant==99);
    NO_CHANGE(s,nvm_affine_take_union_payload(s,3,2,NULL,0,&count));
    CHECK(nvm_affine_union_refine(s,3,2));
    CHECK(nvm_affine_take_union_payload(s,3,2,NULL,0,&count));
    nvm_affine_state_free(other);
    /* Stack moves keep exact nominal type and never reuse stale selection. */
    CHECK(nvm_affine_union_pack(s,3,2,NULL,0));
    NvmAffineType token={99,99};CHECK(nvm_affine_take_local(s,3,&token));
    CHECK(token.tag==TAG_UNION && token.layout==1);
    NO_CHANGE(s,nvm_affine_put_local(s,6,token));
    CHECK(nvm_affine_put_local(s,4,token));
    NO_CHANGE(s,nvm_affine_put_local(s,4,token));
    NO_CHANGE(s,nvm_affine_take_union_payload(s,4,2,NULL,0,&count));
    CHECK(nvm_affine_union_refine(s,4,2));
    CHECK(nvm_affine_take_union_payload(s,4,2,NULL,0,&count));
    CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    nvm_affine_state_free(s);nvm_module_free(m);
}

int main(void) {
    owned_union_transitions();
    scalar_initialization_meets();
    caller_binding_checks();
    NvmModule *m=fixture();NvmAffineState *s=nvm_affine_state_create(m,0,8);CHECK(s);
    CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    NO_CHANGE(s,nvm_affine_scalar_define(s,2));
    NO_CHANGE(s,nvm_affine_move(s,2,3));
    uint16_t fd=0,two[2]={2,3},duplicate[2]={2,2};
    CHECK(nvm_affine_scalar_define(s,0));CHECK(nvm_affine_pack(s,2,&fd,1));
    CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    NO_CHANGE(s,nvm_affine_move(s,2,7)); /* Exact nominal identity. */
    NO_CHANGE(s,nvm_affine_pack(s,5,duplicate,2));
    NO_CHANGE(s,nvm_affine_pack(s,2,&fd,1)); /* No overwrite of live owner. */
    CHECK(nvm_affine_move(s,2,3));
    NO_CHANGE(s,nvm_affine_move(s,2,4));
    CHECK(nvm_affine_owner_access(s,3,&fd,1,false));
    CHECK(nvm_affine_pack(s,2,&fd,1));CHECK(nvm_affine_pack(s,5,two,2));
    NO_CHANGE(s,nvm_affine_unpack(s,5,duplicate,2));
    NO_CHANGE(s,nvm_affine_unpack(s,5,two,1));
    CHECK(!nvm_affine_owner_access(s,2,NULL,0,false));
    NvmAffineState *join=nvm_affine_state_clone(s);CHECK(join);
    CHECK(nvm_affine_state_equal(s,join));
    CHECK(nvm_affine_move(join,5,6));CHECK(!nvm_affine_state_equal(s,join));
    CHECK(nvm_affine_move(join,6,5));CHECK(nvm_affine_state_equal(s,join));
    nvm_affine_state_free(join);
    CHECK(nvm_affine_pack(s,4,&fd,1));
    uint16_t occupied[2]={2,4};
    NO_CHANGE(s,nvm_affine_unpack(s,5,occupied,2));
    CHECK(!nvm_affine_owner_access(s,2,NULL,0,false));
    CHECK(nvm_affine_unpack(s,4,&fd,1));
    consume_pair(s);NO_CHANGE(s,nvm_affine_unpack(s,5,two,2));
    nvm_affine_state_free(s);

    s=nvm_affine_state_create(m,0,8);CHECK(s);make_pair(s);
    uint16_t left=0,right=1,scalar_path[2]={0,0};
    NO_CHANGE(s,nvm_affine_borrow(s,0,5,&left,1,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_region_begin(s));
    CHECK(nvm_affine_borrow(s,0,5,&left,1,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_borrow(s,1,5,&left,1,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_reference_access(s,0,0,false));
    CHECK(!nvm_affine_reference_access(s,0,0,true));
    CHECK(nvm_affine_owner_access(s,5,scalar_path,2,false));
    CHECK(!nvm_affine_owner_access(s,5,scalar_path,2,true));
    NO_CHANGE(s,nvm_affine_move(s,5,6)); /* Later argument cannot move owner. */
    NO_CHANGE(s,nvm_affine_unpack(s,5,two,2));
    NO_CHANGE(s,nvm_affine_borrow(s,2,5,&left,1,NVM_REFERENCE_EXCLUSIVE));
    CHECK(nvm_affine_borrow(s,2,5,&right,1,NVM_REFERENCE_EXCLUSIVE));
    CHECK(nvm_affine_reference_access(s,2,0,true));
    scalar_path[0]=1;CHECK(!nvm_affine_owner_access(s,5,scalar_path,2,false));
    CHECK(nvm_affine_region_begin(s));
    NO_CHANGE(s,nvm_affine_reborrow(s,3,0,NVM_REFERENCE_EXCLUSIVE));
    CHECK(nvm_affine_reborrow(s,3,2,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_reference_access(s,2,0,false));
    CHECK(!nvm_affine_reference_access(s,2,0,true));
    CHECK(nvm_affine_reference_access(s,3,0,false));
    NO_CHANGE(s,nvm_affine_reborrow(s,4,2,NVM_REFERENCE_EXCLUSIVE));
    CHECK(nvm_affine_region_end(s));CHECK(!nvm_affine_reference_access(s,3,0,false));
    CHECK(nvm_affine_reference_access(s,2,0,true));
    CHECK(nvm_affine_region_begin(s));CHECK(nvm_affine_reborrow(s,3,2,NVM_REFERENCE_EXCLUSIVE));
    CHECK(!nvm_affine_reference_access(s,2,0,false));
    CHECK(nvm_affine_reference_access(s,3,0,true));
    CHECK(nvm_affine_region_begin(s));CHECK(nvm_affine_reborrow(s,4,3,NVM_REFERENCE_SHARED));
    CHECK(!nvm_affine_reference_access(s,2,0,false));
    CHECK(!nvm_affine_reference_access(s,3,0,true));
    CHECK(nvm_affine_region_end(s));CHECK(nvm_affine_region_end(s));
    CHECK(nvm_affine_reference_access(s,2,0,true));
    CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    join=nvm_affine_state_clone(s);CHECK(join);CHECK(nvm_affine_state_equal(s,join));
    CHECK(nvm_affine_region_end(join));CHECK(!nvm_affine_state_equal(s,join));
    nvm_affine_state_free(join);
    CHECK(nvm_affine_region_end(s));CHECK(!nvm_affine_reference_access(s,2,0,true));
    NO_CHANGE(s,nvm_affine_region_end(s));consume_pair(s);nvm_affine_state_free(s);

    s=nvm_affine_state_create(m,1,4);CHECK(s);
    CHECK(nvm_affine_reference_access(s,0,0,true));
    CHECK(!nvm_affine_owner_access(s,0,NULL,0,false));
    NO_CHANGE(s,nvm_affine_scalar_define(s,0));
    CHECK(nvm_affine_region_begin(s));CHECK(nvm_affine_reborrow(s,1,0,NVM_REFERENCE_SHARED));
    CHECK(!nvm_affine_reference_access(s,0,0,true));
    CHECK(nvm_affine_region_end(s));CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    CHECK(!nvm_affine_can_exit(s,0)); /* No returned borrowed parameter. */
    nvm_affine_state_free(s);
    m->ownership_data[105]=NVM_REFERENCE_SHARED;
    s=nvm_affine_state_create(m,1,4);CHECK(s);
    CHECK(!nvm_affine_reference_access(s,0,0,true));
    CHECK(nvm_affine_region_begin(s));
    NO_CHANGE(s,nvm_affine_reborrow(s,1,0,NVM_REFERENCE_EXCLUSIVE));
    CHECK(nvm_affine_reborrow(s,1,0,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_region_end(s));nvm_affine_state_free(s);
    m->ownership_data[105]=NVM_REFERENCE_EXCLUSIVE;
    /* Unknown ordinary metadata is transportable, but not an exact fact. */
    slot(m->ownership_data+28,TAG_STRUCT,0,NVM_V2_NO_INDEX);
    CHECK(nvm_affine_state_create(m,0,8)==NULL);
    slot(m->ownership_data+28,TAG_VOID,0,NVM_V2_NO_INDEX);
    CHECK(nvm_affine_state_create(m,0,8)==NULL);
    slot(m->ownership_data+28,TAG_INT,0,NVM_V2_NO_INDEX);
    /* A declared owned result transfers one exact obligation; a sibling
     * resource still live prevents exit. No return signature is inferred. */
    slot(m->ownership_data+20,TAG_STRUCT,0,0);
    m->functions[0].result_count=1;m->functions[0].result_tag=TAG_STRUCT;
    s=nvm_affine_state_create(m,0,8);CHECK(s);
    CHECK(nvm_affine_scalar_define(s,0));CHECK(nvm_affine_pack(s,2,&fd,1));
    CHECK(nvm_affine_can_exit(s,2));CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    CHECK(nvm_affine_pack(s,3,&fd,1));CHECK(!nvm_affine_can_exit(s,2));
    CHECK(nvm_affine_unpack(s,3,&fd,1));CHECK(nvm_affine_can_exit(s,2));
    nvm_affine_state_free(s);
#ifdef AFFINE_ALLOCATION_TEST
    for(unsigned failure=1;;failure++) {
        allocation_attempts=0;fail_at=failure;
        NvmAffineState *trial=nvm_affine_state_create(m,0,8);
        fail_at=0;
        if(trial){nvm_affine_state_free(trial);break;}
        CHECK(failure<20);
    }
    s=nvm_affine_state_create(m,0,8);CHECK(s);make_pair(s);
    CHECK(nvm_affine_region_begin(s));
    NvmAffineState *before=nvm_affine_state_clone(s);CHECK(before);
    allocation_attempts=0;fail_at=1;
    CHECK(!nvm_affine_borrow(s,0,5,&left,1,NVM_REFERENCE_SHARED));
    fail_at=0;CHECK(nvm_affine_state_equal(s,before));nvm_affine_state_free(before);
    CHECK(nvm_affine_borrow(s,0,5,&left,1,NVM_REFERENCE_SHARED));
    for(unsigned failure=1;;failure++) {
        allocation_attempts=0;fail_at=failure;
        NvmAffineState *trial=nvm_affine_state_clone(s);
        fail_at=0;
        if(trial){CHECK(nvm_affine_state_equal(s,trial));nvm_affine_state_free(trial);break;}
        CHECK(failure<20);CHECK(nvm_affine_reference_access(s,0,0,false));
    }
    nvm_affine_state_free(s);
#endif
    s=nvm_affine_state_create(m,1,4);CHECK(s);
    nvm_module_free(m); /* State owns declarations, layouts and paths. */
    CHECK(nvm_affine_reference_access(s,0,0,true));
    nvm_affine_state_free(s);
    printf("%u affine state checks passed\n",checks);return 0;
}
