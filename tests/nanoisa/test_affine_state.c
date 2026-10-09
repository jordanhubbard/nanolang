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
/* I keep unions nominal and concatenate only declaration field ranges:
 * Choice has Some(Handle), Empty, Pair(Handle,Handle); Outer holds Choice. */
static NvmModule *owned_union_fixture(void) {
    AsmResult result;
    NvmModule *m=asm_assemble(".function main 0 7 0 void 0\nRET\n.end\n",&result);CHECK(m);
    uint32_t handle=nvm_add_string(m,"Handle",6),choice=nvm_add_string(m,"Choice",6);
    uint32_t outer=nvm_add_string(m,"Outer",5),some=nvm_add_string(m,"Some",4);
    uint32_t empty=nvm_add_string(m,"Empty",5),pair=nvm_add_string(m,"Pair",4);
    uint32_t field=nvm_add_string(m,"value",5);
    NvmV2LayoutField fd={TAG_INT,NVM_V2_NO_INDEX,field};
    NvmV2LayoutField owners[]={{TAG_STRUCT,0,field},{TAG_STRUCT,0,field},{TAG_STRUCT,0,field}};
    NvmV2LayoutField nested={TAG_UNION,1,field};
    NvmV2Layout items[]={{NVM_V2_LAYOUT_STRUCT,1,handle,&fd},
        {NVM_V2_LAYOUT_UNION,3,choice,owners},{NVM_V2_LAYOUT_UNION,1,outer,&nested}};
    NvmV2Layouts layouts={items,3};m->struct_count=1;m->union_count=2;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=156;m->ownership_data=calloc(156,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,3);word(p+4,3);p[8]=p[9]=p[10]=3;word(p+12,1);
    p[16]=7;slot(p+20,TAG_VOID,0,NVM_V2_NO_INDEX);
    for(unsigned i=0;i<7;i++)slot(p+28+8*i,i==0?TAG_INT:i<3?TAG_STRUCT:TAG_UNION,0,
        i==0?NVM_V2_NO_INDEX:i<3?0:i<5?1:2);
    word(p+84,4);word(p+92,1);p[96]=1;p[98]=1;word(p+100,52);word(p+104,2);
    word(p+108,1);p[112]=3;
    word(p+116,some);p[122]=1;
    word(p+124,empty);p[128]=1;
    word(p+132,pair);p[136]=1;p[138]=2;
    word(p+140,2);p[144]=1;word(p+148,some);p[154]=1;
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK && needs);
    return m;
}
static void owned_union_transfers(void) {
    NvmModule *m=owned_union_fixture();NvmAffineState *s=nvm_affine_state_create(m,0,7);CHECK(s);
    CHECK(nvm_affine_has_complete_unions(s));
    uint16_t fd=0,owner=1,choice=3,both[]={1,2},duplicate[]={1,1};
    CHECK(nvm_affine_scalar_define(s,0));CHECK(nvm_affine_pack(s,1,&fd,1));
    NO_CHANGE(s,nvm_affine_union_define(s,3,1,0));
    NO_CHANGE(s,nvm_affine_pack(s,3,&owner,1));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,2,duplicate,2));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,0,&fd,1));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,9,&owner,1));
    CHECK(nvm_affine_union_pack(s,3,0,&owner,1));
    uint16_t path=0;CHECK(!nvm_affine_owner_access(s,3,&path,1,false));
    CHECK(nvm_affine_pack(s,2,&fd,1));
    uint16_t occupied=2;NO_CHANGE(s,nvm_affine_union_unpack(s,3,0,&occupied,1));
    CHECK(nvm_affine_unpack(s,2,&fd,1));
    CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    NO_CHANGE(s,nvm_affine_union_pack(s,3,1,NULL,0));
    NO_CHANGE(s,nvm_affine_move(s,3,5));
    CHECK(nvm_affine_move(s,3,4));
    uint16_t variant=99;CHECK(nvm_affine_union_variant(s,4,&variant) && variant==0);
    NO_CHANGE(s,nvm_affine_union_unpack(s,3,0,&owner,1));
    NO_CHANGE(s,nvm_affine_union_unpack(s,4,1,NULL,0));
    NO_CHANGE(s,nvm_affine_unpack(s,4,&owner,1));
    CHECK(nvm_affine_union_unpack(s,4,0,&owner,1));
    CHECK(nvm_affine_unpack(s,1,&fd,1));CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    CHECK(nvm_affine_union_pack(s,3,1,NULL,0));
    CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    CHECK(nvm_affine_union_unpack(s,3,1,NULL,0));CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    CHECK(nvm_affine_pack(s,1,&fd,1));CHECK(nvm_affine_pack(s,2,&fd,1));
    CHECK(nvm_affine_union_pack(s,3,2,both,2));
    NO_CHANGE(s,nvm_affine_union_unpack(s,3,2,duplicate,2));
    CHECK(nvm_affine_union_unpack(s,3,2,both,2));
    CHECK(nvm_affine_unpack(s,2,&fd,1));
    CHECK(nvm_affine_union_pack(s,3,0,&owner,1));
    CHECK(nvm_affine_union_pack(s,5,0,&choice,1));
    CHECK(nvm_affine_union_unpack(s,5,0,&choice,1));
    variant=99;CHECK(!nvm_affine_union_variant(s,3,&variant) && variant==99);
    NO_CHANGE(s,nvm_affine_union_unpack(s,3,0,&owner,1));
    CHECK(nvm_affine_union_refine(s,3,0));
    NO_CHANGE(s,nvm_affine_union_refine(s,3,1));
    CHECK(nvm_affine_union_unpack(s,3,0,&owner,1));
    CHECK(nvm_affine_unpack(s,1,&fd,1));CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    CHECK(nvm_affine_union_pack(s,3,1,NULL,0));
    CHECK(nvm_affine_union_pack(s,5,0,&choice,1));
    CHECK(nvm_affine_move(s,5,6));
    CHECK(nvm_affine_union_unpack(s,6,0,&choice,1));
    CHECK(nvm_affine_union_refine(s,3,1));
    CHECK(nvm_affine_union_unpack(s,3,1,NULL,0));
    CHECK(nvm_affine_can_exit(s,UINT16_MAX));
    /* I may forget a selected tag at a join, never the owned token itself. */
    NvmAffineState *empty=nvm_affine_state_clone(s);CHECK(empty);
    CHECK(nvm_affine_union_pack(s,3,1,NULL,0));
    bool changed=true;NO_CHANGE(s,nvm_affine_state_meet_initialization(s,empty,&changed));CHECK(!changed);
    CHECK(nvm_affine_pack(empty,1,&fd,1));CHECK(nvm_affine_union_pack(empty,3,0,&owner,1));
    CHECK(nvm_affine_state_meet_initialization(s,empty,&changed) && changed);
    variant=99;CHECK(!nvm_affine_union_variant(s,3,&variant) && variant==99);
    CHECK(!nvm_affine_can_exit(s,UINT16_MAX));
    NO_CHANGE(s,nvm_affine_union_unpack(s,3,1,NULL,0));
    nvm_affine_state_free(empty);nvm_affine_state_free(s);
    /* State transitions do not yet admit unimplemented bytecode transfers. */
    CHECK(!nvm_verify(m).ok);nvm_module_free(m);
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
int main(void) {
    owned_union_transfers();
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
