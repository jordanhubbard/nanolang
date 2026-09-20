/* I inspect constructor storage only; no bytecode or service executes. */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks, calls, failed, retained;
static long failure = -1;
static bool tracking, transient;
static struct {void *pointer; unsigned call;} allocations[4096];
#define CHECK(x) do {checks++;assert(x);} while(0)
void *variants_test_malloc(size_t size) {
    unsigned call = calls;
    if(tracking) {
        calls++;
        if(failure>=0 && (call==(unsigned)failure || (!transient && call>(unsigned)failure))) {
            failed++;return NULL;
        }
    }
    void *p=malloc(size);
    if(p)memset(p,0xa5,size);
    if(p && tracking) {
        unsigned i=0;while(i<4096 && allocations[i].pointer)i++;
        CHECK(i<4096);allocations[i].pointer=p;allocations[i].call=call;retained++;
    }
    return p;
}
void *variants_test_calloc(size_t n,size_t size) {
    CHECK(!size || n<=SIZE_MAX/size);
    void *p=variants_test_malloc(n*size);if(p)memset(p,0,n*size);return p;
}
void variants_test_free(void *p) {
    if(p)for(unsigned i=0;i<4096;i++)if(allocations[i].pointer==p) {
        allocations[i].pointer=NULL;CHECK(retained);retained--;break;
    }
    free(p);
}
#define malloc variants_test_malloc
#define calloc variants_test_calloc
#define free variants_test_free
#include "../../src/nanoisa/affine_state.c"
#undef malloc
#undef calloc
#undef free
#include "assembler.h"
#include "retained_layouts.h"
static void word(uint8_t *p,uint32_t value) {
    for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(value>>(8*i));
}
static NvmModule *module(unsigned locals) {
    char text[256];snprintf(text,sizeof text,
        ".types 1 0 0\n.entry 0\n.function main 0 %u 0 int 1\nPUSH_I64 0\nRET\n.end\n",locals);
    AsmResult error;NvmModule *m=asm_assemble_unverified(text,&error);CHECK(m);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout row={NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field};
    NvmV2Layouts layouts={&row,1};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=28+8*locals;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,1);word(p+4,1);p[8]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;
    word(p+12,1);p[16]=(uint8_t)locals;p[17]=(uint8_t)(locals>>8);
    for(unsigned i=0;i<=locals;i++){p[20+8*i]=TAG_INT;word(p+24+8*i,NVM_V2_NO_INDEX);}
    return m;
}
static NvmAffineState *construct(unsigned kind,NvmModule *m,MCAnalysis *mixed,LAAnalysis *owner) {
    return kind==0?nvm_affine_state_create(m,0,0):kind==1?mc_checked_state(mixed,0):la_checked_state(owner,0);
}
static unsigned variant_call(const NvmAffineState *s) {
    for(unsigned i=0;i<4096;i++)if(allocations[i].pointer==s->variants)return allocations[i].call;
    CHECK(false);return 0;
}
static void unknown(const NvmAffineState *s,unsigned locals) {
    CHECK(s && s->variants && s->facts->count==locals);
    for(unsigned i=0;i<locals;i++)CHECK(s->variants[i]==NVM_AFFINE_UNKNOWN_VARIANT);
}
static void reset(long at,bool once) {calls=failed=0;failure=at;transient=once;}
static void exercise(unsigned locals) {
    NvmModule *m=module(locals);NvmMixedLayoutView *view=NULL;NvmOwnedArrayLayouts *layouts=NULL;
    CHECK(nvm_describe_mixed_layouts(m,&view).status==NVM_RECORD_DESCRIBED);
    CHECK(nvm_describe_owned_array_layouts(m,&layouts).status==NVM_RECORD_DESCRIBED);
    /* Real descriptor views, not a claim of complete profile/body authority. */
    NvmMixedFloatProof *shape=calloc(1,sizeof *shape);NvmMixedSamplesProof *proof=calloc(1,sizeof *proof);
    MCAnalysis *mixed=calloc(1,sizeof *mixed);LAAnalysis *owner=calloc(1,sizeof *owner);
    NvmOwnerLifetimeFacts *facts=calloc(1,sizeof *facts);CHECK(shape && proof && mixed && owner && facts);
    shape->view=view;proof->shape=shape;mixed->module=m;mixed->proof=proof;
    facts->layouts=layouts;owner->module=m;owner->proof=facts;
    for(unsigned kind=0;kind<3;kind++) {
        CHECK(!retained);tracking=true;reset(-1,false);
        NvmAffineState *s=construct(kind,m,mixed,owner);unknown(s,locals);
        unsigned count=calls,variant=variant_call(s);CHECK(variant<count);
        reset(-1,false);NvmAffineState *copy=nvm_affine_state_clone(s);unknown(copy,locals);
        unsigned clone_count=calls,clone_variant=variant_call(copy);CHECK(clone_variant<clone_count);
        CHECK(copy->variants!=s->variants && copy->live!=s->live && s->facts->users==2);
        if(locals){copy->variants[locals-1]=17;CHECK(s->variants[locals-1]==NVM_AFFINE_UNKNOWN_VARIANT);}
        nvm_affine_state_free(s);CHECK(copy->facts->users==1);
        if(locals)CHECK(copy->variants[locals-1]==17);
        nvm_affine_state_free(copy);CHECK(!retained);
        for(unsigned once=0;once<2;once++) {
            bool hit_variant=false,hit_clone_variant=false;
            for(unsigned at=0;at<count;at++) {
                reset((long)at,once!=0);s=construct(kind,m,mixed,owner);
                CHECK(!s && failed && !retained);if(at==variant)hit_variant=true;
                reset(-1,false);s=construct(kind,m,mixed,owner);unknown(s,locals);
                nvm_affine_state_free(s);CHECK(!retained);
            }
            reset(-1,false);s=construct(kind,m,mixed,owner);unknown(s,locals);
            unsigned baseline=retained;
            for(unsigned at=0;at<clone_count;at++) {
                reset((long)at,once!=0);copy=nvm_affine_state_clone(s);
                CHECK(!copy && failed && retained==baseline && s->facts->users==1);unknown(s,locals);
                if(at==clone_variant)hit_clone_variant=true;
                reset(-1,false);copy=nvm_affine_state_clone(s);unknown(copy,locals);
                nvm_affine_state_free(copy);CHECK(retained==baseline && s->facts->users==1);
            }
            nvm_affine_state_free(s);CHECK(!retained && hit_variant && hit_clone_variant);
        }
        tracking=false;reset(-1,false);
        printf("constructor %u locals %u allocations %u variant %u clone %u clone_variant %u\n",
               kind,locals,count,variant,clone_count,clone_variant);
    }
    nvm_mixed_layout_view_free(view);nvm_owned_array_layouts_free(layouts);
    free(shape);free(proof);free(mixed);free(owner);free(facts);nvm_module_free(m);
}
int main(void) {
    exercise(0);exercise(3);exercise(256);
    CHECK(!retained);printf("%u private affine variant checks passed; no bytecode execution\n",checks);return 0;
}
