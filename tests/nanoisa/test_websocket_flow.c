#include "../../src/nanoisa/websocket_flow.h"
#include "../../src/nanoisa/nvm_v2_sections.h"
#include "../../src/nanoisa/isa.h"
#include "../../src/nsi_websocket_plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#define OK(x) CHECK((x)==NVM_WEBSOCKET_FLOW_OK)
#ifdef FLOW_INSTRUMENT
static bool refuse;static unsigned live;
static void *flow_malloc(size_t n){if(refuse)return NULL;void *p=malloc(n);if(p)live++;return p;}
static void *flow_calloc(size_t n,size_t z){if(n && z>SIZE_MAX/n)return NULL;void *p=flow_malloc(n*z);if(p)memset(p,0,n*z);return p;}
static void flow_free(void *p){if(p){CHECK(live);live--;}free(p);}
#define malloc flow_malloc
#define calloc flow_calloc
#define free flow_free
#include "../../src/nanoisa/websocket_flow.c"
#undef malloc
#undef calloc
#undef free
#endif
#include "websocket_nominal_fixture.h"
static NvmWebSocketFlowCounts counts(NvmWebSocketFlowState *s) {
    NvmWebSocketFlowCounts c;CHECK(nvm_websocket_flow_counts(s,&c));return c;
}
static void fixture_locals(Fixture *f,bool permute) {
    fixture(f,permute);f->function.arity=0;f->function.local_count=9;
    wr16(f->ownership+20,9);wr16(f->ownership+22,0);
    const int types[]={0,3,2,5,6,4,-1,-2,-3};
    for(unsigned i=0;i<9;i++) {
        int t=types[i];uint8_t tag=t==-1?TAG_STRING:t==-2?TAG_INT:t==-3?TAG_BOOL:t<3?TAG_STRUCT:TAG_UNION;
        descriptor(f->ownership+32+8*i,tag,0,t<0?NVM_V2_NO_INDEX:f->bindings.layouts[t]);
    }
    f->module.ownership_size=104;
}
static void message(NvmWebSocketFlowState *s) {
    OK(nvm_websocket_flow_push_scalar(s,TAG_BOOL));OK(nvm_websocket_flow_push_scalar(s,TAG_STRING));
    OK(nvm_websocket_flow_construct(s,2,0));
}
static void lifecycle(bool permute) {
    Fixture f;fixture_locals(&f,permute);NvmWebSocketFlowDeclarations *d=NULL;
    OK(nvm_websocket_flow_declarations(&f.module,&d));NvmWebSocketFlowState *s=NULL;
    OK(nvm_websocket_flow_state(d,0,&s));
    NvmWebSocketNominalBindings bindings=f.bindings;
    memset(&f,0,sizeof f);nvm_websocket_flow_declarations_free(d);
    OK(nvm_websocket_flow_push_scalar(s,TAG_STRING));
    CHECK(nvm_websocket_flow_service(s,1,bindings.imports[0],NVM_WEBSOCKET_FLOW_NO_REFERENCE)==NVM_WEBSOCKET_FLOW_INVALID);
    CHECK(counts(s).stack==1 && !counts(s).owners && !counts(s).obligations);
    OK(nvm_websocket_flow_push_scalar(s,TAG_INT));
    OK(nvm_websocket_flow_service(s,1,bindings.imports[0],NVM_WEBSOCKET_FLOW_NO_REFERENCE));
    NvmWebSocketFlowObligation o;CHECK(nvm_websocket_flow_obligation(s,0,&o));
    CHECK(o.parameters==2 && (o.checks & NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT));
    CHECK(o.acquired_rights && !o.owned_inputs && !o.borrowed_inputs);
    OK(nvm_websocket_flow_put(s,1));NvmWebSocketFlowState *a=NULL,*e=NULL;
    OK(nvm_websocket_flow_refine(s,1,&a,&e));nvm_websocket_flow_state_free(s);
    OK(nvm_websocket_flow_take_result(e,1,NVM_WEBSOCKET_FLOW_ARM_ERROR));
    CHECK(!counts(e).owners);OK(nvm_websocket_flow_field(e,0));OK(nvm_websocket_flow_pop(e));
    OK(nvm_websocket_flow_push_scalar(e,TAG_INT));OK(nvm_websocket_flow_can_exit(e));nvm_websocket_flow_state_free(e);
    s=a;OK(nvm_websocket_flow_take_result(s,1,NVM_WEBSOCKET_FLOW_ARM_OK));OK(nvm_websocket_flow_put(s,0));
    OK(nvm_websocket_flow_region_begin(s));OK(nvm_websocket_flow_borrow(s,0,20));
    message(s);OK(nvm_websocket_flow_push_scalar(s,TAG_BOOL));
    CHECK(nvm_websocket_flow_service(s,2,bindings.imports[1],20)==NVM_WEBSOCKET_FLOW_INVALID);
    CHECK(counts(s).stack==2 && counts(s).owners==1 && counts(s).references==1);
    OK(nvm_websocket_flow_pop(s));OK(nvm_websocket_flow_push_scalar(s,TAG_INT));
    OK(nvm_websocket_flow_service(s,2,bindings.imports[1],20));OK(nvm_websocket_flow_pop(s));
    CHECK(nvm_websocket_flow_obligation(s,1,&o));
    CHECK(o.parameters==3 && o.borrowed_inputs==1 && !o.owned_inputs);
    CHECK((o.checks & NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT) && !(o.checks & NVM_WEBSOCKET_FLOW_CHECK_BYTE));
    CHECK(o.outcomes[0]==NVM_WEBSOCKET_FLOW_INPUT_PRESERVED && o.outcomes[1]==NVM_WEBSOCKET_FLOW_INPUT_PRESERVED);
    CHECK(nvm_websocket_flow_service(s,3,bindings.imports[2],20)==NVM_WEBSOCKET_FLOW_INVALID);
    OK(nvm_websocket_flow_push_scalar(s,TAG_INT));OK(nvm_websocket_flow_service(s,3,bindings.imports[2],20));
    OK(nvm_websocket_flow_store(s,3));a=e=NULL;OK(nvm_websocket_flow_refine(s,3,&a,&e));
    OK(nvm_websocket_flow_take_result(a,3,NVM_WEBSOCKET_FLOW_ARM_OK));
    OK(nvm_websocket_flow_field(a,1));NvmWebSocketFlowValue v;
    CHECK(nvm_websocket_flow_stack(a,0,&v) && v.type.tag==TAG_STRING && !v.owner);
    nvm_websocket_flow_state_free(a);nvm_websocket_flow_state_free(e);
    OK(nvm_websocket_flow_clear_copy(s,3));OK(nvm_websocket_flow_region_end(s));
    OK(nvm_websocket_flow_take(s,0));
    CHECK(nvm_websocket_flow_service(s,4,bindings.imports[3],NVM_WEBSOCKET_FLOW_NO_REFERENCE)==NVM_WEBSOCKET_FLOW_INVALID);
    CHECK(counts(s).owners==1 && counts(s).stack==1);
    OK(nvm_websocket_flow_push_scalar(s,TAG_INT));OK(nvm_websocket_flow_service(s,4,bindings.imports[3],NVM_WEBSOCKET_FLOW_NO_REFERENCE));
    CHECK(!counts(s).owners && counts(s).stack==1);
    CHECK(nvm_websocket_flow_obligation(s,3,&o) && o.owned_inputs==1 && !o.borrowed_inputs);
    CHECK(o.outcomes[0]==NVM_WEBSOCKET_FLOW_INPUT_CONSUMED && o.outcomes[1]==NVM_WEBSOCKET_FLOW_INPUT_CONSUMED);
    CHECK(o.checks & NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT);
    OK(nvm_websocket_flow_pop(s));OK(nvm_websocket_flow_push_scalar(s,TAG_INT));OK(nvm_websocket_flow_can_exit(s));
    nvm_websocket_flow_state_free(s);
}
static void records_and_refusals(void) {
    Fixture f;fixture_locals(&f,false);NvmWebSocketFlowDeclarations *d=NULL;OK(nvm_websocket_flow_declarations(&f.module,&d));
    NvmWebSocketFlowState *s=NULL;OK(nvm_websocket_flow_state(d,0,&s));
    OK(nvm_websocket_flow_push_scalar(s,TAG_BOOL));OK(nvm_websocket_flow_push_scalar(s,TAG_INT));
    CHECK(nvm_websocket_flow_construct(s,2,0)==NVM_WEBSOCKET_FLOW_INVALID && counts(s).stack==2);
    OK(nvm_websocket_flow_pop(s));OK(nvm_websocket_flow_push_scalar(s,TAG_STRING));OK(nvm_websocket_flow_construct(s,2,0));
    OK(nvm_websocket_flow_construct(s,5,0));OK(nvm_websocket_flow_store(s,3));
    OK(nvm_websocket_flow_take_result(s,3,NVM_WEBSOCKET_FLOW_ARM_OK));OK(nvm_websocket_flow_field(s,0));
    NvmWebSocketFlowValue v;CHECK(nvm_websocket_flow_stack(s,0,&v) && v.type.tag==TAG_BOOL);
    CHECK(nvm_websocket_flow_service(s,7,4,NVM_WEBSOCKET_FLOW_NO_REFERENCE)==NVM_WEBSOCKET_FLOW_INVALID);
#ifdef FLOW_INSTRUMENT
    unsigned before=live;refuse=true;NvmWebSocketFlowState *copy=(void *)&f,*saved=copy;
    CHECK(nvm_websocket_flow_clone(s,&copy)==NVM_WEBSOCKET_FLOW_MEMORY && copy==saved && live==before);
    NvmWebSocketFlowDeclarations *bad=(void *)&f,*original=bad;
    CHECK(nvm_websocket_flow_declarations(&f.module,&bad)==NVM_WEBSOCKET_FLOW_MEMORY && bad==original && live==before);
    refuse=false;
#endif
    nvm_websocket_flow_state_free(s);nvm_websocket_flow_declarations_free(d);
    /* I refuse an ordinary same-shaped Message as a declared catalog value. */
    descriptor(f.ownership+48,TAG_STRUCT,0,0);d=(void *)&f;
    NvmWebSocketFlowStatus refusal=nvm_websocket_flow_declarations(&f.module,&d);
    CHECK(refusal==NVM_WEBSOCKET_FLOW_INVALID && d==(void *)&f);
}
int main(void) {
    lifecycle(false);lifecycle(true);records_and_refusals();
#ifdef FLOW_INSTRUMENT
    CHECK(!live);
#endif
    printf("I passed %u WebSocket logical flow checks.\n",checks);return 0;
}
