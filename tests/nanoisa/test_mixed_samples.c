/* I query pending modules only; no VM or generated program runs here. */
#define main previous_shape_fixture_main
#include "test_mixed_float_proof.c"
#undef main
#include "mixed_samples.h"

enum { ALLOC_SHAPE, ALLOC_FACTS, ALLOC_LAYOUT, ALLOC_CURSOR, ALLOC_VIEW, ALLOC_DOMAINS };
static bool tracking;
static long composed_budget=-1;
static unsigned allocs[ALLOC_DOMAINS],failed[ALLOC_DOMAINS],live_allocations;
static void *tracked[65536];
static void *tracked_allocate(unsigned domain,size_t n,size_t size,bool clear) {
    if(tracking){allocs[domain]++;if(!composed_budget){failed[domain]++;return NULL;}if(composed_budget>0)composed_budget--;}
    void *p=clear?calloc(n,size):malloc(n);
    if(p && tracking){unsigned at=0;while(at<65536 && tracked[at])at++;CHECK(at<65536);tracked[at]=p;live_allocations++;}
    return p;
}
void mcs_test_free(void *p) {
    if(p)for(unsigned n=0;n<65536;n++)if(tracked[n]==p){tracked[n]=NULL;CHECK(live_allocations);live_allocations--;break;}
    free(p);
}
#define HOOK(name,domain) \
void *name##_malloc(size_t n){return tracked_allocate(domain,n,0,false);} \
void *name##_calloc(size_t n,size_t s){return tracked_allocate(domain,n,s,true);}
HOOK(mcs_shape,ALLOC_SHAPE)
HOOK(mcs_facts,ALLOC_FACTS)
HOOK(mcs_layout,ALLOC_LAYOUT)
HOOK(mcs_cursor,ALLOC_CURSOR)
HOOK(mcs_view,ALLOC_VIEW)
static NvmMixedSamplesProof *composition(NvmModule *m,NvmMixedShapeStatus wanted) {
    NvmMixedSamplesProof sentinel={0},*p=&sentinel;
    CHECK(!live_allocations);tracking=true;composed_budget=-1;
    NvmMixedShapeResult r=nvm_analyze_mixed_samples(m,&p);
    if(r.status!=wanted)fprintf(stderr,"composition wanted %d got %d f%u pc%u: %s\n",wanted,r.status,r.function,r.pc,r.message);
    CHECK(r.status==wanted);
    if(wanted==NVM_MIXED_SHAPE_PROVED){CHECK(p!=&sentinel && p->affine_checked && !p->runtime_admitted);return p;}
    CHECK(p==&sentinel && !live_allocations);tracking=false;return NULL;
}
static void finished(NvmMixedSamplesProof *p) {nvm_mixed_samples_proof_free(p);CHECK(!live_allocations);tracking=false;}
static const char *close_helper=".function close 1 1 0 int 1\nOWN_UNPACK_LOCAL 0\nRET\n.end\n.parameters 1 struct\n";
static const char *samples_body=
    "PUSH_I64 7\nOWN_PACK 0\nCALL 1\nPUSH_I64 7\nEQ\nASSERT\n"
    "PUSH_F64 1.5\nPUSH_F64 2.5\nARR_LITERAL 3 2\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\n"
    "LOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nSTORE_LOCAL 2\n"
    "LOAD_LOCAL 2\nPUSH_I64 1\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nASSERT\nPUSH_I64 0\nRET\n";
static void samples_and_allocations(void) {
    puts("I check complete mixed shape, independent consumption and allocation failures.");fflush(stdout);
    Type locals[]={{TAG_STRUCT,1},SCALAR(TAG_ARRAY),SCALAR(TAG_ARRAY)};
    NvmModule *m=build(samples_body,locals,3,close_helper,false);
    bool needs=false;NvmV2Result old=nvm_ownership_contracts_validate(m,&needs);CHECK(old!=NVM_V2_OK);CHECK(!nvm_verify(m).ok);
    NvmMixedSamplesProof *p=composition(m,NVM_MIXED_SHAPE_PROVED);
    CHECK(p->checked_functions==2 && p->checked_instructions>25 && p->visits>=p->checked_instructions);
    CHECK(p->max_stack[0]>=2 && p->max_stack[1]==1);
    CHECK(p->managed_count==1 && p->global_to_managed[0]==NVM_V2_NO_INDEX && p->global_to_managed[1]==0 && p->managed_to_global[0]==1);
    CHECK(p->global_to_managed[2]==NVM_V2_NO_INDEX && p->global_to_managed[3]==NVM_V2_NO_INDEX);
    CHECK(p->shape->view->source_to_global[1]==1 && p->shape->view->global_to_managed[1]==NVM_V2_NO_INDEX);
    CHECK(p->shape->view->classes[1]==NVM_MIXED_PENDING_ARRAY_PROOF && p->shape->requires_affine_verification);
    unsigned floats=0,comparisons=0;
    for(uint32_t n=0;n<p->check_count;n++) {
        NvmMixedScalarCheck *c=&p->checks[n];
        if(c->policy==NVM_MIXED_CHECK_FLOAT){floats++;CHECK(c->operand==0 && c->actual_tags==((1u<<TAG_FLOAT)|(1u<<TAG_VOID)) && c->required_tags==(1u<<TAG_FLOAT));}
        else {comparisons++;CHECK(c->actual_tags==(1u<<TAG_INT) && !c->required_tags);}
    }
    CHECK(floats==1 && comparisons==2);finished(p);
    uint8_t *code=malloc(m->code_size),*owned=malloc(m->ownership_size),*layouts=malloc(m->layout_size);CHECK(code&&owned&&layouts);
    memcpy(code,m->code,m->code_size);memcpy(owned,m->ownership_data,m->ownership_size);memcpy(layouts,m->layout_data,m->layout_size);
    memset(allocs,0,sizeof allocs);memset(failed,0,sizeof failed);unsigned refusals=0,successes=0;long first_success=-1;
    for(long n=0;n<2000 && successes<2;n++) {
        NvmMixedSamplesProof sentinel={0};p=&sentinel;tracking=true;composed_budget=n;
        NvmMixedShapeResult r=nvm_analyze_mixed_samples(m,&p);composed_budget=-1;
        if(r.status==NVM_MIXED_SHAPE_MEMORY){CHECK(p==&sentinel);refusals++;}
        else {
            if(r.status!=NVM_MIXED_SHAPE_PROVED)fprintf(stderr,"allocation budget%ld status%d %s\n",n,r.status,r.message);
            CHECK(r.status==NVM_MIXED_SHAPE_PROVED);CHECK(p!=&sentinel && !p->runtime_admitted);
            nvm_mixed_samples_proof_free(p);successes++;if(first_success<0)first_success=n;
        }
        CHECK(!live_allocations);tracking=false;
        CHECK(!memcmp(code,m->code,m->code_size) && !memcmp(owned,m->ownership_data,m->ownership_size) && !memcmp(layouts,m->layout_data,m->layout_size));
    }
    CHECK(successes==2 && refusals>50 && first_success>0);
    CHECK(failed[ALLOC_FACTS]>20 && failed[ALLOC_LAYOUT]>0 && failed[ALLOC_SHAPE]>0 && failed[ALLOC_VIEW]>0);
    printf("I swept all query allocation budgets through first success%ld; Facts/frame%u, layout-decode%u, shape%u, view%u failures.\n",first_success,failed[ALLOC_FACTS],failed[ALLOC_LAYOUT],failed[ALLOC_SHAPE],failed[ALLOC_VIEW]);
    CHECK(nvm_ownership_contracts_validate(m,&needs)==old && !nvm_verify(m).ok);
    free(code);free(owned);free(layouts);nvm_module_free(m);
}
static void scalar_obligations(void) {
    puts("I preserve operand-specific FLOAT-or-VOID and generic equality obligations.");fflush(stdout);
    for(unsigned mask=0;mask<4;mask++) {
        char body[1000];snprintf(body,sizeof body,"%s%sF64_EQ\nPOP\nPUSH_I64 0\nRET\n",
            mask&1?"ARR_NEW 3\nPUSH_I64 -1\nARR_GET\n":"PUSH_F64 1.0\n",
            mask&2?"ARR_NEW 3\nPUSH_I64 99\nARR_GET\n":"PUSH_F64 1.0\n");
        NvmModule *m=build(body,NULL,0,NULL,false);NvmMixedSamplesProof *p=composition(m,NVM_MIXED_SHAPE_PROVED);
        unsigned seen=0;
        for(uint32_t n=0;n<p->check_count;n++) {
            NvmMixedScalarCheck c=p->checks[n];CHECK(c.policy==NVM_MIXED_CHECK_FLOAT && c.operand<2);
            CHECK(c.actual_tags==((1u<<TAG_FLOAT)|(1u<<TAG_VOID)) && c.required_tags==(1u<<TAG_FLOAT));seen|=1u<<c.operand;
        }
        CHECK(seen==mask && p->check_count==(unsigned)(((mask&1)!=0)+((mask&2)!=0)));finished(p);nvm_module_free(m);
    }
    for(unsigned ne=0;ne<2;ne++)for(unsigned order=0;order<2;order++) {
        char body[1000];snprintf(body,sizeof body,"%s%s%s\nPOP\nPUSH_I64 0\nRET\n",
            order?"PUSH_F64 1.0\n":"ARR_NEW 3\nPUSH_I64 0\nARR_GET\n",
            order?"ARR_NEW 3\nPUSH_I64 0\nARR_GET\n":"PUSH_F64 1.0\n",ne?"NE":"EQ");
        NvmModule *m=build(body,NULL,0,NULL,false);NvmMixedSamplesProof *p=composition(m,NVM_MIXED_SHAPE_PROVED);
        CHECK(p->check_count==2);
        for(unsigned n=0;n<2;n++){NvmMixedScalarCheck c=p->checks[n];CHECK(c.policy==NVM_MIXED_COMPARE_VALUE && !c.required_tags && c.operand==n);CHECK(c.actual_tags==((1u<<TAG_FLOAT)|(n==order?(1u<<TAG_VOID):0)));}
        finished(p);nvm_module_free(m);
    }
    const char *refused[]={
        "ARR_NEW 3\nPUSH_I64 0\nARR_GET\nPUSH_F64 1.0\nLT\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_I64 1\nPUSH_F64 1.0\nEQ\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_I64 1\nPUSH_I64 1\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_VOID\nPUSH_F64 1.0\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_I64 1\nASSERT\nPUSH_I64 0\nRET\n"};
    for(unsigned n=0;n<sizeof refused/sizeof *refused;n++) {
        NvmModule *m=build(refused[n],NULL,0,NULL,false);
        NvmMixedFloatProof *shape=expect(m,NVM_MIXED_SHAPE_PROVED);nvm_mixed_float_proof_free(shape);
        composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
    }
    NvmModule *m=build("ARR_NEW 3\nARR_NEW 3\nEQ\nPOP\nPUSH_I64 0\nRET\n",NULL,0,NULL,false);
    composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
}
static void independent_owner_checks(void) {
    puts("I check independent owner exits and reordered observations.");fflush(stdout);
    Type owner[]={{TAG_STRUCT,0}};
    NvmModule *m=build("PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_I64 0\nRET\n",owner,1,NULL,false);
    NvmMixedFloatProof *shape=expect(m,NVM_MIXED_SHAPE_PROVED);nvm_mixed_float_proof_free(shape);
    composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
    m=build("PUSH_I64 7\nOWN_PACK 0\nCALL 1\nRET\n",NULL,0,
        ".function leak 1 1 0 int 1\nPUSH_I64 0\nRET\n.end\n.parameters 1 struct\n",false);
    shape=expect(m,NVM_MIXED_SHAPE_PROVED);nvm_mixed_float_proof_free(shape);composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
    for(unsigned swap=0;swap<2;swap++) {
        char body[1200];snprintf(body,sizeof body,"PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 9\n%sOWN_MOVE_LOCAL 0\nCALL 1\nPOP\nPOP\nPOP\nPUSH_I64 0\nRET\n",swap?"SWAP\n":"");
        m=build(body,owner,1,close_helper,false);composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
    }
    m=build("PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nAGG_GET 0\nPOP\nOWN_MOVE_LOCAL 0\nCALL 1\nRET\n",owner,1,close_helper,false);
    finished(composition(m,NVM_MIXED_SHAPE_PROVED));nvm_module_free(m);
    for(unsigned order=0;order<2;order++) {
        char body[1200];snprintf(body,sizeof body,"PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_BOOL %u\nJMP_FALSE keep\nOWN_MOVE_LOCAL 0\nCALL 1\nPOP\nJMP join\nkeep:\nNOP\njoin:\nPUSH_I64 0\nRET\n",order);
        m=build(body,owner,1,close_helper,false);composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
    }
    /* An unused helper's live owner is independently checked too. */
    m=build("PUSH_I64 0\nRET\n",NULL,0,".function leak 1 1 0 int 1\nPUSH_I64 0\nRET\n.end\n.parameters 1 struct\n",false);
    composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
}
static void ordinary_joins_and_calls(void) {
    puts("I check zero-iteration initialization, full receivers and lower-index callees.");fflush(stdout);
    Type array[]={SCALAR(TAG_ARRAY)};
    const char *refused[]={"LOAD_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",
        "loop:\nPUSH_BOOL 0\nJMP_FALSE done\nARR_NEW 3\nSTORE_LOCAL 0\nJMP loop\ndone:\nLOAD_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n"};
    for(unsigned n=0;n<2;n++){NvmModule *m=build(refused[n],array,1,NULL,false);composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);}
    NvmModule *m=build("ARR_NEW 3\nSTORE_LOCAL 0\nloop:\nPUSH_BOOL 0\nJMP_FALSE done\nLOAD_LOCAL 0\nPUSH_F64 1.0\nARR_PUSH\nSTORE_LOCAL 0\nJMP loop\ndone:\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPOP\nPUSH_I64 0\nRET\n",array,1,NULL,false);
    finished(composition(m,NVM_MIXED_SHAPE_PROVED));nvm_module_free(m);
    for(unsigned order=0;order<2;order++) {
        char body[1200];snprintf(body,sizeof body,"PUSH_BOOL 1\nJMP_FALSE other\nARR_NEW 3\nAGG_PACK 0 %u 0 1\nJMP read\nother:\nARR_NEW 3\nAGG_PACK 0 %u 0 1\nread:\nAGG_GET 0\nPUSH_I64 1\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nPOP\nPUSH_I64 0\nRET\n",1+order,2-order);
        m=build(body,NULL,0,NULL,false);NvmMixedSamplesProof *p=composition(m,NVM_MIXED_SHAPE_PROVED);
        CHECK(p->managed_count==2 && p->global_to_managed[1]==0 && p->global_to_managed[2]==1 && p->global_to_managed[0]==NVM_V2_NO_INDEX);finished(p);nvm_module_free(m);
    }
    m=build("CALL 2\nRET\n",NULL,0,
        ".function leaf 0 0 0 int 1\nARR_NEW 3\nPUSH_I64 0\nARR_GET\nPOP\nPUSH_I64 0\nRET\n.end\n"
        ".function middle 0 0 0 int 1\nCALL 1\nRET\n.end\n",false);
    NvmMixedSamplesProof *p=composition(m,NVM_MIXED_SHAPE_PROVED);CHECK(p->checked_functions==3);nvm_module_free(m);CHECK(p->shape->origins[0].function==1);finished(p);
    m=build("ARR_NEW 3\nAGG_PACK 0 2 0 1\nAGG_PACK 0 3 0 1\nPOP\nPUSH_I64 0\nRET\n",NULL,0,NULL,false);composition(m,NVM_MIXED_SHAPE_UNRESOLVED);nvm_module_free(m);
}
int main(void) {
    samples_and_allocations();scalar_obligations();independent_owner_checks();ordinary_joins_and_calls();
    CHECK(!tracking && !live_allocations);
    printf("%u mixed Samples composition checks passed; no pending module execution\n",checks);return 0;
}
