/* I qualify singleton self-tail proof separately from execution. */
#define RA_TAIL_MAIN retained_tail_fixture_main
#include "test_record_array_tail.c"
#undef RA_TAIL_MAIN

static void self_component(Input *c,unsigned variant) {
    fixture(c,TAG_INT,1,TAG_STRUCT,false);
    uint8_t last[4096];uint32_t length=c->fn[2].code_length;
    CHECK(length<=sizeof last);memcpy(last,c->code+c->fn[2].code_offset,length);
    c->n=c->fn[1].code_offset;
    scalar(c,TAG_BOOL);size_t branch=c->n;arg32(c,OP_JMP_FALSE,0);
    arg16(c,OP_LOAD_LOCAL,0);arg32(c,OP_TAIL_CALL,2);
    size_t at=branch+1;b32(c->code,&at,(uint32_t)(c->n-branch));
    if(variant==4) {
        arg16(c,OP_LOAD_LOCAL,0);arg16(c,OP_AGG_GET,0);scalar(c,TAG_BOOL);op(c,OP_ARR_PUSH);op(c,OP_POP);
    }
    if(variant==5)arg32(c,OP_LOAD_GLOBAL,1);
    else arg16(c,OP_LOAD_LOCAL,0);
    arg32(c,OP_TAIL_CALL,1);
    /* Even an unreachable ordinary self call participates in the graph. */
    if(variant==1){arg16(c,OP_LOAD_LOCAL,0);arg32(c,OP_CALL,1);op(c,OP_RET);}
    c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;
    c->fn[2].code_offset=(uint32_t)c->n;
    if(variant==2 || variant==3) {
        arg16(c,OP_LOAD_LOCAL,0);arg32(c,variant==2?OP_CALL:OP_TAIL_CALL,1);
        if(variant==2)op(c,OP_RET);
    } else {CHECK(c->n+length<sizeof c->code);memcpy(c->code+c->n,last,length);c->n+=length;}
    c->fn[2].code_length=(uint32_t)c->n-c->fn[2].code_offset;authority(c);
}
#ifdef RA_WHITEBOX
static void self_work_boundary(void) {
    Input c;self_component(&c,0);NvmRecordArrayBudget budget={0};
    NvmRecordArrayStructure *structure=NULL;
    CHECK(nvm_record_array_structure_prepare(&c.m,&budget,&structure).status==NVM_ARRAY_ELIGIBLE);
    Analysis measured={0};measured.module=&c.m;measured.structure=structure;
    CHECK(tail_prepare(&measured));uint64_t cost=measured.budget.work;CHECK(cost>0);
    for(unsigned over=0;over<2;over++) {
        Analysis a={0};a.module=&c.m;a.structure=structure;a.budget.work=NVM_RA_STEPS-cost+over;
        int ok=tail_prepare(&a);
        CHECK(over?(!ok && a.budget.limited):(ok && a.budget.work==NVM_RA_STEPS));
    }
    Analysis recovery={0};recovery.module=&c.m;recovery.structure=structure;
    CHECK(tail_prepare(&recovery) && recovery.budget.work==cost);
    nvm_record_array_structure_free(structure);CHECK(!ra_live && !ra_bytes);
    printf("I checked the exact self-tail graph work boundary: %llu steps.\n",(unsigned long long)cost);
}
#endif
int main(void) {
#ifdef RA_WHITEBOX
    self_work_boundary();
    boundaries();
#endif
    Input c;self_component(&c,0);
    NvmRecordArrayOrigins *origins=query(&c,NVM_ARRAY_ELIGIBLE);nvm_record_array_origins_free(origins);
    for(unsigned variant=1;variant<=5;variant++) {
        fprintf(stderr,"I test self-tail negative %u\n",variant);self_component(&c,variant);origins=query(&c,NVM_ARRAY_UNRESOLVED);CHECK(origins==(void *)(uintptr_t)1);
    }
    printf("PASS %u singleton self-tail proof checks\n",checks);return 0;
}
