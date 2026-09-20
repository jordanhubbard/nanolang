/* I inspect global-flow facts only; no program is executed here. */
#define main retained_record_array_origin_controls
#include "test_record_array_origins.c"
#undef main

static void branch_to(Input *c,size_t instruction,size_t destination){
    size_t at=instruction+1;b32(c->code,&at,(uint32_t)(destination-instruction));
}
static void test_tag(Input *c,uint8_t t){op(c,OP_TYPE_CHECK);b8(c->code,&c->n,t);}
static void install_global(Input *c,unsigned slot){arg32(c,OP_LOAD_GLOBAL,slot);record(c,0);op(c,OP_POP);}
static void put_array(Input *c,unsigned slot){array(c,TAG_INT);arg32(c,OP_STORE_GLOBAL,slot);}
static void helper_begin(Input *c,unsigned f,bool initializer){
    CHECK(f<3);c->m.function_count=f+1;c->fn[f].name_idx=initializer?3:2;
    c->fn[f].code_offset=(uint32_t)c->n;c->fn[f].result_tag=TAG_VOID;
}
static void helper_end(Input *c,unsigned f){c->fn[f].code_length=(uint32_t)c->n-c->fn[f].code_offset;authority(c);}
static NvmRecordArrayOrigins *global_query(Input *c,NvmArrayEligibilityStatus expected){
    Input before;memcpy(&before,c,sizeof before);
    NvmRecordArrayOrigins *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult r=nvm_analyze_record_array_origins(&c->m,&p);
    CHECK(!memcmp(c,&before,sizeof before));
    if(r.status!=expected)fprintf(stderr,"I expected global status %u, got %u at %u:%u: %s\n",expected,r.status,r.function,r.pc,r.message);
    CHECK(r.status==expected);
    CHECK(expected==NVM_ARRAY_ELIGIBLE?p!=(void *)(uintptr_t)1:p==(void *)(uintptr_t)1);
    return p;
}
static void global_accept(Input *c){nvm_record_array_origins_free(global_query(c,NVM_ARRAY_ELIGIBLE));}

static void dominating_and_joined_stores(void){
    Input c;init(&c,TAG_INT,true);put_array(&c,0);
    arg32(&c,OP_LOAD_GLOBAL,0);scalar(&c,TAG_INT);scalar(&c,TAG_INT);op(&c,OP_ARR_SLICE);
    arg32(&c,OP_STORE_GLOBAL,2);install_global(&c,2);finish(&c);
    NvmRecordArrayOrigins *p=global_query(&c,NVM_ARRAY_ELIGIBLE);
    NvmRecordValueOrigins value;CHECK(nvm_record_array_field_value(p,0,&value));
    CHECK(value.tags==MASK(TAG_ARRAY)&&value.origins==4&&!value.unknown);
    NvmRecordHeapOrigin original,owner,copy;
    CHECK(nvm_record_array_origin(p,0,&original)&&original.kind==NVM_HEAP_ORIGIN_ARRAY);
    CHECK(nvm_record_array_origin(p,1,&owner)&&owner.kind==NVM_HEAP_ORIGIN_RECORD);
    CHECK(nvm_record_array_origin(p,2,&copy)&&copy.kind==NVM_HEAP_ORIGIN_ARRAY);
    nvm_record_array_origins_free(p);
    init(&c,TAG_INT,false);install_global(&c,0);finish(&c);global_query(&c,NVM_ARRAY_UNRESOLVED);
    for(unsigned other=0;other<3;other++){
        init(&c,TAG_INT,false);scalar(&c,TAG_BOOL);size_t split=c.n;arg32(&c,OP_JMP_FALSE,0);
        put_array(&c,0);size_t skip=c.n;arg32(&c,OP_JMP,0);branch_to(&c,split,c.n);
        if(other==1)put_array(&c,0);
        if(other==2){scalar(&c,TAG_STRING);arg32(&c,OP_STORE_GLOBAL,0);}
        branch_to(&c,skip,c.n);install_global(&c,0);finish(&c);
        if(other==1)global_accept(&c);else global_query(&c,NVM_ARRAY_UNRESOLVED);
    }
}
static void calls_and_contexts(void){
    Input c;init(&c,TAG_INT,false);arg32(&c,OP_CALL,1);install_global(&c,0);finish(&c);
    helper_begin(&c,1,false);put_array(&c,0);op(&c,OP_RET);helper_end(&c,1);global_accept(&c);
    /* One no-write callee sees ARRAY and STRING contexts. Its joined entry
     * must not replace an untouched caller slot with that broader union. */
    init(&c,TAG_INT,false);put_array(&c,0);arg32(&c,OP_CALL,1);install_global(&c,0);
    scalar(&c,TAG_STRING);arg32(&c,OP_STORE_GLOBAL,0);arg32(&c,OP_CALL,1);
    put_array(&c,0);arg32(&c,OP_CALL,1);install_global(&c,0);finish(&c);
    helper_begin(&c,1,false);arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_POP);op(&c,OP_RET);helper_end(&c,1);global_accept(&c);
    /* A may-writing callee really joins unchanged ARRAY and written STRING. */
    init(&c,TAG_INT,false);put_array(&c,0);arg32(&c,OP_CALL,1);install_global(&c,0);finish(&c);
    helper_begin(&c,1,false);scalar(&c,TAG_BOOL);size_t split=c.n;arg32(&c,OP_JMP_FALSE,0);
    scalar(&c,TAG_STRING);arg32(&c,OP_STORE_GLOBAL,0);branch_to(&c,split,c.n);op(&c,OP_RET);helper_end(&c,1);
    global_query(&c,NVM_ARRAY_UNRESOLVED);
    /* A returning writer on only one caller edge leaves initial VOID on
     * the no-call edge. A callee summary cannot initialize that other path. */
    init(&c,TAG_INT,false);scalar(&c,TAG_BOOL);split=c.n;arg32(&c,OP_JMP_FALSE,0);
    arg32(&c,OP_CALL,1);branch_to(&c,split,c.n);install_global(&c,0);finish(&c);
    helper_begin(&c,1,false);put_array(&c,0);op(&c,OP_RET);helper_end(&c,1);global_query(&c,NVM_ARRAY_UNRESOLVED);
    /* A never-called writer cannot turn a root's initial VOID into ARRAY. */
    init(&c,TAG_INT,false);install_global(&c,0);finish(&c);
    helper_begin(&c,1,false);put_array(&c,0);op(&c,OP_RET);helper_end(&c,1);global_query(&c,NVM_ARRAY_UNRESOLVED);
    /* Recursive normal exits establish ARRAY; the recursive edge alone does
     * not manufacture a return before the independent base exit exists. */
    init(&c,TAG_INT,false);scalar(&c,TAG_INT);arg32(&c,OP_CALL,1);install_global(&c,0);finish(&c);
    helper_begin(&c,1,false);c.fn[1].local_count=c.fn[1].arity=1;c.params[0]=TAG_INT;
    c.param_rows[1]=c.params;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_CAST_BOOL);split=c.n;arg32(&c,OP_JMP_FALSE,0);
    put_array(&c,0);op(&c,OP_RET);branch_to(&c,split,c.n);
    arg16(&c,OP_LOAD_LOCAL,0);arg32(&c,OP_CALL,1);op(&c,OP_RET);helper_end(&c,1);global_accept(&c);
}
static void initialization_and_committed_prefixes(void){
    Input c;
    for(unsigned returns=0;returns<2;returns++){
        init(&c,TAG_INT,false);install_global(&c,0);finish(&c);
        helper_begin(&c,1,true);put_array(&c,0);
        if(returns)op(&c,OP_RET);else arg32(&c,OP_JMP,0);
        helper_end(&c,1);
        if(returns)global_accept(&c);else global_query(&c,NVM_ARRAY_UNRESOLVED);
    }
    for(unsigned corrupt=0;corrupt<2;corrupt++){
        init(&c,TAG_INT,false);arg32(&c,OP_LOAD_GLOBAL,0);test_tag(&c,TAG_VOID);
        size_t ready=c.n;arg32(&c,OP_JMP_FALSE,0);put_array(&c,0);branch_to(&c,ready,c.n);
        install_global(&c,0);
        if(corrupt){scalar(&c,TAG_INT);arg32(&c,OP_STORE_GLOBAL,0);scalar(&c,TAG_BOOL);op(&c,OP_ASSERT);}
        finish(&c);
        if(corrupt)global_query(&c,NVM_ARRAY_UNRESOLVED);else global_accept(&c);
    }
    /* This is the unchanged retained-result fixture's initial VOID guard and
     * exact ARRAY result, expressed as a query without VM execution. */
    init(&c,TAG_INT,false);c.fn[0].result_tag=TAG_ARRAY;
    arg32(&c,OP_LOAD_GLOBAL,0);test_tag(&c,TAG_VOID);size_t ready=c.n;arg32(&c,OP_JMP_FALSE,0);
    put_array(&c,0);branch_to(&c,ready,c.n);arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_RET);
    c.fn[0].code_length=(uint32_t)c.n;authority(&c);global_accept(&c);
}
static void stale_predicates(void){
    for(unsigned writer=0;writer<4;writer++){
        Input c;init(&c,TAG_INT,false);put_array(&c,0);arg32(&c,OP_LOAD_GLOBAL,0);test_tag(&c,TAG_ARRAY);arg16(&c,OP_STORE_LOCAL,0);
        size_t join=0;
        if(writer==2){scalar(&c,TAG_BOOL);join=c.n;arg32(&c,OP_JMP_FALSE,0);}
        if(writer==1)arg32(&c,OP_CALL,1);
        else {scalar(&c,TAG_INT);arg32(&c,OP_STORE_GLOBAL,writer==3?1:0);}
        if(writer==2)branch_to(&c,join,c.n);
        arg16(&c,OP_LOAD_LOCAL,0);size_t skip=c.n;arg32(&c,OP_JMP_FALSE,0);
        install_global(&c,0);branch_to(&c,skip,c.n);finish(&c);
        if(writer==1){
            helper_begin(&c,1,false);arg32(&c,OP_CALL,2);op(&c,OP_RET);helper_end(&c,1);
            helper_begin(&c,2,false);scalar(&c,TAG_INT);arg32(&c,OP_STORE_GLOBAL,0);op(&c,OP_RET);helper_end(&c,2);
        }
        if(writer==3)global_accept(&c);else global_query(&c,NVM_ARRAY_UNRESOLVED);
    }
}
static void complete_indices_and_copied_results(void){
    for(unsigned slot=255;slot<=256;slot++){
        Input c;init(&c,TAG_INT,false);put_array(&c,0);arg32(&c,OP_CALL,1);install_global(&c,0);finish(&c);
        helper_begin(&c,1,false);scalar(&c,TAG_INT);arg32(&c,OP_STORE_GLOBAL,slot);op(&c,OP_RET);helper_end(&c,1);
        if(slot==255)global_accept(&c);else global_query(&c,NVM_ARRAY_LIMIT);
    }
    Input *c=malloc(sizeof *c);CHECK(c);init(c,TAG_INT,true);put_array(c,0);install_global(c,0);finish(c);
    NvmRecordEligibilityReport *old=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult before=nvm_analyze_managed_records(&c->m,&old);CHECK(before.status!=NVM_ARRAY_ELIGIBLE&&old==(void *)(uintptr_t)1);
    NvmRecordArrayOrigins *p=global_query(c,NVM_ARRAY_ELIGIBLE);old=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult after=nvm_analyze_managed_records(&c->m,&old);CHECK(before.status==after.status&&!strcmp(before.message,after.message)&&old==(void *)(uintptr_t)1);
    memset(c,0xa5,sizeof *c);free(c);NvmRecordArrayOriginCounts counts;CHECK(nvm_record_array_origins_counts(p,&counts));
    CHECK(counts.origins==2&&counts.fields==1&&counts.checked_field_writes==1);
    NvmRecordValueOrigins value;CHECK(nvm_record_array_field_value(p,0,&value)&&value.tags==MASK(TAG_ARRAY)&&value.origins==1&&!value.unknown);
    UNCHANGED(NvmRecordValueOrigins,nvm_record_array_field_value(p,counts.fields,&out));
    nvm_record_array_origins_free(p);
}
#ifdef RA_WHITEBOX
static void exact_internal_guards(void){
    Value source=tag(TAG_ARRAY);source.relation=1;Value destination={0};
    CHECK(global_merge(&destination,source)&&destination.relation==1);
    source.relation=2;CHECK(global_merge(&destination,source)&&!destination.relation);
    source.relation=1;CHECK(!global_merge(&destination,source)&&!destination.relation);
    source.relation_tag=TAG_ARRAY;CHECK(!global_plain(source).relation&&!global_plain(source).relation_tag);
    Analysis a={0};GlobalFlow flow={0};NvmRecordArrayStructure structure={0};
    a.structure=&structure;a.global_flow=&flow;a.global_count=2;Function *f=&a.functions[0];
    f->locals=2;f->stack=2;f->stride=6;Value state[6]={{0}};
    state[0]=tag(TAG_BOOL);state[0].relation=SLOTS+1;state[0].relation_tag=TAG_ARRAY;
    state[1]=tag(TAG_BOOL);state[1].relation=SLOTS+2;state[1].relation_tag=TAG_ARRAY;
    state[2]=tag(TAG_ARRAY);state[2].relation=1;state[3]=tag(TAG_ARRAY);state[3].relation=2;
    a.budget.work=NVM_RA_STEPS-6;size_t calls=ra_calls;
    CHECK(global_invalidates(&a,f,state,0)&&a.budget.work==NVM_RA_STEPS&&ra_calls==calls);
    CHECK(!state[0].relation&&!state[2].relation&&state[1].relation==SLOTS+2&&state[3].relation==2);
    a.budget.work=NVM_RA_STEPS-5;a.budget.limited=false;Value saved[6];memcpy(saved,state,sizeof saved);
    CHECK(!global_invalidates(&a,f,state,1)&&a.budget.limited&&!memcmp(saved,state,sizeof saved)&&ra_calls==calls);
    memset(&a.budget,0,sizeof a.budget);a.report.origin_count=2;
    a.origin_kind[0]=NVM_HEAP_ORIGIN_ARRAY;a.origin_kind[1]=NVM_HEAP_ORIGIN_RECORD;
    Value predicate=tag(TAG_BOOL);predicate.relation=SLOTS+1;predicate.relation_tag=TAG_ARRAY;
    state[4].tags=MASK(TAG_ARRAY)|MASK(TAG_STRUCT)|MASK(TAG_VOID);state[4].origins=3;
    a.budget.work=NVM_RA_STEPS-ORIGINS-2;int reachable=0;
    CHECK(global_refine(&a,f,state,predicate,1,&reachable)&&reachable&&state[4].tags==MASK(TAG_ARRAY)&&state[4].origins==1&&a.budget.work==NVM_RA_STEPS);
    a.budget.work=NVM_RA_STEPS-ORIGINS-1;memcpy(saved,state,sizeof saved);
    CHECK(!global_refine(&a,f,state,predicate,1,&reachable)&&a.budget.limited&&!memcmp(saved,state,sizeof saved));
    memset(&a.budget,0,sizeof a.budget);state[4].unknown=1;memcpy(saved,state,sizeof saved);
    CHECK(global_refine(&a,f,state,predicate,1,&reachable)&&reachable&&!memcmp(saved,state,sizeof saved));
    state[4].unknown=0;predicate.relation_tag=255;
    CHECK(global_refine(&a,f,state,predicate,1,&reachable)&&!reachable);
    CHECK(global_refine(&a,f,state,predicate,0,&reachable)&&reachable);
    CHECK(ra_calls==calls);
    /* Actual preallocation, scratch, exit-cell and scan limits: I use one
     * decoded global row, not a synthetic successful report. */
    NvmModule module={0};module.function_count=1;VmDecodedInstruction row={0};
    row.instruction.opcode=OP_STORE_GLOBAL;row.instruction.operands[0].u32=0;
    structure.decoded[0].instructions=&row;structure.decoded[0].instruction_count=1;
    const uint64_t bytes=sizeof(GlobalFlow)+sizeof(Value)+3u*SLOTS*3u*sizeof(Value);
    const uint64_t work=sizeof(GlobalFlow)+5;
    for(unsigned boundary=0;boundary<3;boundary++)for(unsigned over=0;over<2;over++){
        memset(&a,0,sizeof a);a.module=&module;a.structure=&structure;
        if(boundary==0)a.budget.bytes=NVM_RA_BYTES-bytes+over;
        if(boundary==1)a.budget.work=NVM_RA_STEPS-work+over;
        if(boundary==2)a.state_cells=CELLS-1+over;
        int ok=global_prepare(&a);CHECK(over?(!ok&&a.result.status==NVM_ARRAY_LIMIT):ok);
        if(!over){CHECK(a.global_count==1&&a.global_flow->writes[0][0]==1);if(boundary==0)CHECK(a.budget.peak==NVM_RA_BYTES);if(boundary==1)CHECK(a.budget.work==NVM_RA_STEPS);if(boundary==2)CHECK(a.state_cells==CELLS);}
        if(a.global_flow){ra_test_free(a.global_flow->exits);ra_test_free(a.global_flow);}CHECK(!ra_live&&!ra_bytes);
    }
}
static void allocation_prefixes(void){
    Input c;init(&c,TAG_INT,true);arg32(&c,OP_CALL,1);install_global(&c,0);finish(&c);
    helper_begin(&c,1,false);put_array(&c,0);op(&c,OP_RET);helper_end(&c,1);
    CHECK(!ra_live&&!ra_bytes);ra_calls=ra_peak=0;
    NvmRecordArrayOrigins *p=global_query(&c,NVM_ARRAY_ELIGIBLE);size_t positions=ra_calls,peak=ra_peak;
    NvmRecordArrayOriginCounts counts;CHECK(nvm_record_array_origins_counts(p,&counts));CHECK(peak<=counts.peak_bytes_reserved);
    nvm_record_array_origins_free(p);CHECK(!ra_live&&!ra_bytes&&positions>20);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<positions;i++){
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;
        global_query(&c,NVM_ARRAY_MEMORY);CHECK(!ra_live&&!ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;global_accept(&c);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I measured %zu global allocation positions and %zu peak payload bytes; every exact MEMORY refusal recovered\n",positions,peak);
}
#endif
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);puts("I begin dominating and joined global stores");dominating_and_joined_stores();
    puts("I begin global calls and multiple contexts");calls_and_contexts();
    puts("I begin initializer and repeated committed prefixes");initialization_and_committed_prefixes();
    puts("I begin stale copied global predicates");stale_predicates();
    puts("I begin complete indices and copied results");complete_indices_and_copied_results();
#ifdef RA_WHITEBOX
    puts("I begin exact global-flow internal limits");exact_internal_guards();
    puts("I begin all global-flow allocation prefixes");allocation_prefixes();
#endif
    printf("PASS %u private global-flow checks; no runtime admission\n",checks);return 0;
}
