/* I copy historical obligations without accepting imported runtime objects. */
#include "../../src/nanoisa/managed_record_map_boundary.h"
int g_argc;char **g_argv;
static NvmRecordMapBoundaryPlan *original_boundary;
static void boundary_original(const NvmModule *);
#define MAP_ORIGINAL_CHECK boundary_original
#define MAP_ORIGIN_MAIN old_map_boundary_origin_main
#include "test_record_map_origins.c"
static NvmRecordMapBoundaryPlan *boundary_prepare(const NvmModule *m) {
    NvmRecordMapBoundaryPlan *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult r=nvm_prepare_record_map_boundary(m,&p);
    if(r.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I refused boundary: %s\n",r.message);
    CHECK(r.status==NVM_ARRAY_ELIGIBLE&&p&&p!=(void *)(uintptr_t)1);return p;
}
static void boundary_original(const NvmModule *m) {
    NvmRecordMapBoundaryPlan *p=boundary_prepare(m);NvmRecordMapBoundaryCounts count;
    CHECK(nvm_record_map_boundary_counts(p,&count));CHECK(count.globals==0);
    CHECK(nvm_record_map_boundary_execution(p));map_public_refused(m);
    original_boundary=p;
}
static void boundary_input(Input *c,unsigned slot) {
    init(c,TAG_INT,false);map_new(c,TAG_STRING,TAG_INT);record(c,0);
    arg32(c,OP_STORE_GLOBAL,slot);
    /* A second allocation at the same nominal layout remains a different
     * historical origin even when it replaces the current global value. */
    map_new(c,TAG_STRING,TAG_INT);record(c,0);arg32(c,OP_STORE_GLOBAL,slot);
    map_finish(c);
}
static void boundary_rows(unsigned slot) {
    Input c;boundary_input(&c,slot);
    NvmRecordMapBoundaryPlan *p=boundary_prepare(&c.m);map_public_refused(&c.m);
    NvmRecordMapBoundaryCounts count;CHECK(nvm_record_map_boundary_counts(p,&count));CHECK(count.globals==slot+1);
    NvmRecordValueOrigins row,saved;memset(&row,0x5a,sizeof row);saved=row;
    CHECK(!nvm_record_map_boundary_global(p,slot+1,&row));CHECK(!memcmp(&row,&saved,sizeof row));
    for(unsigned i=0;i<slot;i++){CHECK(nvm_record_map_boundary_global(p,i,&row));CHECK(row.tags==MASK(TAG_VOID)&&!row.origins&&!row.unknown);}
    CHECK(nvm_record_map_boundary_global(p,slot,&row));
    CHECK(row.tags==(MASK(TAG_VOID)|MASK(TAG_STRUCT))&&row.origins==10&&!row.unknown);
    const NvmRecordMapExecutionPlan *e=nvm_record_map_boundary_execution(p);
    NvmRecordValueOrigins field;CHECK(nvm_record_map_execution_field_value(e,0,&field));CHECK(field.origins==1&&field.tags==MASK(TAG_HASHMAP));
    CHECK(nvm_record_map_execution_field_value(e,1,&field));CHECK(field.origins==4&&field.tags==MASK(TAG_HASHMAP));
    memset(&c,0,sizeof c);CHECK(nvm_record_map_boundary_global(p,slot,&field));CHECK(!memcmp(&row,&field,sizeof row));
    nvm_record_map_boundary_free(p);
}
static void boundary_failures(void) {
    Input c;boundary_input(&c,256);NvmRecordMapBoundaryPlan *p=(void *)(uintptr_t)1;
    CHECK(nvm_prepare_record_map_boundary(&c.m,&p).status!=NVM_ARRAY_ELIGIBLE);CHECK(p==(void *)(uintptr_t)1);
    init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);arg32(&c,OP_STORE_GLOBAL,0);map_finish(&c);
    CHECK(nvm_prepare_record_map_boundary(&c.m,&p).status==NVM_ARRAY_UNRESOLVED);CHECK(p==(void *)(uintptr_t)1);
    CHECK(nvm_prepare_record_map_boundary(NULL,&p).status==NVM_ARRAY_INVALID);CHECK(p==(void *)(uintptr_t)1);
    CHECK(!nvm_record_map_boundary_counts(NULL,NULL));CHECK(!nvm_record_map_boundary_global(NULL,0,NULL));
    CHECK(!nvm_record_map_boundary_execution(NULL));nvm_record_map_boundary_free(NULL);
}
static void boundary_typed_history(void) {
    Input c;init(&c,TAG_INT,false);
    const uint8_t tags[]={TAG_VOID,TAG_INT,TAG_FLOAT,TAG_BOOL,TAG_U8,TAG_STRING};
    for(unsigned i=0;i<sizeof tags;i++){scalar(&c,tags[i]);arg32(&c,OP_STORE_GLOBAL,i);}
    map_new(&c,TAG_STRING,TAG_INT);record(&c,0);op(&c,OP_DUP);
    arg32(&c,OP_STORE_GLOBAL,6);arg32(&c,OP_STORE_GLOBAL,7);map_finish(&c);
    NvmRecordMapBoundaryPlan *p=boundary_prepare(&c.m);NvmRecordValueOrigins row,shared;
    for(unsigned i=0;i<sizeof tags;i++) {
        CHECK(nvm_record_map_boundary_global(p,i,&row));
        CHECK(row.tags==(MASK(TAG_VOID)|MASK(tags[i]))&&!row.origins&&!row.unknown);
    }
    CHECK(nvm_record_map_boundary_global(p,6,&row));CHECK(nvm_record_map_boundary_global(p,7,&shared));
    CHECK(!memcmp(&row,&shared,sizeof row)&&row.origins==2);nvm_record_map_boundary_free(p);
    /* The complete proof must reject an unused unknown-parameter global store. */
    map_basic(&c);c.m.function_count=2;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].arity=1;c.fn[1].local_count=1;c.fn[1].result_tag=TAG_VOID;
    c.params[0]=TAG_VOID;c.param_rows[1]=c.params;
    arg16(&c,OP_LOAD_LOCAL,0);arg32(&c,OP_STORE_GLOBAL,0);op(&c,OP_RET);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
    p=(void *)(uintptr_t)1;CHECK(nvm_prepare_record_map_boundary(&c.m,&p).status==NVM_ARRAY_UNRESOLVED);
    CHECK(p==(void *)(uintptr_t)1);
#ifdef RA_WHITEBOX
    Value v=tag(TAG_STRUCT);v.origins=2;
    CHECK(map_boundary_value_valid(v,1,2));v.origins=1;CHECK(!map_boundary_value_valid(v,1,2));
    v=tag(TAG_ARRAY);v.origins=1;CHECK(map_boundary_value_valid(v,1,2));
    v.origins=4;CHECK(!map_boundary_value_valid(v,1,2));v.origins=0;CHECK(!map_boundary_value_valid(v,1,2));
    v=tag(TAG_INT);v.origins=1;CHECK(!map_boundary_value_valid(v,1,2));
    v.origins=0;v.unknown=1;CHECK(!map_boundary_value_valid(v,1,2));
    v=tag(TAG_HASHMAP);v.origins=4;CHECK(!map_boundary_value_valid(v,1,2));
    v=(Value){0};CHECK(!map_boundary_value_valid(v,1,2));
#endif
}
static void boundary_allocations(void) {
#ifdef RA_WHITEBOX
    Input c;boundary_input(&c,255);CHECK(!ra_live&&!ra_bytes);ra_calls=ra_peak=0;
    NvmRecordMapBoundaryPlan *p=boundary_prepare(&c.m);size_t calls=ra_calls;
    NvmRecordMapBoundaryCounts count;CHECK(nvm_record_map_boundary_counts(p,&count));CHECK(ra_peak<=count.peak_bytes_reserved);
    NvmMapCopyWork work={.limit=UINT64_MAX};uint64_t retained;
    CHECK(rme_plan_retained_cost(p->execution,&work,&retained)==NVM_MAP_COST_OK);
    CHECK(ra_bytes==retained+sizeof *p+256*sizeof *p->globals);
    printf("I measured boundary owner %zu, row %zu, retained %zu, prefixes %zu.\n",sizeof *p,sizeof *p->globals,ra_bytes,calls);
    nvm_record_map_boundary_free(p);CHECK(!ra_live&&!ra_bytes);
    NvmRecordMapBoundaryPlan *prior=boundary_prepare(&c.m);size_t prior_live=ra_live,prior_bytes=ra_bytes;
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<calls;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;p=prior;
        CHECK(nvm_prepare_record_map_boundary(&c.m,&p).status!=NVM_ARRAY_ELIGIBLE);CHECK(p==prior);
        CHECK(ra_live==prior_live&&ra_bytes==prior_bytes);
        NvmRecordValueOrigins row;CHECK(nvm_record_map_boundary_global(prior,255,&row));CHECK(row.origins==10);
        ra_fail=SIZE_MAX;ra_persistent=0;p=boundary_prepare(&c.m);nvm_record_map_boundary_free(p);
        CHECK(ra_live==prior_live&&ra_bytes==prior_bytes);
    }
    nvm_record_map_boundary_free(prior);CHECK(!ra_live&&!ra_bytes);
    uint64_t bytes=NVM_RECORD_ARRAY_EXECUTION_BYTES-count.peak_bytes_reserved;
    uint64_t steps=NVM_RECORD_ARRAY_EXECUTION_STEPS-count.work_reserved;
    p=NULL;CHECK(rmb_prepare_reserved(&c.m,&p,bytes,steps).status==NVM_ARRAY_ELIGIBLE);
    NvmRecordMapBoundaryCounts exact;CHECK(nvm_record_map_boundary_counts(p,&exact));
    CHECK(exact.peak_bytes_reserved==NVM_RECORD_ARRAY_EXECUTION_BYTES&&exact.work_reserved==NVM_RECORD_ARRAY_EXECUTION_STEPS);
    nvm_record_map_boundary_free(p);p=(void *)(uintptr_t)1;
    CHECK(rmb_prepare_reserved(&c.m,&p,bytes+1,steps).status==NVM_ARRAY_LIMIT&&p==(void *)(uintptr_t)1);
    CHECK(rmb_prepare_reserved(&c.m,&p,bytes,steps+1).status==NVM_ARRAY_LIMIT&&p==(void *)(uintptr_t)1);
    CHECK(rmb_prepare_reserved(&c.m,&p,UINT64_MAX,0).status==NVM_ARRAY_LIMIT&&p==(void *)(uintptr_t)1);
    CHECK(!ra_live&&!ra_bytes);p=boundary_prepare(&c.m);nvm_record_map_boundary_free(p);CHECK(!ra_live&&!ra_bytes);
    map_basic(&c);p=boundary_prepare(&c.m);CHECK(!p->globals&&!p->count);
    work=(NvmMapCopyWork){.limit=UINT64_MAX};
    CHECK(rme_plan_retained_cost(p->execution,&work,&retained)==NVM_MAP_COST_OK);
    CHECK(ra_bytes==retained+sizeof *p);nvm_record_map_boundary_free(p);CHECK(!ra_live&&!ra_bytes);
#endif
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);boundary_rows(0);boundary_rows(255);boundary_failures();boundary_typed_history();boundary_allocations();
    for(int i=1;i<argc;i++) {
        map_original_module(argv[i]); /* destroys the actual original module */
        NvmRecordMapBoundaryCounts row;CHECK(nvm_record_map_boundary_counts(original_boundary,&row));CHECK(!row.globals);
        NvmRecordMapExecutionCounts counts;CHECK(nvm_record_map_execution_counts(nvm_record_map_boundary_execution(original_boundary),&counts));
        CHECK(counts.carrier.functions>0);nvm_record_map_boundary_free(original_boundary);original_boundary=NULL;
    }
    printf("PASS %u copied global boundary checks; public admission remains refused\n",checks);return 0;
}
