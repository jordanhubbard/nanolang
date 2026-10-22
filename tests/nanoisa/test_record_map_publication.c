/* I query complete candidates without generating C or changing publishers. */
#include "../../src/nanoisa/record_map_publication.h"
#include "../../src/nanoisa/record_map_admission.h"
#include "../../src/nanoisa/record_map_native_target_internal.h"
static void publication_original(const NvmModule *);
#define MAP_ORIGINAL_CHECK publication_original
#define MAP_ORIGIN_MAIN publication_old_origins
#include "test_record_map_origins.c"
static void publication_original(const NvmModule *m) {
    CHECK(nvm_record_map_publication(m).status==NVM_ARRAY_ELIGIBLE);
}
static void publication_input(Input *c,unsigned variant) {
    init(c,TAG_INT,false);
    map_new(c,TAG_STRING,TAG_INT);op(c,OP_POP);
    scalar(c,TAG_STRING);
    if(variant==1)op(c,OP_STR_TRIM);
    if(variant==2){scalar(c,TAG_STRING);op(c,OP_STR_CONCAT);}
    op(c,OP_POP);if(variant==1){arg32(c,OP_PUSH_STR,2);op(c,OP_POP);}
    map_finish(c);
}
static void publication_cases(void) {
    for(unsigned v=0;v<3;v++) {
        Input c;publication_input(&c,v);Input saved=c;
        CHECK(nvm_verify(&c.m).ok);
        CHECK(nvm_record_map_publication(&c.m).status==(v==1?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_ELIGIBLE));
        CHECK(!memcmp(&c,&saved,sizeof c));
    }
    Input c;publication_input(&c,0);c.strings[1]="a\0b";c.lengths[1]=3;
    CHECK(nvm_verify(&c.m).ok);CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_UNRESOLVED);
    c.ownership[c.o-1]=255;
    CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_INVALID);
    CHECK(!nvm_verify(&c.m).ok);
    publication_input(&c,0);c.code[0]=OP_HM_DELETE;c.code[c.n-1]=255;
    CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_UNRESOLVED);
    CHECK(!nvm_verify(&c.m).ok); /* omission never grants execution authority */
    publication_input(&c,0);c.m.layout_size=1;c.m.ownership_size=0;c.m.ownership_data=NULL;
    CHECK(nvm_record_map_route(&c.m)==NVM_RECORD_MAP_DEFER);
    CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_UNRESOLVED);
    CHECK(!nvm_verify(&c.m).ok);
    CHECK(nvm_record_map_publication(NULL).status==NVM_ARRAY_INVALID);
    publication_input(&c,0);c.m.layout_data=NULL;c.m.layout_size=0;c.m.ownership_data=NULL;c.m.ownership_size=0;
    CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_UNRESOLVED);
}
static void publication_target(void) {
    Input c;publication_input(&c,1);
    NvmRecordMapExecutionPlan *p=NULL;NvmRecordMapImage *image=NULL;
    CHECK(nvm_record_map_admit(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
    CHECK(nvm_record_map_image_copy(p,&image).status==NVM_ARRAY_ELIGIBLE);
    nvm_record_map_execution_free(p);
    NvmRecordMapTargetResult r=nvm_record_map_native_target(image,UINT64_MAX);
    CHECK(r.eligibility.status==NVM_ARRAY_UNRESOLVED);CHECK(r.work>0);
    CHECK(nvm_record_map_native_target(image,r.work).eligibility.status==NVM_ARRAY_UNRESOLVED);
    CHECK(nvm_record_map_native_target(image,r.work-1).eligibility.status==NVM_ARRAY_LIMIT);
    CHECK(nvm_record_map_native_target(image,0).eligibility.status==NVM_ARRAY_LIMIT);
#ifdef RA_WHITEBOX
    uint32_t last=image->plan->carrier->counts.instructions-1;
    NvmRecordMapExecutionInstruction saved=image->plan->instructions[last];
    image->plan->instructions[last].base.recipe=NVM_RA_ROOT_CONSTANT;
    CHECK(nvm_record_map_native_target(image,UINT64_MAX).eligibility.status==NVM_ARRAY_INVALID);
    image->plan->instructions[last]=saved;
    image->plan->instructions[last-1].base.operand_bits[0]^=1;
    CHECK(nvm_record_map_native_target(image,UINT64_MAX).eligibility.status==NVM_ARRAY_INVALID);
    image->plan->instructions[last-1].base.operand_bits[0]^=1;
    CHECK(nvm_record_map_native_target(image,UINT64_MAX).eligibility.status==NVM_ARRAY_UNRESOLVED);
    char *string=image->plan->carrier->snapshot.strings[2];
    image->plan->carrier->snapshot.strings[2]=NULL;
    CHECK(nvm_record_map_native_target(image,UINT64_MAX).eligibility.status==NVM_ARRAY_INVALID);
    image->plan->carrier->snapshot.strings[2]=string;
    CHECK(nvm_record_map_native_target(image,UINT64_MAX).eligibility.status==NVM_ARRAY_UNRESOLVED);
    NvmRecordArrayExecutionDescriptor descriptor=image->plan->carrier->descriptors[0];
    image->plan->carrier->snapshot.functions[0].upvalue_count=1;
    image->plan->carrier->descriptors[0].fields++;
    CHECK(nvm_record_map_native_target(image,UINT64_MAX).eligibility.status==NVM_ARRAY_INVALID);
    image->plan->carrier->descriptors[0]=descriptor;
    image->plan->carrier->snapshot.functions[0].upvalue_count=0;
#endif
    memset(&c,0,sizeof c);
    CHECK(nvm_record_map_native_target(image,UINT64_MAX).eligibility.status==NVM_ARRAY_UNRESOLVED);
    nvm_record_map_image_free(image);
    publication_input(&c,0);p=NULL;image=NULL;
    CHECK(nvm_record_map_admit(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
    CHECK(nvm_record_map_image_copy(p,&image).status==NVM_ARRAY_ELIGIBLE);
    nvm_record_map_execution_free(p);
    r=nvm_record_map_native_target(image,UINT64_MAX);CHECK(r.eligibility.status==NVM_ARRAY_ELIGIBLE);
    CHECK(nvm_record_map_native_target(image,r.work).eligibility.status==NVM_ARRAY_ELIGIBLE);
    CHECK(nvm_record_map_native_target(image,r.work-1).eligibility.status==NVM_ARRAY_LIMIT);
    nvm_record_map_image_free(image);
}
static void publication_scan_limit(void) {
    Input c;init(&c,TAG_INT,false);
    const uint32_t length=1024u*1024u;char *literal=malloc((size_t)length+1);CHECK(literal);
    memset(literal,'x',length);literal[length]=0;c.strings[1]=literal;c.lengths[1]=length;
    for(unsigned i=0;i<40;i++){scalar(&c,TAG_STRING);op(&c,OP_POP);}
    map_finish(&c);NvmRecordMapExecutionPlan *p=NULL;
    CHECK(nvm_record_map_admit(&c.m,&p).status==NVM_ARRAY_ELIGIBLE);
    nvm_record_map_execution_free(p);
    CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_LIMIT);
    CHECK(c.strings[1]==literal&&c.lengths[1]==length&&literal[length]==0);
    free(literal);publication_input(&c,0);
    CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_ELIGIBLE);
    printf("I refused 40 executable references to one 1048576-byte literal at the publication scan limit.\n");
}
static void publication_allocations(void) {
#ifdef RA_WHITEBOX
    Input c;publication_input(&c,2);Input saved=c;
    size_t live=ra_live,bytes=ra_bytes;ra_calls=0;ra_peak=bytes;
    CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_ELIGIBLE);
    size_t positions=ra_calls;CHECK(positions>0);CHECK(ra_live==live&&ra_bytes==bytes);
    printf("I measured %zu publication allocations and %zu peak payload bytes.\n",positions,ra_peak-bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<positions;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;
        CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_MEMORY);
        ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(ra_live==live&&ra_bytes==bytes);CHECK(!memcmp(&c,&saved,sizeof c));
        CHECK(nvm_record_map_publication(&c.m).status==NVM_ARRAY_ELIGIBLE);
        CHECK(ra_live==live&&ra_bytes==bytes);
    }
    printf("I checked %zu transient/persistent allocation denials plus retry.\n",positions*2);
#endif
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    publication_cases();publication_target();publication_scan_limit();publication_allocations();
    for(int i=1;i<argc;i++)map_original_module(argv[i]);
    CHECK(!ra_live&&!ra_bytes);
    printf("PASS %u complete-module publication query checks; publishers unchanged\n",checks);return 0;
}
