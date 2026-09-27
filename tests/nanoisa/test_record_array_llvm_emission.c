/* I qualify actual emitted text preparation and its allocation domain. */
#define RECORD_ARRAY_EXECUTION_MAIN retained_complete_plan_controls
#include "test_record_array_execution.c"
#undef RECORD_ARRAY_EXECUTION_MAIN
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanoisa/nvm2llvm_record_array_private.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
static bool wasm32;
static void emitted(Input *c) {
    Input before;memcpy(&before,c,sizeof before);
    char *text=(void *)(uintptr_t)1;size_t length=SIZE_MAX;NvmRecordArrayGeneratedCost cost;
    CHECK(nvm2llvm_record_array_private(&c->m,wasm32,&text,&length,&cost).status==NVM_ARRAY_ELIGIBLE);
    CHECK(!memcmp(c,&before,sizeof before)&&length==strlen(text));
    CHECK(strstr(text,"define private void @body_0")&&strstr(text,"nrg_generated_create"));
    CHECK(strstr(text,wasm32?nms_runtime_target_wasm32:nms_runtime_target_native)==text);
    CHECK(strstr(text,"@nrg_layout_field")&&strstr(text,"switch i32"));
    CHECK(!strstr(text,"vm_core_execute")&&!strstr(text,"isa_decode"));
    CHECK(cost.plan_bytes<=NVM_RECORD_ARRAY_EXECUTION_BYTES&&cost.consumer_bytes<=NRG_EXTRA_BYTES);
    CHECK(cost.plan_steps<=NVM_RECORD_ARRAY_EXECUTION_STEPS&&cost.consumer_steps<=NRG_EXTRA_STEPS);
    CHECK(ra_peak<=cost.plan_bytes+cost.consumer_bytes);
    ra_test_free(text);CHECK(!ra_live&&!ra_bytes);
}
static void allocation_failures(void) {
    Input c;all_copy_domains(&c);CHECK(!ra_live&&!ra_bytes);
    ra_calls=0;ra_peak=0;emitted(&c);size_t positions=ra_calls;
    CHECK(positions>0);
    for(int persistent=0;persistent<2;persistent++)for(size_t failure=0;failure<positions;failure++) {
        ra_calls=0;ra_peak=0;ra_fail=failure;ra_persistent=persistent;
        char *text=(void *)(uintptr_t)1;size_t length=SIZE_MAX;
        NvmRecordArrayGeneratedCost cost,before;memset(&cost,0xa5,sizeof cost);memcpy(&before,&cost,sizeof cost);
        NvmArrayEligibilityResult r=nvm2llvm_record_array_private(&c.m,wasm32,&text,&length,&cost);
        if(r.status!=NVM_ARRAY_MEMORY)fprintf(stderr,"I expected emitted MEMORY at %zu/%zu mode%d, got%u\n",failure,positions,persistent,r.status);
        CHECK(r.status==NVM_ARRAY_MEMORY&&text==(void *)(uintptr_t)1&&length==SIZE_MAX);
        CHECK(!memcmp(&cost,&before,sizeof cost)&&!ra_live&&!ra_bytes);
        ra_fail=SIZE_MAX;ra_persistent=0;ra_calls=0;ra_peak=0;emitted(&c);
    }
    printf("I checked %zu actual emission allocation positions in both modes with independent recovery.\n",positions);
}
static void work_and_output_boundaries(void) {
    RgOutput b={.status=NVM_ARRAY_ELIGIBLE};
    CHECK(rg_charge(&b,NRG_EXTRA_BYTES,NRG_EXTRA_STEPS));
    CHECK(!rg_charge(&b,1,0)&&b.status==NVM_ARRAY_LIMIT);
    b=(RgOutput){.status=NVM_ARRAY_ELIGIBLE};CHECK(rg_charge(&b,0,NRG_EXTRA_STEPS));
    CHECK(!rg_charge(&b,0,1)&&b.status==NVM_ARRAY_LIMIT);
    b=(RgOutput){.status=NVM_ARRAY_MEMORY};CHECK(!rg_charge(&b,UINT64_MAX,UINT64_MAX)&&b.status==NVM_ARRAY_MEMORY);
    b=(RgOutput){.status=NVM_ARRAY_ELIGIBLE};
    CHECK(rg_write(&b,"%s","retained"));char *original=b.text;size_t used=b.used;
    b.work=NRG_EXTRA_STEPS;
    CHECK(!rg_write(&b,"%s","rejected")&&b.status==NVM_ARRAY_LIMIT&&b.text==original&&b.used==used&&!strcmp(b.text,"retained"));
    ra_test_free(b.text);CHECK(!ra_live&&!ra_bytes);
    b=(RgOutput){.status=NVM_ARRAY_ELIGIBLE};CHECK(rg_write(&b,"%s","retained"));
    original=b.text;used=b.used;size_t capacity=b.capacity;
    ra_fail=ra_calls;
    CHECK(!rg_write(&b,"%8192s","growth")&&b.status==NVM_ARRAY_MEMORY&&b.text==original&&b.used==used&&b.capacity==capacity);
    CHECK(!strcmp(b.text,"retained"));ra_fail=SIZE_MAX;ra_test_free(b.text);CHECK(!ra_live&&!ra_bytes);
}
static void temporary_format_boundaries(void) {
    char text[8];RgOutput b={.status=NVM_ARRAY_ELIGIBLE};
    rl_text(&b,text,sizeof text,"%s","exact07");
    CHECK(b.status==NVM_ARRAY_ELIGIBLE&&b.work==8&&!strcmp(text,"exact07"));
    rl_text(&b,text,sizeof text,"%s","overflow");CHECK(b.status==NVM_ARRAY_LIMIT);
    b=(RgOutput){.status=NVM_ARRAY_ELIGIBLE,.work=NRG_EXTRA_STEPS};
    rl_text(&b,text,sizeof text,"%s","x");CHECK(b.status==NVM_ARRAY_LIMIT);
    b=(RgOutput){.status=NVM_ARRAY_MEMORY};memset(text,0xa5,sizeof text);
    char before[sizeof text];memcpy(before,text,sizeof before);
    rl_text(&b,text,sizeof text,"%s","x");
    CHECK(b.status==NVM_ARRAY_MEMORY&&!memcmp(before,text,sizeof text));
    RlBlock block={.out=&b,.serial=UINT32_MAX};
    b.status=NVM_ARRAY_ELIGIBLE;CHECK(rl_next(&block)==0&&b.status==NVM_ARRAY_LIMIT);
}
static void manifest_and_refusals(void) {
    unsigned count=0;
    for(unsigned opcode=0;opcode<256;opcode++) {
        NvmRecordArrayExecutionInstruction row={0};bool present=rg_recipe((uint8_t)opcode,&row)!=0;
        CHECK(present==nvm_record_array_opcode_supported((uint8_t)opcode));count+=present;
    }
    CHECK(count==93);
    Input c;basic(&c,TAG_INT,false);
    for(unsigned kind=0;kind<3;kind++) {
        char *out=(void *)(uintptr_t)1;size_t length=SIZE_MAX;NvmRecordArrayGeneratedCost cost,before;
        memset(&cost,0xa5,sizeof cost);memcpy(&before,&cost,sizeof cost);
        CHECK(nvm2llvm_record_array_private(kind?&c.m:NULL,wasm32,kind==1?NULL:&out,&length,kind==2?NULL:&cost).status==NVM_ARRAY_INVALID);
        CHECK(out==(void *)(uintptr_t)1&&length==SIZE_MAX&&!memcmp(&before,&cost,sizeof cost));
    }
    puts("I checked all93 emission recipes and all256 decisions, output sentinels and exact work/buffer boundaries.");
}
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    temporary_format_boundaries();
    for(unsigned target=0;target<2;target++){wasm32=target!=0;
        manifest_and_refusals();work_and_output_boundaries();allocation_failures();}
    CHECK(!ra_live&&!ra_bytes);printf("I passed %u LLVM emission checks; no public admission.\n",checks);return 0;
}
