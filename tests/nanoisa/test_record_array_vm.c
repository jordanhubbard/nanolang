/* I execute the distinct private ordinary mixed VM, never a public admission. */
#include "../../src/nanovm/record_array_runtime_private.h"
#define main retained_record_array_plan_controls
#include "test_record_array_execution.c"
#undef main
#ifdef VM_RA_WHITEBOX
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanovm/vm.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
#endif
static void integer(Input *c,int64_t value) {
    op(c,OP_PUSH_I64);uint64_t bits=(uint64_t)value;
    for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)(bits>>(i*8)));
}
static void scalar_variant(Input *c,uint8_t tag) {
    if(tag==TAG_INT)integer(c,19);
    else if(tag==TAG_FLOAT) {
        op(c,OP_PUSH_F64);uint64_t bits=UINT64_C(0x8000000000000000);
        for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)(bits>>(i*8)));
    } else if(tag==TAG_U8){op(c,OP_PUSH_U8);b8(c->code,&c->n,255);}
    else if(tag==TAG_BOOL){op(c,OP_PUSH_BOOL);b8(c->code,&c->n,0);}
    else arg32(c,OP_PUSH_STR,1);
}
static VmRecordArrayPrivate *instance(Input *c) {
    Input before=*c;VmRecordArrayPrivate *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult result=vm_record_array_private_create(&c->m,&p);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"I could not prepare private VM: %u %s\n",result.status,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE&&p!=(void *)(uintptr_t)1&&!memcmp(c,&before,sizeof before));
    return p;
}
static VmRecordArrayObservation observe(VmRecordArrayPrivate *p,VmRecordArrayRoot root,uint32_t index,
                                         const uint32_t *path,uint16_t depth) {
    VmRecordArrayObservation value;
    CHECK(vm_record_array_private_observe(p,root,index,path,depth,&value));return value;
}
static VmRecordArrayPrivateStats idle(VmRecordArrayPrivate *p,VmResult expected) {
    VmRecordArrayPrivateStats stats;
    CHECK(vm_record_array_private_stats(p,&stats));
    CHECK(!stats.active_frames&&!stats.active_stack&&stats.last_status==expected);
    CHECK(stats.preparation_bytes<=NVM_RECORD_ARRAY_EXECUTION_BYTES+VM_RECORD_ARRAY_EXTRA_BYTES);
    CHECK(stats.preparation_steps<=NVM_RECORD_ARRAY_EXECUTION_STEPS+VM_RECORD_ARRAY_EXTRA_STEPS);
    return stats;
}
static void graph_program(Input *c,uint8_t tag) {
    init(c,tag,true);
    array(c,tag);scalar(c,tag);op(c,OP_ARR_PUSH);op(c,OP_DUP);arg32(c,OP_STORE_GLOBAL,0);
    record(c,0);arg32(c,OP_STORE_GLOBAL,1);
    arg32(c,OP_LOAD_GLOBAL,0);integer(c,0);integer(c,1);op(c,OP_ARR_SLICE);arg32(c,OP_STORE_GLOBAL,2);
    arg32(c,OP_LOAD_GLOBAL,0);scalar_variant(c,tag);op(c,OP_ARR_PUSH);op(c,OP_POP);
    /* Both record constructor spellings map the compact ordinal to global2. */
    arg32(c,OP_LOAD_GLOBAL,2);op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);
    b32(c->code,&c->n,1);b16(c->code,&c->n,0);b16(c->code,&c->n,1);arg32(c,OP_STORE_GLOBAL,3);
    finish(c);
}
static void graph_aliases_and_independent_copies(void) {
    for(uint8_t tag=TAG_INT;tag<=TAG_STRING;tag++) {
        Input *c=malloc(sizeof *c);CHECK(c);graph_program(c,tag);
        char counted[]={'a',0,'z',0};c->strings[1]=counted;c->lengths[1]=3;
        VmRecordArrayPrivate *p=instance(c);
        memset(c,0xcc,sizeof *c);free(c);memset(counted,0xcc,sizeof counted);
        CHECK(vm_record_array_private_run(p)==VM_OK);
        VmRecordArrayPrivateStats first=idle(p,VM_OK);CHECK(first.has_result&&first.epoch==1);
        VmRecordArrayObservation result=observe(p,VM_RA_RESULT,0,NULL,0);CHECK(result.tag==TAG_INT&&result.scalar_bits==0);
        VmRecordArrayObservation array_a=observe(p,VM_RA_GLOBAL,0,NULL,0);
        VmRecordArrayObservation record_a=observe(p,VM_RA_GLOBAL,1,NULL,0);
        VmRecordArrayObservation array_b=observe(p,VM_RA_GLOBAL,2,NULL,0);
        VmRecordArrayObservation record_b=observe(p,VM_RA_GLOBAL,3,NULL,0);
        CHECK(array_a.tag==TAG_ARRAY&&array_a.element_tag==tag&&array_a.length==2);
        CHECK(array_b.tag==TAG_ARRAY&&array_b.element_tag==tag&&array_b.length==1&&array_a.identity!=array_b.identity);
        CHECK(record_a.tag==TAG_STRUCT&&record_a.layout==1&&record_a.length==1);
        CHECK(record_b.tag==TAG_STRUCT&&record_b.layout==2&&record_b.identity!=record_a.identity);
        uint32_t field[]={0};
        CHECK(observe(p,VM_RA_GLOBAL,1,field,1).identity==array_a.identity);
        CHECK(observe(p,VM_RA_GLOBAL,3,field,1).identity==array_b.identity);
        VmRecordArrayObservation a=observe(p,VM_RA_GLOBAL,0,field,1),b=observe(p,VM_RA_GLOBAL,2,field,1);
        CHECK(a.tag==tag&&b.tag==tag&&a.scalar_bits==b.scalar_bits);
        if(tag==TAG_STRING) {
            CHECK(a.identity==b.identity&&a.length==3);
            char bytes[4]={0};CHECK(vm_record_array_private_string(p,VM_RA_GLOBAL,0,field,1,0,3,bytes));
            CHECK(!memcmp(bytes,"a\0z",3));
            char sentinel='x';CHECK(!vm_record_array_private_string(p,VM_RA_GLOBAL,0,field,1,3,1,&sentinel)&&sentinel=='x');
        }
        uint32_t second[]={1};VmRecordArrayObservation changed=observe(p,VM_RA_GLOBAL,0,second,1);
        if(tag==TAG_INT)CHECK(changed.scalar_bits==19);
        if(tag==TAG_FLOAT)CHECK(changed.scalar_bits==UINT64_C(0x8000000000000000));
        if(tag==TAG_U8)CHECK(changed.scalar_bits==255);
        if(tag==TAG_BOOL)CHECK(changed.scalar_bits==0);
        UNCHANGED(VmRecordArrayObservation,vm_record_array_private_observe(p,VM_RA_GLOBAL,9,NULL,0,&out));
        UNCHANGED(VmRecordArrayObservation,vm_record_array_private_observe(p,VM_RA_GLOBAL,2,second,1,&out));
        UNCHANGED(VmRecordArrayObservation,vm_record_array_private_observe(p,VM_RA_RESULT,1,NULL,0,&out));
        CHECK(vm_record_array_private_run(p)==VM_OK);
        VmRecordArrayPrivateStats again=idle(p,VM_OK);
        CHECK(again.epoch==2&&again.heap_objects==first.heap_objects&&again.heap_live_bytes==first.heap_live_bytes);
        vm_record_array_private_destroy(p);
    }
}
static void heap_operand(Input *c) {
    array(c,TAG_STRING);arg32(c,OP_PUSH_STR,1);op(c,OP_ARR_PUSH);
}
static void wrong_tag_cleanup(void) {
    static const struct {uint8_t opcode,tag,count;} cases[]={
        {OP_I64_ADD,TAG_INT,2},{OP_I64_SUB,TAG_INT,2},{OP_I64_MUL,TAG_INT,2},
        {OP_I64_DIV_S,TAG_INT,2},{OP_I64_REM_S,TAG_INT,2},{OP_I64_NEG,TAG_INT,1},
        {OP_I64_EQ,TAG_INT,2},{OP_I64_NE,TAG_INT,2},{OP_I64_LT_S,TAG_INT,2},
        {OP_I64_LE_S,TAG_INT,2},{OP_I64_GT_S,TAG_INT,2},{OP_I64_GE_S,TAG_INT,2},
        {OP_F64_ADD,TAG_FLOAT,2},{OP_F64_SUB,TAG_FLOAT,2},{OP_F64_MUL,TAG_FLOAT,2},
        {OP_F64_DIV,TAG_FLOAT,2},{OP_F64_NEG,TAG_FLOAT,1},{OP_F64_EQ,TAG_FLOAT,2},
        {OP_F64_NE,TAG_FLOAT,2},{OP_F64_LT,TAG_FLOAT,2},{OP_F64_LE,TAG_FLOAT,2},
        {OP_F64_GT,TAG_FLOAT,2},{OP_F64_GE,TAG_FLOAT,2},{OP_F64_TO_BITS,TAG_FLOAT,1},
        {OP_F64_FROM_BITS,TAG_INT,1},{OP_BOOL_AND,TAG_BOOL,2},{OP_BOOL_OR,TAG_BOOL,2},
        {OP_BOOL_NOT,TAG_BOOL,1},{OP_CAST_U8,TAG_INT,1},
    };
    for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++)for(unsigned bad=0;bad<cases[i].count;bad++) {
        Input c;init(&c,TAG_STRING,true);
        for(unsigned position=0;position<cases[i].count;position++) {
            if(position==bad)heap_operand(&c);else scalar(&c,cases[i].tag);
        }
        op(&c,cases[i].opcode);op(&c,OP_POP);finish(&c);
        VmRecordArrayPrivate *p=instance(&c);
        VmRecordArrayPrivateStats before=idle(p,VM_OK);
        CHECK(vm_record_array_private_run(p)==VM_ERR_TYPE_ERROR);
        VmRecordArrayPrivateStats after=idle(p,VM_ERR_TYPE_ERROR);
        CHECK(!after.has_result&&after.heap_objects==before.heap_objects&&after.heap_live_bytes==before.heap_live_bytes);
        CHECK(vm_record_array_private_run(p)==VM_ERR_TYPE_ERROR);
        after=idle(p,VM_ERR_TYPE_ERROR);CHECK(after.heap_objects==before.heap_objects&&after.heap_live_bytes==before.heap_live_bytes);
        vm_record_array_private_destroy(p);
    }
}
static void patch_branch(Input *c,size_t instruction,size_t destination) {
    CHECK(instruction+5<=c->n&&destination<=c->n);
    int64_t distance=(int64_t)destination-(int64_t)instruction;CHECK(distance>=INT32_MIN&&distance<=INT32_MAX);
    size_t at=instruction+1;b32(c->code,&at,(uint32_t)(int32_t)distance);
}
static void retained_result_and_globals(void) {
    Input c;init(&c,TAG_INT,true);
    /* I initialize a persistent counter only while its actual tag is VOID. */
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_VOID);
    size_t initialized=c.n;arg32(&c,OP_JMP_FALSE,0);
    integer(&c,0);arg32(&c,OP_STORE_GLOBAL,0);patch_branch(&c,initialized,c.n);
    arg32(&c,OP_LOAD_GLOBAL,0);integer(&c,1);op(&c,OP_I64_ADD);op(&c,OP_DUP);arg32(&c,OP_STORE_GLOBAL,0);
    integer(&c,2);op(&c,OP_I64_LT_S);op(&c,OP_ASSERT);
    integer(&c,7);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;authority(&c);
    VmRecordArrayPrivate *p=instance(&c);CHECK(vm_record_array_private_run(p)==VM_OK);
    CHECK(observe(p,VM_RA_RESULT,0,NULL,0).scalar_bits==7);
    CHECK(observe(p,VM_RA_GLOBAL,0,NULL,0).scalar_bits==1);
    CHECK(vm_record_array_private_run(p)==VM_ERR_ASSERT_FAILED);
    VmRecordArrayPrivateStats stats=idle(p,VM_ERR_ASSERT_FAILED);CHECK(stats.has_result&&stats.epoch==2);
    CHECK(observe(p,VM_RA_RESULT,0,NULL,0).scalar_bits==7);
    CHECK(observe(p,VM_RA_GLOBAL,0,NULL,0).scalar_bits==2);
    vm_record_array_private_destroy(p);
}
static void initializer_prefix(void) {
    Input c;init(&c,TAG_STRING,true);heap_operand(&c);record(&c,0);op(&c,OP_POP);finish(&c);
    c.m.function_count=2;c.fn[1].name_idx=3;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].result_tag=TAG_ARRAY;c.fn[1].result_count=1;
    heap_operand(&c);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;authority(&c);
    VmRecordArrayPrivate *p=instance(&c);VmRecordArrayPrivateStats before=idle(p,VM_OK);
    CHECK(vm_record_array_private_run(p)==VM_OK);
    VmRecordArrayPrivateStats after=idle(p,VM_OK);
    CHECK(after.heap_objects==before.heap_objects&&after.heap_live_bytes==before.heap_live_bytes);
    CHECK(observe(p,VM_RA_RESULT,0,NULL,0).scalar_bits==0);
    vm_record_array_private_destroy(p);
    /* Failure after creating an initializer heap owner never enters main. */
    c.n=c.fn[1].code_offset;heap_operand(&c);op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,0);op(&c,OP_ASSERT);op(&c,OP_RET);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;authority(&c);
    p=instance(&c);before=idle(p,VM_OK);CHECK(vm_record_array_private_run(p)==VM_ERR_ASSERT_FAILED);
    after=idle(p,VM_ERR_ASSERT_FAILED);CHECK(!after.has_result&&after.heap_objects==before.heap_objects);
    vm_record_array_private_destroy(p);
}
#ifdef VM_RA_WHITEBOX
static void independent_recovery(Input *c) {
    ra_fail=SIZE_MAX;ra_persistent=0;
    VmRecordArrayPrivate *p=instance(c);CHECK(vm_record_array_private_run(p)==VM_OK);idle(p,VM_OK);
    vm_record_array_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void allocation_prefixes(void) {
    Input c;graph_program(&c,TAG_STRING);
    CHECK(!ra_live&&!ra_bytes);ra_calls=0;ra_peak=0;
    VmRecordArrayPrivate *p=instance(&c);size_t prepare_calls=ra_calls;
    VmRecordArrayPrivateStats prepared=idle(p,VM_OK);CHECK(ra_peak<=prepared.preparation_bytes);
    CHECK(vm_record_array_private_run(p)==VM_OK);
    ra_calls=0;CHECK(vm_record_array_private_run(p)==VM_OK);size_t execution_calls=ra_calls;
    vm_record_array_private_destroy(p);CHECK(!ra_live&&!ra_bytes&&prepare_calls&&execution_calls);
    for(int persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<prepare_calls;fail++) {
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;
        VmRecordArrayPrivate *sentinel=(void *)(uintptr_t)1;
        NvmArrayEligibilityResult result=vm_record_array_private_create(&c.m,&sentinel);
        if(result.status!=NVM_ARRAY_MEMORY)fprintf(stderr,"I expected create MEMORY at %zu/%zu mode%d, got%u\n",fail,prepare_calls,persistent,result.status);
        CHECK(result.status==NVM_ARRAY_MEMORY&&sentinel==(void *)(uintptr_t)1);
        CHECK(!ra_live&&!ra_bytes);independent_recovery(&c);
    }
    for(int persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<execution_calls;fail++) {
        ra_fail=SIZE_MAX;ra_persistent=0;p=instance(&c);CHECK(vm_record_array_private_run(p)==VM_OK);
        VmRecordArrayObservation previous=observe(p,VM_RA_RESULT,0,NULL,0);
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;
        CHECK(vm_record_array_private_run(p)==VM_ERR_MEMORY);idle(p,VM_ERR_MEMORY);
        VmRecordArrayObservation preserved=observe(p,VM_RA_RESULT,0,NULL,0);
        CHECK(preserved.tag==previous.tag&&preserved.scalar_bits==previous.scalar_bits);
        vm_record_array_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
        independent_recovery(&c);
    }
    printf("I checked %zu preparation and %zu execution allocation positions in both modes with independent recovery.\n",prepare_calls,execution_calls);
}
#endif
