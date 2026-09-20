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
#ifdef VM_RA_WHITEBOX
static bool retired[256];
#endif
static void dispose(VmRecordArrayPrivate *p) {
#ifdef VM_RA_WHITEBOX
    for(unsigned i=0;i<256;i++)retired[i]|=p->vm.profile.opcode_counts[i]!=0;
#endif
    vm_record_array_private_destroy(p);
}
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
    #ifdef VM_RA_WHITEBOX
    vm_profile_enable(&p->vm,true);
    #endif
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
        dispose(p);
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
        dispose(p);
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
    dispose(p);
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
    dispose(p);
    /* Failure after creating an initializer heap owner never enters main. */
    c.n=c.fn[1].code_offset;heap_operand(&c);op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,0);op(&c,OP_ASSERT);op(&c,OP_RET);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;authority(&c);
    p=instance(&c);before=idle(p,VM_OK);CHECK(vm_record_array_private_run(p)==VM_ERR_ASSERT_FAILED);
    after=idle(p,VM_ERR_ASSERT_FAILED);CHECK(!after.has_result&&after.heap_objects==before.heap_objects);
    dispose(p);
}
#ifdef VM_RA_WHITEBOX
static void independent_recovery(Input *c) {
    ra_fail=SIZE_MAX;ra_persistent=0;
    VmRecordArrayPrivate *p=instance(c);CHECK(vm_record_array_private_run(p)==VM_OK);idle(p,VM_OK);
    dispose(p);CHECK(!ra_live&&!ra_bytes);
}
static void allocation_prefixes(void) {
    Input c;graph_program(&c,TAG_STRING);
    CHECK(!ra_live&&!ra_bytes);ra_calls=0;ra_peak=0;
    VmRecordArrayPrivate *p=instance(&c);size_t prepare_calls=ra_calls;
    VmRecordArrayPrivateStats prepared=idle(p,VM_OK);CHECK(ra_peak<=prepared.preparation_bytes);
    CHECK(vm_record_array_private_run(p)==VM_OK);
    ra_calls=0;CHECK(vm_record_array_private_run(p)==VM_OK);size_t execution_calls=ra_calls;
    dispose(p);CHECK(!ra_live&&!ra_bytes&&prepare_calls&&execution_calls);
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
        dispose(p);CHECK(!ra_live&&!ra_bytes);
        independent_recovery(&c);
    }
    printf("I checked %zu preparation and %zu execution allocation positions in both modes with independent recovery.\n",prepare_calls,execution_calls);
}
#endif
static void expect_integer(Input *c,int64_t expected) {
    integer(c,expected);op(c,OP_I64_EQ);op(c,OP_ASSERT);
}
static void expect_boolean(Input *c,bool expected) {
    op(c,OP_PUSH_BOOL);b8(c->code,&c->n,expected);op(c,OP_EQ);op(c,OP_ASSERT);
}
static void floating(Input *c,double value) {
    uint64_t bits;memcpy(&bits,&value,sizeof bits);op(c,OP_PUSH_F64);
    for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)(bits>>(i*8)));
}
static void scalar_handlers(void) {
    Input c;init(&c,TAG_INT,true);op(&c,OP_NOP);
    static const struct {uint8_t opcode;int64_t expected;} integer_ops[]={
        {OP_I64_ADD,9},{OP_I64_SUB,3},{OP_I64_MUL,18},{OP_I64_DIV_S,2},{OP_I64_REM_S,0},
    };
    for(unsigned i=0;i<sizeof integer_ops/sizeof integer_ops[0];i++) {
        integer(&c,6);integer(&c,3);op(&c,integer_ops[i].opcode);expect_integer(&c,integer_ops[i].expected);
    }
    integer(&c,INT64_MIN);op(&c,OP_I64_NEG);expect_integer(&c,INT64_MIN);
    static const struct {uint8_t opcode;double expected;} float_ops[]={
        {OP_F64_ADD,9.0},{OP_F64_SUB,3.0},{OP_F64_MUL,18.0},{OP_F64_DIV,2.0},
    };
    for(unsigned i=0;i<sizeof float_ops/sizeof float_ops[0];i++) {
        floating(&c,6.0);floating(&c,3.0);op(&c,float_ops[i].opcode);
        floating(&c,float_ops[i].expected);op(&c,OP_F64_EQ);op(&c,OP_ASSERT);
    }
    floating(&c,0.0);op(&c,OP_F64_NEG);op(&c,OP_F64_TO_BITS);expect_integer(&c,INT64_MIN);
    integer(&c,INT64_MIN);op(&c,OP_F64_FROM_BITS);op(&c,OP_F64_TO_BITS);expect_integer(&c,INT64_MIN);
    static const uint8_t comparisons[][6]={
        {OP_EQ,OP_NE,OP_LT,OP_LE,OP_GT,OP_GE},
        {OP_I64_EQ,OP_I64_NE,OP_I64_LT_S,OP_I64_LE_S,OP_I64_GT_S,OP_I64_GE_S},
        {OP_F64_EQ,OP_F64_NE,OP_F64_LT,OP_F64_LE,OP_F64_GT,OP_F64_GE},
    };
    for(unsigned family=0;family<3;family++)for(unsigned i=0;i<6;i++) {
        if(family==2){floating(&c,2.0);floating(&c,3.0);}else{integer(&c,2);integer(&c,3);}
        op(&c,comparisons[family][i]);expect_boolean(&c,i==1||i==2||i==3);
    }
    static const uint8_t logic[]={OP_BOOL_AND,OP_BOOL_OR,OP_AND,OP_OR};
    for(unsigned i=0;i<4;i++) {
        op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,1);op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,0);
        op(&c,logic[i]);expect_boolean(&c,i==1||i==3);
    }
    op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,0);op(&c,OP_BOOL_NOT);op(&c,OP_ASSERT);
    op(&c,OP_PUSH_VOID);op(&c,OP_NOT);op(&c,OP_ASSERT);
    integer(&c,257);op(&c,OP_CAST_U8);op(&c,OP_DUP);op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_U8);op(&c,OP_ASSERT);
    op(&c,OP_CAST_INT);expect_integer(&c,1);
    op(&c,OP_PUSH_U8);b8(c.code,&c.n,255);op(&c,OP_CAST_FLOAT);floating(&c,255.0);op(&c,OP_F64_EQ);op(&c,OP_ASSERT);
    integer(&c,1);op(&c,OP_CAST_BOOL);op(&c,OP_ASSERT);
    integer(&c,11);integer(&c,17);op(&c,OP_SWAP);expect_integer(&c,11);expect_integer(&c,17);
    integer(&c,43);arg16(&c,OP_STORE_LOCAL,0);arg16(&c,OP_LOAD_LOCAL,0);expect_integer(&c,43);
    finish(&c);VmRecordArrayPrivate *p=instance(&c);CHECK(vm_record_array_private_run(p)==VM_OK);idle(p,VM_OK);
    dispose(p);
}
static void recursion_program(Input *c,int64_t depth,bool initializer) {
    init(c,TAG_INT,true);integer(c,depth);arg32(c,OP_CALL,1);op(c,OP_RET);
    c->fn[0].code_length=(uint32_t)c->n;c->m.function_count=initializer?3:2;
    c->fn[1].name_idx=2;c->fn[1].code_offset=(uint32_t)c->n;
    c->fn[1].arity=1;c->fn[1].local_count=2;c->fn[1].result_tag=TAG_INT;c->fn[1].result_count=1;
    c->params[0]=TAG_INT;c->param_rows[1]=c->params;c->m.function_param_types=c->param_rows;
    array(c,TAG_INT);arg16(c,OP_STORE_LOCAL,1);
    arg16(c,OP_LOAD_LOCAL,0);integer(c,0);op(c,OP_I64_EQ);
    size_t recursive=c->n;arg32(c,OP_JMP_FALSE,0);
    integer(c,31);op(c,OP_RET);patch_branch(c,recursive,c->n);
    arg16(c,OP_LOAD_LOCAL,0);integer(c,1);op(c,OP_I64_SUB);arg32(c,OP_CALL,1);op(c,OP_RET);
    c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;
    if(initializer) {
        c->fn[2].name_idx=3;c->fn[2].code_offset=(uint32_t)c->n;
        c->fn[2].result_tag=TAG_ARRAY;c->fn[2].result_count=1;
        array(c,TAG_INT);op(c,OP_RET);c->fn[2].code_length=(uint32_t)c->n-c->fn[2].code_offset;
    }
    authority(c);
}
static void frame_boundaries(void) {
    for(unsigned prefix=0;prefix<2;prefix++)for(unsigned overflow=0;overflow<2;overflow++) {
        Input c;recursion_program(&c,overflow?1023:1022,prefix!=0);
        VmRecordArrayPrivate *p=instance(&c);VmRecordArrayPrivateStats before=idle(p,VM_OK);
        VmResult expected=overflow?VM_ERR_CALL_DEPTH:VM_OK;
        CHECK(vm_record_array_private_run(p)==expected);
        VmRecordArrayPrivateStats after=idle(p,expected);
        CHECK(after.maximum_frames==1024&&after.heap_objects==before.heap_objects&&after.heap_live_bytes==before.heap_live_bytes);
        if(!overflow)CHECK(observe(p,VM_RA_RESULT,0,NULL,0).scalar_bits==31);
        else CHECK(!after.has_result);
        dispose(p);
    }
}
static void expect_string(Input *c,uint32_t pool) {
    arg32(c,OP_PUSH_STR,pool);op(c,OP_STR_EQ);op(c,OP_ASSERT);
}
static void string_handlers(void) {
    Input c;init(&c,TAG_STRING,true);
    char *strings[]={"main"," Ab ","Ab","ab","AB","Abab","0"," ab "};
    uint32_t lengths[8];for(unsigned i=0;i<8;i++)lengths[i]=(uint32_t)strlen(strings[i]);
    c.m.strings=strings;c.m.string_lengths=lengths;c.m.string_count=8;
    arg32(&c,OP_PUSH_STR,1);op(&c,OP_STR_LEN);expect_integer(&c,4);
    arg32(&c,OP_PUSH_STR,1);op(&c,OP_STR_TRIM);expect_string(&c,2);
    arg32(&c,OP_PUSH_STR,2);op(&c,OP_STR_TO_LOWER);expect_string(&c,3);
    arg32(&c,OP_PUSH_STR,3);op(&c,OP_STR_TO_UPPER);expect_string(&c,4);
    arg32(&c,OP_PUSH_STR,2);arg32(&c,OP_PUSH_STR,3);op(&c,OP_STR_CONCAT);expect_string(&c,5);
    arg32(&c,OP_PUSH_STR,1);integer(&c,1);integer(&c,2);op(&c,OP_STR_SUBSTR);expect_string(&c,2);
    arg32(&c,OP_PUSH_STR,1);integer(&c,1);op(&c,OP_STR_CHAR_AT);expect_integer(&c,'A');
    arg32(&c,OP_PUSH_STR,1);arg32(&c,OP_PUSH_STR,2);op(&c,OP_STR_CONTAINS);op(&c,OP_ASSERT);
    arg32(&c,OP_PUSH_STR,5);arg32(&c,OP_PUSH_STR,2);op(&c,OP_STR_STARTS_WITH);op(&c,OP_ASSERT);
    arg32(&c,OP_PUSH_STR,5);arg32(&c,OP_PUSH_STR,3);op(&c,OP_STR_ENDS_WITH);op(&c,OP_ASSERT);
    arg32(&c,OP_PUSH_STR,1);arg32(&c,OP_PUSH_STR,2);arg32(&c,OP_PUSH_STR,3);op(&c,OP_STR_REPLACE);expect_string(&c,7);
    arg32(&c,OP_PUSH_STR,5);arg32(&c,OP_PUSH_STR,2);op(&c,OP_STR_SPLIT);op(&c,OP_ARR_LEN);expect_integer(&c,2);
    integer(&c,0);op(&c,OP_STR_FROM_INT);expect_string(&c,6);
    floating(&c,0.0);op(&c,OP_STR_FROM_FLOAT);expect_string(&c,6);
    integer(&c,0);op(&c,OP_CAST_STRING);expect_string(&c,6);
    finish(&c);VmRecordArrayPrivate *p=instance(&c);VmRecordArrayPrivateStats before=idle(p,VM_OK);
    CHECK(vm_record_array_private_run(p)==VM_OK);VmRecordArrayPrivateStats after=idle(p,VM_OK);
    CHECK(after.heap_objects==before.heap_objects&&after.heap_live_bytes==before.heap_live_bytes);
    dispose(p);
}
static void mutation_handlers(void) {
    Input c;init(&c,TAG_INT,true);
    integer(&c,7);integer(&c,9);op(&c,OP_ARR_LITERAL);b8(c.code,&c.n,TAG_INT);b16(c.code,&c.n,2);
    op(&c,OP_DUP);arg16(&c,OP_STORE_LOCAL,0);record(&c,0);arg16(&c,OP_STORE_LOCAL,1);
    arg16(&c,OP_LOAD_LOCAL,0);integer(&c,0);integer(&c,42);op(&c,OP_ARR_SET);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);arg16(&c,OP_STRUCT_GET,0);integer(&c,0);op(&c,OP_ARR_GET);expect_integer(&c,42);
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_ARR_POP);expect_integer(&c,9);
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_ARR_LEN);expect_integer(&c,1);
    arg16(&c,OP_LOAD_LOCAL,1);array(&c,TAG_INT);arg16(&c,OP_STRUCT_SET,0);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_AGG_SET,0);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);arg16(&c,OP_AGG_GET,0);integer(&c,0);op(&c,OP_ARR_GET);expect_integer(&c,42);
    /* Real finite backedge and both branch opcodes, with one shared local. */
    integer(&c,2);arg16(&c,OP_STORE_LOCAL,2);size_t loop=c.n;
    arg16(&c,OP_LOAD_LOCAL,2);integer(&c,1);op(&c,OP_I64_SUB);op(&c,OP_DUP);arg16(&c,OP_STORE_LOCAL,2);
    size_t back=c.n;arg32(&c,OP_JMP_TRUE,0);patch_branch(&c,back,loop);
    size_t skip=c.n;arg32(&c,OP_JMP,0);op(&c,OP_NOP);patch_branch(&c,skip,c.n);
    finish(&c);VmRecordArrayPrivate *p=instance(&c);VmRecordArrayPrivateStats before=idle(p,VM_OK);
    CHECK(vm_record_array_private_run(p)==VM_OK);VmRecordArrayPrivateStats after=idle(p,VM_OK);
    CHECK(after.heap_objects==before.heap_objects&&after.heap_live_bytes==before.heap_live_bytes);
    dispose(p);
}
static void zero_field_constructor(void) {
    Input c;init(&c,TAG_INT,false);arg32(&c,OP_STRUCT_NEW,0);op(&c,OP_POP);finish(&c);
    /* My first layout has no fields. I remove its12-byte field row, and its
     *12-byte array binding; the remaining layout keeps the original binding. */
    CHECK(c.l==44);c.layouts[6]=c.layouts[7]=0;
    memmove(c.layouts+12,c.layouts+24,c.l-24);c.l-=12;c.m.layout_size=(uint32_t)c.l;
    CHECK(c.o>=52);size_t binding_count=c.o-28,chunk_size=c.o-44;
    size_t at=chunk_size;b32(c.ownership,&at,28);at=binding_count;b32(c.ownership,&at,1);
    memmove(c.ownership+binding_count+4,c.ownership+binding_count+16,12);c.o-=12;c.m.ownership_size=(uint32_t)c.o;
    VmRecordArrayPrivate *p=instance(&c);VmRecordArrayPrivateStats before=idle(p,VM_OK);
    CHECK(vm_record_array_private_run(p)==VM_OK);VmRecordArrayPrivateStats after=idle(p,VM_OK);
    CHECK(after.heap_objects==before.heap_objects&&after.heap_live_bytes==before.heap_live_bytes);
    dispose(p);
}

#ifdef VM_RA_WHITEBOX
#define BASE (NVM_RA_FIRST_ERROR_CLEANUP|NVM_RA_CHECK_TAGS)
#define SAFE (NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS)
#define KEEP NVM_RA_RETAIN_BEFORE_RELEASE
#define BOUND NVM_RA_CHECK_BOUNDS
#define NOM NVM_RA_CHECK_NOMINAL
#define SPEC(op,kind,flags) {OP_##op,NVM_RA_ROOT_##kind,(flags)}
static const struct {uint8_t opcode,recipe;uint16_t flags;} operations[]={
    SPEC(NOP,SCALAR,NVM_RA_FIRST_ERROR_CLEANUP),
    SPEC(PUSH_I64,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),SPEC(PUSH_U8,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),
    SPEC(PUSH_F64,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),SPEC(PUSH_BOOL,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),
    SPEC(PUSH_VOID,CONSTANT,NVM_RA_FIRST_ERROR_CLEANUP),SPEC(PUSH_STR,CONSTANT,BASE|KEEP),
    SPEC(I64_ADD,SCALAR,BASE),SPEC(I64_SUB,SCALAR,BASE),SPEC(I64_MUL,SCALAR,BASE),SPEC(I64_DIV_S,SCALAR,BASE),SPEC(I64_REM_S,SCALAR,BASE),SPEC(I64_NEG,SCALAR,BASE),
    SPEC(F64_TO_BITS,SCALAR,BASE),SPEC(F64_FROM_BITS,SCALAR,BASE),SPEC(F64_ADD,SCALAR,BASE),SPEC(F64_SUB,SCALAR,BASE),SPEC(F64_MUL,SCALAR,BASE),SPEC(F64_DIV,SCALAR,BASE),SPEC(F64_NEG,SCALAR,BASE),
    SPEC(CAST_U8,SCALAR,BASE),SPEC(CAST_INT,SCALAR,BASE),SPEC(CAST_FLOAT,SCALAR,BASE),SPEC(CAST_BOOL,SCALAR,BASE),SPEC(TYPE_CHECK,SCALAR,BASE),
    SPEC(EQ,SCALAR,BASE),SPEC(NE,SCALAR,BASE),SPEC(LT,SCALAR,BASE),SPEC(LE,SCALAR,BASE),SPEC(GT,SCALAR,BASE),SPEC(GE,SCALAR,BASE),
    SPEC(I64_EQ,SCALAR,BASE),SPEC(I64_NE,SCALAR,BASE),SPEC(I64_LT_S,SCALAR,BASE),SPEC(I64_LE_S,SCALAR,BASE),SPEC(I64_GT_S,SCALAR,BASE),SPEC(I64_GE_S,SCALAR,BASE),
    SPEC(F64_EQ,SCALAR,BASE),SPEC(F64_NE,SCALAR,BASE),SPEC(F64_LT,SCALAR,BASE),SPEC(F64_LE,SCALAR,BASE),SPEC(F64_GT,SCALAR,BASE),SPEC(F64_GE,SCALAR,BASE),
    SPEC(BOOL_AND,SCALAR,BASE),SPEC(BOOL_OR,SCALAR,BASE),SPEC(BOOL_NOT,SCALAR,BASE),SPEC(AND,SCALAR,BASE),SPEC(OR,SCALAR,BASE),SPEC(NOT,SCALAR,BASE),
    SPEC(STR_LEN,SCALAR,BASE),SPEC(STR_CHAR_AT,SCALAR,BASE),SPEC(ARR_LEN,SCALAR,BASE),SPEC(STR_EQ,SCALAR,BASE),SPEC(STR_CONTAINS,SCALAR,BASE),SPEC(STR_STARTS_WITH,SCALAR,BASE),SPEC(STR_ENDS_WITH,SCALAR,BASE),
    SPEC(CAST_STRING,STRING,BASE|SAFE),SPEC(STR_CONCAT,STRING,BASE|SAFE),SPEC(STR_SUBSTR,STRING,BASE|SAFE),SPEC(STR_TRIM,STRING,BASE|SAFE),SPEC(STR_TO_LOWER,STRING,BASE|SAFE),SPEC(STR_TO_UPPER,STRING,BASE|SAFE),SPEC(STR_REPLACE,STRING,BASE|SAFE),SPEC(STR_FROM_INT,STRING,BASE|SAFE),SPEC(STR_FROM_FLOAT,STRING,BASE|SAFE),SPEC(STR_SPLIT,STRING,BASE|SAFE),
    SPEC(LOAD_LOCAL,LOAD,BASE|KEEP),SPEC(LOAD_GLOBAL,LOAD,BASE|KEEP),SPEC(STORE_LOCAL,STORE,BASE|KEEP),SPEC(STORE_GLOBAL,STORE,BASE|KEEP),
    SPEC(DUP,DUP,BASE|KEEP),SPEC(SWAP,SWAP,BASE),SPEC(POP,DROP,BASE),SPEC(ASSERT,DROP,BASE),
    SPEC(JMP,BRANCH,BASE),SPEC(JMP_TRUE,BRANCH,BASE),SPEC(JMP_FALSE,BRANCH,BASE),
    SPEC(CALL,CALL,BASE|NVM_RA_STAGE_OPERANDS),SPEC(RET,RETURN,BASE|NVM_RA_STAGE_OPERANDS),
    SPEC(STRUCT_NEW,CONSTRUCT,BASE|NOM|SAFE),SPEC(STRUCT_LITERAL,CONSTRUCT,BASE|NOM|SAFE),SPEC(AGG_PACK,CONSTRUCT,BASE|NOM|SAFE),
    SPEC(ARR_NEW,CONSTRUCT,BASE|SAFE),SPEC(ARR_LITERAL,CONSTRUCT,BASE|SAFE),
    SPEC(STRUCT_GET,GET,BASE|NOM|BOUND|KEEP),SPEC(AGG_GET,GET,BASE|NOM|BOUND|KEEP),SPEC(ARR_GET,GET,BASE|BOUND|KEEP),
    SPEC(STRUCT_SET,SET,BASE|NOM|BOUND|KEEP),SPEC(AGG_SET,SET,BASE|NOM|BOUND|KEEP),SPEC(ARR_SET,SET,BASE|SAFE|BOUND|KEEP),
    SPEC(ARR_PUSH,ARRAY_PUSH,BASE|SAFE|KEEP),SPEC(ARR_POP,ARRAY_POP,BASE|BOUND|KEEP),SPEC(ARR_SLICE,ARRAY_COPY,BASE|SAFE|BOUND)
};
#undef SPEC
#undef BASE
#undef SAFE
#undef KEEP
#undef BOUND
#undef NOM
static void recipe_and_dispatch_agreement(void) {
    bool listed[256]={false};CHECK(sizeof operations/sizeof operations[0]==93);
    for(unsigned i=0;i<93;i++) {
        NvmRecordArrayExecutionInstruction row={0};uint8_t opcode=operations[i].opcode;
        CHECK(!listed[opcode]);listed[opcode]=true;
        CHECK(vm_ra_recipe(opcode,&row)&&row.recipe==operations[i].recipe&&row.obligations==operations[i].flags);
        if(!retired[opcode])fprintf(stderr,"I did not retire opcode %u in the runtime corpus\n",opcode);
        CHECK(retired[opcode]);
    }
    for(unsigned i=0;i<256;i++) {
        NvmRecordArrayExecutionInstruction row={0};
        CHECK(!!vm_ra_recipe((uint8_t)i,&row)==listed[i]);
        CHECK(!!nvm_record_array_opcode_supported((uint8_t)i)==listed[i]);
        CHECK(!retired[i]||listed[i]);
    }
    Input c;recursion_program(&c,2,false);VmRecordArrayPrivate *p=instance(&c);
    uint64_t work=p->work;CHECK(vm_ra_coverage(p));p->work=work;
    /* I alter a non-first fact and each independently retained execution map.
     * These are preparation checks only; no changed image is ever executed. */
    NvmRecordArrayExecutionInstruction saved=p->instructions[1];
#define DIFFER(field,expr) do {p->instructions[1].field=(expr);CHECK(!vm_ra_coverage(p));p->instructions[1]=saved;p->work=work;} while(0)
    DIFFER(recipe,NVM_RA_ROOT_DROP);DIFFER(obligations,0);DIFFER(callee,NO);
    DIFFER(operand_bits[0],UINT64_MAX);DIFFER(successors[0],UINT32_MAX);
#undef DIFFER
    p->functions[1].signature.local_count++;CHECK(!vm_ra_coverage(p));p->functions[1].signature.local_count--;p->work=work;
    VmDecodedFunction *df=&p->vm.decoded_module.functions[0];
    uint8_t boundary=df->boundaries[1];df->boundaries[1]=!boundary;
    CHECK(!vm_ra_coverage(p));df->boundaries[1]=boundary;p->work=work;
    VmDispatchFunction *body=&p->vm.dispatch_module.functions[0];
    uint32_t mapping=body->offset_to_index[0];body->offset_to_index[0]=0;
    CHECK(!vm_ra_coverage(p));body->offset_to_index[0]=mapping;p->work=work;
    uint32_t next=body->instructions[1].next_index;body->instructions[1].next_index=NO;
    CHECK(!vm_ra_coverage(p));body->instructions[1].next_index=next;p->work=work;
    CHECK(vm_ra_coverage(p));p->work=work;dispose(p);
    printf("I retired all 93 recipes and checked all 256 opcode decisions and independent maps.\n");
}
static void private_bounds_and_busy(void) {
    Input c;graph_program(&c,TAG_STRING);VmRecordArrayPrivate *p=instance(&c);
    uint64_t bytes=p->bytes,work=p->work;size_t calls=ra_calls;
    p->bytes=VM_RECORD_ARRAY_EXTRA_BYTES;p->work=VM_RECORD_ARRAY_EXTRA_STEPS;
    CHECK(vm_ra_charge(p,0,0));CHECK(!vm_ra_charge(p,1,0)&&p->preparation_status==NVM_ARRAY_LIMIT);
    CHECK(!vm_ra_charge(p,0,1));CHECK(!vm_ra_charge(p,UINT64_MAX,UINT64_MAX));
    CHECK(!vm_ra_zero(p,UINT64_MAX,2));CHECK(ra_calls==calls);
    p->bytes=bytes;p->work=work;p->preparation_status=NVM_ARRAY_ELIGIBLE;
    p->busy=true;CHECK(vm_record_array_private_run(p)==VM_ERR_TYPE_ERROR);
    UNCHANGED(VmRecordArrayPrivateStats,vm_record_array_private_stats(p,&out));
    UNCHANGED(VmRecordArrayObservation,vm_record_array_private_observe(p,VM_RA_RESULT,0,NULL,0,&out));
    vm_record_array_private_destroy(p);CHECK(p->prepared);p->busy=false;
    p->epoch=UINT64_MAX;CHECK(vm_record_array_private_run(p)==VM_ERR_TYPE_ERROR&&p->epoch==UINT64_MAX);
    p->epoch=0;CHECK(vm_record_array_private_run(p)==VM_OK);dispose(p);
    CHECK(!ra_live&&!ra_bytes);
}
#endif
/* My declarations reach the full 256-row DAG limit. My executed graph retains
 * the original 64 allocation-origin limit: one array plus 63 records. */
static void nominal_dag_and_release(void) {
    Input c;init(&c,TAG_STRING,false);heap_operand(&c);
    for(unsigned i=0;i<63;i++)record(&c,i);
    arg32(&c,OP_STORE_GLOBAL,0);finish(&c);
    uint8_t *layouts=calloc(1,4+256*20);CHECK(layouts);size_t n=0;b32(layouts,&n,256);
    for(unsigned i=0;i<256;i++) {
        b8(layouts,&n,NVM_V2_LAYOUT_STRUCT);b8(layouts,&n,0);b16(layouts,&n,1);b32(layouts,&n,NO);
        b8(layouts,&n,i?TAG_STRUCT:TAG_ARRAY);b8(layouts,&n,0);b16(layouts,&n,0);
        b32(layouts,&n,i?i-1:NO);b32(layouts,&n,NO);
    }
    c.m.layout_data=layouts;c.m.layout_size=(uint32_t)n;c.m.struct_count=256;
    c.o=0;b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,256);
    for(unsigned i=0;i<256;i++)b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);
    b32(c.ownership,&c.o,1);b16(c.ownership,&c.o,3);b16(c.ownership,&c.o,0);desc(&c,TAG_INT);
    for(unsigned i=0;i<3;i++)desc(&c,TAG_VOID);
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b16(c.ownership,&c.o,2);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,28);b32(c.ownership,&c.o,1);
    b8(c.ownership,&c.o,TAG_STRING);b8(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,NO);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    c.m.ownership_size=(uint32_t)c.o;
    VmRecordArrayPrivate *p=instance(&c);memset(layouts,0xcc,n);free(layouts);
    CHECK(vm_record_array_private_run(p)==VM_OK);
    uint32_t path[257]={0};VmRecordArrayObservation root=observe(p,VM_RA_GLOBAL,0,NULL,0);
    CHECK(root.tag==TAG_STRUCT&&root.layout==62);
    CHECK(observe(p,VM_RA_GLOBAL,0,path,63).tag==TAG_ARRAY);
    CHECK(observe(p,VM_RA_GLOBAL,0,path,64).tag==TAG_STRING);
#ifdef VM_RA_WHITEBOX
    CHECK(p->layouts[255].rank==256);
    /* This additional heap-layer control exercises maximum release depth.
     * It is not bytecode admission beyond my original 64-origin query. */
    uint64_t objects=p->vm.heap.stats.num_objects,bytes=(p->vm.heap.stats.allocated-p->vm.heap.stats.freed);
    VmArray *a=vm_array_new(&p->vm.heap,TAG_STRING,0);CHECK(a);
    VmString *s=vm_string_new(&p->vm.heap,"deep",4);CHECK(s);
    CHECK(vm_array_push(&p->vm.heap,a,val_string(s)));vm_release(&p->vm.heap,val_string(s));
    NanoValue child=val_array(a);
    for(unsigned i=0;i<256;i++) {
        CHECK(vm_ra_field_accepts(p,i,0,child));
        VmStruct *record_owner=vm_struct_new(&p->vm.heap,i,1);CHECK(record_owner);
        record_owner->fields[0]=child;child=val_struct(record_owner); /* owned move */
    }
    size_t calls=ra_calls;ra_fail=calls;ra_persistent=1;
    vm_release(&p->vm.heap,child);
    CHECK(ra_calls==calls&&p->vm.heap.stats.num_objects==objects&&(p->vm.heap.stats.allocated-p->vm.heap.stats.freed)==bytes);
    ra_fail=SIZE_MAX;ra_persistent=0;
#endif
    dispose(p);
}
static void entry_result_controls(void) {
    Input c;init(&c,TAG_INT,false);integer(&c,29);c.fn[0].code_length=(uint32_t)c.n;authority(&c);
    VmRecordArrayPrivate *p=instance(&c);CHECK(vm_record_array_private_run(p)==VM_OK);
    CHECK(observe(p,VM_RA_RESULT,0,NULL,0).scalar_bits==29);dispose(p);
    init(&c,TAG_INT,false);op(&c,OP_RET);c.fn[0].result_tag=TAG_VOID;c.fn[0].result_count=0;
    c.fn[0].code_length=(uint32_t)c.n;authority(&c);p=instance(&c);
    CHECK(vm_record_array_private_run(p)==VM_OK);CHECK(observe(p,VM_RA_RESULT,0,NULL,0).tag==TAG_VOID);dispose(p);
    for(unsigned explicit_return=0;explicit_return<2;explicit_return++) {
        init(&c,TAG_INT,false);op(&c,explicit_return?OP_RET:OP_NOP);c.fn[0].code_length=(uint32_t)c.n;authority(&c);
        VmRecordArrayPrivate *out=(void *)(uintptr_t)1;
        CHECK(vm_record_array_private_create(&c.m,&out).status==NVM_ARRAY_INVALID&&out==(void *)(uintptr_t)1);
    }
    UNCHANGED(VmRecordArrayPrivateStats,vm_record_array_private_stats(NULL,&out));
    UNCHANGED(VmRecordArrayObservation,vm_record_array_private_observe(NULL,VM_RA_RESULT,0,NULL,0,&out));
    CHECK(vm_record_array_private_run(NULL)==VM_ERR_TYPE_ERROR);vm_record_array_private_destroy(NULL);
}
#ifdef VM_RA_WHITEBOX
static void *wrong_thread(void *arg) {
    VmRecordArrayPrivate *p=arg;
    CHECK(vm_record_array_private_run(p)==VM_ERR_TYPE_ERROR);
    UNCHANGED(VmRecordArrayPrivateStats,vm_record_array_private_stats(p,&out));
    UNCHANGED(VmRecordArrayObservation,vm_record_array_private_observe(p,VM_RA_RESULT,0,NULL,0,&out));
    vm_record_array_private_destroy(p);return NULL;
}
static void boxed_partial_copy_cleanup(void) {
    Input c;graph_program(&c,TAG_STRING);VmRecordArrayPrivate *p=instance(&c);
    VmHeap *heap=&p->vm.heap;size_t objects=heap->stats.num_objects,bytes=heap->stats.allocated-heap->stats.freed;
    VmArray *array_owner=vm_array_new(heap,TAG_STRING,4);CHECK(array_owner);
    VmString *first=vm_string_new(heap,"first-copy",10),*second=vm_string_new(heap,"second-copy",11);CHECK(first&&second);
    CHECK(vm_array_push(heap,array_owner,val_string(first))&&vm_array_push(heap,array_owner,val_string(second)));
    vm_release(heap,val_string(first));vm_release(heap,val_string(second));
    uint32_t first_refs=first->header.ref_count,second_refs=second->header.ref_count;
    size_t with_array=heap->stats.num_objects,with_bytes=heap->stats.allocated-heap->stats.freed;
    second->header.ref_count=UINT32_MAX;
    CHECK(!vm_array_slice(heap,array_owner,0,2));
    CHECK(first->header.ref_count==first_refs&&second->header.ref_count==UINT32_MAX);
    CHECK(heap->stats.num_objects==with_array&&heap->stats.allocated-heap->stats.freed==with_bytes);
    CHECK(!vm_array_push(heap,array_owner,val_string(second))&&array_owner->length==2);
    second->header.ref_count=second_refs;
    VmArray *copy=vm_array_slice(heap,array_owner,0,2);CHECK(copy&&copy!=array_owner&&copy->length==2);
    vm_release(heap,val_array(copy));vm_release(heap,val_array(array_owner));
    CHECK(heap->stats.num_objects==objects&&heap->stats.allocated-heap->stats.freed==bytes);
    dispose(p);CHECK(!ra_live&&!ra_bytes);
}
static void affinity_and_retain_overflow(void) {
    Input c;graph_program(&c,TAG_STRING);VmRecordArrayPrivate *p=instance(&c);
    pthread_t thread;CHECK(!pthread_create(&thread,NULL,wrong_thread,p));CHECK(!pthread_join(thread,NULL));
    CHECK(p->prepared&&!p->epoch&&!p->busy);CHECK(vm_record_array_private_run(p)==VM_OK);
    /* The next run reaches PUSH_STR with a saturated retained pool string.
     * Its caller-owned references survive the first failure and are restored
     * before disposal; the fixture never manufactures an unowned pointer. */
    VmString *s=p->vm.module_constants.strings[1];uint32_t references=s->header.ref_count;
    s->header.ref_count=UINT32_MAX;CHECK(vm_record_array_private_run(p)==VM_ERR_MEMORY);
    CHECK(s->header.ref_count==UINT32_MAX);s->header.ref_count=references;
    idle(p,VM_ERR_MEMORY);CHECK(observe(p,VM_RA_RESULT,0,NULL,0).scalar_bits==0);
    CHECK(vm_record_array_private_run(p)==VM_OK);dispose(p);CHECK(!ra_live&&!ra_bytes);
}
#endif
static void heap_result_alias_and_call(void) {
    Input c;init(&c,TAG_STRING,true);heap_operand(&c);op(&c,OP_DUP);arg32(&c,OP_STORE_GLOBAL,0);
    arg32(&c,OP_CALL,1);record(&c,0);arg32(&c,OP_STORE_GLOBAL,1);finish(&c);
    c.m.function_count=2;c.fn[1].name_idx=2;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].arity=c.fn[1].local_count=1;c.fn[1].result_count=1;c.fn[1].result_tag=TAG_ARRAY;
    c.params[0]=TAG_ARRAY;c.param_rows[1]=c.params;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;authority(&c);
    VmRecordArrayPrivate *p=instance(&c);CHECK(vm_record_array_private_run(p)==VM_OK);
    uint32_t path[]={0};CHECK(observe(p,VM_RA_GLOBAL,0,NULL,0).identity==observe(p,VM_RA_GLOBAL,1,path,1).identity);dispose(p);

    init(&c,TAG_STRING,true);
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_VOID);
    size_t ready=c.n;arg32(&c,OP_JMP_FALSE,0);array(&c,TAG_STRING);arg32(&c,OP_STORE_GLOBAL,0);patch_branch(&c,ready,c.n);
    arg32(&c,OP_LOAD_GLOBAL,0);arg32(&c,OP_PUSH_STR,1);op(&c,OP_ARR_PUSH);op(&c,OP_POP);
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_ARR_LEN);integer(&c,2);op(&c,OP_I64_LT_S);op(&c,OP_ASSERT);
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;
    c.fn[0].result_tag=TAG_ARRAY;authority(&c);
    p=instance(&c);CHECK(vm_record_array_private_run(p)==VM_OK);
    VmRecordArrayObservation first=observe(p,VM_RA_RESULT,0,NULL,0);CHECK(first.tag==TAG_ARRAY&&first.length==1);
    CHECK(first.identity==observe(p,VM_RA_GLOBAL,0,NULL,0).identity);
    CHECK(vm_record_array_private_run(p)==VM_ERR_ASSERT_FAILED);idle(p,VM_ERR_ASSERT_FAILED);
    VmRecordArrayObservation second=observe(p,VM_RA_RESULT,0,NULL,0);
    CHECK(second.identity==first.identity&&second.length==2&&second.epoch==first.epoch+1);
    CHECK(second.identity==observe(p,VM_RA_GLOBAL,0,NULL,0).identity);
    /* Root ownership is preserved. Alias mutation is deliberately not rolled back. */
    dispose(p);
}
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
#define CASE(name) do {printf("I begin %s\n",#name);name();} while(0)
    CASE(graph_aliases_and_independent_copies);CASE(wrong_tag_cleanup);
    CASE(retained_result_and_globals);CASE(heap_result_alias_and_call);CASE(initializer_prefix);CASE(scalar_handlers);
    CASE(frame_boundaries);CASE(string_handlers);CASE(mutation_handlers);
    CASE(zero_field_constructor);CASE(nominal_dag_and_release);CASE(entry_result_controls);
#ifdef VM_RA_WHITEBOX
    CASE(private_bounds_and_busy);CASE(affinity_and_retain_overflow);CASE(boxed_partial_copy_cleanup);
    CASE(recipe_and_dispatch_agreement);CASE(allocation_prefixes);
    CHECK(!ra_live&&!ra_bytes);
#endif
#undef CASE
    printf("I passed %u private mixed VM checks; no public or generated consumer admission.\n",checks);
    return 0;
}
