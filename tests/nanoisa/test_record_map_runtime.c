/* Actual private switch/dispatch qualification; public admission stays closed. */
#include "../../src/nanovm/record_map_runtime_private.h"
static void runtime_original(const NvmModule *,const NvmRecordMapExecutionPlan *,size_t);
#define MAP_EXECUTION_ORIGINAL_CHECK runtime_original
#define MAP_EXECUTION_MAIN retained_map_recipe_controls
#include "test_record_map_execution.c"
int g_argc;char **g_argv;
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#include "../../src/nanovm/vm.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
static VmRecordMapPrivate *originals[8];
static unsigned original_count;
static void runtime_original(const NvmModule *m,const NvmRecordMapExecutionPlan *source,size_t before) {
    (void)before;VmRecordMapPrivate *p=NULL;
    NvmArrayEligibilityResult result=vm_record_map_private_create(source,&p);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"runtime preparation: %u %s\n",result.status,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE&&p);
    VmRecordMapPreparation cost;CHECK(vm_record_map_private_preparation(p,&cost));
    CHECK(cost.runtime_bytes>=sizeof *p+sizeof(VmRecordMapInvocation));
    CHECK(!p->vm.verified&&!p->vm.stack_size&&!p->vm.frame_count&&!p->vm.heap.private_record_dag);
    map_public_transport(m,true);CHECK(original_count<8);originals[original_count++]=p;
    printf("I reserved actual runtime owner%zu/view%zu, %llu bytes/%llu work (overlap %llu/%llu).\n",
        sizeof *p,sizeof(VmRecordMapInvocation),(unsigned long long)cost.runtime_bytes,
        (unsigned long long)cost.runtime_work,(unsigned long long)cost.image_overlap_bytes,
        (unsigned long long)cost.image_overlap_work);
}

static void runtime_integer(Input *c,int64_t value) {
    op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)((uint64_t)value>>(8*i)));
}
static VmRecordMapPrivate *runtime_instance(Input *c) {
    NvmRecordMapExecutionPlan *plan=execution_prepare(&c->m);VmRecordMapPrivate *p=NULL;
    NvmArrayEligibilityResult r=vm_record_map_private_create(plan,&p);
    if(r.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"runtime fixture: %u %s\n",r.status,r.message);
    CHECK(r.status==NVM_ARRAY_ELIGIBLE&&p);nvm_record_map_execution_free(plan);return p;
}
static void runtime_idle(VmRecordMapPrivate *p,VmResult expected) {
    VmRecordMapPrivateStats s;CHECK(vm_record_map_private_stats(p,&s));
    CHECK(s.last_status==expected&&!s.active_frames&&!s.active_stack&&!p->busy);
}
static void runtime_initializer(void) {
    Input c;map_basic(&c);c.m.function_count=2;c.fn[1].name_idx=3;
    c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].result_tag=TAG_STRUCT;c.fn[1].result_count=1;
    map_new(&c,TAG_STRING,TAG_INT);record(&c,0);op(&c,OP_RET);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
    VmRecordMapPrivate *p=runtime_instance(&c);uint64_t objects=p->vm.heap.stats.num_objects;
    CHECK(vm_record_map_private_run(p)==VM_OK);runtime_idle(p,VM_OK);
    CHECK(p->result.tag==TAG_INT&&p->result.as.i64==0&&p->vm.heap.stats.num_objects==objects);
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    c.n=c.fn[1].code_offset;map_new(&c,TAG_STRING,TAG_INT);record(&c,0);
    op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,0);op(&c,OP_ASSERT);op(&c,OP_RET);
    c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
    p=runtime_instance(&c);objects=p->vm.heap.stats.num_objects;
    CHECK(vm_record_map_private_run(p)==VM_ERR_ASSERT_FAILED);runtime_idle(p,VM_ERR_ASSERT_FAILED);
    CHECK(!p->has_result&&p->vm.heap.stats.num_objects==objects);
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void runtime_missing_get(void) {
    for(unsigned wrong=0;wrong<2;wrong++) {
        Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);
        arg16(&c,OP_STORE_LOCAL,0);arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);
        scalar(&c,TAG_STRING);op(&c,OP_HM_HAS);op(&c,OP_BOOL_NOT);op(&c,OP_ASSERT);
        arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);
        op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,wrong?TAG_INT:TAG_VOID);op(&c,OP_ASSERT);map_finish(&c);
        VmRecordMapPrivate *p=runtime_instance(&c);VmResult want=wrong?VM_ERR_ASSERT_FAILED:VM_OK;
        CHECK(vm_record_map_private_run(p)==want);runtime_idle(p,want);CHECK(p->has_result==!wrong);
        vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
}
static void runtime_tail_prefixes(void) {
    Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);
    scalar(&c,TAG_STRING);runtime_integer(&c,37);op(&c,OP_HM_SET);record(&c,0);arg16(&c,OP_STORE_LOCAL,0);
    for(unsigned i=0;i<17;i++)arg16(&c,OP_LOAD_LOCAL,0);
    arg32(&c,OP_TAIL_CALL,1);c.fn[0].code_length=(uint32_t)c.n;c.m.function_count=2;
    c.fn[1].name_idx=2;c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].arity=17;c.fn[1].local_count=19;
    c.fn[1].result_count=1;c.fn[1].result_tag=TAG_INT;uint8_t parameters[17];memset(parameters,TAG_STRUCT,sizeof parameters);
    c.param_rows[1]=parameters;c.m.function_param_types=c.param_rows;
    scalar(&c,TAG_BOOL);op(&c,OP_ASSERT);
    arg16(&c,OP_LOAD_LOCAL,16);arg16(&c,OP_STRUCT_GET,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);
    op(&c,OP_CAST_INT);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
    VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    ra_calls=0;CHECK(vm_record_map_private_run(p)==VM_OK);size_t total=ra_calls;
    CHECK(p->result.tag==TAG_INT&&p->result.as.i64==37&&p->maximum_frames==1);
    vm_record_map_private_destroy(p);CHECK(total&&!ra_live&&!ra_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<total;fail++) {
        p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
        uint64_t objects=p->vm.heap.stats.num_objects,bytes=p->vm.heap.stats.allocated-p->vm.heap.stats.freed;
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;
        CHECK(vm_record_map_private_run(p)==VM_ERR_MEMORY);ra_fail=SIZE_MAX;ra_persistent=0;
        runtime_idle(p,VM_ERR_MEMORY);CHECK(p->result.tag==TAG_INT&&p->result.as.i64==37);
        CHECK(p->vm.heap.stats.num_objects==objects&&p->vm.heap.stats.allocated-p->vm.heap.stats.freed==bytes);
        CHECK(vm_record_map_private_run(p)==VM_OK);runtime_idle(p,VM_OK);
        vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I checked %zu tail/map execution allocation positions in both failure modes.\n",total);
}

static void runtime_patch(Input *c,size_t pc) {
    size_t at=pc+1;b32(c->code,&at,(uint32_t)(int32_t)(c->n-pc));
}
static void runtime_alias_failure(void) {
    Input c;init(&c,TAG_INT,false);c.fn[0].result_tag=TAG_STRUCT;
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_VOID);
    size_t initialized=c.n;arg32(&c,OP_JMP_FALSE,0);
    map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);runtime_integer(&c,0);op(&c,OP_HM_SET);
    record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);runtime_patch(&c,initialized);
    arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_STRUCT_GET,0);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);op(&c,OP_CAST_INT);
    runtime_integer(&c,1);op(&c,OP_I64_ADD);arg16(&c,OP_STORE_LOCAL,1);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);arg16(&c,OP_LOAD_LOCAL,1);op(&c,OP_HM_SET);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);runtime_integer(&c,2);op(&c,OP_I64_LT_S);op(&c,OP_ASSERT);
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;map_authority(&c);
    VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    CHECK(p->result.tag==TAG_STRUCT&&p->result.as.obj==p->vm.globals[0].as.obj);
    void *identity=p->result.as.obj;CHECK(vm_record_map_private_run(p)==VM_ERR_ASSERT_FAILED);
    runtime_idle(p,VM_ERR_ASSERT_FAILED);CHECK(p->has_result&&p->result.as.obj==identity&&p->vm.globals[0].as.obj==identity);
    NanoValue value;CHECK(vm_rm_heap_get(p->result.as.sval->fields[0].as.hashmap,val_string(p->vm.module_constants.strings[1]),&value)==VM_RM_HEAP_OK);
    CHECK(value.tag==TAG_INT&&value.as.i64==2); /* Preserved edge, not mutation rollback. */
    CHECK(p->vm.last_error==VM_ERR_ASSERT_FAILED);
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void runtime_context_headroom(void) {
    for(unsigned map_value=0;map_value<2;map_value++) {
        Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);
        if(map_value)arg16(&c,OP_STRUCT_GET,0);
        scalar(&c,TAG_BOOL);op(&c,OP_ASSERT);size_t duplication=c.n;op(&c,OP_DUP);op(&c,OP_POP);op(&c,OP_POP);map_finish(&c);
        VmRecordMapPrivate *p=runtime_instance(&c);VmState *vm=&p->vm;
        p->epoch=1;p->busy=true;vm->heap.private_record_dag=true;
        VmRecordMapInvocation v={p,vm,vm->module,1,0,0,true};
        VmCallFrame *f=&vm->frames[0];memset(f,0,sizeof *f);f->module=vm->module;
        f->local_count=p->functions[0].signature.local_count;f->owned_callable=val_void();
        for(unsigned i=0;i<f->local_count;i++)vm->stack[vm->stack_size++]=val_void();
        vm->frame_count=1;vm->current_fn=0;vm->ip=0;
        CHECK(vm_rmx_context(&v,vm));v.epoch++;CHECK(!vm_rmx_context(&v,vm));v.epoch--;
        v.root_base=1;CHECK(!vm_rmx_context(&v,vm));v.root_base=0;
        v.active=false;CHECK(!vm_rmx_context(&v,vm));v.active=true;
        vm->verified=true;CHECK(!vm_rmx_context(&v,vm));vm->verified=false;
        VmTrap trap=vm_core_execute_pair(vm,NULL,NULL,NULL,&v,NULL,TAG_INT);CHECK(trap.type==TRAP_ASSERT);
        vm_release(&vm->heap,trap.data.assert_check.condition);
        NanoValue value=stack_peek(vm,0);CHECK(value.tag==(map_value?TAG_HASHMAP:TAG_STRUCT));
        VmHeapHeader *h=value.as.obj;uint32_t refs=h->ref_count,size=vm->stack_size;h->ref_count=UINT32_MAX;
        trap=vm_core_execute_pair(vm,NULL,NULL,NULL,&v,NULL,TAG_INT);
        CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_MEMORY&&vm->stack_size==size&&h->ref_count==UINT32_MAX);
        h->ref_count=refs;vm->ip=(uint32_t)duplication;vm->last_error=VM_OK;
        trap=vm_core_execute_pair(vm,NULL,NULL,NULL,&v,NULL,TAG_INT);CHECK(trap.type==TRAP_NONE);
        vm_rmx_unwind(p);v.active=false;p->busy=false;
        CHECK(vm_record_map_private_run(p)==VM_OK);vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
}

static void runtime_corruption(void) {
    Input c;map_basic(&c);VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    uint64_t epoch=p->epoch;NanoValue previous=p->result;
    p->busy=true;CHECK(vm_record_map_private_run(p)==VM_ERR_TYPE_ERROR);p->busy=false;CHECK(p->epoch==epoch);
    p->epoch=UINT64_MAX;CHECK(vm_record_map_private_run(p)==VM_ERR_TYPE_ERROR);p->epoch=epoch;
    p->vm.verified=true;CHECK(vm_record_map_private_run(p)==VM_ERR_TYPE_ERROR);p->vm.verified=false;
    for(unsigned field=0;field<3;field++) {
        NvmRecordMapExecutionInstruction saved=p->instructions[0];
        if(field==0)p->instructions[0].base.recipe=0;
        if(field==1)p->instructions[0].result_tags=0;
        if(field==2)p->instructions[0].map_operation=NVM_RM_NONE;
        CHECK(vm_record_map_private_run(p)==VM_ERR_DECODE);runtime_idle(p,VM_ERR_DECODE);
        CHECK(p->has_result&&p->result.tag==previous.tag&&p->result.as.i64==previous.as.i64);
        p->instructions[0]=saved;CHECK(vm_record_map_private_run(p)==VM_OK);
    }
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}

static void runtime_growth(void) {
    Input c;init(&c,TAG_INT,false);char text[40][16];char *strings[44];uint32_t lengths[44];
    for(unsigned i=0;i<4;i++){strings[i]=c.strings[i];lengths[i]=c.lengths[i];}
    for(unsigned i=0;i<40;i++){int n=snprintf(text[i],sizeof text[i],"key%u",i);CHECK(n>0);strings[i+4]=text[i];lengths[i+4]=(uint32_t)n;}
    memcpy(text[0],"a\0x",3);memcpy(text[1],"a\0y",3);text[0][3]=text[1][3]=0;lengths[4]=lengths[5]=3;
    c.m.strings=strings;c.m.string_lengths=lengths;c.m.string_count=44;
    map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg16(&c,OP_STORE_LOCAL,0);
    for(unsigned i=0;i<40;i++) {
        arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);arg32(&c,OP_PUSH_STR,i+4);
        runtime_integer(&c,i);op(&c,OP_HM_SET);op(&c,OP_POP);
    }
    for(unsigned i=0;i<40;i++) {
        arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);arg32(&c,OP_PUSH_STR,i+4);op(&c,OP_HM_HAS);op(&c,OP_ASSERT);
        arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);arg32(&c,OP_PUSH_STR,i+4);op(&c,OP_HM_GET);op(&c,OP_CAST_INT);
        runtime_integer(&c,i);op(&c,OP_I64_EQ);op(&c,OP_ASSERT);
    }
    arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);op(&c,OP_HM_LEN);runtime_integer(&c,40);op(&c,OP_I64_EQ);op(&c,OP_ASSERT);
    map_finish(&c);VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    ra_calls=0;CHECK(vm_record_map_private_run(p)==VM_OK);size_t total=ra_calls;
    vm_record_map_private_destroy(p);CHECK(total&&!ra_live&&!ra_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t fail=0;fail<total;fail++) {
        p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
        uint64_t objects=p->vm.heap.stats.num_objects,bytes=p->vm.heap.stats.allocated-p->vm.heap.stats.freed;
        ra_calls=0;ra_fail=fail;ra_persistent=persistent;CHECK(vm_record_map_private_run(p)==VM_ERR_MEMORY);
        ra_fail=SIZE_MAX;ra_persistent=0;runtime_idle(p,VM_ERR_MEMORY);CHECK(p->has_result&&p->result.tag==TAG_INT&&p->result.as.i64==0);
        CHECK(p->vm.heap.stats.num_objects==objects&&p->vm.heap.stats.allocated-p->vm.heap.stats.freed==bytes);
        CHECK(vm_record_map_private_run(p)==VM_OK);vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I checked %zu growth execution positions in both modes with distinct counted-NUL keys.\n",total);
}
static void runtime_tail_headroom(void) {
    Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);op(&c,OP_DUP);
    scalar(&c,TAG_BOOL);op(&c,OP_ASSERT);uint32_t tail=(uint32_t)c.n;arg32(&c,OP_TAIL_CALL,1);
    c.fn[0].code_length=(uint32_t)c.n;c.m.function_count=2;c.fn[1].name_idx=2;c.fn[1].code_offset=(uint32_t)c.n;
    c.fn[1].arity=c.fn[1].local_count=2;c.fn[1].result_count=1;c.fn[1].result_tag=TAG_INT;
    c.params[0]=c.params[1]=TAG_STRUCT;c.param_rows[1]=c.params;c.m.function_param_types=c.param_rows;
    runtime_integer(&c,37);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;map_authority(&c);
    VmRecordMapPrivate *p=runtime_instance(&c);VmState *vm=&p->vm;p->epoch=1;p->busy=true;vm->heap.private_record_dag=true;
    VmRecordMapInvocation v={p,vm,vm->module,1,0,0,true};VmCallFrame *f=&vm->frames[0];memset(f,0,sizeof *f);
    f->module=vm->module;f->local_count=p->functions[0].signature.local_count;f->owned_callable=val_void();
    for(unsigned i=0;i<f->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;
    VmTrap trap=vm_core_execute_pair(vm,NULL,NULL,NULL,&v,NULL,TAG_INT);CHECK(trap.type==TRAP_ASSERT);vm_release(&vm->heap,trap.data.assert_check.condition);
    NanoValue value=stack_peek(vm,0);CHECK(value.as.obj==stack_peek(vm,1).as.obj);VmHeapHeader *h=value.as.obj;
    uint32_t refs=h->ref_count,size=vm->stack_size;h->ref_count=UINT32_MAX-1;
    trap=vm_core_execute_pair(vm,NULL,NULL,NULL,&v,NULL,TAG_INT);CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_MEMORY);
    CHECK(v.active_root==0&&vm->current_fn==0&&vm->frame_count==1&&vm->stack_size==size&&h->ref_count==UINT32_MAX-1);
    h->ref_count=refs;vm->ip=tail;vm->last_error=VM_OK;trap=vm_core_execute_pair(vm,NULL,NULL,NULL,&v,NULL,TAG_INT);
    CHECK(trap.type==TRAP_NONE&&v.active_root==1&&!vm->frame_count&&vm->stack_size==1&&vm->stack[0].as.i64==37);
    vm_rmx_unwind(p);v.active=false;p->busy=false;vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
#ifndef MAP_RUNTIME_MAIN
#define MAP_RUNTIME_MAIN main
#endif
int MAP_RUNTIME_MAIN(int argc,char **argv) {
    runtime_initializer();runtime_missing_get();runtime_tail_prefixes();runtime_alias_failure();runtime_context_headroom();runtime_corruption();runtime_growth();runtime_tail_headroom();
    for(int i=1;i<argc;i++)map_original_module(argv[i]);
    /* Original modules and source plans have been destroyed by their loaders. */
    for(unsigned i=0;i<original_count;i++) {
        VmRecordMapPrivate *p=originals[i];VmResult status=vm_record_map_private_run(p);
        if(status!=VM_OK)fprintf(stderr,"actual runtime %u: %s\n",status,p->vm.error_msg);
        CHECK(status==VM_OK);VmRecordMapPrivateStats stats;CHECK(vm_record_map_private_stats(p,&stats));
        CHECK(stats.has_result&&!stats.active_frames&&!stats.active_stack&&stats.epoch==1);
        CHECK(p->result.tag==TAG_INT&&p->result.as.i64==0);
        CHECK(p->vm.heap.private_record_dag&&!p->vm.verified&&!p->busy);
        printf("I executed original%u with result0, maximum_frames%u, heap objects%llu.\n",i,
            stats.maximum_frames,(unsigned long long)stats.heap_objects);
        vm_record_map_private_destroy(p);
    }
    CHECK(!ra_live&&!ra_bytes);
    printf("PASS %u actual private map runtime checks; public boundaries checked.\n",checks);
    return 0;
}
