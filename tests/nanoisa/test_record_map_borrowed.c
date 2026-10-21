/* First actual borrowed-state gate; public selectors remain closed. */
#include "../../src/nanovm/record_map_borrowed_private.h"
static __attribute__((noinline)) void borrowed_original(const NvmModule *);
#define MAP_ORIGINAL_CHECK borrowed_original
#define MAP_ORIGIN_MAIN retained_borrowed_origin_controls
#include "test_record_map_origins.c"
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
static __attribute__((noinline)) void borrowed_original(const NvmModule *m) {
    VmState vm;vm_init(&vm,m);
    for(unsigned repeat=0;repeat<2;repeat++) {
        VmRecordMapPublicPreparation *p=NULL;
        CHECK(vm_record_map_public_prepare(&vm,&p).status==NVM_ARRAY_ELIGIBLE);
        printf("I measured borrowed reservation %llu bytes / %llu work; view %zu, owner %zu.\n",
            (unsigned long long)p->bytes,(unsigned long long)p->work,
            sizeof(VmRecordMapBorrowedInvocation),sizeof *p);
        vm_record_map_public_preparation_free(p);
        VmResult status=vm_record_map_borrowed_execute_private(m,&vm);
        if(status!=VM_OK)fprintf(stderr,"borrowed original failure %u: %s\n",status,vm.error_msg);
        CHECK(status==VM_OK);printf("I executed repeat%u at stack capacity%u with ordinary GC policy.\n",repeat,vm.stack_capacity);CHECK(vm.stack_size==1&&!vm.frame_count&&!vm.heap.private_record_dag);
        CHECK(vm.module==m&&vm.root_module==m);
        NanoValue result=vm_get_result(&vm);CHECK(result.tag==TAG_INT&&result.as.i64==0);
        CHECK(vm_record_map_borrowed_execute_private(m,&vm)==VM_ERR_TYPE_ERROR);
        CHECK(vm.stack_size==1&&vm_get_result(&vm).as.i64==0);
        vm.stack_size--;vm_release(&vm.heap,result);
    }
    /* Actual old NULL route through this newly compiled vm.c, not a retained
     * CLI binary. Restore metadata immediately after the ordinary baseline. */
    NvmModule *mutable=(NvmModule *)m;uint8_t *layouts=mutable->layout_data,*ownership=mutable->ownership_data;
    uint32_t layout_size=mutable->layout_size,ownership_size=mutable->ownership_size;
    mutable->layout_data=NULL;mutable->layout_size=0;mutable->ownership_data=NULL;mutable->ownership_size=0;
    CHECK(nvm_verify(m).ok);VmResult legacy=vm_execute(&vm);
    mutable->layout_data=layouts;mutable->layout_size=layout_size;mutable->ownership_data=ownership;mutable->ownership_size=ownership_size;
    CHECK(legacy==VM_OK);CHECK(vm.stack_size==1&&vm_get_result(&vm).tag==TAG_INT&&vm_get_result(&vm).as.i64==0);
    vm.stack_size--;vm_release(&vm.heap,vm.stack[0]);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    map_public_transport(m,true);
}
static __attribute__((noinline)) void borrowed_source_global(const char *path) {
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0&&size<1048576);rewind(f);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes&&fread(bytes,1,(size_t)size,f)==(size_t)size);CHECK(!fclose(f));
    NvmV2Module v;NvmModule *m=NULL;CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&v)==NVM_V2_OK);
    CHECK(nvm_v2_to_nvm_module(&v,&m)==NVM_V2_OK);nvm_v2_module_free(&v);free(bytes);CHECK(m&&nvm_verify(m).ok);
    CHECK(m->struct_count==1&&!m->layout_size&&!m->ownership_size);
    Input c={0};b32(c.layouts,&c.l,1);b8(c.layouts,&c.l,NVM_V2_LAYOUT_STRUCT);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,1);b32(c.layouts,&c.l,NO);
    b8(c.layouts,&c.l,TAG_HASHMAP);b8(c.layouts,&c.l,0);b16(c.layouts,&c.l,0);b32(c.layouts,&c.l,NO);b32(c.layouts,&c.l,NO);
    b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,1);b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);while(c.o%4)b8(c.ownership,&c.o,0);
    b32(c.ownership,&c.o,m->function_count);
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *fn=&m->functions[i];b16(c.ownership,&c.o,fn->local_count);b16(c.ownership,&c.o,fn->arity);desc(&c,fn->result_tag);
        for(uint16_t j=0;j<fn->local_count;j++)desc(&c,j<fn->arity?m->function_param_types[i][j]:TAG_VOID);
    }
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b16(c.ownership,&c.o,5);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,48);b32(c.ownership,&c.o,2);desc(&c,TAG_INT);desc(&c,TAG_STRING);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,1);
    b32(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);
    m->layout_data=malloc(c.l);m->ownership_data=malloc(c.o);CHECK(m->layout_data&&m->ownership_data);
    memcpy(m->layout_data,c.layouts,c.l);memcpy(m->ownership_data,c.ownership,c.o);m->layout_size=(uint32_t)c.l;m->ownership_size=(uint32_t)c.o;
    borrowed_original(m);nvm_module_free(m);printf("I checked real source-global input %s.\n",path);
}
static void borrowed_integer(Input *c,int64_t value) {
    op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)b8(c->code,&c->n,(uint8_t)((uint64_t)value>>(8*i)));
}
static void borrowed_consume(VmState *vm,int64_t expected) {
    CHECK(vm->stack_size==1&&!vm->frame_count&&!vm->activation_floor&&!vm->heap.private_record_dag);
    NanoValue value=vm_get_result(vm);CHECK(value.tag==TAG_INT&&value.as.i64==expected);
    vm->stack_size--;vm_release(&vm->heap,value);
}
static __attribute__((noinline)) void borrowed_initializer(void) {
    for(unsigned bad=0;bad<2;bad++) {
        Input c;map_basic(&c);c.m.function_count=3;
        c.fn[1].name_idx=3;c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].result_tag=TAG_INT;c.fn[1].result_count=1;
        arg32(&c,OP_TAIL_CALL,2);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;
        c.fn[2].name_idx=2;c.fn[2].code_offset=(uint32_t)c.n;c.fn[2].result_tag=TAG_INT;c.fn[2].result_count=1;
        op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,!bad);op(&c,OP_ASSERT);borrowed_integer(&c,81);op(&c,OP_RET);
        c.fn[2].code_length=(uint32_t)c.n-c.fn[2].code_offset;map_authority(&c);
        VmState vm;vm_init(&vm,&c.m);
        VmResult status=vm_record_map_borrowed_execute_private(&c.m,&vm);
        CHECK(status==(bad?VM_ERR_ASSERT_FAILED:VM_OK));
        if(!bad)borrowed_consume(&vm,0);
        else CHECK(!vm.stack_size&&!vm.frame_count&&vm.last_error==VM_ERR_ASSERT_FAILED);
        vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
    }
}
static __attribute__((noinline)) void borrowed_missing(void) {
    for(unsigned bad=0;bad<2;bad++) {
        Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);
        arg16(&c,OP_STORE_LOCAL,0);arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);
        scalar(&c,TAG_STRING);op(&c,OP_HM_HAS);op(&c,OP_BOOL_NOT);op(&c,OP_ASSERT);
        arg16(&c,OP_LOAD_LOCAL,0);arg16(&c,OP_STRUCT_GET,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);
        op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,bad?TAG_INT:TAG_VOID);op(&c,OP_ASSERT);map_finish(&c);
        VmState vm;vm_init(&vm,&c.m);CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==(bad?VM_ERR_ASSERT_FAILED:VM_OK));
        if(!bad)borrowed_consume(&vm,0);else CHECK(!vm.stack_size&&!vm.frame_count);
        vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
    }
}
static __attribute__((noinline)) void borrowed_alias_failure(void) {
    Input c;init(&c,TAG_INT,false);
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_VOID);
    size_t jump=c.n;arg32(&c,OP_JMP_FALSE,0);
    map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);borrowed_integer(&c,0);op(&c,OP_HM_SET);
    record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);size_t at=jump+1;b32(c.code,&at,(uint32_t)(c.n-jump));
    arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_STRUCT_GET,0);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);op(&c,OP_CAST_INT);
    borrowed_integer(&c,1);op(&c,OP_I64_ADD);arg16(&c,OP_STORE_LOCAL,1);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);arg16(&c,OP_LOAD_LOCAL,1);op(&c,OP_HM_SET);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);borrowed_integer(&c,2);op(&c,OP_I64_LT_S);op(&c,OP_ASSERT);map_finish(&c);
    VmState vm;vm_init(&vm,&c.m);CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,0);
    NanoValue alias=vm.globals[0];vm_retain(&vm.heap,alias);
    CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_ERR_ASSERT_FAILED);
    CHECK(vm.last_error==VM_ERR_ASSERT_FAILED&&!vm.stack_size&&!vm.frame_count);
    CHECK(vm.globals[0].as.obj==alias.as.obj);
    VmHashMap *map=alias.as.sval->fields[0].as.hashmap;NanoValue key=val_string(vm.module_constants.strings[1]),value;
    CHECK(vm_rm_ordinary_get(&vm.heap,map,key,&value)==VM_RM_HEAP_OK&&value.tag==TAG_INT&&value.as.i64==2);
    CHECK(vm_rm_ordinary_set(&vm.heap,map,key,val_int(0))==VM_RM_HEAP_OK);
    CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,0);
    CHECK(vm.globals[0].as.obj==alias.as.obj);vm_release(&vm.heap,alias);
    vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
}
static void borrowed_tail_fixture(Input *c,bool tail,bool distinct) {
    init(c,TAG_INT,false);
    map_new(c,TAG_STRING,TAG_INT);scalar(c,TAG_STRING);borrowed_integer(c,37);op(c,OP_HM_SET);record(c,0);arg16(c,OP_STORE_LOCAL,0);
    if(distinct) {
        map_new(c,TAG_STRING,TAG_INT);scalar(c,TAG_STRING);borrowed_integer(c,91);op(c,OP_HM_SET);record(c,0);arg16(c,OP_STORE_LOCAL,1);
    }
    for(unsigned i=0;i<17;i++)arg16(c,OP_LOAD_LOCAL,distinct&&i==16?1:0);
    scalar(c,TAG_BOOL);op(c,OP_ASSERT);
    arg32(c,tail?OP_TAIL_CALL:OP_CALL,1);if(!tail)op(c,OP_RET);
    c->fn[0].code_length=(uint32_t)c->n;c->m.function_count=2;
    c->fn[1].name_idx=2;c->fn[1].code_offset=(uint32_t)c->n;c->fn[1].arity=17;c->fn[1].local_count=19;
    c->fn[1].result_count=1;c->fn[1].result_tag=TAG_INT;
    static uint8_t parameters[17];for(unsigned i=0;i<17;i++)parameters[i]=TAG_STRUCT;
    c->param_rows[1]=parameters;c->m.function_param_types=c->param_rows;
    scalar(c,TAG_BOOL);op(c,OP_ASSERT);
    arg16(c,OP_LOAD_LOCAL,0);arg16(c,OP_STRUCT_GET,0);scalar(c,TAG_STRING);op(c,OP_HM_GET);op(c,OP_CAST_INT);
    borrowed_integer(c,37);op(c,OP_I64_EQ);op(c,OP_ASSERT);
    arg16(c,OP_LOAD_LOCAL,16);arg16(c,OP_STRUCT_GET,0);scalar(c,TAG_STRING);op(c,OP_HM_GET);op(c,OP_CAST_INT);op(c,OP_RET);
    c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;map_authority(c);
}
static __attribute__((noinline)) void borrowed_calls_prefixes(void) {
    for(unsigned tail=0;tail<2;tail++)for(unsigned distinct=0;distinct<2;distinct++) {
        Input c;borrowed_tail_fixture(&c,tail,distinct);VmState vm;vm_init(&vm,&c.m);
        CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,distinct?91:37);
        vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
    }
    Input c;borrowed_tail_fixture(&c,true,true);VmState vm;vm_init(&vm,&c.m);
    ra_calls=0;CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);size_t positions=ra_calls;
    borrowed_consume(&vm,91);vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes&&positions);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t denied=0;denied<positions;denied++) {
        vm_init(&vm,&c.m);ra_calls=0;ra_fail=denied;ra_persistent=(int)persistent;
        VmResult result=vm_record_map_borrowed_execute_private(&c.m,&vm);
        ra_fail=SIZE_MAX;ra_persistent=0;
        /* Suspect buffering can fail harmlessly; it is not guest allocation. */
        CHECK(result==VM_OK||result==VM_ERR_MEMORY);
        if(result==VM_OK)borrowed_consume(&vm,91);else CHECK(!vm.stack_size&&!vm.frame_count);
        CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,91);
        vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I checked %zu complete invocation allocation positions in both denial modes.\n",positions);
}

static __attribute__((noinline)) void borrowed_nominal(void) {
    Input c;init(&c,TAG_INT,true);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);
    scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;authority(&c);
    /* UNION layout0 makes record ordinal0 resolve to layout1. */
    c.layouts[32]=TAG_HASHMAP;c.layouts[52]=TAG_HASHMAP;c.o-=48;
    b16(c.ownership,&c.o,5);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,60);
    b32(c.ownership,&c.o,2);desc(&c,TAG_INT);desc(&c,TAG_STRING);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,2);
    for(unsigned i=1;i<=2;i++){b32(c.ownership,&c.o,i);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);}
    c.m.ownership_size=(uint32_t)c.o;VmState vm;vm_init(&vm,&c.m);
    CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,0);
    CHECK(vm.globals[0].as.sval->def_idx==0);
    CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,0);
    CHECK(vm.globals[0].as.sval->def_idx==0);
    vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
}
static __attribute__((noinline)) void borrowed_tree(Input *c,unsigned records,unsigned depth,uint8_t leaf) {
    init(c,TAG_INT,false);c->m.struct_count=records;c->fn[0].result_tag=TAG_INT;
    for(unsigned i=1;i<depth;i++)array(c,TAG_ARRAY);
    array(c,leaf);if(leaf==TAG_U8)borrowed_integer(c,300);else scalar(c,leaf);op(c,OP_ARR_PUSH);
    for(unsigned i=1;i<depth;i++)op(c,OP_ARR_PUSH);
    map_new(c,TAG_STRING,TAG_INT);scalar(c,TAG_STRING);scalar(c,TAG_INT);op(c,OP_HM_SET);
    op(c,OP_STRUCT_LITERAL);b32(c->code,&c->n,0);b16(c->code,&c->n,2);
    for(unsigned i=1;i<records;i++)record(c,i);
    arg32(c,OP_STORE_GLOBAL,0);scalar(c,TAG_INT);op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;
    b32(c->layouts,&c->l,records);
    for(unsigned i=0;i<records;i++) {
        b8(c->layouts,&c->l,NVM_V2_LAYOUT_STRUCT);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,i?1:2);b32(c->layouts,&c->l,NO);
        for(unsigned field=0;field<(i?1u:2u);field++) {
            b8(c->layouts,&c->l,i?TAG_STRUCT:field?TAG_HASHMAP:TAG_ARRAY);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,0);
            b32(c->layouts,&c->l,i?i-1:NO);b32(c->layouts,&c->l,NO);
        }
    }
    b32(c->ownership,&c->o,3);b32(c->ownership,&c->o,records);
    for(unsigned i=0;i<records;i++)b8(c->ownership,&c->o,NVM_LAYOUT_COMPLETE);
    while(c->o%4)b8(c->ownership,&c->o,0);
    b32(c->ownership,&c->o,1);b16(c->ownership,&c->o,c->fn[0].local_count);b16(c->ownership,&c->o,0);desc(c,TAG_INT);
    for(unsigned i=0;i<c->fn[0].local_count;i++)desc(c,TAG_VOID);
    b32(c->ownership,&c->o,4);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,2);
    b16(c->ownership,&c->o,2);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,20+depth*8);b32(c->ownership,&c->o,depth);
    for(unsigned i=0;i<depth;i++){b8(c->ownership,&c->o,i+1==depth?leaf:TAG_ARRAY);b8(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,i+1==depth?NO:i+1);}
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);
    b16(c->ownership,&c->o,5);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,48);b32(c->ownership,&c->o,2);desc(c,TAG_INT);desc(c,TAG_STRING);
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,1);
    b32(c->ownership,&c->o,0);b16(c->ownership,&c->o,1);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);
    c->m.layout_size=(uint32_t)c->l;c->m.ownership_size=(uint32_t)c->o;c->m.code_size=(uint32_t)c->n;
}

static __attribute__((noinline)) void borrowed_nested(void) {
    Input c;borrowed_tree(&c,1,2,TAG_U8);VmState vm;vm_init(&vm,&c.m);
    CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,0);
    VmArray *outer=vm.globals[0].as.sval->fields[0].as.array;
    CHECK(outer->elem_type==TAG_ARRAY&&outer->length==1);
    VmArray *inner=outer->elements[0].as.array;NanoValue octet=vm_array_get(inner,0);
    CHECK(octet.tag==TAG_U8&&octet.as.u8==44);
    VmRecordMapPublicPreparation *p=NULL;CHECK(vm_record_map_public_prepare(&vm,&p).status==NVM_ARRAY_ELIGIBLE);
    CHECK(vm_rmb_value_valid(p,&vm.heap,val_array(outer)));
    uint8_t old=inner->header.obj_type;inner->header.obj_type=TAG_INT;
    /* Deliberate forbidden concurrent mutation: local shell checks do not scan
     * grandchildren. A new invocation's complete import must refuse it. */
    CHECK(vm_rmb_value_valid(p,&vm.heap,val_array(outer)));
    CHECK(!vm_rmb_value_valid(p,&vm.heap,val_array(inner)));
    CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_ERR_TYPE_ERROR);
    inner->header.obj_type=old;vm_record_map_public_preparation_free(p);
    CHECK(vm_record_map_borrowed_execute_private(&c.m,&vm)==VM_OK);borrowed_consume(&vm,0);
    vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
}

static void borrowed_activate_whitebox(VmState *vm,VmRecordMapPublicPreparation *p,VmRecordMapBorrowedInvocation *view) {
    CHECK(stack_reserve(vm,VM_RM_ROOT_CAPACITY)==VM_OK);
    *view=(VmRecordMapBorrowedInvocation){p,vm,vm->module,0,0,true};
    VmCallFrame *f=&vm->frames[0];memset(f,0,sizeof *f);f->module=vm->module;
    f->local_count=p->functions[0].signature.local_count;f->owned_callable=val_void();
    for(unsigned i=0;i<f->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=0;vm->ip=p->functions[0].signature.code_offset;
}
static __attribute__((noinline)) void borrowed_context_controls(void) {
    Input c;borrowed_tail_fixture(&c,true,false);VmState vm;vm_init(&vm,&c.m);
    VmRecordMapPublicPreparation *p=NULL;CHECK(vm_record_map_public_prepare(&vm,&p).status==NVM_ARRAY_ELIGIBLE);
    NanoValue *before=vm.stack;uint32_t capacity=vm.stack_capacity;size_t live=ra_live,bytes=ra_bytes;
    ra_fail=ra_calls;CHECK(stack_reserve(&vm,VM_RM_ROOT_CAPACITY)==VM_ERR_MEMORY);ra_fail=SIZE_MAX;
    CHECK(vm.stack==before&&vm.stack_capacity==capacity&&!vm.stack_size&&!vm.frame_count&&ra_live==live&&ra_bytes==bytes);
    vm.last_error=VM_OK;VmRecordMapBorrowedInvocation view;borrowed_activate_whitebox(&vm,p,&view);
    VmTrap trap=vm_core_execute_views(&vm,NULL,NULL,NULL,NULL,&view);CHECK(trap.type==TRAP_ASSERT);
    vm_release(&vm.heap,trap.data.assert_check.condition);
    uint32_t ip=vm.ip,size=vm.stack_size;NanoValue owner=stack_peek(&vm,0);
    CHECK(owner.as.obj==stack_peek(&vm,16).as.obj);
    VmHeapHeader *header=owner.as.obj;uint32_t refs=header->ref_count;
    header->ref_count=UINT32_MAX-16;
    trap=vm_core_execute_views(&vm,NULL,NULL,NULL,NULL,&view);
    CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_MEMORY&&vm.ip==ip&&vm.stack_size==size);
    CHECK(header->ref_count==UINT32_MAX-16&&view.active_root==0);header->ref_count=refs;
    for(unsigned bad=0;bad<4;bad++) {
        switch(bad){case 0:view.active=false;break;case 1:view.module=NULL;break;case 2:vm.current_fn=UINT32_MAX;break;default:vm.heap.private_record_dag=true;break;}
        trap=vm_core_execute_views(&vm,NULL,NULL,NULL,NULL,&view);
        CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_TYPE_ERROR&&vm.ip==ip&&vm.stack_size==size);
        view.active=true;view.module=&c.m;vm.current_fn=0;vm.heap.private_record_dag=false;
    }
    uint32_t index=vm.decoded_module.functions[0].instruction_indices[ip]-1;
    VmDispatchInstruction *instruction=&vm.dispatch_module.functions[0].instructions[index];
    instruction->call_handle.function_index=1;
    trap=vm_core_execute_views(&vm,NULL,NULL,NULL,NULL,&view);
    CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_DECODE&&vm.ip==ip&&vm.stack_size==size);
    instruction->call_handle.function_index=0;
    uint8_t recipe=p->instructions[index].base.recipe;p->instructions[index].base.recipe=UINT8_MAX;
    trap=vm_core_execute_views(&vm,NULL,NULL,NULL,NULL,&view);
    CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_DECODE&&vm.ip==ip&&vm.stack_size==size);
    p->instructions[index].base.recipe=recipe;vm.last_error=VM_OK;
    trap=vm_core_execute_views(&vm,NULL,NULL,NULL,NULL,&view);CHECK(trap.type==TRAP_ASSERT&&view.active_root==1);
    vm_release(&vm.heap,trap.data.assert_check.condition);
    do {trap=vm_core_execute_views(&vm,NULL,NULL,NULL,NULL,&view);
        if(trap.type==TRAP_ASSERT){CHECK(val_truthy(trap.data.assert_check.condition));vm_release(&vm.heap,trap.data.assert_check.condition);}
    } while(trap.type==TRAP_ASSERT);
    CHECK(trap.type==TRAP_NONE);borrowed_consume(&vm,37);
    view.active=false;vm_record_map_public_preparation_free(p);vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
}
static __attribute__((noinline)) void borrowed_u8_source(const char *path) {
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long size=ftell(f);CHECK(size>0&&size<1048576);rewind(f);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes&&fread(bytes,1,(size_t)size,f)==(size_t)size);CHECK(!fclose(f));
    NvmV2Module v;NvmModule *m=NULL;CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&v)==NVM_V2_OK);
    CHECK(nvm_v2_to_nvm_module(&v,&m)==NVM_V2_OK);nvm_v2_module_free(&v);free(bytes);CHECK(m&&nvm_verify(m).ok);
    CHECK(m->struct_count==1&&!m->layout_size&&!m->ownership_size);
    Input shape;borrowed_tree(&shape,1,2,TAG_U8);Input c={0};
    memcpy(c.layouts,shape.layouts,shape.l);c.l=shape.l;
    b32(c.ownership,&c.o,3);b32(c.ownership,&c.o,1);b8(c.ownership,&c.o,NVM_LAYOUT_COMPLETE);while(c.o%4)b8(c.ownership,&c.o,0);
    b32(c.ownership,&c.o,m->function_count);
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *fn=&m->functions[i];b16(c.ownership,&c.o,fn->local_count);b16(c.ownership,&c.o,fn->arity);desc(&c,fn->result_tag);
        for(uint16_t j=0;j<fn->local_count;j++)desc(&c,j<fn->arity?m->function_param_types[i][j]:TAG_VOID);
    }
    b32(c.ownership,&c.o,4);b32(c.ownership,&c.o,0);
    /* Exact two-row ARRAY and one STRING→INT pair extension from the same
     * field schema; function descriptors above use this original producer. */
    size_t extensions=4+8+(20+2*8)+8+48;
    CHECK(shape.o>=extensions&&c.o+extensions<sizeof c.ownership);
    memcpy(c.ownership+c.o,shape.ownership+shape.o-extensions,extensions);c.o+=extensions;
    m->layout_data=malloc(c.l);m->ownership_data=malloc(c.o);CHECK(m->layout_data&&m->ownership_data);
    memcpy(m->layout_data,c.layouts,c.l);memcpy(m->ownership_data,c.ownership,c.o);
    m->layout_size=(uint32_t)c.l;m->ownership_size=(uint32_t)c.o;
    borrowed_original(m);nvm_module_free(m);printf("I executed exact source U8/map input %s.\n",path);
}
int main(int argc,char **argv) {
    CHECK(argc==1||argc==5||argc==9||argc==13);
    borrowed_initializer();borrowed_missing();borrowed_alias_failure();borrowed_calls_prefixes();borrowed_nominal();borrowed_nested();borrowed_context_controls();
    if(argc==1){Input c;map_basic(&c);borrowed_original(&c.m);}
    for(int i=1;i<argc&&i<=4;i++)map_original_module(argv[i]);
    for(int i=5;i<argc&&i<=8;i++)borrowed_source_global(argv[i]);
    for(int i=9;i<argc;i++)borrowed_u8_source(argv[i]);
    CHECK(!ra_live&&!ra_bytes);
    printf("I checked %u borrowed map execution assertions; public boundaries checked.\n",checks);
    return 0;
}
