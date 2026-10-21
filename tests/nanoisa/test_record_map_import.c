/* I check real ordinary state; this query never executes a selected map module.
 * Test-only noinline keeps multiple ~804KB VmState owners out of one main frame. */
#include "../../src/nanovm/record_map_import_private.h"
static __attribute__((noinline)) void import_original(const NvmModule *);
#define MAP_ORIGINAL_CHECK import_original
#define MAP_ORIGIN_MAIN retained_import_origin_controls
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
static __attribute__((noinline)) void import_check(VmState *vm,VmRmImportStatus expected) {
    VmState saved=*vm;size_t bytes=(size_t)vm->global_capacity*sizeof *vm->globals;
    NanoValue *globals=bytes?malloc(bytes):NULL;if(bytes){CHECK(globals);memcpy(globals,vm->globals,bytes);}
    size_t cycle_bytes=(size_t)vm->heap.cycle_count*sizeof *vm->heap.cycle_buf;
    void **cycles=cycle_bytes?malloc(cycle_bytes):NULL;if(cycle_bytes){CHECK(cycles);memcpy(cycles,vm->heap.cycle_buf,cycle_bytes);}
    VmRmImportReport out,sentinel;memset(&sentinel,0xa5,sizeof sentinel);out=sentinel;
    size_t live=ra_live,payload=ra_bytes;
    VmRmImportStatus actual=vm_record_map_import_check(vm->module,vm,&out);
    if(actual!=expected)fprintf(stderr,"import expected%u got%u\n",expected,actual);
    CHECK(actual==expected);CHECK(!memcmp(vm,&saved,sizeof saved));
    CHECK(!bytes||!memcmp(globals,vm->globals,bytes));CHECK(!cycle_bytes||!memcmp(cycles,vm->heap.cycle_buf,cycle_bytes));
    CHECK(ra_live==live&&ra_bytes==payload);
    if(actual!=VM_RM_IMPORT_OK)CHECK(!memcmp(&out,&sentinel,sizeof out));
    else CHECK(out.preparation_bytes<=NVM_RECORD_ARRAY_EXECUTION_BYTES&&out.preparation_work<=NVM_RECORD_ARRAY_EXECUTION_STEPS);
    free(globals);free(cycles);
}
static VmResult import_legacy_execute(VmState *vm) {
    NvmModule *m=(NvmModule *)vm->module;
    uint8_t *layout=m->layout_data,*ownership=m->ownership_data;uint32_t ls=m->layout_size,os=m->ownership_size;
    m->layout_data=NULL;m->layout_size=0;m->ownership_data=NULL;m->ownership_size=0;
    CHECK(nvm_verify(m).ok);VmResult result=vm_execute(vm);
    m->layout_data=layout;m->layout_size=ls;m->ownership_data=ownership;m->ownership_size=os;
    return result;
}
static __attribute__((noinline)) void import_original(const NvmModule *m) {
    VmState vm;vm_init(&vm,m);import_check(&vm,VM_RM_IMPORT_OK);
    NanoValue retained=val_void();
    for(unsigned repeat=0;repeat<2;repeat++) {
        CHECK(import_legacy_execute(&vm)==VM_OK);NanoValue result=vm_get_result(&vm);
        CHECK(result.tag==TAG_INT&&result.as.i64==0);import_check(&vm,VM_RM_IMPORT_INVALID);
        vm.stack_size--;vm_release(&vm.heap,result);import_check(&vm,VM_RM_IMPORT_OK);
        if(vm.global_count) {
            CHECK(vm.globals[0].tag==TAG_STRUCT);
            if(!repeat){retained=vm.globals[0];vm_retain(&vm.heap,retained);}
            else CHECK(retained.as.obj!=vm.globals[0].as.obj&&retained.as.sval->fields[0].as.hashmap->count==1);
        }
    }
    vm_release(&vm.heap,retained);import_check(&vm,VM_RM_IMPORT_OK);
    CHECK(!vm.heap.private_record_dag);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);map_public_refused(m);
}
static __attribute__((noinline)) void import_capacity(void) {
    Input c;map_basic(&c);VmState vm;vm_init(&vm,&c.m);CHECK(vm_ensure_globals(&vm,8));
    CHECK(vm.global_count<vm.global_capacity);vm.globals[7]=val_int(9);
    import_check(&vm,VM_RM_IMPORT_UNRESOLVED);vm.globals[7]=val_void();import_check(&vm,VM_RM_IMPORT_OK);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    /* The proof permits INT in slot0, and ordinary LOAD_GLOBAL reads it even
     * while the public high-water count is zero. Check the actual slot value. */
    init(&c,TAG_INT,false);arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_PUSH_I64);
    b32(c.code,&c.n,17);b32(c.code,&c.n,0);op(&c,OP_EQ);op(&c,OP_ASSERT);
    scalar(&c,TAG_INT);arg32(&c,OP_STORE_GLOBAL,0);map_finish(&c);
    vm_init(&vm,&c.m);CHECK(vm_ensure_globals(&vm,1));vm.global_count=0;vm.globals[0]=val_int(17);
    import_check(&vm,VM_RM_IMPORT_OK);CHECK(import_legacy_execute(&vm)==VM_OK);
    NanoValue result=vm_get_result(&vm);vm.stack_size--;vm_release(&vm.heap,result);
    vm.global_count=0;vm.globals[0]=val_bool(true);import_check(&vm,VM_RM_IMPORT_UNRESOLVED);
    vm.globals[0]=val_void();vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static __attribute__((noinline)) void import_buffered(void) {
    Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);map_finish(&c);
    VmState vm;vm_init(&vm,&c.m);CHECK(import_legacy_execute(&vm)==VM_OK);
    NanoValue result=vm_get_result(&vm);vm.stack_size--;vm_release(&vm.heap,result);
    NanoValue root=vm.globals[0],map=root.as.sval->fields[0];
    vm_retain(&vm.heap,root);vm_release(&vm.heap,root);vm_retain(&vm.heap,map);vm_release(&vm.heap,map);
    CHECK(root.as.sval->header.buffered&&map.as.hashmap->header.buffered);import_check(&vm,VM_RM_IMPORT_OK);
    VmHashMap *dead=vm_hashmap_new(&vm.heap,TAG_STRING,TAG_INT);CHECK(dead);NanoValue deadv=val_hashmap(dead);
    vm_retain(&vm.heap,deadv);vm_release(&vm.heap,deadv);vm_release(&vm.heap,deadv);
    CHECK(dead->header.buffered&&!dead->header.ref_count&&dead->header.colour==VM_GC_BLACK);import_check(&vm,VM_RM_IMPORT_OK);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    /* Actual failed suspect-buffer growth leaves a live unbuffered PURPLE. */
    vm_init(&vm,&c.m);CHECK(import_legacy_execute(&vm)==VM_OK);result=vm_get_result(&vm);vm.stack_size--;vm_release(&vm.heap,result);
    root=vm.globals[0];CHECK(!vm.heap.cycle_count&&!vm.heap.cycle_capacity);
    vm_retain(&vm.heap,root);ra_calls=0;ra_fail=0;vm_release(&vm.heap,root);ra_fail=SIZE_MAX;
    CHECK(!root.as.sval->header.buffered&&root.as.sval->header.colour==VM_GC_PURPLE);import_check(&vm,VM_RM_IMPORT_OK);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}

static __attribute__((noinline)) void import_run_idle(VmState *vm,Input *c) {
    vm_init(vm,&c->m);CHECK(import_legacy_execute(vm)==VM_OK);
    NanoValue result=vm_get_result(vm);CHECK(result.tag==TAG_INT);vm->stack_size--;vm_release(&vm->heap,result);
}
static __attribute__((noinline)) void import_aliases(void) {
    for(unsigned shared=0;shared<2;shared++) {
        Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);
        if(shared)arg32(&c,OP_LOAD_GLOBAL,0);
        else {map_new(&c,TAG_STRING,TAG_INT);record(&c,0);}
        arg32(&c,OP_STORE_GLOBAL,1);map_finish(&c);VmState vm;import_run_idle(&vm,&c);import_check(&vm,VM_RM_IMPORT_OK);
        if(!shared) {
            NanoValue displaced=vm.globals[1];vm_retain(&vm.heap,vm.globals[0]);vm.globals[1]=vm.globals[0];
            /* Identical present shapes do not authorize merging distinct site obligations. */
            import_check(&vm,VM_RM_IMPORT_UNRESOLVED);
            vm.globals[1]=displaced;vm_release(&vm.heap,vm.globals[0]);import_check(&vm,VM_RM_IMPORT_OK);
        }
        vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    }
    Input c;init(&c,TAG_INT,false);
    for(unsigned i=0;i<2;i++){map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);}
    map_finish(&c);VmState vm;import_run_idle(&vm,&c);
    /* Historical alternatives remain ambiguous even if only the last is live. */
    import_check(&vm,VM_RM_IMPORT_UNRESOLVED);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static __attribute__((noinline)) void import_nominal(void) {
    Input c;init(&c,TAG_INT,true);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);
    scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;authority(&c);
    /* UNION layout0 makes record ordinal0 resolve to layout1. */
    c.layouts[32]=TAG_HASHMAP;c.layouts[52]=TAG_HASHMAP;c.o-=48;
    b16(c.ownership,&c.o,5);b16(c.ownership,&c.o,1);b32(c.ownership,&c.o,60);
    b32(c.ownership,&c.o,2);desc(&c,TAG_INT);desc(&c,TAG_STRING);
    b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,1);b32(c.ownership,&c.o,0);b32(c.ownership,&c.o,2);
    for(unsigned i=1;i<=2;i++){b32(c.ownership,&c.o,i);b16(c.ownership,&c.o,0);b16(c.ownership,&c.o,0);b32(c.ownership,&c.o,0);}
    c.m.ownership_size=(uint32_t)c.o;VmState vm;import_run_idle(&vm,&c);
    CHECK(vm.globals[0].as.sval->def_idx==0);import_check(&vm,VM_RM_IMPORT_OK);
    vm.globals[0].as.sval->def_idx=1;import_check(&vm,VM_RM_IMPORT_UNRESOLVED);
    vm.globals[0].as.sval->def_idx=0;vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static __attribute__((noinline)) void import_topology(void) {
    Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);map_finish(&c);
    VmState vm;import_run_idle(&vm,&c);VmHashMap *map=vm.globals[0].as.sval->fields[0].as.hashmap;
    const char keys[2][3]={{'a',0,'x'},{'a',0,'y'}};
    for(unsigned i=0;i<2;i++){VmString *key=vm_string_new(&vm.heap,keys[i],3);CHECK(key);CHECK(vm_hashmap_set(&vm.heap,map,val_string(key),val_int(i+7)));vm_release(&vm.heap,val_string(key));}
    CHECK(map->count==2);import_check(&vm,VM_RM_IMPORT_OK);
    uint32_t occupied[2],n=0,empty=UINT32_MAX;
    for(uint32_t i=0;i<map->bucket_count;i++){if(map->entries[i].state)occupied[n++]=i;else empty=i;}
    CHECK(n==2&&empty!=UINT32_MAX);VmHMEntry original=map->entries[occupied[0]];
    map->count++;import_check(&vm,VM_RM_IMPORT_INVALID);map->count--;
    map->tombstone_count=1;import_check(&vm,VM_RM_IMPORT_INVALID);map->tombstone_count=0;
    original.key.as.string->hash^=1;import_check(&vm,VM_RM_IMPORT_INVALID);original.key.as.string->hash^=1;
    original.key.as.string->data[3]='!';import_check(&vm,VM_RM_IMPORT_INVALID);original.key.as.string->data[3]=0;
    map->entries[occupied[0]].value=val_bool(true);import_check(&vm,VM_RM_IMPORT_INVALID);map->entries[occupied[0]]=original;
    map->entries[empty]=original;map->count++;import_check(&vm,VM_RM_IMPORT_INVALID);memset(&map->entries[empty],0,sizeof map->entries[empty]);map->count--;
    map->entries[empty]=original;memset(&map->entries[occupied[0]],0,sizeof original);import_check(&vm,VM_RM_IMPORT_INVALID);
    map->entries[occupied[0]]=original;memset(&map->entries[empty],0,sizeof original);import_check(&vm,VM_RM_IMPORT_OK);
    /* Real GC buffering followed by restored malformed membership controls. */
    NanoValue root=vm.globals[0];vm_retain(&vm.heap,root);vm_release(&vm.heap,root);CHECK(vm.heap.cycle_count==1);
    vm.heap.cycle_buf[1]=vm.heap.cycle_buf[0];vm.heap.cycle_count=2;import_check(&vm,VM_RM_IMPORT_INVALID);vm.heap.cycle_count=1;
    root.as.sval->header.buffered=0;import_check(&vm,VM_RM_IMPORT_INVALID);root.as.sval->header.buffered=1;
    import_check(&vm,VM_RM_IMPORT_OK);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static __attribute__((noinline)) void import_prefixes(void) {
    Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);map_finish(&c);
    VmState vm;import_run_idle(&vm,&c);
    /* GC-only identities deliberately have no static origin. Force multiple table growths. */
    for(unsigned i=0;i<80;i++){VmHashMap *map=vm_hashmap_new(&vm.heap,TAG_STRING,TAG_INT);CHECK(map);NanoValue v=val_hashmap(map);vm_retain(&vm.heap,v);vm_release(&vm.heap,v);vm_release(&vm.heap,v);}
    CHECK(vm.heap.cycle_count==80);VmRmImportReport report;size_t caller_bytes=ra_bytes;ra_peak=caller_bytes;ra_calls=0;CHECK(vm_record_map_import_check(&c.m,&vm,&report)==VM_RM_IMPORT_OK);size_t reached=ra_calls;
    CHECK(report.collector_objects==80&&report.graph_objects==2);
    CHECK(ra_bytes==caller_bytes&&ra_peak>=caller_bytes&&ra_peak-caller_bytes<=report.preparation_bytes);
    printf("I measured independent query peak%zu bytes excluding caller heap%zu.\n",ra_peak-caller_bytes,caller_bytes);
    size_t live=ra_live,bytes=ra_bytes;
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t position=0;position<reached;position++) {
        VmState saved=vm;VmRmImportReport out,sentinel;memset(&sentinel,0xa5,sizeof sentinel);out=sentinel;
        VmHeapHeader headers[80];for(unsigned j=0;j<80;j++)headers[j]=*(VmHeapHeader *)vm.heap.cycle_buf[j];
        ra_calls=0;ra_fail=position;ra_persistent=(int)persistent;
        VmRmImportStatus status=vm_record_map_import_check(&c.m,&vm,&out);ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(status==VM_RM_IMPORT_MEMORY);CHECK(!memcmp(&saved,&vm,sizeof vm));CHECK(!memcmp(&out,&sentinel,sizeof out));
        for(unsigned j=0;j<80;j++)CHECK(!memcmp(&headers[j],vm.heap.cycle_buf[j],sizeof headers[j]));
        CHECK(ra_live==live&&ra_bytes==bytes);CHECK(vm_record_map_import_check(&c.m,&vm,&out)==VM_RM_IMPORT_OK);
    }
    printf("I denied all %zu import allocations transiently and persistently, including identity growth.\n",reached);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    VmRmImportState charge={0};CHECK(vm_rmi_charge(&charge,NVM_RECORD_ARRAY_EXECUTION_BYTES,NVM_RECORD_ARRAY_EXECUTION_STEPS)==VM_RM_IMPORT_OK);
    CHECK(vm_rmi_charge(&charge,1,0)==VM_RM_IMPORT_LIMIT);CHECK(vm_rmi_charge(&charge,0,1)==VM_RM_IMPORT_LIMIT);
    CHECK(vm_rmi_charge(&charge,UINT64_MAX,UINT64_MAX)==VM_RM_IMPORT_LIMIT);
}
static __attribute__((noinline)) void import_tree(Input *c,unsigned records,unsigned depth,uint8_t leaf) {
    init(c,TAG_INT,false);c->m.struct_count=records;c->fn[0].result_tag=TAG_INT;
    for(unsigned i=1;i<depth;i++)array(c,TAG_ARRAY);
    array(c,leaf);scalar(c,leaf);op(c,OP_ARR_PUSH);
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

static __attribute__((noinline)) void import_arrays_cycles(void) {
    const uint8_t leaves[]={TAG_INT,TAG_U8,TAG_FLOAT,TAG_BOOL,TAG_STRING};
    for(unsigned leaf=0;leaf<5;leaf++)for(unsigned depth=1;depth<=3;depth++) {
        Input c;import_tree(&c,2,depth,leaves[leaf]);VmState vm;import_run_idle(&vm,&c);import_check(&vm,VM_RM_IMPORT_OK);
        VmStruct *child=vm.globals[0].as.sval->fields[0].as.sval;
        VmArray *array=child->fields[0].as.array;
        if(depth>1) {
            VmArray *inner=array->elements[0].as.array;uint32_t length=inner->length;uint8_t type=inner->elem_type;
            inner->length=0;inner->elem_type=type==TAG_BOOL?TAG_INT:TAG_BOOL;import_check(&vm,VM_RM_IMPORT_UNRESOLVED);
            inner->elem_type=type;inner->length=length;import_check(&vm,VM_RM_IMPORT_OK);
            /* Host creates a real owned cycle; refusal must not normalize it. */
            NanoValue displaced=array->elements[0];vm_retain(&vm.heap,val_array(array));
            vm_array_set(array,0,val_array(array));vm_release(&vm.heap,displaced);
            CHECK(array->elements[0].as.array==array);import_check(&vm,VM_RM_IMPORT_UNRESOLVED);
        }
        vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    }
}

static void *import_foreign(void *arg) {
    VmState *vm=arg;VmRmImportReport out,sentinel;memset(&sentinel,0xa5,sizeof sentinel);out=sentinel;
    CHECK(vm_record_map_import_check(vm->module,vm,&out)==VM_RM_IMPORT_INVALID);
    CHECK(!memcmp(&out,&sentinel,sizeof out));return NULL;
}
static __attribute__((noinline)) void import_headers_limits(void) {
    Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);map_finish(&c);
    VmState vm;import_run_idle(&vm,&c);CHECK(vm_ensure_globals(&vm,8));import_check(&vm,VM_RM_IMPORT_OK);
    VmState thread_saved=vm;size_t thread_bytes=ra_bytes,thread_live=ra_live;
    pthread_t other;CHECK(!pthread_create(&other,NULL,import_foreign,&vm));CHECK(!pthread_join(other,NULL));
    CHECK(!memcmp(&thread_saved,&vm,sizeof vm)&&thread_bytes==ra_bytes&&thread_live==ra_live);
    vm.activation_floor=1;import_check(&vm,VM_RM_IMPORT_INVALID);vm.activation_floor=0;
    vm.references.active=1;import_check(&vm,VM_RM_IMPORT_INVALID);vm.references.active=0;
    vm.callee_references.active=1;import_check(&vm,VM_RM_IMPORT_INVALID);vm.callee_references.active=0;
    vm.heap.gc_running=1;import_check(&vm,VM_RM_IMPORT_INVALID);vm.heap.gc_running=0;
    vm.heap.private_record_dag=1;import_check(&vm,VM_RM_IMPORT_INVALID);vm.heap.private_record_dag=0;
    vm.handler_count=1;import_check(&vm,VM_RM_IMPORT_INVALID);vm.handler_count=0;
    VmState saved=vm;
    CHECK(vm_record_map_import_check(&c.m,&vm,(VmRmImportReport *)&vm)==VM_RM_IMPORT_INVALID);CHECK(!memcmp(&saved,&vm,sizeof vm));
    NanoValue oldglobals[8];CHECK(vm.global_capacity>=8);memcpy(oldglobals,vm.globals,sizeof oldglobals);
    CHECK(vm_record_map_import_check(&c.m,&vm,(VmRmImportReport *)vm.globals)==VM_RM_IMPORT_INVALID);CHECK(!memcmp(oldglobals,vm.globals,sizeof oldglobals));
    NvmRecordMapBoundaryPlan *plan=NULL;CHECK(nvm_prepare_record_map_boundary(&c.m,&plan).status==NVM_ARRAY_ELIGIBLE);
    VmRmImportState state={0};VmRmImportReport measured;CHECK(vm_rmi_check_reserved(plan,&vm,&state,&measured,0,0)==VM_RM_IMPORT_OK);ra_test_free(state.table);
    size_t live=ra_live,bytes=ra_bytes;
    for(unsigned which=0;which<2;which++)for(unsigned over=0;over<2;over++) {
        VmRmImportReport out,sentinel;memset(&sentinel,0xa5,sizeof sentinel);out=sentinel;memset(&state,0,sizeof state);
        uint64_t xb=which?0:NVM_RECORD_ARRAY_EXECUTION_BYTES-measured.preparation_bytes+over;
        uint64_t xw=which?NVM_RECORD_ARRAY_EXECUTION_STEPS-measured.preparation_work+over:0;
        VmRmImportStatus status=vm_rmi_check_reserved(plan,&vm,&state,&out,xb,xw);
        CHECK(status==(over?VM_RM_IMPORT_LIMIT:VM_RM_IMPORT_OK));
        if(over)CHECK(!memcmp(&out,&sentinel,sizeof out));
        else CHECK(which?out.preparation_work==NVM_RECORD_ARRAY_EXECUTION_STEPS:out.preparation_bytes==NVM_RECORD_ARRAY_EXECUTION_BYTES);
        ra_test_free(state.table);CHECK(!memcmp(&vm,&saved,sizeof vm));CHECK(ra_live==live&&ra_bytes==bytes);
    }
    memset(&state,0,sizeof state);VmRmImportReport out,sentinel;memset(&sentinel,0xa5,sizeof sentinel);out=sentinel;ra_calls=0;
    CHECK(vm_rmi_check_reserved(plan,&vm,&state,&out,UINT64_MAX,UINT64_MAX)==VM_RM_IMPORT_LIMIT);
    CHECK(!ra_calls&&!state.table&&!memcmp(&out,&sentinel,sizeof out));CHECK(!memcmp(&vm,&saved,sizeof vm));
    printf("I measured import reservation %llu bytes/%llu work and exact-cap/+1 recovery.\n",(unsigned long long)measured.preparation_bytes,(unsigned long long)measured.preparation_work);
    nvm_record_map_boundary_free(plan);import_check(&vm,VM_RM_IMPORT_OK);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
    import_tree(&c,60,3,TAG_INT);import_run_idle(&vm,&c);CHECK(vm_record_map_import_check(&c.m,&vm,&out)==VM_RM_IMPORT_OK);
    CHECK(out.graph_objects==64);printf("I checked actual63-frame tree with64 static/runtime origins.\n");vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static __attribute__((noinline)) void import_source_global(const char *path) {
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
    import_original(m);nvm_module_free(m);printf("I checked real source-global input %s.\n",path);
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);CHECK(argc==1||argc==5||argc==9);
    printf("I reserve import state%zu/table row%zu/report%zu without changing VmState%zu or VmHeap%zu.\n",
        sizeof(VmRmImportState),sizeof(VmRmImportIdentity),sizeof(VmRmImportReport),sizeof(VmState),sizeof(VmHeap));
    import_capacity();import_buffered();import_aliases();import_nominal();import_topology();import_prefixes();import_arrays_cycles();import_headers_limits();
    for(int i=1;i<argc&&i<=4;i++)map_original_module(argv[i]);
    for(int i=5;i<argc;i++)import_source_global(argv[i]);
    CHECK(!ra_live&&!ra_bytes);printf("PASS %u ordinary import query checks; public admission remains refused\n",checks);return 0;
}
