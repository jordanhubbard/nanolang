/* Selected DELETE qualification; ordinary/private owner boundaries stay distinct. */
#define MAP_RUNTIME_MAIN retained_delete_runtime_neighbors
#include "test_record_map_runtime.c"
static void deletion_fixture(Input *c,bool bad_write,bool fail) {
    init(c,TAG_INT,false);map_new(c,TAG_STRING,TAG_INT);arg16(c,OP_STORE_LOCAL,0);
    arg16(c,OP_LOAD_LOCAL,0);scalar(c,TAG_STRING);scalar(c,bad_write?TAG_BOOL:TAG_INT);op(c,OP_HM_SET);op(c,OP_POP);
    arg16(c,OP_LOAD_LOCAL,0);record(c,0);arg32(c,OP_STORE_GLOBAL,0);
    arg16(c,OP_LOAD_LOCAL,0);scalar(c,TAG_STRING);op(c,OP_HM_DELETE);arg16(c,OP_STORE_LOCAL,1);
    /* Raw instruction returns the same receiver, not source-level VOID. */
    arg16(c,OP_LOAD_LOCAL,0);arg16(c,OP_LOAD_LOCAL,1);op(c,OP_EQ);op(c,OP_ASSERT);
    arg16(c,OP_LOAD_LOCAL,1);scalar(c,TAG_STRING);op(c,OP_HM_HAS);op(c,OP_BOOL_NOT);op(c,OP_ASSERT);
    arg16(c,OP_LOAD_LOCAL,1);scalar(c,TAG_STRING);op(c,OP_HM_GET);op(c,OP_TYPE_CHECK);b8(c->code,&c->n,TAG_VOID);op(c,OP_ASSERT);
    arg16(c,OP_LOAD_LOCAL,1);scalar(c,TAG_STRING);op(c,OP_HM_DELETE);op(c,OP_POP);
    if(fail){op(c,OP_PUSH_BOOL);b8(c->code,&c->n,0);op(c,OP_ASSERT);}
    map_finish(c);
}
static __attribute__((noinline)) void deletion_public(const NvmModule *m,bool fail,bool all_tombstones) {
    VmState vm;vm_init(&vm,m);CHECK(vm_execute(&vm)==(fail?VM_ERR_ASSERT_FAILED:VM_OK));
    CHECK(!vm.frame_count&&vm.stack_size==(fail?0u:1u));
    CHECK(vm.global_capacity&&vm.globals[0].tag==TAG_STRUCT);
    NanoValue old=vm.globals[0];vm_retain(&vm.heap,old);
    VmHashMap *map=old.as.sval->fields[0].as.hashmap;
    CHECK(!map->count&&map->tombstone_count==1);
    if(all_tombstones){for(uint32_t i=0;i<map->bucket_count;i++){CHECK(map->entries[i].key.tag==TAG_VOID&&map->entries[i].value.tag==TAG_VOID);map->entries[i].state=2;}map->tombstone_count=map->bucket_count;}
    if(!fail){CHECK(vm_get_result(&vm).tag==TAG_INT&&vm_get_result(&vm).as.i64==0);vm_release(&vm.heap,stack_pop(&vm));}
    else CHECK(vm.last_error==VM_ERR_ASSERT_FAILED&&!strcmp(vm.error_msg,"Assertion failed"));
    /* Fresh import accepts the tombstoned previous global; old external alias survives replacement. */
    CHECK(vm_execute(&vm)==(fail?VM_ERR_ASSERT_FAILED:VM_OK));
    CHECK(vm.globals[0].as.obj!=old.as.obj&&old.as.sval->fields[0].as.hashmap==map);
    CHECK(!map->count&&map->tombstone_count==(all_tombstones?map->bucket_count:1u));
    vm_release(&vm.heap,old);vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
}
static void deletion_private(Input *c,bool fail) {
    VmRecordMapPrivate *p=runtime_instance(c);CHECK(vm_record_map_private_run(p)==(fail?VM_ERR_ASSERT_FAILED:VM_OK));
    runtime_idle(p,fail?VM_ERR_ASSERT_FAILED:VM_OK);
    if(fail)CHECK(!strcmp(p->vm.error_msg,"Assertion failed"));
    CHECK(p->has_result==!fail&&p->vm.global_capacity&&p->vm.globals[0].tag==TAG_STRUCT);
    VmHashMap *map=p->vm.globals[0].as.sval->fields[0].as.hashmap;CHECK(!map->count&&map->tombstone_count==1);
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};uint32_t path[]={0};VmRecordMapObservation observed;
    CHECK(vm_record_map_private_lookup(p,VM_RM_GLOBAL,0,path,1,"leaf",4,&w,&observed)==VM_RM_OBS_OK&&observed.tag==TAG_VOID);
    CHECK(vm_record_map_private_run(p)==(fail?VM_ERR_ASSERT_FAILED:VM_OK));
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void deletion_original(const char *path) {
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long n=ftell(f);CHECK(n>0&&n<1048576);rewind(f);
    uint8_t *bytes=malloc((size_t)n);CHECK(bytes&&fread(bytes,1,(size_t)n,f)==(size_t)n);CHECK(!fclose(f));
    NvmV2Module wire;CHECK(nvm_v2_module_deserialize(bytes,(size_t)n,&wire)==NVM_V2_OK);
    NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&m)==NVM_V2_OK);nvm_v2_module_free(&wire);free(bytes);CHECK(nvm_verify(m).ok);
    NvmRecordMapExecutionPlan *plan=execution_prepare(m);execution_rows(m,plan);
    VmRecordMapPrivate *p=NULL;CHECK(vm_record_map_private_create(plan,&p).status==NVM_ARRAY_ELIGIBLE);
    nvm_record_map_execution_free(plan);
    VmState *vm=malloc(sizeof *vm);CHECK(vm);vm_init(vm,m);
    for(unsigned i=0;i<2;i++){CHECK(vm_execute(vm)==VM_OK);CHECK(vm_get_result(vm).tag==TAG_INT&&vm_get_result(vm).as.i64==0);vm_release(&vm->heap,stack_pop(vm));}
    vm_destroy(vm);free(vm);nvm_module_free(m);
    /* Original and plan destroyed: private owner independently owns its image. */
    CHECK(vm_record_map_private_run(p)==VM_OK&&p->result.tag==TAG_INT&&p->result.as.i64==0);
    CHECK(vm_record_map_private_run(p)==VM_OK);vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}

static void deletion_prefixes(void) {
    Input c;deletion_fixture(&c,false,false);
    VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    ra_calls=0;CHECK(vm_record_map_private_run(p)==VM_OK);size_t total=ra_calls;
    vm_record_map_private_destroy(p);CHECK(total&&!ra_live&&!ra_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t at=0;at<total;at++) {
        p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
        ra_calls=0;ra_fail=at;ra_persistent=persistent;
        VmResult result=vm_record_map_private_run(p);ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(result==VM_ERR_MEMORY);runtime_idle(p,VM_ERR_MEMORY);
        CHECK(p->has_result&&p->result.tag==TAG_INT&&p->result.as.i64==0);
        CHECK(vm_record_map_private_run(p)==VM_OK);runtime_idle(p,VM_OK);
        vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I checked %zu private DELETE run allocation positions in both denial modes.\n",total);
}

static __attribute__((noinline)) void deletion_public_prefixes(void) {
    Input c;deletion_fixture(&c,false,false);VmState vm;vm_init(&vm,&c.m);
    ra_calls=0;CHECK(vm_execute(&vm)==VM_OK);size_t total=ra_calls;
    vm_destroy(&vm);CHECK(total&&!ra_live&&!ra_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t at=0;at<total;at++) {
        vm_init(&vm,&c.m);ra_calls=0;ra_fail=at;ra_persistent=persistent;
        VmResult result=vm_execute(&vm);ra_fail=SIZE_MAX;ra_persistent=0;
        /* Ordinary suspect buffering allocation can fail without guest failure. */
        CHECK(result==VM_OK||result==VM_ERR_MEMORY);
        if(result==VM_OK){CHECK(vm.stack_size==1&&vm_get_result(&vm).tag==TAG_INT&&vm_get_result(&vm).as.i64==0);vm_release(&vm.heap,stack_pop(&vm));}
        else CHECK(!vm.stack_size&&!vm.frame_count);
        CHECK(vm_execute(&vm)==VM_OK&&vm_get_result(&vm).tag==TAG_INT&&vm_get_result(&vm).as.i64==0);
        vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
    }
    printf("I checked %zu complete public DELETE invocation allocation positions in both denial modes.\n",total);
}

static void deletion_correspondence(void) {
    Input c;deletion_fixture(&c,false,false);VmRecordMapPrivate *p=runtime_instance(&c);
    CHECK(vm_record_map_private_run(p)==VM_OK);uint32_t slot=UINT32_MAX;
    for(uint32_t i=0;i<p->counts.instructions;i++)if(p->instructions[i].base.opcode==OP_HM_DELETE){slot=i;break;}
    CHECK(slot!=UINT32_MAX);NvmRecordMapExecutionInstruction saved=p->instructions[slot];
    for(unsigned field=0;field<3;field++) {
        if(field==0)p->instructions[slot].map_operation=NVM_RM_SET;
        if(field==1)p->instructions[slot].result_tags=MASK(TAG_VOID);
        if(field==2)p->instructions[slot].base.pops=3;
        CHECK(vm_record_map_private_run(p)==VM_ERR_DECODE);runtime_idle(p,VM_ERR_DECODE);
        CHECK(p->has_result&&p->result.tag==TAG_INT&&p->result.as.i64==0);
        p->instructions[slot]=saved;CHECK(vm_record_map_private_run(p)==VM_OK);
    }
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
int main(int argc,char **argv) {
    CHECK(argc==1||argc==5);
    deletion_correspondence();
    deletion_public_prefixes();
    deletion_prefixes();
    execution_failures();
    for(unsigned fail=0;fail<2;fail++){Input c;deletion_fixture(&c,false,fail);NvmRecordMapFrozenContext *p=map_query(&c,NVM_ARRAY_ELIGIBLE);nvm_record_map_frozen_free(p);deletion_public(&c.m,fail,false);deletion_public(&c.m,fail,true);deletion_private(&c,fail);}
    Input bad;deletion_fixture(&bad,true,false);map_query(&bad,NVM_ARRAY_UNRESOLVED);
    for(int i=1;i<argc;i++)deletion_original(argv[i]);
    printf("I checked %u selected DELETE consumer assertions; no publisher test migration.\n",checks);return 0;
}
