/* I exercise selected STRING values through actual owners and ordinary states. */
#define MAP_RUNTIME_MAIN retained_string_runtime_controls
#include "test_record_map_runtime.c"
#include "../../src/nanoisa/nvm2c_map_private.h"
static void string_rows(const NvmModule *m,NvmRecordMapExecutionPlan *p) {
    NvmRecordMapExecutionCounts counts;CHECK(nvm_record_map_execution_counts(p,&counts));
    CHECK(counts.carrier.functions==m->function_count);unsigned maps=0,visited=0;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        NvmRecordArrayExecutionFunction f;CHECK(nvm_record_map_execution_function(p,fi,&f));
        CHECK(f.signature.code_length==m->functions[fi].code_length);CHECK(f.maximum_stack>0);
        uint32_t pc=0;
        for(uint32_t j=0;j<f.instruction_count;j++) {
            NvmRecordMapExecutionInstruction row;CHECK(nvm_record_map_execution_instruction(p,f.instruction_start+j,&row));
            DecodedInstruction ins;uint32_t size=isa_decode(m->code+f.signature.code_offset+pc,f.signature.code_length-pc,&ins);
            CHECK(size);CHECK(row.base.function==fi&&row.base.pc==pc&&row.base.next_pc==pc+size&&row.base.opcode==ins.opcode);
            if(nvm_record_map_opcode(ins.opcode)) {
                maps++;CHECK(row.base.recipe==NVM_RM_INVALID_BASE_RECIPE);CHECK(row.key_tags==MASK(TAG_STRING)&&row.value_tags==MASK(TAG_STRING));
                CHECK(row.base.pushes==1);CHECK(row.base.obligations&NVM_RA_FIRST_ERROR_CLEANUP);
                switch(ins.opcode) {
                case OP_HM_NEW:CHECK(row.map_operation==NVM_RM_NEW&&row.base.pops==0&&row.result_tags==MASK(TAG_HASHMAP));break;
                case OP_HM_DELETE:CHECK(row.map_operation==NVM_RM_DELETE&&row.base.pops==2&&row.result_tags==MASK(TAG_HASHMAP));break;
                case OP_HM_SET:CHECK(row.map_operation==NVM_RM_SET&&row.base.pops==3&&row.result_tags==MASK(TAG_HASHMAP));break;
                case OP_HM_GET:CHECK(row.map_operation==NVM_RM_GET&&row.base.pops==2&&row.result_tags==(MASK(TAG_STRING)|MASK(TAG_VOID)));break;
                case OP_HM_HAS:CHECK(row.map_operation==NVM_RM_HAS&&row.base.pops==2&&row.result_tags==MASK(TAG_BOOL));break;
                case OP_HM_LEN:CHECK(row.map_operation==NVM_RM_LEN&&row.base.pops==1&&row.result_tags==MASK(TAG_INT));break;
                }
            } else CHECK(row.map_operation==NVM_RM_NONE&&!row.key_tags&&!row.value_tags&&!row.result_tags&&row.base.recipe!=NVM_RM_INVALID_BASE_RECIPE);
            pc+=size;visited++;
        }
        CHECK(pc==f.signature.code_length);
    }
    CHECK(visited==counts.carrier.instructions&&maps>0);
    NvmRecordMapExecutionInstruction bad,saved;memset(&bad,0x5a,sizeof bad);saved=bad;
    CHECK(!nvm_record_map_execution_instruction(p,counts.carrier.instructions,&bad));CHECK(!memcmp(&bad,&saved,sizeof bad));
    map_public_transport(m,true);
}
static void string_original(const char *path) {
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long n=ftell(f);CHECK(n>0&&n<1048576);rewind(f);
    uint8_t *bytes=malloc((size_t)n);CHECK(bytes&&fread(bytes,1,(size_t)n,f)==(size_t)n);CHECK(!fclose(f));
    NvmV2Module wire;CHECK(nvm_v2_module_deserialize(bytes,(size_t)n,&wire)==NVM_V2_OK);
    NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&wire,&m)==NVM_V2_OK);nvm_v2_module_free(&wire);free(bytes);CHECK(nvm_verify(m).ok);
#ifdef NANO_RECORD_MAP_PRIVATE_NATIVE
    char error[512]={0};char *native=nvm2c_emit_record_map_private(m,error,sizeof error);CHECK(native);
    char *public_native=nvm2c_emit(m,error,sizeof error);CHECK(public_native&&!strcmp(public_native,native));free(public_native);free(native);
#endif
    NvmRecordMapExecutionPlan *plan=execution_prepare(m);string_rows(m,plan);
    VmRecordMapPrivate *p=NULL;CHECK(vm_record_map_private_create(plan,&p).status==NVM_ARRAY_ELIGIBLE);
    nvm_record_map_execution_free(plan);
    VmState *vm=malloc(sizeof *vm);CHECK(vm);vm_init(vm,m);
    for(unsigned i=0;i<2;i++){CHECK(vm_execute(vm)==VM_OK);CHECK(vm_get_result(vm).tag==TAG_INT&&vm_get_result(vm).as.i64==0);vm_release(&vm->heap,stack_pop(vm));}
    vm_destroy(vm);free(vm);nvm_module_free(m);
    /* Original and plan destroyed: private owner independently owns its image. */
    CHECK(vm_record_map_private_run(p)==VM_OK&&p->result.tag==TAG_INT&&p->result.as.i64==0);
    CHECK(vm_record_map_private_run(p)==VM_OK);vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}


static void string_authority(Input *c) {
    map_authority(c);
    /* Replace the final kind5 payload with one canonical STRING scalar. */
    CHECK(c->o>=68);c->o-=68;
    b16(c->ownership,&c->o,5);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,52);
    b32(c->ownership,&c->o,1);desc(c,TAG_STRING);
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);
    b32(c->ownership,&c->o,2);
    for(unsigned i=0;i<2;i++){b32(c->ownership,&c->o,i);b16(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);}
    c->m.ownership_size=(uint32_t)c->o;
}
static void string_fixture_mode(Input *c,bool bad_write,bool fail,bool native) {
    init(c,TAG_INT,false);map_new(c,TAG_STRING,TAG_STRING);
    scalar(c,TAG_STRING);scalar(c,bad_write?TAG_INT:TAG_STRING);op(c,OP_HM_SET);
    if(native){op(c,OP_AGG_PACK);b8(c->code,&c->n,AGG_RECORD);b32(c->code,&c->n,0);b16(c->code,&c->n,0);b16(c->code,&c->n,1);}
    else record(c,0);
    arg32(c,OP_STORE_GLOBAL,0);
    arg32(c,OP_LOAD_GLOBAL,0);arg16(c,native?OP_AGG_GET:OP_STRUCT_GET,0);scalar(c,TAG_STRING);op(c,OP_HM_GET);
    scalar(c,TAG_STRING);op(c,OP_EQ);op(c,OP_ASSERT);
    /* An absent lookup remains VOID even after a successful lookup. */
    arg32(c,OP_LOAD_GLOBAL,0);arg16(c,native?OP_AGG_GET:OP_STRUCT_GET,0);arg32(c,OP_PUSH_STR,2);op(c,OP_HM_GET);
    op(c,OP_TYPE_CHECK);b8(c->code,&c->n,TAG_VOID);op(c,OP_ASSERT);
    /* GET consumes the last map owner; the same interned key/value survives
     * through the independently retained result edge. */
    map_new(c,TAG_STRING,TAG_STRING);scalar(c,TAG_STRING);scalar(c,TAG_STRING);op(c,OP_HM_SET);
    scalar(c,TAG_STRING);op(c,OP_HM_GET);scalar(c,TAG_STRING);op(c,OP_EQ);op(c,OP_ASSERT);
    /* This concatenation is absent from the constant pool and other maps.
     * GET must retain it before releasing its only container owner. */
    map_new(c,TAG_STRING,TAG_STRING);scalar(c,TAG_STRING);
    if(native)arg32(c,OP_CALL,1);
    else {scalar(c,TAG_STRING);scalar(c,TAG_STRING);op(c,OP_STR_CONCAT);}
    op(c,OP_HM_SET);
    scalar(c,TAG_STRING);op(c,OP_HM_GET);
    if(native){arg32(c,OP_CALL,1);op(c,OP_POP);} /* Only returned STRING is operand-rooted at this collection point. */
    op(c,OP_STR_LEN);runtime_integer(c,8);op(c,OP_EQ);op(c,OP_ASSERT);
    if(fail){op(c,OP_PUSH_BOOL);b8(c->code,&c->n,0);op(c,OP_ASSERT);}
    scalar(c,TAG_INT);op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;
    if(native){
        c->m.function_count=3;c->fn[1].name_idx=2;c->fn[1].code_offset=(uint32_t)c->n;
        c->fn[1].result_tag=TAG_STRING;c->fn[1].result_count=1;c->fn[1].local_count=1;
        scalar(c,TAG_STRING);scalar(c,TAG_STRING);op(c,OP_STR_CONCAT);arg16(c,OP_STORE_LOCAL,0);
        arg16(c,OP_LOAD_LOCAL,0);arg32(c,OP_TAIL_CALL,2);c->fn[1].code_length=(uint32_t)c->n-c->fn[1].code_offset;
        c->fn[2].name_idx=2;c->fn[2].code_offset=(uint32_t)c->n;c->fn[2].arity=1;c->fn[2].local_count=1;
        c->fn[2].result_tag=TAG_STRING;c->fn[2].result_count=1;c->params[0]=TAG_STRING;c->param_rows[2]=c->params;c->m.function_param_types=c->param_rows;
        arg16(c,OP_LOAD_LOCAL,0);op(c,OP_RET);c->fn[2].code_length=(uint32_t)c->n-c->fn[2].code_offset;
    }
    string_authority(c);
}
static void string_fixture(Input *c,bool bad,bool fail){string_fixture_mode(c,bad,fail,false);}
static __attribute__((noinline)) void string_public(Input *c,bool fail) {
    VmState vm;vm_init(&vm,&c->m);
    for(unsigned run=0;run<2;run++) {
        CHECK(vm_execute(&vm)==(fail?VM_ERR_ASSERT_FAILED:VM_OK));
        CHECK(!vm.frame_count&&vm.stack_size==(fail?0u:1u));
        CHECK(vm.global_capacity&&vm.globals[0].tag==TAG_STRUCT);
        VmHashMap *map=vm.globals[0].as.sval->fields[0].as.hashmap;
        CHECK(map->val_type==TAG_STRING&&map->count==1);
        if(fail)CHECK(!strcmp(vm.error_msg,"Assertion failed"));
        else {CHECK(vm_get_result(&vm).tag==TAG_INT&&vm_get_result(&vm).as.i64==0);vm_release(&vm.heap,stack_pop(&vm));}
    }
    vm_destroy(&vm);CHECK(!ra_live&&!ra_bytes);
}
static void string_private(Input *c,bool fail) {
    VmRecordMapPrivate *p=runtime_instance(c);
    for(unsigned run=0;run<2;run++) {
        CHECK(vm_record_map_private_run(p)==(fail?VM_ERR_ASSERT_FAILED:VM_OK));runtime_idle(p,fail?VM_ERR_ASSERT_FAILED:VM_OK);
        if(!fail)CHECK(p->has_result&&p->result.tag==TAG_INT&&p->result.as.i64==0);
        uint32_t path[]={0};VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};char out[4]={0};
        CHECK(vm_record_map_private_lookup_string(p,VM_RM_GLOBAL,0,path,1,c->strings[1],c->lengths[1],0,4,&w,out)==VM_RM_OBS_OK);
        CHECK(!memcmp(out,c->strings[1],4));uint64_t exact=w.used;
        VmRecordMapObservation observed;VmRmObservationWork missing={0,VM_RM_OBSERVATION_WORK_MAX,false};
        CHECK(vm_record_map_private_lookup(p,VM_RM_GLOBAL,0,path,1,"absent",6,&missing,&observed)==VM_RM_OBS_OK&&observed.tag==TAG_VOID);
        char untouched[4];memset(untouched,0xa5,sizeof untouched);missing=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};
        CHECK(vm_record_map_private_lookup_string(p,VM_RM_GLOBAL,0,path,1,"absent",6,0,0,&missing,untouched)!=VM_RM_OBS_OK);
        for(unsigned j=0;j<4;j++)CHECK((unsigned char)untouched[j]==0xa5);
        for(unsigned invalid=0;invalid<3;invalid++) {
            missing=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};
            uint32_t start=invalid==0?5:invalid==1?UINT32_MAX:1,length=invalid==2?UINT32_MAX:0;
            CHECK(vm_record_map_private_lookup_string(p,VM_RM_GLOBAL,0,path,1,c->strings[1],c->lengths[1],start,length,&missing,untouched)!=VM_RM_OBS_OK);
            for(unsigned j=0;j<4;j++)CHECK((unsigned char)untouched[j]==0xa5);
        }


        w=(VmRmObservationWork){0,exact-1,false};memset(out,0xa5,sizeof out);
        CHECK(vm_record_map_private_lookup_string(p,VM_RM_GLOBAL,0,path,1,c->strings[1],c->lengths[1],0,4,&w,out)==VM_RM_OBS_LIMIT);
        for(unsigned i=0;i<4;i++)CHECK((unsigned char)out[i]==0xa5);
        w=(VmRmObservationWork){0,exact,false};
        CHECK(vm_record_map_private_lookup_string(p,VM_RM_GLOBAL,0,path,1,c->strings[1],c->lengths[1],0,4,&w,out)==VM_RM_OBS_OK&&!memcmp(out,c->strings[1],4));
    }
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void string_prefixes(void) {
    Input c;string_fixture(&c,false,false);
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
    printf("I checked %zu private STRING-value run allocation positions in both denial modes.\n",total);
}

static __attribute__((noinline)) void string_public_prefixes(void) {
    Input c;string_fixture(&c,false,false);VmState vm;vm_init(&vm,&c.m);
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
    printf("I checked %zu complete public STRING-value invocation allocation positions in both denial modes.\n",total);
}

static void string_correspondence(void) {
    Input c;string_fixture(&c,false,false);VmRecordMapPrivate *p=runtime_instance(&c);
    CHECK(vm_record_map_private_run(p)==VM_OK);uint32_t slot=UINT32_MAX;
    uint8_t tag=0xa5;CHECK(!nvm_record_map_execution_uniform_pair(NULL,&tag)&&tag==0xa5);
    CHECK(nvm_record_map_execution_uniform_pair(p->plan,&tag)&&tag==TAG_STRING);
    const NvmRecordMapExecutionPlan *plan=p->plan;uint64_t epoch=p->epoch;NanoValue prior=p->result;
    p->plan=NULL;CHECK(vm_record_map_private_run(p)==VM_ERR_TYPE_ERROR);
    CHECK(p->epoch==epoch&&!p->busy&&p->result.tag==prior.tag&&p->result.as.i64==prior.as.i64);
    p->plan=plan;CHECK(vm_record_map_private_run(p)==VM_OK);

    for(uint32_t i=0;i<p->counts.instructions;i++)if(p->instructions[i].base.opcode==OP_HM_GET){slot=i;break;}
    CHECK(slot!=UINT32_MAX);NvmRecordMapExecutionInstruction saved=p->instructions[slot];
    for(unsigned field=0;field<3;field++) {
        if(field==0)p->instructions[slot].value_tags=MASK(TAG_INT);
        if(field==1)p->instructions[slot].result_tags=MASK(TAG_VOID);
        if(field==2)p->instructions[slot].base.pops=3;
        CHECK(vm_record_map_private_run(p)==VM_ERR_DECODE);runtime_idle(p,VM_ERR_DECODE);
        CHECK(p->has_result&&p->result.tag==TAG_INT&&p->result.as.i64==0);
        p->instructions[slot]=saved;CHECK(vm_record_map_private_run(p)==VM_OK);
    }
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
/* Direct executor whitebox: the full invocation gates above cover preflight.
 * I restore deliberately saturated counters before releasing real owners. */
static void string_get_headroom(void) {
    for(unsigned ordinary=0;ordinary<2;ordinary++) {
        Input c;string_fixture(&c,false,false);VmRecordMapPrivate *p=NULL;VmState *vm;
        if(ordinary){vm=malloc(sizeof *vm);CHECK(vm);vm_init(vm,&c.m);CHECK(vm_execute(vm)==VM_OK);vm_release(&vm->heap,stack_pop(vm));}
        else {p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);vm=&p->vm;}
        NanoValue map=vm->globals[0].as.sval->fields[0];NanoValue key=val_string(vm->module_constants.strings[1]);
        vm_retain(&vm->heap,map);vm_retain(&vm->heap,key);CHECK(stack_push(vm,map)==VM_OK&&stack_push(vm,key)==VM_OK);
        uint32_t refs=((VmHeapHeader *)key.as.obj)->ref_count;((VmHeapHeader *)key.as.obj)->ref_count=UINT32_MAX;
        VmRecordMapInvocation private_view={0};private_view.vm=vm;
        VmRecordMapBorrowedInvocation public_view={0};public_view.vm=vm;
        VmResult result=ordinary?vm_rmb_map_instruction(&public_view,OP_HM_GET,TAG_STRING):vm_rmx_map_instruction(&private_view,OP_HM_GET,TAG_STRING);
        CHECK(result==VM_ERR_MEMORY);
        CHECK(vm->stack_size==(ordinary?2u:0u));CHECK(((VmHeapHeader *)key.as.obj)->ref_count==(ordinary?UINT32_MAX:UINT32_MAX-1));
        ((VmHeapHeader *)key.as.obj)->ref_count=ordinary?refs:refs-1;
        if(!ordinary){vm_retain(&vm->heap,map);vm_retain(&vm->heap,key);CHECK(stack_push(vm,map)==VM_OK&&stack_push(vm,key)==VM_OK);}
        CHECK((ordinary?vm_rmb_map_instruction(&public_view,OP_HM_GET,TAG_STRING):vm_rmx_map_instruction(&private_view,OP_HM_GET,TAG_STRING))==VM_OK);
        CHECK(vm->stack_size==1&&stack_peek(vm,0).tag==TAG_STRING&&stack_peek(vm,0).as.obj==key.as.obj);
        vm_release(&vm->heap,stack_pop(vm));
        if(ordinary){vm_destroy(vm);free(vm);}else vm_record_map_private_destroy(p);
        CHECK(!ra_live&&!ra_bytes);
    }
}
static void string_core_saturation(void) {
    for(unsigned ordinary=0;ordinary<2;ordinary++) {
        Input c;init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_STRING);scalar(&c,TAG_STRING);
        scalar(&c,TAG_STRING);scalar(&c,TAG_STRING);op(&c,OP_STR_CONCAT);op(&c,OP_HM_SET);
        record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_STRUCT_GET,0);
        scalar(&c,TAG_STRING);op(&c,OP_PUSH_BOOL);b8(c.code,&c.n,1);op(&c,OP_ASSERT);
        op(&c,OP_HM_GET);op(&c,OP_STR_LEN);runtime_integer(&c,8);op(&c,OP_EQ);op(&c,OP_ASSERT);
        scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;string_authority(&c);
        VmRecordMapPrivate *p=NULL;VmRecordMapPublicPreparation *prep=NULL;VmState *vm;
        if(ordinary){vm=malloc(sizeof *vm);CHECK(vm);vm_init(vm,&c.m);CHECK(vm_record_map_public_prepare(vm,&prep).status==NVM_ARRAY_ELIGIBLE);CHECK(stack_reserve(vm,VM_RM_ROOT_CAPACITY)==VM_OK);}
        else {p=runtime_instance(&c);vm=&p->vm;p->epoch=1;p->busy=true;vm->heap.private_record_dag=true;}
        VmRecordMapInvocation private_view={p,vm,vm->module,1,0,0,true};
        VmRecordMapBorrowedInvocation public_view={prep,vm,vm->module,0,0,true};
        VmCallFrame *f=&vm->frames[0];memset(f,0,sizeof *f);f->module=vm->module;f->local_count=c.fn[0].local_count;f->owned_callable=val_void();
        for(unsigned i=0;i<f->local_count;i++)vm->stack[vm->stack_size++]=val_void();
        vm->frame_count=1;vm->current_fn=0;vm->ip=0;
        VmTrap trap=vm_core_execute_pair(vm,NULL,NULL,NULL,ordinary?NULL:&private_view,ordinary?&public_view:NULL,TAG_STRING);
        CHECK(trap.type==TRAP_ASSERT&&val_truthy(trap.data.assert_check.condition));vm_release(&vm->heap,trap.data.assert_check.condition);
        NanoValue map=stack_peek(vm,1),key=stack_peek(vm,0),stored=val_void();CHECK(map.tag==TAG_HASHMAP&&key.tag==TAG_STRING);
        for(uint32_t i=0;i<map.as.hashmap->bucket_count;i++)if(map.as.hashmap->entries[i].state==1)stored=map.as.hashmap->entries[i].value;
        CHECK(stored.tag==TAG_STRING&&stored.as.obj!=key.as.obj);uint32_t refs=((VmHeapHeader *)stored.as.obj)->ref_count,size=vm->stack_size,ip=vm->ip;
        if(ordinary){uint32_t index=vm->decoded_module.functions[0].instruction_indices[ip]-1;uint64_t mask=prep->instructions[index].result_tags;prep->instructions[index].result_tags=MASK(TAG_INT)|MASK(TAG_VOID);
            trap=vm_core_execute_pair(vm,NULL,NULL,NULL,NULL,&public_view,TAG_STRING);CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_DECODE&&vm->stack_size==size&&vm->ip==ip);prep->instructions[index].result_tags=mask;vm->last_error=VM_OK;}
        ((VmHeapHeader *)stored.as.obj)->ref_count=UINT32_MAX;
        trap=vm_core_execute_pair(vm,NULL,NULL,NULL,ordinary?NULL:&private_view,ordinary?&public_view:NULL,TAG_STRING);
        CHECK(trap.type==TRAP_ERROR&&trap.data.error.code==VM_ERR_MEMORY);CHECK(vm->stack_size==size-(ordinary?0u:2u));
        CHECK(((VmHeapHeader *)stored.as.obj)->ref_count==UINT32_MAX&&vm->globals[0].as.sval->fields[0].as.obj==map.as.obj);((VmHeapHeader *)stored.as.obj)->ref_count=refs;
        if(ordinary){vm_rmb_unwind(vm);public_view.active=false;vm_record_map_public_preparation_free(prep);CHECK(vm_execute(vm)==VM_OK);vm_destroy(vm);free(vm);}
        else {vm_rmx_unwind(p);private_view.active=false;p->busy=false;CHECK(vm_record_map_private_run(p)==VM_OK);vm_record_map_private_destroy(p);}
        CHECK(!ra_live&&!ra_bytes);
    }
}
static void string_native(const char *directory) {
    for(unsigned variant=0;variant<3;variant++) {
        Input c;
        if(!variant)string_fixture_mode(&c,false,false,true);
        else {
            init(&c,TAG_INT,false);map_new(&c,TAG_STRING,TAG_STRING);
            op(&c,OP_AGG_PACK);b8(c.code,&c.n,AGG_RECORD);b32(c.code,&c.n,0);b16(c.code,&c.n,0);b16(c.code,&c.n,1);
            arg32(&c,OP_STORE_GLOBAL,0);arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_AGG_GET,0);
            scalar(&c,TAG_STRING);op(&c,OP_HM_GET);
            if(variant==1){op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_VOID);op(&c,OP_ASSERT);}
            else {op(&c,OP_STR_LEN);op(&c,OP_POP);}
            scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;string_authority(&c);
        }
        VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==(variant==2?VM_ERR_TYPE_ERROR:VM_OK));
        vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
        char error[512]={0};char *text=nvm2c_emit(&c.m,error,sizeof error);
        if(!text)fprintf(stderr,"SS native variant %u: %s\n",variant,error);
        CHECK(text);
#ifdef NANO_RECORD_MAP_PRIVATE_NATIVE
        char *private_text=nvm2c_emit_record_map_private(&c.m,error,sizeof error);CHECK(private_text&&!strcmp(private_text,text));free(private_text);
#endif
        char path[4096];CHECK(snprintf(path,sizeof path,"%s/string-%u.c",directory,variant)>0);
        FILE *f=fopen(path,"wb");CHECK(f);size_t n=strlen(text);CHECK(fwrite(text,1,n,f)==n);CHECK(!fclose(f));free(text);
    }
}
int main(int argc,char **argv) {
    CHECK(argc==1||argc==5);
    if(getenv("MAP_STRING_NATIVE_DIR"))string_native(getenv("MAP_STRING_NATIVE_DIR"));
    string_core_saturation();string_get_headroom();string_correspondence();string_prefixes();string_public_prefixes();
    for(unsigned fail=0;fail<2;fail++){Input c;string_fixture(&c,false,fail);NvmRecordMapFrozenContext *facts=map_query(&c,NVM_ARRAY_ELIGIBLE);nvm_record_map_frozen_free(facts);string_public(&c,fail);string_private(&c,fail);}
    Input counted;string_fixture(&counted,false,false);counted.strings[1]="le\0f";counted.lengths[1]=4;
    string_public(&counted,false);string_private(&counted,false);
    Input bad;string_fixture(&bad,true,false);map_query(&bad,NVM_ARRAY_UNRESOLVED);
    for(int i=1;i<argc;i++)string_original(argv[i]);
    printf("I checked %u selected STRING-value consumer assertions.\n",checks);return 0;
}
