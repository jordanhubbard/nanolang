/* Const borrowed-state preparation only; no selected execution. */
#include "../../src/nanovm/record_map_public_prepare_private.h"
static __attribute__((noinline)) void prepare_original(const NvmModule *);
#define MAP_ORIGINAL_CHECK prepare_original
#define MAP_ORIGIN_MAIN retained_public_prepare_origin_controls
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

static VmResult import_legacy_execute(VmState *vm) {
    NvmModule *m=(NvmModule *)vm->module;
    uint8_t *layout=m->layout_data,*ownership=m->ownership_data;uint32_t ls=m->layout_size,os=m->ownership_size;
    m->layout_data=NULL;m->layout_size=0;m->ownership_data=NULL;m->ownership_size=0;
    CHECK(nvm_verify(m).ok);VmResult result=vm_execute(vm);
    m->layout_data=layout;m->layout_size=ls;m->ownership_data=ownership;m->ownership_size=os;
    return result;
}

static __attribute__((noinline)) void prepare_check(VmState *vm,NvmArrayEligibilityStatus expected) {
    VmState saved=*vm;
    size_t live=ra_live,bytes=ra_bytes;
    VmRecordMapPublicPreparation *sentinel=(void *)(uintptr_t)0x1234,*p=sentinel;
    NvmArrayEligibilityResult r=vm_record_map_public_prepare(vm,&p);
    if(r.status!=expected)fprintf(stderr,"prepare expected %u got %u: %s\n",expected,r.status,r.message);
    CHECK(r.status==expected);CHECK(!memcmp(vm,&saved,sizeof saved));
    if(r.status==NVM_ARRAY_ELIGIBLE) {
        CHECK(p!=sentinel&&p->module==vm->module&&p->original_stack==vm->stack);
        CHECK(p->bytes<=NVM_RECORD_ARRAY_EXECUTION_BYTES&&p->work<=NVM_RECORD_ARRAY_EXECUTION_STEPS);
        CHECK(p->rounded_stack_capacity>=VM_RM_ROOT_CAPACITY);
        CHECK(!vm->heap.private_record_dag);
        printf("I prepared %u instructions at %llu bytes / %llu work without executing.\n",
            p->counts.carrier.instructions,(unsigned long long)p->bytes,(unsigned long long)p->work);
        vm_record_map_public_preparation_free(p);
    } else CHECK(p==sentinel);
    CHECK(ra_live==live&&ra_bytes==bytes);
}
static __attribute__((noinline)) void prepare_mutations(void) {
    Input c;map_basic(&c);VmState vm;vm_init(&vm,&c.m);
    NanoValue *stack=vm.stack;vm.stack=NULL;prepare_check(&vm,NVM_ARRAY_INVALID);vm.stack=stack;
    CHECK(vm_ensure_globals(&vm,1));
    size_t live=ra_live,bytes=ra_bytes;
    vm.stack[0]=val_void();NanoValue saved=vm.stack[0];
    CHECK(vm_record_map_public_prepare(&vm,(VmRecordMapPublicPreparation **)(void *)vm.stack).status==NVM_ARRAY_INVALID);
    CHECK(!memcmp(&saved,&vm.stack[0],sizeof saved));
    saved=vm.globals[0];
    CHECK(vm_record_map_public_prepare(&vm,(VmRecordMapPublicPreparation **)(void *)vm.globals).status==NVM_ARRAY_INVALID);
    CHECK(!memcmp(&saved,&vm.globals[0],sizeof saved));
    if(vm.module_constants.count) {
        VmString *str=vm.module_constants.strings[0];
        CHECK(vm_record_map_public_prepare(&vm,(VmRecordMapPublicPreparation **)(void *)vm.module_constants.strings).status==NVM_ARRAY_INVALID);
        CHECK(vm.module_constants.strings[0]==str);
        str->hash^=1;prepare_check(&vm,NVM_ARRAY_INVALID);str->hash^=1;
        str->data[str->length]=1;prepare_check(&vm,NVM_ARRAY_INVALID);str->data[str->length]=0;
    }
    CHECK(ra_live==live&&ra_bytes==bytes);
    VmDispatchInstruction *row=&vm.dispatch_module.functions[0].instructions[0];
    row->call_handle.function_index=1;prepare_check(&vm,NVM_ARRAY_INVALID);row->call_handle.function_index=0;
    uint32_t next=row->next_index;row->next_index=UINT32_MAX;prepare_check(&vm,NVM_ARRAY_INVALID);row->next_index=next;
    uint8_t boundary=vm.decoded_module.functions[0].boundaries[0];
    vm.decoded_module.functions[0].boundaries[0]=0;prepare_check(&vm,NVM_ARRAY_INVALID);
    vm.decoded_module.functions[0].boundaries[0]=boundary;
    uint32_t count=vm.dispatch_module.function_count;
    vm.dispatch_module.function_count=0;prepare_check(&vm,NVM_ARRAY_INVALID);vm.dispatch_module.function_count=count;
    count=vm.decoded_module.function_count;
    vm.decoded_module.function_count=0;prepare_check(&vm,NVM_ARRAY_INVALID);vm.decoded_module.function_count=count;
    count=vm.module_constants.count;
    vm.module_constants.count=count+1;prepare_check(&vm,NVM_ARRAY_INVALID);vm.module_constants.count=count;
    vm.activation_floor=1;prepare_check(&vm,NVM_ARRAY_INVALID);vm.activation_floor=0;
    vm.references.active=1;prepare_check(&vm,NVM_ARRAY_INVALID);vm.references.active=0;
    vm.callee_references.active=1;prepare_check(&vm,NVM_ARRAY_INVALID);vm.callee_references.active=0;
    vm.heap.gc_running=1;prepare_check(&vm,NVM_ARRAY_INVALID);vm.heap.gc_running=0;
    vm.heap.private_record_dag=1;prepare_check(&vm,NVM_ARRAY_INVALID);vm.heap.private_record_dag=0;
    row->super_operand=1;prepare_check(&vm,NVM_ARRAY_INVALID);row->super_operand=0;
    uint32_t target=row->call_target;row->call_target=0;prepare_check(&vm,NVM_ARRAY_INVALID);row->call_target=target;
    uint32_t pc=vm.decoded_module.functions[0].instructions[0].next_byte_offset;
    vm.decoded_module.functions[0].instructions[0].next_byte_offset=0;
    prepare_check(&vm,NVM_ARRAY_INVALID);vm.decoded_module.functions[0].instructions[0].next_byte_offset=pc;
    prepare_check(&vm,NVM_ARRAY_ELIGIBLE);
    CHECK(stack_reserve(&vm,VM_RM_ROOT_CAPACITY)==VM_OK);
    stack=vm.stack;uint32_t capacity=vm.stack_capacity;
    prepare_check(&vm,NVM_ARRAY_ELIGIBLE);CHECK(vm.stack==stack&&vm.stack_capacity==capacity);
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static __attribute__((noinline)) void prepare_original(const NvmModule *m) {
    VmState vm;vm_init(&vm,m);prepare_check(&vm,NVM_ARRAY_ELIGIBLE);
    for(unsigned i=0;i<2;i++) {
        CHECK(import_legacy_execute(&vm)==VM_OK);NanoValue value=vm_get_result(&vm);
        CHECK(value.tag==TAG_INT&&value.as.i64==0);prepare_check(&vm,NVM_ARRAY_INVALID);
        vm.stack_size--;vm_release(&vm.heap,value);prepare_check(&vm,NVM_ARRAY_ELIGIBLE);
    }
    vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);map_public_transport(m,true);
}
static __attribute__((noinline)) void prepare_source_global(const char *path) {
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
    prepare_original(m);nvm_module_free(m);printf("I checked real source-global input %s.\n",path);
}
static __attribute__((noinline)) void prepare_parameters_rounding(void) {
    Input c;init(&c,TAG_INT,false);c.m.function_count=2;
    map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_CALL,1);op(&c,OP_POP);
    scalar(&c,TAG_INT);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;
    c.fn[1].code_offset=(uint32_t)c.n;c.fn[1].arity=1;c.fn[1].local_count=1;
    c.fn[1].result_tag=TAG_STRUCT;c.fn[1].result_count=1;
    c.params[0]=TAG_STRUCT;c.param_rows[1]=c.params;c.m.function_param_types=c.param_rows;
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);c.fn[1].code_length=(uint32_t)c.n-c.fn[1].code_offset;
    map_authority(&c);VmState vm;vm_init(&vm,&c.m);prepare_check(&vm,NVM_ARRAY_ELIGIBLE);
    VmRecordMapPublicPreparation *proof=NULL;CHECK(vm_record_map_public_prepare(&vm,&proof).status==NVM_ARRAY_ELIGIBLE);
    NvmArrayEligibilityStatus status=NVM_ARRAY_INVALID;uint64_t work=proof->work;
    c.params[0]=TAG_ARRAY;CHECK(vm_rmp_charge(proof,&status,0,32u*proof->counts.carrier.functions));
    CHECK(!vm_rmp_functions(proof,&vm,&status));c.params[0]=TAG_STRUCT;
    CHECK(vm_rmp_charge(proof,&status,0,32u*proof->counts.carrier.functions));
    CHECK(vm_rmp_functions(proof,&vm,&status));CHECK(proof->work==work+136u);
    vm_record_map_public_preparation_free(proof);
    c.params[0]=255;prepare_check(&vm,NVM_ARRAY_UNRESOLVED);c.params[0]=TAG_STRUCT;
    prepare_check(&vm,NVM_ARRAY_ELIGIBLE);
    NanoValue *old=vm.stack;uint32_t capacity=vm.stack_capacity;
    vm.stack=NULL;vm.stack_capacity=0;prepare_check(&vm,NVM_ARRAY_ELIGIBLE);vm.stack=old;vm.stack_capacity=capacity;
    NanoValue *odd=ra_test_realloc(vm.stack,32769u*sizeof *odd);CHECK(odd);vm.stack=odd;vm.stack_capacity=32769;
    VmRecordMapPublicPreparation *p=NULL;CHECK(vm_record_map_public_prepare(&vm,&p).status==NVM_ARRAY_ELIGIBLE);
    CHECK(p->rounded_stack_capacity==524304&&vm.stack==odd&&vm.stack_capacity==32769);
    vm_record_map_public_preparation_free(p);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
static __attribute__((noinline)) void prepare_prefixes_limits(void) {
    Input c;map_basic(&c);VmState vm;vm_init(&vm,&c.m);
    for(unsigned i=0;i<80;i++) {
        VmHashMap *map=vm_hashmap_new(&vm.heap,TAG_STRING,TAG_INT);CHECK(map);
        NanoValue v=val_hashmap(map);vm_retain(&vm.heap,v);vm_release(&vm.heap,v);vm_release(&vm.heap,v);
    }
    CHECK(vm.heap.cycle_count==80);
    VmRecordMapPublicPreparation *prior=NULL,*measured=NULL;
    CHECK(vm_record_map_public_prepare(&vm,&prior).status==NVM_ARRAY_ELIGIBLE);
    size_t before=ra_bytes;ra_peak=before;ra_calls=0;
    CHECK(vm_record_map_public_prepare(&vm,&measured).status==NVM_ARRAY_ELIGIBLE);
    size_t reached=ra_calls;uint64_t mb=measured->bytes,mw=measured->work;
    printf("I measured preparation live peak %zu bytes beyond caller/prior %zu against reservation %llu.\n",ra_peak-before,before,(unsigned long long)mb);
    CHECK(ra_peak>=before&&ra_peak-before<=mb);
    vm_record_map_public_preparation_free(measured);
    size_t live=ra_live,bytes=ra_bytes;VmState saved=vm;
    NvmRecordMapExecutionInstruction first=prior->instructions[0];
    VmHeapHeader headers[80];for(unsigned i=0;i<80;i++)headers[i]=*(VmHeapHeader *)vm.heap.cycle_buf[i];
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<reached;i++) {
        VmRecordMapPublicPreparation *out=prior;
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;
        NvmArrayEligibilityResult r=vm_record_map_public_prepare(&vm,&out);
        ra_fail=SIZE_MAX;ra_persistent=0;
        if(r.status!=NVM_ARRAY_MEMORY)fprintf(stderr,"allocation %zu/%zu expected MEMORY got%u\n",i,reached,r.status);
        CHECK(r.status==NVM_ARRAY_MEMORY&&out==prior);
        CHECK(!memcmp(&saved,&vm,sizeof vm)&&!memcmp(&first,prior->instructions,sizeof first));
        for(unsigned j=0;j<80;j++)CHECK(!memcmp(&headers[j],vm.heap.cycle_buf[j],sizeof headers[j]));
        CHECK(ra_live==live&&ra_bytes==bytes);
        out=NULL;CHECK(vm_record_map_public_prepare(&vm,&out).status==NVM_ARRAY_ELIGIBLE);
        vm_record_map_public_preparation_free(out);CHECK(ra_live==live&&ra_bytes==bytes);
    }
    for(unsigned work=0;work<2;work++)for(unsigned over=0;over<2;over++) {
        uint64_t xb=work?0:NVM_RECORD_ARRAY_EXECUTION_BYTES-mb+over;
        uint64_t xw=work?NVM_RECORD_ARRAY_EXECUTION_STEPS-mw+over:0;
        VmRecordMapPublicPreparation *out=prior;
        NvmArrayEligibilityResult r=vm_rmp_prepare_reserved(&vm,&out,xb,xw);
        CHECK(r.status==(over?NVM_ARRAY_LIMIT:NVM_ARRAY_ELIGIBLE));
        if(over)CHECK(out==prior);
        else {CHECK(work?out->work==NVM_RECORD_ARRAY_EXECUTION_STEPS:out->bytes==NVM_RECORD_ARRAY_EXECUTION_BYTES);vm_record_map_public_preparation_free(out);}
        CHECK(!memcmp(&saved,&vm,sizeof vm)&&ra_live==live&&ra_bytes==bytes);
    }
    VmRecordMapPublicPreparation *out=prior;ra_calls=0;
    CHECK(vm_rmp_prepare_reserved(&vm,&out,UINT64_MAX,UINT64_MAX).status==NVM_ARRAY_LIMIT);
    CHECK(!ra_calls&&out==prior&&!memcmp(&saved,&vm,sizeof vm));
    printf("I denied %zu preparation allocations transiently/persistently and recovered, exact caps from %llu/%llu.\n",
        reached,(unsigned long long)mb,(unsigned long long)mw);
    vm_record_map_public_preparation_free(prior);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}

int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);CHECK(argc==1||argc==5||argc==9);
    Input c;map_basic(&c);prepare_original(&c.m);prepare_mutations();prepare_parameters_rounding();prepare_prefixes_limits();
    for(int i=1;i<argc&&i<=4;i++)map_original_module(argv[i]);
    for(int i=5;i<argc;i++)prepare_source_global(argv[i]);
    CHECK(!ra_live&&!ra_bytes);
    printf("I passed %u borrowed-state preparation checks; execution remains refused.\n",checks);
    return 0;
}
