/* Constructor-only map storage checks: no guest entry is exposed or executed. */
#include "../../src/nanovm/record_map_runtime_private.h"
static void preparation_original(const NvmModule *,const NvmRecordMapExecutionPlan *,size_t);
#define MAP_EXECUTION_ORIGINAL_CHECK preparation_original
#define MAP_EXECUTION_MAIN retained_map_execution_controls
#include "test_record_map_execution.c"
int g_argc=0;char **g_argv=NULL;
#ifdef VM_RM_WHITEBOX
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
static VmRecordMapPrivate *retained_originals[8];
static NvmRecordMapExecutionInstruction retained_first[8];
static unsigned retained_count;
static VmRecordMapPrivate *prepare_runtime(const NvmRecordMapExecutionPlan *source) {
    VmRecordMapPrivate *out=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult result=vm_record_map_private_create(source,&out);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"map preparation status=%u %s\n",result.status,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE&&out&&out!=(void *)(uintptr_t)1);
    return out;
}
static void preparation_original(const NvmModule *m,const NvmRecordMapExecutionPlan *source,size_t before) {
    (void)before;
    VmRecordMapPrivate *p=prepare_runtime(source);VmRecordMapPreparation stats;
    CHECK(vm_record_map_private_preparation(p,&stats));CHECK(stats.functions==m->function_count);
    CHECK(stats.image_overlap_bytes<=NVM_RECORD_ARRAY_EXECUTION_BYTES);
    CHECK(stats.image_overlap_work<=NVM_RECORD_ARRAY_EXECUTION_STEPS);
    CHECK(stats.runtime_bytes<stats.image_overlap_bytes&&stats.runtime_work<stats.image_overlap_work);
    for(uint32_t i=0;i<stats.instructions;i++) {
        NvmRecordMapExecutionInstruction a,b;
        CHECK(nvm_record_map_execution_instruction(source,i,&a));
        CHECK(vm_record_map_private_instruction(p,i,&b));
        CHECK(!memcmp(&a,&b,sizeof a));
    }
    printf("I prepared %u functions/%u rows/%u fields: owner=%llu field=%llu layout=%llu runtime=%llu/%llu overlap=%llu/%llu.\n",
        stats.functions,stats.instructions,stats.fields,(unsigned long long)stats.owner_size,
        (unsigned long long)stats.field_size,(unsigned long long)stats.layout_size,
        (unsigned long long)stats.runtime_bytes,(unsigned long long)stats.runtime_work,
        (unsigned long long)stats.image_overlap_bytes,(unsigned long long)stats.image_overlap_work);
    map_public_transport(m,true);
    CHECK(retained_count<8);retained_originals[retained_count]=p;
    CHECK(vm_record_map_private_instruction(p,0,&retained_first[retained_count]));retained_count++;
}
static void preparation_independent(void) {
    Input c;map_basic(&c);NvmRecordMapExecutionPlan *source=execution_prepare(&c.m);
    VmRecordMapPrivate *p=prepare_runtime(source);
    VmRecordMapPreparation stats;CHECK(vm_record_map_private_preparation(p,&stats));
    NvmRecordMapExecutionInstruction first;CHECK(vm_record_map_private_instruction(p,0,&first));
    nvm_record_map_execution_free(source);memset(&c,0xa5,sizeof c);
    NvmRecordMapExecutionInstruction row;CHECK(vm_record_map_private_instruction(p,0,&row)&&!memcmp(&row,&first,sizeof row));
    memset(&row,0x5a,sizeof row);NvmRecordMapExecutionInstruction saved=row;
    CHECK(!vm_record_map_private_instruction(p,stats.instructions,&row)&&!memcmp(&row,&saved,sizeof row));
#ifdef VM_RM_WHITEBOX
    CHECK(!p->vm.verified&&!p->vm.frame_count&&!p->vm.stack_size);
    CHECK(p->plan==nvm_record_map_image_plan(p->image));
#endif
    vm_record_map_private_destroy(p);
}

static void preparation_constant_variants(void) {
    for(unsigned variant=0;variant<4;variant++) {
        Input c;map_basic(&c);char counted[]={'a',0,'z',0};
        if(variant>=1){c.m.function_param_types=c.param_rows;c.param_rows[0]=c.params;}
        if(variant==2){c.strings[1]="";c.lengths[1]=0;c.strings[2]="";c.lengths[2]=0;}
        if(variant==3){c.strings[1]=counted;c.lengths[1]=3;}
        NvmRecordMapExecutionPlan *source=execution_prepare(&c.m);
        VmRecordMapPrivate *p=prepare_runtime(source);nvm_record_map_execution_free(source);
        memset(&c,0xa5,sizeof c);memset(counted,0xa5,sizeof counted);
#ifdef VM_RM_WHITEBOX
        CHECK(p->vm.module_constants.count==4);
        if(variant==2)CHECK(p->vm.module_constants.strings[1]==p->vm.module_constants.strings[2]);
        if(variant==3)CHECK(p->vm.module_constants.strings[1]->length==3&&
            !memcmp(p->vm.module_constants.strings[1]->data,"a\0z",3));
#endif
        vm_record_map_private_destroy(p);
    }
}
#ifdef VM_RM_WHITEBOX
static void preparation_limits(void) {
    Input c;map_basic(&c);NvmRecordMapExecutionPlan *source=execution_prepare(&c.m);
    VmRecordMapPrivate *prior=prepare_runtime(source);VmRecordMapPreparation base;
    CHECK(vm_record_map_private_preparation(prior,&base));size_t bytes=ra_bytes,live=ra_live;
    for(unsigned dimension=0;dimension<2;dimension++) {
        uint64_t extra=dimension?NVM_RECORD_ARRAY_EXECUTION_STEPS-base.image_overlap_work:
                                 NVM_RECORD_ARRAY_EXECUTION_BYTES-base.image_overlap_bytes;
        VmRecordMapPrivate *out=prior;
        CHECK(vm_rm_create_reserved(source,&out,dimension?0:extra,dimension?extra:0).status==NVM_ARRAY_ELIGIBLE);
        VmRecordMapPreparation cost;CHECK(vm_record_map_private_preparation(out,&cost));
        CHECK(dimension?cost.image_overlap_work==NVM_RECORD_ARRAY_EXECUTION_STEPS:
                        cost.image_overlap_bytes==NVM_RECORD_ARRAY_EXECUTION_BYTES);
        vm_record_map_private_destroy(out);CHECK(ra_bytes==bytes&&ra_live==live);
        out=prior;size_t calls=ra_calls;
        CHECK(vm_rm_create_reserved(source,&out,dimension?0:extra+1,dimension?extra+1:0).status==NVM_ARRAY_LIMIT&&out==prior);
        CHECK(ra_bytes==bytes&&ra_live==live);calls=ra_calls;
        CHECK(vm_rm_create_reserved(source,&out,dimension?0:UINT64_MAX,dimension?UINT64_MAX:0).status==NVM_ARRAY_LIMIT&&out==prior);
        CHECK(ra_calls==calls&&ra_bytes==bytes&&ra_live==live);
        CHECK(vm_record_map_private_preparation(prior,&cost));
        uint8_t key=99,value=98;
        CHECK(nvm_record_map_execution_map_field(source,0,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
        out=prepare_runtime(source);vm_record_map_private_destroy(out);
    }
    vm_record_map_private_destroy(prior);nvm_record_map_execution_free(source);CHECK(!ra_live&&!ra_bytes);
}
static void preparation_corrupt(void) {
    Input c;map_basic(&c);NvmRecordMapExecutionPlan *source=execution_prepare(&c.m);
    VmRecordMapPrivate *p=prepare_runtime(source);CHECK(vm_rm_coverage(p,TAG_INT));
    uint32_t fi=0,ii=0;
    VmDispatchInstruction *dispatch=&p->vm.dispatch_module.functions[fi].instructions[ii];
    VmDecodedInstruction *decoded=&p->vm.decoded_module.functions[fi].instructions[ii];
    for(unsigned variant=0;variant<8;variant++) {
        NvmRecordMapExecutionInstruction saved=p->instructions[0];
        VmDispatchInstruction ds=*dispatch;VmDecodedInstruction dec=*decoded;
        /* Coverage is an internal diagnostic. Give repeated scans their same
         * bounded allowance; this is not a constructor/runtime credit API. */
        p->work=0;
        if(variant==0)p->instructions[0].base.recipe=NVM_RA_ROOT_SCALAR;
        if(variant==1)p->instructions[0].map_operation=NVM_RM_NONE;
        if(variant==2)p->instructions[0].key_tags=MASK(TAG_INT);
        if(variant==3)decoded->next_byte_offset++;
        if(variant==4)dispatch->next_index=1234;
        if(variant==5)dispatch->call_handle.resolved=true;
        if(variant==6)p->instructions[0].base.successors[0]++;
        if(variant==7)dispatch->instruction.operands[0].u8=TAG_BOOL;
        CHECK(!vm_rm_coverage(p,TAG_INT));p->instructions[0]=saved;*dispatch=ds;*decoded=dec;
        p->work=0;CHECK(vm_rm_coverage(p,TAG_INT));
    }
    nvm_record_map_execution_free(source);vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static size_t preparation_live_runtime(const VmRecordMapPrivate *p) {
    size_t bytes=sizeof *p+(size_t)p->counts.instructions*sizeof *p->instructions+
        (size_t)p->field_count*sizeof *p->fields+
        (size_t)p->vm.stack_capacity*sizeof *p->vm.stack+
        (size_t)p->vm.global_capacity*sizeof *p->vm.globals+
        (size_t)p->counts.functions*(sizeof(VmDecodedFunction)+sizeof(VmDispatchFunction))+
        (size_t)p->vm.module_constants.count*sizeof *p->vm.module_constants.strings+
        (size_t)p->vm.heap.intern_bucket_count*sizeof *p->vm.heap.intern_buckets;
    for(uint32_t f=0;f<p->counts.functions;f++) {
        const VmDecodedFunction *d=&p->vm.decoded_module.functions[f];
        size_t cap=0;while(cap<d->instruction_count)cap=cap?cap*2:16;
        bytes+=cap*sizeof *d->instructions+(size_t)(d->code_size+1)*9+
               (size_t)d->instruction_count*sizeof(VmDispatchInstruction);
    }
    size_t strings=0;
    for(uint32_t b=0;b<p->vm.heap.intern_bucket_count;b++)
        for(const VmString *v=p->vm.heap.intern_buckets[b];v;v=v->intern_next) {
            bytes+=sizeof *v+v->length+1;strings++;
        }
    CHECK(strings==p->vm.heap.intern_count);return bytes;
}
static void preparation_allocations(void) {
    Input c;map_basic(&c);
    enum { STRINGS=400 };char text[STRINGS][24];char *strings[STRINGS];uint32_t lengths[STRINGS];
    for(unsigned i=0;i<STRINGS;i++) {
        if(i<4){strings[i]=c.strings[i];lengths[i]=c.lengths[i];}
        else {int n=snprintf(text[i],sizeof text[i],"constant-%u",i);CHECK(n>0);strings[i]=text[i];lengths[i]=(uint32_t)n;}
    }
    c.m.strings=strings;c.m.string_lengths=lengths;c.m.string_count=STRINGS;
    NvmRecordMapExecutionPlan *source=execution_prepare(&c.m);
    size_t source_bytes=ra_bytes,source_live=ra_live;
    NvmRecordMapImage *sample=NULL;CHECK(nvm_record_map_image_copy(source,&sample).status==NVM_ARRAY_ELIGIBLE);
    size_t image_bytes=ra_bytes-source_bytes;nvm_record_map_image_free(sample);
    CHECK(ra_bytes==source_bytes&&ra_live==source_live);ra_calls=0;ra_peak=ra_bytes;
    VmRecordMapPrivate *prior=prepare_runtime(source);size_t positions=ra_calls;
    VmRecordMapPreparation stats;CHECK(vm_record_map_private_preparation(prior,&stats));
    CHECK(ra_peak<=stats.image_overlap_bytes);CHECK(prior->vm.heap.intern_bucket_count==1024);
    CHECK(prior->vm.heap.intern_count==STRINGS);
    size_t retained=ra_bytes-source_bytes,prior_bytes=ra_bytes,prior_live=ra_live;
    CHECK(retained==image_bytes+preparation_live_runtime(prior));
    CHECK(retained<stats.image_overlap_bytes);
    for(unsigned persistent=0;persistent<2;persistent++)for(size_t i=0;i<positions;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)persistent;ra_peak=ra_bytes;
        VmRecordMapPrivate *out=prior;NvmArrayEligibilityResult result=vm_record_map_private_create(source,&out);
        if(result.status==NVM_ARRAY_ELIGIBLE) {
            /* Intern table growth failure is deliberately nonfatal: denser
             * chains retain exactly the same constants and correct lifetime. */
            CHECK(out!=prior&&out->vm.module_constants.count==STRINGS);
            CHECK(out->vm.heap.intern_count==STRINGS);
            CHECK(ra_bytes-prior_bytes==image_bytes+preparation_live_runtime(out));
            CHECK(ra_peak-prior_bytes+source_bytes<=stats.image_overlap_bytes);
            vm_record_map_private_destroy(out);
        } else CHECK(result.status==NVM_ARRAY_MEMORY&&out==prior);
        CHECK(ra_bytes==prior_bytes&&ra_live==prior_live);
        ra_fail=SIZE_MAX;ra_persistent=0;
        uint8_t key=99,value=98;CHECK(nvm_record_map_execution_map_field(source,0,0,&key,&value)&&key==TAG_STRING&&value==TAG_INT);
        CHECK(vm_record_map_private_preparation(prior,&stats));
        out=prepare_runtime(source);vm_record_map_private_destroy(out);
        CHECK(ra_bytes==prior_bytes&&ra_live==prior_live);
    }
    vm_record_map_private_destroy(prior);CHECK(ra_bytes==source_bytes&&ra_live==source_live);
    nvm_record_map_execution_free(source);CHECK(!ra_live&&!ra_bytes);
    printf("I checked %zu preparation allocation prefixes in both modes, including intern growth, retained %zu bytes.\n",positions,retained);
}
#endif
int main(int argc,char **argv) {
    retained_map_execution_controls(argc,argv);
    for(unsigned i=0;i<retained_count;i++) {
        NvmRecordMapExecutionInstruction row;
        CHECK(vm_record_map_private_instruction(retained_originals[i],0,&row)&&!memcmp(&row,&retained_first[i],sizeof row));
        vm_record_map_private_destroy(retained_originals[i]);
    }
    preparation_independent();preparation_constant_variants();
#ifdef VM_RM_WHITEBOX
    CHECK(!ra_live&&!ra_bytes);preparation_limits();preparation_corrupt();preparation_allocations();
#endif
    printf("I checked private map preparation and independence; execution remains refused.\n");
    return 0;
}
