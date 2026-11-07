/* Preparation only: no selected executor is present. */
#include "../../src/nanovm/union_field_runtime_private.h"
#define UFE_MAIN retained_union_recipe_controls
#include "test_union_field_execution.c"
#include "record_array_alloc.h"
int g_argc=0;char **g_argv=NULL;
#ifdef VM_UF_WHITEBOX
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
static VmUnionFieldPrivate *preparation_query(const NvmModule *m) {
    VmUnionFieldPrivate *p=(void *)(uintptr_t)1;
    NvmArrayEligibilityResult r=vm_union_field_private_prepare(m,&p);
    if(r.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"union preparation status=%u: %s\n",r.status,r.message);
    CHECK(r.status==NVM_ARRAY_ELIGIBLE&&p&&p!=(void *)(uintptr_t)1);return p;
}
static void preparation_check(const VmUnionFieldPrivate *p) {
    VmUnionFieldPreparation s;CHECK(vm_union_field_private_preparation(p,&s));
    CHECK(s.functions&&s.instructions&&s.stack_capacity==VM_MAX_FRAMES*512u+1u);
    CHECK(s.image_overlap_bytes<=NVM_RECORD_ARRAY_EXECUTION_BYTES&&s.image_overlap_work<=NVM_RECORD_ARRAY_EXECUTION_STEPS);
    CHECK(s.runtime_bytes<s.image_overlap_bytes&&s.runtime_work<s.image_overlap_work);
#ifdef VM_UF_WHITEBOX
    CHECK(p->vm.module==p->vm.root_module&&p->vm.module==nvm_union_field_image_module_internal(p->image));
    CHECK(!p->vm.verified&&!p->vm.stack_size&&!p->vm.frame_count&&!p->vm.heap.gc_running);
    CHECK(!p->vm.dispatch_profile.fuse_load_local_field&&p->record_layout==1&&p->union_layout==0);
    CHECK(p->vm.stack[0].tag==TAG_VOID&&p->vm.stack[s.stack_capacity-1].tag==TAG_VOID);
    CHECK(p->vm.heap.intern_bucket_count>=256&&(p->vm.heap.stats.num_objects==0)==(p->vm.module_constants.count==0));
    CHECK(vm_uf_coverage((VmUnionFieldPrivate *)p));
#endif
}
static void preparation_basic(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);
#ifdef VM_UF_WHITEBOX
    CHECK(!ra_live&&!ra_bytes);ra_peak=0;
#endif
    printf("union unchanged VmState=%zu VmHeap=%zu NanoValue=%zu\n",sizeof(VmState),sizeof(VmHeap),sizeof(NanoValue));
    VmUnionFieldPrivate *p=preparation_query(&v.c.module);VmUnionFieldPreparation s;
    CHECK(vm_union_field_private_preparation(p,&s));preparation_check(p);
    memset(&v,0xa5,sizeof v);memset(code,0xa5,sizeof code);preparation_check(p);
    printf("union preparation owner=%llu instruction=%llu runtime=%llu/%llu image=%llu/%llu source=%llu/%llu stack=%u\n",
        (unsigned long long)s.owner_size,(unsigned long long)s.instruction_size,
        (unsigned long long)s.runtime_bytes,(unsigned long long)s.runtime_work,
        (unsigned long long)s.image_overlap_bytes,(unsigned long long)s.image_overlap_work,
        (unsigned long long)s.source_bytes_reserved,(unsigned long long)s.source_work_reserved,s.stack_capacity);
#ifdef VM_UF_WHITEBOX
    printf("union actual retained=%zu peak=%zu calls=%zu allocations=%zu\n",ra_bytes,ra_peak,ra_calls,ra_live);
    CHECK(ra_bytes<=s.runtime_bytes+s.image_overlap_bytes&&ra_peak<=s.image_overlap_bytes);
#endif
    vm_union_field_private_destroy(p);vm_union_field_private_destroy(NULL);
#ifdef VM_UF_WHITEBOX
    CHECK(!ra_live&&!ra_bytes);
#endif
}
static void preparation_original(const char *path) {
    FILE *file=fopen(path,"rb");CHECK(file);CHECK(!fseek(file,0,SEEK_END));
    long size=ftell(file);CHECK(size>0&&(unsigned long)size<=UINT32_MAX);rewind(file);
    uint8_t *bytes=malloc((size_t)size);CHECK(bytes);
    CHECK(fread(bytes,1,(size_t)size,file)==(size_t)size);CHECK(!fclose(file));
    NvmV2Module original_wire;memset(&original_wire,0,sizeof original_wire);
    CHECK(nvm_v2_module_deserialize(bytes,(size_t)size,&original_wire)==NVM_V2_OK);
    NvmModule *m=NULL;CHECK(nvm_v2_to_nvm_module(&original_wire,&m)==NVM_V2_OK);
    CHECK(m&&nvm_verify(m).ok);
    CHECK(!m->layout_size&&!m->ownership_size&&m->struct_count==1&&m->union_count==1&&m->string_count>=6);
    UnionCase v;union_case(&v);Case *c=&v.c;size_t suffix=v.local+8;
    uint8_t extension[64];size_t extension_size=c->o-suffix;CHECK(extension_size<=sizeof extension);
    memcpy(extension,c->ownership+suffix,extension_size);c->o=12;
    u32(c->ownership,&c->o,m->function_count);
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *f=&m->functions[i];CHECK(f->local_count<64);
        u16(c->ownership,&c->o,f->local_count);u16(c->ownership,&c->o,f->arity);
        proof_desc(c->ownership,&c->o,f->result_count?f->result_tag:TAG_VOID);
        for(uint32_t j=0;j<f->local_count;j++)proof_desc(c->ownership,&c->o,
            j<f->arity&&m->function_param_types&&m->function_param_types[i]?m->function_param_types[i][j]:TAG_VOID);
    }
    CHECK(c->o+extension_size<=sizeof c->ownership);memcpy(c->ownership+c->o,extension,extension_size);c->o+=extension_size;
    m->layout_data=c->layouts;m->layout_size=(uint32_t)c->l;
    m->ownership_data=c->ownership;m->ownership_size=(uint32_t)c->o;
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
    VmUnionFieldPrivate *p=preparation_query(m);preparation_check(p);
    VmUnionFieldPreparation stats;CHECK(vm_union_field_private_preparation(p,&stats));
    printf("prepared original %s rows=%u functions=%u image=%llu/%llu\n",path,stats.instructions,stats.functions,
        (unsigned long long)stats.image_overlap_bytes,(unsigned long long)stats.image_overlap_work);
    m->layout_data=NULL;m->layout_size=0;m->ownership_data=NULL;m->ownership_size=0;
    nvm_module_free(m);nvm_v2_module_free(&original_wire);free(bytes);memset(&v,0,sizeof v);
    preparation_check(p);vm_union_field_private_destroy(p);
}
#ifdef VM_UF_WHITEBOX
static size_t runtime_retained(const VmUnionFieldPrivate *p) {
    const VmState *v=&p->vm;size_t bytes=sizeof *p+(size_t)p->counts.instructions*sizeof *p->instructions;
    bytes+=(size_t)v->stack_capacity*sizeof *v->stack+(size_t)v->global_capacity*sizeof *v->globals;
    bytes+=(size_t)v->decoded_module.function_count*sizeof *v->decoded_module.functions;
    bytes+=(size_t)v->dispatch_module.function_count*sizeof *v->dispatch_module.functions;
    for(uint32_t f=0;f<p->counts.functions;f++) {
        const VmDecodedFunction *d=&v->decoded_module.functions[f];const VmDispatchFunction *t=&v->dispatch_module.functions[f];
        bytes+=((size_t)d->code_size+1)*(sizeof *d->boundaries+sizeof *d->instruction_indices+sizeof *t->offset_to_index);
        size_t cap=d->instruction_count?16:0;while(cap<d->instruction_count)cap*=2;
        bytes+=cap*sizeof *d->instructions+(size_t)t->instruction_count*sizeof *t->instructions;
    }
    bytes+=(size_t)v->module_constants.count*sizeof *v->module_constants.strings;
    bytes+=(size_t)v->heap.intern_bucket_count*sizeof *v->heap.intern_buckets;
    for(uint32_t b=0;b<v->heap.intern_bucket_count;b++)for(VmString *x=v->heap.intern_buckets[b];x;x=x->intern_next)
        bytes+=sizeof *x+(size_t)x->length+1;
    CHECK(!v->heap.cycle_buf&&!v->heap.cycle_count&&!v->heap.cycle_capacity);return bytes;
}
static void preparation_oracle(const NvmModule *m) {
    CHECK(!ra_live&&!ra_bytes);ra_peak=0;
    NvmUnionFieldExecutionPlan *source=execution_query(m);size_t source_peak=ra_peak;
    NvmUnionFieldImage *image=NULL;CHECK(nvm_union_field_image_copy(source,&image).status==NVM_ARRAY_ELIGIBLE);
    size_t image_peak=ra_peak;nvm_union_field_execution_free(source);size_t image_retained=ra_bytes;
    nvm_union_field_image_free(image);CHECK(!ra_bytes&&!ra_live);ra_peak=0;
    VmUnionFieldPrivate *p=preparation_query(m);VmUnionFieldPreparation stats;CHECK(vm_union_field_private_preparation(p,&stats));
    size_t physical=runtime_retained(p);CHECK(ra_bytes==physical+image_retained);
    CHECK(ra_peak<=stats.image_overlap_bytes&&physical<=stats.runtime_bytes);
    printf("union independent image=%zu runtime=%zu retained=%zu peak=%zu source_peak=%zu image_peak=%zu\n",
        image_retained,physical,ra_bytes,ra_peak,source_peak,image_peak);
    vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void preparation_prefixes(void) {
    UnionCase v;uint8_t code[128];size_t n=execution_case(&v,code);UnionCase saved=v;uint8_t saved_code[128];memcpy(saved_code,code,n);
    CHECK(!ra_live&&!ra_bytes);ra_calls=0;VmUnionFieldPrivate *p=preparation_query(&v.c.module);size_t positions=ra_calls;
    vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    VmUnionFieldPrivate *prior=preparation_query(&v.c.module);size_t live=ra_live,bytes=ra_bytes;
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<positions;i++) {
        p=prior;ra_calls=0;ra_fail=i;ra_persistent=(int)mode;
        NvmArrayEligibilityResult r=vm_union_field_private_prepare(&v.c.module,&p);
        if(r.status!=NVM_ARRAY_MEMORY)fprintf(stderr,"union preparation prefix=%zu mode=%u status=%u %s\n",i,mode,r.status,r.message);
        CHECK(r.status==NVM_ARRAY_MEMORY&&p==prior&&ra_live==live&&ra_bytes==bytes);
        CHECK(!memcmp(&v,&saved,sizeof v)&&!memcmp(code,saved_code,n));
        ra_fail=SIZE_MAX;ra_persistent=0;preparation_check(prior);
        p=preparation_query(&v.c.module);preparation_check(p);vm_union_field_private_destroy(p);CHECK(ra_live==live&&ra_bytes==bytes);
    }
    vm_union_field_private_destroy(prior);CHECK(!ra_live&&!ra_bytes);
    printf("union preparation prefixes=%zu transient/persistent with valid retry\n",positions);
}
static void preparation_constants(void) {
    for(unsigned mode=0;mode<4;mode++) {
        UnionCase v;uint8_t code[128];execution_case(&v,code);
        char counted[]={'a',0,'z',0};char *strings[8];uint32_t lengths[8];
        for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
        strings[6]=strings[7]=mode==1?"":mode==2?counted:"duplicate";
        lengths[6]=lengths[7]=mode==1?0:mode==2?3:9;
        uint8_t zero=0;uint8_t *parameters[1]={mode==3?&zero:NULL};
        v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=8;
        if(mode==3)v.c.module.function_param_types=parameters;
        preparation_oracle(&v.c.module);
        VmUnionFieldPrivate *p=preparation_query(&v.c.module);
        CHECK(p->vm.module_constants.strings[6]==p->vm.module_constants.strings[7]);
        CHECK(p->vm.module_constants.strings[6]->length==lengths[6]);
        CHECK(!memcmp(p->vm.module_constants.strings[6]->data,strings[6],lengths[6]));
        CHECK(p->vm.module_constants.strings[6]->header.ref_count==2);
        vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
    UnionCase v;uint8_t code[128];execution_case(&v,code);v.c.module.string_count=0;
    VmUnionFieldPrivate *out=(void *)(uintptr_t)1;
    CHECK(vm_union_field_private_prepare(&v.c.module,&out).status!=NVM_ARRAY_ELIGIBLE&&out==(void *)(uintptr_t)1);
    CHECK(!ra_live&&!ra_bytes);
    /* Actual zero-constant primitive, not a valid zero-name union module. */
    VmState *vm=calloc(1,sizeof *vm);CHECK(vm);vm_heap_init(&vm->heap);
    NvmModule empty={0};CHECK(vm->heap.intern_bucket_count==256&&vm->heap.intern_buckets);
    CHECK(vm_module_constants_build(vm,&empty,&vm->module_constants));
    CHECK(!vm->module_constants.count&&!vm->module_constants.strings);
    vm_heap_destroy(&vm->heap);free(vm);CHECK(!ra_live&&!ra_bytes);
}
static void preparation_growth(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);
    enum {COUNT=406};char storage[COUNT-6][24];char *strings[COUNT];uint32_t lengths[COUNT];
    for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
    for(unsigned i=6;i<COUNT;i++) {
        int n=snprintf(storage[i-6],sizeof storage[i-6],"growth-value-%u",i);CHECK(n>0&&n<(int)sizeof storage[i-6]);
        strings[i]=storage[i-6];lengths[i]=(uint32_t)n;
    }
    v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=COUNT;
    preparation_oracle(&v.c.module);ra_calls=0;VmUnionFieldPrivate *p=preparation_query(&v.c.module);size_t positions=ra_calls;
    CHECK(p->vm.heap.intern_bucket_count==1024&&p->vm.heap.intern_count==COUNT);
    vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    VmUnionFieldPrivate *prior=preparation_query(&v.c.module);size_t live=ra_live,bytes=ra_bytes;unsigned optional=0;
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<positions;i++) {
        p=prior;ra_calls=0;ra_fail=i;ra_persistent=(int)mode;
        NvmArrayEligibilityResult r=vm_union_field_private_prepare(&v.c.module,&p);
        if(r.status==NVM_ARRAY_ELIGIBLE) {
            CHECK(!mode&&p!=prior&&ra_calls>i);optional++;
            CHECK(p->vm.heap.intern_count==COUNT&&p->vm.module_constants.count==COUNT);
            for(unsigned j=0;j<COUNT;j++)CHECK(p->vm.module_constants.strings[j]->length==lengths[j]&&
                !memcmp(p->vm.module_constants.strings[j]->data,strings[j],lengths[j]));
            vm_union_field_private_destroy(p);
        } else {
            if(r.status!=NVM_ARRAY_MEMORY)fprintf(stderr,"growth prefix=%zu mode=%u status=%u %s\n",i,mode,r.status,r.message);
            CHECK(r.status==NVM_ARRAY_MEMORY&&p==prior);
        }
        CHECK(ra_live==live&&ra_bytes==bytes);ra_fail=SIZE_MAX;ra_persistent=0;preparation_check(prior);
        p=preparation_query(&v.c.module);CHECK(p->vm.heap.intern_bucket_count==1024&&p->vm.heap.intern_count==COUNT);
        vm_union_field_private_destroy(p);CHECK(ra_live==live&&ra_bytes==bytes);
    }
    CHECK(optional==2);vm_union_field_private_destroy(prior);CHECK(!ra_live&&!ra_bytes);
    printf("union growth constants=%u prefixes=%zu transient/persistent optional-growth-successes=%u\n",COUNT,positions,optional);
}
static void preparation_calls(bool tail,bool blocked) {
    UnionCase v;union_case(&v);uint8_t code[256];size_t n=0;
    NvmFunctionEntry functions[3]={{0}};functions[0].result_count=1;functions[0].result_tag=TAG_INT;
    u8(code,&n,OP_CALL);u32(code,&n,1);
    if(blocked){proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);}
    proof_pack(code,&n,AGG_RECORD,0,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);functions[0].code_length=(uint32_t)n;
    functions[1].code_offset=(uint32_t)n;functions[1].name_idx=1;
    if(blocked) {
        /* This callee never returns. The caller's union use/RET must not be
         * interpreted as unreachable evidence for a successful proof. */
        u8(code,&n,OP_JMP);u32(code,&n,0);
    } else {
        functions[1].result_count=1;functions[1].result_tag=TAG_UNION;
        if(tail){u8(code,&n,OP_TAIL_CALL);u32(code,&n,2);}
        else {proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);u8(code,&n,OP_RET);}
    }
    functions[1].code_length=(uint32_t)n-functions[1].code_offset;
    if(tail&&!blocked) {
        functions[2].code_offset=(uint32_t)n;functions[2].name_idx=2;
        functions[2].result_count=1;functions[2].result_tag=TAG_UNION;
        proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);u8(code,&n,OP_RET);
        functions[2].code_length=(uint32_t)n-functions[2].code_offset;
    }
    proof_header(&v,code,n);proof_functions(&v,functions,tail&&!blocked?3:2);
    if(blocked) {
        VmUnionFieldPrivate *out=(void *)(uintptr_t)1;
        CHECK(vm_union_field_private_prepare(&v.c.module,&out).status==NVM_ARRAY_UNRESOLVED&&out==(void *)(uintptr_t)1);
    } else {
        VmUnionFieldPrivate *p=preparation_query(&v.c.module);preparation_check(p);vm_union_field_private_destroy(p);
    }
    CHECK(!ra_live&&!ra_bytes);
}
static void preparation_variants(void) {
    for(unsigned mode=0;mode<5;mode++) {
        UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
        if(mode==1||mode==2){v.c.ownership[v.variant+6]=0;v.c.ownership[v.variant+12]=0;v.c.ownership[v.variant+14]=1;}
        if(mode<3) {
            if(mode==1){proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,1,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);}
            else {proof_pack(code,&n,AGG_VARIANT,mode==2?0:1,0);u8(code,&n,OP_AGG_TAG);}
            u8(code,&n,OP_RET);
        } else if(mode==3) {
            u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);size_t branch=n;u8(code,&n,OP_JMP_FALSE);u32(code,&n,0);
            proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
            patch(code,branch+1,(uint32_t)(n-branch));u8(code,&n,OP_HALT);
            proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_POP);
        } else {
            proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
            u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);u8(code,&n,OP_LOAD_GLOBAL);u32(code,&n,0);
            u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
        }
        proof_header(&v,code,n);preparation_oracle(&v.c.module);
        VmUnionFieldPrivate *p=preparation_query(&v.c.module);preparation_check(p);
        if(mode==4)CHECK(p->vm.global_count==1&&p->vm.globals[0].tag==TAG_VOID);
        if(mode==3) {
            unsigned proved=0,unestablished=0;
            for(uint32_t j=0;j<p->counts.instructions;j++) {
                if(p->instructions[j].site.operation==UFS_HALT&&p->instructions[j].site.state==UFS_PROVED)proved++;
                if(p->instructions[j].site.state==UFS_NOT_ESTABLISHED)unestablished++;
            }
            CHECK(proved==1&&unestablished==1);
        }
        vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
    preparation_calls(false,false);preparation_calls(true,false);preparation_calls(false,true);
}
static void preparation_corruption(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);VmUnionFieldPrivate *p=preparation_query(&v.c.module);
    NvmUnionFieldExecutionInstruction old=p->instructions[1];p->instructions[1].site.layout=1;
    CHECK(!vm_uf_coverage(p));p->instructions[1]=old;CHECK(vm_uf_coverage(p));
    p->instructions[1].base.operand_bits[MAX_OPERANDS-1]^=1;CHECK(!vm_uf_coverage(p));p->instructions[1]=old;
    VmDispatchInstruction *d=&p->vm.dispatch_module.functions[0].instructions[1];VmDispatchInstruction save=*d;
    d->super_op=VM_SUPER_LOAD_LOCAL_FIELD;CHECK(!vm_uf_coverage(p));*d=save;
    d->call_handle.module=p->vm.module;CHECK(!vm_uf_coverage(p));*d=save;
    p->functions[0].maximum_stack++;CHECK(!vm_uf_coverage(p));p->functions[0].maximum_stack--;
    p->vm.decoded_module.functions[0].boundaries[1]^=1;CHECK(!vm_uf_coverage(p));p->vm.decoded_module.functions[0].boundaries[1]^=1;
    CHECK(vm_uf_coverage(p));vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void preparation_reservations(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);NvmUnionFieldExecutionPlan *source=execution_query(&v.c.module);
    NvmUnionFieldImage *prior=NULL;CHECK(nvm_union_field_image_copy(source,&prior).status==NVM_ARRAY_ELIGIBLE);
    NvmUnionFieldImageCost c;CHECK(nvm_union_field_image_cost(prior,&c));size_t live=ra_live,bytes=ra_bytes;
    for(unsigned dimension=0;dimension<2;dimension++) {
        uint64_t extra=dimension?NVM_RECORD_ARRAY_EXECUTION_STEPS-c.preparation_work_reserved:NVM_RECORD_ARRAY_EXECUTION_BYTES-c.preparation_bytes_reserved;
        NvmUnionFieldImage *p=prior;CHECK(nvm_union_field_image_copy_reserved(source,dimension?0:extra,dimension?extra:0,&p).status==NVM_ARRAY_ELIGIBLE);
        NvmUnionFieldImageCost exact;CHECK(nvm_union_field_image_cost(p,&exact));
        CHECK(dimension?exact.preparation_work_reserved==NVM_RECORD_ARRAY_EXECUTION_STEPS:exact.preparation_bytes_reserved==NVM_RECORD_ARRAY_EXECUTION_BYTES);
        nvm_union_field_image_free(p);CHECK(ra_live==live&&ra_bytes==bytes);
        p=prior;size_t calls=ra_calls;
        CHECK(nvm_union_field_image_copy_reserved(source,dimension?0:extra+1,dimension?extra+1:0,&p).status==NVM_ARRAY_LIMIT&&p==prior);
        CHECK(ra_live==live&&ra_bytes==bytes);
        printf("union image final-cap+1 dimension=%u reversible allocations=%zu\n",dimension,ra_calls-calls);
        calls=ra_calls;
        CHECK(nvm_union_field_image_copy_reserved(source,dimension?0:UINT64_MAX,dimension?UINT64_MAX:0,&p).status==NVM_ARRAY_LIMIT&&p==prior);
        CHECK(ra_calls==calls&&ra_live==live&&ra_bytes==bytes);CHECK(nvm_union_field_image_cost(prior,&exact));
    }
    nvm_union_field_image_free(prior);nvm_union_field_execution_free(source);CHECK(!ra_live&&!ra_bytes);
    char *large=malloc(1024u*1024u+1);CHECK(large);memset(large,'q',1024u*1024u);large[1024u*1024u]=0;
    char *strings[7];uint32_t lengths[7];for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
    strings[6]=large;lengths[6]=1024u*1024u;v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=7;
    source=execution_query(&v.c.module);NvmUnionFieldImage *large_image=NULL;
    CHECK(nvm_union_field_image_copy(source,&large_image).status==NVM_ARRAY_ELIGIBLE);
    nvm_union_field_image_free(large_image);nvm_union_field_execution_free(source);
    VmUnionFieldPrivate *p=(void *)(uintptr_t)1;NvmArrayEligibilityResult r=vm_union_field_private_prepare(&v.c.module,&p);
    CHECK(r.status==NVM_ARRAY_LIMIT&&p==(void *)(uintptr_t)1&&!ra_live&&!ra_bytes);free(large);
    execution_case(&v,code);p=preparation_query(&v.c.module);vm_union_field_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
#endif
int main(int argc,char **argv) {
    preparation_basic();
#ifdef VM_UF_WHITEBOX
    UnionCase v;uint8_t code[128];execution_case(&v,code);preparation_oracle(&v.c.module);
    preparation_prefixes();preparation_constants();preparation_growth();preparation_variants();preparation_corruption();preparation_reservations();
#endif
    for(int i=1;i<argc;i++)preparation_original(argv[i]);
    printf("I passed %u union preparation checks; no execution admission.\n",checks);return 0;
}
