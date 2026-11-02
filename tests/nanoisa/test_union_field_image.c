#include "../../src/nanoisa/union_field_snapshot_private.h"
static void image_original(const NvmUnionFieldExecutionPlan *);
#define UFE_ORIGINAL_CHECK image_original
#define UFE_MAIN union_recipe_controls_main
#include "test_union_field_execution.c"
static NvmUnionFieldImage *image_query(const NvmUnionFieldExecutionPlan *p) {
    NvmUnionFieldImage *image=NULL;NvmArrayEligibilityResult r=nvm_union_field_image_copy(p,&image);
    if(r.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"union image status=%u: %s\n",r.status,r.message);
    CHECK(r.status==NVM_ARRAY_ELIGIBLE&&image);return image;
}
static void image_rows(const NvmUnionFieldExecutionPlan *p,const NvmUnionFieldImage *image,bool public_accepted) {
    NvmRecordArrayExecutionCounts a,b;
    CHECK(nvm_union_field_execution_counts(p,&a)&&nvm_union_field_image_counts(image,&b));
    CHECK(a.functions==b.functions&&a.instructions==b.instructions&&a.initializer==b.initializer&&a.entry==b.entry);
    for(uint32_t i=0;i<a.instructions;i++) {
        NvmUnionFieldExecutionInstruction x,y;
        CHECK(nvm_union_field_execution_instruction(p,i,&x)&&nvm_union_field_image_instruction(image,i,&y));
        CHECK(x.base.opcode==y.base.opcode&&x.base.recipe==y.base.recipe&&x.base.pc==y.base.pc);
        CHECK(x.site.state==y.site.state&&x.site.operation==y.site.operation&&x.site.variant==y.site.variant);
        CHECK(x.site.receiver.origins==y.site.receiver.origins&&x.site.result.origins==y.site.result.origins);
    }
    for(unsigned k=0;k<3;k++) {
        uint32_t n,m;CHECK(nvm_union_field_execution_size(p,(NvmRecordArraySnapshotKind)k,0,&n));
        CHECK(nvm_union_field_image_size(image,(NvmRecordArraySnapshotKind)k,0,&m)&&m==n);
        uint8_t *x=malloc(n?n:1),*y=malloc(n?n:1);CHECK(x&&y);
        CHECK(nvm_union_field_execution_bytes(p,(NvmRecordArraySnapshotKind)k,0,0,n,x));
        CHECK(nvm_union_field_image_bytes(image,(NvmRecordArraySnapshotKind)k,0,0,n,y)&&!memcmp(x,y,n));free(x);free(y);
    }
    NvmUnionFieldImageCost cost;CHECK(nvm_union_field_image_cost(image,&cost));
    CHECK(cost.source_retained_bytes&&cost.preparation_bytes_reserved>=a.peak_bytes_reserved+cost.source_retained_bytes);
#ifdef UFP_ALLOC
    CHECK(ufi_same(p,image->plan));
    CHECK(nvm_verify(&p->carrier->snapshot).ok==public_accepted);
    CHECK(nvm_verify(&image->plan->carrier->snapshot).ok==public_accepted);
    char error[512]={0};char *emitted=nvm2c_emit(&image->plan->carrier->snapshot,error,sizeof error);
    CHECK(!!emitted==public_accepted);if(!public_accepted)CHECK(error[0]);free(emitted);
    /* Conversion allocation ownership belongs to the linked original boundary
     * controls; do not send unwrapped converter allocations to wrapped frees. */
#else
    (void)public_accepted;
#endif
}
static void image_original(const NvmUnionFieldExecutionPlan *p) {
    NvmUnionFieldImage *image=image_query(p);image_rows(p,image,false);nvm_union_field_image_free(image);
}
static void image_basic(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);
    NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);NvmUnionFieldImage *image=image_query(p);
    image_rows(p,image,false);NvmUnionFieldOrigin origin;CHECK(nvm_union_field_image_origin(image,0,&origin));
    nvm_union_field_execution_free(p);memset(&v,0,sizeof v);memset(code,0,sizeof code);
    NvmUnionFieldOrigin after;CHECK(nvm_union_field_image_origin(image,0,&after)&&after.member.tags==origin.member.tags);
    uint32_t layout;CHECK(nvm_union_field_image_nominal(image,TAG_STRUCT,0,&layout)&&layout==1);
    CHECK(nvm_union_field_image_nominal(image,TAG_UNION,0,&layout)&&layout==0);
    NvmUnionFieldExecutionInstruction unchanged;memset(&unchanged,0xa5,sizeof unchanged);NvmUnionFieldExecutionInstruction saved=unchanged;
    CHECK(!nvm_union_field_image_instruction(image,UINT32_MAX,&unchanged)&&!memcmp(&unchanged,&saved,sizeof saved));
    nvm_union_field_image_free(image);nvm_union_field_image_free(NULL);
}
static void image_variants(void) {
    for(unsigned mode=0;mode<4;mode++) {
        UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
        if(mode<2){v.c.ownership[v.variant+6]=0;v.c.ownership[v.variant+12]=0;v.c.ownership[v.variant+14]=1;}
        if(mode==0){proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,1,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);}
        else if(mode==1){proof_pack(code,&n,AGG_VARIANT,0,0);u8(code,&n,OP_AGG_TAG);}
        else {proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);u8(code,&n,OP_AGG_GET);u16(code,&n,0);}
        u8(code,&n,mode==3?OP_HALT:OP_RET);
        if(mode>=2){u8(code,&n,OP_HALT);proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_POP);}
        if(mode==3) {
            proof_header(&v,code,n);NvmUnionFieldExecutionPlan *refused=(void *)(uintptr_t)1;
            CHECK(nvm_prepare_union_field_execution(&v.c.module,&refused).status==NVM_ARRAY_UNRESOLVED&&refused==(void *)(uintptr_t)1);
            n=0;u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);size_t branch=n;
            u8(code,&n,OP_JMP_FALSE);u32(code,&n,0);proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
            u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
            patch(code,branch+1,(uint32_t)(n-branch));u8(code,&n,OP_HALT);
        }
        proof_header(&v,code,n);NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);
        NvmUnionFieldImage *image=image_query(p);image_rows(p,image,true);
        nvm_union_field_execution_free(p);nvm_union_field_image_free(image);
    }
}
#ifdef UFP_ALLOC
static void image_corruption(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);CHECK(!ra_live&&!ra_bytes);
    NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);NvmUnionFieldImage *prior=image_query(p),*out=prior;
    size_t bytes=ra_bytes,live=ra_live;
#define UFI_BAD(field,value) do {uint64_t saved=(field);(field)=(value);out=prior;CHECK(nvm_union_field_image_copy(p,&out).status==NVM_ARRAY_INVALID&&out==prior&&ra_bytes==bytes&&ra_live==live);(field)=saved;}while(0)
    UFI_BAD(p->carrier->counts.initializer,0);UFI_BAD(p->carrier->counts.frame_limit,1);
    UFI_BAD(p->carrier->functions[0].maximum_stack,p->carrier->functions[0].maximum_stack+1);
    UFI_BAD(p->carrier->maximum_stacks[0],p->carrier->maximum_stacks[0]+1);
    UFI_BAD(p->carrier->functions[0].parameter_tags_present,1);
    UFI_BAD(p->carrier->instructions[0].successors[1],123);
    UFI_BAD(p->carrier->instructions[0].operand_bits[MAX_OPERANDS-1],123);
    UFI_BAD(p->carrier->instructions[0].operand_types[MAX_OPERANDS-1],255);
    UFI_BAD(p->carrier->descriptors[0].layout,0);
    UFI_BAD(p->sites[1].result.origins,UINT64_C(1)<<63);UFI_BAD(p->sites[1].variant,1);
    UFI_BAD(p->sites[1].reserved[0],1);UFI_BAD(p->proof->origins[0].member.tags,BIT(TAG_BOOL));
    UFI_BAD(p->proof->results[0].tags,BIT(TAG_BOOL));UFI_BAD(p->proof->stacks[0],p->proof->stacks[0]+1);
#undef UFI_BAD
    image_rows(p,prior,false);out=image_query(p);nvm_union_field_image_free(out);
    nvm_union_field_image_free(prior);nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);
}
static void image_snapshot_limits(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);
    NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);NvmUnionFieldImage *prior=image_query(p),*out=prior;
    size_t bytes=ra_bytes,live=ra_live;
    /* Well-framed wrong historical scalar write must fail independent proof. */
    uint8_t saved=p->carrier->snapshot.code[0];p->carrier->snapshot.code[0]=OP_PUSH_F64;
    CHECK(nvm_union_field_image_copy(p,&out).status!=NVM_ARRAY_ELIGIBLE&&out==prior&&ra_bytes==bytes&&ra_live==live);
    p->carrier->snapshot.code[0]=saved;
    saved=p->carrier->snapshot.layout_data[13];p->carrier->snapshot.layout_data[13]=1;
    CHECK(nvm_union_field_image_copy(p,&out).status!=NVM_ARRAY_ELIGIBLE&&out==prior&&ra_bytes==bytes&&ra_live==live);
    p->carrier->snapshot.layout_data[13]=saved;
    image_rows(p,prior,false);
    /* Real image LIMIT from an independently accepted large source snapshot.
     * These bytes are temporary test input, never a serialized artifact. */
    char *strings[7];uint32_t lengths[7];
    for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
    strings[6]="";lengths[6]=0;v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=7;
    NvmUnionFieldExecutionPlan *empty=execution_query(&v.c.module);NvmRecordArrayExecutionCounts counts;
    CHECK(nvm_union_field_execution_counts(empty,&counts));nvm_union_field_execution_free(empty);
    uint32_t length=(uint32_t)((NVM_RECORD_ARRAY_EXECUTION_STEPS-counts.work_reserved)/2-128);
    char *large=malloc((size_t)length+1);CHECK(large);memset(large,'x',length);large[length]=0;
    strings[6]=large;lengths[6]=length;NvmUnionFieldExecutionPlan *wide=execution_query(&v.c.module);
    size_t before=ra_bytes;out=prior;
    CHECK(nvm_union_field_image_copy(wide,&out).status==NVM_ARRAY_LIMIT&&out==prior&&ra_bytes==before);
    execution_rows(wide);nvm_union_field_execution_free(wide);free(large);
    out=image_query(p);nvm_union_field_image_free(out);
    /* Exact-cap tests use the actual new positive constructor reservation,
     * not the image public API, and preserve a real existing plan output. */
    NvmUnionFieldExecutionPlan *trial=p;
    uint64_t extra_bytes=NVM_RECORD_ARRAY_EXECUTION_BYTES-p->carrier->counts.peak_bytes_reserved;
    uint64_t extra_work=NVM_RECORD_ARRAY_EXECUTION_STEPS-p->carrier->counts.work_reserved;
    CHECK(ufe_prepare_reserved(&p->carrier->snapshot,&trial,extra_bytes,extra_work).status==NVM_ARRAY_ELIGIBLE);
    CHECK(trial->carrier->counts.peak_bytes_reserved==NVM_RECORD_ARRAY_EXECUTION_BYTES&&
          trial->carrier->counts.work_reserved==NVM_RECORD_ARRAY_EXECUTION_STEPS);
    nvm_union_field_execution_free(trial);trial=p;
    CHECK(ufe_prepare_reserved(&p->carrier->snapshot,&trial,extra_bytes+1,extra_work).status==NVM_ARRAY_LIMIT&&trial==p);
    CHECK(ufe_prepare_reserved(&p->carrier->snapshot,&trial,extra_bytes,extra_work+1).status==NVM_ARRAY_LIMIT&&trial==p);
    size_t calls=ra_calls;
    CHECK(ufe_prepare_reserved(&p->carrier->snapshot,&trial,UINT64_MAX,0).status==NVM_ARRAY_LIMIT&&trial==p&&ra_calls==calls);
    CHECK(ufe_prepare_reserved(&p->carrier->snapshot,&trial,0,UINT64_MAX).status==NVM_ARRAY_LIMIT&&trial==p&&ra_calls==calls);
    CHECK(ra_bytes==bytes&&ra_live==live);image_rows(p,prior,false);
    nvm_union_field_image_free(prior);nvm_union_field_execution_free(p);CHECK(!ra_bytes&&!ra_live);
}
static void image_optional_storage(void) {
    for(unsigned mode=0;mode<4;mode++) {
        UnionCase v;uint8_t code[128];execution_case(&v,code);
        uint8_t zero=0,*rows[1]={mode?&zero:NULL};v.c.module.function_param_types=rows;
        NvmMetadataEntry metadata={0,0};NvmDebugEntry debug={0};
        if(mode>=2){v.c.module.metadata=&metadata;v.c.module.metadata_count=1;}
        if(mode>=3){v.c.module.debug_entries=&debug;v.c.module.debug_count=1;}
        NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);size_t source=ra_bytes;
        NvmUnionFieldImage *image=image_query(p);NvmMapCopyWork work={0,UINT64_MAX,false};uint64_t retained;
        CHECK(ufe_plan_retained_cost(image->plan,&work,&retained)==NVM_MAP_COST_OK);
        CHECK(ra_bytes==source+sizeof *image+retained&&retained==source);image_rows(p,image,false);
        nvm_union_field_execution_free(p);nvm_union_field_image_free(image);CHECK(!ra_live&&!ra_bytes);
    }
}
static void image_globals(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
    u8(code,&n,OP_STORE_GLOBAL);u32(code,&n,0);u8(code,&n,OP_LOAD_GLOBAL);u32(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(&v,code,n);NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);
    NvmUnionFieldImage *prior=image_query(p),*out=prior;image_rows(p,prior,false);
    NvmRecordValueOrigins global;CHECK(nvm_union_field_image_global(prior,0,&global));
    CHECK(global.tags==(BIT(TAG_VOID)|BIT(TAG_STRUCT))&&global.origins==2&&!global.unknown);
    size_t bytes=ra_bytes,live=ra_live;NvmRecordValueOrigins saved=p->proof->globals[0];
    p->proof->globals[0].origins=UINT64_C(1)<<63;
    CHECK(nvm_union_field_image_copy(p,&out).status==NVM_ARRAY_INVALID&&out==prior&&ra_bytes==bytes&&ra_live==live);
    p->proof->globals[0]=saved;out=image_query(p);nvm_union_field_image_free(out);
    nvm_union_field_execution_free(p);memset(&v,0,sizeof v);memset(code,0,sizeof code);
    CHECK(nvm_union_field_image_global(prior,0,&global)&&global.tags==saved.tags&&global.origins==saved.origins);
    nvm_union_field_image_free(prior);CHECK(!ra_live&&!ra_bytes);
}
static void image_allocations(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);CHECK(!ra_live&&!ra_bytes);
    NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);size_t source_bytes=ra_bytes;
    ra_peak=ra_bytes;ra_calls=0;NvmUnionFieldImage *prior=image_query(p);
    size_t positions=ra_calls,peak=ra_peak,total=ra_bytes;
    NvmMapCopyWork work={0,UINT64_MAX,false};uint64_t destination=0;
    CHECK(ufe_plan_retained_cost(prior->plan,&work,&destination)==NVM_MAP_COST_OK);
    CHECK(total==source_bytes+sizeof *prior+destination&&prior->cost.source_retained_bytes==source_bytes);
    CHECK(peak<=prior->cost.preparation_bytes_reserved&&sizeof *prior==32);
    image_rows(p,prior,false);size_t live=ra_live;
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<positions;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)mode;NvmUnionFieldImage *out=prior;
        NvmArrayEligibilityResult r=nvm_union_field_image_copy(p,&out);
        CHECK(r.status==NVM_ARRAY_MEMORY&&out==prior&&ra_live==live&&ra_bytes==total);
        ra_fail=SIZE_MAX;ra_persistent=0;image_rows(p,prior,false);
        out=image_query(p);nvm_union_field_image_free(out);CHECK(ra_live==live&&ra_bytes==total);
    }
    printf("union image owner=%zu source=%zu destination=%llu live=%zu peak=%zu reserved=%llu work=%llu prefixes=%zu\n",
        sizeof *prior,source_bytes,(unsigned long long)destination,total,peak,
        (unsigned long long)prior->cost.preparation_bytes_reserved,(unsigned long long)prior->cost.preparation_work_reserved,positions);
    nvm_union_field_execution_free(p);NvmUnionVariantFact fact;CHECK(nvm_union_field_image_variant(prior,0,0,&fact)&&fact.field_count==1);
    nvm_union_field_image_free(prior);CHECK(!ra_live&&!ra_bytes);
}
#endif
int main(int argc,char **argv) {
    CHECK(argc==1||argc==7);image_basic();image_variants();
#ifdef UFP_ALLOC
    (void)argv;image_corruption();image_snapshot_limits();image_optional_storage();image_globals();image_allocations();
#else
    for(int i=1;i<argc;i++)execution_original(argv[i]);
#endif
    printf("PASS %u union image checks; no execution admission\n",checks);return 0;
}
