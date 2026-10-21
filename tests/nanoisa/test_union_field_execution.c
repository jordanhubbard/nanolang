/* I test copied recipes, never selected execution. */
#define UFP_MAIN union_proof_controls_main
#include "test_union_field_proof.c"
#undef UFP_MAIN
#include "../../src/nanoisa/managed_union_field_execution_private.h"
static size_t execution_case(UnionCase *v,uint8_t *code) {
    union_case(v);size_t n=0;
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    proof_pack(code,&n,AGG_RECORD,0,1);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(v,code,n);return n;
}
static NvmUnionFieldExecutionPlan *execution_query(const NvmModule *m) {
    NvmUnionFieldExecutionPlan *p=NULL;NvmArrayEligibilityResult r=nvm_prepare_union_field_execution(m,&p);
    if(r.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"union recipes status=%u fn=%u pc=%u: %s\n",r.status,r.function,r.pc,r.message);
    CHECK(r.status==NVM_ARRAY_ELIGIBLE&&p);return p;
}
static void execution_rows(NvmUnionFieldExecutionPlan *p) {
    NvmRecordArrayExecutionCounts c;CHECK(nvm_union_field_execution_counts(p,&c));
    CHECK(c.functions&&c.instructions&&c.records==1);
    for(uint32_t i=0;i<c.instructions;i++) {
        NvmUnionFieldExecutionInstruction r;CHECK(nvm_union_field_execution_instruction(p,i,&r));
        CHECK(r.base.function==r.site.function&&r.base.pc==r.site.pc);
        if(r.base.opcode==OP_HALT)CHECK(!r.base.successor_count&&r.base.recipe==NVM_UF_INVALID_BASE_RECIPE);
        if(r.site.state==UFS_PROVED) {
            if(r.site.operation==UFS_RECORD_PACK || r.site.operation==UFS_RECORD_GET)CHECK(r.site.nominal==0&&r.site.layout==1);
            if(r.site.operation==UFS_VARIANT_PACK || r.site.operation==UFS_PAYLOAD_GET || r.site.operation==UFS_TAG)CHECK(r.site.nominal==0&&r.site.layout==0);
        }
    }
    NvmUnionFieldExecutionInstruction unchanged;memset(&unchanged,0xa5,sizeof unchanged);
    NvmUnionFieldExecutionInstruction saved=unchanged;
    CHECK(!nvm_union_field_execution_instruction(p,c.instructions,&unchanged)&&!memcmp(&saved,&unchanged,sizeof saved));
}
static void execution_basic(void) {
    UnionCase v;uint8_t code[128];size_t n=execution_case(&v,code);
    NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);execution_rows(p);proof_public_refusal(&v.c.module);
    uint8_t copy[128]={0};CHECK(nvm_union_field_execution_bytes(p,NVM_RA_SNAPSHOT_CODE,0,0,(uint32_t)n,copy)&&!memcmp(copy,code,n));
    NvmUnionFieldExecutionInstruction r;
    CHECK(nvm_union_field_execution_instruction(p,1,&r)&&r.site.operation==UFS_VARIANT_PACK&&r.site.result.tags==(1u<<TAG_UNION));
    CHECK(nvm_union_field_execution_instruction(p,3,&r)&&r.site.operation==UFS_RECORD_GET&&r.site.result.tags==(1u<<TAG_UNION));
    CHECK(nvm_union_field_execution_instruction(p,4,&r)&&r.site.operation==UFS_PAYLOAD_GET&&r.site.result.tags==(1u<<TAG_INT));
    memset(code,0,sizeof code);memset(&v,0,sizeof v);execution_rows(p);
    memset(copy,0,sizeof copy);CHECK(nvm_union_field_execution_bytes(p,NVM_RA_SNAPSHOT_CODE,0,0,(uint32_t)n,copy)&&copy[0]==OP_PUSH_I64);
    nvm_union_field_execution_free(p);
}
static void execution_empty_halt(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_AGG_TAG);u8(code,&n,OP_RET);
    proof_header(&v,code,n);NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);execution_rows(p);
    NvmUnionFieldExecutionInstruction r;CHECK(nvm_union_field_execution_instruction(p,0,&r)&&r.site.variant==1&&r.site.member==0);
    nvm_union_field_execution_free(p);
    n=0;proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    u8(code,&n,OP_HALT);proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_POP);
    proof_header(&v,code,n);p=execution_query(&v.c.module);execution_rows(p);
    CHECK(nvm_union_field_execution_instruction(p,4,&r)&&r.base.opcode==OP_HALT&&!r.base.successor_count&&r.site.state==UFS_NOT_ESTABLISHED);
    CHECK(nvm_union_field_execution_instruction(p,5,&r)&&r.site.state==UFS_NOT_ESTABLISHED&&r.site.operation==UFS_VARIANT_PACK&&!r.site.result.tags);
    nvm_union_field_execution_free(p);
}
#ifdef UFP_ALLOC
static void execution_all_opcodes(void) {
    unsigned count=0;
    for(unsigned op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction r={0};
        int expected=nvm_record_array_opcode_supported((uint8_t)op)||op==OP_AGG_TAG||op==OP_HALT;
        CHECK(!!ufe_recipe((uint8_t)op,&r)==!!expected);count+=(unsigned)expected;
        if(expected&&uf_sensitive((uint8_t)op))CHECK(r.recipe==NVM_UF_INVALID_BASE_RECIPE);
    }
    CHECK(count==96);
}
static void execution_corruption(void) {
    UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
    proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_POP);
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
    proof_header(&v,code,n);NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);
    UfeOrigins o={0};o.count=p->proof->counts.origins;
    for(uint32_t i=0;i<o.count;i++) {
        o.rows[i]=p->proof->origins[i];
        if(o.rows[i].kind==NVM_HEAP_ORIGIN_RECORD)o.records|=UINT64_C(1)<<i;
        else o.variants[o.rows[i].variant]|=UINT64_C(1)<<i;
    }
    NvmUnionVariantFact variants[2];CHECK(nvm_union_field_proof_variant(p->proof,0,0,&variants[0]));
    CHECK(nvm_union_field_proof_variant(p->proof,0,1,&variants[1]));
    for(uint32_t i=0;i<p->site_count;i++) {
        NvmRecordArrayExecutionInstruction base=p->carrier->instructions[i];NvmUnionFieldSite site=p->sites[i];
        CHECK(ufe_corresponds(&base,&site,1,0,variants,&o));
        NvmUnionFieldSite bad=site;
        if(site.result.origins) {
            bad.result.origins=UINT64_C(1)<<63;CHECK(!ufe_corresponds(&base,&bad,1,0,variants,&o));
            bad=site;bad.result.origins=o.records;CHECK(!ufe_corresponds(&base,&bad,1,0,variants,&o) || site.result.tags==(1u<<TAG_STRUCT));
        }
        if(site.operation==UFS_PAYLOAD_GET) {
            bad=site;bad.receiver.origins=o.variants[1];CHECK(!ufe_corresponds(&base,&bad,1,0,variants,&o));
            bad=site;bad.variant=1;CHECK(!ufe_corresponds(&base,&bad,1,0,variants,&o));
        }
        if(site.state==UFS_PROVED) {
            base.pops++;CHECK(!ufe_corresponds(&base,&site,1,0,variants,&o));base=p->carrier->instructions[i];
            base.pushes++;CHECK(!ufe_corresponds(&base,&site,1,0,variants,&o));base=p->carrier->instructions[i];
        }
        base.obligations^=NVM_RA_RETAIN_BEFORE_RELEASE;CHECK(!ufe_corresponds(&base,&site,1,0,variants,&o));
    }
    nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);
}
static void execution_setter_effects(void) {
    UnionCase v;union_case(&v);uint8_t code[256];size_t n=0;
    u8(code,&n,OP_PUSH_BOOL);u8(code,&n,1);size_t branch=n;u8(code,&n,OP_JMP_FALSE);u32(code,&n,0);
    proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
    proof_pack(code,&n,AGG_VARIANT,1,0);u8(code,&n,OP_AGG_SET);u16(code,&n,0);
    u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_TAG);u8(code,&n,OP_POP);
    proof_i64(code,&n,0);u8(code,&n,OP_RET);patch(code,branch+1,(uint32_t)(n-branch));u8(code,&n,OP_HALT);
    proof_header(&v,code,n);NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);
    UfeOrigins o={0};o.count=p->proof->counts.origins;
    for(uint32_t i=0;i<o.count;i++) {
        o.rows[i]=p->proof->origins[i];
        if(o.rows[i].kind==NVM_HEAP_ORIGIN_RECORD)o.records|=UINT64_C(1)<<i;
        else o.variants[o.rows[i].variant]|=UINT64_C(1)<<i;
    }
    NvmUnionVariantFact variants[2];CHECK(nvm_union_field_proof_variant(p->proof,0,0,&variants[0]));
    CHECK(nvm_union_field_proof_variant(p->proof,0,1,&variants[1]));
    unsigned seen=0;
    for(uint32_t i=0;i<p->site_count;i++) {
        NvmRecordArrayExecutionInstruction r=p->carrier->instructions[i];NvmUnionFieldSite site=p->sites[i];
        CHECK(ufe_corresponds(&r,&site,1,0,variants,&o));
        if(site.state!=UFS_PROVED)continue;
        seen|=1u<<site.operation;
        r.pops++;CHECK(!ufe_corresponds(&r,&site,1,0,variants,&o));r=p->carrier->instructions[i];
        r.pushes++;CHECK(!ufe_corresponds(&r,&site,1,0,variants,&o));
    }
    CHECK((seen&((1u<<UFS_RECORD_SET)|(1u<<UFS_TAG)|(1u<<UFS_HALT)))==((1u<<UFS_RECORD_SET)|(1u<<UFS_TAG)|(1u<<UFS_HALT)));
    /* Budget helper evidence, not a constructor reservation seam. */
    NvmRecordArrayExecutionPlan *c=p->carrier;uint64_t bytes=c->bytes,work=c->work;
    CHECK(rae_charge(c,NVM_RECORD_ARRAY_EXECUTION_BYTES-bytes,NVM_RECORD_ARRAY_EXECUTION_STEPS-work));
    CHECK(!rae_charge(c,1,0)&&c->bytes==NVM_RECORD_ARRAY_EXECUTION_BYTES&&c->work==NVM_RECORD_ARRAY_EXECUTION_STEPS);
    CHECK(!rae_charge(c,0,1)&&c->bytes==NVM_RECORD_ARRAY_EXECUTION_BYTES&&c->work==NVM_RECORD_ARRAY_EXECUTION_STEPS);
    c->bytes=bytes;c->work=work;c->failure=NVM_ARRAY_INVALID;
    nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);
}
static void execution_storage_edges(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);CHECK(!ra_live&&!ra_bytes);
    NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);size_t base=ra_bytes;
    nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);
    char *strings[7];uint32_t lengths[7];
    for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
    strings[6]="";lengths[6]=0;uint8_t zero=0;uint8_t *parameters[1]={&zero};
    v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=7;
    v.c.module.function_param_types=parameters;
    p=execution_query(&v.c.module);
    CHECK(ra_bytes==base+sizeof(char *)+sizeof(uint32_t)+1+sizeof(uint8_t *)+1);
    CHECK(p->carrier->snapshot.function_param_types&&p->carrier->snapshot.function_param_types[0]&&
          !p->carrier->snapshot.function_param_types[0][0]&&p->carrier->snapshot.strings[6]&&!p->carrier->snapshot.strings[6][0]);
    uint32_t size=99;CHECK(nvm_union_field_execution_size(p,NVM_RA_SNAPSHOT_STRING,6,&size)&&!size);
    uint8_t tag=255;CHECK(!nvm_union_field_execution_parameter(p,0,0,&tag)&&tag==255);
    nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);
}
static void execution_allocations(void) {
    UnionCase v;uint8_t code[128];size_t n=execution_case(&v,code);UnionCase saved=v;uint8_t code_copy[128];memcpy(code_copy,code,n);
    CHECK(!ra_live&&!ra_bytes);ra_fail=SIZE_MAX;ra_persistent=0;ra_calls=0;ra_peak=0;
    NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);size_t positions=ra_calls,retained=ra_bytes,peak=ra_peak;
    NvmRecordArrayExecutionCounts c;CHECK(nvm_union_field_execution_counts(p,&c));
    CHECK(sizeof(NvmUnionFieldSite)==72&&sizeof(NvmUnionFieldSiteExport)==16&&sizeof(NvmUnionFieldExecutionPlan)==32);
    CHECK(!p->carrier->origins&&p->site_count==c.instructions&&peak<=c.peak_bytes_reserved);
    size_t before_declarations=ra_bytes;NvmUnionFieldDeclarations *decl=NULL;
    CHECK(nvm_union_field_declarations_prepare(&v.c.module,&decl).status==NVM_UFD_PREPARED);
    size_t declaration_bytes=ra_bytes-before_declarations;nvm_union_field_declarations_free(decl);
    CHECK(ra_bytes==retained);
    const NvmModule *m=&p->carrier->snapshot;
    size_t snapshot=(size_t)m->function_count*sizeof *m->functions+m->code_size+m->layout_size+m->ownership_size+
        (size_t)m->metadata_count*sizeof *m->metadata+(size_t)m->debug_count*sizeof *m->debug_entries+
        (size_t)m->string_count*(sizeof *m->strings+sizeof *m->string_lengths);
    for(uint32_t i=0;i<m->string_count;i++)snapshot+=(size_t)m->string_lengths[i]+1;
    if(m->function_param_types) {
        snapshot+=(size_t)m->function_count*sizeof *m->function_param_types;
        for(uint32_t i=0;i<m->function_count;i++)if(m->function_param_types[i])snapshot+=m->functions[i].arity?m->functions[i].arity:1;
    }
    NvmUnionFieldProofCounts pc;CHECK(nvm_union_field_execution_proof_counts(p,&pc));
    size_t proof_bytes=sizeof *p->proof+declaration_bytes+(size_t)pc.origins*sizeof *p->proof->origins+
        (size_t)(pc.globals+pc.functions)*sizeof *p->proof->globals+(size_t)pc.functions*sizeof *p->proof->stacks;
    size_t expected=sizeof *p+sizeof *p->carrier+snapshot+proof_bytes+
        (size_t)c.instructions*(sizeof *p->sites+sizeof *p->carrier->instructions);
    CHECK(expected==retained);
    nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);
    NvmUnionFieldExecutionPlan *prior=execution_query(&v.c.module);size_t prior_live=ra_live,prior_bytes=ra_bytes;
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<positions;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)mode;p=prior;
        NvmArrayEligibilityResult r=nvm_prepare_union_field_execution(&v.c.module,&p);
        if(r.status!=NVM_ARRAY_MEMORY)fprintf(stderr,"recipe prefix=%zu mode=%u status=%u %s\n",i,mode,r.status,r.message);
        CHECK(r.status==NVM_ARRAY_MEMORY&&p==prior&&ra_live==prior_live&&ra_bytes==prior_bytes);
        CHECK(!memcmp(&saved,&v,sizeof v)&&!memcmp(code_copy,code,n));
        ra_fail=SIZE_MAX;ra_persistent=0;execution_rows(prior);
        p=execution_query(&v.c.module);nvm_union_field_execution_free(p);CHECK(ra_live==prior_live&&ra_bytes==prior_bytes);
    }
    nvm_union_field_execution_free(prior);CHECK(!ra_live&&!ra_bytes);
    NvmUnionFieldProof *proof=NULL;NvmUnionFieldSiteExport sites={0};
    CHECK(analyze_union_sites(&v.c.module,&proof,&sites).status==NVM_ARRAY_ELIGIBLE);
    size_t live=ra_live,bytes=ra_bytes;NvmUnionFieldProof *proof_saved=proof;NvmUnionFieldSiteExport sites_saved=sites;
    /* Only the writable output envelopes overlap, not arbitrary heap inputs. */
    CHECK(analyze_union_sites(&v.c.module,(NvmUnionFieldProof **)&sites,&sites).status==NVM_ARRAY_INVALID);
    CHECK(!memcmp(&sites,&sites_saved,sizeof sites)&&proof==proof_saved&&ra_live==live&&ra_bytes==bytes);
    ra_calls=0;ra_fail=0;CHECK(analyze_union_sites(&v.c.module,&proof,&sites).status==NVM_ARRAY_MEMORY);
    CHECK(proof==proof_saved&&!memcmp(&sites,&sites_saved,sizeof sites)&&ra_live==live&&ra_bytes==bytes);
    ra_fail=SIZE_MAX;ra_calls=0;
    NvmUnionFieldProof *trial=NULL;NvmUnionFieldSiteExport trial_sites={0};
    CHECK(analyze_union_sites(&v.c.module,&trial,&trial_sites).status==NVM_ARRAY_ELIGIBLE);
    size_t paired_positions=ra_calls;nvm_union_field_proof_free(trial);ra_test_free(trial_sites.rows);
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<paired_positions;i++) {
        ra_calls=0;ra_fail=i;ra_persistent=(int)mode;
        CHECK(analyze_union_sites(&v.c.module,&proof,&sites).status==NVM_ARRAY_MEMORY);
        CHECK(proof==proof_saved&&!memcmp(&sites,&sites_saved,sizeof sites)&&ra_live==live&&ra_bytes==bytes);
        CHECK(!memcmp(&saved,&v,sizeof v)&&!memcmp(code_copy,code,n));
        ra_fail=SIZE_MAX;ra_persistent=0;trial=NULL;memset(&trial_sites,0,sizeof trial_sites);
        CHECK(analyze_union_sites(&v.c.module,&trial,&trial_sites).status==NVM_ARRAY_ELIGIBLE);
        nvm_union_field_proof_free(trial);ra_test_free(trial_sites.rows);CHECK(ra_live==live&&ra_bytes==bytes);
    }
    nvm_union_field_proof_free(proof);ra_test_free(sites.rows);CHECK(!ra_live&&!ra_bytes);
    printf("union atomic proof/sites prefixes=%zu both denial modes\n",paired_positions);
    printf("union recipes owner=%zu site=%zu carrier=%zu origin_scratch=%zu retained=%zu peak=%zu bytes=%llu work=%llu prefixes=%zu\n",
        sizeof(NvmUnionFieldExecutionPlan),sizeof(NvmUnionFieldSite),sizeof(NvmRecordArrayExecutionPlan),sizeof(UfeOrigins),retained,peak,
        (unsigned long long)c.peak_bytes_reserved,(unsigned long long)c.work_reserved,positions);
}
static void execution_cost_oracles(void) {
    for(unsigned variant=0;variant<6;variant++) {
        UnionCase v;uint8_t code[128];execution_case(&v,code);
        char *strings[7];uint32_t lengths[7];
        for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
        strings[6]="";lengths[6]=0;uint8_t zero=0;uint8_t *parameters[1]={NULL};
        NvmMetadataEntry metadata={0,0};NvmDebugEntry debug={0};
        if(variant>=1)v.c.module.function_param_types=parameters;
        if(variant>=2)parameters[0]=&zero;
        if(variant>=3){v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=7;}
        if(variant>=4){v.c.module.metadata=&metadata;v.c.module.metadata_count=1;}
        if(variant>=5){v.c.module.debug_entries=&debug;v.c.module.debug_count=1;}
        CHECK(!ra_live&&!ra_bytes);ra_peak=0;
        NvmUnionFieldExecutionPlan *p=execution_query(&v.c.module);
        size_t retained=ra_bytes,peak=ra_peak,calls=ra_calls,live=ra_live;
        NvmMapCopyWork work={0,UINT64_MAX,false};uint64_t bytes=99;
        CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_OK&&bytes==retained);
        uint64_t needed=184+20*(uint64_t)v.c.module.function_count+12*(uint64_t)v.c.module.string_count;
        CHECK(work.used==needed&&ra_calls==calls&&ra_bytes==retained&&ra_live==live&&ra_peak==peak);
        work=(NvmMapCopyWork){0,needed,false};bytes=99;
        CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_OK&&bytes==retained&&work.used==needed);
        work=(NvmMapCopyWork){0,needed-1,false};bytes=99;
        CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_LIMIT&&bytes==99&&work.limited);
        work=(NvmMapCopyWork){0,UINT64_MAX,true};bytes=99;
        CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_LIMIT&&bytes==99);
        work=(NvmMapCopyWork){0,UINT64_MAX,false};NvmMapCopyWork saved_work=work;
        CHECK(ufe_plan_retained_cost(p,&work,&work.used)==NVM_MAP_COST_INVALID&&!memcmp(&work,&saved_work,sizeof work));
        /* Full independent constructors make the sub-owner oracles measurable. */
        NvmUnionFieldProof *proof=NULL;
        CHECK(nvm_union_field_proof_prepare(&v.c.module,&proof).status==NVM_ARRAY_ELIGIBLE);
        size_t proof_bytes=ra_bytes-retained;calls=ra_calls;
        work=(NvmMapCopyWork){0,100,false};bytes=99;
        CHECK(ufe_proof_retained_cost(proof,&work,&bytes)==NVM_MAP_COST_OK&&bytes==proof_bytes&&work.used==100&&ra_calls==calls);
        work=(NvmMapCopyWork){0,99,false};bytes=99;
        CHECK(ufe_proof_retained_cost(proof,&work,&bytes)==NVM_MAP_COST_LIMIT&&bytes==99);
        nvm_union_field_proof_free(proof);CHECK(ra_bytes==retained);
        NvmUnionFieldDeclarations *decl=NULL;
        CHECK(nvm_union_field_declarations_prepare(&v.c.module,&decl).status==NVM_UFD_PREPARED);
        size_t declaration_bytes=ra_bytes-retained;calls=ra_calls;
        work=(NvmMapCopyWork){0,64,false};bytes=99;
        CHECK(nvm_union_declarations_retained_cost(decl,&work,&bytes)==NVM_MAP_COST_OK&&bytes==declaration_bytes&&work.used==64&&ra_calls==calls);
        work=(NvmMapCopyWork){0,63,false};bytes=99;
        CHECK(nvm_union_declarations_retained_cost(decl,&work,&bytes)==NVM_MAP_COST_LIMIT&&bytes==99);
        nvm_union_field_declarations_free(decl);CHECK(ra_bytes==retained);
#define UFC_BAD(field,value) do {uint64_t saved=(field);(field)=(value);work=(NvmMapCopyWork){0,UINT64_MAX,false};bytes=99;CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_INVALID&&bytes==99);(field)=saved;}while(0)
        UFC_BAD(p->carrier->snapshot.function_count,257);UFC_BAD(p->carrier->snapshot.string_count,NVM_MAX_STRINGS+1);
        UFC_BAD(p->carrier->counts.instructions,65537);UFC_BAD(p->site_count,p->site_count+1);
        UFC_BAD(p->carrier->counts.metadata,p->carrier->counts.metadata+1);
        UFC_BAD(p->carrier->counts.debug,p->carrier->counts.debug+1);
        UFC_BAD(p->carrier->counts.sections,p->carrier->counts.sections+1);
        UFC_BAD(p->carrier->functions[0].signature.arity,1);
        UFC_BAD(p->carrier->functions[0].signature.local_count,255);
        UFC_BAD(p->carrier->functions[0].instruction_start,1);
        UFC_BAD(p->carrier->functions[0].instruction_count,0);
        UFC_BAD(p->carrier->functions[0].parameter_tags_present,!p->carrier->functions[0].parameter_tags_present);
        UFC_BAD(p->proof->counts.origins,65);UFC_BAD(p->proof->counts.globals,257);
        UFC_BAD(p->proof->counts.functions,257);UFC_BAD(p->reserved,1);
#undef UFC_BAD
#define UFC_NULL(field) do {void *saved=(field);(field)=NULL;work=(NvmMapCopyWork){0,UINT64_MAX,false};bytes=99;CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_INVALID&&bytes==99);(field)=saved;}while(0)
        UFC_NULL(p->sites);UFC_NULL(p->carrier->instructions);UFC_NULL(p->carrier->snapshot.functions);
        UFC_NULL(p->carrier->snapshot.strings);UFC_NULL(p->carrier->snapshot.string_lengths);
        UFC_NULL(p->carrier->snapshot.strings[0]);UFC_NULL(p->proof->origins);UFC_NULL(p->proof->results);UFC_NULL(p->proof->stacks);
#undef UFC_NULL
        /* A pointer for a zero count cannot become an uncharged allocation. */
        CHECK(!p->proof->counts.globals&&!p->proof->globals);
        p->proof->globals=p->proof->results;work=(NvmMapCopyWork){0,UINT64_MAX,false};bytes=99;
        CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_INVALID&&bytes==99);p->proof->globals=NULL;
        work=(NvmMapCopyWork){0,needed,false};bytes=99;calls=ra_calls;
        CHECK(ufe_plan_retained_cost(p,&work,&bytes)==NVM_MAP_COST_OK&&bytes==retained&&ra_calls==calls);
        printf("union retained oracle case=%u bytes=%zu peak=%zu cost_work=%llu\n",variant,retained,peak,(unsigned long long)needed);
        nvm_union_field_execution_free(p);CHECK(!ra_live&&!ra_bytes);
    }
}

#endif
static void execution_original(const char *path) {
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
    NvmUnionFieldExecutionPlan *proof=NULL;NvmArrayEligibilityResult result=nvm_prepare_union_field_execution(m,&proof);
    if(result.status!=NVM_ARRAY_ELIGIBLE)fprintf(stderr,"original %s status=%u fn=%u pc=%u: %s\n",path,result.status,result.function,result.pc,result.message);
    CHECK(result.status==NVM_ARRAY_ELIGIBLE);execution_rows(proof);proof_public_refusal(m);
    NvmRecordArrayExecutionCounts counts;CHECK(nvm_union_field_execution_counts(proof,&counts));
    printf("original %s instructions=%u functions=%u reserved=%llu work=%llu\n",path,counts.instructions,counts.functions,
        (unsigned long long)counts.peak_bytes_reserved,(unsigned long long)counts.work_reserved);
    m->layout_data=NULL;m->layout_size=0;m->ownership_data=NULL;m->ownership_size=0;
    nvm_module_free(m);nvm_v2_module_free(&original_wire);free(bytes);memset(&v,0,sizeof v);
    NvmUnionFieldOrigin row;CHECK(nvm_union_field_execution_origin(proof,0,&row));
    nvm_union_field_execution_free(proof);
}
static void execution_bad_name(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);
    NvmUnionFieldExecutionPlan *prior=execution_query(&v.c.module),*out=prior;
    char bad[]="__init__\0suffix";
    char *strings[]={bad,"Outer","value","boxed","Some","None"};
    uint32_t lengths[]={sizeof bad-1,5,5,5,4,4};
    v.c.module.strings=strings;v.c.module.string_lengths=lengths;
    CHECK(nvm_prepare_union_field_execution(&v.c.module,&out).status==NVM_ARRAY_UNRESOLVED&&out==prior);
    execution_rows(prior);nvm_union_field_execution_free(prior);
}
static void execution_snapshot_limit(void) {
    UnionCase v;uint8_t code[128];execution_case(&v,code);
    NvmUnionFieldExecutionPlan *prior=execution_query(&v.c.module),*out=prior;
    const uint32_t length=17u*1024u*1024u;char *large=malloc((size_t)length+1);CHECK(large);
    memset(large,'x',length);large[length]=0;
    char *strings[7];uint32_t lengths[7];
    for(unsigned i=0;i<6;i++){strings[i]=v.c.module.strings[i];lengths[i]=v.c.module.string_lengths[i];}
    strings[6]=large;lengths[6]=length;
    char **old_strings=v.c.module.strings;uint32_t *old_lengths=v.c.module.string_lengths;
    v.c.module.strings=strings;v.c.module.string_lengths=lengths;v.c.module.string_count=7;
    CHECK(nvm_prepare_union_field_execution(&v.c.module,&out).status==NVM_ARRAY_LIMIT&&out==prior);
    CHECK(large[0]=='x'&&large[length-1]=='x'&&!large[length]);execution_rows(prior);
    v.c.module.strings=old_strings;v.c.module.string_lengths=old_lengths;v.c.module.string_count=6;free(large);
    out=execution_query(&v.c.module);nvm_union_field_execution_free(out);nvm_union_field_execution_free(prior);
}
int main(int argc,char **argv) {
    CHECK(argc==1 || argc==7);
#ifdef UFP_ALLOC
    execution_all_opcodes();execution_corruption();execution_setter_effects();execution_storage_edges();execution_allocations();execution_cost_oracles();
    printf("PASS %u union recipe allocation checks\n",checks);return 0;
#endif
    execution_basic();execution_empty_halt();execution_bad_name();execution_snapshot_limit();
    for(int i=1;i<argc;i++)execution_original(argv[i]);
    nvm_union_field_execution_free(NULL);
    printf("PASS %u union recipe checks; public execution remains refused\n",checks);return 0;
}
