/* Owning TU only: carrier holds snapshot/recipes and never old origin facts. */
_Static_assert(NVM_UF_INVALID_BASE_RECIPE>NVM_RA_ROOT_TAIL,"union sentinel cannot alias an old recipe");
struct NvmUnionFieldExecutionPlan {
    NvmRecordArrayExecutionPlan *carrier;
    NvmUnionFieldProof *proof;
    NvmUnionFieldSite *sites;
    uint32_t site_count,reserved;
};
void nvm_union_field_execution_free(NvmUnionFieldExecutionPlan *p) {
    if(!p)return;
    nvm_record_array_execution_free(p->carrier);
    nvm_union_field_proof_free(p->proof);free(p->sites);free(p);
}
static int ufe_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
    if(!uf_sensitive(op))return rae_old_recipe(op,r);
    r->recipe=NVM_UF_INVALID_BASE_RECIPE;
    r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;
    if(op==OP_HALT)return 1;
    r->obligations|=NVM_RA_CHECK_TAGS|NVM_RA_CHECK_NOMINAL;
    if(op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_AGG_PACK)
        r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;
    else if(op!=OP_AGG_TAG)
        r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;
    return 1;
}
static int ufe_empty(NvmRecordValueOrigins v) { return !v.tags && !v.origins && !v.unknown; }
typedef struct {
    NvmUnionFieldOrigin rows[64];
    uint64_t records,variants[2];
    uint32_t count;
} UfeOrigins;
static int ufe_origins_value(const UfeOrigins *o,NvmRecordValueOrigins v) {
    uint64_t unions=o->variants[0]|o->variants[1];
    if(v.unknown || (v.origins&~(o->records|unions)))return 0;
    if(!!(v.tags&BIT(TAG_STRUCT))!=!!(v.origins&o->records) ||
       !!(v.tags&BIT(TAG_UNION))!=!!(v.origins&unions))return 0;
    return 1;
}
static int ufe_origin_correspondence(const UfeOrigins *o,const NvmUnionFieldSite *s) {
    if(!ufe_origins_value(o,s->receiver) || !ufe_origins_value(o,s->input) || !ufe_origins_value(o,s->result))return 0;
    if(s->state!=UFS_PROVED)return 1;
    if(s->operation==UFS_VARIANT_PACK || s->operation==UFS_RECORD_PACK || s->operation==UFS_RECORD_NEW) {
        uint64_t site=0;
        for(uint32_t i=0;i<o->count;i++) {
            const NvmUnionFieldOrigin *r=&o->rows[i];
            if(r->function==s->function && r->pc==s->pc) {
                if(site || r->nominal_ordinal!=s->nominal || r->layout_index!=s->layout ||
                   r->member_count!=s->member || r->variant!=s->variant ||
                   r->kind!=(s->operation==UFS_VARIANT_PACK?NVM_UNION_FIELD_ORIGIN:NVM_HEAP_ORIGIN_RECORD) ||
                   (s->input.tags&~r->member.tags) || (s->input.origins&~r->member.origins))return 0;
                site=UINT64_C(1)<<i;
            }
        }
        return site && site==s->result.origins;
    }
    if(s->operation==UFS_PAYLOAD_GET)
        return s->variant<2 && s->receiver.origins && !(s->receiver.origins&~o->variants[s->variant]);
    if(s->operation==UFS_RECORD_GET || s->operation==UFS_RECORD_SET) {
        uint64_t children=0;uint16_t tags=0;
        for(uint32_t i=0;i<o->count;i++)if(s->receiver.origins&(UINT64_C(1)<<i)) {
            const NvmUnionFieldOrigin *r=&o->rows[i];
            if(r->kind!=NVM_HEAP_ORIGIN_RECORD || r->nominal_ordinal!=s->nominal || r->layout_index!=s->layout)return 0;
            if(s->operation==UFS_RECORD_SET && ((s->input.origins&~r->member.origins) || (s->input.tags&~r->member.tags)))return 0;
            children|=r->member.origins;tags|=r->member.tags;
        }
        if(s->operation==UFS_RECORD_GET)return children==s->result.origins && tags==s->result.tags;
    }
    return 1;
}
/* Internal immutable rows came from this constructor's successful analysis.
 * This checks decoded correspondence, not an externally supplied authority. */
static int ufe_corresponds(const NvmRecordArrayExecutionInstruction *r,const NvmUnionFieldSite *s,
                          uint32_t record_layout,uint32_t union_layout,const NvmUnionVariantFact variants[2],const UfeOrigins *origins) {
    if(!ufe_origin_correspondence(origins,s))return 0;
    NvmRecordArrayExecutionInstruction expected={0};
    if(!ufe_recipe(r->opcode,&expected) || expected.recipe!=r->recipe || expected.obligations!=r->obligations)return 0;
    if(r->function!=s->function || r->pc!=s->pc || s->reserved[0] || s->reserved[1] || s->reserved[2] ||
       s->receiver.unknown || s->input.unknown || s->result.unknown)return 0;
    if(!uf_sensitive(r->opcode))return s->state==UFS_NOT_REQUIRED && s->operation==UFS_NONE &&
        ufe_empty(s->receiver) && ufe_empty(s->input) && ufe_empty(s->result) &&
        s->nominal==UINT32_MAX && s->layout==UINT32_MAX && s->member==UINT16_MAX && s->variant==UINT8_MAX;
    if(r->recipe!=NVM_UF_INVALID_BASE_RECIPE)return 0;
    DecodedInstruction in={0};in.opcode=r->opcode;in.operands[0].u8=(uint8_t)r->operand_bits[0];
    uint8_t operation=uf_site_operation(&in);
    if(s->state==UFS_NOT_ESTABLISHED)return s->operation==operation &&
        ufe_empty(s->receiver) && ufe_empty(s->input) && ufe_empty(s->result) &&
        s->nominal==UINT32_MAX && s->layout==UINT32_MAX && s->member==UINT16_MAX && s->variant==UINT8_MAX;
    if(s->state!=UFS_PROVED)return 0;
    if(r->opcode==OP_HALT)return s->operation==UFS_HALT && !r->successor_count && !r->pops && !r->pushes &&
        ufe_empty(s->receiver) && ufe_empty(s->input) && ufe_empty(s->result) &&
        s->nominal==UINT32_MAX && s->layout==UINT32_MAX && s->member==UINT16_MAX && s->variant==UINT8_MAX;
    if(s->nominal!=0)return 0;
    switch(s->operation) {
    case UFS_RECORD_NEW:case UFS_RECORD_PACK:case UFS_VARIANT_PACK: {
        int variant=s->operation==UFS_VARIANT_PACK;
        if(s->operation!=operation || !ufe_empty(s->receiver) || s->layout!=(variant?union_layout:record_layout) ||
           s->result.tags!=BIT(variant?TAG_UNION:TAG_STRUCT) || !s->result.origins)return 0;
        if(r->opcode==OP_AGG_PACK) {
            if(r->operand_bits[0]!=(uint64_t)(variant?AGG_VARIANT:AGG_RECORD) || r->operand_bits[1]!=s->nominal ||
               r->operand_bits[2]!=(variant?s->variant:0) || r->operand_bits[3]!=s->member)return 0;
        } else if(r->operand_bits[0]!=s->nominal || (r->opcode==OP_STRUCT_LITERAL && r->operand_bits[1]!=s->member))return 0;
        uint16_t count=1;
        if(variant) {
            if(s->variant>=2)return 0;
            count=variants[s->variant].field_count;
        } else if(s->variant!=UINT8_MAX)return 0;
        return r->pushes==1 && r->pops==(r->opcode==OP_STRUCT_NEW?0:count) && s->member==count && (count?(s->input.tags==BIT(variant?TAG_INT:TAG_UNION) &&
             (variant?!s->input.origins:!!s->input.origins)):ufe_empty(s->input));
    }
    case UFS_RECORD_GET:
        return r->pops==1 && r->pushes==1 && (r->opcode==OP_STRUCT_GET || r->opcode==OP_AGG_GET) && !r->operand_bits[0] && !s->member &&
            s->layout==record_layout && s->variant==UINT8_MAX && s->receiver.tags==BIT(TAG_STRUCT) &&
            s->receiver.origins && ufe_empty(s->input) && s->result.tags==BIT(TAG_UNION) && s->result.origins;
    case UFS_PAYLOAD_GET:
        return r->pops==1 && r->pushes==1 && r->opcode==OP_AGG_GET && !r->operand_bits[0] && !s->member && s->layout==union_layout &&
            s->variant<2 && variants[s->variant].field_count==1 && s->receiver.tags==BIT(TAG_UNION) &&
            s->receiver.origins && ufe_empty(s->input) && s->result.tags==BIT(TAG_INT) && !s->result.origins;
    case UFS_RECORD_SET:
        return r->pops==2 && r->pushes==1 && (r->opcode==OP_STRUCT_SET || r->opcode==OP_AGG_SET) && !r->operand_bits[0] && !s->member &&
            s->layout==record_layout && s->variant==UINT8_MAX && s->receiver.tags==BIT(TAG_STRUCT) && s->receiver.origins &&
            s->input.tags==BIT(TAG_UNION) && s->input.origins && s->result.tags==s->receiver.tags && s->result.origins==s->receiver.origins;
    case UFS_TAG:
        return r->pops==1 && r->pushes==1 && r->opcode==OP_AGG_TAG && s->layout==union_layout && s->variant==UINT8_MAX && s->member==UINT16_MAX &&
            s->receiver.tags==BIT(TAG_UNION) && s->receiver.origins && ufe_empty(s->input) && s->result.tags==BIT(TAG_INT) && !s->result.origins;
    default:return 0;
    }
}
NvmArrayEligibilityResult nvm_prepare_union_field_execution(const NvmModule *m,NvmUnionFieldExecutionPlan **out) {
    NvmArrayEligibilityResult result={NVM_ARRAY_INVALID,0,0,"I require a module and copied union recipe output."};
    if(!m || !out)return result;
    uint64_t bytes=UINT64_C(64)*1024*1024+65536u+sizeof(NvmUnionFieldExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    uint64_t work=UINT64_C(16777216)+65536u+sizeof(NvmUnionFieldExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    if(bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || work>NVM_RECORD_ARRAY_EXECUTION_STEPS) {result.status=NVM_ARRAY_LIMIT;return result;}
    NvmUnionFieldExecutionPlan *p=calloc(1,sizeof *p);
    if(!p){result.status=NVM_ARRAY_MEMORY;return result;}
    p->carrier=calloc(1,sizeof *p->carrier);
    if(!p->carrier){result.status=NVM_ARRAY_MEMORY;nvm_union_field_execution_free(p);return result;}
    NvmRecordArrayExecutionPlan *c=p->carrier;
    c->bytes=bytes;c->work=work;c->failure=NVM_ARRAY_INVALID;c->counts.initializer=UINT32_MAX;
    if(!rae_snapshot(c,m))goto fail;
    /* Match counted initializer identity to analysis's C-string lookup. */
    for(uint32_t fi=0;fi<c->snapshot.function_count;fi++) {
        uint32_t name=c->snapshot.functions[fi].name_idx;
        if(name>=c->snapshot.string_count)goto fail;
        uint32_t length=c->snapshot.string_lengths[name];
        if(!rae_charge(c,0,(uint64_t)length+1))goto fail;
        if(memchr(c->snapshot.strings[name],0,length)){c->failure=NVM_ARRAY_UNRESOLVED;goto fail;}
    }
    NvmUnionFieldSiteExport sites={0};
    result=analyze_union_sites(&c->snapshot,&p->proof,&sites);
    if(result.status!=NVM_ARRAY_ELIGIBLE){nvm_union_field_execution_free(p);return result;}
    p->sites=sites.rows;p->site_count=sites.count;
    if(p->site_count>65536 || !rae_charge(c,(uint64_t)p->site_count*sizeof *p->sites,0))goto fail;
    if(!rae_charge(c,0,(uint64_t)c->snapshot.function_count*4u))goto fail;
    for(uint32_t i=0;i<c->snapshot.function_count;i++)
        if(!nvm_union_field_proof_stack(p->proof,i,&c->maximum_stacks[i]))goto fail;
    if(!rae_charge(c,0,256))goto fail;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction row={0};
        int supported=nvm_record_array_opcode_supported((uint8_t)op) || op==OP_AGG_TAG || op==OP_HALT;
        if(!!supported!=!!ufe_recipe((uint8_t)op,&row))goto fail;
    }
    uint32_t record_layout,union_layout;NvmDeclarationLayout record,uni;
    NvmUnionVariantFact variants[2];NvmV2LayoutField rf,uf;
    if(!rae_charge(c,0,32))goto fail;
    if(!nvm_union_field_proof_nominal(p->proof,TAG_STRUCT,0,&record_layout) ||
       !nvm_union_field_proof_nominal(p->proof,TAG_UNION,0,&union_layout) ||
       !nvm_union_field_proof_layout(p->proof,record_layout,&record) ||
       !nvm_union_field_proof_layout(p->proof,union_layout,&uni) ||
       !nvm_union_field_proof_variant(p->proof,0,0,&variants[0]) ||
       !nvm_union_field_proof_variant(p->proof,0,1,&variants[1]) ||
       !nvm_union_field_proof_field(p->proof,record_layout,0,&rf) ||
       !nvm_union_field_proof_field(p->proof,union_layout,0,&uf))goto fail;
    if(record_layout==union_layout || record.fields!=1 || uni.fields!=1 ||
       c->snapshot.struct_count!=1 || c->snapshot.union_count!=1)goto fail;
    c->counts.records=1;c->descriptors[0]=(NvmRecordArrayExecutionDescriptor){0,record_layout,record.fields};
    UfeOrigins origins={0};NvmUnionFieldProofCounts facts;
    if(!rae_charge(c,0,4) || !nvm_union_field_proof_counts(p->proof,&facts) || facts.origins>64)goto fail;
    origins.count=facts.origins;
    if(!rae_charge(c,0,(uint64_t)origins.count*5u))goto fail;
    for(uint32_t i=0;i<origins.count;i++) {
        NvmUnionFieldOrigin *r=&origins.rows[i];
        if(!nvm_union_field_proof_origin(p->proof,i,r) || r->nominal_ordinal!=0 || r->member.unknown)goto fail;
        if(r->kind==NVM_HEAP_ORIGIN_RECORD) {
            if(r->layout_index!=record_layout || r->variant!=UINT8_MAX || r->member_count!=1 || r->member.tags!=BIT(TAG_UNION))goto fail;
            origins.records|=UINT64_C(1)<<i;
        } else if(r->kind==NVM_UNION_FIELD_ORIGIN) {
            if(r->layout_index!=union_layout || r->variant>=2 || r->member_count!=variants[r->variant].field_count ||
               (r->member_count?(r->member.tags!=BIT(TAG_INT) || r->member.origins):!ufe_empty(r->member)))goto fail;
            origins.variants[r->variant]|=UINT64_C(1)<<i;
        } else goto fail;
    }
    if(!rae_charge(c,0,origins.count))goto fail;
    for(uint32_t i=0;i<origins.count;i++)if(!ufe_origins_value(&origins,origins.rows[i].member))goto fail;
    if(!rae_instructions_policy(c,ufe_recipe,1) || c->counts.instructions!=p->site_count)goto fail;
    /* Each row's constant checks plus at most64 frozen origin visits. */
    if(!rae_charge(c,0,(uint64_t)p->site_count*80u))goto fail;
    for(uint32_t i=0;i<p->site_count;i++)
        if(!ufe_corresponds(&c->instructions[i],&p->sites[i],record_layout,union_layout,variants,&origins))goto fail;
    if(!rae_same(c,m))goto fail;
    c->counts.strings=c->snapshot.string_count;c->counts.sections=c->snapshot.section_count;
    c->counts.metadata=c->snapshot.metadata_count;c->counts.debug=c->snapshot.debug_count;
    c->counts.functions=c->snapshot.function_count;c->counts.entry=c->snapshot.header.entry_point;
    c->counts.frame_limit=NVM_RECORD_ARRAY_EXECUTION_FRAMES;
    c->counts.peak_bytes_reserved=c->bytes;c->counts.work_reserved=c->work;
    *out=p;result.status=NVM_ARRAY_ELIGIBLE;
    snprintf(result.message,sizeof result.message,"I copied union recipes without executable admission.");return result;
fail:
    result.status=c->failure;
    snprintf(result.message,sizeof result.message,"I could not copy exact bounded union recipes.");
    nvm_union_field_execution_free(p);return result;
}
bool nvm_union_field_execution_counts(const NvmUnionFieldExecutionPlan *p,NvmRecordArrayExecutionCounts *out) {
    return p && nvm_record_array_execution_counts(p->carrier,out);
}
bool nvm_union_field_execution_instruction(const NvmUnionFieldExecutionPlan *p,uint32_t i,NvmUnionFieldExecutionInstruction *out) {
    if(!p || !out || i>=p->site_count || i>=p->carrier->counts.instructions)return false;
    NvmUnionFieldExecutionInstruction row={p->carrier->instructions[i],p->sites[i]};*out=row;return true;
}
bool nvm_union_field_execution_function(const NvmUnionFieldExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionFunction *out) {
    return p && nvm_record_array_execution_function(p->carrier,i,out);
}
bool nvm_union_field_execution_parameter(const NvmUnionFieldExecutionPlan *p,uint32_t fi,uint16_t i,uint8_t *out) {
    return p && nvm_record_array_execution_parameter(p->carrier,fi,i,out);
}
bool nvm_union_field_execution_header(const NvmUnionFieldExecutionPlan *p,NvmHeader *out) {
    return p && nvm_record_array_execution_header(p->carrier,out);
}
bool nvm_union_field_execution_size(const NvmUnionFieldExecutionPlan *p,NvmRecordArraySnapshotKind k,uint32_t i,uint32_t *out) {
    return p && nvm_record_array_execution_size(p->carrier,k,i,out);
}
bool nvm_union_field_execution_bytes(const NvmUnionFieldExecutionPlan *p,NvmRecordArraySnapshotKind k,uint32_t i,uint32_t offset,uint32_t count,void *out) {
    return p && nvm_record_array_execution_bytes(p->carrier,k,i,offset,count,out);
}
bool nvm_union_field_execution_proof_counts(const NvmUnionFieldExecutionPlan *p,NvmUnionFieldProofCounts *out) {
    return p && nvm_union_field_proof_counts(p->proof,out);
}
#define UFE_QUERY(name,type) \
bool nvm_union_field_execution_##name(const NvmUnionFieldExecutionPlan *p,uint32_t i,type *out) { \
    return p && nvm_union_field_proof_##name(p->proof,i,out); }
UFE_QUERY(origin,NvmUnionFieldOrigin)
UFE_QUERY(global,NvmRecordValueOrigins)
UFE_QUERY(result,NvmRecordValueOrigins)
UFE_QUERY(layout,NvmDeclarationLayout)
#undef UFE_QUERY
bool nvm_union_field_execution_field(const NvmUnionFieldExecutionPlan *p,uint32_t i,uint16_t j,NvmV2LayoutField *out) {
    return p && nvm_union_field_proof_field(p->proof,i,j,out);
}
bool nvm_union_field_execution_variant(const NvmUnionFieldExecutionPlan *p,uint32_t i,uint16_t j,NvmUnionVariantFact *out) {
    return p && nvm_union_field_proof_variant(p->proof,i,j,out);
}
bool nvm_union_field_execution_nominal(const NvmUnionFieldExecutionPlan *p,uint8_t tag,uint32_t i,uint32_t *out) {
    return p && nvm_union_field_proof_nominal(p->proof,tag,i,out);
}
