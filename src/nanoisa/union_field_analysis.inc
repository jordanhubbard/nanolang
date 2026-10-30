/* Opt-in union facts; no old projection or execution owner enters this path. */
struct NvmUnionFieldProof {
    NvmUnionFieldDeclarations *declarations;
    NvmUnionFieldOrigin *origins;
    NvmRecordValueOrigins *globals,*results;
    uint16_t *stacks;
    NvmUnionFieldProofCounts counts;
};
void nvm_union_field_proof_free(NvmUnionFieldProof *p) {
    if(!p)return;
    nvm_union_field_declarations_free(p->declarations);
    free(p->origins);free(p->globals);free(p->results);free(p->stacks);free(p);
}
bool nvm_union_field_proof_counts(const NvmUnionFieldProof *p,NvmUnionFieldProofCounts *out) {
    if(!p || !out)return false;
    *out=p->counts;return true;
}
#define UF_COPY_QUERY(name,type,member,count) \
bool name(const NvmUnionFieldProof *p,uint32_t i,type *out) { \
    if(!p || !out || i>=p->counts.count)return false; \
    *out=p->member[i];return true; \
}
UF_COPY_QUERY(nvm_union_field_proof_origin,NvmUnionFieldOrigin,origins,origins)
UF_COPY_QUERY(nvm_union_field_proof_global,NvmRecordValueOrigins,globals,globals)
UF_COPY_QUERY(nvm_union_field_proof_result,NvmRecordValueOrigins,results,functions)
UF_COPY_QUERY(nvm_union_field_proof_stack,uint16_t,stacks,functions)
#undef UF_COPY_QUERY
bool nvm_union_field_proof_layout(const NvmUnionFieldProof *p,uint32_t i,NvmDeclarationLayout *out) {
    return p && nvm_union_field_declarations_layout(p->declarations,i,out);
}
bool nvm_union_field_proof_field(const NvmUnionFieldProof *p,uint32_t i,uint16_t j,NvmV2LayoutField *out) {
    return p && nvm_union_field_declarations_field(p->declarations,i,j,out);
}
bool nvm_union_field_proof_variant(const NvmUnionFieldProof *p,uint32_t i,uint16_t j,NvmUnionVariantFact *out) {
    return p && nvm_union_field_declarations_variant(p->declarations,i,j,out);
}
bool nvm_union_field_proof_nominal(const NvmUnionFieldProof *p,uint8_t tag,uint32_t i,uint32_t *out) {
    return p && nvm_union_field_declarations_nominal(p->declarations,tag,i,out);
}
static int uf_prepare_records(Analysis *a,NvmUnionAnalysisContext *u) {
    a->plan=analysis_alloc(a,1,sizeof *a->plan);
    if(!a->plan)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy union record mappings.");
    NvmRecordPlan *p=a->plan;
    p->layouts.items=analysis_alloc(a,2,sizeof *p->layouts.items);
    p->record_to_layout=analysis_alloc(a,1,sizeof *p->record_to_layout);
    p->layout_to_record=analysis_alloc(a,2,sizeof *p->layout_to_record);
    if(!p->layouts.items || !p->record_to_layout || !p->layout_to_record)
        return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy union layout mappings.");
    p->layouts.count=2;p->record_count=1;p->authority=NVM_RECORD_AUTHORITY_ORDINARY;
    if(!analysis_work(a,4))return 0;
    if(!nvm_union_field_declarations_nominal(u->declarations,TAG_STRUCT,0,&p->record_to_layout[0]))
        return stop(a,NVM_ARRAY_INVALID,0,0,"I require an exact copied record nominal.");
    for(uint32_t i=0;i<2;i++) {
        NvmDeclarationLayout row;
        if(!analysis_work(a,8))return 0;
        if(!nvm_union_field_declarations_layout(u->declarations,i,&row) || row.fields!=1)
            return stop(a,NVM_ARRAY_INVALID,0,0,"I require the exact union-field layout rows.");
        p->layout_to_record[i]=i==p->record_to_layout[0]?0:NVM_V2_NO_INDEX;
        NvmV2Layout *l=&p->layouts.items[i];l->kind=row.kind;l->name_idx=row.name;
        l->fields=analysis_alloc(a,1,sizeof *l->fields);
        if(!l->fields)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy a union layout field.");
        l->field_count=1;
        if(!nvm_union_field_declarations_field(u->declarations,i,0,l->fields))
            return stop(a,NVM_ARRAY_INVALID,0,0,"I require a copied union field.");
    }
    return 1;
}
static int uf_site(Analysis *a,NvmUnionAnalysisContext *u,uint32_t fi,uint32_t pc) {
    Function *f=&a->functions[fi];VmDecodedInstruction *d=&f->decoded.instructions[pc];
    DecodedInstruction *in=&d->instruction;
    if(in->opcode!=OP_AGG_PACK || in->operands[0].u8!=AGG_VARIANT)
        return record_site(a,fi,pc);
    NvmUnionVariantFact v;uint32_t layout;
    if(!analysis_work(a,8))return 0;
    if(!nvm_union_field_declarations_nominal(u->declarations,TAG_UNION,in->operands[1].u32,&layout) ||
       !nvm_union_field_declarations_variant(u->declarations,in->operands[1].u32,in->operands[2].u16,&v) ||
       v.layout!=layout || v.field_count!=in->operands[3].u16)
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I require an exact nominal variant construction.");
    if(a->report.origin_count==ORIGINS)
        return stop(a,NVM_ARRAY_LIMIT,fi,d->byte_offset,"I reached my shared union origin bound.");
    uint32_t o=a->report.origin_count++;f->origins[pc]=(int16_t)o;
    a->origin_kind[o]=NVM_UNION_FIELD_ORIGIN;a->record_ordinal[o]=in->operands[1].u32;
    a->record_fields[o]=v.field_count;u->variant[o]=(uint8_t)in->operands[2].u16;
    a->report.origins[o]=(NvmArrayOrigin){fi,d->byte_offset,0,TAG_UNION,0};
    return 1;
}
/* Exact kind/nominal assignment, never pointer-derived provenance. */
static int uf_value(Analysis *a,Value v,uint8_t expected,uint32_t fi,uint32_t pc) {
    if(!analysis_work(a,ORIGINS+1))return 0;
    if(v.unknown || v.tags!=BIT(expected))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require exact known union-field value tags.");
    if(expected!=TAG_STRUCT && expected!=TAG_UNION) {
        if(v.origins)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I refuse scalar heap origins.");
        return 1;
    }
    uint8_t kind=expected==TAG_STRUCT?NVM_HEAP_ORIGIN_RECORD:NVM_UNION_FIELD_ORIGIN;
    uint64_t matched=0;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(v.origins&(UINT64_C(1)<<i)) {
        if(a->origin_kind[i]!=kind || a->record_ordinal[i]!=0)
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require exact union-field nominal origins.");
        matched|=UINT64_C(1)<<i;
    }
    if(!matched || matched!=v.origins)
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require complete union-field origins.");
    return 1;
}
/* During fixed point, unknown/bottom may block successors. Final validation
 * calls the same obligation with final=true and rejects the reachable site. */
static int uf_access(Analysis *a,NvmUnionAnalysisContext *u,Value receiver,
                     uint16_t field,int writing,Value value,uint32_t fi,uint32_t pc,
                     int final,Value *result) {
    if(receiver.unknown || !receiver.tags || !receiver.origins) {
        if(final)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I cannot resolve this reachable union-field receiver.");
        return 2;
    }
    if(receiver.tags==BIT(TAG_STRUCT)) {
        if(!uf_value(a,receiver,TAG_STRUCT,fi,pc))return 0;
        if(field)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require the exact containing record field.");
        if(writing) {
            if(value.unknown || !value.tags || !value.origins) {
                if(final)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I cannot resolve this historical union write.");
                return 2;
            }
            if(!uf_value(a,value,TAG_UNION,fi,pc))return 0;
            if(!final)write_record(a,receiver,field,value);
            *result=receiver;
        } else *result=read_record(a,receiver,field);
    } else {
        if(!uf_value(a,receiver,TAG_UNION,fi,pc))return 0;
        if(writing)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I have not qualified union payload mutation.");
        if(field)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require the sole INT payload field.");
        if(!analysis_work(a,ORIGINS))return 0;
        *result=(Value){0};
        for(uint32_t i=0;i<a->report.origin_count;i++)if(receiver.origins&(UINT64_C(1)<<i)) {
            if(a->record_fields[i]!=1)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I refuse an empty variant at payload access.");
            merge(result,u->payloads[i]);
        }
    }
    if(result->unknown || !result->tags || (!result->origins && result->tags==BIT(TAG_UNION))) {
        if(final)return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I cannot resolve this reachable union-field result.");
        return 2;
    }
    return !a->budget.limited;
}
static int uf_operation(Analysis *a,NvmUnionAnalysisContext *u,uint32_t fi,uint32_t pc) {
    Function *f=&a->functions[fi];VmDecodedInstruction *d=&f->decoded.instructions[pc];
    DecodedInstruction *in=&d->instruction;uint8_t op=in->opcode;
    Value *stack=f->states+(size_t)pc*f->stride+f->locals;uint16_t depth=f->depths[pc];
    if(op==OP_AGG_TAG)return uf_value(a,stack[depth-1],TAG_UNION,fi,d->byte_offset);
    if(op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_AGG_PACK) {
        uint32_t o=(uint32_t)f->origins[pc];uint16_t n=a->record_fields[o];
        if(n && !uf_value(a,stack[depth-n],a->origin_kind[o]==NVM_UNION_FIELD_ORIGIN?TAG_INT:TAG_UNION,fi,d->byte_offset))return 0;
        return 1;
    }
    int writing=op==OP_STRUCT_SET || op==OP_AGG_SET;Value result={0};
    return uf_access(a,u,stack[depth-(writing?2:1)],in->operands[0].u16,writing,
                     writing?stack[depth-1]:(Value){0},fi,d->byte_offset,1,&result)==1;
}
static int uf_summary(Analysis *a,Value v,uint32_t fi,uint32_t pc) {
    if(!analysis_work(a,ORIGINS+1))return 0;
    uint16_t allowed=LEAVES|BIT(TAG_STRUCT)|BIT(TAG_UNION);
    if(v.unknown || !v.tags || (v.tags&~allowed))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require complete bounded historical union summaries.");
    uint64_t valid=0;uint16_t heap_tags=0;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(v.origins&(UINT64_C(1)<<i)) {
        uint16_t bit=a->origin_kind[i]==NVM_HEAP_ORIGIN_RECORD?BIT(TAG_STRUCT):
            a->origin_kind[i]==NVM_UNION_FIELD_ORIGIN?BIT(TAG_UNION):0;
        if(!bit || !(v.tags&bit))return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require matching historical union origin kinds.");
        valid|=UINT64_C(1)<<i;heap_tags|=bit;
    }
    if(valid!=v.origins || heap_tags!=(v.tags&(BIT(TAG_STRUCT)|BIT(TAG_UNION))))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require every historical heap tag to have exact origins.");
    return 1;
}
static NvmRecordValueOrigins uf_copy_value(Value v) {
    NvmRecordValueOrigins out={0};out.tags=v.tags;out.origins=v.origins;return out;
}
/* Dense optional output is private to this owning TU; never a trusted token. */
typedef struct { NvmUnionFieldSite *rows;uint32_t count,reserved; } NvmUnionFieldSiteExport;
static int uf_outputs_overlap(const void *a,size_t an,const void *b,size_t bn) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y?y-x<an:x-y<bn;
}
static int uf_sensitive(uint8_t op) { return record_operation(op) || op==OP_AGG_TAG || op==OP_HALT; }
static uint8_t uf_site_operation(const DecodedInstruction *in) {
    switch(in->opcode) {
    case OP_STRUCT_NEW:return UFS_RECORD_NEW;
    case OP_STRUCT_LITERAL:return UFS_RECORD_PACK;
    case OP_AGG_PACK:return in->operands[0].u8==AGG_VARIANT?UFS_VARIANT_PACK:UFS_RECORD_PACK;
    case OP_STRUCT_GET:return UFS_RECORD_GET;
    case OP_STRUCT_SET:case OP_AGG_SET:return UFS_RECORD_SET;
    case OP_AGG_TAG:return UFS_TAG;
    case OP_HALT:return UFS_HALT;
    default:return UFS_NONE;
    }
}
static int uf_export_sites(Analysis *a,NvmUnionAnalysisContext *u,NvmUnionFieldSiteExport *out) {
    uint32_t count=0;
    if(!analysis_work(a,a->module->function_count))return 0;
    for(uint32_t fi=0;fi<a->module->function_count;fi++) {
        uint32_t n=a->functions[fi].decoded.instruction_count;
        if(n>65536u-count)return stop(a,NVM_ARRAY_LIMIT,fi,0,"I reached the union site count bound.");
        count+=n;
    }
    if(!analysis_work(a,(uint64_t)count*512u))return 0;
    NvmUnionFieldSite *rows=count?analysis_alloc(a,count,sizeof *rows):NULL;
    if(count && !rows)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy final union sites.");
    uint32_t index=0;
    for(uint32_t fi=0;fi<a->module->function_count;fi++) {
        Function *f=&a->functions[fi];
        for(uint32_t pc=0;pc<f->decoded.instruction_count;pc++,index++) {
            VmDecodedInstruction *d=&f->decoded.instructions[pc];DecodedInstruction *in=&d->instruction;
            NvmUnionFieldSite *r=&rows[index];r->function=fi;r->pc=d->byte_offset;
            r->nominal=r->layout=UINT32_MAX;r->member=UINT16_MAX;r->variant=UINT8_MAX;
            if(!uf_sensitive(in->opcode))continue;
            r->state=UFS_NOT_ESTABLISHED;r->operation=uf_site_operation(in);
            if(!f->seen[pc])continue; /* Never certifies unreachability or execution. */
            r->state=UFS_PROVED;
            if(in->opcode==OP_HALT)continue;
            Value receiver={0},input={0},result={0};uint16_t depth=f->depths[pc];
            Value *stack=f->states+(size_t)pc*f->stride+f->locals;
            uint8_t nominal_tag=TAG_VOID;
            if(in->opcode==OP_STRUCT_NEW || in->opcode==OP_STRUCT_LITERAL || in->opcode==OP_AGG_PACK) {
                int32_t o=f->origins[pc];
                if(o<0 || (uint32_t)o>=a->report.origin_count)goto invalid;
                r->member=a->record_fields[o];r->nominal=a->record_ordinal[o];
                nominal_tag=a->origin_kind[o]==NVM_UNION_FIELD_ORIGIN?TAG_UNION:TAG_STRUCT;
                if(r->member>1 || depth<r->member)goto invalid;
                if(r->member)input=stack[depth-r->member];
                result=tag(nominal_tag);result.origins=UINT64_C(1)<<o;
                if(nominal_tag==TAG_UNION)r->variant=u->variant[o];
            } else if(in->opcode==OP_AGG_TAG) {
                if(!depth)goto invalid;
                receiver=stack[depth-1];result=tag(TAG_INT);nominal_tag=TAG_UNION;r->nominal=0;
            } else {
                int writing=in->opcode==OP_STRUCT_SET || in->opcode==OP_AGG_SET;
                if(depth<(writing?2:1))goto invalid;
                receiver=stack[depth-(writing?2:1)];if(writing)input=stack[depth-1];
                if(uf_access(a,u,receiver,in->operands[0].u16,writing,input,fi,d->byte_offset,1,&result)!=1)goto fail;
                r->member=in->operands[0].u16;r->nominal=0;
                nominal_tag=receiver.tags==BIT(TAG_STRUCT)?TAG_STRUCT:TAG_UNION;
                r->operation=writing?UFS_RECORD_SET:nominal_tag==TAG_STRUCT?UFS_RECORD_GET:UFS_PAYLOAD_GET;
                if(nominal_tag==TAG_UNION) {
                    for(uint32_t j=0;j<a->report.origin_count;j++)if(receiver.origins&(UINT64_C(1)<<j)) {
                        if(r->variant!=UINT8_MAX && r->variant!=u->variant[j])goto invalid;
                        r->variant=u->variant[j];
                    }
                }
            }
            /* Existing validation helpers retain their own explicit charges. */
            if((receiver.tags && !uf_summary(a,receiver,fi,d->byte_offset)) ||
               (input.tags && !uf_summary(a,input,fi,d->byte_offset)) ||
               !uf_summary(a,result,fi,d->byte_offset))goto fail;
            if(!analysis_work(a,4))goto fail;
            if(!nvm_union_field_declarations_nominal(u->declarations,nominal_tag,r->nominal,&r->layout))goto invalid;
            r->receiver=uf_copy_value(receiver);r->input=uf_copy_value(input);r->result=uf_copy_value(result);
        }
    }
    *out=(NvmUnionFieldSiteExport){rows,count,0};return 1;
invalid:
    stop(a,NVM_ARRAY_INVALID,0,0,"I require exact final union site correspondence.");
fail:
    free(rows);return 0;
}
static int uf_freeze(Analysis *a,NvmUnionAnalysisContext *u,NvmUnionFieldProof **out) {
    uint32_t origins=a->report.origin_count,globals=a->global_count,functions=a->module->function_count;
    if(!analysis_work(a,origins+globals+functions))return 0;
    /* Historical rows are checked independently of reachable final operands. */
    for(uint32_t i=0;i<origins;i++) {
        Value v;
        if(a->origin_kind[i]==NVM_HEAP_ORIGIN_RECORD) {
            v=a->fields[a->field_start[i]];
            if(!uf_value(a,v,TAG_UNION,a->report.origins[i].function,a->report.origins[i].pc))return 0;
        } else if(a->origin_kind[i]==NVM_UNION_FIELD_ORIGIN) {
            if(a->record_fields[i] && !uf_value(a,u->payloads[i],TAG_INT,a->report.origins[i].function,a->report.origins[i].pc))return 0;
        } else return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I have not qualified other heap origins with union fields.");
    }
    for(uint32_t i=0;i<globals;i++)if(!uf_summary(a,a->globals[i],0,0))return 0;
    for(uint32_t i=0;i<functions;i++)
        if(a->module->functions[i].result_count && !uf_summary(a,a->functions[i].result,i,0))return 0;
    NvmUnionFieldProof *p=analysis_alloc(a,1,sizeof *p);
    if(!p)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate copied union proof.");
    if(origins)p->origins=analysis_alloc(a,origins,sizeof *p->origins);
    if(globals)p->globals=analysis_alloc(a,globals,sizeof *p->globals);
    p->results=analysis_alloc(a,functions,sizeof *p->results);
    p->stacks=analysis_alloc(a,functions,sizeof *p->stacks);
    if((origins && !p->origins) || (globals && !p->globals) || !p->results || !p->stacks) {
        nvm_union_field_proof_free(p);return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate copied union rows.");
    }
    if(!analysis_work(a,(uint64_t)origins*5u+globals+functions)) {
        nvm_union_field_proof_free(p);return 0;
    }
    for(uint32_t i=0;i<origins;i++) {
        NvmUnionFieldOrigin *row=&p->origins[i];
        row->function=a->report.origins[i].function;row->pc=a->report.origins[i].pc;
        row->nominal_ordinal=a->record_ordinal[i];row->kind=a->origin_kind[i];
        row->variant=row->kind==NVM_UNION_FIELD_ORIGIN?u->variant[i]:UINT8_MAX;
        row->member_count=a->record_fields[i];
        if(!nvm_union_field_declarations_nominal(u->declarations,
            row->kind==NVM_UNION_FIELD_ORIGIN?TAG_UNION:TAG_STRUCT,row->nominal_ordinal,&row->layout_index)) {
            nvm_union_field_proof_free(p);return stop(a,NVM_ARRAY_INVALID,0,0,"I require copied nominal identity during union freeze.");
        }
        if(row->member_count)row->member=uf_copy_value(row->kind==NVM_UNION_FIELD_ORIGIN?
            u->payloads[i]:a->fields[a->field_start[i]]);
    }
    for(uint32_t i=0;i<globals;i++)p->globals[i]=uf_copy_value(a->globals[i]);
    for(uint32_t i=0;i<functions;i++) {
        if(a->module->functions[i].result_count)p->results[i]=uf_copy_value(a->functions[i].result);
        p->stacks[i]=a->structure->stacks[i];
    }
    p->counts=(NvmUnionFieldProofCounts){origins,globals,functions,a->budget.peak,a->budget.work};
    p->declarations=u->declarations;u->declarations=NULL;
    *out=p;return 1;
}
