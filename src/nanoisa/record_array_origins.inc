struct NvmRecordArrayOrigins {
    NvmOwnershipDeclarationPlan *declarations;
    NvmRecordEligibilityReport *origins;
    uint16_t required[ORIGINS];
    NvmRecordArrayOriginCounts counts;
};
static int publish_records(Analysis *,NvmRecordEligibilityReport **);
static int ra_prepare_records(Analysis *a) {
    NvmDeclarationCounts counts;
    if(!nvm_ownership_declarations_counts(a->structure->declarations,&counts) || counts.layouts>256)
        return stop(a,NVM_ARRAY_INVALID,0,0,"I require complete declaration counts.");
    a->plan=analysis_alloc(a,1,sizeof *a->plan);
    if(!a->plan)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate copied record maps.");
    NvmRecordPlan *p=a->plan;
    p->layouts.items=analysis_alloc(a,counts.layouts?counts.layouts:1,sizeof *p->layouts.items);
    p->record_to_layout=analysis_alloc(a,counts.layouts?counts.layouts:1,sizeof *p->record_to_layout);
    p->layout_to_record=analysis_alloc(a,counts.layouts?counts.layouts:1,sizeof *p->layout_to_record);
    if(!p->layouts.items || !p->record_to_layout || !p->layout_to_record)
        return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate copied layout maps.");
    p->layouts.count=counts.layouts;p->authority=NVM_RECORD_AUTHORITY_ORDINARY;
    uint32_t fields=0;
    for(uint32_t i=0;i<counts.layouts;i++) {
        NvmDeclarationLayout row;
        if(!nvm_ownership_declarations_layout(a->structure->declarations,i,&row))
            return stop(a,NVM_ARRAY_INVALID,0,0,"I require a copied declaration row.");
        if(!analysis_work(a,(uint64_t)row.fields*4u+1))return 0;
        a->layout_fields[i]=fields;
        if(row.fields>65536u-fields)return stop(a,NVM_ARRAY_LIMIT,0,0,"I reached my copied field bound.");
        fields+=row.fields;p->layout_to_record[i]=NVM_V2_NO_INDEX;
        if(row.kind==NVM_V2_LAYOUT_STRUCT) {
            p->layout_to_record[i]=p->record_count;p->record_to_layout[p->record_count++]=i;
            if(row.flags!=NVM_LAYOUT_COMPLETE)return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require complete ordinary records.");
        } else if(row.kind!=NVM_V2_LAYOUT_UNION || row.flags)
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require validated ordinary records and scalar union declarations.");
        NvmV2Layout *layout=&p->layouts.items[i];layout->kind=row.kind;layout->name_idx=row.name;
        if(row.fields) {
            layout->fields=analysis_alloc(a,row.fields,sizeof *layout->fields);
            if(!layout->fields)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy declaration fields.");
        }
        layout->field_count=row.fields;
        for(uint16_t j=0;j<row.fields;j++)
            if(!nvm_ownership_declarations_field(a->structure->declarations,i,j,&layout->fields[j]))
                return stop(a,NVM_ARRAY_INVALID,0,0,"I require each copied field fact.");
    }
    if(p->record_count!=a->module->struct_count)
        return stop(a,NVM_ARRAY_INVALID,0,0,"I require the original nominal record count.");
    a->field_elements=analysis_alloc(a,fields?fields:1,1);
    if(!a->field_elements)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy field element constraints.");
    if(!analysis_work(a,(uint64_t)counts.bindings*4u+fields))return 0;
    for(uint32_t i=0;i<counts.bindings;i++) {
        NvmOrdinaryArrayBinding binding;NvmOrdinaryArrayType element;
        if(!nvm_ownership_declarations_binding(a->structure->declarations,i,&binding) ||
           !nvm_ownership_declarations_type(a->structure->declarations,binding.element_type,&element) ||
           binding.layout>=p->layouts.count || binding.field>=p->layouts.items[binding.layout].field_count)
            return stop(a,NVM_ARRAY_INVALID,0,0,"I require exact copied array bindings.");
        if(element.referent!=NVM_V2_NO_INDEX || !(element.tag==TAG_INT || element.tag==TAG_U8 ||
             element.tag==TAG_FLOAT || element.tag==TAG_BOOL || element.tag==TAG_STRING))
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require flat exact field element tags.");
        a->field_elements[a->layout_fields[binding.layout]+binding.field]=element.tag;
    }
    for(uint32_t i=0;i<p->layouts.count;i++) {
        NvmV2Layout *layout=&p->layouts.items[i];
        if(layout->kind!=NVM_V2_LAYOUT_STRUCT)continue;
        for(uint16_t j=0;j<layout->field_count;j++) {
            uint8_t tag=layout->fields[j].type_tag;
            if(tag==TAG_ARRAY) {
                if(!a->field_elements[a->layout_fields[i]+j])
                    return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require every array field binding.");
            } else if(!(tag==TAG_INT || tag==TAG_U8 || tag==TAG_FLOAT || tag==TAG_BOOL || tag==TAG_STRING || tag==TAG_STRUCT))
                return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require the bounded ordinary record field schema.");
        }
    }
    return 1;
}
static int ra_check_constraints(Analysis *a) {
    if(!analysis_work(a,ORIGINS*16u))return 0;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        Value writes=a->all_writes[i],children=a->children[i];
        if(!(RECORD_ARRAY_LEAVES&BIT(a->report.origins[i].declared_tag)) ||
           writes.unknown || writes.origins || (writes.tags&~RECORD_ARRAY_LEAVES) ||
           children.unknown || children.origins || (children.tags&~RECORD_ARRAY_LEAVES) ||
           (a->report.origins[i].child_tags&~RECORD_ARRAY_LEAVES))
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require flat scalar/string content for every array origin.");
        if(a->required_elements[i] && (writes.tags&~a->required_elements[i]))
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require every aliased array write to preserve its exact field element type.");
    }
    return 1;
}
static int ra_publish(Analysis *a,NvmRecordArrayOrigins **out) {
    NvmRecordArrayOrigins *p=analysis_alloc(a,1,sizeof *p);
    if(!p)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate the origin report.");
    if(!analysis_work(a,(uint64_t)a->field_count+ORIGINS*2u) || !publish_records(a,&p->origins)) {free(p);return 0;}
    memcpy(p->required,a->required_elements,sizeof p->required);
    p->counts=(NvmRecordArrayOriginCounts){p->origins->origin_count,p->origins->field_value_count,
        p->origins->checked_field_writes,p->origins->checked_array_writes,p->origins->runtime_tag_checks,
        a->budget.work,a->budget.peak};
    p->declarations=a->structure->declarations;a->structure->declarations=NULL;
    *out=p;return 1;
}
void nvm_record_array_origins_free(NvmRecordArrayOrigins *p) {
    if(!p)return;
    nvm_ownership_declarations_free(p->declarations);nvm_record_eligibility_free(p->origins);free(p);
}
bool nvm_record_array_origins_counts(const NvmRecordArrayOrigins *p,NvmRecordArrayOriginCounts *out) {
    if(!p||!out)return false;
    *out=p->counts;return true;
}
bool nvm_record_array_origin(const NvmRecordArrayOrigins *p,uint32_t i,NvmRecordHeapOrigin *out) {
    if(!p||!out||i>=p->origins->origin_count)return false;
    *out=p->origins->origins[i];return true;
}
bool nvm_record_array_field_value(const NvmRecordArrayOrigins *p,uint32_t i,NvmRecordValueOrigins *out) {
    if(!p||!out||i>=p->origins->field_value_count)return false;
    *out=p->origins->fields[i];return true;
}
bool nvm_record_array_required_elements(const NvmRecordArrayOrigins *p,uint32_t i,uint16_t *out) {
    if(!p||!out||i>=p->origins->origin_count)return false;
    *out=p->required[i];return true;
}
bool nvm_record_array_declaration_counts(const NvmRecordArrayOrigins *p,NvmDeclarationCounts *out) {
    return p && nvm_ownership_declarations_counts(p->declarations,out);
}
bool nvm_record_array_declaration_layout(const NvmRecordArrayOrigins *p,uint32_t i,NvmDeclarationLayout *out) {
    return p && nvm_ownership_declarations_layout(p->declarations,i,out);
}
bool nvm_record_array_declaration_field(const NvmRecordArrayOrigins *p,uint32_t i,uint16_t f,NvmV2LayoutField *out) {
    return p && nvm_ownership_declarations_field(p->declarations,i,f,out);
}
bool nvm_record_array_declaration_type(const NvmRecordArrayOrigins *p,uint32_t i,NvmOrdinaryArrayType *out) {
    return p && nvm_ownership_declarations_type(p->declarations,i,out);
}
bool nvm_record_array_declaration_binding(const NvmRecordArrayOrigins *p,uint32_t i,NvmOrdinaryArrayBinding *out) {
    return p && nvm_ownership_declarations_binding(p->declarations,i,out);
}
bool nvm_record_array_declaration_variant(const NvmRecordArrayOrigins *p,uint32_t i,uint16_t v,NvmUnionVariantFact *out) {
    return p && nvm_ownership_declarations_variant(p->declarations,i,v,out);
}
