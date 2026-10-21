struct NvmRecordArrayOrigins {
    NvmOwnershipDeclarationPlan *declarations;
    NvmRecordEligibilityReport *origins;
    uint16_t required[ORIGINS];
    NvmRecordArrayOriginCounts counts;
};
static int publish_records(Analysis *,NvmRecordEligibilityReport **);
static int ra_prepare_records(Analysis *a,NvmRecordMapAnalysisContext *maps) {
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
    a->field_elements=analysis_alloc(a,fields?fields:1,sizeof *a->field_elements);
    if(!a->field_elements)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy field element constraints.");
    if(!analysis_work(a,(uint64_t)counts.bindings*4u+fields))return 0;
    for(uint32_t i=0;i<counts.bindings;i++) {
        NvmOrdinaryArrayBinding binding;NvmOrdinaryArrayType element;
        if(!nvm_ownership_declarations_binding(a->structure->declarations,i,&binding) ||
           !nvm_ownership_declarations_type(a->structure->declarations,binding.element_type,&element) ||
           binding.layout>=p->layouts.count || binding.field>=p->layouts.items[binding.layout].field_count)
            return stop(a,NVM_ARRAY_INVALID,0,0,"I require exact copied array bindings.");
        if(element.tag==TAG_ARRAY) {
            NvmOrdinaryArrayType child;
            if(!nvm_ownership_declarations_type(a->structure->declarations,element.referent,&child))
                return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require exact nested primitive element authority.");
            if(child.tag==TAG_ARRAY) {
                if(!analysis_work(a,1))return 0;
                NvmOrdinaryArrayType leaf;
                if(!nvm_ownership_declarations_type(a->structure->declarations,child.referent,&leaf) ||
                   leaf.tag<TAG_INT || leaf.tag>TAG_STRING || leaf.referent!=NVM_V2_NO_INDEX)
                    return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require exact depth-three primitive authority.");
                a->nested_arrays|=NESTED_PRESENT|NESTED_DEPTH3;
                if(leaf.tag==TAG_U8)a->nested_arrays|=NESTED_U8;
                if(leaf.tag==TAG_FLOAT)a->nested_arrays|=NESTED_FLOAT;
                if(leaf.tag==TAG_BOOL)a->nested_arrays|=NESTED_BOOL;
                if(leaf.tag==TAG_STRING)a->nested_arrays|=NESTED_DEPTH3_STRING;
                a->field_elements[a->layout_fields[binding.layout]+binding.field]=
                    (uint8_t)(FIELD_DEPTH3_FIRST+leaf.tag-TAG_INT);
                continue;
            }
            if(child.tag<TAG_INT || child.tag>TAG_STRING || child.referent!=NVM_V2_NO_INDEX)
                return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require exact nested primitive element authority.");
            a->nested_arrays|=NESTED_PRESENT;
            if(child.tag==TAG_U8 || child.tag==TAG_FLOAT || child.tag==TAG_BOOL) {
                uint32_t field=a->layout_fields[binding.layout]+binding.field;
                a->field_elements[field]=child.tag==TAG_U8?FIELD_NESTED_U8:
                    child.tag==TAG_FLOAT?FIELD_NESTED_FLOAT:FIELD_NESTED_BOOL;
                a->nested_arrays|=child.tag==TAG_U8?NESTED_U8:
                    child.tag==TAG_FLOAT?NESTED_FLOAT:NESTED_BOOL;
                continue;
            }
            if(child.tag==TAG_STRING) {
                if(!a->nested_string_fields) {
                    size_t bytes=(fields+7u)/8u;
                    a->nested_string_fields=analysis_alloc(a,bytes,1);
                    if(!a->nested_string_fields)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy nested STRING field constraints.");
                    if(!analysis_work(a,bytes))return 0;
                }
                uint32_t field=a->layout_fields[binding.layout]+binding.field;
                a->nested_string_fields[field/8]|=(uint8_t)(1u<<(field%8));
                continue;
            }
        } else if(element.referent!=NVM_V2_NO_INDEX || !(element.tag==TAG_INT || element.tag==TAG_U8 ||
             element.tag==TAG_FLOAT || element.tag==TAG_BOOL || element.tag==TAG_STRING))
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require exact bounded field element tags.");
        a->field_elements[a->layout_fields[binding.layout]+binding.field]=BIT(element.tag);
    }
    for(uint32_t i=0;i<p->layouts.count;i++) {
        NvmV2Layout *layout=&p->layouts.items[i];
        if(layout->kind!=NVM_V2_LAYOUT_STRUCT)continue;
        for(uint16_t j=0;j<layout->field_count;j++) {
            uint8_t tag=layout->fields[j].type_tag;
            if(tag==TAG_ARRAY) {
                if(!record_field_element(a,a->layout_fields[i]+j))
                    return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require every array field binding.");
            } else if(!(tag==TAG_INT || tag==TAG_U8 || tag==TAG_FLOAT || tag==TAG_BOOL || tag==TAG_STRING || tag==TAG_STRUCT || (maps && tag==TAG_HASHMAP)))
                return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require the bounded ordinary record field schema.");
        }
    }
    return 1;
}
/* Each bit names one exact (depth, leaf) shape, never a physical tag union. */
static uint16_t ra_child_domain(uint16_t domain) {
    uint16_t out=0;
    for(uint8_t depth=2;depth<=3;depth++)for(uint8_t leaf=TAG_INT;leaf<=TAG_STRING;leaf++)
        if(domain&nvm_record_array_requirement(depth,leaf))out|=nvm_record_array_requirement((uint8_t)(depth-1),leaf);
    return out;
}
static uint16_t ra_parent_domain(uint16_t parents,uint16_t children) {
    uint16_t out=0;
    for(uint8_t depth=2;depth<=3;depth++)for(uint8_t leaf=TAG_INT;leaf<=TAG_STRING;leaf++) {
        uint16_t shape=nvm_record_array_requirement(depth,leaf);
        if((parents&shape) && (children&nvm_record_array_requirement((uint8_t)(depth-1),leaf)))out|=shape;
    }
    return out;
}
static int ra_check_depth_three(Analysis *a) {
    int result=0;uint64_t converted_children=0;
    uint16_t *domains=analysis_alloc(a,ORIGINS,sizeof *domains);
    if(!domains)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate bounded depth-three shape domains.");
    if(!analysis_work(a,ORIGINS*16u))goto done;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        uint8_t declared=a->report.origins[i].declared_tag;
        Value content=a->children[i];merge(&content,a->all_writes[i]);
        if(content.unknown)goto invalid;
        if(declared==TAG_ARRAY) {
            if((content.tags&~BIT(TAG_ARRAY)) || (content.tags && !content.origins))goto invalid;
            for(uint8_t depth=2;depth<=3;depth++)for(uint8_t leaf=TAG_INT;leaf<=TAG_STRING;leaf++)
                domains[i]|=nvm_record_array_requirement(depth,leaf);
        } else {
            if(declared<TAG_INT || declared>TAG_STRING || content.origins ||
               (content.tags&~RECORD_ARRAY_LEAVES))goto invalid;
            domains[i]=BIT(declared);
        }
        if(a->required_elements[i]) {
            if(!nvm_record_array_requirement_depth(a->required_elements[i]))goto invalid;
            domains[i]&=a->required_elements[i];
        }
        if(!domains[i])goto invalid;
    }
    /* Every changing round removes a domain bit or installs one exact required
     * identity. The fixed bound includes all15 bits and64 origin identities. */
    for(uint32_t round=0;round<=ORIGINS*16u;round++) {
        bool changed=false;
        for(uint32_t i=0;i<a->report.origin_count;i++) {
            if(!analysis_work(a,1))goto done;
            if(a->origin_kind[i]!=NVM_HEAP_ORIGIN_ARRAY || a->report.origins[i].declared_tag!=TAG_ARRAY)continue;
            Value content=a->children[i];merge(&content,a->all_writes[i]);
            for(uint32_t j=0;j<a->report.origin_count;j++) {
                if(!analysis_work(a,1))goto done;
                if(!(content.origins&(UINT64_C(1)<<j)))continue;
                if(!analysis_work(a,32))goto done;
                if(a->origin_kind[j]!=NVM_HEAP_ORIGIN_ARRAY)goto invalid;
                uint16_t parent=ra_parent_domain(domains[i],domains[j]);
                uint16_t child=(uint16_t)(domains[j]&ra_child_domain(parent));
                if(!parent || !child)goto invalid;
                if(parent!=domains[i]){domains[i]=parent;changed=true;}
                if(child!=domains[j]){domains[j]=child;changed=true;}
                if(a->required_elements[i]) {
                    uint16_t required=ra_child_domain(a->required_elements[i]);
                    if(!nvm_record_array_requirement_depth(required) || !(domains[j]&required) ||
                       (a->required_elements[j] && a->required_elements[j]!=required))goto invalid;
                    if(!a->required_elements[j]){a->required_elements[j]=required;changed=true;}
                    if(domains[j]!=required){domains[j]=required;changed=true;}
                    if(a->required_elements[i]==NVM_RECORD_ARRAY_OF_U8)converted_children|=UINT64_C(1)<<j;
                }
            }
        }
        if(!changed)break;
        if(round==ORIGINS*16u){stop(a,NVM_ARRAY_LIMIT,0,0,"I reached my depth-three closure bound.");goto done;}
    }
    for(uint32_t i=0;i<a->report.origin_count;i++)if(a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        if(!analysis_work(a,16))goto done;
        uint8_t declared=a->report.origins[i].declared_tag;
        Value writes=a->all_writes[i],children=a->children[i];
        uint16_t allowed=declared==TAG_ARRAY?BIT(TAG_ARRAY):RECORD_ARRAY_LEAVES;
        if(writes.unknown || children.unknown || (writes.tags&~allowed) ||
           (children.tags&~allowed) || (a->report.origins[i].child_tags&~allowed))goto invalid;
        uint16_t required=a->required_elements[i];
        if(required) {
            uint8_t depth=nvm_record_array_requirement_depth(required);
            if(!depth)goto invalid;
            uint16_t expected=depth>1?BIT(TAG_ARRAY):required;
            uint16_t accepted=(uint16_t)(expected|((converted_children&(UINT64_C(1)<<i))?BIT(TAG_INT):0));
            if(expected!=BIT(declared) || (writes.tags&~accepted))goto invalid;
        }
    }
    result=1;goto done;
invalid:
    stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require acyclic exact depth-three primitive origins and every historical write.");
done:
    free(domains);return result;
}

/* I impose nested child obligations over the final monotone flow summaries.
 * Every past write remains in all_writes, including removed/replaced aliases. */
static int ra_check_constraints(Analysis *a) {
    if(a->nested_arrays&NESTED_DEPTH3)return ra_check_depth_three(a);
    uint64_t converted_children=0;
    if(!analysis_work(a,ORIGINS*16u))return 0;
    if(a->nested_arrays && !analysis_work(a,ORIGINS*ORIGINS*4u))return 0;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY &&
        a->report.origins[i].declared_tag==TAG_ARRAY) {
        Value content=a->children[i];merge(&content,a->all_writes[i]);
        if(content.unknown || (content.tags&~BIT(TAG_ARRAY)) ||
           (content.tags && !content.origins))
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require proved nested array children.");
        uint16_t requirement=a->required_elements[i];
        uint8_t leaf=nvm_record_array_requirement_leaf(requirement);
        for(uint32_t j=0;j<a->report.origin_count;j++)if(content.origins&(UINT64_C(1)<<j)) {
            uint8_t child=a->report.origins[j].declared_tag;
            if(a->origin_kind[j]!=NVM_HEAP_ORIGIN_ARRAY ||
               (child!=TAG_INT && !(a->nested_string_fields && child==TAG_STRING) &&
                !(child==TAG_U8 && (a->nested_arrays&NESTED_U8)) &&
                !(child==TAG_FLOAT && (a->nested_arrays&NESTED_FLOAT)) &&
                !(child==TAG_BOOL && (a->nested_arrays&NESTED_BOOL))) ||
               (leaf!=TAG_VOID && child!=leaf))
                return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require each nested child to preserve its exact primitive leaf.");
            if(requirement==NVM_RECORD_ARRAY_OF_U8)
                converted_children|=UINT64_C(1)<<j;
            leaf=child;a->required_elements[j]|=BIT(child);
        }
    }
    for(uint32_t i=0;i<a->report.origin_count;i++)if(a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        Value writes=a->all_writes[i],children=a->children[i];
        uint8_t declared=a->report.origins[i].declared_tag;
        uint16_t allowed=declared==TAG_ARRAY?BIT(TAG_ARRAY):RECORD_ARRAY_LEAVES;
        if(!(allowed&BIT(declared)) || writes.unknown || children.unknown ||
           (declared!=TAG_ARRAY && (writes.origins || children.origins)) ||
           (writes.tags&~allowed) || (children.tags&~allowed) ||
           (a->report.origins[i].child_tags&~allowed))
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require exact flat or bounded depth-two array content.");
        uint16_t expected=nvm_record_array_requirement_leaf(a->required_elements[i])!=TAG_VOID?BIT(TAG_ARRAY):a->required_elements[i];
        /* Exact packed children retain their leaf type after the existing INT conversion. */
        uint16_t accepted_writes=expected | ((converted_children&(UINT64_C(1)<<i))?BIT(TAG_INT):0);
        if(expected && (expected!=BIT(declared) || (writes.tags&~accepted_writes)))
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
