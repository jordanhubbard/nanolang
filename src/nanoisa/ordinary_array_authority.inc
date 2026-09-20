/* I publish only a complete owned non-admitting declaration report. */
static NvmOrdinaryArrayResult oaa_result(NvmOrdinaryArrayStatus status) {
    const char *message="I could not establish complete ordinary array declarations.";
    if(status==NVM_OAA_DESCRIBED)message="I copied ordinary array declarations without executable admission.";
    else if(status==NVM_OAA_INVALID)message="I require consistent complete declaration bytes.";
    else if(status==NVM_OAA_LIMIT)message="I reached my bounded declaration budget.";
    else if(status==NVM_OAA_MEMORY)message="I could not allocate my declaration report.";
    return (NvmOrdinaryArrayResult){status,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX,message};
}
void nvm_ordinary_array_authority_free(NvmOrdinaryArrayAuthority *p) {
    if(!p)return;
    nvm_v2_layouts_free(&p->layouts);free(p->types);free(p->bindings);free(p);
}
bool nvm_ordinary_array_authority_counts(const NvmOrdinaryArrayAuthority *p,NvmOrdinaryArrayCounts *out) {
    if(!p||!out)return false;
    *out=(NvmOrdinaryArrayCounts){p->layouts.count,p->type_count,p->binding_count};return true;
}
bool nvm_ordinary_array_authority_type(const NvmOrdinaryArrayAuthority *p,uint32_t i,NvmOrdinaryArrayType *out) {
    if(!p||!out||i>=p->type_count)return false;
    *out=p->types[i];return true;
}
bool nvm_ordinary_array_authority_binding(const NvmOrdinaryArrayAuthority *p,uint32_t i,NvmOrdinaryArrayBinding *out) {
    if(!p||!out||i>=p->binding_count)return false;
    *out=p->bindings[i];return true;
}
/* I identify explicit limits before invoking the legacy decoder. Truncated
 * structure is invalid here; its later conflated failure remains UNKNOWN. */
static NvmOrdinaryArrayStatus oaa_preflight(const NvmModule *m,uint32_t *work) {
    if(m->layout_size>OAA_BYTES || m->ownership_size>OAA_BYTES)return NVM_OAA_LIMIT;
    NvmV2Cursor c;nvm_v2_cursor_init(&c,m->layout_data,m->layout_size);
    uint32_t count,total=0;
    if(nvm_v2_u32(&c,&count)!=NVM_V2_OK)return NVM_OAA_INVALID;
    if(count>NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_LAYOUTS)return NVM_OAA_LIMIT;
    for(uint32_t i=0;i<count;i++) {
        const uint8_t *h,*fields;
        if(nvm_v2_take(&c,8,&h)!=NVM_V2_OK)return NVM_OAA_INVALID;
        uint16_t n=(uint16_t)((uint16_t)h[2]|((uint16_t)h[3]<<8));
        if(n>NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_FIELDS-total)return NVM_OAA_LIMIT;
        total+=n;
        if(nvm_v2_take(&c,(size_t)n*12,&fields)!=NVM_V2_OK)return NVM_OAA_INVALID;
    }
    if(c.pos!=c.size)return NVM_OAA_INVALID;
    if(m->function_count && !m->functions)return NVM_OAA_INVALID;
    /* I conservatively charge all borrowed-root/sibling walks per descriptor,
     * including the legacy scalar-root scan. Refusal may precede that work. */
    uint32_t used=total*4u+count*4u;
    uint32_t descriptor_cost=total*3u+count+33u;
    if(m->function_count>(OAA_STEPS-used)/descriptor_cost)return NVM_OAA_LIMIT;
    for(uint32_t i=0;i<m->function_count;i++) {
        uint32_t descriptors=(uint32_t)m->functions[i].local_count+1u;
        if(descriptors>(OAA_STEPS-used)/descriptor_cost)return NVM_OAA_LIMIT;
        uint32_t cost=descriptors*descriptor_cost;
        if(cost>OAA_STEPS-used)return NVM_OAA_LIMIT;
        used+=cost;
    }
    /* Existing union validation compares up to256 names pairwise per union.
     * Retained per-kind equality is checked before the common reader runs. */
    if(m->union_count>(OAA_STEPS-used)/65536u)return NVM_OAA_LIMIT;
    used+=m->union_count*65536u;*work=used;return NVM_OAA_DESCRIBED;
}
NvmOrdinaryArrayResult nvm_describe_ordinary_array_authority(
        const NvmModule *m,NvmOrdinaryArrayAuthority **out) {
    if(!m||!out||(!m->layout_data != !m->layout_size)||
       (!m->ownership_data != !m->ownership_size))return oaa_result(NVM_OAA_INVALID);
    if(!m->layout_size || !m->ownership_size)return oaa_result(NVM_OAA_UNKNOWN);
    uint32_t work=0;NvmOrdinaryArrayStatus status=oaa_preflight(m,&work);
    if(status!=NVM_OAA_DESCRIBED)return oaa_result(status);
    NvmOrdinaryArrayAuthority *p=calloc(1,sizeof *p);
    if(!p)return oaa_result(NVM_OAA_MEMORY);
    p->failure=NVM_OAA_INVALID;p->steps=work;
    NvmV2Result r=nvm_ownership_layouts_private_decode(m->layout_data,m->layout_size,&p->layouts);
    if(r!=NVM_V2_OK){status=r==NVM_V2_ERR_TRUNCATED?NVM_OAA_UNKNOWN:NVM_OAA_INVALID;goto fail;}
    uint32_t records=0,enums=0,unions=0;
    for(uint32_t i=0;i<p->layouts.count;i++) {
        const NvmV2Layout *row=&p->layouts.items[i];
        if(row->name_idx!=NVM_V2_NO_INDEX && row->name_idx>=m->string_count){status=NVM_OAA_INVALID;goto fail;}
        records+=row->kind==NVM_V2_LAYOUT_STRUCT;enums+=row->kind==NVM_V2_LAYOUT_ENUM;unions+=row->kind==NVM_V2_LAYOUT_UNION;
        if(row->kind!=NVM_V2_LAYOUT_STRUCT)p->unknown=true;
        for(uint16_t f=0;f<row->field_count;f++) {
            const NvmV2LayoutField *field=&row->fields[f];uint32_t ref=field->nested_idx;
            if(field->name_idx!=NVM_V2_NO_INDEX && field->name_idx>=m->string_count){status=NVM_OAA_INVALID;goto fail;}
            /* The prior-only legacy decoder permits tag/referent combinations
             * that this exact ordinary query must check independently. */
            if(field->type_tag==TAG_STRUCT) {
                if(ref>=p->layouts.count || p->layouts.items[ref].kind!=NVM_V2_LAYOUT_STRUCT){status=NVM_OAA_INVALID;goto fail;}
            } else if(field->type_tag==TAG_ENUM && ref!=NVM_V2_NO_INDEX) {
                if(ref>=p->layouts.count || p->layouts.items[ref].kind!=NVM_V2_LAYOUT_ENUM){status=NVM_OAA_INVALID;goto fail;}
            } else if(ref!=NVM_V2_NO_INDEX){status=NVM_OAA_INVALID;goto fail;}
            if(!scalar(field->type_tag) && field->type_tag!=TAG_STRING &&
               field->type_tag!=TAG_STRUCT && field->type_tag!=TAG_ARRAY)p->unknown=true;
        }
    }
    if(records!=m->struct_count || enums!=m->enum_count || unions!=m->union_count){status=NVM_OAA_INVALID;goto fail;}
    bool needs=false;
    r=ownership_validate_owned(m,&p->layouts,&needs,p);
    if(r!=NVM_V2_OK){status=p->failure;goto fail;}
    if(needs || m->import_count || m->module_ref_count || m->service_size || m->service_data ||
       m->passive_size || m->passive_data)p->unknown=true;
    uint32_t arrays=0;
    for(uint32_t i=0;i<p->layouts.count;i++) {
        if(p->flags[i]!=NVM_LAYOUT_COMPLETE)p->unknown=true;
        for(uint16_t f=0;f<p->layouts.items[i].field_count;f++)
            arrays+=p->layouts.items[i].fields[f].type_tag==TAG_ARRAY;
    }
    if(arrays && !p->binding_count)p->unknown=true;
    if(p->unknown){status=NVM_OAA_UNKNOWN;goto fail;}
    *out=p;return oaa_result(NVM_OAA_DESCRIBED);
fail:
    nvm_ordinary_array_authority_free(p);return oaa_result(status);
}
