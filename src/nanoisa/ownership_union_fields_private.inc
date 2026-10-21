#include "ownership_union_fields_private.h"
struct NvmUnionFieldDeclarations {
    NvmV2Layouts layouts;
    NvmUnionVariantFact variants[2];
    uint8_t flags[2];
    uint16_t record_layout, union_layout, payload_variant, empty_variant;
};
static NvmUnionFieldDeclarationResult ufd_result(NvmUnionFieldDeclarationStatus s,uint64_t b,uint64_t w) {
    const char *message="I require consistent union declaration bytes.";
    if(s==NVM_UFD_PREPARED)message="I copied union declarations without execution authority.";
    else if(s==NVM_UFD_UNKNOWN)message="I could not establish the supported union declarations.";
    else if(s==NVM_UFD_LIMIT)message="I reached the union declaration bound.";
    else if(s==NVM_UFD_MEMORY)message="I could not allocate union declarations.";
    return (NvmUnionFieldDeclarationResult){s,message,b,w};
}
void nvm_union_field_declarations_free(NvmUnionFieldDeclarations *p) {
    if(!p)return;
    nvm_v2_layouts_free(&p->layouts);free(p);
}
/* The public internal constructor supplies zero extras. Whitebox tests may
 * conservatively pre-reserve additional work/bytes; never credit/refund. */
static NvmUnionFieldDeclarationResult ufd_prepare_reserved(const NvmModule *m,
        NvmUnionFieldDeclarations **out,uint64_t extra_bytes,uint64_t extra_work) {
    uint64_t work=64,bytes=0;
    if(extra_work>OAA_STEPS-work || extra_bytes>OAA_BYTES)
        return ufd_result(NVM_UFD_LIMIT,bytes,work);
    work+=extra_work;bytes=extra_bytes;
    if(!m||!out || (!m->layout_data != !m->layout_size) ||
       (!m->ownership_data != !m->ownership_size))return ufd_result(NVM_UFD_INVALID,bytes,work);
    if(!m->layout_size || !m->ownership_size)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    if(m->layout_size>OAA_BYTES || m->ownership_size>OAA_BYTES || m->function_count>256)
        return ufd_result(NVM_UFD_LIMIT,bytes,work);
    if(m->function_count && !m->functions)return ufd_result(NVM_UFD_INVALID,bytes,work);
    uint64_t source=(uint64_t)m->layout_size+m->ownership_size;
    /* Prepay the complete framing extent before any cursor traversal. */
    if(source>OAA_STEPS-work)return ufd_result(NVM_UFD_LIMIT,bytes,work);
    work+=source;
    NvmV2Cursor c;nvm_v2_cursor_init(&c,m->layout_data,m->layout_size);
    uint32_t n;const uint8_t *row;
#define UFD_READ(expr) do { if((expr)!=NVM_V2_OK)return ufd_result(NVM_UFD_INVALID,bytes,work); } while(0)
    UFD_READ(nvm_v2_u32(&c,&n));
    if(n!=2 || m->struct_count!=1 || m->union_count!=1 || m->enum_count)
        return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    for(uint32_t i=0;i<2;i++) {
        UFD_READ(nvm_v2_take(&c,8,&row));
        uint16_t fields=(uint16_t)(row[2]|((uint16_t)row[3]<<8));
        if(fields!=1)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
        if(row[0]!=(i?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION))
            return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
        UFD_READ(nvm_v2_take(&c,12,&row));
        uint32_t ref=(uint32_t)row[4]|((uint32_t)row[5]<<8)|((uint32_t)row[6]<<16)|((uint32_t)row[7]<<24);
        if(row[0]!=(i?TAG_UNION:TAG_INT) || ref!=(i?0:NVM_V2_NO_INDEX))
            return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    }
    if(c.pos!=c.size)return ufd_result(NVM_UFD_INVALID,bytes,work);
    nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    UFD_READ(nvm_v2_u32(&c,&n));
    if(n!=3)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    UFD_READ(nvm_v2_u32(&c,&n));
    if(n!=2)return ufd_result(NVM_UFD_INVALID,bytes,work);
    UFD_READ(nvm_v2_take(&c,2,&row));
    uint8_t flags[2]={row[0],row[1]};
    if(flags[0]!=NVM_LAYOUT_COMPLETE || flags[1]!=NVM_LAYOUT_COMPLETE)
        return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    UFD_READ(nvm_v2_align4(&c));UFD_READ(nvm_v2_u32(&c,&n));
    if(n!=m->function_count)return ufd_result(NVM_UFD_INVALID,bytes,work);
    uint64_t descriptors=0;
    for(uint32_t i=0;i<n;i++) {
        uint16_t locals,params;
        UFD_READ(nvm_v2_u16(&c,&locals));UFD_READ(nvm_v2_u16(&c,&params));
        if(locals>256)return ufd_result(NVM_UFD_LIMIT,bytes,work);
        if(locals!=m->functions[i].local_count || params!=m->functions[i].arity || params>locals)
            return ufd_result(NVM_UFD_INVALID,bytes,work);
        descriptors+=(uint32_t)locals+1;
        for(uint32_t d=0;d<=(uint32_t)locals;d++) {
            UFD_READ(nvm_v2_take(&c,8,&row));
            if(row[1])return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
        }
    }
    UFD_READ(nvm_v2_u32(&c,&n));
    if(n!=4)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    UFD_READ(nvm_v2_u32(&c,&n));
    if(n)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    UFD_READ(nvm_v2_u32(&c,&n));
    if(n!=1)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    uint16_t kind,revision;
    UFD_READ(nvm_v2_u16(&c,&kind));UFD_READ(nvm_v2_u16(&c,&revision));
    if(kind!=1 || revision!=1)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    UFD_READ(nvm_v2_u32(&c,&n));
    if(n!=28)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    UFD_READ(nvm_v2_take(&c,28,&row));
    if(c.pos!=c.size)return ufd_result(NVM_UFD_INVALID,bytes,work);
    NvmV2Cursor u;nvm_v2_cursor_init(&u,row,28);
    uint32_t unions,layout;uint16_t variants,reserved;
    UFD_READ(nvm_v2_u32(&u,&unions));UFD_READ(nvm_v2_u32(&u,&layout));
    UFD_READ(nvm_v2_u16(&u,&variants));UFD_READ(nvm_v2_u16(&u,&reserved));
    if(unions!=1 || layout!=0 || variants!=2 || reserved)
        return ufd_result(NVM_UFD_INVALID,bytes,work);
#undef UFD_READ
    if(m->import_count || m->module_ref_count || m->service_size || m->service_data ||
       m->passive_size || m->passive_data)return ufd_result(NVM_UFD_UNKNOWN,bytes,work);
    uint64_t more=4*source+32*descriptors+128*2+64*2;
    if(more>OAA_STEPS-work)return ufd_result(NVM_UFD_LIMIT,bytes,work);
    work+=more;
    uint64_t payload=sizeof(NvmUnionFieldDeclarations)+2*sizeof(NvmV2Layout)+2*sizeof(NvmV2LayoutField);
    if(payload>OAA_BYTES-bytes)return ufd_result(NVM_UFD_LIMIT,bytes,work);
    bytes+=payload;
    NvmUnionFieldDeclarations *p=calloc(1,sizeof *p);
    if(!p)return ufd_result(NVM_UFD_MEMORY,bytes,work);
    bool oom=false;
    NvmV2Result decoded=nvm_ownership_mixed_layouts_private_decode_detailed(
        m->layout_data,m->layout_size,&p->layouts,&oom);
    NvmUnionFieldDeclarationStatus status=NVM_UFD_INVALID;
    if(decoded!=NVM_V2_OK){if(oom)status=NVM_UFD_MEMORY;goto fail;}
    /* Exact prior-only shape prevents decoder DFS allocations in this route. */
    if(p->layouts.count!=2 || p->layouts.items[0].kind!=NVM_V2_LAYOUT_UNION ||
       p->layouts.items[1].kind!=NVM_V2_LAYOUT_STRUCT)goto fail;
    for(uint32_t i=0;i<2;i++) {
        const NvmV2Layout *l=&p->layouts.items[i];
        if(l->field_count!=1 || l->name_idx>=m->string_count || l->fields[0].name_idx>=m->string_count)goto fail;
        const NvmV2LayoutField *f=&l->fields[0];
        if(f->type_tag!=(i?TAG_UNION:TAG_INT) || f->nested_idx!=(i?0:NVM_V2_NO_INDEX))goto fail;
    }
    OwnershipProjection projection={0};bool needs=false;
    if(ownership_validate_owned(m,&p->layouts,&needs,NULL,&projection,NULL)!=NVM_V2_OK ||
       projection.ambiguous || projection.count!=2 || !projection.view.data)goto fail;
    uint32_t starts[1]={0};uint16_t counts[1]={0};
    projection.rows=p->variants;projection.capacity=2;projection.starts=starts;projection.variants=counts;
    nvm_v2_cursor_init(&u,projection.view.data,projection.view.size);
    if(union_facts_read(&u,m,&p->layouts,flags,NVM_V2_NO_INDEX,0,NULL,&projection)!=NVM_V2_OK ||
       projection.ambiguous || projection.count!=2 || starts[0] || counts[0]!=2)goto fail;
    if(p->variants[0].field_count+p->variants[1].field_count!=1)goto fail;
    p->payload_variant=p->variants[0].field_count?0:1;
    p->empty_variant=(uint16_t)(1-p->payload_variant);
    p->record_layout=1;p->union_layout=0;p->flags[0]=flags[0];p->flags[1]=flags[1];
    *out=p;return ufd_result(NVM_UFD_PREPARED,bytes,work);
fail:
    nvm_union_field_declarations_free(p);return ufd_result(status,bytes,work);
}
NvmUnionFieldDeclarationResult nvm_union_field_declarations_prepare(const NvmModule *m,NvmUnionFieldDeclarations **out) {
    return ufd_prepare_reserved(m,out,0,0);
}
bool nvm_union_field_declarations_layout(const NvmUnionFieldDeclarations *p,uint32_t i,NvmDeclarationLayout *out) {
    if(!p||!out||i>=2)return false;
    const NvmV2Layout *l=&p->layouts.items[i];
    *out=(NvmDeclarationLayout){l->kind,p->flags[i],l->field_count,l->name_idx};return true;
}
bool nvm_union_field_declarations_field(const NvmUnionFieldDeclarations *p,uint32_t i,uint16_t f,NvmV2LayoutField *out) {
    if(!p||!out||i>=2||f)return false;
    *out=p->layouts.items[i].fields[0];return true;
}
bool nvm_union_field_declarations_variant(const NvmUnionFieldDeclarations *p,uint32_t ordinal,uint16_t v,NvmUnionVariantFact *out) {
    if(!p||!out||ordinal||v>=2)return false;
    *out=p->variants[v];return true;
}
bool nvm_union_field_declarations_nominal(const NvmUnionFieldDeclarations *p,uint8_t tag,uint32_t ordinal,uint32_t *out) {
    if(!p||!out||ordinal||(tag!=TAG_STRUCT&&tag!=TAG_UNION))return false;
    *out=tag==TAG_STRUCT?p->record_layout:p->union_layout;return true;
}
