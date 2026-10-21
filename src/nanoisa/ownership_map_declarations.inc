/* I separate framing classification from copied declaration validation. */
typedef enum { MAP_ROUTE_ABSENT, MAP_ROUTE_PRESENT, MAP_ROUTE_INVALID, MAP_ROUTE_LIMIT } MapRoute;
static MapRoute map_route(const NvmModule *m,uint64_t *work) {
    if(!m || (!m->layout_data != !m->layout_size) || (!m->ownership_data != !m->ownership_size))return MAP_ROUTE_INVALID;
    if(!m->layout_size && !m->ownership_size)return MAP_ROUTE_ABSENT;
    if(!m->layout_size || !m->ownership_size)return MAP_ROUTE_INVALID;
    if(m->layout_size>OAA_BYTES || m->ownership_size>OAA_BYTES)return MAP_ROUTE_LIMIT;
    *work=(uint64_t)m->layout_size+m->ownership_size;
    if(*work>NVM_MAP_DECL_WORK)return MAP_ROUTE_LIMIT;
    bool field_map=false;NvmV2Cursor c;uint32_t count,total=0;const uint8_t *data;
    nvm_v2_cursor_init(&c,m->layout_data,m->layout_size);
    if(nvm_v2_u32(&c,&count)!=NVM_V2_OK)return MAP_ROUTE_INVALID;
    if(count>256)return MAP_ROUTE_LIMIT;
    for(uint32_t i=0;i<count;i++) {
        if(nvm_v2_take(&c,8,&data)!=NVM_V2_OK)return MAP_ROUTE_INVALID;
        uint16_t fields=(uint16_t)((uint16_t)data[2]|((uint16_t)data[3]<<8));
        if(data[1] || data[0]>NVM_V2_LAYOUT_KIND_MAX)return MAP_ROUTE_INVALID;
        if(fields>65536u-total)return MAP_ROUTE_LIMIT;
        total+=fields;
        for(uint16_t f=0;f<fields;f++) {
            if(nvm_v2_take(&c,12,&data)!=NVM_V2_OK)return MAP_ROUTE_INVALID;
            if(data[0]>=TAG_COUNT || data[1] || data[2] || data[3])return MAP_ROUTE_INVALID;
            field_map|=data[0]==TAG_HASHMAP;
        }
    }
    if(c.pos!=c.size)return MAP_ROUTE_INVALID;
    uint32_t version,layouts,functions;
    nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || !ownership_version(version) ||
       nvm_v2_u32(&c,&layouts)!=NVM_V2_OK || layouts!=count ||
       nvm_v2_take(&c,layouts,&data)!=NVM_V2_OK)return MAP_ROUTE_INVALID;
    for(uint32_t i=0;i<layouts;i++)if(data[i]&~(NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE))return MAP_ROUTE_INVALID;
    if(nvm_v2_align4(&c)!=NVM_V2_OK || nvm_v2_u32(&c,&functions)!=NVM_V2_OK ||
       functions!=m->function_count || (functions && !m->functions))return MAP_ROUTE_INVALID;
    for(uint32_t i=0;i<functions;i++) {
        uint16_t locals,params;
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK ||
           locals!=m->functions[i].local_count || params!=m->functions[i].arity || params>locals ||
           nvm_v2_take(&c,((size_t)locals+1)*8,&data)!=NVM_V2_OK)return MAP_ROUTE_INVALID;
        for(uint32_t j=0;j<=(uint32_t)locals;j++)if(data[j*8]>=TAG_COUNT ||
            data[j*8+1]>NVM_REFERENCE_EXCLUSIVE || data[j*8+2] || data[j*8+3])return MAP_ROUTE_INVALID;
    }
    if(version==3) {
        NvmV2Cursor bound=c;uint32_t path_bytes,extensions;
        if(nvm_v2_u32(&bound,&path_bytes)!=NVM_V2_OK ||
           nvm_v2_take(&bound,path_bytes,NULL)!=NVM_V2_OK ||
           nvm_v2_u32(&bound,&extensions)!=NVM_V2_OK)return MAP_ROUTE_INVALID;
        if(extensions>NVM_OWNERSHIP_MAX_EXTENSIONS)return MAP_ROUTE_LIMIT;
        if(*work>NVM_MAP_DECL_WORK || 8u>NVM_MAP_DECL_WORK-*work)return MAP_ROUTE_LIMIT;
        *work+=8;
    }
    NvmOwnershipExtensionView map={0};NvmOwnershipExtensions other={0};
    NvmV2Result r=NVM_V2_OK;
    if(version==2)r=paths_read(&c,NVM_V2_NO_INDEX,NULL,0,NULL,true);
    else if(version==3)r=extension_suffix_read_map(&c,NVM_V2_NO_INDEX,NULL,0,NULL,&other,&map);
    if(r!=NVM_V2_OK || c.pos!=c.size)return MAP_ROUTE_INVALID;
    /* I conservatively precharged the entire framing byte span before scanning. */
    return field_map || map.data?MAP_ROUTE_PRESENT:MAP_ROUTE_ABSENT;
}
static NvmMapDeclarationResult map_result(NvmMapDeclarationStatus status) {
    const char *s="I require complete exact map declarations.";
    if(status==NVM_MAP_ABSENT)s="I found no map declaration route.";
    else if(status==NVM_MAP_DESCRIBED)s="I copied exact map declarations without execution authority.";
    else if(status==NVM_MAP_LIMIT)s="I reached my map declaration bound.";
    else if(status==NVM_MAP_MEMORY)s="I could not allocate map declarations.";
    return (NvmMapDeclarationResult){status,s};
}
void nvm_map_declarations_free(NvmMapDeclarationPlan *p) {
    if(!p)return;
    free(p->scalars);free(p->pairs);free(p->bindings);free(p->starts);
    nvm_ownership_declarations_free(p->base);free(p);
}
NvmMapDeclarationResult nvm_prepare_map_declarations(const NvmModule *m,NvmMapDeclarationPlan **out) {
    if(!out)return map_result(NVM_MAP_INVALID);
    uint64_t work=0;MapRoute route=map_route(m,&work);
    if(route!=MAP_ROUTE_PRESENT)return map_result(route==MAP_ROUTE_ABSENT?NVM_MAP_ABSENT:
                                                 route==MAP_ROUTE_LIMIT?NVM_MAP_LIMIT:NVM_MAP_INVALID);
    if(sizeof(NvmMapDeclarationPlan)>NVM_MAP_DECL_BYTES)return map_result(NVM_MAP_LIMIT);
    NvmMapDeclarationPlan *p=calloc(1,sizeof *p);
    if(!p)return map_result(NVM_MAP_MEMORY);
    p->bytes=sizeof *p;p->work=work;p->failure=NVM_MAP_INVALID;
    NvmDeclarationResult base=prepare_ownership_declarations_map(m,&p->base,p);
    if(base.status!=NVM_DECL_PREPARED) {
        NvmMapDeclarationStatus failure=p->failure;
        if(base.status==NVM_DECL_MEMORY)failure=NVM_MAP_MEMORY;
        else if(base.status==NVM_DECL_LIMIT)failure=NVM_MAP_LIMIT;
        nvm_map_declarations_free(p);return map_result(failure);
    }
    *out=p;return map_result(NVM_MAP_DESCRIBED);
}
bool nvm_map_declarations_counts(const NvmMapDeclarationPlan *p,NvmMapDeclarationCounts *out) {
    if(!p||!out)return false;
    *out=p->counts;return true;
}
bool nvm_map_declarations_scalar(const NvmMapDeclarationPlan *p,uint32_t index,uint8_t *out) {
    if(!p||!out||index>=p->counts.scalars)return false;
    *out=p->scalars[index];return true;
}
bool nvm_map_declarations_pair(const NvmMapDeclarationPlan *p,uint32_t index,NvmMapDeclarationPair *out) {
    if(!p||!out||index>=p->counts.pairs)return false;
    *out=p->pairs[index];return true;
}
bool nvm_map_declarations_binding(const NvmMapDeclarationPlan *p,uint32_t index,NvmMapDeclarationBinding *out) {
    if(!p||!out||index>=p->counts.bindings)return false;
    *out=p->bindings[index];return true;
}
bool nvm_map_declarations_field(const NvmMapDeclarationPlan *p,uint32_t layout,uint16_t field,NvmMapDeclarationPair *out) {
    if(!p||!out||layout>=p->layouts)return false;
    uint32_t low=p->starts[layout],high=p->starts[layout+1];
    while(low<high) {
        uint32_t mid=low+(high-low)/2;const NvmMapDeclarationBinding *b=&p->bindings[mid];
        if(b->field==field){*out=p->pairs[b->pair];return true;}
        if(b->field<field)low=mid+1;else high=mid;
    }
    return false;
}
