/* I retain checked field observations without executable authority. */
static const char field_read_key[]="nano.field-read.v1";
static bool field_read_entry(const NvmModule *mod,const NvmMetadataEntry *e) {
    return mod->string_lengths[e->key_idx]==sizeof field_read_key-1 &&
        !memcmp(mod->strings[e->key_idx],field_read_key,sizeof field_read_key-1);
}
static bool field_read_record(const NvmModule *mod,const NvmMetadataEntry *e,NvmFieldRead *out) {
    uint32_t size=mod->string_lengths[e->value_idx];if(size<28)return false;
    const uint8_t *p=(const uint8_t *)mod->strings[e->value_idx];
    uint32_t np=le_read_u32(p+16),nr=le_read_u32(p+20),nj=le_read_u32(p+24);
    if(!np || !nr || np>size-28 || nr>size-28-np || nj!=size-28-np-nr)return false;
    NvmFieldRead v={le_read_u32(p),le_read_u32(p+4),le_read_u32(p+8),p[12],p[13],
        (const char *)p+28,(const char *)p+28+np,(const char *)p+28+np+nr,np,nr,nj};
    if(v.function>=mod->function_count || !v.line || !v.column || v.category<1 || v.category>3 ||
       v.coordinates<1 || v.coordinates>2 || p[14] || p[15] || !nj || v.projection[0]!='.' ||
       !borrow_place_utf8(v.source_path,np) || !borrow_place_utf8(v.root,nr) ||
       !borrow_place_utf8(v.projection,nj))return false;
    *out=v;return true;
}
static bool same_field_read(const NvmFieldRead *a,const NvmFieldRead *b) {
    return a->function==b->function && a->line==b->line && a->column==b->column &&
        a->coordinates==b->coordinates && a->source_path_length==b->source_path_length &&
        !memcmp(a->source_path,b->source_path,a->source_path_length);
}
bool nvm_field_read_at(const NvmModule *mod,uint32_t index,NvmFieldRead *out) {
    if(!out || !nvm_metadata_valid(mod) || index>=mod->metadata_count)return false;
    NvmFieldRead selected;
    if(!field_read_entry(mod,&mod->metadata[index]) || !field_read_record(mod,&mod->metadata[index],&selected))return false;
    for(uint32_t i=0;i<mod->metadata_count;i++) {
        if(i==index || !field_read_entry(mod,&mod->metadata[i]))continue;
        NvmFieldRead other;
        if(!field_read_record(mod,&mod->metadata[i],&other) || same_field_read(&selected,&other))return false;
    }
    *out=selected;return true;
}
bool nvm_add_field_read(NvmModule *mod,const NvmFieldRead *v) {
    if(!v || !nvm_metadata_valid(mod) || v->function>=mod->function_count || !v->line ||
       !v->column || v->category<1 || v->category>3 || v->coordinates<1 || v->coordinates>2 ||
       !v->source_path || !v->root || !v->projection || !v->source_path_length || !v->root_length ||
       !v->projection_length || v->projection[0]!='.')return false;
    uint32_t np=v->source_path_length,nr=v->root_length,nj=v->projection_length;
    if(np>UINT32_MAX-28 || nr>UINT32_MAX-28-np || nj>UINT32_MAX-28-np-nr ||
       !borrow_place_utf8(v->source_path,np) || !borrow_place_utf8(v->root,nr) ||
       !borrow_place_utf8(v->projection,nj))return false;
    for(uint32_t i=0;i<mod->metadata_count;i++) {
        if(!field_read_entry(mod,&mod->metadata[i]))continue;
        NvmFieldRead other;
        if(!field_read_record(mod,&mod->metadata[i],&other) || same_field_read(v,&other))return false;
    }
    uint32_t size=28+np+nr+nj;uint8_t *data=calloc(1,size);if(!data)return false;
    le_write_u32(data,v->function);le_write_u32(data+4,v->line);le_write_u32(data+8,v->column);
    data[12]=(uint8_t)v->category;data[13]=(uint8_t)v->coordinates;
    le_write_u32(data+16,np);le_write_u32(data+20,nr);le_write_u32(data+24,nj);
    memcpy(data+28,v->source_path,np);memcpy(data+28+np,v->root,nr);memcpy(data+28+np+nr,v->projection,nj);
    uint32_t key=nvm_add_string(mod,field_read_key,sizeof field_read_key-1);
    uint32_t value=key==UINT32_MAX?UINT32_MAX:nvm_add_string(mod,(const char *)data,size);
    free(data);return value!=UINT32_MAX && nvm_add_metadata(mod,key,value);
}
