/* I transport positive checked source events independently of executable proof. */
static const char borrow_place_key[]="nano.borrow-place.v1";
static bool borrow_place_utf8(const char *p,size_t n) {
    size_t i=0;
    while(i<n) {
        unsigned c=(unsigned char)p[i++],more=0;uint32_t value=0,min=0;
        if(c<128){if(!c)return false;continue;}
        if(c>=0xc2 && c<=0xdf){more=1;value=c&31;min=128;}
        else if(c>=0xe0 && c<=0xef){more=2;value=c&15;min=2048;}
        else if(c>=0xf0 && c<=0xf4){more=3;value=c&7;min=65536;}
        else return false;
        if(more>n-i)return false;
        while(more--){unsigned d=(unsigned char)p[i++];if((d&192)!=128)return false;value=(value<<6)|(d&63);}
        if(value<min || value>0x10ffff || (value>=0xd800 && value<=0xdfff))return false;
    }
    return true;
}
static bool borrow_place_entry(const NvmModule *mod,const NvmMetadataEntry *e) {
    return mod->string_lengths[e->key_idx]==sizeof borrow_place_key-1 &&
        !memcmp(mod->strings[e->key_idx],borrow_place_key,sizeof borrow_place_key-1);
}
static bool borrow_place_record(const NvmModule *mod,const NvmMetadataEntry *e,NvmBorrowPlace *out) {
    uint32_t size=mod->string_lengths[e->value_idx];if(size<40)return false;
    const uint8_t *p=(const uint8_t *)mod->strings[e->value_idx];
    uint32_t np=le_read_u32(p+28),nr=le_read_u32(p+32),nj=le_read_u32(p+36);
    if(!np || !nr || np>size-40 || nr>size-40-np || nj!=size-40-np-nr)return false;
    NvmBorrowPlace v={le_read_u32(p),le_read_u32(p+4),le_read_u32(p+8),le_read_u32(p+12),
        le_read_u32(p+16),le_read_u32(p+20),p[24],p[25],(const char *)p+40,
        (const char *)p+40+np,(const char *)p+40+np+nr,np,nr,nj};
    if(v.function>=mod->function_count || !v.call_line || !v.call_column || !v.argument_line ||
       !v.argument_column || v.mode<1 || v.mode>2 || v.coordinates<1 || v.coordinates>2 || p[26] || p[27] ||
       !borrow_place_utf8(v.source_path,np) || !borrow_place_utf8(v.root,nr) ||
       !borrow_place_utf8(v.projection,nj))return false;
    *out=v;return true;
}
static bool same_borrow_place(const NvmBorrowPlace *a,const NvmBorrowPlace *b) {
    return a->function==b->function && a->call_line==b->call_line && a->call_column==b->call_column &&
        a->argument_line==b->argument_line && a->argument_column==b->argument_column &&
        a->ordinal==b->ordinal && a->coordinates==b->coordinates &&
        a->source_path_length==b->source_path_length && !memcmp(a->source_path,b->source_path,a->source_path_length);
}
bool nvm_borrow_place_at(const NvmModule *mod,uint32_t index,NvmBorrowPlace *out) {
    if(!out || !nvm_metadata_valid(mod) || index>=mod->metadata_count)return false;
    NvmBorrowPlace selected;
    if(!borrow_place_entry(mod,&mod->metadata[index]) || !borrow_place_record(mod,&mod->metadata[index],&selected))return false;
    for(uint32_t i=0;i<mod->metadata_count;i++) {
        if(i==index || !borrow_place_entry(mod,&mod->metadata[i]))continue;
        NvmBorrowPlace other;
        if(!borrow_place_record(mod,&mod->metadata[i],&other) || same_borrow_place(&selected,&other))return false;
    }
    *out=selected;return true;
}
bool nvm_add_borrow_place(NvmModule *mod,const NvmBorrowPlace *v) {
    if(!v || !nvm_metadata_valid(mod) || v->function>=mod->function_count || !v->call_line ||
       !v->call_column || !v->argument_line || !v->argument_column || v->mode<1 || v->mode>2 ||
       v->coordinates<1 || v->coordinates>2 || !v->source_path || !v->root || !v->projection ||
       !v->source_path_length || !v->root_length)return false;
    uint32_t np=v->source_path_length,nr=v->root_length,nj=v->projection_length;
    if(np>UINT32_MAX-40 || nr>UINT32_MAX-40-np || nj>UINT32_MAX-40-np-nr ||
       !borrow_place_utf8(v->source_path,np) || !borrow_place_utf8(v->root,nr) ||
       !borrow_place_utf8(v->projection,nj))return false;
    for(uint32_t i=0;i<mod->metadata_count;i++) {
        if(!borrow_place_entry(mod,&mod->metadata[i]))continue;
        NvmBorrowPlace other;
        if(!borrow_place_record(mod,&mod->metadata[i],&other) || same_borrow_place(v,&other))return false;
    }
    uint32_t size=40+np+nr+nj;uint8_t *data=calloc(1,size);if(!data)return false;
    le_write_u32(data,v->function);le_write_u32(data+4,v->call_line);le_write_u32(data+8,v->call_column);
    le_write_u32(data+12,v->argument_line);le_write_u32(data+16,v->argument_column);le_write_u32(data+20,v->ordinal);
    data[24]=(uint8_t)v->mode;data[25]=(uint8_t)v->coordinates;
    le_write_u32(data+28,np);le_write_u32(data+32,nr);le_write_u32(data+36,nj);
    memcpy(data+40,v->source_path,np);memcpy(data+40+np,v->root,nr);memcpy(data+40+np+nr,v->projection,nj);
    uint32_t key=nvm_add_string(mod,borrow_place_key,sizeof borrow_place_key-1);
    uint32_t value=key==UINT32_MAX?UINT32_MAX:nvm_add_string(mod,(const char *)data,size);
    free(data);return value!=UINT32_MAX && nvm_add_metadata(mod,key,value);
}
