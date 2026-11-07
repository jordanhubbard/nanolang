#include "record_map_admission.h"
/* No allocation or public verification. Independently scan both spans so a
 * malformed layout cannot hide an explicitly framed kind5 (and conversely).
 * The old map declaration query retains its original schedule unchanged. */
static bool map_public_layout_scan(const NvmModule *m,bool *found,uint32_t *layouts) {
    NvmV2Cursor c;const uint8_t *h,*f;uint32_t count;
    if(!m->layout_data)return !m->layout_size;
    size_t size=m->layout_size>OAA_BYTES?OAA_BYTES:m->layout_size;
    nvm_v2_cursor_init(&c,m->layout_data,size);
    if(nvm_v2_u32(&c,&count)!=NVM_V2_OK)return false;
    *layouts=count;
    bool valid=true;
    for(uint32_t i=0;i<count;i++) {
        if(nvm_v2_take(&c,8,&h)!=NVM_V2_OK)return false;
        if(h[1] || h[0]>NVM_V2_LAYOUT_KIND_MAX)valid=false;
        uint16_t fields=(uint16_t)((uint16_t)h[2]|((uint16_t)h[3]<<8));
        for(uint16_t j=0;j<fields;j++) {
            if(nvm_v2_take(&c,12,&f)!=NVM_V2_OK)return false;
            if(f[0]==TAG_HASHMAP)*found=true;
            if(f[0]>=TAG_COUNT || f[1] || f[2] || f[3])valid=false;
        }
    }
    return valid && c.pos==c.size;
}
static bool map_public_ownership_scan(const NvmModule *m,bool *found,uint32_t layouts) {
    NvmV2Cursor c;uint32_t version,count,functions;const uint8_t *flags;
    if(!m->ownership_data)return !m->ownership_size;
    size_t size=m->ownership_size>OAA_BYTES?OAA_BYTES:m->ownership_size;
    nvm_v2_cursor_init(&c,m->ownership_data,size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || nvm_v2_u32(&c,&count)!=NVM_V2_OK ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK)return false;
    bool valid=ownership_version(version) && count==layouts;
    for(uint32_t i=0;i<count;i++)if(flags[i]&~(NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE))valid=false;
    if(nvm_v2_align4(&c)!=NVM_V2_OK || nvm_v2_u32(&c,&functions)!=NVM_V2_OK)return false;
    for(uint32_t i=0;i<functions;i++) {
        uint16_t locals,params;
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK ||
           nvm_v2_take(&c,((size_t)locals+1)*8,NULL)!=NVM_V2_OK)return false;
        if(params>locals)valid=false;
    }
    if(version==3) {
        /* Detect selected identity before full suffix validation, so invalid
         * revision/padding/later frames cannot turn PRESENT into absence. */
        NvmV2Cursor probe=c;uint32_t paths,extensions;
        if(nvm_v2_u32(&probe,&paths)!=NVM_V2_OK || nvm_v2_take(&probe,paths,NULL)!=NVM_V2_OK ||
           nvm_v2_u32(&probe,&extensions)!=NVM_V2_OK)return false;
        for(uint32_t i=0;i<extensions;i++) {
            uint16_t kind,revision;uint32_t bytes;
            if(nvm_v2_u16(&probe,&kind)!=NVM_V2_OK)return false;
            if(kind==NVM_OWNERSHIP_EXTENSION_MAP_FIELDS)*found=true;
            if(nvm_v2_u16(&probe,&revision)!=NVM_V2_OK || nvm_v2_u32(&probe,&bytes)!=NVM_V2_OK ||
               nvm_v2_take(&probe,bytes,NULL)!=NVM_V2_OK || nvm_v2_align4(&probe)!=NVM_V2_OK)return false;
        }
        NvmOwnershipExtensions other={0};NvmOwnershipExtensionView map={0};
        if(extension_suffix_read_map(&c,NVM_V2_NO_INDEX,NULL,0,NULL,&other,&map)!=NVM_V2_OK)return false;
    } else if(version==2 && paths_read(&c,NVM_V2_NO_INDEX,NULL,0,NULL,true)!=NVM_V2_OK)return false;
    return valid && c.pos==c.size;
}
NvmRecordMapRoute nvm_record_map_route(const NvmModule *m) {
    if(!m)return NVM_RECORD_MAP_DEFER;
    if(!m->layout_data && !m->layout_size && !m->ownership_data && !m->ownership_size)return NVM_RECORD_MAP_ABSENT;
    bool found=false;uint32_t layouts=0;
    bool layout_ok=map_public_layout_scan(m,&found,&layouts);
    bool ownership_ok=map_public_ownership_scan(m,&found,layouts);
    if(!found)return layout_ok && ownership_ok?NVM_RECORD_MAP_ABSENT:NVM_RECORD_MAP_DEFER;
    if(m->layout_size>OAA_BYTES || m->ownership_size>OAA_BYTES)return NVM_RECORD_MAP_LIMIT;
    if(!m->layout_data || !m->layout_size || !m->ownership_data || !m->ownership_size ||
       !layout_ok || !ownership_ok)return NVM_RECORD_MAP_INVALID;
    return NVM_RECORD_MAP_PRESENT;
}

bool nvm_record_map_wire_selected(const NvmV2Module *m) {
    if(!m)return false;
    NvmModule framed={0};framed.ownership_data=(uint8_t *)m->ownership_data;
    framed.ownership_size=m->ownership_size;
    bool found=false;
    (void)map_public_ownership_scan(&framed,&found,m->layouts.count);
    if(found)return true;
    if(!m->layouts.items)return false;
    uint64_t work=0;
    for(uint32_t i=0;i<m->layouts.count;i++) {
        if(++work>OAA_BYTES)return false;
        const NvmV2Layout *l=&m->layouts.items[i];
        if(!l->fields && l->field_count)return false;
        for(uint16_t j=0;j<l->field_count;j++) {
            if(++work>OAA_BYTES)return false;
            if(l->fields[j].type_tag==TAG_HASHMAP)return true;
        }
    }
    return false; /* Old full validation still required, not valid absence. */
}
