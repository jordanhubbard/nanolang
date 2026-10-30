/* I keep all added storage in an opt-in declaration sidecar. */
#include "ownership_declaration_projection.h"
#ifndef NVM_MAP_DECL_BYTES
#define NVM_MAP_DECL_BYTES (2u*1024u*1024u)
#endif
#ifndef NVM_MAP_DECL_WORK
#define NVM_MAP_DECL_WORK (4u*1024u*1024u)
#endif
struct NvmMapDeclarationPlan {
    NvmOwnershipDeclarationPlan *base;
    uint8_t *scalars;
    NvmMapDeclarationPair *pairs;
    NvmMapDeclarationBinding *bindings;
    uint32_t *starts;
    NvmMapDeclarationCounts counts;
    uint32_t layouts;
    uint64_t bytes,work;
    NvmMapDeclarationStatus failure;
};
typedef struct {
    uint8_t scalar_used[5],pair_used[25];
    NvmV2Cursor extent,rows;
    NvmMapDeclarationPair previous_pair;
    NvmMapDeclarationBinding previous_binding;
    uint32_t scalars,pairs,bindings,index,layout;
    uint8_t previous_tag;
} MapDeclarationScratch;
static bool map_charge(NvmMapDeclarationPlan *p,uint64_t bytes,uint64_t work) {
    if(p->bytes>NVM_MAP_DECL_BYTES || bytes>NVM_MAP_DECL_BYTES-p->bytes ||
       p->work>NVM_MAP_DECL_WORK || work>NVM_MAP_DECL_WORK-p->work) {
        p->failure=NVM_MAP_LIMIT;return false;
    }
    p->bytes+=bytes;p->work+=work;return true;
}
static void *map_alloc(NvmMapDeclarationPlan *p,size_t n,size_t width) {
    if(width && n>SIZE_MAX/width){p->failure=NVM_MAP_LIMIT;return NULL;}
    if(!map_charge(p,(uint64_t)n*width,0))return NULL;
    void *v=calloc(n,width);if(!v)p->failure=NVM_MAP_MEMORY;return v;
}
static bool map_payload(NvmMapDeclarationPlan *p,NvmOwnershipExtensionView view,
                        const NvmV2Layouts *layouts,const uint8_t *flags) {
    if(!map_charge(p,sizeof(MapDeclarationScratch),(uint64_t)view.size*2u))return false;
    MapDeclarationScratch s={0};const uint8_t *scalar_rows,*pair_rows,*binding_rows;
    nvm_v2_cursor_init(&s.extent,view.data,view.size);
    if(view.revision!=1 || nvm_v2_u32(&s.extent,&s.scalars)!=NVM_V2_OK)return false;
    if(!s.scalars || s.scalars>5){p->failure=s.scalars?NVM_MAP_LIMIT:NVM_MAP_INVALID;return false;}
    if(nvm_v2_take(&s.extent,(size_t)s.scalars*8,&scalar_rows)!=NVM_V2_OK ||
       nvm_v2_u32(&s.extent,&s.pairs)!=NVM_V2_OK)return false;
    if(!s.pairs || s.pairs>25){p->failure=s.pairs?NVM_MAP_LIMIT:NVM_MAP_INVALID;return false;}
    if(nvm_v2_take(&s.extent,(size_t)s.pairs*8,&pair_rows)!=NVM_V2_OK ||
       nvm_v2_u32(&s.extent,&s.bindings)!=NVM_V2_OK)return false;
    if(!s.bindings || s.bindings>65536){p->failure=s.bindings?NVM_MAP_LIMIT:NVM_MAP_INVALID;return false;}
    if(nvm_v2_take(&s.extent,(size_t)s.bindings*12,&binding_rows)!=NVM_V2_OK ||
       s.extent.pos!=s.extent.size || layouts->count>256)return false;
    p->counts=(NvmMapDeclarationCounts){s.scalars,s.pairs,s.bindings};p->layouts=layouts->count;
    p->scalars=map_alloc(p,s.scalars,sizeof *p->scalars);
    if(!p->scalars)return false;
    p->pairs=map_alloc(p,s.pairs,sizeof *p->pairs);
    if(!p->pairs)return false;
    p->bindings=map_alloc(p,s.bindings,sizeof *p->bindings);
    if(!p->bindings)return false;
    p->starts=map_alloc(p,257,sizeof *p->starts);
    if(!p->scalars || !p->pairs || !p->bindings || !p->starts)return false;
    nvm_v2_cursor_init(&s.rows,scalar_rows,(size_t)s.scalars*8);
    for(s.index=0;s.index<s.scalars;s.index++) {
        uint8_t tag,z;uint16_t zz;uint32_t ref;
        if(!map_charge(p,0,1))return false;
        if(nvm_v2_u8(&s.rows,&tag)!=NVM_V2_OK || nvm_v2_u8(&s.rows,&z)!=NVM_V2_OK ||
           nvm_v2_u16(&s.rows,&zz)!=NVM_V2_OK || nvm_v2_u32(&s.rows,&ref)!=NVM_V2_OK ||
           z || zz || ref!=NVM_V2_NO_INDEX || (!scalar(tag)&&tag!=TAG_STRING) ||
           (s.index && tag<=s.previous_tag))return false;
        p->scalars[s.index]=tag;s.previous_tag=tag;
    }
    nvm_v2_cursor_init(&s.rows,pair_rows,(size_t)s.pairs*8);
    for(s.index=0;s.index<s.pairs;s.index++) {
        NvmMapDeclarationPair row;
        if(!map_charge(p,0,4))return false;
        if(nvm_v2_u32(&s.rows,&row.key_type)!=NVM_V2_OK || nvm_v2_u32(&s.rows,&row.value_type)!=NVM_V2_OK ||
           row.key_type>=s.scalars || row.value_type>=s.scalars || (s.index &&
           (row.key_type<s.previous_pair.key_type || (row.key_type==s.previous_pair.key_type &&
            row.value_type<=s.previous_pair.value_type))))return false;
        p->pairs[s.index]=row;s.previous_pair=row;
        s.scalar_used[row.key_type]=1;s.scalar_used[row.value_type]=1;
    }
    nvm_v2_cursor_init(&s.rows,binding_rows,(size_t)s.bindings*12);
    for(s.index=0;s.index<s.bindings;s.index++) {
        NvmMapDeclarationBinding row;uint16_t reserved;
        if(!map_charge(p,0,5))return false;
        if(nvm_v2_u32(&s.rows,&row.layout)!=NVM_V2_OK || nvm_v2_u16(&s.rows,&row.field)!=NVM_V2_OK ||
           nvm_v2_u16(&s.rows,&reserved)!=NVM_V2_OK || nvm_v2_u32(&s.rows,&row.pair)!=NVM_V2_OK ||
           reserved || row.layout>=layouts->count || row.pair>=s.pairs ||
           row.field>=layouts->items[row.layout].field_count || (s.index &&
           (row.layout<s.previous_binding.layout || (row.layout==s.previous_binding.layout &&
            row.field<=s.previous_binding.field))))return false;
        const NvmV2Layout *l=&layouts->items[row.layout];
        if(l->kind!=NVM_V2_LAYOUT_STRUCT || flags[row.layout]!=NVM_LAYOUT_COMPLETE ||
           l->fields[row.field].type_tag!=TAG_HASHMAP || l->fields[row.field].nested_idx!=NVM_V2_NO_INDEX)return false;
        p->bindings[s.index]=row;s.previous_binding=row;s.pair_used[row.pair]=1;
    }
    s.index=0;
    for(s.layout=0;s.layout<layouts->count;s.layout++) {
        const NvmV2Layout *l=&layouts->items[s.layout];
        if(!map_charge(p,0,1))return false;
        p->starts[s.layout]=s.index;
        for(uint16_t f=0;f<l->field_count;f++) {
            if(!map_charge(p,0,2))return false;
            if(l->fields[f].type_tag!=TAG_HASHMAP)continue;
            if(s.index>=s.bindings || p->bindings[s.index].layout!=s.layout ||
               p->bindings[s.index].field!=f)return false;
            s.index++;
        }
    }
    p->starts[layouts->count]=s.index;
    if(s.index!=s.bindings)return false;
    for(uint32_t i=0;i<s.scalars;i++){if(!map_charge(p,0,1)||!s.scalar_used[i])return false;}
    for(uint32_t i=0;i<s.pairs;i++){if(!map_charge(p,0,1)||!s.pair_used[i])return false;}
    return true;
}
