/* I share one private declaration grammar with the checked ownership reader. */
#include "ordinary_array_authority.h"
#include "ownership_layouts_private.h"
#define OAA_TYPES 4096u
#define OAA_BINDINGS 65536u
#define OAA_STEPS 1048576u
#define OAA_BYTES (16u * 1024u * 1024u)
struct NvmOrdinaryArrayAuthority {
    NvmV2Layouts layouts;
    NvmOrdinaryArrayType *types;
    NvmOrdinaryArrayBinding *bindings;
    uint8_t flags[NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_LAYOUTS];
    uint32_t type_count, binding_count, steps;
    NvmOrdinaryArrayStatus failure;
    bool unknown;
};
static bool oaa_charge(NvmOrdinaryArrayAuthority *p,uint32_t n) {
    if(n>OAA_STEPS-p->steps){p->failure=NVM_OAA_LIMIT;return false;}
    p->steps+=n;return true;
}
static NvmV2Result oaa_array_fields(NvmOwnershipExtensionView view,
                                   NvmOrdinaryArrayAuthority *p) {
    NvmV2Cursor c;nvm_v2_cursor_init(&c,view.data,view.size);
    uint32_t types,bindings;NvmV2Result r;
    if((r=nvm_v2_u32(&c,&types))!=NVM_V2_OK)return r;
    if(types>OAA_TYPES){p->failure=NVM_OAA_LIMIT;return NVM_V2_ERR_INDEX_RANGE;}
    /* I scan exact extents before either table allocation. */
    const uint8_t *rows;
    if((r=nvm_v2_take(&c,(size_t)types*8,&rows))!=NVM_V2_OK ||
       (r=nvm_v2_u32(&c,&bindings))!=NVM_V2_OK)return r;
    if(bindings>OAA_BINDINGS){p->failure=NVM_OAA_LIMIT;return NVM_V2_ERR_INDEX_RANGE;}
    const uint8_t *binding_rows;
    if((r=nvm_v2_take(&c,(size_t)bindings*12,&binding_rows))!=NVM_V2_OK)return r;
    if(c.pos!=c.size)return NVM_V2_ERR_SECTION_RANGE;
    size_t bytes=sizeof *p+(size_t)p->layouts.count*sizeof(NvmV2Layout);
    for(uint32_t i=0;i<p->layouts.count;i++)bytes+=(size_t)p->layouts.items[i].field_count*sizeof(NvmV2LayoutField);
    /* Fixed ceilings make these products representable on wasm32 and native. */
    size_t extra=(size_t)types*sizeof *p->types+(size_t)bindings*sizeof *p->bindings;
    if(bytes>OAA_BYTES || extra>OAA_BYTES-bytes){p->failure=NVM_OAA_LIMIT;return NVM_V2_ERR_INDEX_RANGE;}
    p->type_count=types;p->binding_count=bindings;
    if(types)p->types=calloc(types,sizeof *p->types);
    if(bindings)p->bindings=calloc(bindings,sizeof *p->bindings);
    if((types&&!p->types)||(bindings&&!p->bindings)){p->failure=NVM_OAA_MEMORY;return NVM_V2_ERR_TRUNCATED;}
    NvmV2Cursor t;nvm_v2_cursor_init(&t,rows,(size_t)types*8);
    for(uint32_t i=0;i<types;i++) {
        if(!oaa_charge(p,1))return NVM_V2_ERR_INDEX_RANGE;
        uint8_t tag,z;uint16_t zz;uint32_t ref;
        if((r=nvm_v2_u8(&t,&tag))!=NVM_V2_OK || (r=nvm_v2_u8(&t,&z))!=NVM_V2_OK ||
           (r=nvm_v2_u16(&t,&zz))!=NVM_V2_OK || (r=nvm_v2_u32(&t,&ref))!=NVM_V2_OK)return r;
        if(z||zz)return NVM_V2_ERR_RESERVED_FLAGS;
        if(scalar(tag)||tag==TAG_STRING){if(ref!=NVM_V2_NO_INDEX)return NVM_V2_ERR_INDEX_RANGE;}
        else if(tag==TAG_ARRAY){if(ref>=types)return NVM_V2_ERR_INDEX_RANGE;p->unknown=true;}
        else if(tag==TAG_STRUCT){
            if(ref>=p->layouts.count || p->layouts.items[ref].kind!=NVM_V2_LAYOUT_STRUCT ||
               p->flags[ref]!=NVM_LAYOUT_COMPLETE)return NVM_V2_ERR_SECTION_TYPE;
            p->unknown=true;
        } else return NVM_V2_ERR_SECTION_TYPE;
        p->types[i]=(NvmOrdinaryArrayType){tag,ref};
    }
    /* Each ARRAY row has one edge. Iterative coloring visits it once; cached
     * depths prevent declaration order from hiding a chain longer than64. */
    uint8_t color[OAA_TYPES]={0},depths[OAA_TYPES]={0};uint32_t path[64];
    for(uint32_t root=0;root<types;root++) {
        uint32_t at=root,n=0;unsigned depth=0;
        while(!color[at]) {
            if(!oaa_charge(p,1))return NVM_V2_ERR_INDEX_RANGE;
            if(n==64){p->failure=NVM_OAA_LIMIT;return NVM_V2_ERR_INDEX_RANGE;}
            color[at]=1;path[n++]=at;
            if(p->types[at].tag!=TAG_ARRAY)break;
            at=p->types[at].referent;
            if(color[at]==1)return NVM_V2_ERR_SECTION_TYPE;
        }
        if(color[at]==2)depth=depths[at];
        while(n){uint32_t item=path[--n];if(++depth>64){p->failure=NVM_OAA_LIMIT;return NVM_V2_ERR_INDEX_RANGE;}color[item]=2;depths[item]=(uint8_t)depth;}
    }
    nvm_v2_cursor_init(&t,binding_rows,(size_t)bindings*12);
    for(uint32_t i=0;i<bindings;i++) {
        if(!oaa_charge(p,1))return NVM_V2_ERR_INDEX_RANGE;
        uint32_t layout,element;uint16_t field,z;
        if((r=nvm_v2_u32(&t,&layout))!=NVM_V2_OK || (r=nvm_v2_u16(&t,&field))!=NVM_V2_OK ||
           (r=nvm_v2_u16(&t,&z))!=NVM_V2_OK || (r=nvm_v2_u32(&t,&element))!=NVM_V2_OK)return r;
        if(z)return NVM_V2_ERR_RESERVED_FLAGS;
        if(layout>=p->layouts.count || element>=types || p->flags[layout]!=NVM_LAYOUT_COMPLETE ||
           p->layouts.items[layout].kind!=NVM_V2_LAYOUT_STRUCT || field>=p->layouts.items[layout].field_count ||
           p->layouts.items[layout].fields[field].type_tag!=TAG_ARRAY)return NVM_V2_ERR_SECTION_TYPE;
        if(i && (layout<p->bindings[i-1].layout || (layout==p->bindings[i-1].layout && field<=p->bindings[i-1].field)))return NVM_V2_ERR_SECTION_TYPE;
        p->bindings[i]=(NvmOrdinaryArrayBinding){layout,field,element};
    }
    uint32_t next=0;
    for(uint32_t i=0;i<p->layouts.count;i++)for(uint16_t f=0;f<p->layouts.items[i].field_count;f++) {
        if(!oaa_charge(p,1))return NVM_V2_ERR_INDEX_RANGE;
        if(p->flags[i]==NVM_LAYOUT_COMPLETE && p->layouts.items[i].fields[f].type_tag==TAG_ARRAY) {
            if(next>=bindings || p->bindings[next].layout!=i || p->bindings[next].field!=f)return NVM_V2_ERR_SECTION_TYPE;
            next++;
        }
    }
    return next==bindings?NVM_V2_OK:NVM_V2_ERR_SECTION_TYPE;
}
