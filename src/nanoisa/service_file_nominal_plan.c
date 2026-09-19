#include "service_file_nominal.h"
#include "nvm_v2_sections.h"
#include "ownership_contracts.h"
#include "isa.h"
#include "../nsi_file_catalog.h"
#include <stdlib.h>
#include <string.h>

struct NvmFileNominalPlan {
    NvmFileNominalBindings bindings;
    uint32_t count;
    NvmFileNominalLayout rows[];
};
static uint32_t catalog_at(const NvmFileNominalBindings *b,uint32_t global) {
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++)if(b->layouts[i]==global)return i;
    return NVM_V2_NO_INDEX;
}
static bool text(const NvmModule *m,uint32_t index,const char *wanted) {
    if(index>=m->string_count || !m->strings || !m->string_lengths || !m->strings[index])return false;
    size_t length=strlen(wanted);
    return m->string_lengths[index]==length && !memcmp(m->strings[index],wanted,length);
}
static bool name_index(const NvmModule *m,uint32_t index) {
    return index==NVM_V2_NO_INDEX || index<m->string_count;
}
static uint8_t type_kind(uint32_t ordinal) {
    return ordinal<3?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION;
}
static uint8_t type_tag(uint32_t ordinal) {
    return ordinal<3?TAG_STRUCT:TAG_UNION;
}
static uint8_t type_flags(uint32_t ordinal) {
    if(ordinal==NVM_V2_NO_INDEX)return 0;
    return (uint8_t)(NVM_LAYOUT_COMPLETE|((ordinal==0 || ordinal==3)?NVM_LAYOUT_RESOURCE:0));
}
static NvmFileCategory category(uint32_t ordinal) {
    if(ordinal==NVM_V2_NO_INDEX)return NVM_FILE_CATEGORY_UNKNOWN;
    if(ordinal==0)return NVM_FILE_CATEGORY_FILE;
    if(ordinal==3)return NVM_FILE_CATEGORY_OPEN_RESULT;
    return ordinal<3?NVM_FILE_CATEGORY_RECORD:NVM_FILE_CATEGORY_SCALAR_RESULT;
}
/* I resolve only the immutable catalog, never a same-shaped module type. */
static bool member_type(const NvmFileNominalBindings *b,const char *id,uint8_t *tag,uint32_t *nested) {
    *nested=NVM_V2_NO_INDEX;
    if(!id){*tag=TAG_VOID;return true;}
    if(!strcmp(id,"nsi:core/int")){*tag=TAG_INT;return true;}
    if(!strcmp(id,"nsi:core/bool")){*tag=TAG_BOOL;return true;}
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        const NlFilePlanType *t=nl_file_catalog_type(i);
        if(!t)return false;
        if(!strcmp(id,t->id)){*tag=type_tag(i);*nested=b->layouts[i];return true;}
    }
    return false;
}
static bool imports_valid(const NvmModule *m,const NvmFileNominalBindings *b) {
    if(m->import_count!=NVM_SERVICE_BINDING_COUNT || !m->imports ||
       m->module_ref_count || m->callback_contract_count)return false;
    for(uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        uint32_t index=b->imports[i];
        if(index>=m->import_count)return false;
        const NvmImportEntry *im=&m->imports[index];
        const NlFilePlanMethod *method=nl_file_catalog_method(i);
        if(!method || !method->param_count || method->param_count-1!=im->param_count ||
           im->kind!=NVM_IMPORT_SERVICE || im->return_type!=TAG_UNION ||
           !text(m,im->module_name_idx,nl_file_catalog_interface()) ||
           !text(m,im->function_name_idx,method->id))return false;
        const uint8_t *params=m->import_param_types?m->import_param_types[index]:NULL;
        if(im->param_count && !params)return false;
        for(uint16_t j=0;j<im->param_count;j++) {
            uint8_t tag;uint32_t nested;
            if(method->params[j].direction!=NL_NSI_DIR_IN ||
               !member_type(b,method->params[j].type_id,&tag,&nested) || params[j]!=tag)return false;
        }
        uint8_t result_tag;uint32_t result_layout;
        const NlFilePlanParam *result=&method->params[im->param_count];
        if(result->direction!=NL_NSI_DIR_RETURN ||
           !member_type(b,result->type_id,&result_tag,&result_layout) || result_tag!=TAG_UNION)
            return false;
    }
    return true;
}
/* Allocation-free exact prior-only codec preflight. I also check canonical
 * length, all names and actual per-kind counts before publishing any map. */
static NvmFileNominalStatus layouts_read(const NvmModule *m,const NvmFileNominalBindings *b,
                                        uint32_t *count_out,NvmFileNominalLayout *rows) {
    if(!m->layout_data || !m->layout_size)return NVM_FILE_NOMINAL_INVALID;
    NvmV2Cursor c;nvm_v2_cursor_init(&c,m->layout_data,m->layout_size);
    uint32_t count,records=0,enums=0,unions=0;
    if(nvm_v2_u32(&c,&count)!=NVM_V2_OK)return NVM_FILE_NOMINAL_INVALID;
    if(count>NVM_FILE_NOMINAL_MAX_LAYOUTS)return NVM_FILE_NOMINAL_LIMIT;
    if(count<NVM_FILE_NOMINAL_TYPES || count>(c.size-c.pos)/8)return NVM_FILE_NOMINAL_INVALID;
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++)if(b->layouts[i]>=count)return NVM_FILE_NOMINAL_INVALID;
    for(uint32_t i=0;i<count;i++) {
        uint8_t kind,pad;uint16_t fields;uint32_t name;
        if(nvm_v2_u8(&c,&kind)!=NVM_V2_OK || nvm_v2_u8(&c,&pad)!=NVM_V2_OK ||
           nvm_v2_u16(&c,&fields)!=NVM_V2_OK || nvm_v2_u32(&c,&name)!=NVM_V2_OK ||
           pad || kind>NVM_V2_LAYOUT_KIND_MAX || !name_index(m,name) || fields>(c.size-c.pos)/12)
            return NVM_FILE_NOMINAL_INVALID;
        uint32_t ordinal=catalog_at(b,i),source=NVM_V2_NO_INDEX;
        if(kind==NVM_V2_LAYOUT_STRUCT)source=records++;
        else if(kind==NVM_V2_LAYOUT_ENUM)enums++;
        else if(kind==NVM_V2_LAYOUT_UNION)source=unions++;
        const NlFilePlanType *type=ordinal==NVM_V2_NO_INDEX?NULL:nl_file_catalog_type(ordinal);
        if(ordinal!=NVM_V2_NO_INDEX && (!type || kind!=type_kind(ordinal) ||
           fields!=type->member_count || !text(m,name,type->id)))return NVM_FILE_NOMINAL_INVALID;
        for(uint32_t j=0;j<fields;j++) {
            uint8_t tag,p0,p1,p2;uint32_t nested,field_name;
            if(nvm_v2_u8(&c,&tag)!=NVM_V2_OK || nvm_v2_u8(&c,&p0)!=NVM_V2_OK ||
               nvm_v2_u8(&c,&p1)!=NVM_V2_OK || nvm_v2_u8(&c,&p2)!=NVM_V2_OK ||
               nvm_v2_u32(&c,&nested)!=NVM_V2_OK || nvm_v2_u32(&c,&field_name)!=NVM_V2_OK ||
               tag>=TAG_COUNT || p0 || p1 || p2 || !name_index(m,field_name) ||
               (nested!=NVM_V2_NO_INDEX && nested>=i))return NVM_FILE_NOMINAL_INVALID;
            if(type) {
                uint8_t expected_tag;uint32_t expected_nested;
                const NlFilePlanMember *member=&type->members[j];
                if(!member_type(b,member->type_id,&expected_tag,&expected_nested) ||
                   tag!=expected_tag || nested!=expected_nested || !text(m,field_name,member->id))
                    return NVM_FILE_NOMINAL_INVALID;
            }
        }
        if(rows)rows[i]=(NvmFileNominalLayout){i,ordinal,source,kind,type_flags(ordinal),category(ordinal)};
    }
    if(c.pos!=c.size || records!=m->struct_count || enums!=m->enum_count || unions!=m->union_count)
        return NVM_FILE_NOMINAL_INVALID;
    *count_out=count;return NVM_FILE_NOMINAL_DESCRIBED;
}
static bool descriptor_read(NvmV2Cursor *c,const NvmFileNominalBindings *b,
                             bool parameter,int signature_tag) {
    uint8_t tag,mode;uint16_t pad;uint32_t layout;
    if(nvm_v2_u8(c,&tag)!=NVM_V2_OK || nvm_v2_u8(c,&mode)!=NVM_V2_OK ||
       nvm_v2_u16(c,&pad)!=NVM_V2_OK || nvm_v2_u32(c,&layout)!=NVM_V2_OK ||
       pad || tag>=TAG_COUNT || mode>2 || (!parameter && mode) ||
       (signature_tag>=0 && tag!=signature_tag))return false;
    uint32_t ordinal=layout==NVM_V2_NO_INDEX?NVM_V2_NO_INDEX:catalog_at(b,layout);
    if(layout!=NVM_V2_NO_INDEX && (ordinal==NVM_V2_NO_INDEX || tag!=type_tag(ordinal)))return false;
    if(mode && (mode!=2 || ordinal!=0))return false;
    /* A bare STRUCT/UNION tag with NO_INDEX remains unresolved. No catalog
     * authority or initialized/live-owner claim is inferred from that tag. */
    return true;
}
static bool ownership_valid(const NvmModule *m,const NvmFileNominalBindings *b,uint32_t layouts) {
    if(!m->ownership_data || !m->ownership_size)return false;
    NvmV2Cursor c;nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    uint32_t version,count;const uint8_t *flags;
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || version!=NVM_OWNERSHIP_VERSION ||
       nvm_v2_u32(&c,&count)!=NVM_V2_OK || count!=layouts ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK)return false;
    for(uint32_t i=0;i<count;i++)if(flags[i]!=type_flags(catalog_at(b,i)))return false;
    if(nvm_v2_u32(&c,&count)!=NVM_V2_OK || count!=m->function_count ||
       (count && !m->functions) || count>(c.size-c.pos)/12)return false;
    for(uint32_t i=0;i<count;i++) {
        uint16_t locals,params;const NvmFunctionEntry *fn=&m->functions[i];
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK ||
           locals!=fn->local_count || params!=fn->arity || params>locals || fn->result_count>1 ||
           (size_t)locals+1>(c.size-c.pos)/8)return false;
        if(!descriptor_read(&c,b,false,fn->result_count?fn->result_tag:TAG_VOID))return false;
        const uint8_t *tags=m->function_param_types?m->function_param_types[i]:NULL;
        if(params && !tags)return false;
        for(uint32_t j=0;j<locals;j++)
            if(!descriptor_read(&c,b,j<params,j<params?tags[j]:-1))return false;
    }
    return c.pos==c.size;
}
static bool plan_extent(size_t count,size_t *bytes) {
    if(count>(SIZE_MAX-sizeof(NvmFileNominalPlan))/sizeof(NvmFileNominalLayout))return false;
    *bytes=sizeof(NvmFileNominalPlan)+count*sizeof(NvmFileNominalLayout);return true;
}
NvmFileNominalStatus nvm_file_nominal_plan(const NvmModule *m,NvmFileNominalPlan **out) {
    if(!m || !out)return NVM_FILE_NOMINAL_INVALID;
    NvmFileNominalBindings bindings;
    if(nvm_file_nominal_decode(m->service_data,m->service_size,&bindings)!=NVM_SERVICE_OK ||
       !imports_valid(m,&bindings))return NVM_FILE_NOMINAL_INVALID;
    uint32_t count=0;
    NvmFileNominalStatus result=layouts_read(m,&bindings,&count,NULL);
    if(result!=NVM_FILE_NOMINAL_DESCRIBED)return result;
    if(!ownership_valid(m,&bindings,count))return NVM_FILE_NOMINAL_INVALID;
    size_t bytes;
    if(!plan_extent(count,&bytes))return NVM_FILE_NOMINAL_LIMIT;
    NvmFileNominalPlan *plan=malloc(bytes);
    if(!plan)return NVM_FILE_NOMINAL_MEMORY;
    plan->bindings=bindings;plan->count=count;
    uint32_t filled=0;result=layouts_read(m,&bindings,&filled,plan->rows);
    if(result!=NVM_FILE_NOMINAL_DESCRIBED || filled!=count){free(plan);return NVM_FILE_NOMINAL_INVALID;}
    *out=plan;return NVM_FILE_NOMINAL_DESCRIBED;
}
void nvm_file_nominal_plan_free(NvmFileNominalPlan *p) { free(p); }
uint32_t nvm_file_nominal_layout_count(const NvmFileNominalPlan *p) { return p?p->count:0; }
bool nvm_file_nominal_layout(const NvmFileNominalPlan *p,uint32_t i,NvmFileNominalLayout *out) {
    if(!p || !out || i>=p->count)return false;
    *out=p->rows[i];return true;
}
bool nvm_file_nominal_type(const NvmFileNominalPlan *p,uint32_t i,NvmFileNominalLayout *out) {
    return p && i<NVM_FILE_NOMINAL_TYPES && nvm_file_nominal_layout(p,p->bindings.layouts[i],out);
}
bool nvm_file_nominal_source(const NvmFileNominalPlan *p,uint8_t kind,uint32_t ordinal,
                              NvmFileNominalLayout *out) {
    if(!p || !out || (kind!=NVM_V2_LAYOUT_STRUCT && kind!=NVM_V2_LAYOUT_UNION))return false;
    for(uint32_t i=0;i<p->count;i++)if(p->rows[i].layout_kind==kind && p->rows[i].source_ordinal==ordinal) {
        *out=p->rows[i];return true;
    }
    return false;
}
bool nvm_file_nominal_import(const NvmFileNominalPlan *p,uint32_t ordinal,uint32_t *out) {
    if(!p || !out || ordinal>=NVM_SERVICE_BINDING_COUNT)return false;
    *out=p->bindings.imports[ordinal];return true;
}
