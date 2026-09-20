#include "shadow_admission_diagnostic.h"
#include "owned_array_admission.h"
SHADOW_QUERY(NvmOwnedArrayRoute, nvm_owned_array_route,
             (const NvmModule *m), (m), SHADOW_OWNER) {
    if(!m || nvm_service_execution_pending(m))return NVM_OWNER_ARRAY_NOT_SELECTED;
    if(!m->ownership_data && !m->ownership_size)return NVM_OWNER_ARRAY_NOT_SELECTED;
    if(!m->ownership_data || !m->ownership_size)
        return NVM_OWNER_ARRAY_INVALID;
    NvmV2Cursor c;uint32_t version,count,layouts;const uint8_t *flags;
    nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK ||
       (version!=NVM_OWNERSHIP_VERSION && version!=NVM_OWNERSHIP_PATH_VERSION) ||
       nvm_v2_u32(&c,&count)!=NVM_V2_OK || count>NVM_RECORD_PLAN_MAX_LAYOUTS ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK)return NVM_OWNER_ARRAY_INVALID;
    bool resource=false;
    for(uint32_t n=0;n<count;n++)resource|=(flags[n]&NVM_LAYOUT_RESOURCE)!=0;
    if(!resource)return NVM_OWNER_ARRAY_NOT_SELECTED;
    for(uint32_t n=0;n<count;n++)
        if((flags[n]&~(NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE)) ||
           ((flags[n]&NVM_LAYOUT_RESOURCE) && !(flags[n]&NVM_LAYOUT_COMPLETE)))
            return NVM_OWNER_ARRAY_INVALID;
    uint32_t functions;
    if(nvm_v2_align4(&c)!=NVM_V2_OK || nvm_v2_u32(&c,&functions)!=NVM_V2_OK ||
       functions!=m->function_count)return NVM_OWNER_ARRAY_INVALID;
    if(!m->layout_data || !m->layout_size)return NVM_OWNER_ARRAY_INVALID;
    nvm_v2_cursor_init(&c,m->layout_data,m->layout_size);
    if(nvm_v2_u32(&c,&layouts)!=NVM_V2_OK || layouts!=count)return NVM_OWNER_ARRAY_INVALID;
    uint32_t total=0;bool selected=false;
    for(uint32_t n=0;n<count;n++) {
        uint8_t kind,reserved;uint16_t fields;uint32_t name;const uint8_t *field;
        if(nvm_v2_u8(&c,&kind)!=NVM_V2_OK || nvm_v2_u8(&c,&reserved)!=NVM_V2_OK ||
           nvm_v2_u16(&c,&fields)!=NVM_V2_OK || nvm_v2_u32(&c,&name)!=NVM_V2_OK ||
           fields>NVM_RECORD_PLAN_MAX_FIELDS-total || reserved || kind>NVM_V2_LAYOUT_KIND_MAX)
            return NVM_OWNER_ARRAY_INVALID;
        total+=fields;
        for(uint16_t f=0;f<fields;f++) {
            if(nvm_v2_take(&c,12,&field)!=NVM_V2_OK || field[0]>=TAG_COUNT ||
               field[1] || field[2] || field[3])return NVM_OWNER_ARRAY_INVALID;
            if((flags[n]&NVM_LAYOUT_RESOURCE) && field[0]==TAG_ARRAY)selected=true;
        }
    }
    if(c.pos!=c.size)return NVM_OWNER_ARRAY_INVALID;
    return selected?NVM_OWNER_ARRAY_SELECTED:NVM_OWNER_ARRAY_NOT_SELECTED;
}
/* I complete only parameter tags in a bounded borrowed view. This cursor pass
 * grants no authority; the unchanged complete query validates all declarations,
 * paths, bodies and ownership before any plan can escape. */
static NvmOwnerAuthorityStatus owner_array_parameter_view(const NvmModule *m,
        NvmModule *view,uint8_t *rows[8],uint8_t tags[8][8]) {
    if(!m->function_count || !m->functions)return NVM_OWNER_AUTH_INVALID;
    if(m->function_count>8)return NVM_OWNER_AUTH_LIMIT;
    NvmV2Cursor c;uint32_t version,count;const uint8_t *flags;
    nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK ||
       (version!=NVM_OWNERSHIP_VERSION && version!=NVM_OWNERSHIP_PATH_VERSION) ||
       nvm_v2_u32(&c,&count)!=NVM_V2_OK || count>NVM_RECORD_PLAN_MAX_LAYOUTS ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK ||
       nvm_v2_u32(&c,&count)!=NVM_V2_OK || count!=m->function_count)
        return NVM_OWNER_AUTH_INVALID;
    for(uint32_t f=0;f<count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];uint16_t locals,params;
        if(fn->arity>8 || fn->local_count>256)return NVM_OWNER_AUTH_LIMIT;
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK ||
           locals!=fn->local_count || params!=fn->arity || params>locals)
            return NVM_OWNER_AUTH_INVALID;
        rows[f]=tags[f];
        for(uint32_t n=0;n<=(uint32_t)locals;n++) {
            const uint8_t *descriptor;
            if(nvm_v2_take(&c,8,&descriptor)!=NVM_V2_OK)return NVM_OWNER_AUTH_INVALID;
            if(!n || n>params)continue;
            uint8_t tag=descriptor[0];
            if(tag!=TAG_INT && tag!=TAG_BOOL && tag!=TAG_U8 &&
               tag!=TAG_STRING && tag!=TAG_STRUCT)return NVM_OWNER_AUTH_UNRESOLVED;
            if(m->function_param_types && m->function_param_types[f]) {
                uint8_t supplied=m->function_param_types[f][n-1];
                if(supplied!=TAG_VOID && supplied!=tag)return NVM_OWNER_AUTH_UNRESOLVED;
            }
            tags[f][n-1]=tag;
        }
    }
    *view=*m;view->function_param_types=rows;
    return NVM_OWNER_AUTH_PREPARED;
}
NvmOwnerAuthorityResult nvm_owned_array_admit(const NvmModule *m,NvmOwnedArrayPlan **out) {
    if(!out)return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_INVALID,0,0,"I require an owner ARRAY admission output."};
    if(nvm_service_execution_pending(m))
        return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_UNRESOLVED,0,0,"I refuse service contracts before owner ARRAY admission."};
    NvmOwnedArrayRoute route=nvm_owned_array_route(m);
    if(route!=NVM_OWNER_ARRAY_SELECTED)
        return (NvmOwnerAuthorityResult){route==NVM_OWNER_ARRAY_INVALID?NVM_OWNER_AUTH_INVALID:NVM_OWNER_AUTH_UNRESOLVED,0,0,"I require an exact owner ARRAY candidate."};
    uint8_t tags[8][8]={{0}};uint8_t *rows[8]={0};NvmModule view;
    NvmOwnerAuthorityStatus prepared=owner_array_parameter_view(m,&view,rows,tags);
    if(prepared!=NVM_OWNER_AUTH_PREPARED)
        return (NvmOwnerAuthorityResult){prepared,0,0,"I require bounded exact owner ARRAY parameter declarations."};
    /* The query destroys every analysis frame borrowing view before returning;
     * its plan owns copied descriptors, signatures and obligations. */
    NvmOwnedArrayPlan *p=NULL;NvmOwnerAuthorityResult r=nvm_prepare_owned_array_authority(&view,&p);
    if(r.status!=NVM_OWNER_AUTH_PREPARED)return r;
    NvmOwnerSignature entry;bool supported=nvm_owned_array_plan_signature(p,0,&entry) &&
        !entry.parameter_count && (entry.result.tag==TAG_INT || entry.result.tag==TAG_BOOL || entry.result.tag==TAG_U8) &&
        (m->header.flags&NVM_FLAG_HAS_MAIN) && m->header.entry_point==0;
    for(uint32_t f=0;f<m->function_count;f++) {
        const char *name=nvm_get_string(m,m->functions[f].name_idx);
        if(name && !strcmp(name,"__init__"))supported=false;
        NvmOwnerSignature signature;
        if(!nvm_owned_array_plan_signature(p,f,&signature)){supported=false;continue;}
        if(m->function_param_types && m->function_param_types[f])
            for(uint16_t n=0;n<signature.parameter_count;n++) {
                uint8_t declared=m->function_param_types[f][n];
                if(declared!=TAG_VOID && declared!=signature.parameters[n].tag)supported=false;
            }
    }
    if(!supported){nvm_owned_array_plan_free(p);return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_UNRESOLVED,0,0,"I require exact owner ARRAY parameter tags and a no-argument scalar entry without initializers."};}
    *out=p;return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_PREPARED,0,0,"I admit exact owner ARRAY runtime authority."};
}
static NvmVerifyResult verify_owned_arrays(const NvmModule *m,uint32_t function,uint16_t *depth) {
    NvmOwnedArrayPlan *p=NULL;NvmOwnerAuthorityResult r=nvm_owned_array_admit(m,&p);
    if(r.status!=NVM_OWNER_AUTH_PREPARED)return fail("I refuse owner ARRAY admission (status %u) at function %u offset %u: %s",(unsigned)r.status,r.function,r.pc,r.message);
    NvmOwnerSignature signature;bool valid=nvm_owned_array_plan_signature(p,function,&signature);
    nvm_owned_array_plan_free(p);
    if(!valid)return fail("I require an existing owner ARRAY function index");
    if(depth)*depth=signature.max_stack;
    return ok_result();
}
