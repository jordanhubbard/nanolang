/* I live inside verifier.c so my structural delegation cannot be requested by
 * a public caller. Only a freshly completed private composition reaches it. */
#include "mixed_samples_internal.h"
struct NvmMixedSamplesPlan {
    NvmMixedSamplesProof *proof;
    NvmMixedSignature functions[8];
    NvmMixedDeclaration locals[8][256];
    uint32_t function_count;
    uint8_t *layout_bytes;
    uint32_t layout_size;
};
static NvmMixedShapeResult mp_result(NvmMixedShapeStatus status,const char *message) {
    return (NvmMixedShapeResult){status,0,0,message};
}
void nvm_mixed_samples_plan_free(NvmMixedSamplesPlan *p) {
    if(p){nvm_mixed_samples_proof_free(p->proof);free(p->layout_bytes);free(p);}
}
static bool mp_declaration(NvmV2Cursor *c,const NvmMixedSamplesProof *proof,
                            NvmMixedDeclaration *out) {
    NvmMixedDeclaration d={0};uint16_t reserved;
    if(nvm_v2_u8(c,&d.tag)!=NVM_V2_OK || nvm_v2_u8(c,&d.mode)!=NVM_V2_OK ||
       nvm_v2_u16(c,&reserved)!=NVM_V2_OK || reserved || d.mode ||
       nvm_v2_u32(c,&d.global_layout)!=NVM_V2_OK)return false;
    const NvmMixedLayoutView *v=proof->shape->view;
    if(d.tag==TAG_STRUCT) {
        if(d.global_layout>=v->layouts.count)return false;
        if(v->classes[d.global_layout]==NVM_MIXED_RESOURCE)d.category=NVM_MIXED_VALUE_OWNER;
        else if(proof->global_to_managed[d.global_layout]!=NVM_V2_NO_INDEX)
            d.category=NVM_MIXED_VALUE_ORDINARY;
        /* An unused pending declaration remains explicitly unavailable. */
    } else if(d.global_layout==NVM_V2_NO_INDEX) {
        if(d.tag==TAG_ARRAY)d.category=NVM_MIXED_VALUE_ORDINARY;
        else if(d.tag==TAG_INT || d.tag==TAG_BOOL || d.tag==TAG_U8 ||
                d.tag==TAG_FLOAT || d.tag==TAG_VOID)d.category=NVM_MIXED_VALUE_SCALAR;
        else return false;
    } else return false;
    *out=d;return true;
}
static bool mp_value_signature(NvmMixedDeclaration d,bool result) {
    if(d.mode)return false;
    if(d.category==NVM_MIXED_VALUE_OWNER)return d.tag==TAG_STRUCT;
    return d.category==NVM_MIXED_VALUE_SCALAR && (d.tag==TAG_INT || d.tag==TAG_BOOL ||
           d.tag==TAG_U8 || (result && d.tag==TAG_VOID));
}
static bool mp_signatures(const NvmModule *module,NvmMixedSamplesPlan *p) {
    const NvmMixedLayoutView *v=p->proof->shape->view;
    NvmV2Cursor c;uint32_t version,count,functions;const uint8_t *flags;
    nvm_v2_cursor_init(&c,v->ownership,v->ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK ||
       (version!=NVM_OWNERSHIP_VERSION && version!=NVM_OWNERSHIP_PATH_VERSION) ||
       nvm_v2_u32(&c,&count)!=NVM_V2_OK || count!=v->layouts.count ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK ||
       nvm_v2_u32(&c,&functions)!=NVM_V2_OK || functions!=module->function_count)return false;
    p->function_count=functions;
    for(uint32_t f=0;f<functions;f++) {
        const NvmFunctionEntry *fn=&module->functions[f];NvmMixedSignature *sig=&p->functions[f];
        if(nvm_v2_u16(&c,&sig->local_count)!=NVM_V2_OK || sig->local_count>256 ||
           nvm_v2_u16(&c,&sig->parameter_count)!=NVM_V2_OK || sig->parameter_count>8 ||
           sig->parameter_count>sig->local_count || sig->parameter_count!=fn->arity ||
           sig->local_count!=fn->local_count || !mp_declaration(&c,p->proof,&sig->result) ||
           !mp_value_signature(sig->result,true) || sig->result.tag!=fn->result_tag ||
           fn->result_count!=(sig->result.tag==TAG_VOID?0:1))return false;
        sig->max_stack=p->proof->max_stack[f];
        if((uint32_t)sig->local_count+sig->max_stack>UINT16_MAX)return false;
        if(sig->result.category==NVM_MIXED_VALUE_OWNER)
            sig->result_fields=v->layouts.items[sig->result.global_layout].field_count;
        for(uint16_t n=0;n<sig->local_count;n++) {
            if(!mp_declaration(&c,p->proof,&p->locals[f][n]))return false;
            if(n<sig->parameter_count) {
                if(!mp_value_signature(p->locals[f][n],false))return false;
                sig->parameters[n]=p->locals[f][n];
                if(module->function_param_types && module->function_param_types[f] &&
                   module->function_param_types[f][n]!=sig->parameters[n].tag)return false;
            }
        }
    }
    /* The earlier descriptor reader validated trailing retained path transport.
     * I copy the original bytes; this signature query neither drops nor reorders it. */
    return true;
}
NvmMixedShapeResult nvm_mixed_samples_prepare(const NvmModule *module,NvmMixedSamplesPlan **out) {
    if(!out)return mp_result(NVM_MIXED_SHAPE_INVALID,"I require an output for private mixed preparation.");
    if(nvm_service_bindings_present(module))
        return mp_result(NVM_MIXED_SHAPE_UNRESOLVED,"I refuse service contracts before mixed preparation.");
    NvmMixedSamplesProof *proof=NULL;
    NvmMixedShapeResult r=nvm_analyze_mixed_samples(module,&proof);
    if(r.status!=NVM_MIXED_SHAPE_PROVED)return r;
    bool owner=false,array=false;
    for(uint32_t n=0;n<proof->shape->view->layouts.count;n++)
        owner|=proof->shape->view->classes[n]==NVM_MIXED_RESOURCE;
    for(uint32_t n=0;n<proof->shape->origin_count;n++)array|=proof->shape->origins[n].tag==TAG_ARRAY;
    /* This checked candidate is distinct from ordinary and qualified STRING
     * profiles. It grants no backend permission, including when it succeeds. */
    if(!owner || !array || !proof->managed_count) {
        nvm_mixed_samples_proof_free(proof);
        return mp_result(NVM_MIXED_SHAPE_UNRESOLVED,"I require an owner and proved ordinary FLOAT-array record graph.");
    }
    /* I delegate only ownership to my fresh complete composition, not retained
     * structure, function ranges, names, metadata or other shared checks. */
    NvmVerifyResult structure=verify_structure_checked(module,true,NULL,true);
    if(!structure.ok || !nvm_metadata_valid(module)) {
        nvm_mixed_samples_proof_free(proof);
        return mp_result(NVM_MIXED_SHAPE_INVALID,"I require complete common structure for private mixed preparation.");
    }
    NvmMixedSamplesPlan *p=calloc(1,sizeof *p);
    if(!p){nvm_mixed_samples_proof_free(proof);return mp_result(NVM_MIXED_SHAPE_MEMORY,"I could not allocate my private mixed plan.");}
    p->proof=proof;
    if(!mp_signatures(module,p)) {
        nvm_mixed_samples_plan_free(p);
        return mp_result(NVM_MIXED_SHAPE_INVALID,"I require exact checked mixed signature transport.");
    }
    p->layout_bytes=malloc(module->layout_size);
    if(!p->layout_bytes){nvm_mixed_samples_plan_free(p);return mp_result(NVM_MIXED_SHAPE_MEMORY,"I could not retain my mixed layout transport.");}
    memcpy(p->layout_bytes,module->layout_data,module->layout_size);p->layout_size=module->layout_size;
    *out=p;
    return mp_result(NVM_MIXED_SHAPE_PROVED,"I prepared mixed facts; runtime checks and admission remain unavailable.");
}
bool nvm_mixed_samples_signature(const NvmMixedSamplesPlan *p,uint32_t f,NvmMixedSignature *out) {
    if(!p || !out || f>=p->function_count)return false;
    *out=p->functions[f];return true;
}
bool nvm_mixed_samples_local(const NvmMixedSamplesPlan *p,uint32_t f,uint16_t local,NvmMixedDeclaration *out) {
    if(!p || !out || f>=p->function_count || local>=p->functions[f].local_count)return false;
    *out=p->locals[f][local];return true;
}
bool nvm_mixed_samples_record(const NvmMixedSamplesPlan *p,uint32_t source,NvmMixedRecordIdentity *out) {
    if(!p || !out)return false;
    const NvmMixedLayoutView *v=p->proof->shape->view;
    if(source>=v->record_count)return false;
    uint32_t global=v->source_to_global[source];
    if(global>=v->layouts.count)return false;
    uint32_t managed=p->proof->global_to_managed[global];
    if(managed==NVM_V2_NO_INDEX || managed>=p->proof->managed_count ||
       p->proof->managed_to_global[managed]!=global || v->classes[global]==NVM_MIXED_RESOURCE)return false;
    *out=(NvmMixedRecordIdentity){source,global,managed,v->layouts.items[global].field_count};return true;
}
bool nvm_mixed_samples_transport(const NvmMixedSamplesPlan *p,NvmMixedTransport *out) {
    if(!p || !out)return false;
    const NvmMixedLayoutView *v=p->proof->shape->view;
    *out=(NvmMixedTransport){p->layout_bytes,v->ownership,p->layout_size,v->ownership_size};return true;
}
const NvmMixedSamplesProof *nvm_mixed_samples_obligations(const NvmMixedSamplesPlan *p) {
    return p?p->proof:NULL;
}

/* I inspect only bounded transport bytes, without allocating or claiming their
 * validity. The full descriptor reader and independent analyses remain mandatory. */
bool nvm_mixed_samples_candidate(const NvmModule *module) {
    if(!module || nvm_service_bindings_present(module) ||
       !module->ownership_data || !module->layout_data)return false;
    NvmV2Cursor c;uint32_t version,count,layout_count;const uint8_t *flags;
    nvm_v2_cursor_init(&c,module->ownership_data,module->ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK ||
       (version!=NVM_OWNERSHIP_VERSION && version!=NVM_OWNERSHIP_PATH_VERSION) ||
       nvm_v2_u32(&c,&count)!=NVM_V2_OK || count>NVM_RECORD_PLAN_MAX_LAYOUTS ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK)return false;
    bool resource=false;for(uint32_t n=0;n<count;n++)resource|=(flags[n]&NVM_LAYOUT_RESOURCE)!=0;
    if(!resource)return false;
    nvm_v2_cursor_init(&c,module->layout_data,module->layout_size);
    if(nvm_v2_u32(&c,&layout_count)!=NVM_V2_OK || layout_count!=count)return false;
    uint32_t total=0;
    for(uint32_t n=0;n<count;n++) {
        uint8_t kind,reserved;uint16_t fields;uint32_t name;const uint8_t *bytes;
        if(nvm_v2_u8(&c,&kind)!=NVM_V2_OK || nvm_v2_u8(&c,&reserved)!=NVM_V2_OK ||
           nvm_v2_u16(&c,&fields)!=NVM_V2_OK || nvm_v2_u32(&c,&name)!=NVM_V2_OK ||
           fields>NVM_RECORD_PLAN_MAX_FIELDS-total)return false;
        total+=fields;
        for(uint16_t f=0;f<fields;f++) {
            if(nvm_v2_take(&c,12,&bytes)!=NVM_V2_OK)return false;
            if(bytes[0]==TAG_ARRAY)return true;
        }
    }
    return false;
}
NvmMixedShapeResult nvm_mixed_samples_admit(const NvmModule *module,NvmMixedSamplesPlan **out) {
    if(!out)return mp_result(NVM_MIXED_SHAPE_INVALID,"I require an output for mixed admission.");
    if(!nvm_mixed_samples_candidate(module))
        return mp_result(NVM_MIXED_SHAPE_UNRESOLVED,"I require the explicit mixed Samples candidate.");
    NvmMixedSamplesPlan *plan=NULL;
    NvmMixedShapeResult result=nvm_mixed_samples_prepare(module,&plan);
    if(result.status!=NVM_MIXED_SHAPE_PROVED)return result;
    uint8_t entry=plan->functions[0].result.tag;
    bool supported=entry==TAG_INT || entry==TAG_BOOL || entry==TAG_U8;
    for(uint32_t f=0;f<module->function_count;f++) {
        const char *name=nvm_get_string(module,module->functions[f].name_idx);
        if(name && !strcmp(name,"__init__"))supported=false;
    }
    if(!supported) {
        nvm_mixed_samples_plan_free(plan);
        return mp_result(NVM_MIXED_SHAPE_UNRESOLVED,"I require a scalar mixed entry without initializer functions.");
    }
    *out=plan;
    return mp_result(NVM_MIXED_SHAPE_PROVED,"I admit my closed mixed runtime with exact scalar checks and owner obligations.");
}
static NvmVerifyResult verify_mixed_samples(const NvmModule *module,uint32_t function,
                                           uint16_t *out_max_stack) {
    NvmMixedSamplesPlan *plan=NULL;
    NvmMixedShapeResult result=nvm_mixed_samples_admit(module,&plan);
    if(result.status!=NVM_MIXED_SHAPE_PROVED)
        return fail("I refuse mixed admission (status %u) at function %u offset %u: %s",
                    (unsigned)result.status,result.function,result.pc,result.message);
    if(function>=plan->function_count){nvm_mixed_samples_plan_free(plan);return fail("I require an existing mixed function index");}
    uint16_t max_stack=plan->functions[function].max_stack;
    nvm_mixed_samples_plan_free(plan);
    if(out_max_stack)*out_max_stack=max_stack;
    return ok_result();
}
