#include "service_bindings_module.h"
#include "service_classification_private.h"
#include "../nsi_file_catalog.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>

bool nvm_service_bindings_present(const NvmModule *m) {
    if (!m) return false;
    if (m->service_data || m->service_size) return true;
    if (m->imports) for (uint32_t i=0;i<m->import_count;i++)
        if (m->imports[i].kind==NVM_IMPORT_SERVICE) return true;
    return false;
}
bool nvm_v2_service_bindings_present(const NvmV2Module *m) {
    if (!m) return false;
    if (m->service_data || m->service_size ||
        (m->extra_features & NVM_V2_FEATURE_SERVICE_BINDINGS)) return true;
    if (m->imports.items) for (uint32_t i=0;i<m->imports.count;i++)
        if (m->imports.items[i].kind==NVM_V2_IMPORT_SERVICE) return true;
    return false;
}
bool nvm_v2_file_instructions_present(const NvmV2Module *m) {
    if(!m || !m->functions.items || !m->code || m->code_size>SIZE_MAX)return false;
    for(uint32_t i=0;i<m->functions.count;i++) {
        const NvmV2Function *f=&m->functions.items[i];
        if(f->code_offset>m->code_size || f->code_length>m->code_size-f->code_offset)continue;
        if(isa_code_has_file_instructions(m->code+(size_t)f->code_offset,(size_t)f->code_length))return true;
    }
    return false;
}
bool nvm_service_execution_pending(const NvmModule *m) {
    return nvm_service_bindings_present(m) || nvm_file_instructions_present(m);
}
NvmServiceClassification nvm_service_classify(const NvmModule *module) {
    return (NvmServiceClassification){module, nvm_service_execution_pending(module)};
}
bool nvm_service_pending_classified(const NvmModule *module,
                                    const NvmServiceClassification *facts) {
    return facts && facts->module == module ? facts->pending
                                           : nvm_service_execution_pending(module);
}
static bool exact_bytes(const uint8_t *bytes,uint32_t length,const char *text) {
    size_t n=strlen(text);
    return bytes && length==n && memcmp(bytes,text,n)==0;
}
/* I derive positional wire categories from the one immutable catalog. The
 * nominal File/Result identities and ownership outcomes stay in that catalog. */
static bool category_signature(const NlFilePlanMethod *method,
                               const uint8_t *params,uint16_t count,
                               const uint8_t *results,uint16_t result_count) {
    if (!method || !method->param_count || method->param_count-1!=count ||
        result_count!=1 || !results || results[0]!=TAG_UNION || (count && !params)) return false;
    for (uint16_t i=0;i<count;i++) {
        const NlFilePlanParam *p=&method->params[i];
        uint8_t tag;
        if (!strcmp(p->type_id,"nsi:core/int")) tag=TAG_INT;
        else if (!strcmp(p->type_id,"nsi:nanolang/filesystem#File")) tag=TAG_STRUCT;
        else return false;
        if (p->direction!=NL_NSI_DIR_IN || params[i]!=tag) return false;
    }
    return method->params[count].direction==NL_NSI_DIR_RETURN;
}
static bool nvm_name(const NvmModule *m,uint32_t i,const char *s) {
    return i<m->string_count && m->strings && m->string_lengths &&
        exact_bytes((const uint8_t *)m->strings[i],m->string_lengths[i],s);
}
static bool v2_name(const NvmV2Module *m,uint32_t i,const char *s) {
    if (i>=m->constants.count || !m->constants.items) return false;
    const NvmV2Constant *c=&m->constants.items[i];
    return c->tag==TAG_STRING && exact_bytes(c->payload,c->length,s);
}
/* Version dispatch is transport-only. A malformed v2 claim cannot fall back
 * to v1 or to shared executable ownership validation. */
static bool nominal_version(const uint8_t *bytes,uint32_t size) {
    return bytes && size>=2 && bytes[0]==NVM_FILE_NOMINAL_VERSION && bytes[1]==0;
}
static NvmV2Result nominal_status(NvmFileNominalStatus status) {
    if(status==NVM_FILE_NOMINAL_DESCRIBED)return NVM_V2_OK;
    if(status==NVM_FILE_NOMINAL_MEMORY)return NVM_V2_ERR_TRUNCATED;
    return status==NVM_FILE_NOMINAL_LIMIT?NVM_V2_ERR_INDEX_RANGE:NVM_V2_ERR_SECTION_TYPE;
}
static NvmV2Result nominal_module(const NvmModule *m) {
    NvmFileNominalPlan *plan=NULL;
    NvmV2Result status=nominal_status(nvm_file_nominal_plan(m,&plan));
    nvm_file_nominal_plan_free(plan);return status;
}
static bool table_bytes(size_t count,size_t width,size_t *bytes) {
    if(width && count>SIZE_MAX/width)return false;
    *bytes=count*width;return true;
}
/* I adapt only metadata. No bridge/verifier callback, renumbering or ownership
 * flag projection is involved. All pointer-array views die before return. */
static NvmV2Result nominal_v2(const NvmV2Module *m) {
    if(m->imports.count!=NVM_SERVICE_BINDING_COUNT || !m->imports.items ||
       m->links.count || m->callbacks.count || !m->ownership_data || !m->ownership_size ||
       m->layouts.count<NVM_FILE_NOMINAL_TYPES || m->layouts.count>NVM_FILE_NOMINAL_MAX_LAYOUTS ||
       !m->layouts.items || (m->constants.count && !m->constants.items) ||
       (m->functions.count && !m->functions.items) || (m->signatures.count && !m->signatures.items) ||
       m->functions.count>m->ownership_size/12)return NVM_V2_ERR_SECTION_TYPE;
    size_t names_bytes,lengths_bytes,functions_bytes,params_bytes;
    if(!table_bytes(m->constants.count,sizeof(char *),&names_bytes) ||
       !table_bytes(m->constants.count,sizeof(uint32_t),&lengths_bytes) ||
       !table_bytes(m->functions.count,sizeof(NvmFunctionEntry),&functions_bytes) ||
       !table_bytes(m->functions.count,sizeof(uint8_t *),&params_bytes))return NVM_V2_ERR_INDEX_RANGE;
    size_t layout_bytes=4;
    NvmModule view={0};
    for(uint32_t i=0;i<m->layouts.count;i++) {
        const NvmV2Layout *layout=&m->layouts.items[i];
        size_t fields_bytes;
        if(layout->kind>NVM_V2_LAYOUT_KIND_MAX || (layout->field_count && !layout->fields))
            return NVM_V2_ERR_SECTION_TYPE;
        if(!table_bytes(layout->field_count,12,&fields_bytes) ||
           fields_bytes>SIZE_MAX-8 || layout_bytes>SIZE_MAX-8-fields_bytes)
            return NVM_V2_ERR_INDEX_RANGE;
        layout_bytes+=8+fields_bytes;
        if(layout_bytes>UINT32_MAX)return NVM_V2_ERR_INDEX_RANGE;
        for(uint32_t j=0;j<layout->field_count;j++)
            if(layout->fields[j].nested_idx!=NVM_V2_NO_INDEX && layout->fields[j].nested_idx>=i)
                return NVM_V2_ERR_INDEX_RANGE;
        view.struct_count+=layout->kind==NVM_V2_LAYOUT_STRUCT;
        view.union_count+=layout->kind==NVM_V2_LAYOUT_UNION;
        view.enum_count+=layout->kind==NVM_V2_LAYOUT_ENUM;
    }
    NvmV2Result result=NVM_V2_ERR_TRUNCATED;
    view.strings=names_bytes?malloc(names_bytes):NULL;
    view.string_lengths=lengths_bytes?malloc(lengths_bytes):NULL;
    view.functions=functions_bytes?calloc(1,functions_bytes):NULL;
    view.function_param_types=params_bytes?calloc(1,params_bytes):NULL;
    if((names_bytes && !view.strings) || (lengths_bytes && !view.string_lengths) ||
       (functions_bytes && !view.functions) || (params_bytes && !view.function_param_types))goto done;
    view.string_count=m->constants.count;view.function_count=m->functions.count;
    result=NVM_V2_ERR_SECTION_TYPE;
    for(uint32_t i=0;i<m->constants.count;i++) {
        const NvmV2Constant *c=&m->constants.items[i];
        if(c->tag!=TAG_STRING || (c->length && !c->payload))goto done;
        view.strings[i]=(char *)c->payload;view.string_lengths[i]=c->length;
    }
    for(uint32_t i=0;i<m->functions.count;i++) {
        const NvmV2Function *f=&m->functions.items[i];
        if(f->signature_idx>=m->signatures.count)goto done;
        const NvmV2Signature *sig=&m->signatures.items[f->signature_idx];
        if(sig->result_count>1 || (sig->result_count && !sig->result_tags) ||
           (sig->param_count && !sig->param_tags))goto done;
        view.functions[i].arity=sig->param_count;view.functions[i].local_count=f->local_count;
        view.functions[i].result_count=(uint8_t)sig->result_count;
        view.functions[i].result_tag=sig->result_count?sig->result_tags[0]:TAG_VOID;
        view.function_param_types[i]=(uint8_t *)sig->param_tags;
    }
    NvmImportEntry imports[NVM_SERVICE_BINDING_COUNT]={0};
    uint8_t *params[NVM_SERVICE_BINDING_COUNT]={0};
    view.imports=imports;view.import_param_types=params;view.import_count=NVM_SERVICE_BINDING_COUNT;
    for(uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        const NvmV2Import *im=&m->imports.items[i];
        if(im->kind!=NVM_V2_IMPORT_SERVICE || im->signature_idx>=m->signatures.count)goto done;
        const NvmV2Signature *sig=&m->signatures.items[im->signature_idx];
        if(sig->result_count!=1 || !sig->result_tags || (sig->param_count && !sig->param_tags))goto done;
        imports[i].module_name_idx=im->module_name_idx;imports[i].function_name_idx=im->symbol_name_idx;
        imports[i].kind=NVM_IMPORT_SERVICE;imports[i].param_count=sig->param_count;
        imports[i].return_type=sig->result_tags[0];params[i]=(uint8_t *)sig->param_tags;
    }
    /* Checked extent above makes the existing structural encoder bounded. The
     * private query below checks tags, names, canonical edges and exact flags. */
    view.layout_data=malloc(layout_bytes);
    if(!view.layout_data){result=NVM_V2_ERR_TRUNCATED;goto done;}
    view.layout_size=(uint32_t)layout_bytes;
    result=nvm_v2_layouts_encode(&m->layouts,view.layout_data,layout_bytes);
    if(result!=NVM_V2_OK)goto done;
    view.service_data=(uint8_t *)m->service_data;view.service_size=m->service_size;
    view.ownership_data=(uint8_t *)m->ownership_data;view.ownership_size=m->ownership_size;
    result=nominal_module(&view);
done:
    free(view.layout_data);free(view.strings);free(view.string_lengths);
    free(view.functions);free(view.function_param_types);return result;
}
NvmV2Result nvm_service_bindings_validate(const NvmModule *m) {
    if (!m) return NVM_V2_ERR_INDEX_RANGE;
    if (nvm_file_instructions_present(m) && !nominal_version(m->service_data,m->service_size))
        return NVM_V2_ERR_SECTION_TYPE;
    if (!nvm_service_bindings_present(m)) return NVM_V2_OK;
    if (nominal_version(m->service_data,m->service_size)) return nominal_module(m);
    NvmServiceBindings value;
    if (nvm_service_bindings_decode(m->service_data,m->service_size,&value)!=NVM_SERVICE_OK)
        return NVM_V2_ERR_SECTION_TYPE;
    if (m->import_count!=NVM_SERVICE_BINDING_COUNT || !m->imports ||
        m->module_ref_count || m->callback_contract_count) return NVM_V2_ERR_FEATURE_MISMATCH;
    for (uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        uint32_t index=value.imports[i];
        if (index>=m->import_count) return NVM_V2_ERR_INDEX_RANGE;
        const NvmImportEntry *im=&m->imports[index];
        const NlFilePlanMethod *method=nl_file_catalog_method(i);
        const uint8_t *params=m->import_param_types ? m->import_param_types[index] : NULL;
        if (im->kind!=NVM_IMPORT_SERVICE ||
            !nvm_name(m,im->module_name_idx,nl_file_catalog_interface()) ||
            !nvm_name(m,im->function_name_idx,method->id) ||
            !category_signature(method,params,im->param_count,&im->return_type,1))
            return NVM_V2_ERR_SECTION_TYPE;
    }
    return NVM_V2_OK;
}
NvmV2Result nvm_v2_service_bindings_validate(const NvmV2Module *m) {
    if (!m) return NVM_V2_ERR_INDEX_RANGE;
    if (nvm_v2_file_instructions_present(m) && !nominal_version(m->service_data,m->service_size))
        return NVM_V2_ERR_SECTION_TYPE;
    if (!nvm_v2_service_bindings_present(m)) return NVM_V2_OK;
    if (nominal_version(m->service_data,m->service_size)) return nominal_v2(m);
    NvmServiceBindings value;
    if (nvm_service_bindings_decode(m->service_data,m->service_size,&value)!=NVM_SERVICE_OK)
        return NVM_V2_ERR_SECTION_TYPE;
    if (m->imports.count!=NVM_SERVICE_BINDING_COUNT || !m->imports.items ||
        m->links.count || m->callbacks.count) return NVM_V2_ERR_FEATURE_MISMATCH;
    for (uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        uint32_t index=value.imports[i];
        if (index>=m->imports.count) return NVM_V2_ERR_INDEX_RANGE;
        const NvmV2Import *im=&m->imports.items[index];
        if (im->signature_idx>=m->signatures.count || !m->signatures.items)
            return NVM_V2_ERR_INDEX_RANGE;
        const NvmV2Signature *sig=&m->signatures.items[im->signature_idx];
        const NlFilePlanMethod *method=nl_file_catalog_method(i);
        if (im->kind!=NVM_V2_IMPORT_SERVICE ||
            !v2_name(m,im->module_name_idx,nl_file_catalog_interface()) ||
            !v2_name(m,im->symbol_name_idx,method->id) ||
            !category_signature(method,sig->param_tags,sig->param_count,
                                sig->result_tags,sig->result_count))
            return NVM_V2_ERR_SECTION_TYPE;
    }
    return NVM_V2_OK;
}
NvmV2Result nvm_service_bindings_attach(NvmModule *m,const NlFilePlan *plan,
                                       const NvmServiceBindings *bindings) {
    if (!m || !plan || nl_file_plan_method_count(plan)!=NVM_SERVICE_BINDING_COUNT ||
        strcmp(nl_file_plan_interface(plan),nl_file_catalog_interface())) return NVM_V2_ERR_SECTION_TYPE;
    for (uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++)
        if (nl_file_plan_method(plan,i)!=nl_file_catalog_method(i)) return NVM_V2_ERR_SECTION_TYPE;
    uint8_t staged[NVM_SERVICE_BINDING_BYTES];size_t size=0;
    if (nvm_service_bindings_encode(bindings,staged,sizeof staged,&size)!=NVM_SERVICE_OK)
        return NVM_V2_ERR_SECTION_TYPE;
    NvmModule candidate=*m;candidate.service_data=staged;candidate.service_size=(uint32_t)size;
    NvmV2Result r=nvm_service_bindings_validate(&candidate);
    if (r!=NVM_V2_OK) return r;
    if (m->service_data || m->service_size)
        return m->service_data && m->service_size==size && !memcmp(m->service_data,staged,size)
            ? NVM_V2_OK : NVM_V2_ERR_FEATURE_MISMATCH;
    uint8_t *owned=malloc(size);
    if (!owned) return NVM_V2_ERR_TRUNCATED;
    memcpy(owned,staged,size);m->service_data=owned;m->service_size=(uint32_t)size;
    return NVM_V2_OK;
}

NvmV2Result nvm_file_nominal_attach(NvmModule *m,const NlFilePlan *plan,
                                    const NvmFileNominalBindings *bindings) {
    if(!m || !plan || nl_file_plan_type_count(plan)!=NVM_FILE_NOMINAL_TYPES ||
       nl_file_plan_method_count(plan)!=NVM_SERVICE_BINDING_COUNT ||
       strcmp(nl_file_plan_interface(plan),nl_file_catalog_interface()))return NVM_V2_ERR_SECTION_TYPE;
    for(uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++)
        if(nl_file_plan_method(plan,i)!=nl_file_catalog_method(i))return NVM_V2_ERR_SECTION_TYPE;
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++)
        if(nl_file_plan_type(plan,i)!=nl_file_catalog_type(i))return NVM_V2_ERR_SECTION_TYPE;
    uint8_t staged[NVM_FILE_NOMINAL_BYTES];size_t size=0;
    if(nvm_file_nominal_encode(bindings,staged,sizeof staged,&size)!=NVM_SERVICE_OK)
        return NVM_V2_ERR_SECTION_TYPE;
    NvmModule candidate=*m;candidate.service_data=staged;candidate.service_size=(uint32_t)size;
    NvmV2Result result=nominal_module(&candidate);
    if(result!=NVM_V2_OK)return result;
    if(m->service_data || m->service_size)
        return m->service_data && m->service_size==size && !memcmp(m->service_data,staged,size)
            ?NVM_V2_OK:NVM_V2_ERR_FEATURE_MISMATCH;
    uint8_t *owned=malloc(size);
    if(!owned)return NVM_V2_ERR_TRUNCATED;
    memcpy(owned,staged,size);m->service_data=owned;m->service_size=(uint32_t)size;
    return NVM_V2_OK;
}
