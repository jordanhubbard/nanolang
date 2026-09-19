#include "service_bindings_module.h"
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
NvmV2Result nvm_service_bindings_validate(const NvmModule *m) {
    if (!m) return NVM_V2_ERR_INDEX_RANGE;
    if (!nvm_service_bindings_present(m)) return NVM_V2_OK;
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
    if (!nvm_v2_service_bindings_present(m)) return NVM_V2_OK;
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
