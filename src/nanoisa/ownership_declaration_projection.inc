#include "ownership_declaration_projection.h"
struct NvmOwnershipDeclarationPlan {
    NvmOrdinaryArrayAuthority *ordinary;
    NvmUnionVariantFact *rows;
    uint32_t unions, count;
    uint32_t starts[NVM_OWNERSHIP_MAX_UNIONS];
    uint16_t variants[NVM_OWNERSHIP_MAX_UNIONS];
};
static NvmDeclarationResult declaration_result(NvmDeclarationStatus status) {
    const char *message="I could not establish complete mixed declarations.";
    if(status==NVM_DECL_PREPARED)message="I copied complete declarations without execution authority.";
    else if(status==NVM_DECL_INVALID)message="I require consistent complete declaration bytes.";
    else if(status==NVM_DECL_LIMIT)message="I reached my declaration storage or work bound.";
    else if(status==NVM_DECL_MEMORY)message="I could not allocate my declaration projection.";
    return (NvmDeclarationResult){status,message};
}
void nvm_ownership_declarations_free(NvmOwnershipDeclarationPlan *p) {
    if(!p)return;
    nvm_ordinary_array_authority_free(p->ordinary);free(p->rows);free(p);
}
static NvmDeclarationResult declaration_publish(const OwnershipModuleFacts *facts,
        NvmOrdinaryArrayAuthority *ordinary,OwnershipProjection *projection,
        NvmOwnershipDeclarationPlan **out) {
    /* I charge both owned halves before either additional allocation. The
     * common preflight already charged two complete union-reader passes. */
    size_t used=sizeof *ordinary+(size_t)ordinary->layouts.count*sizeof(NvmV2Layout);
    for(uint32_t i=0;i<ordinary->layouts.count;i++)
        used+=(size_t)ordinary->layouts.items[i].field_count*sizeof(NvmV2LayoutField);
    used+=(size_t)ordinary->type_count*sizeof(NvmOrdinaryArrayType);
    used+=(size_t)ordinary->binding_count*sizeof(NvmOrdinaryArrayBinding);
    size_t rows=(size_t)projection->count*sizeof(NvmUnionVariantFact);
    if(projection->count>65536u || used>OAA_BYTES || sizeof(NvmOwnershipDeclarationPlan)>OAA_BYTES-used ||
       rows>OAA_BYTES-used-sizeof(NvmOwnershipDeclarationPlan)) {
        nvm_ordinary_array_authority_free(ordinary);return declaration_result(NVM_DECL_LIMIT);
    }
    NvmOwnershipDeclarationPlan *p=calloc(1,sizeof *p);
    if(!p){nvm_ordinary_array_authority_free(ordinary);return declaration_result(NVM_DECL_MEMORY);}
    p->ordinary=ordinary;p->unions=facts->union_count;p->count=projection->count;
    if(p->count) {
        p->rows=calloc(p->count,sizeof *p->rows);
        if(!p->rows){nvm_ownership_declarations_free(p);return declaration_result(NVM_DECL_MEMORY);}
        projection->rows=p->rows;projection->capacity=p->count;
        projection->starts=p->starts;projection->variants=p->variants;
        NvmV2Cursor cursor;nvm_v2_cursor_init(&cursor,projection->view.data,projection->view.size);
        NvmV2Result result=union_facts_read(&cursor,facts,&ordinary->layouts,NVM_V2_NO_INDEX,0,NULL,projection);
        if(result!=NVM_V2_OK || projection->count!=p->count) {
            nvm_ownership_declarations_free(p);return declaration_result(NVM_DECL_INVALID);
        }
    }
    *out=p;return declaration_result(NVM_DECL_PREPARED);
}
NvmDeclarationResult nvm_prepare_ownership_declarations(
        const NvmModule *m,NvmOwnershipDeclarationPlan **out) {
    if(!out)return declaration_result(NVM_DECL_INVALID);
    OwnershipProjection projection={0};NvmOrdinaryArrayAuthority *ordinary=NULL;
    NvmOrdinaryArrayResult prepared=oaa_prepare(m,&ordinary,&projection);
    if(prepared.status!=NVM_OAA_DESCRIBED) {
        NvmDeclarationStatus status=prepared.status==NVM_OAA_UNKNOWN?NVM_DECL_UNKNOWN:
            prepared.status==NVM_OAA_LIMIT?NVM_DECL_LIMIT:
            prepared.status==NVM_OAA_MEMORY?NVM_DECL_MEMORY:NVM_DECL_INVALID;
        return declaration_result(status);
    }
    OwnershipModuleFacts facts=ownership_legacy_facts(m);
    return declaration_publish(&facts,ordinary,&projection,out);
}
/* I reuse the exact private layout decoder through a bounded layout-only wire
 * copy. This is not a module/legacy conversion: selectors and constants stay
 * in the caller's retained V2 tables, and no auxiliary section is discarded. */
NvmDeclarationResult nvm_prepare_ownership_declarations_v2(
        const NvmV2Module *m,NvmOwnershipDeclarationPlan **out) {
    if(!m || !out || (m->layouts.count&&!m->layouts.items) ||
       (m->functions.count&&!m->functions.items) ||
       (m->constants.count&&!m->constants.items) ||
       (m->signatures.count&&!m->signatures.items) ||
       (!m->ownership_data != !m->ownership_size))return declaration_result(NVM_DECL_INVALID);
    if(!m->ownership_size)return declaration_result(NVM_DECL_UNKNOWN);
    if(m->ownership_size>OAA_BYTES || m->layouts.count>NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_LAYOUTS)
        return declaration_result(NVM_DECL_LIMIT);
    OwnershipModuleFacts facts={NULL,m,m->functions.count,0,m->ownership_data,m->ownership_size};
    uint32_t fields=0;
    for(uint32_t i=0;i<m->layouts.count;i++) {
        const NvmV2Layout *l=&m->layouts.items[i];
        if(l->field_count && !l->fields)return declaration_result(NVM_DECL_INVALID);
        if(l->field_count>NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_FIELDS-fields)
            return declaration_result(NVM_DECL_LIMIT);
        fields+=l->field_count;
        facts.union_count+=l->kind==NVM_V2_LAYOUT_UNION;
    }
    /* All dimensions are bounded before arithmetic or selector traversal.
     * I charge encode/decode/cycle/name/descriptor visits conservatively. */
    uint32_t work=fields*8u+m->layouts.count*8u+
        2u*NVM_OWNERSHIP_MAX_PATHS*(NVM_OWNERSHIP_MAX_PATH_DEPTH+4u);
    uint32_t descriptor_cost=fields*3u+m->layouts.count+33u;
    if(m->functions.count>(OAA_STEPS-work)/descriptor_cost)return declaration_result(NVM_DECL_LIMIT);
    for(uint32_t i=0;i<m->functions.count;i++) {
        OwnershipFunctionFacts f;
        if(!ownership_function_facts(&facts,i,&f) || f.params>f.locals || f.results>1)
            return declaration_result(NVM_DECL_INVALID);
        uint32_t descriptors=(uint32_t)f.locals+1u;
        if(descriptors>(OAA_STEPS-work)/descriptor_cost)return declaration_result(NVM_DECL_LIMIT);
        work+=descriptors*descriptor_cost;
    }
    if(facts.union_count>(OAA_STEPS-work)/131072u)return declaration_result(NVM_DECL_LIMIT);
    work+=facts.union_count*131072u;
    size_t encoded=4u+(size_t)m->layouts.count*8u+(size_t)fields*12u;
    /* The shared decoder's forward-DAG workspace is one byte plus two u32s
     * per layout. I include its temporary and the layout wire in peak storage.
     * Fixed profile ceilings also bound later types/bindings/union plan storage. */
    size_t peak=encoded+sizeof(NvmOrdinaryArrayAuthority)+
        (size_t)m->layouts.count*(sizeof(NvmV2Layout)+1u+2u*sizeof(uint32_t))+
        (size_t)fields*sizeof(NvmV2LayoutField);
    if(peak>OAA_BYTES)return declaration_result(NVM_DECL_LIMIT);
    uint8_t *wire=calloc(1,encoded);
    if(!wire)return declaration_result(NVM_DECL_MEMORY);
    NvmV2Result result=nvm_v2_layouts_encode(&m->layouts,wire,encoded);
    if(result!=NVM_V2_OK){free(wire);return declaration_result(NVM_DECL_INVALID);}
    NvmOrdinaryArrayAuthority *ordinary=calloc(1,sizeof *ordinary);
    if(!ordinary){free(wire);return declaration_result(NVM_DECL_MEMORY);}
    ordinary->failure=NVM_OAA_INVALID;ordinary->steps=work;
    bool allocation_failed=false;
    result=nvm_ownership_mixed_layouts_private_decode_detailed(wire,encoded,
        &ordinary->layouts,&allocation_failed);
    free(wire);
    if(result!=NVM_V2_OK) {
        nvm_ordinary_array_authority_free(ordinary);
        return declaration_result(allocation_failed?NVM_DECL_MEMORY:NVM_DECL_INVALID);
    }
    OwnershipProjection projection={0};
    NvmOrdinaryArrayStatus status=oaa_check_declarations(&facts,ordinary,&projection);
    if(status!=NVM_OAA_DESCRIBED) {
        nvm_ordinary_array_authority_free(ordinary);
        return declaration_result(status==NVM_OAA_UNKNOWN?NVM_DECL_UNKNOWN:
            status==NVM_OAA_LIMIT?NVM_DECL_LIMIT:status==NVM_OAA_MEMORY?NVM_DECL_MEMORY:NVM_DECL_INVALID);
    }
    return declaration_publish(&facts,ordinary,&projection,out);
}
bool nvm_ownership_declarations_counts(const NvmOwnershipDeclarationPlan *p,NvmDeclarationCounts *out) {
    if(!p||!out)return false;
    *out=(NvmDeclarationCounts){p->ordinary->layouts.count,p->ordinary->type_count,
        p->ordinary->binding_count,p->unions,p->count};return true;
}
bool nvm_ownership_declarations_layout(const NvmOwnershipDeclarationPlan *p,uint32_t i,NvmDeclarationLayout *out) {
    if(!p||!out||i>=p->ordinary->layouts.count)return false;
    const NvmV2Layout *row=&p->ordinary->layouts.items[i];
    *out=(NvmDeclarationLayout){row->kind,p->ordinary->flags[i],row->field_count,row->name_idx};return true;
}
bool nvm_ownership_declarations_field(const NvmOwnershipDeclarationPlan *p,uint32_t i,uint16_t field,NvmV2LayoutField *out) {
    if(!p||!out||i>=p->ordinary->layouts.count||field>=p->ordinary->layouts.items[i].field_count)return false;
    *out=p->ordinary->layouts.items[i].fields[field];return true;
}
bool nvm_ownership_declarations_type(const NvmOwnershipDeclarationPlan *p,uint32_t i,NvmOrdinaryArrayType *out) {
    return p && nvm_ordinary_array_authority_type(p->ordinary,i,out);
}
bool nvm_ownership_declarations_binding(const NvmOwnershipDeclarationPlan *p,uint32_t i,NvmOrdinaryArrayBinding *out) {
    return p && nvm_ordinary_array_authority_binding(p->ordinary,i,out);
}
bool nvm_ownership_declarations_variant(const NvmOwnershipDeclarationPlan *p,uint32_t ordinal,uint16_t variant,NvmUnionVariantFact *out) {
    if(!p||!out||ordinal>=p->unions||variant>=p->variants[ordinal])return false;
    *out=p->rows[p->starts[ordinal]+variant];return true;
}
