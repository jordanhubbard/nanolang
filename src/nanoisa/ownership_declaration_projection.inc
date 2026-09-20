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
    /* I charge both owned halves before either additional allocation. The
     * common preflight already charged two complete union-reader passes. */
    size_t used=sizeof *ordinary+(size_t)ordinary->layouts.count*sizeof(NvmV2Layout);
    for(uint32_t i=0;i<ordinary->layouts.count;i++)
        used+=(size_t)ordinary->layouts.items[i].field_count*sizeof(NvmV2LayoutField);
    used+=(size_t)ordinary->type_count*sizeof(NvmOrdinaryArrayType);
    used+=(size_t)ordinary->binding_count*sizeof(NvmOrdinaryArrayBinding);
    size_t rows=(size_t)projection.count*sizeof(NvmUnionVariantFact);
    if(projection.count>65536u || used>OAA_BYTES || sizeof(NvmOwnershipDeclarationPlan)>OAA_BYTES-used ||
       rows>OAA_BYTES-used-sizeof(NvmOwnershipDeclarationPlan)) {
        nvm_ordinary_array_authority_free(ordinary);return declaration_result(NVM_DECL_LIMIT);
    }
    NvmOwnershipDeclarationPlan *p=calloc(1,sizeof *p);
    if(!p){nvm_ordinary_array_authority_free(ordinary);return declaration_result(NVM_DECL_MEMORY);}
    p->ordinary=ordinary;p->unions=m->union_count;p->count=projection.count;
    if(p->count) {
        p->rows=calloc(p->count,sizeof *p->rows);
        if(!p->rows){nvm_ownership_declarations_free(p);return declaration_result(NVM_DECL_MEMORY);}
        projection.rows=p->rows;projection.capacity=p->count;
        projection.starts=p->starts;projection.variants=p->variants;
        NvmV2Cursor cursor;nvm_v2_cursor_init(&cursor,projection.view.data,projection.view.size);
        NvmV2Result result=union_facts_read(&cursor,m,&ordinary->layouts,NVM_V2_NO_INDEX,0,NULL,&projection);
        if(result!=NVM_V2_OK || projection.count!=p->count) {
            nvm_ownership_declarations_free(p);return declaration_result(NVM_DECL_INVALID);
        }
    }
    *out=p;return declaration_result(NVM_DECL_PREPARED);
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
