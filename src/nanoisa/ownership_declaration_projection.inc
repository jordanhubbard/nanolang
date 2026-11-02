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
static NvmDeclarationResult prepare_ownership_declarations_map(
        const NvmModule *m,NvmOwnershipDeclarationPlan **out,NvmMapDeclarationPlan *map) {
    if(!out)return declaration_result(NVM_DECL_INVALID);
    OwnershipProjection projection={0};NvmOrdinaryArrayAuthority *ordinary=NULL;
    NvmOrdinaryArrayResult prepared=oaa_prepare_map(m,&ordinary,&projection,map);
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
        NvmV2Result result=union_facts_read(&cursor,m,&ordinary->layouts,ordinary->flags,NVM_V2_NO_INDEX,0,NULL,&projection);
        if(result!=NVM_V2_OK || projection.count!=p->count) {
            nvm_ownership_declarations_free(p);return declaration_result(NVM_DECL_INVALID);
        }
    }
    *out=p;return declaration_result(NVM_DECL_PREPARED);
}
NvmDeclarationResult nvm_prepare_ownership_declarations(const NvmModule *m,NvmOwnershipDeclarationPlan **out) {
    return prepare_ownership_declarations_map(m,out,NULL);
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

NvmMapCostStatus nvm_ownership_declarations_retained_cost(const NvmOwnershipDeclarationPlan *p,
                                                       NvmMapCopyWork *w,uint64_t *out) {
    if(!nvm_map_copy_charge(w,16))return NVM_MAP_COST_LIMIT;
    if(!p || !out || !p->ordinary || p->unions>256 || p->count>65536)return NVM_MAP_COST_INVALID;
    const NvmOrdinaryArrayAuthority *a=p->ordinary;
    if(a->layouts.count>256 || a->type_count>4096 || a->binding_count>65536)return NVM_MAP_COST_INVALID;
    uint64_t bytes=0;NvmMapCostStatus status;
#define MDC_TERM(ptr,count) do {status=nvm_map_copy_term(w,(ptr),(count),sizeof *(ptr),&bytes);if(status!=NVM_MAP_COST_OK)return status;}while(0)
    MDC_TERM(p,1);MDC_TERM(a,1);MDC_TERM(a->layouts.items,a->layouts.count);
    MDC_TERM(p->rows,p->count);MDC_TERM(a->types,a->type_count);MDC_TERM(a->bindings,a->binding_count);
    uint32_t fields=0;
    for(uint32_t i=0;i<a->layouts.count;i++) {
        if(!nvm_map_copy_charge(w,8))return NVM_MAP_COST_LIMIT;
        const NvmV2Layout *row=&a->layouts.items[i];
        if(row->field_count>65536u-fields)return NVM_MAP_COST_INVALID;
        fields+=row->field_count;MDC_TERM(row->fields,row->field_count);
    }
#undef MDC_TERM
    *out=bytes;return NVM_MAP_COST_OK;
}
