_Static_assert(NVM_RECORD_MAP_ORIGIN!=NVM_HEAP_ORIGIN_ARRAY && NVM_RECORD_MAP_ORIGIN!=NVM_HEAP_ORIGIN_RECORD,"map identity has a distinct private kind");
typedef struct { uint32_t ordinal,pair; } MapFrozenOrigin;
_Static_assert(sizeof(MapFrozenOrigin)==8,"map origin row has an exact copied size");
struct NvmRecordMapFrozenContext {
    NvmRecordArrayOrigins *base;
    NvmMapDeclarationPlan *declarations;
    MapFrozenOrigin *rows;
    NvmRecordMapCounts counts;
};
void nvm_record_map_frozen_free(NvmRecordMapFrozenContext *p) {
    if(!p)return;
    nvm_record_array_origins_free(p->base);nvm_map_declarations_free(p->declarations);
    free(p->rows);free(p);
}
static int map_freeze(Analysis *a,NvmRecordMapAnalysisContext *maps,NvmRecordMapFrozenContext **out) {
    NvmRecordMapFrozenContext *p=analysis_alloc(a,1,sizeof *p);
    if(!p)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate frozen map facts.");
    uint32_t count=0;
    if(!analysis_work(a,ORIGINS*2u)){free(p);return 0;}
    for(uint32_t i=0;i<a->report.origin_count;i++)count+=a->origin_kind[i]==NVM_RECORD_MAP_ORIGIN;
    if(count) {
        p->rows=analysis_alloc(a,count,sizeof *p->rows);
        if(!p->rows){free(p);return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not copy map origin rows.");}
    }
    uint32_t index=0;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(a->origin_kind[i]==NVM_RECORD_MAP_ORIGIN)
        p->rows[index++]=(MapFrozenOrigin){i,0};
    if(!ra_publish(a,&p->base)){nvm_record_map_frozen_free(p);return 0;}
    p->counts=(NvmRecordMapCounts){a->report.origin_count,count,a->field_count,maps->checked,a->budget.peak,a->budget.work};
    p->declarations=maps->declarations;maps->declarations=NULL;
    *out=p;return 1;
}
bool nvm_record_map_frozen_counts(const NvmRecordMapFrozenContext *p,NvmRecordMapCounts *out) {
    if(!p||!out)return false;
    *out=p->counts;return true;
}
bool nvm_record_map_frozen_origin(const NvmRecordMapFrozenContext *p,uint32_t origin,
                                  uint8_t *key,uint8_t *value) {
    if(!p||!key||!value||key==value)return false;
    for(uint32_t i=0;i<p->counts.map_origins;i++)if(p->rows[i].ordinal==origin) {
        NvmMapDeclarationPair pair;uint8_t k,v;
        if(!nvm_map_declarations_pair(p->declarations,p->rows[i].pair,&pair) ||
           !nvm_map_declarations_scalar(p->declarations,pair.key_type,&k) ||
           !nvm_map_declarations_scalar(p->declarations,pair.value_type,&v))return false;
        *key=k;*value=v;return true;
    }
    return false;
}
bool nvm_record_map_frozen_field(const NvmRecordMapFrozenContext *p,uint32_t layout,uint16_t field,
                                 uint8_t *key,uint8_t *value) {
    NvmMapDeclarationPair pair;uint8_t k,v;
    if(!p||!key||!value||key==value || !nvm_map_declarations_field(p->declarations,layout,field,&pair) ||
       !nvm_map_declarations_scalar(p->declarations,pair.key_type,&k) ||
       !nvm_map_declarations_scalar(p->declarations,pair.value_type,&v))return false;
    *key=k;*value=v;return true;
}
bool nvm_record_map_heap_origin(const NvmRecordMapFrozenContext *p,uint32_t i,NvmRecordHeapOrigin *out) {
    return p && nvm_record_array_origin(p->base,i,out);
}
bool nvm_record_map_field_value(const NvmRecordMapFrozenContext *p,uint32_t i,NvmRecordValueOrigins *out) {
    return p && nvm_record_array_field_value(p->base,i,out);
}
