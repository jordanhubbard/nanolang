/* Included only in the origin proof TU. Mutable summaries never escape. */
static int map_declarations_exact(Analysis *a,NvmRecordMapAnalysisContext *maps) {
    NvmMapDeclarationCounts counts;
    if(!analysis_work(a,NVM_MAP_SIMPLE_QUERY_WORK) ||
       !nvm_map_declarations_counts(maps->declarations,&counts))
        return stop(a,NVM_ARRAY_INVALID,0,0,"I require copied map counts.");
    if(counts.pairs!=1)return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require one exact STRING-to-INT declaration pair.");
    for(uint32_t i=0;i<counts.pairs;i++) {
        NvmMapDeclarationPair pair;uint8_t key,value;
        if(!analysis_work(a,3u*NVM_MAP_SIMPLE_QUERY_WORK))return 0;
        if(!nvm_map_declarations_pair(maps->declarations,i,&pair) ||
           !nvm_map_declarations_scalar(maps->declarations,pair.key_type,&key) ||
           !nvm_map_declarations_scalar(maps->declarations,pair.value_type,&value))
            return stop(a,NVM_ARRAY_INVALID,0,0,"I require exact map pair referents.");
        if(key!=TAG_STRING || value!=TAG_INT)
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I prove only exact STRING-to-INT maps.");
    }
    return 1;
}
static int map_receiver(Analysis *a,NvmRecordMapAnalysisContext *maps,
                        uint32_t fn,uint32_t pc,Value receiver) {
    if(!maps || receiver.unknown || receiver.tags!=BIT(TAG_HASHMAP) || !receiver.origins)
        return stop(a,NVM_ARRAY_UNRESOLVED,fn,pc,"I require exact known map origins.");
    if(!analysis_work(a,ORIGINS))return 0;
    uint64_t known=0;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(receiver.origins&(UINT64_C(1)<<i)) {
        if(a->origin_kind[i]!=NVM_RECORD_MAP_ORIGIN)
            return stop(a,NVM_ARRAY_UNRESOLVED,fn,pc,"I refuse non-map receiver origins.");
        known|=UINT64_C(1)<<i;
    }
    if(known!=receiver.origins)return stop(a,NVM_ARRAY_UNRESOLVED,fn,pc,"I refuse unknown map origin bits.");
    return 1;
}
static int map_exact_scalar(Analysis *a,uint32_t fn,uint32_t pc,Value value,uint8_t expected) {
    if(value.unknown || value.origins || value.tags!=BIT(expected))
        return stop(a,NVM_ARRAY_UNRESOLVED,fn,pc,"I require exact map key/value scalar tags.");
    return 1;
}
static int map_operation(Analysis *a,NvmRecordMapAnalysisContext *maps,uint32_t fn,
                         const VmDecodedInstruction *d,const Value *stack,uint16_t depth) {
    uint8_t op=d->instruction.opcode;
    if(op==OP_HM_NEW)return 1; /* Every encoded creation was checked, including dead code. */
    unsigned pops=op==OP_HM_SET?3u:op==OP_HM_LEN?1u:2u;
    if(depth<pops)return stop(a,NVM_ARRAY_INVALID,fn,d->byte_offset,"I require the verified map stack effect.");
    unsigned base=depth-pops;
    if(!map_receiver(a,maps,fn,d->byte_offset,stack[base]))return 0;
    if(op!=OP_HM_LEN && !map_exact_scalar(a,fn,d->byte_offset,stack[base+1],TAG_STRING))return 0;
    if(op==OP_HM_SET && !map_exact_scalar(a,fn,d->byte_offset,stack[base+2],TAG_INT))return 0;
    maps->checked++;return 1;
}
static int map_final(Analysis *a,NvmRecordMapAnalysisContext *maps) {
    if(!analysis_work(a,ORIGINS*2u+a->global_count))return 0;
    for(uint32_t i=0;i<a->global_count;i++)if(a->globals[i].tags&BIT(TAG_HASHMAP))
        return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I keep standalone map globals outside this field profile.");
    for(uint32_t i=0;i<a->report.origin_count;i++)if(a->origin_kind[i]==NVM_RECORD_MAP_ORIGIN) {
        Value key=maps->keys[i],value=maps->values[i];
        if(key.unknown || value.unknown || key.origins || value.origins ||
           (key.tags&~BIT(TAG_STRING)) || (value.tags&~BIT(TAG_INT)))
            return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require every historical map write to preserve its exact pair.");
    }
    return 1;
}
