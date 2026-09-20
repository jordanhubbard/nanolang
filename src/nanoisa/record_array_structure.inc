/* I prepare original-module structure only. No public route calls this entry. */
void nvm_record_array_structure_free(NvmRecordArrayStructure *p) {
    if(!p)return;
    for(uint32_t i=0;i<p->functions;i++)vm_decoded_function_free(&p->decoded[i]);
    nvm_ownership_declarations_free(p->declarations);free(p);
}
static NvmArrayEligibilityResult ra_result(NvmArrayEligibilityStatus status,uint32_t fn,
                                           const char *message) {
    NvmArrayEligibilityResult r={.status=status,.function=fn};
    snprintf(r.message,sizeof r.message,"%s",message);return r;
}
NvmArrayEligibilityResult nvm_record_array_structure_prepare(const NvmModule *m,
        NvmRecordArrayBudget *b,NvmRecordArrayStructure **out) {
    if(!m || !b || !out)return ra_result(NVM_ARRAY_INVALID,0,"I require private preparation inputs.");
    /* A fresh budget is construction state, not a reusable verification token. */
    if(b->bytes || b->work || b->peak || b->limited || b->memory)
        return ra_result(NVM_ARRAY_INVALID,0,"I require fresh private accounting.");
    /* Fixed automatic scratch across preparation/analysis call chains. Heap
     * capacities are charged separately; allocator metadata is not measured. */
    if(!nvm_ra_bytes(b,65536) || !nvm_ra_steps(b,65536))
        return ra_result(NVM_ARRAY_LIMIT,0,"I reached my fixed scratch reservation.");
    if(m->service_data || m->service_size || m->import_count || m->module_ref_count ||
       m->passive_data || m->passive_size || m->callback_contract_count)
        return ra_result(NVM_ARRAY_UNRESOLVED,0,"I require a closed ordinary module without host contracts.");
    if(m->function_count>256 || m->string_count>NVM_MAX_STRINGS || m->code_size>16u*1024u*1024u)
        return ra_result(NVM_ARRAY_LIMIT,0,"I reached my private module bound.");
    if(!m->function_count || !m->functions || (m->code_size && !m->code) ||
       (m->string_count && (!m->strings || !m->string_lengths)) ||
       m->section_count>NVM_MAX_SECTIONS || m->header.section_count>NVM_MAX_SECTIONS)
        return ra_result(NVM_ARRAY_INVALID,0,"I require consistent original module storage.");
    if(m->header.magic[0]!=NVM_MAGIC_0 || m->header.magic[1]!=NVM_MAGIC_1 ||
       m->header.magic[2]!=NVM_MAGIC_2 || m->header.magic[3]!=NVM_MAGIC_3 ||
       m->header.format_version!=NVM_FORMAT_VERSION ||
       (m->header.flags&~(NVM_FLAG_HAS_MAIN|NVM_FLAG_NEEDS_EXTERN|NVM_FLAG_DEBUG_INFO)))
        return ra_result(NVM_ARRAY_INVALID,0,"I require the original in-memory module header.");
    if(!(m->header.flags&NVM_FLAG_HAS_MAIN) || m->header.entry_point>=m->function_count)
        return ra_result(NVM_ARRAY_UNRESOLVED,0,"I require a bounded module entry.");
    for(uint32_t i=0;i<m->string_count;i++) {
        if(!m->strings[i])return ra_result(NVM_ARRAY_INVALID,0,"I require a present counted string.");
        if(!nvm_ra_steps(b,(uint64_t)m->string_lengths[i]+1))
            return ra_result(NVM_ARRAY_LIMIT,0,"I reached my counted-string work bound.");
        if(m->strings[i][m->string_lengths[i]]!='\0')
            return ra_result(NVM_ARRAY_INVALID,0,"I require terminated counted strings.");
    }
    if(!nvm_ra_steps(b,(uint64_t)m->metadata_count*32u+1))
        return ra_result(NVM_ARRAY_LIMIT,0,"I reached my metadata work bound.");
    if(!nvm_metadata_valid(m))return ra_result(NVM_ARRAY_INVALID,0,"I require consistent source metadata.");
    uint32_t counts[256]={0};uint32_t total=0;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        const NvmFunctionEntry *f=&m->functions[fi];
        if(!f->code_length && f->result_count)
            return ra_result(NVM_ARRAY_INVALID,fi,"I require a value-producing body for a declared result.");
        if(f->local_count>256)return ra_result(NVM_ARRAY_LIMIT,fi,"I reached my local bound.");
        if(f->upvalue_count || f->result_count>1 ||
           !(f->result_tag<=TAG_STRING || f->result_tag==TAG_ARRAY || f->result_tag==TAG_STRUCT))
            return ra_result(NVM_ARRAY_UNRESOLVED,fi,"I require ordinary direct-call signatures.");
        if(f->code_offset>m->code_size || f->code_length>m->code_size-f->code_offset)
            return ra_result(NVM_ARRAY_INVALID,fi,"I require bounded original code ranges.");
        if(!nvm_ra_steps(b,(uint64_t)f->code_length+256u+m->function_count))
            return ra_result(NVM_ARRAY_LIMIT,fi,"I reached my structural scan bound.");
        for(uint16_t p=0;p<f->arity;p++) {
            if(p>=256)return ra_result(NVM_ARRAY_LIMIT,fi,"I reached my parameter bound.");
            uint8_t tag=m->function_param_types && m->function_param_types[fi]?m->function_param_types[fi][p]:TAG_VOID;
            if(!(tag<=TAG_STRING || tag==TAG_ARRAY || tag==TAG_STRUCT))
                return ra_result(NVM_ARRAY_UNRESOLVED,fi,"I require ordinary parameter tags.");
        }
        for(uint32_t pc=0;pc<f->code_length;) {
            DecodedInstruction in;uint32_t width=isa_decode(m->code+f->code_offset+pc,f->code_length-pc,&in);
            if(!width)return ra_result(NVM_ARRAY_INVALID,fi,"I require decodable original instructions.");
            if(++total>65536)return ra_result(NVM_ARRAY_LIMIT,fi,"I reached my instruction bound.");
            counts[fi]++;
            if(!nvm_record_array_opcode_supported(in.opcode) ||
               (in.opcode==OP_AGG_PACK && (in.operands[0].u8!=AGG_RECORD || in.operands[2].u16)))
                return ra_result(NVM_ARRAY_UNRESOLVED,fi,"I have no ordinary record-array transfer for this opcode.");
            if((in.opcode==OP_LOAD_GLOBAL || in.opcode==OP_STORE_GLOBAL) && in.operands[0].u32>=256)
                return ra_result(NVM_ARRAY_LIMIT,fi,"I reached my global bound.");
            pc+=width;
        }
    }
    NvmVerifyResult r=verify_module_ranges(m);
    if(!r.ok)return ra_result(NVM_ARRAY_INVALID,0,r.error_msg);
    r=verify_module_contracts_and_imports(m);
    if(!r.ok)return ra_result(NVM_ARRAY_INVALID,0,r.error_msg);
    /* Reserve the complete independently bounded declaration preparation,
     * including its transient staging; retain this conservative reservation. */
    if(!nvm_ra_bytes(b,UINT64_C(16)*1024*1024) || !nvm_ra_steps(b,1048576))
        return ra_result(NVM_ARRAY_LIMIT,0,"I reached my declaration reservation bound.");
    NvmRecordArrayStructure *p=nvm_ra_alloc(b,1,sizeof *p);
    if(!p)return ra_result(b->limited?NVM_ARRAY_LIMIT:NVM_ARRAY_MEMORY,0,"I could not allocate private structure.");
    NvmDeclarationResult declarations=nvm_prepare_ownership_declarations(m,&p->declarations);
    if(declarations.status!=NVM_DECL_PREPARED) {
        nvm_record_array_structure_free(p);
        return ra_result(declarations.status==NVM_DECL_MEMORY?NVM_ARRAY_MEMORY:
            declarations.status==NVM_DECL_LIMIT?NVM_ARRAY_LIMIT:
            declarations.status==NVM_DECL_UNKNOWN?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_INVALID,0,declarations.message);
    }
    p->functions=m->function_count;
    for(uint32_t fi=0;fi<p->functions;fi++) {
        uint64_t capacity=16;while(capacity<counts[fi])capacity*=2;
        /* vm_decode's old+new realloc peak and byte-index arrays, retained
         * conservatively for the whole preparation. Every scan is charged. */
        uint64_t decoded=(uint64_t)(m->functions[fi].code_length+1)*5u+
            (counts[fi]?capacity*3u*sizeof(VmDecodedInstruction):0);
        if(!nvm_ra_bytes(b,decoded) || !nvm_ra_steps(b,(uint64_t)m->functions[fi].code_length+counts[fi]*16u)) {
            nvm_record_array_structure_free(p);return ra_result(NVM_ARRAY_LIMIT,fi,"I reached my decoded storage bound.");
        }
        r=verify_function_body(m,fi,NULL,0,&p->stacks[fi],b,&p->decoded[fi]);
        if(!r.ok) {
            nvm_record_array_structure_free(p);
            return ra_result(b->limited?NVM_ARRAY_LIMIT:b->memory || strstr(r.error_msg,"allocat")?
                NVM_ARRAY_MEMORY:NVM_ARRAY_INVALID,fi,r.error_msg);
        }
    }
    *out=p;return ra_result(NVM_ARRAY_ELIGIBLE,0,"I copied bounded structure, not executable authority.");
}
