/* Private native target boundary. The public emitter never calls this route. */
static int nvm2c_map_target_error(char *error,size_t size,const char *message) {
    if(error && size)snprintf(error,size,"%s",message);
    return 0;
}
static int nvm2c_map_target(const NvmRecordMapImage *image,char *error,size_t size) {
    const NvmModule *m=nvm_record_map_image_borrow(image);
    const NvmRecordMapExecutionPlan *p=nvm_record_map_image_plan(image);
    NvmRecordMapExecutionCounts counts;
    if(!m || !p || !nvm_record_map_image_counts(image,&counts) ||
       counts.carrier.functions!=m->function_count ||
       m->function_count>NVM_RECORD_ARRAY_ADMISSION_FUNCTIONS || counts.carrier.entry>=m->function_count ||
       m->import_count || counts.carrier.records!=m->struct_count)
        return nvm2c_map_target_error(error,size,"I require a closed copied native map module.");
    const NvmFunctionEntry *entry=&m->functions[counts.carrier.entry];
    if(entry->arity || entry->upvalue_count || entry->result_count!=1 || entry->result_tag!=TAG_INT)
        return nvm2c_map_target_error(error,size,"I require a capture-free zero-argument INT native map entry.");
    if(counts.carrier.initializer!=UINT32_MAX) {
        if(counts.carrier.initializer>=m->function_count)return 0;
        const NvmFunctionEntry *init=&m->functions[counts.carrier.initializer];
        if(init->arity || init->upvalue_count || init->result_count>1 ||
           (init->result_count && init->result_tag!=TAG_INT))
            return nvm2c_map_target_error(error,size,"I require a zero-argument VOID/INT native map initializer.");
    }
    /* Preserve the public native layout/tag identity restriction for every row,
     * including declarations whose constructors are unreachable or absent. */
    for(uint32_t ordinal=0;ordinal<counts.carrier.records;ordinal++) {
        NvmRecordArrayExecutionDescriptor descriptor;NvmDeclarationLayout layout;
        if(!nvm_record_map_execution_descriptor(p,ordinal,&descriptor) ||
           descriptor.ordinal!=ordinal || descriptor.layout!=ordinal ||
           !nvm_record_map_execution_layout(p,descriptor.layout,&layout) ||
           layout.kind!=NVM_V2_LAYOUT_STRUCT || layout.fields!=descriptor.fields)
            return nvm2c_map_target_error(error,size,"I require identical native record ordinal and layout identities.");
    }
    uint32_t index=0;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        const NvmFunctionEntry *f=&m->functions[fi];
        if(f->code_offset>m->code_size || f->code_length>m->code_size-f->code_offset)return 0;
        for(uint32_t pc=0;pc<f->code_length;) {
            DecodedInstruction in;NvmRecordMapExecutionInstruction row;
            uint32_t width=isa_decode(m->code+f->code_offset+pc,f->code_length-pc,&in);
            const InstructionInfo *info=width?isa_get_info(in.opcode):NULL;
            if(!info || index>=counts.carrier.instructions ||
               !nvm_record_map_image_instruction(image,index++,&row) ||
               row.base.function!=fi || row.base.pc!=pc || row.base.next_pc!=pc+width ||
               row.base.opcode!=in.opcode || row.base.operand_count!=info->operand_count)return 0;
            uint32_t offset=1;
            for(uint8_t a=0;a<info->operand_count;a++) {
                uint32_t bytes=isa_operand_size(info->operands[a]);uint64_t bits=0;
                if(bytes>8 || offset>width || bytes>width-offset ||
                   row.base.operand_types[a]!=(uint8_t)info->operands[a])return 0;
                for(uint32_t k=0;k<bytes;k++)bits|=(uint64_t)m->code[f->code_offset+pc+offset+k]<<(8u*k);
                if(bits!=row.base.operand_bits[a])return 0;
                offset+=bytes;
            }
            if(offset!=width)return 0;
            if(row.base.recipe==NVM_RA_ROOT_STRING || in.opcode==OP_CAST_STRING ||
               in.opcode==OP_STRUCT_NEW || in.opcode==OP_STRUCT_LITERAL ||
               in.opcode==OP_STRUCT_GET || in.opcode==OP_STRUCT_SET)
                return nvm2c_map_target_error(error,size,"I have not qualified this native map string producer or STRUCT transport.");
            uint8_t operation=in.opcode==OP_HM_NEW?NVM_RM_NEW:in.opcode==OP_HM_SET?NVM_RM_SET:
                in.opcode==OP_HM_GET?NVM_RM_GET:in.opcode==OP_HM_HAS?NVM_RM_HAS:
                in.opcode==OP_HM_LEN?NVM_RM_LEN:NVM_RM_NONE;
            if(row.map_operation!=operation)return 0;
            if(operation) {
                uint16_t result=operation==NVM_RM_GET?(1u<<TAG_INT)|(1u<<TAG_VOID):
                    operation==NVM_RM_HAS?(1u<<TAG_BOOL):operation==NVM_RM_LEN?(1u<<TAG_INT):(1u<<TAG_HASHMAP);
                if(!operation || row.map_operation!=operation || row.base.recipe!=NVM_RM_INVALID_BASE_RECIPE ||
                   row.key_tags!=(1u<<TAG_STRING) || row.value_tags!=(1u<<TAG_INT) || row.result_tags!=result)return 0;
            } else if(row.base.recipe>NVM_RA_ROOT_TAIL)return 0;
            if(in.opcode==OP_PUSH_STR) {
                uint32_t at=in.operands[0].u32;
                if(at>=m->string_count || !m->strings || !m->string_lengths || !m->strings[at] ||
                   memchr(m->strings[at],0,m->string_lengths[at]))
                    return nvm2c_map_target_error(error,size,"I require NUL-free executable native map string literals.");
            }
            if(in.opcode==OP_AGG_PACK) {
                NvmRecordArrayExecutionDescriptor descriptor;NvmDeclarationLayout layout;
                uint32_t ordinal=in.operands[1].u32;
                if(in.operands[0].u8!=AGG_RECORD || in.operands[2].u16 ||
                   !nvm_record_map_execution_descriptor(p,ordinal,&descriptor) || descriptor.ordinal!=ordinal ||
                   descriptor.fields!=in.operands[3].u16 ||
                   !nvm_record_map_execution_layout(p,descriptor.layout,&layout) ||
                   layout.kind!=NVM_V2_LAYOUT_STRUCT || layout.fields!=descriptor.fields)return 0;
            }
            pc+=width;
        }
    }
    return index==counts.carrier.instructions;
}
/* Internal seam for source-destruction controls, never an exported trust API. */
static char *nvm2c_map_image_emit(const NvmRecordMapImage *image,char *error,size_t size) {
    if(!nvm2c_map_target(image,error,size)) {
        if(error && size && !error[0])snprintf(error,size,"I require matching private native map recipes and target bounds.");
        return NULL;
    }
    return nvm2c_emit_core(nvm_record_map_image_borrow(image),error,size,nvm_record_map_image_plan(image));
}
static char *nvm2c_emit_record_map_source(const NvmModule *source,char *error,size_t size,bool public_root) {
    if(error && size)error[0]=0;
    NvmRecordMapExecutionPlan *plan=NULL;NvmRecordMapImage *image=NULL;
    NvmArrayEligibilityResult result=public_root?nvm_record_map_admit(source,&plan):
        nvm_prepare_record_map_execution(source,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE) {
        nvm2c_map_target_error(error,size,result.message);return NULL;
    }
    result=nvm_record_map_image_copy(plan,&image);
    nvm_record_map_execution_free(plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE) {
        nvm2c_map_target_error(error,size,result.message);return NULL;
    }
    char *output=nvm2c_map_image_emit(image,error,size);
    nvm_record_map_image_free(image);
    return output;
}

#ifdef NANO_RECORD_MAP_PRIVATE_NATIVE
char *nvm2c_emit_record_map_private(const NvmModule *source,char *error,size_t size) {
    return nvm2c_emit_record_map_source(source,error,size,false);
}
#endif
