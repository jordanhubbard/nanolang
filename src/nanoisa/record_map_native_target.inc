#include "record_map_native_target_internal.h"
/* Four operands, each at most eight bytes, have a fixed bounded decode/compare
 * schedule. The existing ISA maximum is stricter than that coarse bound. */
_Static_assert(MAX_OPERANDS==4,"I budget four native target operands.");
_Static_assert(ISA_MAX_INSTRUCTION_SIZE<=1+4*8,"I require a bounded native target instruction.");
NvmRecordMapTargetResult nvm_record_map_native_target(const NvmRecordMapImage *image,uint64_t allowance) {
    NvmRecordMapTargetResult result={{NVM_ARRAY_INVALID,0,0,{0}},0};
    const char *unsupported=NULL;
#define RMT_CHARGE(n) do {uint64_t amount=(uint64_t)(n); \
    if(result.work>allowance || amount>allowance-result.work){result.eligibility.status=NVM_ARRAY_LIMIT;return result;} \
    result.work+=amount;}while(0)
#define RMT_INVALID(text) do {snprintf(result.eligibility.message,sizeof result.eligibility.message,"%s",(text));return result;}while(0)
#define RMT_UNSUPPORTED(message) do {if(!unsupported)unsupported=(message);}while(0)
    RMT_CHARGE(64);
    const NvmModule *m=nvm_record_map_image_borrow(image);
    const NvmRecordMapExecutionPlan *p=nvm_record_map_image_plan(image);
    NvmRecordMapExecutionCounts counts;
    if(!m || !p || !nvm_record_map_image_counts(image,&counts) ||
       counts.carrier.functions!=m->function_count || counts.carrier.entry>=m->function_count ||
       counts.carrier.records!=m->struct_count || (m->function_count && !m->functions))
        RMT_INVALID("I require a closed copied native map module.");
    if(m->function_count>NVM_RECORD_ARRAY_ADMISSION_FUNCTIONS || m->import_count)
        RMT_UNSUPPORTED("I require a closed copied native map module.");
    const NvmFunctionEntry *entry=&m->functions[counts.carrier.entry];
    if(entry->arity || entry->upvalue_count || entry->result_count!=1 || entry->result_tag!=TAG_INT)
        RMT_UNSUPPORTED("I require a capture-free zero-argument INT native map entry.");
    if(counts.carrier.initializer!=UINT32_MAX) {
        if(counts.carrier.initializer>=m->function_count)return result;
        const NvmFunctionEntry *init=&m->functions[counts.carrier.initializer];
        if(init->arity || init->upvalue_count || init->result_count>1 ||
           (init->result_count && init->result_tag!=TAG_INT))
            RMT_UNSUPPORTED("I require a zero-argument VOID/INT native map initializer.");
    }
    /* Preserve the public native layout/tag identity restriction for every row,
     * including declarations whose constructors are unreachable or absent. */
    for(uint32_t ordinal=0;ordinal<counts.carrier.records;ordinal++) {
        RMT_CHARGE(32);
        NvmRecordArrayExecutionDescriptor descriptor;NvmDeclarationLayout layout;
        if(!nvm_record_map_execution_descriptor(p,ordinal,&descriptor) ||
           descriptor.ordinal!=ordinal ||
           !nvm_record_map_execution_layout(p,descriptor.layout,&layout) ||
           layout.kind!=NVM_V2_LAYOUT_STRUCT || layout.fields!=descriptor.fields)
            RMT_INVALID("I require identical native record ordinal and layout identities.");
        if(descriptor.layout!=ordinal)
            RMT_UNSUPPORTED("I require identical native record ordinal and layout identities.");
    }
    uint32_t index=0;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        RMT_CHARGE(16);
        const NvmFunctionEntry *f=&m->functions[fi];
        if((f->code_length && !m->code) || f->code_offset>m->code_size || f->code_length>m->code_size-f->code_offset)return result;
        for(uint32_t pc=0;pc<f->code_length;) {
            RMT_CHARGE(128);
            DecodedInstruction in;NvmRecordMapExecutionInstruction row;
            uint32_t width=isa_decode(m->code+f->code_offset+pc,f->code_length-pc,&in);
            RMT_CHARGE(width);
            const InstructionInfo *info=width?isa_get_info(in.opcode):NULL;
            if(!info || width>ISA_MAX_INSTRUCTION_SIZE || info->operand_count>MAX_OPERANDS || index>=counts.carrier.instructions ||
               !nvm_record_map_image_instruction(image,index++,&row) ||
               row.base.function!=fi || row.base.pc!=pc || row.base.next_pc!=pc+width ||
               row.base.opcode!=in.opcode || row.base.operand_count!=info->operand_count)return result;
            uint32_t offset=1;
            for(uint8_t a=0;a<info->operand_count;a++) {
                uint32_t bytes=isa_operand_size(info->operands[a]);uint64_t bits=0;
                if(bytes>8 || offset>width || bytes>width-offset ||
                   row.base.operand_types[a]!=(uint8_t)info->operands[a])return result;
                for(uint32_t k=0;k<bytes;k++)bits|=(uint64_t)m->code[f->code_offset+pc+offset+k]<<(8u*k);
                if(bits!=row.base.operand_bits[a])return result;
                offset+=bytes;
            }
            if(offset!=width)return result;
            NvmRecordArrayExecutionInstruction expected={0};
            if(!rme_recipe(in.opcode,&expected) || row.base.recipe!=expected.recipe ||
               row.base.obligations!=expected.obligations)return result;
            /* NUL-free literal leaves and closed immutable transport make
             * concatenation byte-preserving. Other producers remain unqualified. */
            if((row.base.recipe==NVM_RA_ROOT_STRING && in.opcode!=OP_STR_CONCAT) || in.opcode==OP_CAST_STRING ||
               in.opcode==OP_STRUCT_NEW || in.opcode==OP_STRUCT_LITERAL ||
               in.opcode==OP_STRUCT_GET || in.opcode==OP_STRUCT_SET)
                RMT_UNSUPPORTED("I have not qualified this native map string producer or STRUCT transport.");
            uint8_t operation=in.opcode==OP_HM_NEW?NVM_RM_NEW:in.opcode==OP_HM_SET?NVM_RM_SET:
                in.opcode==OP_HM_GET?NVM_RM_GET:in.opcode==OP_HM_HAS?NVM_RM_HAS:
                in.opcode==OP_HM_LEN?NVM_RM_LEN:in.opcode==OP_HM_DELETE?NVM_RM_DELETE:NVM_RM_NONE;
            if(row.map_operation!=operation)return result;
            if(operation) {
                uint16_t result_tags=operation==NVM_RM_GET?(1u<<TAG_INT)|(1u<<TAG_VOID):
                    operation==NVM_RM_HAS?(1u<<TAG_BOOL):operation==NVM_RM_LEN?(1u<<TAG_INT):(1u<<TAG_HASHMAP);
                if(!operation || row.map_operation!=operation || row.base.recipe!=NVM_RM_INVALID_BASE_RECIPE ||
                   row.key_tags!=(1u<<TAG_STRING) || row.value_tags!=(1u<<TAG_INT) || row.result_tags!=result_tags)return result;
            } else if(row.base.recipe>NVM_RA_ROOT_TAIL)return result;
            if(in.opcode==OP_PUSH_STR) {
                uint32_t at=in.operands[0].u32;
                if(at>=m->string_count || !m->strings || !m->string_lengths || !m->strings[at])
                    RMT_INVALID("I require NUL-free executable native map string literals.");
                RMT_CHARGE(m->string_lengths[at]);
                if(memchr(m->strings[at],0,m->string_lengths[at]))
                    RMT_UNSUPPORTED("I require NUL-free executable native map string literals.");
            }
            if(in.opcode==OP_AGG_PACK) {
                NvmRecordArrayExecutionDescriptor descriptor;NvmDeclarationLayout layout;
                uint32_t ordinal=in.operands[1].u32;
                if(in.operands[0].u8!=AGG_RECORD || in.operands[2].u16 ||
                   !nvm_record_map_execution_descriptor(p,ordinal,&descriptor) || descriptor.ordinal!=ordinal ||
                   descriptor.fields!=in.operands[3].u16 ||
                   !nvm_record_map_execution_layout(p,descriptor.layout,&layout) ||
                   layout.kind!=NVM_V2_LAYOUT_STRUCT || layout.fields!=descriptor.fields)return result;
            }
            pc+=width;
        }
    }
    if(index!=counts.carrier.instructions)return result;
    result.eligibility.status=unsupported?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_ELIGIBLE;
    if(unsupported)snprintf(result.eligibility.message,sizeof result.eligibility.message,"%s",unsupported);
#undef RMT_CHARGE
#undef RMT_INVALID
#undef RMT_UNSUPPORTED
    return result;
}
