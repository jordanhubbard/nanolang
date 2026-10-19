/* I expose a verified local-resource slice; declarations never grant execution. */
#include "affine_state.h"
#include "affine_bytecode.h"

static int owned_facts(const NvmModule *m) {
    NvmV2Layouts layouts={0};NvmAffineState *state=NULL;
    VmDecodedFunction code={0};char error[VM_DECODE_ERROR_SIZE];
    bool ok=false;
    if(m->ownership_size<4 || !m->ownership_data ||
       m->ownership_data[0]!=NVM_OWNERSHIP_VERSION || m->ownership_data[1] ||
       m->ownership_data[2] || m->ownership_data[3] || m->function_count!=1 || m->header.entry_point!=0 || m->import_count ||
       m->module_ref_count || m->passive_size || m->union_count || m->enum_count ||
       !nvm_get_string(m,m->functions[0].name_idx) ||
       !strcmp(nvm_get_string(m,m->functions[0].name_idx),"__init__") ||
       m->functions[0].arity || m->functions[0].upvalue_count ||
       m->functions[0].result_count!=1 || m->functions[0].result_tag!=TAG_INT ||
       m->functions[0].local_count>128 || (m->header.flags & NVM_FLAG_NEEDS_EXTERN))goto done;
    uint32_t globals=0;
    if(nvm_ownership_globals(m,NULL,0,&globals)!=NVM_V2_OK || globals ||
       nvm_v2_layouts_decode(m->layout_data,m->layout_size,&layouts)!=NVM_V2_OK ||
       !layouts.count || layouts.count>16 || layouts.count!=m->struct_count)goto done;
    for(uint32_t i=0;i<layouts.count;i++) {
        NvmLayoutAuthority authority;
        NvmV2Layout *l=&layouts.items[i];
        if(nvm_ownership_layout_authority(m,i,&authority)!=NVM_V2_OK ||
           authority!=NVM_LAYOUT_AUTHORITY_RESOURCE || l->kind!=NVM_V2_LAYOUT_STRUCT ||
           !l->field_count || l->field_count>16)goto done;
        for(uint16_t f=0;f<l->field_count;f++)
            if((l->fields[f].type_tag!=TAG_INT && l->fields[f].type_tag!=TAG_BOOL) ||
               l->fields[f].nested_idx!=NVM_V2_NO_INDEX)goto done;
    }
    state=nvm_affine_state_create(m,0,m->functions[0].local_count);
    if(!state || !nvm_affine_analyze_function(m,0).ok ||
       !vm_decode_function(m,0,&code,error) || code.instruction_count>4096)goto done;
    for(uint16_t i=0;i<m->functions[0].local_count;i++) {
        NvmAffineType t;
        if(!nvm_affine_local_type(state,i,&t) ||
           (t.tag!=TAG_INT && t.tag!=TAG_BOOL &&
            !(t.tag==TAG_STRUCT && t.layout<layouts.count && nvm_affine_type_is_owned(state,t))))goto done;
    }
    printf("{\"profile\":\"owned-local-v1\",\"entry\":0,\"layouts\":[");
    for(uint32_t i=0;i<layouts.count;i++) {
        if(i){putchar(',');}printf("{\"identity\":%u,\"resource\":true,\"fields\":[",i);
        for(uint16_t f=0;f<layouts.items[i].field_count;f++) {
            if(f){putchar(',');}printf("%u",layouts.items[i].fields[f].type_tag);
        }
        printf("]}");
    }
    printf("],\"functions\":[{\"name\":\"owned_main\",\"locals\":%u,\"result\":1,\"params\":[],\"names\":[",m->functions[0].local_count);
    for(uint16_t i=0;i<m->functions[0].local_count;i++){if(i){putchar(',');}printf("\"\"");}
    printf("],\"local_types\":[");
    for(uint16_t i=0;i<m->functions[0].local_count;i++) {
        NvmAffineType t;if(!nvm_affine_local_type(state,i,&t))goto done;
        if(i){putchar(',');}printf("[%u,%u]",t.tag,t.layout);
    }
    printf("],\"size\":%u,\"code\":[",code.code_size);
    for(uint32_t i=0;i<code.instruction_count;i++) {
        const VmDecodedInstruction *di=&code.instructions[i];
        const DecodedInstruction *in=&di->instruction;int64_t args[4]={0};
        for(uint8_t a=0;a<in->operand_count && a<4;a++)switch(in->operand_types[a]) {
            case OPERAND_U8:args[a]=in->operands[a].u8;break;
            case OPERAND_U16:args[a]=in->operands[a].u16;break;
            case OPERAND_U32:args[a]=in->operands[a].u32;break;
            case OPERAND_I32:args[a]=in->operands[a].i32;break;
            case OPERAND_I64:args[a]=in->operands[a].i64;break;
            default:break;
        }
        if(i){putchar(',');}printf("{\"pc\":%u,\"op\":\"%s\",\"arg\":%" PRId64 ",\"args\":[",di->byte_offset,isa_get_info(in->opcode)->name,args[0]);
        for(uint8_t a=0;a<in->operand_count && a<4;a++){if(a)putchar(',');printf("%" PRId64,args[a]);}
        printf("]}");
    }
    puts("]}]}");ok=!ferror(stdout);
done:
    vm_decoded_function_free(&code);nvm_affine_state_free(state);nvm_v2_layouts_free(&layouts);
    if(!ok)fputs("I require a verified bounded local resource module with scalar fields.\n",stderr);
    return ok?0:1;
}
