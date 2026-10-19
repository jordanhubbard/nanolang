/* I expose a verified local-resource slice; declarations never grant execution. */
#include "affine_state.h"
#include "affine_bytecode.h"

static int owned_facts(const NvmModule *m) {
    NvmV2Layouts layouts={0};NvmAffineState *state=NULL;
    VmDecodedFunction code={0};char error[VM_DECODE_ERROR_SIZE];
    bool ok=false;uint32_t total=0;uint8_t depths[16]={0};
    if(m->ownership_size<4 || !m->ownership_data ||
       m->ownership_data[0]!=NVM_OWNERSHIP_VERSION || m->ownership_data[1] ||
       m->ownership_data[2] || m->ownership_data[3] || !m->function_count || m->function_count>NVM_OWNED_MAX_FUNCTIONS || m->header.entry_point!=0 || m->import_count ||
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
        depths[i]=1;
        for(uint16_t f=0;f<l->field_count;f++) {
            uint8_t tag=l->fields[f].type_tag;uint32_t child=l->fields[f].nested_idx;
            if(tag==TAG_STRUCT) {
                if(child>=i)goto done;
                uint8_t depth=(uint8_t)(depths[child]+1);
                if(depth>3)goto done;
                if(depth>depths[i])depths[i]=depth;
            } else if((tag!=TAG_INT && tag!=TAG_BOOL) || child!=NVM_V2_NO_INDEX)goto done;
        }
    }
    bool borrowed=!nvm_affine_value_call_graph(m);
    if(borrowed && (m->function_count!=2 || m->functions[1].arity!=1))goto done;
    if(borrowed)for(uint32_t i=0;i<layouts.count;i++)if(depths[i]>2)goto done;
    printf("{\"profile\":\"owned-local-v1\",\"entry\":0,\"layouts\":[");
    for(uint32_t i=0;i<layouts.count;i++) {
        if(i){putchar(',');}printf("{\"identity\":%u,\"resource\":true,\"fields\":[",i);
        for(uint16_t f=0;f<layouts.items[i].field_count;f++) {
            if(f){putchar(',');}printf("%u",layouts.items[i].fields[f].type_tag);
        }
        printf("],\"field_types\":[");
        for(uint16_t f=0;f<layouts.items[i].field_count;f++) {
            if(f)putchar(',');
            printf("[%u,%u]",layouts.items[i].fields[f].type_tag,layouts.items[i].fields[f].nested_idx);
        }
        printf("]}");
    }
    printf("],\"functions\":[");
    for(uint32_t function=0;function<m->function_count;function++) {
        const NvmFunctionEntry *fn=&m->functions[function];
        NvmAffineType parameters[NVM_AFFINE_MAX_PARAMETERS],result;
        NvmReferenceMode mode=0;
        bool helper=borrowed && function==1;
        uint16_t count=0,fields=0;
        if(fn->local_count>128)goto done;
        state=nvm_affine_state_create(m,function,fn->local_count);
        if(!state || !nvm_affine_analyze_function(m,function).ok ||
           !vm_decode_function(m,function,&code,error) || code.instruction_count>4096)goto done;
        total+=code.instruction_count;if(total>4096)goto done;
        for(uint16_t i=0;i<fn->local_count;i++) {
            NvmAffineType t;
            if(!(helper && i==0 ? nvm_affine_parameter_at(state,0,&t,&mode) : nvm_affine_local_type(state,i,&t)) ||
               (t.tag!=TAG_INT && t.tag!=TAG_BOOL &&
                !(t.tag==TAG_STRUCT && t.layout<layouts.count && nvm_affine_type_is_owned(state,t))))goto done;
        }

        if(!nvm_get_string(m,fn->name_idx) || !strcmp(nvm_get_string(m,fn->name_idx),"__init__") ||
           !(helper ? (count=1, nvm_affine_parameter_at(state,0,&parameters[0],&mode) &&
                       (mode==NVM_REFERENCE_SHARED || mode==NVM_REFERENCE_EXCLUSIVE)) :
                       nvm_affine_value_parameters(state,parameters,NVM_AFFINE_MAX_PARAMETERS,&count)) || count!=fn->arity ||
           !nvm_affine_value_result(state,&result,&fields) ||
           fn->result_count!=(result.tag==TAG_VOID?0:1) ||
           (result.tag!=TAG_VOID && result.tag!=TAG_INT && result.tag!=TAG_BOOL &&
            !(result.tag==TAG_STRUCT && result.layout<layouts.count && nvm_affine_type_is_owned(state,result))))goto done;
        if(helper && (result.tag!=TAG_INT && result.tag!=TAG_BOOL))goto done;
        if(helper) {
            if(parameters[0].tag!=TAG_STRUCT || parameters[0].layout>=layouts.count)goto done;
            const NvmV2Layout *leaf=&layouts.items[parameters[0].layout];
            for(uint16_t field=0;field<leaf->field_count;field++)
                if((leaf->fields[field].type_tag!=TAG_INT && leaf->fields[field].type_tag!=TAG_BOOL) ||
                   leaf->fields[field].nested_idx!=NVM_V2_NO_INDEX)goto done;
        }
        for(uint16_t p=0;p<count;p++)
            if(parameters[p].tag!=TAG_INT && parameters[p].tag!=TAG_BOOL &&
               !(parameters[p].tag==TAG_STRUCT && parameters[p].layout<layouts.count && nvm_affine_type_is_owned(state,parameters[p])))goto done;
        if(function)putchar(',');
        printf("{\"name\":\"owned_%u\",\"locals\":%u,\"result_type\":[%u,%u],\"result\":%u,\"result_count\":%u,\"params\":[",function,fn->local_count,result.tag,result.layout,result.tag,fn->result_count);
        for(uint16_t p=0;p<count;p++){if(p)putchar(',');printf("[%u,%u]",parameters[p].tag,parameters[p].layout);}
        printf("],\"param_modes\":[");
        for(uint16_t p=0;p<count;p++){if(p)putchar(',');printf("%u",helper?(unsigned)mode:0);}
        printf("],\"names\":[");
        for(uint16_t i=0;i<fn->local_count;i++){if(i){putchar(',');}printf("\"\"");}
        printf("],\"local_types\":[");
        for(uint16_t i=0;i<fn->local_count;i++) {
            NvmAffineType t;if(!(helper && i==0 ? nvm_affine_parameter_at(state,0,&t,&mode) : nvm_affine_local_type(state,i,&t)))goto done;
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
            printf("],\"target\":%u}", di->resolved_target==UINT32_MAX ? UINT32_MAX :
                di->resolved_target-fn->code_offset);
        }
        printf("]}");
        vm_decoded_function_free(&code);nvm_affine_state_free(state);state=NULL;
    }
    puts("]}");ok=!ferror(stdout);
done:
    vm_decoded_function_free(&code);nvm_affine_state_free(state);nvm_v2_layouts_free(&layouts);
    if(!ok)fputs("I require a verified bounded resource module with exact fields.\n",stderr);
    return ok?0:1;
}
