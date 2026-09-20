/* I share only private declaration ownership/actual extents with file_flow.c.
 * This plan never reaches a public admission or executing consumer. */
#include "file_code.h"
struct NvmFileCodePlan {
    NvmFileFlowDeclarations *declarations;
    uint8_t *code;
    NvmFileCodeInstruction *instructions;
    uint32_t count, code_size, instruction_count;
    size_t storage_bound;
    NvmFileCodeFunction functions[NVM_FILE_FLOW_FUNCTIONS];
    uint32_t starts[NVM_FILE_FLOW_FUNCTIONS], function_order[NVM_FILE_FLOW_FUNCTIONS];
    uint16_t instruction_order[NVM_FILE_CODE_INSTRUCTIONS];
    uint64_t calls[NVM_FILE_FLOW_FUNCTIONS];
};
static bool file_code_supported_mode(uint8_t op,bool indirect) {
    if(indirect && (op==OP_FUNCREF || op==OP_CALL_INDIRECT))return true;
    switch(op) {
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
    case OP_DUP: case OP_POP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
    case OP_REGION_BEGIN: case OP_REGION_END: case OP_BORROW_LOCAL_EXCLUSIVE:
    case OP_FILE_SERVICE: case OP_FILE_RESULT_BRANCH: case OP_FILE_RESULT_TAKE:
    case OP_FILE_DROP_LOCAL: case OP_FILE_DROP_STACK: case OP_FILE_END_BORROW:
    case OP_CALL: case OP_CALL_REF: case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:
    case OP_RET: case OP_ASSERT:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S:
    case OP_I64_REM_S: case OP_I64_NEG:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S:
    case OP_I64_GT_S: case OP_I64_GE_S: case OP_AND: case OP_OR: case OP_NOT:
    case OP_AGG_PACK: case OP_UNION_CONSTRUCT: case OP_AGG_GET: case OP_UNION_FIELD:
    case OP_AGG_TAG: case OP_UNION_TAG:return true;
    default:return false;
    }
}
static bool file_code_supported(uint8_t op) { return file_code_supported_mode(op,false); }
static bool file_code_budget(size_t *bytes,size_t count,size_t width) {
    if(count && width>SIZE_MAX/count)return false;
    size_t add=count*width;
    if(*bytes>NVM_FILE_FLOW_BYTES || add>NVM_FILE_FLOW_BYTES-*bytes)return false;
    *bytes+=add;return true;
}
static NvmFileFlowStatus file_code_preflight(const NvmModule *m,uint32_t *instructions,
                                             size_t *budget) {
    if(!m || !m->functions || !m->function_count || !m->code || !m->code_size)
        return NVM_FILE_FLOW_INVALID;
    if(m->function_count>NVM_FILE_FLOW_FUNCTIONS || m->code_size>NVM_FILE_CODE_BYTES)
        return NVM_FILE_FLOW_LIMIT;
    if(m->header.entry_point>=m->function_count)return NVM_FILE_FLOW_INVALID;
    uint32_t total=0,covered=0;size_t locals=0;
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];
        if(!fn->code_length || fn->code_offset>m->code_size ||
           fn->code_length>m->code_size-fn->code_offset || fn->arity>fn->local_count ||
           fn->name_idx>=m->string_count || fn->result_tag>=TAG_COUNT ||
           (fn->result_count==0)!=(fn->result_tag==TAG_VOID))
            return NVM_FILE_FLOW_INVALID;
        if(fn->local_count>NVM_FILE_FLOW_LOCALS)return NVM_FILE_FLOW_LIMIT;
        locals+=fn->local_count;
        for(uint32_t prior=0;prior<f;prior++) {
            const NvmFunctionEntry *p=&m->functions[prior];
            if(fn->code_offset<p->code_offset+p->code_length &&
               p->code_offset<fn->code_offset+fn->code_length)return NVM_FILE_FLOW_INVALID;
        }
        covered+=fn->code_length;uint32_t pc=0,count=0;
        while(pc<fn->code_length) {
            DecodedInstruction decoded={0};
            uint32_t width=isa_decode(m->code+fn->code_offset+pc,fn->code_length-pc,&decoded);
            if(!width)return NVM_FILE_FLOW_INVALID;
            if(++count>NVM_FILE_CODE_FUNCTION_INSTRUCTIONS || ++total>NVM_FILE_CODE_INSTRUCTIONS)
                return NVM_FILE_FLOW_LIMIT;
            pc+=width;
        }
    }
    if(covered!=m->code_size)return NVM_FILE_FLOW_UNRESOLVED;
    size_t bytes=0,nominal_bound=0;
    if(!nvm_file_nominal_storage_bound(NVM_FILE_NOMINAL_MAX_LAYOUTS,&nominal_bound) ||
       !file_code_budget(&bytes,1,sizeof(NvmFileFlowDeclarations)) ||
       !file_code_budget(&bytes,locals,sizeof(NvmFileFlowDeclaration)) ||
       !file_code_budget(&bytes,1,nominal_bound) ||
       !file_code_budget(&bytes,1,sizeof(NvmFileCodePlan)) ||
       !file_code_budget(&bytes,m->code_size,1) ||
       !file_code_budget(&bytes,total,sizeof(NvmFileCodeInstruction)) ||
       /* Fixed CFG indegrees/selection and transient decoded/preflight facts. */
       !file_code_budget(&bytes,NVM_FILE_CODE_FUNCTION_INSTRUCTIONS,sizeof(uint16_t)+sizeof(bool)) ||
       !file_code_budget(&bytes,1,sizeof(DecodedInstruction)))return NVM_FILE_FLOW_LIMIT;
    *instructions=total;*budget=bytes;return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus file_code_operands(NvmFileCodePlan *p,uint32_t function,
                                            NvmFileCodeInstruction *in,bool indirect) {
    const DecodedInstruction *d=&in->decoded;
    const NvmFileFlowFunction *fn=&p->functions[function].declaration;
    if(!file_code_supported_mode(d->opcode,indirect))return NVM_FILE_FLOW_UNRESOLVED;
    switch(d->opcode) {
    case OP_FUNCREF:
        return d->operands[0].u32<p->count?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_INVALID;
    case OP_CALL_INDIRECT:
        if(d->operands[0].u16>NVM_FILE_FLOW_LOCALS || d->operands[1].u16>1)
            return NVM_FILE_FLOW_INVALID;
        break;
    case OP_PUSH_BOOL:return d->operands[0].u8<=1?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_INVALID;
    case OP_LOAD_LOCAL: case OP_STORE_LOCAL: case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
    case OP_FILE_DROP_LOCAL: case OP_FILE_RESULT_BRANCH: case OP_FILE_RESULT_TAKE:
        if(d->operands[0].u16>=fn->locals)return NVM_FILE_FLOW_INVALID;
        if(d->opcode==OP_FILE_RESULT_TAKE && d->operands[1].u8>1)return NVM_FILE_FLOW_INVALID;
        break;
    case OP_BORROW_LOCAL_EXCLUSIVE:
        if(d->operands[0].u16>=NVM_FILE_FLOW_REFERENCES || d->operands[1].u16>=fn->locals)
            return NVM_FILE_FLOW_INVALID;
        break;
    case OP_FILE_END_BORROW:
        if(d->operands[0].u16>=NVM_FILE_FLOW_REFERENCES)return NVM_FILE_FLOW_INVALID;
        break;
    case OP_CALL: case OP_CALL_REF: {
        uint32_t target=d->operands[0].u32;
        if(target>=p->count)return NVM_FILE_FLOW_INVALID;
        NvmFileFlowFunction callee;
        if(!nvm_file_flow_function(p->declarations,target,&callee))return NVM_FILE_FLOW_INVALID;
        uint32_t borrowed=0;
        for(uint16_t i=0;i<callee.parameters;i++) {
            NvmFileFlowDeclaration type;
            if(!nvm_file_flow_declaration(p->declarations,target,i,&type))return NVM_FILE_FLOW_INVALID;
            borrowed+=type.mode!=0;
        }
        if(d->opcode==OP_CALL_REF) {
            if(d->operands[1].u16>=NVM_FILE_FLOW_REFERENCES)return NVM_FILE_FLOW_INVALID;
            if(borrowed!=1)return NVM_FILE_FLOW_UNRESOLVED;
        } else if(borrowed)return NVM_FILE_FLOW_INVALID;
        p->calls[function]|=UINT64_C(1)<<target;
        break;
    }
    case OP_FILE_SERVICE: {
        uint32_t ordinal=NVM_SERVICE_BINDING_COUNT;
        for(uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
            uint32_t import;
            if(nvm_file_nominal_import(p->declarations->nominal,i,&import) && import==d->operands[0].u32) {
                ordinal=i;break;
            }
        }
        const NlFilePlanMethod *method=nl_file_catalog_method(ordinal);
        if(!method)return NVM_FILE_FLOW_INVALID;
        uint16_t reference=d->operands[1].u16;
        if(method->input_mode==NL_FILE_INPUT_EXCLUSIVE) {
            if(reference>=NVM_FILE_FLOW_REFERENCES)return NVM_FILE_FLOW_INVALID;
        } else if(reference!=NVM_FILE_FLOW_NO_REFERENCE)return NVM_FILE_FLOW_INVALID;
        in->catalog_ordinal=ordinal;break;
    }
    case OP_AGG_PACK: case OP_UNION_CONSTRUCT: {
        uint8_t kind=NVM_V2_LAYOUT_UNION;uint32_t source=d->operands[0].u32;
        uint16_t variant=d->operands[1].u16,count=d->operands[2].u16;
        if(d->opcode==OP_AGG_PACK) {
            uint8_t aggregate=d->operands[0].u8;
            if(aggregate>AGG_TUPLE)return NVM_FILE_FLOW_INVALID;
            if(aggregate==AGG_TUPLE)return NVM_FILE_FLOW_UNRESOLVED;
            kind=aggregate==AGG_RECORD?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION;
            source=d->operands[1].u32;variant=d->operands[2].u16;count=d->operands[3].u16;
        }
        NvmFileNominalLayout layout;
        if(!nvm_file_nominal_source(p->declarations->nominal,kind,source,&layout))return NVM_FILE_FLOW_INVALID;
        if(layout.category!=NVM_FILE_CATEGORY_RECORD && layout.category!=NVM_FILE_CATEGORY_SCALAR_RESULT)
            return NVM_FILE_FLOW_UNRESOLVED;
        const NlFilePlanType *type=nl_file_catalog_type(layout.catalog_ordinal);
        bool record=layout.category==NVM_FILE_CATEGORY_RECORD;
        if(!type || (record?variant!=0:variant>=type->member_count) ||
           count!=(record?type->member_count:1))return NVM_FILE_FLOW_INVALID;
        in->catalog_ordinal=layout.catalog_ordinal;break;
    }
    default:break;
    }
    return NVM_FILE_FLOW_OK;
}
static bool file_code_index(const NvmFileCodePlan *p,uint32_t f,int64_t pc,uint16_t *out) {
    const NvmFileCodeFunction *fn=&p->functions[f];
    if(pc<0 || pc>=fn->code_length)return false;
    for(uint16_t i=0;i<fn->instruction_count;i++)
        if(p->instructions[p->starts[f]+i].byte_offset==(uint32_t)pc){*out=i;return true;}
    return false;
}
static NvmFileFlowStatus file_code_cfg(NvmFileCodePlan *p,uint32_t f,bool acyclic) {
    NvmFileCodeFunction *fn=&p->functions[f];
    uint16_t indegree[NVM_FILE_CODE_FUNCTION_INSTRUCTIONS]={0};
    bool selected[NVM_FILE_CODE_FUNCTION_INSTRUCTIONS]={0};
    for(uint16_t i=0;i<fn->instruction_count;i++) {
        NvmFileCodeInstruction *in=&p->instructions[p->starts[f]+i];
        uint8_t op=in->decoded.opcode;
        if(op!=OP_RET && op!=OP_JMP) {
            int64_t next=(int64_t)in->byte_offset+in->decoded.byte_length;
            if(!file_code_index(p,f,next,&in->successors[in->successor_count]))return NVM_FILE_FLOW_INVALID;
            in->successor_count++;
        }
        if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE || op==OP_FILE_RESULT_BRANCH) {
            int32_t displacement=in->decoded.operands[op==OP_FILE_RESULT_BRANCH?1:0].i32;
            int64_t target=(int64_t)in->byte_offset+displacement;
            if(!file_code_index(p,f,target,&in->successors[in->successor_count]))return NVM_FILE_FLOW_INVALID;
            in->successor_count++;
        }
        for(uint8_t e=0;e<in->successor_count;e++)indegree[in->successors[e]]++;
    }
    if(!acyclic)return NVM_FILE_FLOW_OK;
    for(uint16_t rank=0;rank<fn->instruction_count;rank++) {
        uint16_t found=fn->instruction_count;
        for(uint16_t i=0;i<fn->instruction_count;i++)if(!selected[i] && !indegree[i]){found=i;break;}
        if(found==fn->instruction_count)return NVM_FILE_FLOW_UNRESOLVED;
        selected[found]=true;p->instruction_order[p->starts[f]+rank]=found;
        const NvmFileCodeInstruction *in=&p->instructions[p->starts[f]+found];
        for(uint8_t e=0;e<in->successor_count;e++)indegree[in->successors[e]]--;
    }
    return NVM_FILE_FLOW_OK;
}
void nvm_file_code_free(NvmFileCodePlan *p) {
    if(p){nvm_file_flow_declarations_free(p->declarations);free(p->instructions);free(p->code);free(p);}
}
static NvmFileFlowStatus file_code_prepare_profile(const NvmModule *m,NvmFileCodePlan **out,bool acyclic,bool indirect) {
    if(!out)return NVM_FILE_FLOW_INVALID;
    uint32_t total;size_t budget;
    NvmFileFlowStatus status=file_code_preflight(m,&total,&budget);
    if(status!=NVM_FILE_FLOW_OK)return status;
    NvmFileFlowDeclarations *declarations=NULL;
    status=file_flow_declarations_mode(m,&declarations,indirect);if(status!=NVM_FILE_FLOW_OK)return status;
    const NvmFileFlowFunction *entry=&declarations->functions[m->header.entry_point];
    if(!scalar(entry->result.tag) || entry->result.global_index!=NVM_V2_NO_INDEX) {
        nvm_file_flow_declarations_free(declarations);return NVM_FILE_FLOW_UNRESOLVED;
    }
    for(uint16_t i=0;i<entry->parameters;i++) {
        NvmFileFlowDeclaration type=declarations->locals[declarations->offsets[m->header.entry_point]+i];
        if(type.mode || !scalar(type.tag) || type.global_index!=NVM_V2_NO_INDEX) {
            nvm_file_flow_declarations_free(declarations);return NVM_FILE_FLOW_UNRESOLVED;
        }
    }
    NvmFileCodePlan *p=calloc(1,sizeof *p);
    if(!p){nvm_file_flow_declarations_free(declarations);return NVM_FILE_FLOW_MEMORY;}
    p->declarations=declarations;p->count=m->function_count;p->code_size=m->code_size;
    p->instruction_count=total;p->storage_bound=budget;
    p->code=malloc(m->code_size);p->instructions=calloc(total,sizeof *p->instructions);
    if(!p->code || !p->instructions){status=NVM_FILE_FLOW_MEMORY;goto fail;}
    memcpy(p->code,m->code,m->code_size);
    uint32_t cursor=0;
    for(uint32_t f=0;f<p->count;f++) {
        NvmFileCodeFunction *fn=&p->functions[f];
        fn->code_offset=m->functions[f].code_offset;fn->code_length=m->functions[f].code_length;
        fn->declaration=declarations->functions[f];p->starts[f]=cursor;
        uint32_t pc=0;
        while(pc<fn->code_length) {
            NvmFileCodeInstruction *in=&p->instructions[cursor++];in->byte_offset=pc;
            in->catalog_ordinal=NVM_V2_NO_INDEX;
            in->successors[0]=in->successors[1]=NVM_FILE_CODE_NO_SUCCESSOR;
            uint32_t width=isa_decode(p->code+fn->code_offset+pc,fn->code_length-pc,&in->decoded);
            if(!width){status=NVM_FILE_FLOW_INVALID;goto fail;}
            status=file_code_operands(p,f,in,indirect);if(status!=NVM_FILE_FLOW_OK)goto fail;
            pc+=width;fn->instruction_count++;
        }
        status=file_code_cfg(p,f,acyclic);if(status!=NVM_FILE_FLOW_OK)goto fail;
    }
    uint64_t done=0;
    for(uint32_t rank=0;rank<p->count;rank++) {
        uint32_t found=p->count;
        for(uint32_t f=0;f<p->count;f++)if(!(done&(UINT64_C(1)<<f)) && !(p->calls[f]&~done)){found=f;break;}
        if(found==p->count){status=NVM_FILE_FLOW_UNRESOLVED;goto fail;}
        p->function_order[rank]=found;done|=UINT64_C(1)<<found;
    }
    *out=p;return NVM_FILE_FLOW_OK;
fail:
    nvm_file_code_free(p);return status;
}
static NvmFileFlowStatus file_code_prepare_mode(const NvmModule *m,NvmFileCodePlan **out,bool acyclic) {
    return file_code_prepare_profile(m,out,acyclic,false);
}
/* The old API always retains its original DAG refusal. Only the separate
 * non-admitting cyclic query calls the private syntactic preparation mode. */
NvmFileFlowStatus nvm_file_code_prepare(const NvmModule *m,NvmFileCodePlan **out) {
    return file_code_prepare_mode(m,out,true);
}
uint32_t nvm_file_code_function_count(const NvmFileCodePlan *p){return p?p->count:0;}
bool nvm_file_code_function(const NvmFileCodePlan *p,uint32_t f,NvmFileCodeFunction *out) {
    if(!p || !out || f>=p->count)return false;
    *out=p->functions[f];return true;
}
bool nvm_file_code_local(const NvmFileCodePlan *p,uint32_t f,uint16_t local,NvmFileFlowDeclaration *out) {
    return p && nvm_file_flow_declaration(p->declarations,f,local,out);
}
bool nvm_file_code_instruction(const NvmFileCodePlan *p,uint32_t f,uint16_t i,NvmFileCodeInstruction *out) {
    if(!p || !out || f>=p->count || i>=p->functions[f].instruction_count)return false;
    *out=p->instructions[p->starts[f]+i];return true;
}
bool nvm_file_code_instruction_order(const NvmFileCodePlan *p,uint32_t f,uint16_t rank,uint16_t *out) {
    if(!p || !out || f>=p->count || rank>=p->functions[f].instruction_count)return false;
    *out=p->instruction_order[p->starts[f]+rank];return true;
}
bool nvm_file_code_function_order(const NvmFileCodePlan *p,uint32_t rank,uint32_t *out) {
    if(!p || !out || rank>=p->count)return false;
    *out=p->function_order[rank];return true;
}
