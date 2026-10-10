/* I share only private declaration ownership/actual extents with file_flow.c.
 * This plan never reaches a public admission or executing consumer. */
struct CODE_TYPE(Plan) {
    FLOW_TYPE(Declarations) *declarations;
    uint8_t *code;
    CODE_TYPE(Instruction) *instructions;
    uint32_t count, code_size, instruction_count;
    size_t storage_bound;
    CODE_TYPE(Function) functions[FLOW_CONST(FUNCTIONS)];
    uint32_t starts[FLOW_CONST(FUNCTIONS)], function_order[FLOW_CONST(FUNCTIONS)];
    uint16_t instruction_order[CODE_CONST(INSTRUCTIONS)];
    uint64_t calls[FLOW_CONST(FUNCTIONS)];
};
#ifndef CODE_STRINGS
#define CODE_STRINGS 0
#endif
static bool file_code_supported_mode(uint8_t op,bool indirect) {
    if(indirect && (op==OP_FUNCREF || isa_is_file_indirect_call(op)))return true;
    if((indirect || CODE_STRINGS) && (op==OP_PUSH_STR || op==OP_STR_EQ || op==OP_STR_LEN))return true;
    switch(op) {
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
    case OP_DUP: case OP_POP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
    case OP_REGION_BEGIN: case OP_REGION_END: case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE:
    case OP_FILE_SERVICE: case OP_FILE_RESULT_BRANCH: case OP_FILE_RESULT_TAKE:
    case OP_FILE_DROP_LOCAL: case OP_FILE_DROP_STACK: case OP_FILE_END_BORROW:
    case OP_CALL: case OP_CALL_REF: case OP_FILE_CALL_REFS: case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:
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
    if(*bytes>FLOW_CONST(BYTES) || add>FLOW_CONST(BYTES)-*bytes)return false;
    *bytes+=add;return true;
}
static FLOW_TYPE(Status) file_code_preflight(const NvmModule *m,uint32_t *instructions,
                                             size_t *budget) {
    if(!m || !m->functions || !m->function_count || !m->code || !m->code_size)
        return FLOW_CONST(INVALID);
    if(m->function_count>FLOW_CONST(FUNCTIONS) || m->code_size>CODE_CONST(BYTES))
        return FLOW_CONST(LIMIT);
    if(m->header.entry_point>=m->function_count)return FLOW_CONST(INVALID);
    uint32_t total=0,covered=0;size_t locals=0;
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];
        if(!fn->code_length || fn->code_offset>m->code_size ||
           fn->code_length>m->code_size-fn->code_offset || fn->arity>fn->local_count ||
           fn->name_idx>=m->string_count || fn->result_tag>=TAG_COUNT ||
           (fn->result_count==0)!=(fn->result_tag==TAG_VOID))
            return FLOW_CONST(INVALID);
        if(fn->local_count>FLOW_CONST(LOCALS))return FLOW_CONST(LIMIT);
        locals+=fn->local_count;
        for(uint32_t prior=0;prior<f;prior++) {
            const NvmFunctionEntry *p=&m->functions[prior];
            if(fn->code_offset<p->code_offset+p->code_length &&
               p->code_offset<fn->code_offset+fn->code_length)return FLOW_CONST(INVALID);
        }
        covered+=fn->code_length;uint32_t pc=0,count=0;
        while(pc<fn->code_length) {
            DecodedInstruction decoded={0};
            uint32_t width=isa_decode(m->code+fn->code_offset+pc,fn->code_length-pc,&decoded);
            if(!width)return FLOW_CONST(INVALID);
            if(++count>CODE_CONST(FUNCTION_INSTRUCTIONS) || ++total>CODE_CONST(INSTRUCTIONS))
                return FLOW_CONST(LIMIT);
            pc+=width;
        }
    }
    if(covered!=m->code_size)return FLOW_CONST(UNRESOLVED);
    size_t bytes=0,nominal_bound=0;
    if(!NOMINAL_FN(storage_bound)(NOMINAL_MAX_LAYOUTS,&nominal_bound) ||
       !file_code_budget(&bytes,1,sizeof(FLOW_TYPE(Declarations))) ||
       !file_code_budget(&bytes,locals,sizeof(FLOW_TYPE(Declaration))) ||
       !file_code_budget(&bytes,1,nominal_bound) ||
       !file_code_budget(&bytes,1,sizeof(CODE_TYPE(Plan))) ||
       !file_code_budget(&bytes,m->code_size,1) ||
       !file_code_budget(&bytes,total,sizeof(CODE_TYPE(Instruction))) ||
       /* Fixed CFG indegrees/selection and transient decoded/preflight facts. */
       !file_code_budget(&bytes,CODE_CONST(FUNCTION_INSTRUCTIONS),sizeof(uint16_t)+sizeof(bool)) ||
       !file_code_budget(&bytes,1,sizeof(DecodedInstruction)))return FLOW_CONST(LIMIT);
    *instructions=total;*budget=bytes;return FLOW_CONST(OK);
}
static FLOW_TYPE(Status) file_code_operands(CODE_TYPE(Plan) *p,uint32_t function,
                                            CODE_TYPE(Instruction) *in,bool indirect,const NvmModule *module) {
    const DecodedInstruction *d=&in->decoded;
    const FLOW_TYPE(Function) *fn=&p->functions[function].declaration;
    if(!(indirect?file_code_supported_mode(d->opcode,true):file_code_supported(d->opcode)))return FLOW_CONST(UNRESOLVED);
    switch(d->opcode) {
    case OP_PUSH_STR:
        if(d->operands[0].u32>=module->string_count || !nvm_get_string(module,d->operands[0].u32))return FLOW_CONST(INVALID);
        return nvm_get_string_len(module,d->operands[0].u32)>1024u*1024u?FLOW_CONST(LIMIT):FLOW_CONST(OK);
    case OP_FUNCREF:
        return d->operands[0].u32<p->count?FLOW_CONST(OK):FLOW_CONST(INVALID);
    case OP_FILE_CALL_INDIRECT_REFS: {
        uint16_t parameters=d->operands[0].u16;uint32_t index=d->operands[2].u32;
        if(parameters>FLOW_CONST(LOCALS) || d->operands[1].u16>1)return FLOW_CONST(INVALID);
        const char *map=nvm_get_string(module,index);
        if(!map || nvm_get_string_len(module,index)!=2u*parameters)return FLOW_CONST(INVALID);
        bool borrowed=false;
        for(uint16_t i=0;i<parameters;i++) {
            uint16_t ref=(uint16_t)(uint8_t)map[2u*i]|((uint16_t)(uint8_t)map[2u*i+1u]<<8);
            if(ref!=FLOW_CONST(NO_REFERENCE) && ref>=FLOW_CONST(REFERENCES))return FLOW_CONST(INVALID);
            borrowed|=ref!=FLOW_CONST(NO_REFERENCE);in->call_references[i]=ref;
        }
        if(!borrowed)return FLOW_CONST(INVALID);
        break;
    }
    case OP_CALL_INDIRECT:
        if(d->operands[0].u16>FLOW_CONST(LOCALS) || d->operands[1].u16>1)
            return FLOW_CONST(INVALID);
        break;
    case OP_PUSH_BOOL:return d->operands[0].u8<=1?FLOW_CONST(OK):FLOW_CONST(INVALID);
    case OP_LOAD_LOCAL: case OP_STORE_LOCAL: case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
    case OP_FILE_DROP_LOCAL: case OP_FILE_RESULT_BRANCH: case OP_FILE_RESULT_TAKE:
        if(d->operands[0].u16>=fn->locals)return FLOW_CONST(INVALID);
        if(d->opcode==OP_FILE_RESULT_TAKE && d->operands[1].u8>1)return FLOW_CONST(INVALID);
        break;
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE:
        if(d->operands[0].u16>=FLOW_CONST(REFERENCES) || d->operands[1].u16>=fn->locals)
            return FLOW_CONST(INVALID);
        break;
    case OP_FILE_END_BORROW:
        if(d->operands[0].u16>=FLOW_CONST(REFERENCES))return FLOW_CONST(INVALID);
        break;
    case OP_CALL: case OP_CALL_REF: case OP_FILE_CALL_REFS: {
        uint32_t target=d->operands[0].u32;
        if(target>=p->count)return FLOW_CONST(INVALID);
        FLOW_TYPE(Function) callee;
        if(!FLOW_FN(function)(p->declarations,target,&callee))return FLOW_CONST(INVALID);
        const char *map=NULL;
        if(d->opcode==OP_FILE_CALL_REFS) {
            map=nvm_get_string(module,d->operands[1].u32);
            if(!map || nvm_get_string_len(module,d->operands[1].u32)!=2u*callee.parameters)
                return FLOW_CONST(INVALID);
        }
        uint32_t borrowed=0;
        for(uint16_t i=0;i<callee.parameters;i++) {
            FLOW_TYPE(Declaration) type;
            if(!FLOW_FN(declaration)(p->declarations,target,i,&type))return FLOW_CONST(INVALID);
            borrowed+=type.mode!=0;
            uint16_t reference=FLOW_CONST(NO_REFERENCE);
            if(map)reference=(uint16_t)(uint8_t)map[2u*i]|((uint16_t)(uint8_t)map[2u*i+1u]<<8);
            else if(type.mode && d->opcode==OP_CALL_REF)reference=d->operands[1].u16;
            if(type.mode ? reference>=FLOW_CONST(REFERENCES) : reference!=FLOW_CONST(NO_REFERENCE))
                return FLOW_CONST(INVALID);
            in->call_references[i]=reference;
        }
        if(d->opcode==OP_CALL_REF) {
            if(d->operands[1].u16>=FLOW_CONST(REFERENCES))return FLOW_CONST(INVALID);
            if(borrowed!=1)return FLOW_CONST(UNRESOLVED);
        } else if(d->opcode==OP_FILE_CALL_REFS) {
            if(!borrowed)return FLOW_CONST(INVALID);
        } else if(borrowed)return FLOW_CONST(INVALID);
        p->calls[function]|=UINT64_C(1)<<target;
        break;
    }
    case OP_FILE_SERVICE: {
        uint32_t ordinal=FLOW_IMPORT_COUNT(p->declarations->nominal);
        for(uint32_t i=0;i<FLOW_IMPORT_COUNT(p->declarations->nominal);i++) {
            uint32_t import;
            if(NOMINAL_FN(import)(p->declarations->nominal,i,&import) && import==d->operands[0].u32) {
                ordinal=i;break;
            }
        }
        const NlServicePlanMethod *method=FLOW_CATALOG_METHOD(p->declarations->nominal,ordinal);
        if(!method)return FLOW_CONST(INVALID);
        uint16_t reference=d->operands[1].u16;
        if(method->input_mode==NL_SERVICE_INPUT_EXCLUSIVE) {
            if(reference>=FLOW_CONST(REFERENCES))return FLOW_CONST(INVALID);
        } else if(reference!=FLOW_CONST(NO_REFERENCE))return FLOW_CONST(INVALID);
        in->catalog_ordinal=ordinal;break;
    }
    case OP_AGG_PACK: case OP_UNION_CONSTRUCT: {
        uint8_t kind=NVM_V2_LAYOUT_UNION;uint32_t source=d->operands[0].u32;
        uint16_t variant=d->operands[1].u16,count=d->operands[2].u16;
        if(d->opcode==OP_AGG_PACK) {
            uint8_t aggregate=d->operands[0].u8;
            if(aggregate>AGG_TUPLE)return FLOW_CONST(INVALID);
            if(aggregate==AGG_TUPLE)return FLOW_CONST(UNRESOLVED);
            kind=aggregate==AGG_RECORD?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION;
            source=d->operands[1].u32;variant=d->operands[2].u16;count=d->operands[3].u16;
        }
        NominalLayout layout;
        if(!NOMINAL_FN(source)(p->declarations->nominal,kind,source,&layout))return FLOW_CONST(INVALID);
        if(layout.category!=NOMINAL_CATEGORY_RECORD && layout.category!=NOMINAL_CATEGORY_SCALAR_RESULT)
            return FLOW_CONST(UNRESOLVED);
        const NlServicePlanType *type=FLOW_CATALOG_TYPE(p->declarations->nominal,layout.catalog_ordinal);
        bool record=layout.category==NOMINAL_CATEGORY_RECORD;
        if(!type || (record?variant!=0:variant>=type->member_count) ||
           count!=(record?type->member_count:1))return FLOW_CONST(INVALID);
        in->catalog_ordinal=layout.catalog_ordinal;break;
    }
    default:break;
    }
    return FLOW_CONST(OK);
}
static bool file_code_index(const CODE_TYPE(Plan) *p,uint32_t f,int64_t pc,uint16_t *out) {
    const CODE_TYPE(Function) *fn=&p->functions[f];
    if(pc<0 || pc>=fn->code_length)return false;
    for(uint16_t i=0;i<fn->instruction_count;i++)
        if(p->instructions[p->starts[f]+i].byte_offset==(uint32_t)pc){*out=i;return true;}
    return false;
}
static FLOW_TYPE(Status) file_code_cfg(CODE_TYPE(Plan) *p,uint32_t f,bool acyclic) {
    CODE_TYPE(Function) *fn=&p->functions[f];
    uint16_t indegree[CODE_CONST(FUNCTION_INSTRUCTIONS)]={0};
    bool selected[CODE_CONST(FUNCTION_INSTRUCTIONS)]={0};
    for(uint16_t i=0;i<fn->instruction_count;i++) {
        CODE_TYPE(Instruction) *in=&p->instructions[p->starts[f]+i];
        uint8_t op=in->decoded.opcode;
        if(op!=OP_RET && op!=OP_JMP) {
            int64_t next=(int64_t)in->byte_offset+in->decoded.byte_length;
            if(!file_code_index(p,f,next,&in->successors[in->successor_count]))return FLOW_CONST(INVALID);
            in->successor_count++;
        }
        if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE || op==OP_FILE_RESULT_BRANCH) {
            int32_t displacement=in->decoded.operands[op==OP_FILE_RESULT_BRANCH?1:0].i32;
            int64_t target=(int64_t)in->byte_offset+displacement;
            if(!file_code_index(p,f,target,&in->successors[in->successor_count]))return FLOW_CONST(INVALID);
            in->successor_count++;
        }
        for(uint8_t e=0;e<in->successor_count;e++)indegree[in->successors[e]]++;
    }
    if(!acyclic)return FLOW_CONST(OK);
    for(uint16_t rank=0;rank<fn->instruction_count;rank++) {
        uint16_t found=fn->instruction_count;
        for(uint16_t i=0;i<fn->instruction_count;i++)if(!selected[i] && !indegree[i]){found=i;break;}
        if(found==fn->instruction_count)return FLOW_CONST(UNRESOLVED);
        selected[found]=true;p->instruction_order[p->starts[f]+rank]=found;
        const CODE_TYPE(Instruction) *in=&p->instructions[p->starts[f]+found];
        for(uint8_t e=0;e<in->successor_count;e++)indegree[in->successors[e]]--;
    }
    return FLOW_CONST(OK);
}
void CODE_FN(free)(CODE_TYPE(Plan) *p) {
    if(p){FLOW_FN(declarations_free)(p->declarations);free(p->instructions);free(p->code);free(p);}
}
static FLOW_TYPE(Status) file_code_prepare_profile(const NvmModule *m,CODE_TYPE(Plan) **out,bool acyclic,bool indirect) {
    if(!out)return FLOW_CONST(INVALID);
    uint32_t total;size_t budget;
    FLOW_TYPE(Status) status=file_code_preflight(m,&total,&budget);
    if(status!=FLOW_CONST(OK))return status;
    FLOW_TYPE(Declarations) *declarations=NULL;
    status=file_flow_declarations_mode(m,&declarations,indirect);if(status!=FLOW_CONST(OK))return status;
    const FLOW_TYPE(Function) *entry=&declarations->functions[m->header.entry_point];
    if(entry->result.tag==TAG_STRING || !scalar(entry->result.tag) || entry->result.global_index!=NVM_V2_NO_INDEX) {
        FLOW_FN(declarations_free)(declarations);return FLOW_CONST(UNRESOLVED);
    }
    for(uint16_t i=0;i<entry->parameters;i++) {
        FLOW_TYPE(Declaration) type=declarations->locals[declarations->offsets[m->header.entry_point]+i];
        if(type.mode || !scalar(type.tag) || type.global_index!=NVM_V2_NO_INDEX) {
            FLOW_FN(declarations_free)(declarations);return FLOW_CONST(UNRESOLVED);
        }
    }
    CODE_TYPE(Plan) *p=calloc(1,sizeof *p);
    if(!p){FLOW_FN(declarations_free)(declarations);return FLOW_CONST(MEMORY);}
    p->declarations=declarations;p->count=m->function_count;p->code_size=m->code_size;
    p->instruction_count=total;p->storage_bound=budget;
    p->code=malloc(m->code_size);p->instructions=calloc(total,sizeof *p->instructions);
    if(!p->code || !p->instructions){status=FLOW_CONST(MEMORY);goto fail;}
    memcpy(p->code,m->code,m->code_size);
    uint32_t cursor=0;
    for(uint32_t f=0;f<p->count;f++) {
        CODE_TYPE(Function) *fn=&p->functions[f];
        fn->code_offset=m->functions[f].code_offset;fn->code_length=m->functions[f].code_length;
        fn->declaration=declarations->functions[f];p->starts[f]=cursor;
        uint32_t pc=0;
        while(pc<fn->code_length) {
            CODE_TYPE(Instruction) *in=&p->instructions[cursor++];in->byte_offset=pc;
            in->catalog_ordinal=NVM_V2_NO_INDEX;
            in->successors[0]=in->successors[1]=CODE_CONST(NO_SUCCESSOR);
            uint32_t width=isa_decode(p->code+fn->code_offset+pc,fn->code_length-pc,&in->decoded);
            if(!width){status=FLOW_CONST(INVALID);goto fail;}
            status=file_code_operands(p,f,in,indirect,m);if(status!=FLOW_CONST(OK))goto fail;
            pc+=width;fn->instruction_count++;
        }
        status=file_code_cfg(p,f,acyclic);if(status!=FLOW_CONST(OK))goto fail;
    }
    uint64_t done=0;
    for(uint32_t rank=0;rank<p->count;rank++) {
        uint32_t found=p->count;
        for(uint32_t f=0;f<p->count;f++)if(!(done&(UINT64_C(1)<<f)) && !(p->calls[f]&~done)){found=f;break;}
        if(found==p->count){status=FLOW_CONST(UNRESOLVED);goto fail;}
        p->function_order[rank]=found;done|=UINT64_C(1)<<found;
    }
    *out=p;return FLOW_CONST(OK);
fail:
    CODE_FN(free)(p);return status;
}
static FLOW_TYPE(Status) file_code_prepare_mode(const NvmModule *m,CODE_TYPE(Plan) **out,bool acyclic) {
    return file_code_prepare_profile(m,out,acyclic,false);
}
/* The old API always retains its original DAG refusal. Only the separate
 * non-admitting cyclic query calls the private syntactic preparation mode. */
FLOW_TYPE(Status) CODE_FN(prepare)(const NvmModule *m,CODE_TYPE(Plan) **out) {
    return file_code_prepare_mode(m,out,true);
}
uint32_t CODE_FN(function_count)(const CODE_TYPE(Plan) *p){return p?p->count:0;}
bool CODE_FN(function)(const CODE_TYPE(Plan) *p,uint32_t f,CODE_TYPE(Function) *out) {
    if(!p || !out || f>=p->count)return false;
    *out=p->functions[f];return true;
}
bool CODE_FN(local)(const CODE_TYPE(Plan) *p,uint32_t f,uint16_t local,FLOW_TYPE(Declaration) *out) {
    return p && FLOW_FN(declaration)(p->declarations,f,local,out);
}
bool CODE_FN(instruction)(const CODE_TYPE(Plan) *p,uint32_t f,uint16_t i,CODE_TYPE(Instruction) *out) {
    if(!p || !out || f>=p->count || i>=p->functions[f].instruction_count)return false;
    *out=p->instructions[p->starts[f]+i];return true;
}
bool CODE_FN(instruction_order)(const CODE_TYPE(Plan) *p,uint32_t f,uint16_t rank,uint16_t *out) {
    if(!p || !out || f>=p->count || rank>=p->functions[f].instruction_count)return false;
    *out=p->instruction_order[p->starts[f]+rank];return true;
}
bool CODE_FN(function_order)(const CODE_TYPE(Plan) *p,uint32_t rank,uint32_t *out) {
    if(!p || !out || rank>=p->count)return false;
    *out=p->function_order[rank];return true;
}
