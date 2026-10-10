/* I share exact checked decode/declaration storage, but publish only callable
 * target facts. Execution still requires complete ownership/hosted conjunction. */
typedef struct {
    SERVICE_TYPE(FlowDeclaration) type;
    uint64_t targets;
    uint8_t initialized; /* 1: uninitialized alternative, 2: initialized. */
    uint8_t arms; /* 1: Ok, 2: Error, 3: either; zero for non-Results. */
} FileIndirectValue;
typedef struct {
    bool reached;
    uint16_t stack_count;
    FileIndirectValue locals[SERVICE_CONST(FLOW_LOCALS)],stack[SERVICE_CONST(FLOW_STACK)];
} FileIndirectState;
typedef struct {
    FileIndirectState incoming[SERVICE_CONST(CODE_FUNCTION_INSTRUCTIONS)],working,edge;
    bool pending[SERVICE_CONST(CODE_FUNCTION_INSTRUCTIONS)];
} FileIndirectWorkspace;
struct SERVICE_TYPE(IndirectTargets) {
    SERVICE_TYPE(CodePlan) *plan;
    SERVICE_TYPE(IndirectSummary) summary;
    SERVICE_TYPE(IndirectCall) *calls;
    uint64_t graph[SERVICE_CONST(FLOW_FUNCTIONS)];
    /* Monotone same-module summaries. Empty is bottom, never a wildcard. */
    uint64_t parameters[SERVICE_CONST(FLOW_FUNCTIONS)][SERVICE_CONST(FLOW_LOCALS)];
    uint64_t results[SERVICE_CONST(FLOW_FUNCTIONS)];
};
static FileIndirectValue fi_value(SERVICE_TYPE(FlowDeclaration) t,uint8_t initialized) {
    FileIndirectValue v={0};v.type=t;v.initialized=initialized;
    if(t.category==NOMINAL_CATEGORY_OWNER_RESULT || t.category==SERVICE_CONST(CATEGORY_SCALAR_RESULT))v.arms=3;
    return v;
}
static bool fi_catalog(const SERVICE_TYPE(CodePlan) *p,uint32_t ordinal,SERVICE_TYPE(FlowDeclaration) *out) {
    SERVICE_TYPE(NominalLayout) l;
    if(!SERVICE_FN(nominal_type)(p->declarations->nominal,ordinal,&l))return false;
    *out=(SERVICE_TYPE(FlowDeclaration)){l.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
        0,l.global_index,l.catalog_ordinal,l.category};return true;
}
static bool fi_member(const SERVICE_TYPE(CodePlan) *p,uint32_t context,const char *id,SERVICE_TYPE(FlowDeclaration) *out) {
    if(!id){*out=scalar_type(TAG_VOID);return true;}
    if(!strcmp(id,"nsi:core/int")){*out=scalar_type(TAG_INT);return true;}
    if(!strcmp(id,"nsi:core/bool")){*out=scalar_type(TAG_BOOL);return true;}
    if(!strcmp(id,"nsi:core/string")){*out=scalar_type(TAG_STRING);return true;}
    for(uint32_t i=FLOW_TYPE_BASE(context);i<FLOW_TYPE_BASE(context)+FLOW_TYPE_COUNT(p->declarations->nominal,context);i++) {
        const NlServicePlanType *t=FLOW_CATALOG_TYPE(p->declarations->nominal,i);
        if(t && !strcmp(t->id,id))return fi_catalog(p,i,out);
    }
    return false;
}
static SERVICE_TYPE(FlowStatus) fi_push(FileIndirectState *s,FileIndirectValue v) {
    if(s->stack_count==SERVICE_CONST(FLOW_STACK))return SERVICE_CONST(FLOW_LIMIT);
    s->stack[s->stack_count++]=v;return SERVICE_CONST(FLOW_OK);
}
static SERVICE_TYPE(FlowStatus) fi_take(FileIndirectState *s,SERVICE_TYPE(FlowDeclaration) expected) {
    if(!s->stack_count)return SERVICE_CONST(FLOW_INVALID);
    FileIndirectValue *v=&s->stack[s->stack_count-1];
    if(v->initialized!=2 || !same(v->type,expected))return SERVICE_CONST(FLOW_INVALID);
    *v=(FileIndirectValue){0};s->stack_count--;return SERVICE_CONST(FLOW_OK);
}
static SERVICE_TYPE(FlowStatus) fi_join(FileIndirectState *to,const FileIndirectState *from,
                                  uint16_t locals,bool *changed) {
    *changed=false;
    if(!from->reached)return SERVICE_CONST(FLOW_OK);
    if(!to->reached){*to=*from;*changed=true;return SERVICE_CONST(FLOW_OK);}
    if(to->stack_count!=from->stack_count)return SERVICE_CONST(FLOW_INVALID);
    for(uint16_t group=0;group<2;group++) {
        uint16_t n=group?to->stack_count:locals;
        for(uint16_t i=0;i<n;i++) {
            FileIndirectValue *a=group?&to->stack[i]:&to->locals[i];
            const FileIndirectValue *b=group?&from->stack[i]:&from->locals[i];
            if(!same(a->type,b->type))return SERVICE_CONST(FLOW_UNRESOLVED);
            uint64_t targets=a->targets|b->targets;
            uint8_t init=(uint8_t)(a->initialized|b->initialized),arms=(uint8_t)(a->arms|b->arms);
            *changed|=targets!=a->targets || init!=a->initialized || arms!=a->arms;
            a->targets=targets;a->initialized=init;a->arms=arms;
        }
    }
    return SERVICE_CONST(FLOW_OK);
}
static bool fi_signature(const SERVICE_TYPE(CodePlan) *p,uint32_t a,uint32_t b) {
    const SERVICE_TYPE(FlowFunction) *x=&p->functions[a].declaration,*y=&p->functions[b].declaration;
    if(x->parameters!=y->parameters || x->result_count!=y->result_count || !same(x->result,y->result))return false;
    for(uint16_t i=0;i<x->parameters;i++)
        if(!same(p->declarations->locals[p->declarations->offsets[a]+i],
                 p->declarations->locals[p->declarations->offsets[b]+i]))return false;
    return true;
}
static SERVICE_TYPE(FlowStatus) fi_candidates(const SERVICE_TYPE(CodePlan) *p,const FileIndirectState *s,
                                         const SERVICE_TYPE(CodeInstruction) *in,uint64_t *bits,uint32_t *first) {
    const DecodedInstruction *d=&in->decoded;
    if(!s->stack_count)return SERVICE_CONST(FLOW_INVALID);
    const FileIndirectValue *v=&s->stack[s->stack_count-1];
    if(v->initialized!=2 || !same(v->type,scalar_type(TAG_FUNCTION)) || !v->targets)
        return SERVICE_CONST(FLOW_UNRESOLVED);
    if(p->count<64 && (v->targets>>p->count))return SERVICE_CONST(FLOW_INVALID);
    uint32_t selected=p->count;
    for(uint32_t f=0;f<p->count;f++)if(v->targets&(UINT64_C(1)<<f)) {
        const SERVICE_TYPE(FlowFunction) *fn=&p->functions[f].declaration;
        if(fn->parameters!=d->operands[0].u16 || fn->result_count!=d->operands[1].u16)
            return SERVICE_CONST(FLOW_INVALID);
        for(uint16_t i=0;i<fn->parameters;i++) {
            SERVICE_TYPE(FlowDeclaration) type=p->declarations->locals[p->declarations->offsets[f]+i];
            if(d->opcode==OP_FILE_CALL_INDIRECT_REFS) {
                if(type.mode ? in->call_references[i]>=SERVICE_CONST(FLOW_REFERENCES) :
                   in->call_references[i]!=SERVICE_CONST(FLOW_NO_REFERENCE))return SERVICE_CONST(FLOW_INVALID);
            } else if(type.mode)return SERVICE_CONST(FLOW_UNRESOLVED);
        }
        if(selected==p->count)selected=f;
        else if(!fi_signature(p,selected,f))return SERVICE_CONST(FLOW_UNRESOLVED);
    }
    *bits=v->targets;*first=selected;return SERVICE_CONST(FLOW_OK);
}
static void fi_grow(uint64_t *to,uint64_t bits,bool *changed) {
    *changed|=(*to|bits)!=*to;*to|=bits;
}
static SERVICE_TYPE(FlowStatus) fi_call(SERVICE_TYPE(IndirectTargets) *r,FileIndirectState *s,
                                  uint64_t candidates,uint32_t first,bool *changed) {
    const SERVICE_TYPE(CodePlan) *p=r->plan;
    const SERVICE_TYPE(FlowFunction) *fn=&p->functions[first].declaration;
    for(uint16_t i=fn->parameters;i>0;i--) {
        SERVICE_TYPE(FlowDeclaration) type=p->declarations->locals[p->declarations->offsets[first]+i-1];
        if(type.mode)continue; /* Reference-slot liveness belongs to full flow. */
        if(!s->stack_count)return SERVICE_CONST(FLOW_INVALID);
        FileIndirectValue value=s->stack[s->stack_count-1];
        SERVICE_TYPE(FlowStatus) status=fi_take(s,type);if(status!=SERVICE_CONST(FLOW_OK))return status;
        if(type.tag==TAG_FUNCTION)for(uint32_t f=0;f<p->count;f++)
            if(candidates&(UINT64_C(1)<<f))fi_grow(&r->parameters[f][i-1],value.targets,changed);
    }
    if(!fn->result_count)return SERVICE_CONST(FLOW_OK);
    FileIndirectValue result=fi_value(fn->result,2);
    if(fn->result.tag==TAG_FUNCTION)for(uint32_t f=0;f<p->count;f++)
        if(candidates&(UINT64_C(1)<<f))result.targets|=r->results[f];
    return fi_push(s,result);
}
static SERVICE_TYPE(FlowStatus) fi_scalar(FileIndirectState *s,uint8_t input,uint8_t output,uint16_t count,bool equality) {
    if(s->stack_count<count)return SERVICE_CONST(FLOW_INVALID);
    if(equality) {
        input=s->stack[s->stack_count-1].type.tag;
        if(input!=TAG_INT && input!=TAG_BOOL)return SERVICE_CONST(FLOW_UNRESOLVED);
    }
    for(uint16_t i=0;i<count;i++) {
        SERVICE_TYPE(FlowStatus) status=fi_take(s,scalar_type(input));if(status!=SERVICE_CONST(FLOW_OK))return status;
    }
    return output==TAG_COUNT?SERVICE_CONST(FLOW_OK):fi_push(s,fi_value(scalar_type(output),2));
}
static SERVICE_TYPE(FlowStatus) fi_transfer(SERVICE_TYPE(IndirectTargets) *r,uint32_t f,FileIndirectState *s,
                                      const SERVICE_TYPE(CodeInstruction) *in,bool *changed) {
    const SERVICE_TYPE(CodePlan) *p=r->plan;
    const DecodedInstruction *d=&in->decoded;uint16_t local=d->operands[0].u16;
    FileIndirectValue value;SERVICE_TYPE(FlowDeclaration) type;SERVICE_TYPE(FlowStatus) status;
    switch(d->opcode) {
    case OP_NOP:case OP_JMP:case OP_REGION_BEGIN:case OP_REGION_END:case OP_FILE_END_BORROW:
        return SERVICE_CONST(FLOW_OK);
    case OP_FILE_RESULT_BRANCH:
        value=s->locals[local];
        return value.initialized==2 && value.arms?SERVICE_CONST(FLOW_OK):SERVICE_CONST(FLOW_INVALID);
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE:
        value=s->locals[d->operands[1].u16];
        return value.initialized==2 && value.type.category==NOMINAL_CATEGORY_OWNER?SERVICE_CONST(FLOW_OK):SERVICE_CONST(FLOW_INVALID);
    case OP_PUSH_STR:return fi_push(s,fi_value(scalar_type(TAG_STRING),2));
    case OP_STR_EQ:return fi_scalar(s,TAG_STRING,TAG_BOOL,2,false);
    case OP_STR_LEN:return fi_scalar(s,TAG_STRING,TAG_INT,1,false);
    case OP_PUSH_I64:return fi_push(s,fi_value(scalar_type(TAG_INT),2));
    case OP_PUSH_BOOL:return fi_push(s,fi_value(scalar_type(TAG_BOOL),2));
    case OP_PUSH_VOID:return fi_push(s,fi_value(scalar_type(TAG_VOID),2));
    case OP_FUNCREF:
        value=fi_value(scalar_type(TAG_FUNCTION),2);value.targets=UINT64_C(1)<<d->operands[0].u32;
        return fi_push(s,value);
    case OP_LOAD_LOCAL:case OP_OWN_MOVE_LOCAL:
        value=s->locals[local];
        if(value.initialized!=2 || value.type.mode || owned(value.type)!=(d->opcode==OP_OWN_MOVE_LOCAL))
            return SERVICE_CONST(FLOW_INVALID);
        status=fi_push(s,value);
        if(status==SERVICE_CONST(FLOW_OK) && d->opcode==OP_OWN_MOVE_LOCAL)s->locals[local]=fi_value(value.type,1);
        return status;
    case OP_STORE_LOCAL:case OP_OWN_STORE_LOCAL:
        if(!s->stack_count)return SERVICE_CONST(FLOW_INVALID);
        value=s->stack[s->stack_count-1];
        /* I mirror the ownership check's explicit end of a copy lifetime.
         * Clearing a callable also forgets its target set. Owners and formal
         * loans still require their checked move/borrow operations. */
        if(d->opcode==OP_STORE_LOCAL && value.initialized==2 &&
           same(value.type,scalar_type(TAG_VOID)) && s->locals[local].type.tag!=TAG_VOID) {
            type=s->locals[local].type;
            if(type.mode || owned(type))return SERVICE_CONST(FLOW_INVALID);
            s->locals[local]=fi_value(type,1);
            return fi_take(s,scalar_type(TAG_VOID));
        }
        if(value.initialized!=2 || value.type.mode || !same(value.type,s->locals[local].type) ||
           owned(value.type)!=(d->opcode==OP_OWN_STORE_LOCAL) ||
           (d->opcode==OP_OWN_STORE_LOCAL && s->locals[local].initialized!=1))return SERVICE_CONST(FLOW_INVALID);
        s->locals[local]=value;return fi_take(s,value.type);
    case OP_DUP:
        if(!s->stack_count || owned(s->stack[s->stack_count-1].type))return SERVICE_CONST(FLOW_INVALID);
        return fi_push(s,s->stack[s->stack_count-1]);
    case OP_POP:case OP_FILE_DROP_STACK:
        if(!s->stack_count)return SERVICE_CONST(FLOW_INVALID);
        value=s->stack[s->stack_count-1];
        if(owned(value.type)!=(d->opcode==OP_FILE_DROP_STACK))return SERVICE_CONST(FLOW_INVALID);
        return fi_take(s,value.type);
    case OP_FILE_DROP_LOCAL:
        value=s->locals[local];
        if(value.initialized!=2 || !owned(value.type))return SERVICE_CONST(FLOW_INVALID);
        s->locals[local]=fi_value(value.type,1);return SERVICE_CONST(FLOW_OK);
    case OP_FILE_RESULT_TAKE: {
        value=s->locals[local];uint8_t arm=(uint8_t)(1u<<d->operands[1].u8);
        if(value.initialized!=2 || value.arms!=arm)return SERVICE_CONST(FLOW_UNRESOLVED);
        const NlServicePlanType *t=FLOW_CATALOG_TYPE(p->declarations->nominal,value.type.catalog_ordinal);
        if(!t || t->member_count!=2 || !fi_member(p,value.type.catalog_ordinal,t->members[d->operands[1].u8].type_id,&type))return SERVICE_CONST(FLOW_INVALID);
        status=fi_push(s,fi_value(type,2));
        if(status==SERVICE_CONST(FLOW_OK))s->locals[local]=fi_value(value.type,1);
        return status;
    }
    case OP_RET: {
        const SERVICE_TYPE(FlowFunction) *fn=&p->functions[f].declaration;
        if(s->stack_count!=fn->result_count)return SERVICE_CONST(FLOW_INVALID);
        if(fn->result_count && fn->result.tag==TAG_FUNCTION)
            fi_grow(&r->results[f],s->stack[0].targets,changed);
        return fn->result_count?fi_take(s,fn->result):SERVICE_CONST(FLOW_OK);
    }
    case OP_ASSERT:case OP_JMP_TRUE:case OP_JMP_FALSE:return fi_scalar(s,TAG_BOOL,TAG_COUNT,1,false);
    case OP_ADD:case OP_SUB:case OP_MUL:case OP_DIV:case OP_MOD:
    case OP_I64_ADD:case OP_I64_SUB:case OP_I64_MUL:case OP_I64_DIV_S:case OP_I64_REM_S:
        return fi_scalar(s,TAG_INT,TAG_INT,2,false);
    case OP_NEG:case OP_I64_NEG:return fi_scalar(s,TAG_INT,TAG_INT,1,false);
    case OP_EQ:case OP_NE:return fi_scalar(s,TAG_INT,TAG_BOOL,2,true);
    case OP_LT:case OP_LE:case OP_GT:case OP_GE:
    case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:
        return fi_scalar(s,TAG_INT,TAG_BOOL,2,false);
    case OP_AND:case OP_OR:return fi_scalar(s,TAG_BOOL,TAG_BOOL,2,false);
    case OP_NOT:return fi_scalar(s,TAG_BOOL,TAG_BOOL,1,false);
    case OP_CALL:case OP_CALL_REF:case OP_FILE_CALL_REFS:
        return fi_call(r,s,UINT64_C(1)<<d->operands[0].u32,d->operands[0].u32,changed);
    case OP_FILE_CALL_INDIRECT_REFS:case OP_CALL_INDIRECT: {
        uint64_t bits;uint32_t first;
        status=fi_candidates(p,s,in,&bits,&first);if(status!=SERVICE_CONST(FLOW_OK))return status;
        status=fi_take(s,scalar_type(TAG_FUNCTION));if(status!=SERVICE_CONST(FLOW_OK))return status;
        fi_grow(&r->graph[f],bits,changed);
        return fi_call(r,s,bits,first,changed);
    }
    case OP_FILE_SERVICE: {
        const NlServicePlanMethod *method=FLOW_CATALOG_METHOD(p->declarations->nominal,in->catalog_ordinal);
        if(!method || !method->param_count)return SERVICE_CONST(FLOW_INVALID);
        size_t first=method->input_mode==NL_SERVICE_INPUT_EXCLUSIVE?1u:0u;
        for(size_t i=method->param_count-1;i>first;i--) {
            if(!fi_member(p,FLOW_METHOD_OWNER(in->catalog_ordinal),method->params[i-1].type_id,&type))return SERVICE_CONST(FLOW_INVALID);
            status=fi_take(s,type);if(status!=SERVICE_CONST(FLOW_OK))return status;
        }
        if(!fi_member(p,FLOW_METHOD_OWNER(in->catalog_ordinal),method->params[method->param_count-1].type_id,&type))return SERVICE_CONST(FLOW_INVALID);
        return fi_push(s,fi_value(type,2));
    }
    case OP_AGG_PACK:case OP_UNION_CONSTRUCT: {
        const NlServicePlanType *t=FLOW_CATALOG_TYPE(p->declarations->nominal,in->catalog_ordinal);
        if(!t || !fi_catalog(p,in->catalog_ordinal,&type))return SERVICE_CONST(FLOW_INVALID);
        bool record=type.category==SERVICE_CONST(CATEGORY_RECORD);
        uint16_t variant=d->operands[d->opcode==OP_AGG_PACK?2:1].u16;
        size_t count=record?t->member_count:1;
        for(size_t i=count;i>0;i--) {
            SERVICE_TYPE(FlowDeclaration) member;
            if(!fi_member(p,in->catalog_ordinal,t->members[record?i-1:variant].type_id,&member))return SERVICE_CONST(FLOW_INVALID);
            status=fi_take(s,member);if(status!=SERVICE_CONST(FLOW_OK))return status;
        }
        value=fi_value(type,2);if(!record)value.arms=(uint8_t)(1u<<variant);
        return fi_push(s,value);
    }
    case OP_AGG_GET:case OP_UNION_FIELD:case OP_AGG_TAG:case OP_UNION_TAG: {
        if(!s->stack_count)return SERVICE_CONST(FLOW_INVALID);
        value=s->stack[s->stack_count-1];
        if(value.initialized!=2 || owned(value.type) || value.type.mode)return SERVICE_CONST(FLOW_INVALID);
        const NlServicePlanType *t=FLOW_CATALOG_TYPE(p->declarations->nominal,value.type.catalog_ordinal);
        if(!t)return SERVICE_CONST(FLOW_INVALID);
        bool tag=d->opcode==OP_AGG_TAG || d->opcode==OP_UNION_TAG;
        if(value.type.category==SERVICE_CONST(CATEGORY_RECORD) && d->opcode==OP_AGG_GET) {
            if(local>=t->member_count || !fi_member(p,value.type.catalog_ordinal,t->members[local].type_id,&type))return SERVICE_CONST(FLOW_INVALID);
        } else if(value.type.category==SERVICE_CONST(CATEGORY_SCALAR_RESULT)) {
            if(tag)type=scalar_type(TAG_INT);
            else {
                if(local || (value.arms!=1 && value.arms!=2))return SERVICE_CONST(FLOW_UNRESOLVED);
                if(!fi_member(p,value.type.catalog_ordinal,t->members[value.arms==2].type_id,&type))return SERVICE_CONST(FLOW_INVALID);
            }
        } else return SERVICE_CONST(FLOW_UNRESOLVED);
        s->stack[s->stack_count-1]=fi_value(type,2);return SERVICE_CONST(FLOW_OK);
    }
    default:return SERVICE_CONST(FLOW_UNRESOLVED);
    }
}
static SERVICE_TYPE(FlowStatus) fi_function(SERVICE_TYPE(IndirectTargets) *r,uint32_t f,
                                      FileIndirectWorkspace *w,uint32_t *failed_pc,bool publish,bool *changed) {
    SERVICE_TYPE(CodePlan) *p=r->plan;const SERVICE_TYPE(CodeFunction) *fn=&p->functions[f];
    memset(w,0,sizeof *w);w->incoming[0].reached=true;w->pending[0]=true;
    for(uint16_t l=0;l<fn->declaration.locals;l++) {
        FileIndirectValue value=fi_value(p->declarations->locals[p->declarations->offsets[f]+l],
                                            l<fn->declaration.parameters?2:1);
        if(l<fn->declaration.parameters && value.type.tag==TAG_FUNCTION)value.targets=r->parameters[f][l];
        w->incoming[0].locals[l]=value;
    }
    for(;;) {
        uint16_t i=0;while(i<fn->instruction_count && !w->pending[i])i++;
        if(i==fn->instruction_count)break;
        const SERVICE_TYPE(CodeInstruction) *in=&p->instructions[p->starts[f]+i];
        *failed_pc=in->byte_offset;
        if(r->summary.visits==SERVICE_CONST(INDIRECT_VISITS))return SERVICE_CONST(FLOW_LIMIT);
        r->summary.visits++;w->pending[i]=false;w->working=w->incoming[i];
        /* An as-yet empty callable summary cannot invent a return type or
         * successors. A later whole-module pass resumes from growing inputs. */
        if(!publish && isa_is_file_indirect_call(in->decoded.opcode) && w->working.stack_count) {
            const FileIndirectValue *v=&w->working.stack[w->working.stack_count-1];
            if(v->initialized==2 && same(v->type,scalar_type(TAG_FUNCTION)) && !v->targets)continue;
        }
        SERVICE_TYPE(FlowStatus) status=fi_transfer(r,f,&w->working,in,changed);
        if(status!=SERVICE_CONST(FLOW_OK))return status;
        for(uint8_t e=0;e<in->successor_count;e++) {
            w->edge=w->working;
            if(in->decoded.opcode==OP_FILE_RESULT_BRANCH) {
                FileIndirectValue *v=&w->edge.locals[in->decoded.operands[0].u16];
                v->arms&=(uint8_t)(1u<<e);
                if(!v->arms)continue;
            }
            bool edge_changed=false;uint16_t target=in->successors[e];
            status=fi_join(&w->incoming[target],&w->edge,fn->declaration.locals,&edge_changed);
            if(status!=SERVICE_CONST(FLOW_OK))return status;
            w->pending[target]|=edge_changed;
        }
    }
    if(!publish)return SERVICE_CONST(FLOW_OK);
    /* Only the completed module and CFG fixed points supply published calls. */
    for(uint16_t i=0;i<fn->instruction_count;i++) {
        const SERVICE_TYPE(CodeInstruction) *in=&p->instructions[p->starts[f]+i];
        if(!isa_is_file_indirect_call(in->decoded.opcode))continue;
        *failed_pc=in->byte_offset;
        if(!w->incoming[i].reached)return SERVICE_CONST(FLOW_UNRESOLVED);
        uint64_t bits;uint32_t first;
        SERVICE_TYPE(FlowStatus) status=fi_candidates(p,&w->incoming[i],in,&bits,&first);
        if(status!=SERVICE_CONST(FLOW_OK))return status;
        const SERVICE_TYPE(FlowFunction) *callee=&p->functions[first].declaration;
        r->calls[r->summary.calls++]=(SERVICE_TYPE(IndirectCall)){f,in->byte_offset,i,callee->parameters,
            bits,callee->result_count,callee->result};
        r->graph[f]|=bits;
    }
    return SERVICE_CONST(FLOW_OK);
}
static SERVICE_TYPE(IndirectResult) fi_result(SERVICE_TYPE(FlowStatus) s,uint32_t f,uint32_t pc) {
    SERVICE_TYPE(IndirectStatus) status=SERVICE_CONST(INDIRECT_INVALID);const char *message="I found invalid callable target facts.";
    switch(s) {
    case SERVICE_CONST(FLOW_OK):status=SERVICE_CONST(INDIRECT_DESCRIBED);message="I describe exact targets without File execution authority.";break;
    case SERVICE_CONST(FLOW_UNRESOLVED):status=SERVICE_CONST(INDIRECT_UNRESOLVED);message="I cannot resolve this complete callable target contract.";break;
    case SERVICE_CONST(FLOW_LIMIT):status=SERVICE_CONST(INDIRECT_LIMIT);message="I reached my bounded target-query limit.";break;
    case SERVICE_CONST(FLOW_MEMORY):status=SERVICE_CONST(INDIRECT_MEMORY);message="I could not allocate the target-query staging.";break;
    default:break;
    }
    return (SERVICE_TYPE(IndirectResult)){status,f,pc,message};
}
SERVICE_TYPE(IndirectResult) SERVICE_FN(indirect_targets)(const NvmModule *m,SERVICE_TYPE(IndirectTargets) **out) {
    uint32_t f=NVM_V2_NO_INDEX,pc=NVM_V2_NO_INDEX,total=0;size_t bytes=0;
    if(!out)return fi_result(SERVICE_CONST(FLOW_INVALID),f,pc);
    SERVICE_TYPE(FlowStatus) status=file_code_preflight(m,&total,&bytes);
    if(status!=SERVICE_CONST(FLOW_OK))return fi_result(status,f,pc);
    if(!file_code_budget(&bytes,1,sizeof(SERVICE_TYPE(IndirectTargets))) ||
       !file_code_budget(&bytes,total,sizeof(SERVICE_TYPE(IndirectCall))) ||
       !file_code_budget(&bytes,1,sizeof(FileIndirectWorkspace)) || bytes>SERVICE_CONST(INDIRECT_BYTES))
        return fi_result(SERVICE_CONST(FLOW_LIMIT),f,pc);
    SERVICE_TYPE(CodePlan) *plan=NULL;
    status=file_code_prepare_profile(m,&plan,false,true);
    if(status!=SERVICE_CONST(FLOW_OK))return fi_result(status,f,pc);
    SERVICE_TYPE(IndirectTargets) *r=calloc(1,sizeof *r);
    if(!r){SERVICE_FN(code_free)(plan);return fi_result(SERVICE_CONST(FLOW_MEMORY),f,pc);}
    r->plan=plan;r->summary=(SERVICE_TYPE(IndirectSummary)){plan->count,total,0,0,bytes};
    r->calls=calloc(total,sizeof *r->calls);
    FileIndirectWorkspace *w=calloc(1,sizeof *w);
    if(!r->calls || !w){status=SERVICE_CONST(FLOW_MEMORY);goto fail;}
    memcpy(r->graph,plan->calls,sizeof r->graph);
    bool changed;
    do {
        changed=false;
        for(f=0;f<plan->count;f++) {
            pc=NVM_V2_NO_INDEX;status=fi_function(r,f,w,&pc,false,&changed);
            if(status!=SERVICE_CONST(FLOW_OK))goto fail;
        }
    } while(changed);
    for(f=0;f<plan->count;f++) {
        pc=NVM_V2_NO_INDEX;status=fi_function(r,f,w,&pc,true,&changed);
        if(status!=SERVICE_CONST(FLOW_OK))goto fail;
    }
    if(changed){status=SERVICE_CONST(FLOW_INVALID);goto fail;}
    /* Transitive closure includes dead direct edges and every indirect target. */
    for(uint32_t via=0;via<plan->count;via++)
        for(uint32_t source=0;source<plan->count;source++)
            if(r->graph[source]&(UINT64_C(1)<<via))r->graph[source]|=r->graph[via];
    for(f=0;f<plan->count;f++)if(r->graph[f]&(UINT64_C(1)<<f)) {
        pc=NVM_V2_NO_INDEX;status=SERVICE_CONST(FLOW_UNRESOLVED);goto fail;
    }
    free(w);*out=r;return fi_result(SERVICE_CONST(FLOW_OK),NVM_V2_NO_INDEX,NVM_V2_NO_INDEX);
fail:
    free(w);SERVICE_FN(indirect_targets_free)(r);return fi_result(status,f,pc);
}
void SERVICE_FN(indirect_targets_free)(SERVICE_TYPE(IndirectTargets) *r) {
    if(r){SERVICE_FN(code_free)(r->plan);free(r->calls);free(r);}
}
bool SERVICE_FN(indirect_targets_summary)(const SERVICE_TYPE(IndirectTargets) *r,SERVICE_TYPE(IndirectSummary) *out) {
    if(!r || !out)return false;
    *out=r->summary;return true;
}
bool SERVICE_FN(indirect_targets_call)(const SERVICE_TYPE(IndirectTargets) *r,uint32_t index,SERVICE_TYPE(IndirectCall) *out) {
    if(!r || !out || index>=r->summary.calls)return false;
    *out=r->calls[index];return true;
}
bool SERVICE_FN(indirect_targets_parameter)(const SERVICE_TYPE(IndirectTargets) *r,uint32_t index,uint16_t parameter,SERVICE_TYPE(FlowDeclaration) *out) {
    if(!r || !out || index>=r->summary.calls || parameter>=r->calls[index].parameters)return false;
    uint32_t f=0;while(f<r->plan->count && !(r->calls[index].candidates&(UINT64_C(1)<<f)))f++;
    return SERVICE_FN(code_local)(r->plan,f,parameter,out);
}
