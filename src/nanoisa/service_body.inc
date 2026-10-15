/* Same translation unit as qualified logical states: actual sizeof accounting,
 * no copied state layout and no public verifier/admission recursion. */
struct BODY_TYPE(Report) {
    CODE_TYPE(Plan) *plan;
    BODY_TYPE(Instruction) *facts;
    bool checked[FLOW_CONST(FUNCTIONS)];
};
typedef struct {
    size_t bytes;
    FLOW_TYPE(State) *incoming[CODE_CONST(FUNCTION_INSTRUCTIONS)];
    FLOW_TYPE(State) *working, *edges[2];
} FileBodyWorkspace;
static void file_body_release(FileBodyWorkspace *w,FLOW_TYPE(State) **s) {
    if(*s){FLOW_FN(state_free)(*s);*s=NULL;w->bytes-=sizeof(FLOW_TYPE(State));}
}
static bool file_body_room(const FileBodyWorkspace *w,size_t count) {
    size_t bytes=w->bytes;return file_code_budget(&bytes,count,sizeof(FLOW_TYPE(State)));
}
static FLOW_TYPE(Status) file_body_clone(FileBodyWorkspace *w,const FLOW_TYPE(State) *s,
                                         FLOW_TYPE(State) **out) {
    if(!file_body_room(w,1))return FLOW_CONST(LIMIT);
    FLOW_TYPE(Status) status=FLOW_FN(clone)(s,out);
    if(status==FLOW_CONST(OK))w->bytes+=sizeof(FLOW_TYPE(State));
    return status;
}
static FLOW_TYPE(Status) file_body_scalar(FLOW_TYPE(State) *s,uint8_t input,uint8_t output,
                                          uint16_t count,bool equality) {
    if(s->stack_count<count)return FLOW_CONST(INVALID);
    for(uint16_t i=0;i<count;i++) {
        FLOW_TYPE(Value) v=s->stack[s->stack_count-count+i];
        if(!v.initialized || v.owner || !same(v.type,scalar_type(v.type.tag)))return FLOW_CONST(INVALID);
        if(equality) {
            if(v.type.tag!=TAG_INT && v.type.tag!=TAG_BOOL)return FLOW_CONST(UNRESOLVED);
            if(i && v.type.tag!=s->stack[s->stack_count-count].type.tag)return FLOW_CONST(UNRESOLVED);
        } else if(v.type.tag!=input)return FLOW_CONST(INVALID);
    }
    for(uint16_t i=0;i<count;i++)pop(s);
    if(output!=TAG_COUNT)s->stack[s->stack_count++]=initial_value(scalar_type(output),true,0);
    return FLOW_CONST(OK);
}
static FLOW_TYPE(Status) file_body_projection(FLOW_TYPE(State) *s,uint16_t field,bool tag,bool record) {
    if(!s->stack_count)return FLOW_CONST(INVALID);
    FLOW_TYPE(Value) v=s->stack[s->stack_count-1];
    if(!v.initialized || v.owner || v.type.mode)return FLOW_CONST(INVALID);
    if(record && !tag && v.type.category==NOMINAL_CATEGORY_RECORD)return FLOW_FN(field)(s,field);
    if(v.type.category!=NOMINAL_CATEGORY_SCALAR_RESULT)return FLOW_CONST(UNRESOLVED);
    FLOW_TYPE(Declaration) result=scalar_type(TAG_INT);
    if(!tag) {
        if(field)return FLOW_CONST(INVALID);
        if(v.arm!=FLOW_CONST(ARM_OK) && v.arm!=FLOW_CONST(ARM_ERROR))return FLOW_CONST(UNRESOLVED);
        const NlServicePlanType *type=FLOW_CATALOG_TYPE(s->declarations->nominal,v.type.catalog_ordinal);
        if(!type || type->member_count!=2 ||
           !member_type(s,v.type.catalog_ordinal,type->members[v.arm==FLOW_CONST(ARM_ERROR)].type_id,&result) || owned(result))
            return FLOW_CONST(INVALID);
    }
    s->stack[s->stack_count-1]=initial_value(result,true,0);return FLOW_CONST(OK);
}
static FLOW_TYPE(Status) file_body_transfer(BODY_TYPE(Report) *r,FLOW_TYPE(State) *s,
                                            const CODE_TYPE(Instruction) *in,BODY_TYPE(Instruction) *fact) {
    const DecodedInstruction *d=&in->decoded;uint16_t local=d->operands[0].u16;
    FLOW_TYPE(Status) status;
    switch(d->opcode) {
    case OP_NOP: case OP_JMP: case OP_FILE_RESULT_BRANCH:return FLOW_CONST(OK);
    case OP_PUSH_STR:return FLOW_FN(push_scalar)(s,TAG_STRING);
    case OP_STR_EQ:return file_body_scalar(s,TAG_STRING,TAG_BOOL,2,false);
    case OP_STR_LEN:return file_body_scalar(s,TAG_STRING,TAG_INT,1,false);
    case OP_PUSH_I64:return FLOW_FN(push_scalar)(s,TAG_INT);
    case OP_PUSH_BOOL:return FLOW_FN(push_scalar)(s,TAG_BOOL);
    case OP_PUSH_VOID:return FLOW_FN(push_scalar)(s,TAG_VOID);
    case OP_DUP:return FLOW_FN(dup)(s);
    case OP_POP:return FLOW_FN(pop)(s);
    case OP_LOAD_LOCAL:return FLOW_FN(load)(s,local);
    case OP_STORE_LOCAL:return FLOW_FN(store)(s,local);
    case OP_OWN_MOVE_LOCAL:return FLOW_FN(take)(s,local);
    case OP_OWN_STORE_LOCAL:return FLOW_FN(put)(s,local);
    case OP_REGION_BEGIN:return FLOW_FN(region_begin)(s);
    case OP_REGION_END:return FLOW_FN(region_end)(s);
    case OP_BORROW_LOCAL_SHARED:return FLOW_FN(borrow_shared)(s,d->operands[1].u16,local);
    case OP_BORROW_LOCAL_EXCLUSIVE:return FLOW_FN(borrow)(s,d->operands[1].u16,local);
    case OP_FILE_END_BORROW:return FLOW_FN(end_borrow)(s,local);
    case OP_FILE_RESULT_TAKE:return FLOW_FN(take_result)(s,local,
        d->operands[1].u8?FLOW_CONST(ARM_ERROR):FLOW_CONST(ARM_OK));
    case OP_FILE_DROP_LOCAL:
        fact->cleanup=BODY_CONST(CLEANUP_DROP_LOCAL);fact->cleanup_local=local;
        return FLOW_FN(drop_local)(s,local);
    case OP_FILE_DROP_STACK:
        fact->cleanup=BODY_CONST(CLEANUP_DROP_STACK);return FLOW_FN(drop_stack)(s);
    case OP_RET:
        status=FLOW_FN(can_exit)(s);fact->exit_checked=status==FLOW_CONST(OK);
        fact->cleanup=BODY_CONST(CLEANUP_RETURN);return status;
    case OP_ASSERT:
        fact->cleanup=BODY_CONST(CLEANUP_ASSERT);
        return file_body_scalar(s,TAG_BOOL,TAG_COUNT,1,false);
    case OP_JMP_TRUE: case OP_JMP_FALSE:return file_body_scalar(s,TAG_BOOL,TAG_COUNT,1,false);
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD:
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S: case OP_I64_REM_S:
        return file_body_scalar(s,TAG_INT,TAG_INT,2,false);
    case OP_NEG: case OP_I64_NEG:return file_body_scalar(s,TAG_INT,TAG_INT,1,false);
    case OP_EQ: case OP_NE:return file_body_scalar(s,TAG_INT,TAG_BOOL,2,true);
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S:
    case OP_I64_GT_S: case OP_I64_GE_S:return file_body_scalar(s,TAG_INT,TAG_BOOL,2,false);
    case OP_AND: case OP_OR:return file_body_scalar(s,TAG_BOOL,TAG_BOOL,2,false);
    case OP_NOT:return file_body_scalar(s,TAG_BOOL,TAG_BOOL,1,false);
    case OP_AGG_PACK: case OP_UNION_CONSTRUCT:return FLOW_FN(construct)(s,in->catalog_ordinal,
        d->operands[d->opcode==OP_AGG_PACK?2:1].u16);
    case OP_AGG_GET: case OP_UNION_FIELD:
        return file_body_projection(s,local,false,d->opcode==OP_AGG_GET);
    case OP_AGG_TAG: case OP_UNION_TAG:return file_body_projection(s,0,true,false);
    case OP_FILE_SERVICE:
        fact->cleanup=BODY_CONST(CLEANUP_SERVICE);
        status=FLOW_FN(service)(s,in->byte_offset,d->operands[0].u32,d->operands[1].u16);break;
    case OP_CALL: case OP_CALL_REF:case OP_FILE_CALL_REFS: {
        uint32_t target=d->operands[0].u32;
        if(target>=r->plan->count || !r->checked[target])return FLOW_CONST(UNRESOLVED);
        const FLOW_TYPE(Function) *callee=&r->plan->functions[target].declaration;
        uint16_t references[FLOW_CONST(LOCALS)];
        for(uint16_t i=0;i<callee->parameters;i++) {
            FLOW_TYPE(Declaration) type=s->declarations->locals[s->declarations->offsets[target]+i];
            references[i]=type.mode?in->call_references[i]:FLOW_CONST(NO_REFERENCE);
        }
        fact->cleanup=BODY_CONST(CLEANUP_CALL);
        status=FLOW_FN(call)(s,in->byte_offset,target,references,callee->parameters);
        if(status==FLOW_CONST(OK))fact->discharged_checks=FLOW_CONST(CHECK_CALLEE);
        break;
    }
    default:return FLOW_CONST(UNRESOLVED);
    }
    if(status!=FLOW_CONST(OK))return status;
    for(uint16_t i=0;i<s->obligation_count;i++)if(s->obligations[i].site==in->byte_offset) {
        fact->has_obligation=true;fact->obligation=s->obligations[i];
        fact->pending_checks|=fact->obligation.checks & ~fact->discharged_checks;
        return FLOW_CONST(OK);
    }
    return FLOW_CONST(INVALID);
}
static FLOW_TYPE(Status) file_body_function(BODY_TYPE(Report) *r,uint32_t f,FileBodyWorkspace *w) {
    CODE_TYPE(Plan) *p=r->plan;
    if(!file_body_room(w,1))return FLOW_CONST(LIMIT);
    FLOW_TYPE(Status) status=FLOW_FN(state)(p->declarations,f,&w->incoming[0]);
    if(status!=FLOW_CONST(OK))return status;
    w->bytes+=sizeof(FLOW_TYPE(State));
    bool returned=false;
    for(uint16_t rank=0;rank<p->functions[f].instruction_count;rank++) {
        uint16_t index=p->instruction_order[p->starts[f]+rank];
        w->working=w->incoming[index];w->incoming[index]=NULL;
        if(!w->working)continue;
        CODE_TYPE(Instruction) *in=&p->instructions[p->starts[f]+index];
        BODY_TYPE(Instruction) *fact=&r->facts[p->starts[f]+index];
        fact->reachable=true;fact->input_stack=w->working->stack_count;
        fact->cleanup_local=FLOW_CONST(NO_REFERENCE);
        fact->pending_checks=FLOW_CONST(CHECK_CLEANUP);
        status=file_body_transfer(r,w->working,in,fact);
        if(status!=FLOW_CONST(OK))return status;
        fact->output_stack=w->working->stack_count;returned|=fact->exit_checked;
        if(in->decoded.opcode==OP_FILE_RESULT_BRANCH) {
            if(!file_body_room(w,2))return FLOW_CONST(LIMIT);
            status=FLOW_FN(refine)(w->working,in->decoded.operands[0].u16,&w->edges[0],&w->edges[1]);
            if(status!=FLOW_CONST(OK))return status;
            w->bytes+=2*sizeof(FLOW_TYPE(State));fact->refinement=true;
        } else if(in->successor_count) {
            status=file_body_clone(w,w->working,&w->edges[0]);
            if(status!=FLOW_CONST(OK))return status;
            if(in->successor_count==2) {
                status=file_body_clone(w,w->working,&w->edges[1]);
                if(status!=FLOW_CONST(OK))return status;
            }
        }
        file_body_release(w,&w->working);
        for(uint8_t edge=0;edge<in->successor_count;edge++) {
            if(!w->edges[edge]->reachable){file_body_release(w,&w->edges[edge]);continue;}
            FLOW_TYPE(State) **destination=&w->incoming[in->successors[edge]];
            if(!*destination){*destination=w->edges[edge];w->edges[edge]=NULL;}
            else {
                bool changed=false;status=FLOW_FN(join)(*destination,w->edges[edge],&changed);
                if(status!=FLOW_CONST(OK))return status;
                file_body_release(w,&w->edges[edge]);
            }
        }
    }
    if(!returned)return FLOW_CONST(INVALID);
    r->checked[f]=true;return FLOW_CONST(OK);
}
void BODY_FN(free)(BODY_TYPE(Report) *r) {
    if(r){CODE_FN(free)(r->plan);free(r->facts);free(r);}
}
FLOW_TYPE(Status) BODY_FN(analyze)(const NvmModule *module,BODY_TYPE(Report) **out) {
    if(!out)return FLOW_CONST(INVALID);
    CODE_TYPE(Plan) *p=NULL;FLOW_TYPE(Status) status=CODE_FN(prepare)(module,&p);
    if(status!=FLOW_CONST(OK))return status;
    size_t bytes=p->storage_bound;
    if(!file_code_budget(&bytes,1,sizeof(BODY_TYPE(Report))) ||
       !file_code_budget(&bytes,p->instruction_count,sizeof(BODY_TYPE(Instruction))) ||
       !file_code_budget(&bytes,1,sizeof(FileBodyWorkspace)) ||
       !file_code_budget(&bytes,FLOW_CONST(LOCALS),sizeof(uint16_t))) {
        CODE_FN(free)(p);return FLOW_CONST(LIMIT);
    }
    BODY_TYPE(Report) *r=calloc(1,sizeof *r);
    if(!r){CODE_FN(free)(p);return FLOW_CONST(MEMORY);}
    r->plan=p;r->facts=calloc(p->instruction_count,sizeof *r->facts);
    if(!r->facts){BODY_FN(free)(r);return FLOW_CONST(MEMORY);}
    FileBodyWorkspace w={0};w.bytes=bytes;
    for(uint32_t rank=0;rank<p->count;rank++) {
        status=file_body_function(r,p->function_order[rank],&w);
        if(status!=FLOW_CONST(OK))break;
    }
    file_body_release(&w,&w.working);
    for(uint16_t i=0;i<CODE_CONST(FUNCTION_INSTRUCTIONS);i++)file_body_release(&w,&w.incoming[i]);
    for(uint8_t i=0;i<2;i++)file_body_release(&w,&w.edges[i]);
    if(status!=FLOW_CONST(OK)){BODY_FN(free)(r);return status;}
    *out=r;return FLOW_CONST(OK);
}
uint32_t BODY_FN(function_count)(const BODY_TYPE(Report) *r) {
    return r?CODE_FN(function_count)(r->plan):0;
}
bool BODY_FN(function)(const BODY_TYPE(Report) *r,uint32_t f,CODE_TYPE(Function) *out) {
    return r && CODE_FN(function)(r->plan,f,out);
}
bool BODY_FN(local)(const BODY_TYPE(Report) *r,uint32_t f,uint16_t local,FLOW_TYPE(Declaration) *out) {
    return r && CODE_FN(local)(r->plan,f,local,out);
}
bool BODY_FN(instruction)(const BODY_TYPE(Report) *r,uint32_t f,uint16_t i,
                              CODE_TYPE(Instruction) *instruction,BODY_TYPE(Instruction) *fact) {
    if(!r || !instruction || !fact || f>=r->plan->count ||
       i>=r->plan->functions[f].instruction_count)return false;
    *instruction=r->plan->instructions[r->plan->starts[f]+i];
    *fact=r->facts[r->plan->starts[f]+i];return true;
}
