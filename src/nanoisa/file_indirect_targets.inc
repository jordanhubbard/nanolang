/* I share exact checked decode/declaration storage, but publish only callable
 * target facts. This report is not consumed by any File execution selector. */
#include "file_indirect_targets.h"
typedef struct {
    NvmFileFlowDeclaration type;
    uint64_t targets;
    uint8_t initialized; /* 1: uninitialized alternative, 2: initialized. */
    uint8_t arms; /* 1: Ok, 2: Error, 3: either; zero for non-Results. */
} FileIndirectValue;
typedef struct {
    bool reached;
    uint16_t stack_count;
    FileIndirectValue locals[NVM_FILE_FLOW_LOCALS],stack[NVM_FILE_FLOW_STACK];
} FileIndirectState;
typedef struct {
    FileIndirectState incoming[NVM_FILE_CODE_FUNCTION_INSTRUCTIONS],working,edge;
    bool pending[NVM_FILE_CODE_FUNCTION_INSTRUCTIONS];
} FileIndirectWorkspace;
struct NvmFileIndirectTargets {
    NvmFileCodePlan *plan;
    NvmFileIndirectSummary summary;
    NvmFileIndirectCall *calls;
    uint64_t graph[NVM_FILE_FLOW_FUNCTIONS];
};
static FileIndirectValue fi_value(NvmFileFlowDeclaration t,uint8_t initialized) {
    FileIndirectValue v={0};v.type=t;v.initialized=initialized;
    if(t.category==NVM_FILE_CATEGORY_OPEN_RESULT || t.category==NVM_FILE_CATEGORY_SCALAR_RESULT)v.arms=3;
    return v;
}
static bool fi_catalog(const NvmFileCodePlan *p,uint32_t ordinal,NvmFileFlowDeclaration *out) {
    NvmFileNominalLayout l;
    if(!nvm_file_nominal_type(p->declarations->nominal,ordinal,&l))return false;
    *out=(NvmFileFlowDeclaration){l.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
        0,l.global_index,l.catalog_ordinal,l.category};return true;
}
static bool fi_member(const NvmFileCodePlan *p,const char *id,NvmFileFlowDeclaration *out) {
    if(!id){*out=scalar_type(TAG_VOID);return true;}
    if(!strcmp(id,"nsi:core/int")){*out=scalar_type(TAG_INT);return true;}
    if(!strcmp(id,"nsi:core/bool")){*out=scalar_type(TAG_BOOL);return true;}
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        const NlFilePlanType *t=nl_file_catalog_type(i);
        if(t && !strcmp(t->id,id))return fi_catalog(p,i,out);
    }
    return false;
}
static NvmFileFlowStatus fi_push(FileIndirectState *s,FileIndirectValue v) {
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count++]=v;return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus fi_take(FileIndirectState *s,NvmFileFlowDeclaration expected) {
    if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
    FileIndirectValue *v=&s->stack[s->stack_count-1];
    if(v->initialized!=2 || !same(v->type,expected))return NVM_FILE_FLOW_INVALID;
    *v=(FileIndirectValue){0};s->stack_count--;return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus fi_join(FileIndirectState *to,const FileIndirectState *from,
                                  uint16_t locals,bool *changed) {
    *changed=false;
    if(!from->reached)return NVM_FILE_FLOW_OK;
    if(!to->reached){*to=*from;*changed=true;return NVM_FILE_FLOW_OK;}
    if(to->stack_count!=from->stack_count)return NVM_FILE_FLOW_INVALID;
    for(uint16_t group=0;group<2;group++) {
        uint16_t n=group?to->stack_count:locals;
        for(uint16_t i=0;i<n;i++) {
            FileIndirectValue *a=group?&to->stack[i]:&to->locals[i];
            const FileIndirectValue *b=group?&from->stack[i]:&from->locals[i];
            if(!same(a->type,b->type))return NVM_FILE_FLOW_UNRESOLVED;
            uint64_t targets=a->targets|b->targets;
            uint8_t init=(uint8_t)(a->initialized|b->initialized),arms=(uint8_t)(a->arms|b->arms);
            *changed|=targets!=a->targets || init!=a->initialized || arms!=a->arms;
            a->targets=targets;a->initialized=init;a->arms=arms;
        }
    }
    return NVM_FILE_FLOW_OK;
}
static bool fi_signature(const NvmFileCodePlan *p,uint32_t a,uint32_t b) {
    const NvmFileFlowFunction *x=&p->functions[a].declaration,*y=&p->functions[b].declaration;
    if(x->parameters!=y->parameters || x->result_count!=y->result_count || !same(x->result,y->result))return false;
    for(uint16_t i=0;i<x->parameters;i++)
        if(!same(p->declarations->locals[p->declarations->offsets[a]+i],
                 p->declarations->locals[p->declarations->offsets[b]+i]))return false;
    return true;
}
static NvmFileFlowStatus fi_candidates(const NvmFileCodePlan *p,const FileIndirectState *s,
                                         const DecodedInstruction *d,uint64_t *bits,uint32_t *first) {
    if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
    const FileIndirectValue *v=&s->stack[s->stack_count-1];
    if(v->initialized!=2 || !same(v->type,scalar_type(TAG_FUNCTION)) || !v->targets)
        return NVM_FILE_FLOW_UNRESOLVED;
    if(p->count<64 && (v->targets>>p->count))return NVM_FILE_FLOW_INVALID;
    uint32_t selected=p->count;
    for(uint32_t f=0;f<p->count;f++)if(v->targets&(UINT64_C(1)<<f)) {
        const NvmFileFlowFunction *fn=&p->functions[f].declaration;
        if(fn->parameters!=d->operands[0].u16 || fn->result_count!=d->operands[1].u16)
            return NVM_FILE_FLOW_INVALID;
        for(uint16_t i=0;i<fn->parameters;i++)
            if(p->declarations->locals[p->declarations->offsets[f]+i].mode)return NVM_FILE_FLOW_UNRESOLVED;
        if(selected==p->count)selected=f;
        else if(!fi_signature(p,selected,f))return NVM_FILE_FLOW_UNRESOLVED;
    }
    *bits=v->targets;*first=selected;return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus fi_call(const NvmFileCodePlan *p,FileIndirectState *s,uint32_t f) {
    const NvmFileFlowFunction *fn=&p->functions[f].declaration;
    for(uint16_t i=fn->parameters;i>0;i--) {
        NvmFileFlowDeclaration type=p->declarations->locals[p->declarations->offsets[f]+i-1];
        if(type.mode)continue; /* Reference-slot liveness belongs to full flow. */
        NvmFileFlowStatus status=fi_take(s,type);if(status!=NVM_FILE_FLOW_OK)return status;
    }
    return fn->result_count?fi_push(s,fi_value(fn->result,2)):NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus fi_scalar(FileIndirectState *s,uint8_t input,uint8_t output,uint16_t count,bool equality) {
    if(s->stack_count<count)return NVM_FILE_FLOW_INVALID;
    if(equality) {
        input=s->stack[s->stack_count-1].type.tag;
        if(input!=TAG_INT && input!=TAG_BOOL)return NVM_FILE_FLOW_UNRESOLVED;
    }
    for(uint16_t i=0;i<count;i++) {
        NvmFileFlowStatus status=fi_take(s,scalar_type(input));if(status!=NVM_FILE_FLOW_OK)return status;
    }
    return output==TAG_COUNT?NVM_FILE_FLOW_OK:fi_push(s,fi_value(scalar_type(output),2));
}
static NvmFileFlowStatus fi_transfer(const NvmFileCodePlan *p,uint32_t f,FileIndirectState *s,
                                      const NvmFileCodeInstruction *in) {
    const DecodedInstruction *d=&in->decoded;uint16_t local=d->operands[0].u16;
    FileIndirectValue value;NvmFileFlowDeclaration type;NvmFileFlowStatus status;
    switch(d->opcode) {
    case OP_NOP:case OP_JMP:case OP_REGION_BEGIN:case OP_REGION_END:case OP_FILE_END_BORROW:
        return NVM_FILE_FLOW_OK;
    case OP_FILE_RESULT_BRANCH:
        value=s->locals[local];
        return value.initialized==2 && value.arms?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_INVALID;
    case OP_BORROW_LOCAL_EXCLUSIVE:
        value=s->locals[d->operands[1].u16];
        return value.initialized==2 && value.type.category==NVM_FILE_CATEGORY_FILE?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_INVALID;
    case OP_PUSH_I64:return fi_push(s,fi_value(scalar_type(TAG_INT),2));
    case OP_PUSH_BOOL:return fi_push(s,fi_value(scalar_type(TAG_BOOL),2));
    case OP_PUSH_VOID:return fi_push(s,fi_value(scalar_type(TAG_VOID),2));
    case OP_FUNCREF:
        value=fi_value(scalar_type(TAG_FUNCTION),2);value.targets=UINT64_C(1)<<d->operands[0].u32;
        return fi_push(s,value);
    case OP_LOAD_LOCAL:case OP_OWN_MOVE_LOCAL:
        value=s->locals[local];
        if(value.initialized!=2 || value.type.mode || owned(value.type)!=(d->opcode==OP_OWN_MOVE_LOCAL))
            return NVM_FILE_FLOW_INVALID;
        status=fi_push(s,value);
        if(status==NVM_FILE_FLOW_OK && d->opcode==OP_OWN_MOVE_LOCAL)s->locals[local]=fi_value(value.type,1);
        return status;
    case OP_STORE_LOCAL:case OP_OWN_STORE_LOCAL:
        if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
        value=s->stack[s->stack_count-1];
        if(value.initialized!=2 || value.type.mode || !same(value.type,s->locals[local].type) ||
           owned(value.type)!=(d->opcode==OP_OWN_STORE_LOCAL) ||
           (d->opcode==OP_OWN_STORE_LOCAL && s->locals[local].initialized!=1))return NVM_FILE_FLOW_INVALID;
        s->locals[local]=value;return fi_take(s,value.type);
    case OP_DUP:
        if(!s->stack_count || owned(s->stack[s->stack_count-1].type))return NVM_FILE_FLOW_INVALID;
        return fi_push(s,s->stack[s->stack_count-1]);
    case OP_POP:case OP_FILE_DROP_STACK:
        if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
        value=s->stack[s->stack_count-1];
        if(owned(value.type)!=(d->opcode==OP_FILE_DROP_STACK))return NVM_FILE_FLOW_INVALID;
        return fi_take(s,value.type);
    case OP_FILE_DROP_LOCAL:
        value=s->locals[local];
        if(value.initialized!=2 || !owned(value.type))return NVM_FILE_FLOW_INVALID;
        s->locals[local]=fi_value(value.type,1);return NVM_FILE_FLOW_OK;
    case OP_FILE_RESULT_TAKE: {
        value=s->locals[local];uint8_t arm=(uint8_t)(1u<<d->operands[1].u8);
        if(value.initialized!=2 || value.arms!=arm)return NVM_FILE_FLOW_UNRESOLVED;
        const NlFilePlanType *t=nl_file_catalog_type(value.type.catalog_ordinal);
        if(!t || t->member_count!=2 || !fi_member(p,t->members[d->operands[1].u8].type_id,&type))return NVM_FILE_FLOW_INVALID;
        status=fi_push(s,fi_value(type,2));
        if(status==NVM_FILE_FLOW_OK)s->locals[local]=fi_value(value.type,1);
        return status;
    }
    case OP_RET: {
        const NvmFileFlowFunction *fn=&p->functions[f].declaration;
        if(s->stack_count!=fn->result_count)return NVM_FILE_FLOW_INVALID;
        return fn->result_count?fi_take(s,fn->result):NVM_FILE_FLOW_OK;
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
    case OP_CALL:case OP_CALL_REF:return fi_call(p,s,d->operands[0].u32);
    case OP_CALL_INDIRECT: {
        uint64_t bits;uint32_t first;
        status=fi_candidates(p,s,d,&bits,&first);if(status!=NVM_FILE_FLOW_OK)return status;
        status=fi_take(s,scalar_type(TAG_FUNCTION));if(status!=NVM_FILE_FLOW_OK)return status;
        return fi_call(p,s,first);
    }
    case OP_FILE_SERVICE:
        if(in->catalog_ordinal==1)status=fi_take(s,scalar_type(TAG_INT));
        else if(in->catalog_ordinal==4) {
            if(!fi_catalog(p,0,&type))return NVM_FILE_FLOW_INVALID;
            status=fi_take(s,type);
        } else status=NVM_FILE_FLOW_OK;
        if(status!=NVM_FILE_FLOW_OK)return status;
        if(!fi_catalog(p,in->catalog_ordinal+3,&type))return NVM_FILE_FLOW_INVALID;
        return fi_push(s,fi_value(type,2));
    case OP_AGG_PACK:case OP_UNION_CONSTRUCT: {
        const NlFilePlanType *t=nl_file_catalog_type(in->catalog_ordinal);
        if(!t || !fi_catalog(p,in->catalog_ordinal,&type))return NVM_FILE_FLOW_INVALID;
        bool record=type.category==NVM_FILE_CATEGORY_RECORD;
        uint16_t variant=d->operands[d->opcode==OP_AGG_PACK?2:1].u16;
        size_t count=record?t->member_count:1;
        for(size_t i=count;i>0;i--) {
            NvmFileFlowDeclaration member;
            if(!fi_member(p,t->members[record?i-1:variant].type_id,&member))return NVM_FILE_FLOW_INVALID;
            status=fi_take(s,member);if(status!=NVM_FILE_FLOW_OK)return status;
        }
        value=fi_value(type,2);if(!record)value.arms=(uint8_t)(1u<<variant);
        return fi_push(s,value);
    }
    case OP_AGG_GET:case OP_UNION_FIELD:case OP_AGG_TAG:case OP_UNION_TAG: {
        if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
        value=s->stack[s->stack_count-1];
        if(value.initialized!=2 || owned(value.type) || value.type.mode)return NVM_FILE_FLOW_INVALID;
        const NlFilePlanType *t=nl_file_catalog_type(value.type.catalog_ordinal);
        if(!t)return NVM_FILE_FLOW_INVALID;
        bool tag=d->opcode==OP_AGG_TAG || d->opcode==OP_UNION_TAG;
        if(value.type.category==NVM_FILE_CATEGORY_RECORD && d->opcode==OP_AGG_GET) {
            if(local>=t->member_count || !fi_member(p,t->members[local].type_id,&type))return NVM_FILE_FLOW_INVALID;
        } else if(value.type.category==NVM_FILE_CATEGORY_SCALAR_RESULT) {
            if(tag)type=scalar_type(TAG_INT);
            else {
                if(local || (value.arms!=1 && value.arms!=2))return NVM_FILE_FLOW_UNRESOLVED;
                if(!fi_member(p,t->members[value.arms==2].type_id,&type))return NVM_FILE_FLOW_INVALID;
            }
        } else return NVM_FILE_FLOW_UNRESOLVED;
        s->stack[s->stack_count-1]=fi_value(type,2);return NVM_FILE_FLOW_OK;
    }
    default:return NVM_FILE_FLOW_UNRESOLVED;
    }
}
static NvmFileFlowStatus fi_function(NvmFileIndirectTargets *r,uint32_t f,
                                      FileIndirectWorkspace *w,uint32_t *failed_pc) {
    NvmFileCodePlan *p=r->plan;const NvmFileCodeFunction *fn=&p->functions[f];
    memset(w,0,sizeof *w);w->incoming[0].reached=true;w->pending[0]=true;
    for(uint16_t l=0;l<fn->declaration.locals;l++)
        w->incoming[0].locals[l]=fi_value(p->declarations->locals[p->declarations->offsets[f]+l],
                                            l<fn->declaration.parameters?2:1);
    for(;;) {
        uint16_t i=0;while(i<fn->instruction_count && !w->pending[i])i++;
        if(i==fn->instruction_count)break;
        const NvmFileCodeInstruction *in=&p->instructions[p->starts[f]+i];
        *failed_pc=in->byte_offset;
        if(r->summary.visits==NVM_FILE_INDIRECT_VISITS)return NVM_FILE_FLOW_LIMIT;
        r->summary.visits++;w->pending[i]=false;w->working=w->incoming[i];
        NvmFileFlowStatus status=fi_transfer(p,f,&w->working,in);
        if(status!=NVM_FILE_FLOW_OK)return status;
        for(uint8_t e=0;e<in->successor_count;e++) {
            w->edge=w->working;
            if(in->decoded.opcode==OP_FILE_RESULT_BRANCH) {
                FileIndirectValue *v=&w->edge.locals[in->decoded.operands[0].u16];
                v->arms&=(uint8_t)(1u<<e);
                if(!v->arms)continue;
            }
            bool changed=false;uint16_t target=in->successors[e];
            status=fi_join(&w->incoming[target],&w->edge,fn->declaration.locals,&changed);
            if(status!=NVM_FILE_FLOW_OK)return status;
            w->pending[target]|=changed;
        }
    }
    /* Only the completed fixed point supplies published calls and graph edges. */
    for(uint16_t i=0;i<fn->instruction_count;i++) {
        const NvmFileCodeInstruction *in=&p->instructions[p->starts[f]+i];
        if(in->decoded.opcode!=OP_CALL_INDIRECT)continue;
        *failed_pc=in->byte_offset;
        if(!w->incoming[i].reached)return NVM_FILE_FLOW_UNRESOLVED;
        uint64_t bits;uint32_t first;
        NvmFileFlowStatus status=fi_candidates(p,&w->incoming[i],&in->decoded,&bits,&first);
        if(status!=NVM_FILE_FLOW_OK)return status;
        const NvmFileFlowFunction *callee=&p->functions[first].declaration;
        r->calls[r->summary.calls++]=(NvmFileIndirectCall){f,in->byte_offset,i,callee->parameters,
            bits,callee->result_count,callee->result};
        r->graph[f]|=bits;
    }
    return NVM_FILE_FLOW_OK;
}
static NvmFileIndirectResult fi_result(NvmFileFlowStatus s,uint32_t f,uint32_t pc) {
    NvmFileIndirectStatus status=NVM_FILE_INDIRECT_INVALID;const char *message="I found invalid callable target facts.";
    switch(s) {
    case NVM_FILE_FLOW_OK:status=NVM_FILE_INDIRECT_DESCRIBED;message="I describe exact targets without File execution authority.";break;
    case NVM_FILE_FLOW_UNRESOLVED:status=NVM_FILE_INDIRECT_UNRESOLVED;message="I cannot resolve this complete callable target contract.";break;
    case NVM_FILE_FLOW_LIMIT:status=NVM_FILE_INDIRECT_LIMIT;message="I reached my bounded target-query limit.";break;
    case NVM_FILE_FLOW_MEMORY:status=NVM_FILE_INDIRECT_MEMORY;message="I could not allocate the target-query staging.";break;
    default:break;
    }
    return (NvmFileIndirectResult){status,f,pc,message};
}
NvmFileIndirectResult nvm_file_indirect_targets(const NvmModule *m,NvmFileIndirectTargets **out) {
    uint32_t f=NVM_V2_NO_INDEX,pc=NVM_V2_NO_INDEX,total=0;size_t bytes=0;
    if(!out)return fi_result(NVM_FILE_FLOW_INVALID,f,pc);
    NvmFileFlowStatus status=file_code_preflight(m,&total,&bytes);
    if(status!=NVM_FILE_FLOW_OK)return fi_result(status,f,pc);
    if(!file_code_budget(&bytes,1,sizeof(NvmFileIndirectTargets)) ||
       !file_code_budget(&bytes,total,sizeof(NvmFileIndirectCall)) ||
       !file_code_budget(&bytes,1,sizeof(FileIndirectWorkspace)) || bytes>NVM_FILE_INDIRECT_BYTES)
        return fi_result(NVM_FILE_FLOW_LIMIT,f,pc);
    NvmFileCodePlan *plan=NULL;
    status=file_code_prepare_profile(m,&plan,false,true);
    if(status!=NVM_FILE_FLOW_OK)return fi_result(status,f,pc);
    NvmFileIndirectTargets *r=calloc(1,sizeof *r);
    if(!r){nvm_file_code_free(plan);return fi_result(NVM_FILE_FLOW_MEMORY,f,pc);}
    r->plan=plan;r->summary=(NvmFileIndirectSummary){plan->count,total,0,0,bytes};
    r->calls=calloc(total,sizeof *r->calls);
    FileIndirectWorkspace *w=calloc(1,sizeof *w);
    if(!r->calls || !w){status=NVM_FILE_FLOW_MEMORY;goto fail;}
    memcpy(r->graph,plan->calls,sizeof r->graph);
    for(f=0;f<plan->count;f++) {
        pc=NVM_V2_NO_INDEX;status=fi_function(r,f,w,&pc);
        if(status!=NVM_FILE_FLOW_OK)goto fail;
    }
    /* Transitive closure includes dead direct edges and every indirect target. */
    for(uint32_t via=0;via<plan->count;via++)
        for(uint32_t source=0;source<plan->count;source++)
            if(r->graph[source]&(UINT64_C(1)<<via))r->graph[source]|=r->graph[via];
    for(f=0;f<plan->count;f++)if(r->graph[f]&(UINT64_C(1)<<f)) {
        pc=NVM_V2_NO_INDEX;status=NVM_FILE_FLOW_UNRESOLVED;goto fail;
    }
    free(w);*out=r;return fi_result(NVM_FILE_FLOW_OK,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX);
fail:
    free(w);nvm_file_indirect_targets_free(r);return fi_result(status,f,pc);
}
void nvm_file_indirect_targets_free(NvmFileIndirectTargets *r) {
    if(r){nvm_file_code_free(r->plan);free(r->calls);free(r);}
}
bool nvm_file_indirect_targets_summary(const NvmFileIndirectTargets *r,NvmFileIndirectSummary *out) {
    if(!r || !out)return false;
    *out=r->summary;return true;
}
bool nvm_file_indirect_targets_call(const NvmFileIndirectTargets *r,uint32_t index,NvmFileIndirectCall *out) {
    if(!r || !out || index>=r->summary.calls)return false;
    *out=r->calls[index];return true;
}
bool nvm_file_indirect_targets_parameter(const NvmFileIndirectTargets *r,uint32_t index,uint16_t parameter,NvmFileFlowDeclaration *out) {
    if(!r || !out || index>=r->summary.calls || parameter>=r->calls[index].parameters)return false;
    uint32_t f=0;while(f<r->plan->count && !(r->calls[index].candidates&(UINT64_C(1)<<f)))f++;
    return nvm_file_code_local(r->plan,f,parameter,out);
}
