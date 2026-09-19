/* Same translation unit as qualified logical states: actual sizeof accounting,
 * no copied state layout and no public verifier/admission recursion. */
#include "file_body.h"
struct NvmFileBodyReport {
    NvmFileCodePlan *plan;
    NvmFileBodyInstruction *facts;
    bool checked[NVM_FILE_FLOW_FUNCTIONS];
};
typedef struct {
    size_t bytes;
    NvmFileFlowState *incoming[NVM_FILE_CODE_FUNCTION_INSTRUCTIONS];
    NvmFileFlowState *working, *edges[2];
} FileBodyWorkspace;
static void file_body_release(FileBodyWorkspace *w,NvmFileFlowState **s) {
    if(*s){nvm_file_flow_state_free(*s);*s=NULL;w->bytes-=sizeof(NvmFileFlowState);}
}
static bool file_body_room(const FileBodyWorkspace *w,size_t count) {
    size_t bytes=w->bytes;return file_code_budget(&bytes,count,sizeof(NvmFileFlowState));
}
static NvmFileFlowStatus file_body_clone(FileBodyWorkspace *w,const NvmFileFlowState *s,
                                         NvmFileFlowState **out) {
    if(!file_body_room(w,1))return NVM_FILE_FLOW_LIMIT;
    NvmFileFlowStatus status=nvm_file_flow_clone(s,out);
    if(status==NVM_FILE_FLOW_OK)w->bytes+=sizeof(NvmFileFlowState);
    return status;
}
static NvmFileFlowStatus file_body_scalar(NvmFileFlowState *s,uint8_t input,uint8_t output,
                                          uint16_t count,bool equality) {
    if(s->stack_count<count)return NVM_FILE_FLOW_INVALID;
    for(uint16_t i=0;i<count;i++) {
        NvmFileFlowValue v=s->stack[s->stack_count-count+i];
        if(!v.initialized || v.owner || !same(v.type,scalar_type(v.type.tag)))return NVM_FILE_FLOW_INVALID;
        if(equality) {
            if(v.type.tag!=TAG_INT && v.type.tag!=TAG_BOOL)return NVM_FILE_FLOW_UNRESOLVED;
            if(i && v.type.tag!=s->stack[s->stack_count-count].type.tag)return NVM_FILE_FLOW_UNRESOLVED;
        } else if(v.type.tag!=input)return NVM_FILE_FLOW_INVALID;
    }
    for(uint16_t i=0;i<count;i++)pop(s);
    if(output!=TAG_COUNT)s->stack[s->stack_count++]=initial_value(scalar_type(output),true,0);
    return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus file_body_projection(NvmFileFlowState *s,uint16_t field,bool tag,bool record) {
    if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
    NvmFileFlowValue v=s->stack[s->stack_count-1];
    if(!v.initialized || v.owner || v.type.mode)return NVM_FILE_FLOW_INVALID;
    if(record && !tag && v.type.category==NVM_FILE_CATEGORY_RECORD)return nvm_file_flow_field(s,field);
    if(v.type.category!=NVM_FILE_CATEGORY_SCALAR_RESULT)return NVM_FILE_FLOW_UNRESOLVED;
    NvmFileFlowDeclaration result=scalar_type(TAG_INT);
    if(!tag) {
        if(field)return NVM_FILE_FLOW_INVALID;
        if(v.arm!=NVM_FILE_FLOW_ARM_OK && v.arm!=NVM_FILE_FLOW_ARM_ERROR)return NVM_FILE_FLOW_UNRESOLVED;
        const NlFilePlanType *type=nl_file_catalog_type(v.type.catalog_ordinal);
        if(!type || type->member_count!=2 ||
           !member_type(s,type->members[v.arm==NVM_FILE_FLOW_ARM_ERROR].type_id,&result) || owned(result))
            return NVM_FILE_FLOW_INVALID;
    }
    s->stack[s->stack_count-1]=initial_value(result,true,0);return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus file_body_transfer(NvmFileBodyReport *r,NvmFileFlowState *s,
                                            const NvmFileCodeInstruction *in,NvmFileBodyInstruction *fact) {
    const DecodedInstruction *d=&in->decoded;uint16_t local=d->operands[0].u16;
    NvmFileFlowStatus status;
    switch(d->opcode) {
    case OP_NOP: case OP_JMP: case OP_FILE_RESULT_BRANCH:return NVM_FILE_FLOW_OK;
    case OP_PUSH_I64:return nvm_file_flow_push_scalar(s,TAG_INT);
    case OP_PUSH_BOOL:return nvm_file_flow_push_scalar(s,TAG_BOOL);
    case OP_PUSH_VOID:return nvm_file_flow_push_scalar(s,TAG_VOID);
    case OP_DUP:return nvm_file_flow_dup(s);
    case OP_POP:return nvm_file_flow_pop(s);
    case OP_LOAD_LOCAL:return nvm_file_flow_load(s,local);
    case OP_STORE_LOCAL:return nvm_file_flow_store(s,local);
    case OP_OWN_MOVE_LOCAL:return nvm_file_flow_take(s,local);
    case OP_OWN_STORE_LOCAL:return nvm_file_flow_put(s,local);
    case OP_REGION_BEGIN:return nvm_file_flow_region_begin(s);
    case OP_REGION_END:return nvm_file_flow_region_end(s);
    case OP_BORROW_LOCAL_EXCLUSIVE:return nvm_file_flow_borrow(s,d->operands[1].u16,local);
    case OP_FILE_END_BORROW:return nvm_file_flow_end_borrow(s,local);
    case OP_FILE_RESULT_TAKE:return nvm_file_flow_take_result(s,local,
        d->operands[1].u8?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK);
    case OP_FILE_DROP_LOCAL:
        fact->cleanup=NVM_FILE_BODY_CLEANUP_DROP_LOCAL;fact->cleanup_local=local;
        return nvm_file_flow_drop_local(s,local);
    case OP_FILE_DROP_STACK:
        fact->cleanup=NVM_FILE_BODY_CLEANUP_DROP_STACK;return nvm_file_flow_drop_stack(s);
    case OP_RET:
        status=nvm_file_flow_can_exit(s);fact->exit_checked=status==NVM_FILE_FLOW_OK;
        fact->cleanup=NVM_FILE_BODY_CLEANUP_RETURN;return status;
    case OP_ASSERT:
        fact->cleanup=NVM_FILE_BODY_CLEANUP_ASSERT;
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
    case OP_AGG_PACK: case OP_UNION_CONSTRUCT:return nvm_file_flow_construct(s,in->catalog_ordinal,
        d->operands[d->opcode==OP_AGG_PACK?2:1].u16);
    case OP_AGG_GET: case OP_UNION_FIELD:
        return file_body_projection(s,local,false,d->opcode==OP_AGG_GET);
    case OP_AGG_TAG: case OP_UNION_TAG:return file_body_projection(s,0,true,false);
    case OP_FILE_SERVICE:
        fact->cleanup=NVM_FILE_BODY_CLEANUP_SERVICE;
        status=nvm_file_flow_service(s,in->byte_offset,d->operands[0].u32,d->operands[1].u16);break;
    case OP_CALL: case OP_CALL_REF: {
        uint32_t target=d->operands[0].u32;
        if(target>=r->plan->count || !r->checked[target])return NVM_FILE_FLOW_UNRESOLVED;
        const NvmFileFlowFunction *callee=&r->plan->functions[target].declaration;
        uint16_t references[NVM_FILE_FLOW_LOCALS];
        for(uint16_t i=0;i<callee->parameters;i++) {
            NvmFileFlowDeclaration type=s->declarations->locals[s->declarations->offsets[target]+i];
            references[i]=type.mode?d->operands[1].u16:NVM_FILE_FLOW_NO_REFERENCE;
        }
        fact->cleanup=NVM_FILE_BODY_CLEANUP_CALL;
        status=nvm_file_flow_call(s,in->byte_offset,target,references,callee->parameters);
        if(status==NVM_FILE_FLOW_OK)fact->discharged_checks=NVM_FILE_FLOW_CHECK_CALLEE;
        break;
    }
    default:return NVM_FILE_FLOW_UNRESOLVED;
    }
    if(status!=NVM_FILE_FLOW_OK)return status;
    for(uint16_t i=0;i<s->obligation_count;i++)if(s->obligations[i].site==in->byte_offset) {
        fact->has_obligation=true;fact->obligation=s->obligations[i];
        fact->pending_checks|=fact->obligation.checks & ~fact->discharged_checks;
        return NVM_FILE_FLOW_OK;
    }
    return NVM_FILE_FLOW_INVALID;
}
static NvmFileFlowStatus file_body_function(NvmFileBodyReport *r,uint32_t f,FileBodyWorkspace *w) {
    NvmFileCodePlan *p=r->plan;
    if(!file_body_room(w,1))return NVM_FILE_FLOW_LIMIT;
    NvmFileFlowStatus status=nvm_file_flow_state(p->declarations,f,&w->incoming[0]);
    if(status!=NVM_FILE_FLOW_OK)return status;
    w->bytes+=sizeof(NvmFileFlowState);
    bool returned=false;
    for(uint16_t rank=0;rank<p->functions[f].instruction_count;rank++) {
        uint16_t index=p->instruction_order[p->starts[f]+rank];
        w->working=w->incoming[index];w->incoming[index]=NULL;
        if(!w->working)continue;
        NvmFileCodeInstruction *in=&p->instructions[p->starts[f]+index];
        NvmFileBodyInstruction *fact=&r->facts[p->starts[f]+index];
        fact->reachable=true;fact->input_stack=w->working->stack_count;
        fact->cleanup_local=NVM_FILE_FLOW_NO_REFERENCE;
        fact->pending_checks=NVM_FILE_FLOW_CHECK_CLEANUP;
        status=file_body_transfer(r,w->working,in,fact);
        if(status!=NVM_FILE_FLOW_OK)return status;
        fact->output_stack=w->working->stack_count;returned|=fact->exit_checked;
        if(in->decoded.opcode==OP_FILE_RESULT_BRANCH) {
            if(!file_body_room(w,2))return NVM_FILE_FLOW_LIMIT;
            status=nvm_file_flow_refine(w->working,in->decoded.operands[0].u16,&w->edges[0],&w->edges[1]);
            if(status!=NVM_FILE_FLOW_OK)return status;
            w->bytes+=2*sizeof(NvmFileFlowState);fact->refinement=true;
        } else if(in->successor_count) {
            status=file_body_clone(w,w->working,&w->edges[0]);
            if(status!=NVM_FILE_FLOW_OK)return status;
            if(in->successor_count==2) {
                status=file_body_clone(w,w->working,&w->edges[1]);
                if(status!=NVM_FILE_FLOW_OK)return status;
            }
        }
        file_body_release(w,&w->working);
        for(uint8_t edge=0;edge<in->successor_count;edge++) {
            if(!w->edges[edge]->reachable){file_body_release(w,&w->edges[edge]);continue;}
            NvmFileFlowState **destination=&w->incoming[in->successors[edge]];
            if(!*destination){*destination=w->edges[edge];w->edges[edge]=NULL;}
            else {
                bool changed=false;status=nvm_file_flow_join(*destination,w->edges[edge],&changed);
                if(status!=NVM_FILE_FLOW_OK)return status;
                file_body_release(w,&w->edges[edge]);
            }
        }
    }
    if(!returned)return NVM_FILE_FLOW_INVALID;
    r->checked[f]=true;return NVM_FILE_FLOW_OK;
}
void nvm_file_body_free(NvmFileBodyReport *r) {
    if(r){nvm_file_code_free(r->plan);free(r->facts);free(r);}
}
NvmFileFlowStatus nvm_file_body_analyze(const NvmModule *module,NvmFileBodyReport **out) {
    if(!out)return NVM_FILE_FLOW_INVALID;
    NvmFileCodePlan *p=NULL;NvmFileFlowStatus status=nvm_file_code_prepare(module,&p);
    if(status!=NVM_FILE_FLOW_OK)return status;
    size_t bytes=p->storage_bound;
    if(!file_code_budget(&bytes,1,sizeof(NvmFileBodyReport)) ||
       !file_code_budget(&bytes,p->instruction_count,sizeof(NvmFileBodyInstruction)) ||
       !file_code_budget(&bytes,1,sizeof(FileBodyWorkspace)) ||
       !file_code_budget(&bytes,NVM_FILE_FLOW_LOCALS,sizeof(uint16_t))) {
        nvm_file_code_free(p);return NVM_FILE_FLOW_LIMIT;
    }
    NvmFileBodyReport *r=calloc(1,sizeof *r);
    if(!r){nvm_file_code_free(p);return NVM_FILE_FLOW_MEMORY;}
    r->plan=p;r->facts=calloc(p->instruction_count,sizeof *r->facts);
    if(!r->facts){nvm_file_body_free(r);return NVM_FILE_FLOW_MEMORY;}
    FileBodyWorkspace w={0};w.bytes=bytes;
    for(uint32_t rank=0;rank<p->count;rank++) {
        status=file_body_function(r,p->function_order[rank],&w);
        if(status!=NVM_FILE_FLOW_OK)break;
    }
    file_body_release(&w,&w.working);
    for(uint16_t i=0;i<NVM_FILE_CODE_FUNCTION_INSTRUCTIONS;i++)file_body_release(&w,&w.incoming[i]);
    for(uint8_t i=0;i<2;i++)file_body_release(&w,&w.edges[i]);
    if(status!=NVM_FILE_FLOW_OK){nvm_file_body_free(r);return status;}
    *out=r;return NVM_FILE_FLOW_OK;
}
uint32_t nvm_file_body_function_count(const NvmFileBodyReport *r) {
    return r?nvm_file_code_function_count(r->plan):0;
}
bool nvm_file_body_function(const NvmFileBodyReport *r,uint32_t f,NvmFileCodeFunction *out) {
    return r && nvm_file_code_function(r->plan,f,out);
}
bool nvm_file_body_local(const NvmFileBodyReport *r,uint32_t f,uint16_t local,NvmFileFlowDeclaration *out) {
    return r && nvm_file_code_local(r->plan,f,local,out);
}
bool nvm_file_body_instruction(const NvmFileBodyReport *r,uint32_t f,uint16_t i,
                              NvmFileCodeInstruction *instruction,NvmFileBodyInstruction *fact) {
    if(!r || !instruction || !fact || f>=r->plan->count ||
       i>=r->plan->functions[f].instruction_count)return false;
    *instruction=r->plan->instructions[r->plan->starts[f]+i];
    *fact=r->facts[r->plan->starts[f]+i];return true;
}
