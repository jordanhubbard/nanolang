/* I infer only provenance. Ownership, exact signatures, initialization and
 * runtime admission remain independent checks over the resulting target sets. */
typedef struct {
    uint8_t locals[NVM_AFFINE_MAX_LOCALS],stack[NVM_AFFINE_MAX_STACK];
    uint16_t height;
    bool reached;
} TargetFrame;
struct NvmAffineTargets {
    uint32_t count;
    VmDecodedFunction code[NVM_OWNED_MAX_FUNCTIONS];
    uint8_t *sites[NVM_OWNED_MAX_FUNCTIONS];
    uint8_t parameters[NVM_OWNED_MAX_FUNCTIONS][NVM_AFFINE_MAX_PARAMETERS];
    uint8_t results[NVM_OWNED_MAX_FUNCTIONS];
    uint8_t edges[NVM_OWNED_MAX_FUNCTIONS];
};
void nvm_affine_targets_free(NvmAffineTargets *plan) {
    if (!plan) return;
    for (uint32_t f=0;f<plan->count;f++) {
        vm_decoded_function_free(&plan->code[f]);free(plan->sites[f]);
    }
    free(plan);
}
bool nvm_affine_targets_at(const NvmAffineTargets *plan,uint32_t function,
                           uint32_t offset,uint8_t *mask) {
    if (!plan || function>=plan->count || !mask) return false;
    const VmDecodedInstruction *in=vm_decoded_function_at(&plan->code[function],offset);
    if (!in || in->instruction.opcode!=OP_CALL_INDIRECT) return false;
    uint8_t found=plan->sites[function][in-plan->code[function].instructions];
    if (!found) return false;
    *mask=found;return true;
}
static bool targets_join(TargetFrame *frames,uint32_t index,const TargetFrame *from,
                         Worklist *work,uint16_t locals) {
    TargetFrame *to=&frames[index];bool changed=false;
    if (!to->reached) {*to=*from;to->reached=true;changed=true;}
    else {
        if (to->height!=from->height) return false;
        for (uint16_t i=0;i<locals;i++) {
            uint8_t mask=to->locals[i]|from->locals[i];
            if (mask!=to->locals[i]) {to->locals[i]=mask;changed=true;}
        }
        for (uint16_t i=0;i<to->height;i++) {
            uint8_t mask=to->stack[i]|from->stack[i];
            if (mask!=to->stack[i]) {to->stack[i]=mask;changed=true;}
        }
    }
    return !changed || enqueue(work,index);
}
static bool targets_function(NvmAffineTargets *plan,const NvmModule *m,uint32_t f,
                              const NvmV2Layouts *layouts,bool *changed) {
    const NvmFunctionEntry *fn=&m->functions[f];
    const VmDecodedFunction *code=&plan->code[f];uint32_t count=code->instruction_count;
    TargetFrame *frames=calloc(count,sizeof(*frames));Worklist work={0};
    NvmAffineState *state=nvm_affine_state_create(m,f,fn->local_count);
    work.capacity=count;work.items=malloc(count*sizeof(*work.items));work.queued=calloc(count,sizeof(*work.queued));
    bool ok=false;
    if (!frames || !state || !work.items || !work.queued) goto done;
    frames[0].reached=true;
    for (uint16_t p=0;p<fn->arity;p++)frames[0].locals[p]=plan->parameters[f][p];
    if (!enqueue(&work,0)) goto done;
    /* Each queued revisit adds at least one of the eight target bits to one
     * live local/stack slot. First reach is the additional visit per node. */
    uint32_t visits=0,limit=count*(1u+8u*(fn->local_count+NVM_AFFINE_MAX_STACK));
    while (work.pending) {
        if (++visits>limit) goto done;
        uint32_t index=work.items[work.head];work.head=(work.head+1)%count;
        work.pending--;work.queued[index]=false;
        TargetFrame next=frames[index];uint16_t height=next.height;
        const VmDecodedInstruction *d=&code->instructions[index];
        const DecodedInstruction *in=&d->instruction;uint8_t op=in->opcode;
        const InstructionInfo *info=isa_get_info(op);
        if (!info || (!supported(op,true) && op!=OP_CALL_INDIRECT) || op==OP_CALL_REF) goto done;
        if (op==OP_RET) {
            if (height!=fn->result_count) goto done;
            if (fn->result_tag==TAG_FUNCTION && height==1) {
                uint8_t mask=plan->results[f]|next.stack[0];
                if (mask!=plan->results[f]) {plan->results[f]=mask;*changed=true;}
            }
            continue;
        }
        if (op==OP_HALT) continue;
        int pop=info->pop_count,push_count=info->push_count;
        uint8_t produced=0,call_targets=0;
        if (op==OP_CALL) {
            uint32_t target=in->operands[0].u32;
            if (!target || target>=m->function_count) goto done;
            pop=m->functions[target].arity;push_count=m->functions[target].result_count;
            call_targets=(uint8_t)(1u<<target);
        } else if (op==OP_CALL_INDIRECT) {
            pop=(int)in->operands[0].u16+1;push_count=in->operands[1].u16;
            if (height<pop || push_count>1) goto done;
            call_targets=next.stack[height-1];
            if (call_targets&1u) goto done;
            plan->sites[f][index]|=call_targets;
        } else if (op==OP_OWN_PACK) {
            uint32_t layout=in->operands[0].u32;
            if (layout>=layouts->count) goto done;
            pop=layouts->items[layout].field_count;push_count=1;
        } else if (op==OP_OWN_UNPACK_LOCAL) {
            NvmAffineType type;
            if (!nvm_affine_local_type(state,in->operands[0].u16,&type) || type.layout>=layouts->count) goto done;
            pop=0;push_count=layouts->items[type.layout].field_count;
        } else if (op==OP_OWN_UNPACK_VARIANT) {pop=0;push_count=in->operands[2].u16;}
        else if (op==OP_AGG_PACK) {pop=in->operands[3].u16;push_count=1;}
        if (pop<0 || push_count<0 || height<pop || height-pop+push_count>(int)NVM_AFFINE_MAX_STACK) goto done;
        uint16_t first=(uint16_t)(height-pop);
        if (op==OP_CALL || op==OP_CALL_INDIRECT) {
            uint16_t arity=(uint16_t)(pop-(op==OP_CALL_INDIRECT));
            for (uint32_t target=1;target<m->function_count;target++) if(call_targets&(1u<<target)) {
                const NvmFunctionEntry *callee=&m->functions[target];
                if (callee->arity!=arity || callee->result_count!=push_count) goto done;
                for (uint16_t p=0;p<arity;p++) {
                    uint8_t mask=plan->parameters[target][p]|next.stack[first+p];
                    if (mask!=plan->parameters[target][p]) {plan->parameters[target][p]=mask;*changed=true;}
                }
                produced|=plan->results[target];
            }
            plan->edges[f]|=call_targets;
        } else if (op==OP_FUNCREF) {
            uint32_t target=in->operands[0].u32;
            if (target>=m->function_count || m->functions[target].upvalue_count) goto done;
            produced=(uint8_t)(1u<<target);
        } else if (op==OP_LOAD_LOCAL) {
            uint16_t local=in->operands[0].u16;
            if (local>=fn->local_count) goto done;
            produced=next.locals[local];
        } else if (op==OP_STORE_LOCAL || op==OP_OWN_STORE_LOCAL) {
            uint16_t local=in->operands[0].u16;
            if (local>=fn->local_count || !height) goto done;
            next.locals[local]=op==OP_STORE_LOCAL?next.stack[height-1]:0;
        }
        if (op==OP_DUP) {
            if (!height || height==NVM_AFFINE_MAX_STACK) goto done;
            next.stack[height]=next.stack[height-1];
        } else if (op==OP_SWAP) {
            if (height<2) goto done;
            uint8_t mask=next.stack[height-1];next.stack[height-1]=next.stack[height-2];next.stack[height-2]=mask;
        } else for (int p=0;p<push_count;p++)next.stack[first+p]=produced;
        next.height=(uint16_t)(height-pop+push_count);
        uint32_t successors[2],length=0;
        if (op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE || op==OP_MATCH_TAG) {
            const VmDecodedInstruction *target=vm_decoded_function_at(code,d->resolved_target-fn->code_offset);
            if (!target) goto done;
            successors[length++]=(uint32_t)(target-code->instructions);
        }
        if (op!=OP_JMP)successors[length++]=index+1;
        for (uint32_t i=0;i<length;i++)
            if (successors[i]>=count || !targets_join(frames,successors[i],&next,&work,fn->local_count))goto done;
    }
    ok=true;
done:
    free(frames);free(work.items);free(work.queued);nvm_affine_state_free(state);return ok;
}
NvmAffineTargets *nvm_affine_targets_create(const NvmModule *m) {
    if (!m || !m->functions || !m->function_count || m->function_count>NVM_OWNED_MAX_FUNCTIONS ||
        m->function_count>8 || m->header.entry_point || m->functions[0].arity ||
        m->functions[0].result_count!=1 ||
        (m->functions[0].result_tag!=TAG_INT && m->functions[0].result_tag!=TAG_BOOL &&
         m->functions[0].result_tag!=TAG_U8) || m->import_count || m->module_ref_count) return NULL;
    /* I retain the validated bound outside the mutable plan while solving. */
    const uint32_t function_count=m->function_count;
    NvmAffineTargets *plan=calloc(1,sizeof(*plan));NvmV2Layouts layouts={0};
    if (!plan) return NULL;
    plan->count=function_count;
    if (nvm_v2_layouts_decode(m->layout_data,m->layout_size,&layouts)!=NVM_V2_OK) goto fail;
    uint8_t globals[NVM_OWNERSHIP_MAX_SCALAR_GLOBALS];uint32_t global_count=0;
    if (nvm_ownership_scalar_globals(m,globals,sizeof(globals),&global_count)!=NVM_V2_OK) goto fail;
    for (uint32_t f=0;f<function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];
        if (fn->upvalue_count || fn->local_count>NVM_AFFINE_MAX_LOCALS ||
            fn->code_length>NVM_AFFINE_MAX_INSTRUCTIONS*ISA_MAX_INSTRUCTION_SIZE) goto fail;
        NvmAffineState *state=nvm_affine_state_create(m,f,fn->local_count);
        NvmAffineType params[NVM_AFFINE_MAX_PARAMETERS],result;uint16_t arity=0,fields=0;
        bool valid=nvm_affine_value_parameters(state,params,NVM_AFFINE_MAX_PARAMETERS,&arity) &&
            arity==fn->arity && nvm_affine_value_result(state,&result,&fields);
        nvm_affine_state_free(state);
        if (!valid) goto fail;
        char error[VM_DECODE_ERROR_SIZE];
        if (!vm_decode_function(m,f,&plan->code[f],error) || !plan->code[f].instruction_count ||
            plan->code[f].instruction_count>NVM_AFFINE_MAX_INSTRUCTIONS) goto fail;
        plan->sites[f]=calloc(plan->code[f].instruction_count,1);
        if (!plan->sites[f]) goto fail;
        /* I preserve the existing rule that even dead direct edges are acyclic. */
        for(uint32_t i=0;i<plan->code[f].instruction_count;i++) {
            const DecodedInstruction *in=&plan->code[f].instructions[i].instruction;
            if (in->opcode==OP_CALL) {
                uint32_t target=in->operands[0].u32;
                if (!target || target>=m->function_count) goto fail;
                plan->edges[f]|=(uint8_t)(1u<<target);
            }
        }
    }
    /* Only parameter and returned-value summaries drive another global pass.
     * Each bit can be added once; I reserve one final unchanged pass. */
    uint32_t limit=8u*function_count*(NVM_AFFINE_MAX_PARAMETERS+1u)+1u;
    bool converged=false;
    for (uint32_t pass=0;pass<limit;pass++) {
        bool changed=false;
        for (uint32_t f=0;f<function_count;f++)
            if (!targets_function(plan,m,f,&layouts,&changed)) goto fail;
        if (!changed) {converged=true;break;}
    }
    if (!converged) goto fail;
    for(uint32_t f=0;f<function_count;f++) {
        for(uint32_t i=0;i<plan->code[f].instruction_count;i++)
            if (plan->code[f].instructions[i].instruction.opcode==OP_CALL_INDIRECT && !plan->sites[f][i])goto fail;
    }
    for(uint32_t k=0;k<function_count;k++)
        for(uint32_t f=0;f<function_count;f++)
            if(plan->edges[f]&(1u<<k))plan->edges[f]|=plan->edges[k];
    for(uint32_t f=0;f<function_count;f++)if(plan->edges[f]&(1u<<f))goto fail;
    nvm_v2_layouts_free(&layouts);return plan;
fail:
    nvm_v2_layouts_free(&layouts);nvm_affine_targets_free(plan);return NULL;
}
