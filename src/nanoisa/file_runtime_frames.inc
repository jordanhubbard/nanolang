/* Owning translation unit: file_runtime.c. No allocation after carrier begin. */
static bool frf_current(const NvmFileRuntime *c,NvmFileHostedFunction *fn,
                        NvmFileCodeInstruction *in,NvmFileBodyInstruction *fact) {
    if(!c || c->busy || c->phase!=FR_ACTIVE || !c->acquired || c->complete ||
       c->report.status!=NVM_FILE_RUNTIME_OK || !c->frame_count || c->frame_count>c->storage.frames)return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    return !f->waiting && nvm_file_hosted_function(c->plan,f->function,fn) &&
           f->instruction<=UINT16_MAX && nvm_file_hosted_instruction(c->plan,f->function,
               (uint16_t)f->instruction,in,fact) && fact->reachable &&
           f->stack_count==fact->input_stack;
}
static NvmFileRuntimeStatus frf_ready(NvmFileRuntime *c,NvmFileHostedFunction *fn,
                                      NvmFileCodeInstruction *in,NvmFileBodyInstruction *fact) {
    NvmFileRuntimeStatus status=fr_ready(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!frf_current(c,fn,in,fact))return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    c->report.function=c->frames[c->frame_count-1].function;
    c->report.instruction=in->byte_offset;return NVM_FILE_RUNTIME_OK;
}
static NvmFileRuntimeStatus frf_bad(NvmFileRuntime *c) {
    return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
}
static bool frf_matches(NvmFileRuntimeView v,NvmFileFlowDeclaration type) {
    bool owner=type.category==NVM_FILE_CATEGORY_FILE || type.category==NVM_FILE_CATEGORY_OPEN_RESULT;
    return v.initialized && fr_same(v.type,type) && v.formal==(type.mode==2) &&
           v.owning==(owner && !type.mode);
}
static bool frf_range(const NvmFileRuntime *c,uint64_t base,uint64_t count) {
    return base<=c->storage.values && count<=c->storage.values-base;
}
static bool frf_empty_range(const NvmFileRuntime *c,uint32_t base,uint32_t count) {
    if(!frf_range(c,base,count))return false;
    for(uint32_t i=0;i<count;i++)if(!fr_empty(c,base+i))return false;
    return true;
}
static bool frf_stack(const NvmFileRuntime *c,const FileRuntimeFrame *f,
                      const NvmFileHostedFunction *fn,uint32_t count) {
    if(count>fn->operand_peak || !frf_range(c,f->stack_base,fn->operand_peak) ||
       !frf_empty_range(c,f->staging_base,fn->staging_slots))return false;
    for(uint32_t i=0;i<fn->operand_peak;i++) {
        const NvmFileRuntimeView *v=&c->values[f->stack_base+i].view;
        if(i<count?(!v->initialized || v->formal):v->initialized)return false;
    }
    return true;
}
static bool frf_held(const NvmFileRuntime *c,uint32_t root) {
    for(uint32_t i=0;i<c->storage.references;i++)
        if(c->references[i].live && c->references[i].owner_root==root)return true;
    return false;
}
static bool frf_layout(const NvmFileRuntime *c,uint32_t function,uint64_t base,
                       uint16_t depth,FileRuntimeFrame *out,NvmFileHostedFunction *fn) {
    if(depth>=c->storage.frames || !nvm_file_hosted_function(c->plan,function,fn) ||
       !fn->staging_slots || fn->locals!=fn->code.declaration.locals)return false;
    uint64_t extent=c->mode==NVM_FILE_RUNTIME_VM?fn->vm_value_slots:fn->native_value_slots;
    uint64_t stage=base+fn->locals,stack=stage+fn->staging_slots;
    uint64_t refs=(uint64_t)depth*NVM_FILE_FLOW_REFERENCES;
    if(!frf_range(c,base,extent) || stack>c->storage.values ||
       !frf_range(c,stack,fn->operand_peak) || refs>c->storage.references ||
       NVM_FILE_FLOW_REFERENCES>c->storage.references-refs ||
       fn->frames>c->storage.frames-depth)return false;
    *out=(FileRuntimeFrame){.function=function,.locals_base=(uint32_t)base,
        .stack_base=(uint32_t)stack,.staging_base=(uint32_t)stage,
        .reference_base=(uint32_t)refs,.region_base=c->region_count};return true;
}
static bool frf_next_fact(const NvmFileRuntime *c,uint32_t function,uint16_t index,uint32_t count) {
    NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    return nvm_file_hosted_instruction(c->plan,function,index,&in,&fact) &&
           fact.reachable && fact.input_stack==count;
}
NvmFileRuntimeStatus nvm_file_runtime_frame_start(NvmFileRuntime *c) {
    NvmFileRuntimeStatus status=fr_ready(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame frame;NvmFileHostedFunction fn;
    if(!c->acquired || c->frame_count || c->region_count ||
       !frf_layout(c,c->current_root,0,0,&frame,&fn) || fn.code.declaration.parameters ||
       !frf_next_fact(c,c->current_root,0,0) || !frf_empty_range(c,0,c->storage.values))return frf_bad(c);
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live)return frf_bad(c);
    c->frames[0]=frame;c->frame_count=1;
    return nvm_file_runtime_site(c,frame.function,0);
}
bool nvm_file_runtime_frame_view(const NvmFileRuntime *c,NvmFileRuntimeFrameView *out) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    if(!out || !frf_current(c,&fn,&in,&fact))return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    *out=(NvmFileRuntimeFrameView){c->mode,f->function,f->instruction,in.byte_offset,c->frame_count,
        fn.locals,fn.staging_slots,fn.operand_peak,f->locals_base,f->staging_base,f->stack_base,
        f->stack_count,f->reference_base,f->region_base};return true;
}
bool nvm_file_runtime_frame_local(const NvmFileRuntime *c,uint16_t local,uint32_t *out) {
    NvmFileRuntimeFrameView f;
    if(!out || !nvm_file_runtime_frame_view(c,&f) || local>=f.locals)return false;
    *out=f.locals_base+local;return true;
}
bool nvm_file_runtime_frame_operand(const NvmFileRuntime *c,uint16_t index,uint32_t *out) {
    NvmFileRuntimeFrameView f;
    if(!out || !nvm_file_runtime_frame_view(c,&f) || index>=f.stack_count ||
       !fr_live(c,f.stack_base+index))return false;
    *out=f.stack_base+index;return true;
}
bool nvm_file_runtime_frame_reserve(const NvmFileRuntime *c,uint16_t index,uint32_t *out) {
    NvmFileRuntimeFrameView f;
    if(!out || !nvm_file_runtime_frame_view(c,&f) || index>=f.operand_peak ||
       !fr_empty(c,f.stack_base+index))return false;
    *out=f.stack_base+index;return true;
}
bool nvm_file_runtime_frame_reference(const NvmFileRuntime *c,uint16_t index,uint32_t *out) {
    NvmFileRuntimeFrameView f;
    if(!out || !nvm_file_runtime_frame_view(c,&f) || index>=NVM_FILE_FLOW_REFERENCES)return false;
    *out=f.reference_base+index;return true;
}
NvmFileRuntimeStatus nvm_file_runtime_frame_store(NvmFileRuntime *c) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];uint16_t local=in.decoded.operands[0].u16;
    bool owner=in.decoded.opcode==OP_OWN_STORE_LOCAL;NvmFileFlowDeclaration expected;
    if((!owner && in.decoded.opcode!=OP_STORE_LOCAL) || !f->stack_count || local>=fn.locals ||
       !frf_stack(c,f,&fn,f->stack_count) || !nvm_file_hosted_local(c->plan,f->function,local,&expected))return frf_bad(c);
    uint32_t src=f->stack_base+f->stack_count-1,dst=f->locals_base+local;
    NvmFileRuntimeView v=c->values[src].view;
    if(expected.mode || !frf_matches(v,expected) || v.owning!=owner ||
       (owner?(!fr_empty(c,dst) || frf_held(c,src)):
        (fr_live(c,dst) && !frf_matches(c->values[dst].view,expected))))
        return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    if(!owner)fr_clear(c,dst);
    return nvm_file_runtime_move(c,src,dst);
}
NvmFileRuntimeStatus nvm_file_runtime_frame_next(NvmFileRuntime *c,uint8_t successor) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    if(in.decoded.opcode==OP_CALL || in.decoded.opcode==OP_CALL_REF || in.decoded.opcode==OP_RET ||
       successor>=in.successor_count || !frf_stack(c,f,&fn,fact.output_stack) ||
       !frf_next_fact(c,f->function,in.successors[successor],fact.output_stack))return frf_bad(c);
    f->stack_count=fact.output_stack;f->instruction=in.successors[successor];
    return nvm_file_runtime_site(c,f->function,(uint16_t)f->instruction);
}
NvmFileRuntimeStatus nvm_file_runtime_frame_call(NvmFileRuntime *c) {
    NvmFileHostedFunction fn,callee;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1],child;
    uint32_t target=in.decoded.operands[0].u32;
    if((in.decoded.opcode!=OP_CALL && in.decoded.opcode!=OP_CALL_REF) || in.successor_count!=1 ||
       !fact.has_obligation || fact.obligation.kind!=NVM_FILE_FLOW_CALL || fact.obligation.target!=target ||
       fact.obligation.site!=in.byte_offset || !(fact.discharged_checks&NVM_FILE_FLOW_CHECK_CALLEE) ||
       !frf_stack(c,f,&fn,f->stack_count) || !nvm_file_hosted_function(c->plan,target,&callee))return frf_bad(c);
    uint16_t parameters=callee.code.declaration.parameters,borrowed=0,values=0;
    uint32_t reference=NVM_FILE_RUNTIME_NO_SLOT;
    for(uint16_t i=0;i<parameters;i++) {
        NvmFileFlowDeclaration type;
        if(!nvm_file_hosted_local(c->plan,target,i,&type))return frf_bad(c);
        if(type.mode) {
            if(type.mode!=2 || ++borrowed>1 || in.decoded.opcode!=OP_CALL_REF ||
               in.decoded.operands[1].u16>=NVM_FILE_FLOW_REFERENCES)return frf_bad(c);
            reference=f->reference_base+in.decoded.operands[1].u16;
            NvmFileFlowDeclaration wanted=c->types[0];wanted.mode=2;
            if(!fr_same(type,wanted) || !fr_reference(c,reference))
                return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
        } else values++;
    }
    if(parameters!=fact.obligation.parameters || borrowed!=fact.obligation.borrowed_inputs ||
       values>f->stack_count || fn.staging_slots<parameters+1u ||
       callee.code.declaration.result_count!=fact.obligation.result_count ||
       (callee.code.declaration.result_count && !fr_same(callee.code.declaration.result,fact.obligation.result)))return frf_bad(c);
    uint32_t prefix=f->stack_count-values;
    if((uint32_t)prefix+callee.code.declaration.result_count!=fact.output_stack ||
       fact.output_stack>fn.operand_peak || !frf_next_fact(c,f->function,in.successors[0],fact.output_stack))return frf_bad(c);
    uint64_t base=(uint64_t)f->stack_base+(c->mode==NVM_FILE_RUNTIME_VM?prefix:fn.operand_peak);
    if(!frf_layout(c,target,base,c->frame_count,&child,&callee) || !frf_next_fact(c,target,0,0))return frf_bad(c);
    uint32_t end=child.stack_base+callee.operand_peak;
    for(uint32_t i=child.locals_base;i<end;i++) {
        bool argument=i>=f->stack_base+prefix && i<f->stack_base+f->stack_count;
        if(!argument && !fr_empty(c,i))return frf_bad(c);
    }
    for(uint32_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)
        if(c->references[child.reference_base+i].live)return frf_bad(c);
    uint32_t arg=f->stack_base+prefix;uint16_t owned=0;
    for(uint16_t i=0;i<parameters;i++) {
        NvmFileFlowDeclaration type;
        if(!nvm_file_hosted_local(c->plan,target,i,&type))return frf_bad(c);
        if(type.mode)continue;
        NvmFileRuntimeView value=c->values[arg].view;
        if(!frf_matches(value,type) || value.formal || (value.owning && frf_held(c,arg)))
            return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
        owned+=value.owning;arg++;
    }
    if(owned!=fact.obligation.owned_inputs)return frf_bad(c);
    /* Complete ALL staging before installing ANY local: VM child locals may
     * cover a later argument's original operand position. No reverse moves. */
    arg=f->stack_base+prefix;
    for(uint16_t i=0;i<parameters;i++) {
        NvmFileFlowDeclaration type;
        if(!nvm_file_hosted_local(c->plan,target,i,&type))return frf_bad(c);
        if(type.mode)continue;
        status=nvm_file_runtime_move(c,arg++,f->staging_base+i);if(status!=NVM_FILE_RUNTIME_OK)return status;
    }
    for(uint16_t i=0;i<parameters;i++) {
        NvmFileFlowDeclaration type;
        if(!nvm_file_hosted_local(c->plan,target,i,&type))return frf_bad(c);
        if(type.mode)status=nvm_file_runtime_bind_formal(c,reference,child.locals_base+i,child.reference_base+i);
        else status=nvm_file_runtime_move(c,f->staging_base+i,child.locals_base+i);
        if(status!=NVM_FILE_RUNTIME_OK)return status;
    }
    f->stack_count=prefix;f->waiting=true;c->frames[c->frame_count++]=child;
    return nvm_file_runtime_site(c,child.function,0);
}
NvmFileRuntimeStatus nvm_file_runtime_frame_return(NvmFileRuntime *c) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];uint8_t results=fn.code.declaration.result_count;
    if(in.decoded.opcode!=OP_RET || !fact.exit_checked || in.successor_count ||
       f->stack_count!=results || !frf_stack(c,f,&fn,results) || c->region_count!=f->region_base)return frf_bad(c);
    uint32_t result=results?f->stack_base:NVM_FILE_RUNTIME_NO_SLOT;
    if(results && (!frf_matches(c->values[result].view,fn.code.declaration.result) ||
       c->values[result].view.formal || frf_held(c,result)))return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    for(uint16_t i=0;i<fn.locals;i++) {
        NvmFileRuntimeView v=c->values[f->locals_base+i].view;NvmFileFlowDeclaration type;
        if(!nvm_file_hosted_local(c->plan,f->function,i,&type) ||
           (v.initialized && v.owning) || (type.mode && !frf_matches(v,type)))return frf_bad(c);
        if(v.formal) {
            uint32_t ref=f->reference_base+i;
            if(i>=fn.code.declaration.parameters ||
               !frf_matches(v,type) || c->values[f->locals_base+i].reference!=ref ||
               !fr_reference(c,ref) || !c->references[ref].formal ||
               c->references[ref].formal_root!=f->locals_base+i)return frf_bad(c);
        }
    }
    for(uint32_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++) {
        FileRuntimeReference *r=&c->references[f->reference_base+i];
        if(r->live && (!r->formal || i>=fn.code.declaration.parameters ||
           !fr_live(c,f->locals_base+i) || !c->values[f->locals_base+i].view.formal ||
           r->formal_root!=f->locals_base+i || !fr_reference(c,f->reference_base+i)))return frf_bad(c);
    }
    FileRuntimeFrame *parent=NULL;NvmFileHostedFunction caller={0};
    NvmFileCodeInstruction call={0};NvmFileBodyInstruction call_fact={0};uint32_t staged=NVM_FILE_RUNTIME_NO_SLOT;
    if(c->frame_count>1) {
        parent=&c->frames[c->frame_count-2];
        if(!parent->waiting || !nvm_file_hosted_function(c->plan,parent->function,&caller) ||
           !nvm_file_hosted_instruction(c->plan,parent->function,(uint16_t)parent->instruction,&call,&call_fact) ||
           (call.decoded.opcode!=OP_CALL && call.decoded.opcode!=OP_CALL_REF) || call.successor_count!=1 ||
           call.decoded.operands[0].u32!=f->function || !call_fact.has_obligation ||
           !frf_empty_range(c,parent->staging_base,caller.staging_slots) ||
           call_fact.output_stack!=parent->stack_count+results ||
           !frf_next_fact(c,parent->function,call.successors[0],call_fact.output_stack))return frf_bad(c);
        for(uint32_t i=0;i<parent->stack_count;i++)if(!fr_live(c,parent->stack_base+i) ||
            c->values[parent->stack_base+i].view.formal)return frf_bad(c);
        staged=parent->staging_base+caller.staging_slots-1;
        if(results) { status=nvm_file_runtime_move(c,result,staged);if(status!=NVM_FILE_RUNTIME_OK)return status; }
    }
    /* Removing formal aliases never releases an ancestor borrow epoch. */
    for(uint32_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)if(c->references[f->reference_base+i].live) {
        status=nvm_file_runtime_end_reference(c,f->reference_base+i);if(status!=NVM_FILE_RUNTIME_OK)return status;
    }
    if(!parent) {
        if(f->function!=c->current_root)return frf_bad(c);
        return nvm_file_runtime_complete_root(c,result);
    }
    uint32_t end=f->stack_base+fn.operand_peak;
    for(uint32_t i=f->locals_base;i<end;i++)fr_clear(c,i);
    *f=(FileRuntimeFrame){0};c->frame_count--;
    if(results) {
        status=nvm_file_runtime_move(c,staged,parent->stack_base+parent->stack_count);
        if(status!=NVM_FILE_RUNTIME_OK)return status;
    }
    if(!frf_stack(c,parent,&caller,call_fact.output_stack))return frf_bad(c);
    parent->stack_count=call_fact.output_stack;parent->waiting=false;parent->instruction=call.successors[0];
    return nvm_file_runtime_site(c,parent->function,(uint16_t)parent->instruction);
}
NvmFileRuntimeStatus nvm_file_runtime_frame_region_begin(NvmFileRuntime *c) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    if(in.decoded.opcode!=OP_REGION_BEGIN || c->region_count<f->region_base ||
       c->region_count-f->region_base>=NVM_FILE_FLOW_REFERENCES)return frf_bad(c);
    return nvm_file_runtime_region_begin(c);
}
NvmFileRuntimeStatus nvm_file_runtime_frame_region_end(NvmFileRuntime *c) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(in.decoded.opcode!=OP_REGION_END || c->region_count<=c->frames[c->frame_count-1].region_base)return frf_bad(c);
    return nvm_file_runtime_region_end(c);
}
NvmFileRuntimeStatus nvm_file_runtime_frame_borrow(NvmFileRuntime *c) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    uint16_t reference=in.decoded.operands[0].u16,local=in.decoded.operands[1].u16;
    if(in.decoded.opcode!=OP_BORROW_LOCAL_EXCLUSIVE || reference>=NVM_FILE_FLOW_REFERENCES ||
       local>=fn.locals || c->region_count<=f->region_base)return frf_bad(c);
    return nvm_file_runtime_borrow(c,f->locals_base+local,f->reference_base+reference);
}
NvmFileRuntimeStatus nvm_file_runtime_frame_end_reference(NvmFileRuntime *c) {
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];uint16_t index=in.decoded.operands[0].u16;
    if(in.decoded.opcode!=OP_FILE_END_BORROW || index>=NVM_FILE_FLOW_REFERENCES ||
       !fr_reference(c,f->reference_base+index) || c->references[f->reference_base+index].formal)
        return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    return nvm_file_runtime_end_reference(c,f->reference_base+index);
}
