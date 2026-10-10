/* Owning translation unit: file_runtime.c. No allocation after carrier begin. */
static bool frf_current(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(HostedFunction) *fn,
                        SERVICE_TYPE(CodeInstruction) *in,SERVICE_TYPE(BodyInstruction) *fact) {
    if(!c || c->busy || c->phase!=FR_ACTIVE || !c->acquired || c->complete ||
       c->report.status!=SERVICE_CONST(RUNTIME_OK) || !c->frame_count || c->frame_count>c->storage.frames)return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    return !f->waiting && fr_function(c,f->function,fn) &&
           f->instruction<=UINT16_MAX && fr_instruction(c,f->function,
               (uint16_t)f->instruction,f->variant,in,fact) && fact->reachable &&
           f->stack_count==fact->input_stack;
}
static SERVICE_TYPE(RuntimeStatus) frf_ready(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(HostedFunction) *fn,
                                      SERVICE_TYPE(CodeInstruction) *in,SERVICE_TYPE(BodyInstruction) *fact) {
    SERVICE_TYPE(RuntimeStatus) status=fr_ready(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!frf_current(c,fn,in,fact) || (c->kind!=FR_ACYCLIC &&
       !c->frames[c->frame_count-1].instruction_open))return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    c->report.function=c->frames[c->frame_count-1].function;
    c->report.instruction=in->byte_offset;return SERVICE_CONST(RUNTIME_OK);
}
static SERVICE_TYPE(RuntimeStatus) frf_bad(SERVICE_TYPE(Runtime) *c) {
    return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
}
static bool frf_matches(SERVICE_TYPE(RuntimeView) v,SERVICE_TYPE(FlowDeclaration) type) {
    bool owner=type.category==NOMINAL_CATEGORY_OWNER || type.category==NOMINAL_CATEGORY_OWNER_RESULT;
    return v.initialized && fr_same(v.type,type) && v.formal==(type.mode!=0) &&
           v.owning==(owner && !type.mode);
}
static bool frf_range(const SERVICE_TYPE(Runtime) *c,uint64_t base,uint64_t count) {
    return base<=c->storage.values && count<=c->storage.values-base;
}
static bool frf_empty_range(const SERVICE_TYPE(Runtime) *c,uint32_t base,uint32_t count) {
    if(!frf_range(c,base,count))return false;
    for(uint32_t i=0;i<count;i++)if(!fr_empty(c,base+i))return false;
    return true;
}
static bool frf_stack(const SERVICE_TYPE(Runtime) *c,const FileRuntimeFrame *f,
                      const SERVICE_TYPE(HostedFunction) *fn,uint32_t count) {
    if(count>fn->operand_peak || !frf_range(c,f->stack_base,fn->operand_peak) ||
       !frf_empty_range(c,f->staging_base,fn->staging_slots))return false;
    for(uint32_t i=0;i<fn->operand_peak;i++) {
        const SERVICE_TYPE(RuntimeView) *v=&c->values[f->stack_base+i].view;
        if(i<count?(!v->initialized || v->formal):v->initialized)return false;
    }
    return true;
}
static bool frf_held(const SERVICE_TYPE(Runtime) *c,uint32_t root) {
    for(uint32_t i=0;i<c->storage.references;i++)
        if(c->references[i].live && c->references[i].owner_root==root)return true;
    return false;
}
static bool frf_layout(const SERVICE_TYPE(Runtime) *c,uint32_t function,uint64_t base,
                       uint16_t depth,FileRuntimeFrame *out,SERVICE_TYPE(HostedFunction) *fn) {
    if(depth>=c->storage.frames || !fr_function(c,function,fn) ||
       !fn->staging_slots || fn->locals!=fn->code.declaration.locals)return false;
    uint64_t extent=c->mode==SERVICE_CONST(RUNTIME_VM)?fn->vm_value_slots:fn->native_value_slots;
    uint64_t stage=base+fn->locals,stack=stage+fn->staging_slots;
    uint64_t refs=(uint64_t)depth*SERVICE_CONST(FLOW_REFERENCES);
    if(!frf_range(c,base,extent) || stack>c->storage.values ||
       !frf_range(c,stack,fn->operand_peak) || refs>c->storage.references ||
       SERVICE_CONST(FLOW_REFERENCES)>c->storage.references-refs ||
       fn->frames>c->storage.frames-depth)return false;
    *out=(FileRuntimeFrame){.function=function,.locals_base=(uint32_t)base,
        .stack_base=(uint32_t)stack,.staging_base=(uint32_t)stage,
        .reference_base=(uint32_t)refs,.region_base=c->region_count};return true;
}
static bool frf_next_fact(const SERVICE_TYPE(Runtime) *c,uint32_t function,uint16_t index,uint8_t variant,uint32_t count) {
    SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    return fr_instruction(c,function,index,variant,&in,&fact) &&
           fact.reachable && fact.input_stack==count;
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_start)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(RuntimeStatus) status=fr_ready(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame frame;SERVICE_TYPE(HostedFunction) fn;
    if(!c->acquired || c->frame_count || c->region_count ||
       !frf_layout(c,c->current_root,0,0,&frame,&fn) || fn.code.declaration.parameters ||
       !frf_next_fact(c,c->current_root,0,0,0) || !frf_empty_range(c,0,c->storage.values))return frf_bad(c);
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live)return frf_bad(c);
    c->frames[0]=frame;c->frame_count=1;
    if(c->kind!=FR_ACYCLIC && frc_validate(c,&frame)!=SERVICE_CONST(RUNTIME_OK))return c->report.status;
    return SERVICE_FN(runtime_site)(c,frame.function,0);
}
bool SERVICE_FN(runtime_frame_view)(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeFrameView) *out) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    if(!out || !frf_current(c,&fn,&in,&fact))return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    *out=(SERVICE_TYPE(RuntimeFrameView)){c->mode,f->function,f->instruction,in.byte_offset,c->frame_count,
        fn.locals,fn.staging_slots,fn.operand_peak,f->locals_base,f->staging_base,f->stack_base,
        f->stack_count,f->reference_base,f->region_base};return true;
}
bool SERVICE_FN(runtime_frame_local)(const SERVICE_TYPE(Runtime) *c,uint16_t local,uint32_t *out) {
    SERVICE_TYPE(RuntimeFrameView) f;
    if(!out || !SERVICE_FN(runtime_frame_view)(c,&f) || local>=f.locals)return false;
    *out=f.locals_base+local;return true;
}
bool SERVICE_FN(runtime_frame_operand)(const SERVICE_TYPE(Runtime) *c,uint16_t index,uint32_t *out) {
    SERVICE_TYPE(RuntimeFrameView) f;
    if(!out || !SERVICE_FN(runtime_frame_view)(c,&f) || index>=f.stack_count ||
       !fr_live(c,f.stack_base+index))return false;
    *out=f.stack_base+index;return true;
}
bool SERVICE_FN(runtime_frame_reserve)(const SERVICE_TYPE(Runtime) *c,uint16_t index,uint32_t *out) {
    SERVICE_TYPE(RuntimeFrameView) f;
    if(!out || !SERVICE_FN(runtime_frame_view)(c,&f) || index>=f.operand_peak ||
       !fr_empty(c,f.stack_base+index))return false;
    *out=f.stack_base+index;return true;
}
bool SERVICE_FN(runtime_frame_scratch)(const SERVICE_TYPE(Runtime) *c,uint32_t *out) {
    SERVICE_TYPE(RuntimeFrameView) f;
    if(!out || !SERVICE_FN(runtime_frame_view)(c,&f) || !f.staging_slots)return false;
    uint64_t slot=(uint64_t)f.staging_base+f.staging_slots-1;
    if(slot>=c->storage.values || !fr_empty(c,(uint32_t)slot))return false;
    *out=(uint32_t)slot;return true;
}
bool SERVICE_FN(runtime_frame_reference)(const SERVICE_TYPE(Runtime) *c,uint16_t index,uint32_t *out) {
    SERVICE_TYPE(RuntimeFrameView) f;
    if(!out || !SERVICE_FN(runtime_frame_view)(c,&f) || index>=SERVICE_CONST(FLOW_REFERENCES))return false;
    *out=f.reference_base+index;return true;
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_store)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];uint16_t local=in.decoded.operands[0].u16;
    bool owner=in.decoded.opcode==OP_OWN_STORE_LOCAL;SERVICE_TYPE(FlowDeclaration) expected;
    if((!owner && in.decoded.opcode!=OP_STORE_LOCAL) || !f->stack_count || local>=fn.locals ||
       !frf_stack(c,f,&fn,f->stack_count) || !fr_local(c,f->function,local,&expected))return frf_bad(c);
    uint32_t src=f->stack_base+f->stack_count-1,dst=f->locals_base+local;
    SERVICE_TYPE(RuntimeView) v=c->values[src].view;
    if(!owner && v.initialized && !v.owning && !v.formal &&
       v.type.tag==TAG_VOID && !v.type.mode && v.type.global_index==NVM_V2_NO_INDEX &&
       expected.tag!=TAG_VOID && !expected.mode &&
       expected.category!=NOMINAL_CATEGORY_OWNER && expected.category!=NOMINAL_CATEGORY_OWNER_RESULT) {
        if(fr_live(c,dst) && !frf_matches(c->values[dst].view,expected))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
        fr_clear(c,dst);fr_clear(c,src);return SERVICE_CONST(RUNTIME_OK);
    }
    if(expected.mode || !frf_matches(v,expected) || v.owning!=owner ||
       (owner?(!fr_empty(c,dst) || frf_held(c,src)):
        (fr_live(c,dst) && !frf_matches(c->values[dst].view,expected))))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    if(!owner)fr_clear(c,dst);
    return SERVICE_FN(runtime_move)(c,src,dst);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_next)(SERVICE_TYPE(Runtime) *c,uint8_t successor) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    uint8_t next;
    if(!frc_edge(c,f,successor,&next) || isa_is_file_indirect_call(in.decoded.opcode) || in.decoded.opcode==OP_CALL || (in.decoded.opcode==OP_CALL_REF || in.decoded.opcode==OP_FILE_CALL_REFS) || in.decoded.opcode==OP_RET ||
       successor>=in.successor_count || !frf_stack(c,f,&fn,fact.output_stack) ||
       !frf_next_fact(c,f->function,in.successors[successor],next,fact.output_stack))return frf_bad(c);
    FileRuntimeFrame completed=*f;completed.stack_count=fact.output_stack;
    completed.instruction=in.successors[successor];completed.variant=next;completed.instruction_open=false;
    if(c->kind!=FR_ACYCLIC && frc_validate(c,&completed)!=SERVICE_CONST(RUNTIME_OK))return c->report.status;
    *f=completed;
    return SERVICE_FN(runtime_site)(c,f->function,(uint16_t)f->instruction);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_call)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(HostedFunction) fn,callee;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1],child;
    uint32_t target=in.decoded.operands[0].u32;
    bool indirect=isa_is_file_indirect_call(in.decoded.opcode);
    uint32_t callable=SERVICE_CONST(RUNTIME_NO_SLOT);
    if(indirect) {
        if(c->kind!=FR_INDIRECT || !f->stack_count)return frf_bad(c);
        callable=f->stack_base+f->stack_count-1;
        if(!fr_live(c,callable))return frf_bad(c);
        const FileRuntimeValue *v=&c->values[callable];
        if(!frf_matches(v->view,fr_scalar_type(TAG_FUNCTION)) || v->view.fields ||
           v->callable_plan!=c->indirect_plan)return frf_bad(c);
        target=v->callable_target;
        if(!fr_call_target(c,f,&in,target))return frf_bad(c);
    }
    if((!indirect && in.decoded.opcode!=OP_CALL && (in.decoded.opcode!=OP_CALL_REF && in.decoded.opcode!=OP_FILE_CALL_REFS)) || in.successor_count!=1 ||
       !fact.has_obligation || fact.obligation.kind!=SERVICE_CONST(FLOW_CALL) || fact.obligation.target!=(indirect?NVM_V2_NO_INDEX:target) ||
       fact.obligation.site!=in.byte_offset || !(fact.discharged_checks&SERVICE_CONST(FLOW_CHECK_CALLEE)) ||
       !frf_stack(c,f,&fn,f->stack_count) || !fr_function(c,target,&callee))return frf_bad(c);
    uint16_t parameters=callee.code.declaration.parameters,borrowed=0,values=0;
    uint32_t references[SERVICE_CONST(FLOW_LOCALS)]={0};
    for(uint16_t i=0;i<parameters;i++) {
        SERVICE_TYPE(FlowDeclaration) type;
        if(!fr_local(c,target,i,&type))return frf_bad(c);
        if(type.mode) {
            if((indirect && in.decoded.opcode!=OP_FILE_CALL_INDIRECT_REFS) || (type.mode!=1 && type.mode!=2) || ++borrowed>SERVICE_CONST(FLOW_REFERENCES) || (in.decoded.opcode!=OP_CALL_REF && in.decoded.opcode!=OP_FILE_CALL_REFS && in.decoded.opcode!=OP_FILE_CALL_INDIRECT_REFS) ||
               in.call_references[i]>=SERVICE_CONST(FLOW_REFERENCES) ||
               (in.decoded.opcode==OP_CALL_REF && in.call_references[i]!=in.decoded.operands[1].u16))return frf_bad(c);
            uint32_t reference=f->reference_base+in.call_references[i];
            references[i]=reference;
            if(!fr_reference(c,reference))return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
            SERVICE_TYPE(FlowDeclaration) wanted=c->values[c->references[reference].owner_root].view.type;wanted.mode=type.mode;
            if(!fr_same(type,wanted) || c->references[reference].shared!=(type.mode==1))
                return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
            for(uint16_t j=0;j<i;j++) {
                SERVICE_TYPE(FlowDeclaration) prior;
                if(!fr_local(c,target,j,&prior))return frf_bad(c);
                if(prior.mode && (prior.mode!=1 || type.mode!=1) && c->references[references[j]].owner_root==c->references[reference].owner_root)
                    return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
            }
        } else {
            if((!indirect || in.decoded.opcode==OP_FILE_CALL_INDIRECT_REFS) && in.call_references[i]!=SERVICE_CONST(FLOW_NO_REFERENCE))return frf_bad(c);
            values++;
        }
    }
    if((in.decoded.opcode==OP_CALL_REF && borrowed!=1) ||
       ((in.decoded.opcode==OP_FILE_CALL_REFS || in.decoded.opcode==OP_FILE_CALL_INDIRECT_REFS) && !borrowed) ||
       parameters!=fact.obligation.parameters || borrowed!=fact.obligation.borrowed_inputs ||
       (uint32_t)values+(indirect?1u:0u)>f->stack_count || fn.staging_slots<parameters+1u+(indirect?1u:0u) ||
       callee.code.declaration.result_count!=fact.obligation.result_count ||
       (callee.code.declaration.result_count && !fr_same(callee.code.declaration.result,fact.obligation.result)))return frf_bad(c);
    uint32_t prefix=f->stack_count-values-(indirect?1u:0u);uint8_t next;
    if(!frc_edge(c,f,0,&next))return frf_bad(c);
    if((uint32_t)prefix+callee.code.declaration.result_count!=fact.output_stack ||
       fact.output_stack>fn.operand_peak || !frf_next_fact(c,f->function,in.successors[0],next,fact.output_stack))return frf_bad(c);
    uint64_t base=(uint64_t)f->stack_base+(c->mode==SERVICE_CONST(RUNTIME_VM)?prefix:fn.operand_peak);
    if(!frf_layout(c,target,base,c->frame_count,&child,&callee) || !frf_next_fact(c,target,0,0,0))return frf_bad(c);
    uint32_t end=child.stack_base+callee.operand_peak;
    for(uint32_t i=child.locals_base;i<end;i++) {
        bool argument=i>=f->stack_base+prefix && i<f->stack_base+f->stack_count;
        if(!argument && !fr_empty(c,i))return frf_bad(c);
    }
    for(uint32_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++)
        if(c->references[child.reference_base+i].live)return frf_bad(c);
    uint32_t arg=f->stack_base+prefix;uint16_t owned=0;
    for(uint16_t i=0;i<parameters;i++) {
        SERVICE_TYPE(FlowDeclaration) type;
        if(!fr_local(c,target,i,&type))return frf_bad(c);
        if(type.mode)continue;
        SERVICE_TYPE(RuntimeView) value=c->values[arg].view;
        if(!frf_matches(value,type) || value.formal || (value.owning && frf_held(c,arg)))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
        owned+=value.owning;arg++;
    }
    if(owned!=fact.obligation.owned_inputs)return frf_bad(c);
    /* Complete ALL staging before installing ANY local: VM child locals may
     * cover a later argument's original operand position. No reverse moves. */
    if(indirect)fr_clear(c,callable);
    arg=f->stack_base+prefix;
    for(uint16_t i=0;i<parameters;i++) {
        SERVICE_TYPE(FlowDeclaration) type;
        if(!fr_local(c,target,i,&type))return frf_bad(c);
        if(type.mode)continue;
        status=SERVICE_FN(runtime_move)(c,arg++,f->staging_base+i);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    }
    for(uint16_t i=0;i<parameters;i++) {
        SERVICE_TYPE(FlowDeclaration) type;
        if(!fr_local(c,target,i,&type))return frf_bad(c);
        if(type.mode)status=SERVICE_FN(runtime_bind_formal)(c,references[i],child.locals_base+i,child.reference_base+i);
        else status=SERVICE_FN(runtime_move)(c,f->staging_base+i,child.locals_base+i);
        if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    }
    f->stack_count=prefix;f->selected_target=target;f->waiting=true;f->instruction_open=false;c->frames[c->frame_count++]=child;
    if(c->kind!=FR_ACYCLIC && frc_validate(c,&child)!=SERVICE_CONST(RUNTIME_OK))return c->report.status;
    return SERVICE_FN(runtime_site)(c,child.function,0);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_return)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];uint8_t results=fn.code.declaration.result_count;
    if(in.decoded.opcode!=OP_RET || !fact.exit_checked || in.successor_count ||
       f->stack_count!=results || !frf_stack(c,f,&fn,results) || c->region_count!=f->region_base)return frf_bad(c);
    uint32_t result=results?f->stack_base:SERVICE_CONST(RUNTIME_NO_SLOT);
    if(results && (!frf_matches(c->values[result].view,fn.code.declaration.result) ||
       c->values[result].view.formal || frf_held(c,result)))return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    for(uint16_t i=0;i<fn.locals;i++) {
        SERVICE_TYPE(RuntimeView) v=c->values[f->locals_base+i].view;SERVICE_TYPE(FlowDeclaration) type;
        if(!fr_local(c,f->function,i,&type) ||
           (v.initialized && v.owning) || (type.mode && !frf_matches(v,type)))return frf_bad(c);
        if(v.formal) {
            uint32_t ref=f->reference_base+i;
            if(i>=fn.code.declaration.parameters ||
               !frf_matches(v,type) || c->values[f->locals_base+i].reference!=ref ||
               !fr_reference(c,ref) || !c->references[ref].formal ||
               c->references[ref].formal_root!=f->locals_base+i)return frf_bad(c);
        }
    }
    for(uint32_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++) {
        FileRuntimeReference *r=&c->references[f->reference_base+i];
        if(r->live && (!r->formal || i>=fn.code.declaration.parameters ||
           !fr_live(c,f->locals_base+i) || !c->values[f->locals_base+i].view.formal ||
           r->formal_root!=f->locals_base+i || !fr_reference(c,f->reference_base+i)))return frf_bad(c);
    }
    uint8_t next=0;
    FileRuntimeFrame *parent=NULL;SERVICE_TYPE(HostedFunction) caller={0};
    SERVICE_TYPE(CodeInstruction) call={0};SERVICE_TYPE(BodyInstruction) call_fact={0};uint32_t staged=SERVICE_CONST(RUNTIME_NO_SLOT);
    if(c->frame_count>1) {
        parent=&c->frames[c->frame_count-2];
        if(!parent->waiting || !fr_function(c,parent->function,&caller) ||
           !fr_instruction(c,parent->function,(uint16_t)parent->instruction,parent->variant,&call,&call_fact) ||
           call.successor_count!=1 || !fr_call_target(c,parent,&call,f->function) ||
           parent->selected_target!=f->function || !call_fact.has_obligation ||
           !frf_empty_range(c,parent->staging_base,caller.staging_slots) ||
           call_fact.output_stack!=parent->stack_count+results ||
           !frc_edge(c,parent,0,&next) ||
           !frf_next_fact(c,parent->function,call.successors[0],next,call_fact.output_stack))return frf_bad(c);
        for(uint32_t i=0;i<parent->stack_count;i++)if(!fr_live(c,parent->stack_base+i) ||
            c->values[parent->stack_base+i].view.formal)return frf_bad(c);
        staged=parent->staging_base+caller.staging_slots-1;
        if(results) { status=SERVICE_FN(runtime_move)(c,result,staged);if(status!=SERVICE_CONST(RUNTIME_OK))return status; }
    }
    /* Removing formal aliases never releases an ancestor borrow epoch. */
    for(uint32_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++)if(c->references[f->reference_base+i].live) {
        status=SERVICE_FN(runtime_end_reference)(c,f->reference_base+i);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    }
    if(!parent) {
        if(f->function!=c->current_root)return frf_bad(c);
        return SERVICE_FN(runtime_complete_root)(c,result);
    }
    uint32_t end=f->stack_base+fn.operand_peak;
    for(uint32_t i=f->locals_base;i<end;i++)fr_clear(c,i);
    *f=(FileRuntimeFrame){0};c->frame_count--;
    c->transfer_active=true;
    if(results) {
        status=SERVICE_FN(runtime_move)(c,staged,parent->stack_base+parent->stack_count);
        if(status!=SERVICE_CONST(RUNTIME_OK)){c->transfer_active=false;return status;}
    }
    c->transfer_active=false;
    if(!frf_stack(c,parent,&caller,call_fact.output_stack))return frf_bad(c);
    parent->stack_count=call_fact.output_stack;parent->waiting=false;parent->instruction=call.successors[0];
    parent->variant=next;parent->instruction_open=false;parent->selected_target=0;
    if(c->kind!=FR_ACYCLIC && frc_validate(c,parent)!=SERVICE_CONST(RUNTIME_OK))return c->report.status;
    return SERVICE_FN(runtime_site)(c,parent->function,(uint16_t)parent->instruction);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_region_begin)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    if(in.decoded.opcode!=OP_REGION_BEGIN || c->region_count<f->region_base ||
       c->region_count-f->region_base>=SERVICE_CONST(FLOW_REFERENCES))return frf_bad(c);
    return SERVICE_FN(runtime_region_begin)(c);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_region_end)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(in.decoded.opcode!=OP_REGION_END || c->region_count<=c->frames[c->frame_count-1].region_base)return frf_bad(c);
    return SERVICE_FN(runtime_region_end)(c);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_borrow)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    uint16_t reference=in.decoded.operands[0].u16,local=in.decoded.operands[1].u16;
    if((in.decoded.opcode!=OP_BORROW_LOCAL_EXCLUSIVE && in.decoded.opcode!=OP_BORROW_LOCAL_SHARED) || reference>=SERVICE_CONST(FLOW_REFERENCES) ||
       local>=fn.locals || c->region_count<=f->region_base)return frf_bad(c);
    return in.decoded.opcode==OP_BORROW_LOCAL_SHARED?
        SERVICE_FN(runtime_borrow_shared)(c,f->locals_base+local,f->reference_base+reference):
        SERVICE_FN(runtime_borrow)(c,f->locals_base+local,f->reference_base+reference);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_frame_end_reference)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];uint16_t index=in.decoded.operands[0].u16;
    if(in.decoded.opcode!=OP_FILE_END_BORROW || index>=SERVICE_CONST(FLOW_REFERENCES) ||
       !fr_reference(c,f->reference_base+index) || c->references[f->reference_base+index].formal)
        return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    return SERVICE_FN(runtime_end_reference)(c,f->reference_base+index);
}
