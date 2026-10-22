/* One synchronous scope. No owner/VM/heap layout or persistent policy changes. */
static VmResult vm_uf_call_root(VmUnionFieldInvocation *v,uint32_t function,VmUnionFieldRunOutcome *outcome) {
    VmState *vm=v->vm;const VmUnionFieldPrivate *p=v->owner;
    if(!v->active || !vm_uf_idle(p) || function>=p->counts.functions)return VM_ERR_TYPE_ERROR;
    const NvmRecordArrayExecutionFunction *root=&p->functions[function];
    const NvmFunctionEntry *fn=&root->signature;
    if(fn->arity || fn->local_count>vm->stack_capacity ||
       root->maximum_stack>vm->stack_capacity-fn->local_count)return VM_ERR_STACK_OVERFLOW;
    v->active_root=function;v->root_base=0;
    VmCallFrame *frame=&vm->frames[0];memset(frame,0,sizeof *frame);
    frame->fn_idx=function;frame->local_count=fn->local_count;frame->instruction_ip=fn->code_offset;
    frame->owned_callable=val_void();frame->module=v->module;
    for(uint16_t i=0;i<fn->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=function;vm->ip=fn->code_offset;
    for(;;) {
        VmTrap trap=vm_core_execute_union(vm,NULL,NULL,NULL,NULL,NULL,TAG_VOID,v);
        if(trap.type==TRAP_ASSERT) {
            bool success=trap.data.assert_check.condition.tag==TAG_BOOL && trap.data.assert_check.condition.as.boolean;
            vm_release(&vm->heap,trap.data.assert_check.condition);
            if(!success)return vm_error(vm,VM_ERR_ASSERT_FAILED,"Assertion failed");
            continue;
        }
        if(trap.type==TRAP_ERROR)return trap.data.error.code;
        if(trap.type==TRAP_HALT) {*outcome=VM_UNION_FIELD_HALTED;return VM_OK;}
        if(trap.type!=TRAP_NONE)return VM_ERR_INVALID_OPCODE;
        if(vm->frame_count || vm->stack_size!=fn->result_count ||
           (fn->result_count && (vm->stack[0].tag!=fn->result_tag || !vm_uf_shell(p,vm->stack[0]))))return VM_ERR_TYPE_ERROR;
        *outcome=VM_UNION_FIELD_RETURNED;return VM_OK;
    }
}
static void vm_uf_unwind(VmUnionFieldInvocation *v) {
    VmState *vm=v->vm;
    while(vm->stack_size)vm_release(&vm->heap,stack_pop(vm));
    for(uint32_t i=0;i<vm->frame_count;i++) {
        vm_release(&vm->heap,vm->frames[i].owned_callable);
        memset(&vm->frames[i],0,sizeof vm->frames[i]);
    }
    vm->frame_count=0;vm->activation_floor=0;vm->current_fn=0;vm->ip=0;
    vm_clear_reference_activations(vm);effect_prune(vm,0);
    vm->module=v->module;vm->root_module=v->module;vm->halt_requested=false;
}
VmResult vm_union_field_private_execute(VmUnionFieldPrivate *p,VmUnionFieldRunResult *out) {
    if(!p || !out || !vm_rmo_disjoint(p,sizeof *p,out,sizeof *out) || !vm_uf_idle(p) || !vm_uf_invocation_tables(p))return VM_ERR_TYPE_ERROR;
    VmState *vm=&p->vm;
    if(!vm_rmo_disjoint(out,sizeof *out,vm->stack,(size_t)vm->stack_capacity*sizeof *vm->stack) ||
       !vm_rmo_disjoint(out,sizeof *out,vm->globals,(size_t)vm->global_capacity*sizeof *vm->globals))return VM_ERR_TYPE_ERROR;
    uint32_t entry=p->counts.entry,initializer=p->counts.initializer;
    if(!(vm->module->header.flags&NVM_FLAG_HAS_MAIN) || entry>=p->counts.functions || entry!=vm->module->header.entry_point)return VM_ERR_TYPE_ERROR;
    const NvmFunctionEntry *root=&p->functions[entry].signature;
    if(root->arity || root->result_count!=1 || root->result_tag!=TAG_INT ||
       !vm_rm_signature_same(root,&vm->module->functions[entry]))return VM_ERR_TYPE_ERROR;
    if(initializer!=UINT32_MAX) {
        if(initializer>=p->counts.functions)return VM_ERR_TYPE_ERROR;
        const NvmFunctionEntry *init=&p->functions[initializer].signature;
        if(init->arity || !((init->result_count==0 && init->result_tag==TAG_VOID) ||
           (init->result_count==1 && init->result_tag==TAG_INT)) ||
           !vm_rm_signature_same(init,&vm->module->functions[initializer]))return VM_ERR_TYPE_ERROR;
    }
    vm->last_error=VM_OK;vm->error_msg[0]='\0';
    VmUnionFieldInvocation view={p,vm,vm->module,UINT32_MAX,0,true};
    VmUnionFieldRunResult result;memset(&result,0,sizeof result);result.result=val_void();
    VmResult status=VM_OK;
    if(initializer!=UINT32_MAX) {
        VmUnionFieldRunOutcome ignored;
        status=vm_uf_call_root(&view,initializer,&ignored);
        vm_uf_unwind(&view); /* Both returned and halted initializer drain here. */
    }
    if(status==VM_OK)status=vm_uf_call_root(&view,entry,&result.outcome);
    if(status==VM_OK && result.outcome==VM_UNION_FIELD_RETURNED)result.result=vm->stack[0];
    if(status!=VM_OK && vm->last_error==VM_OK)vm_error(vm,status,"I stopped the closed private union invocation.");
    vm_uf_unwind(&view);
    view.active=false;view.owner=NULL;view.vm=NULL;view.module=NULL;
    if(status==VM_OK)*out=result;
    return status;
}
