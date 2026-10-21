/* Root supervision only. All guest control flow uses the real scoped core. */
static VmResult vm_rmx_call_root(VmRecordMapInvocation *v,uint32_t function,uint8_t value_tag) {
    VmRecordMapPrivate *p=v->owner;VmState *vm=v->vm;
    if(function>=p->counts.functions)return VM_ERR_UNDEFINED_FUNCTION;
    const NvmFunctionEntry *fn=&p->functions[function].signature;
    if(fn->arity)return VM_ERR_TYPE_ERROR;
    if(vm->frame_count || vm->stack_size>vm->stack_capacity ||
       fn->local_count>vm->stack_capacity-vm->stack_size ||
       p->functions[function].maximum_stack>vm->stack_capacity-vm->stack_size-fn->local_count)
        return VM_ERR_STACK_OVERFLOW;
    /* Reset both identities for initializer and entry independently. */
    uint32_t base=vm->stack_size;v->active_root=function;v->root_base=base;
    VmCallFrame *frame=&vm->frames[0];memset(frame,0,sizeof *frame);
    frame->fn_idx=function;frame->stack_base=base;frame->local_count=fn->local_count;
    frame->instruction_ip=fn->code_offset;frame->owned_callable=val_void();frame->module=vm->module;
    for(uint16_t i=0;i<fn->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=function;vm->ip=fn->code_offset;
    vm->activation_floor=0;vm->halt_requested=false;
    if(p->maximum_frames<1)p->maximum_frames=1;
    for(;;) {
        VmTrap trap=vm_core_execute_pair(vm,NULL,NULL,NULL,v,NULL,value_tag);
        if(trap.type==TRAP_ASSERT) {
            bool success=val_truthy(trap.data.assert_check.condition);
            vm_release(&vm->heap,trap.data.assert_check.condition);
            if(!success)return vm_error(vm,VM_ERR_ASSERT_FAILED,"Assertion failed");
            continue;
        }
        if(trap.type==TRAP_ERROR)return trap.data.error.code;
        if(trap.type!=TRAP_NONE)return VM_ERR_INVALID_OPCODE;
        if(vm->frame_count || vm->stack_size!=base+fn->result_count)return VM_ERR_TYPE_ERROR;
        if(fn->result_count && (!vm_rmx_value_valid(p,vm->stack[base]) ||
           !result_tag_matches(fn->result_tag,vm->stack[base].tag)))return VM_ERR_TYPE_ERROR;
        return VM_OK;
    }
}
static void vm_rmx_unwind(VmRecordMapPrivate *p) {
    VmState *vm=&p->vm;
    while(vm->stack_size)vm_release(&vm->heap,stack_pop(vm));
    for(uint32_t i=0;i<vm->frame_count;i++) {
        vm_release(&vm->heap,vm->frames[i].owned_callable);
        memset(&vm->frames[i],0,sizeof vm->frames[i]);
    }
    vm->frame_count=0;vm->activation_floor=0;vm->current_fn=0;vm->ip=0;
    vm->module=nvm_record_map_image_borrow(p->image);vm->halt_requested=false;
}
VmResult vm_record_map_private_run(VmRecordMapPrivate *p) {
    if(!p || !p->prepared || p->busy || !pthread_equal(p->vm.owner_thread,pthread_self()) ||
       p->epoch==UINT64_MAX)return VM_ERR_TYPE_ERROR;
    VmState *vm=&p->vm;
    if(vm->stack_size || vm->frame_count || vm->activation_floor || vm->callbacks || vm->linked_module_count ||
       vm->handler_count || vm->references.active || vm->callee_references.active || vm->reference_generation || vm->opcode_trace ||
       vm->heap.gc_running || vm->verified || vm->module!=nvm_record_map_image_borrow(p->image) ||
       vm->root_module!=vm->module || !vm->decoded_module_valid || !vm->dispatch_module_valid ||
       vm->dispatch_profile.fuse_load_local_field)return VM_ERR_TYPE_ERROR;
    /* Fixed per-invocation query budget, not prepaid guest execution work.
     * Read only this owner's immutable image plan, before any activation. */
    uint64_t pair_work=0;const uint64_t pair_limit=NVM_RECORD_MAP_PAIR_QUERY_WORK;
    uint8_t value_tag=TAG_VOID;
    if(NVM_RECORD_MAP_PAIR_QUERY_WORK>pair_limit-pair_work)return VM_ERR_TYPE_ERROR;
    pair_work+=NVM_RECORD_MAP_PAIR_QUERY_WORK;
    if(pair_work!=pair_limit || p->plan!=nvm_record_map_image_plan(p->image) ||
       !nvm_record_map_execution_uniform_pair(p->plan,&value_tag))return VM_ERR_TYPE_ERROR;
    /* Only constructor-owned constants and prior closed result/global graphs
     * exist here. The policy stays enabled for those roots after this call. */
    vm->heap.private_record_dag=true;
    p->epoch++;p->busy=true;vm->last_error=VM_OK;vm->error_msg[0]='\0';
    VmRecordMapInvocation view={p,vm,vm->module,p->epoch,UINT32_MAX,0,true};
    VmResult status=VM_OK;NanoValue replacement=val_void();bool publish=false;
    if(!(vm->module->header.flags&NVM_FLAG_HAS_MAIN))status=VM_ERR_UNDEFINED_FUNCTION;
    if(status==VM_OK && p->counts.initializer!=UINT32_MAX)status=vm_rmx_call_root(&view,p->counts.initializer,value_tag);
    if(status==VM_OK)status=vm_rmx_call_root(&view,p->counts.entry,value_tag);
    if(status==VM_OK) {
        if(p->functions[p->counts.entry].signature.result_count)replacement=stack_pop(vm);
        publish=true;
    }
    if(status!=VM_OK && vm->last_error==VM_OK)vm_error(vm,status,"I stopped the closed private map invocation.");
    vm_rmx_unwind(p);
    if(publish) {
        NanoValue previous=p->result;p->result=replacement;p->has_result=true;
        vm_release(&vm->heap,previous);
    }
    /* Invalidate the view before any caller can observe the idle owner. */
    view.active=false;view.owner=NULL;view.vm=NULL;view.module=NULL;
    p->last_status=status;p->busy=false;return status;
}
bool vm_record_map_private_stats(const VmRecordMapPrivate *p,VmRecordMapPrivateStats *out) {
    if(!p || !out || !p->prepared || p->busy || !pthread_equal(p->vm.owner_thread,pthread_self()))return false;
    VmRecordMapPrivateStats value={p->epoch,p->vm.heap.stats.num_objects,
        p->vm.heap.stats.allocated-p->vm.heap.stats.freed,p->vm.frame_count,p->vm.stack_size,
        p->maximum_frames,p->last_status,p->has_result};
    *out=value;return true;
}
