/* I enter the real scoped VM and supervise only its assertion/completion traps.
 * All branches, opcode dispatch, CALL and RET remain in vm_core_execute_scoped. */
static VmResult vm_ra_call_root(VmRecordArrayPrivate *p,uint32_t function) {
    VmState *vm=&p->vm;
    if(function>=p->counts.functions)return VM_ERR_UNDEFINED_FUNCTION;
    const NvmFunctionEntry *fn=&p->functions[function].signature;
    if(fn->arity)return VM_ERR_TYPE_ERROR;
    if(vm->frame_count || fn->local_count>vm->stack_capacity-vm->stack_size)return VM_ERR_STACK_OVERFLOW;
    uint32_t base=vm->stack_size;
    VmCallFrame *frame=&vm->frames[0];memset(frame,0,sizeof *frame);
    frame->fn_idx=function;frame->stack_base=base;frame->local_count=fn->local_count;
    frame->instruction_ip=fn->code_offset;
    frame->owned_callable=val_void();frame->module=vm->module;
    for(uint16_t i=0;i<fn->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=function;vm->ip=fn->code_offset;
    vm->activation_floor=0;vm->halt_requested=false;
    if(p->maximum_frames<1)p->maximum_frames=1;
    for(;;) {
        VmTrap trap=vm_core_execute_scoped(vm,NULL,NULL,p);
        if(trap.type==TRAP_ASSERT) {
            bool success=val_truthy(trap.data.assert_check.condition);
            vm_release(&vm->heap,trap.data.assert_check.condition);
            if(!success)return VM_ERR_ASSERT_FAILED;
            continue;
        }
        if(trap.type==TRAP_ERROR)return trap.data.error.code;
        if(trap.type!=TRAP_NONE)return VM_ERR_INVALID_OPCODE;
        if(vm->frame_count || vm->stack_size!=base+fn->result_count)return VM_ERR_TYPE_ERROR;
        if(fn->result_count && (!vm_ra_value_valid(p,vm->stack[base]) ||
            !result_tag_matches(fn->result_tag,vm->stack[base].tag)))return VM_ERR_TYPE_ERROR;
        return VM_OK;
    }
}
static void vm_ra_unwind(VmRecordArrayPrivate *p) {
    VmState *vm=&p->vm;
    vm_frames_clear_bindings(vm, 0);
    while(vm->stack_size)vm_release(&vm->heap,stack_pop(vm));
    for(uint32_t i=0;i<vm->frame_count;i++) {
        vm_release(&vm->heap,vm->frames[i].owned_callable);
        memset(&vm->frames[i],0,sizeof vm->frames[i]);
    }
    vm->frame_count=0;vm->activation_floor=0;vm->current_fn=0;vm->ip=0;
    vm->module=nvm_record_array_image_borrow(p->image);
    vm->halt_requested=false;
}
VmResult vm_record_array_private_run(VmRecordArrayPrivate *p) {
    if(!p || !p->prepared || p->busy || !pthread_equal(p->vm.owner_thread,pthread_self()))return VM_ERR_TYPE_ERROR;
    if(p->epoch==UINT64_MAX)return VM_ERR_TYPE_ERROR;
    /* Prior result and globals remain rooted throughout initialization and
     * entry. The selected initializer's at-most-one result stays below the
     * entry frame, matching ordinary vm_execute's actual stack-base offset. */
    p->epoch++;p->busy=true;p->vm.last_error=VM_OK;p->vm.error_msg[0]='\0';
    VmResult result=VM_OK;NanoValue replacement=val_void();bool replace=false;
    if(!(p->vm.module->header.flags&NVM_FLAG_HAS_MAIN))result=VM_ERR_UNDEFINED_FUNCTION;
    if(result==VM_OK && p->counts.initializer!=UINT32_MAX)
        result=vm_ra_call_root(p,p->counts.initializer);
    if(result==VM_OK)result=vm_ra_call_root(p,p->counts.entry);
    if(result==VM_OK) {
        if(p->functions[p->counts.entry].signature.result_count)replacement=stack_pop(&p->vm);
        replace=true;
    }
    /* This cannot allocate or replace the first runtime failure. It releases
     * initializer results and all suspended frame roots before publication. */
    vm_ra_unwind(p);
    if(replace) {
        NanoValue previous=p->result;
        p->result=replacement;p->has_result=true;
        vm_release(&p->vm.heap,previous);
    }
    p->last_status=result;p->busy=false;return result;
}
