#include "record_map_borrowed_private.h"
static VmResult vm_rmb_call_root(VmRecordMapBorrowedInvocation *v,uint32_t function) {
    const VmRecordMapPublicPreparation *p=v->preparation;VmState *vm=v->vm;
    if(function>=p->counts.carrier.functions)return VM_ERR_UNDEFINED_FUNCTION;
    const NvmFunctionEntry *fn=&p->functions[function].signature;
    if(fn->arity || vm->frame_count)return VM_ERR_TYPE_ERROR;
    uint32_t base=vm->stack_size;
    if(base>vm->stack_capacity || fn->local_count>vm->stack_capacity-base ||
       p->functions[function].maximum_stack>vm->stack_capacity-base-fn->local_count)return VM_ERR_STACK_OVERFLOW;
    v->active_root=function;v->root_base=base;
    VmCallFrame *frame=&vm->frames[0];memset(frame,0,sizeof *frame);
    frame->fn_idx=function;frame->stack_base=base;frame->local_count=fn->local_count;
    frame->instruction_ip=fn->code_offset;frame->owned_callable=val_void();frame->module=vm->module;
    for(uint16_t i=0;i<fn->local_count;i++)vm->stack[vm->stack_size++]=val_void();
    vm->frame_count=1;vm->current_fn=function;vm->ip=fn->code_offset;
    vm->activation_floor=0;vm->halt_requested=false;
    for(;;) {
        VmTrap trap=vm_core_execute_views(vm,NULL,NULL,NULL,NULL,v);
        if(trap.type==TRAP_ASSERT) {
            bool success=val_truthy(trap.data.assert_check.condition);
            vm_release(&vm->heap,trap.data.assert_check.condition);
            if(!success)return vm_error(vm,VM_ERR_ASSERT_FAILED,"Assertion failed");
            continue;
        }
        if(trap.type==TRAP_ERROR)return trap.data.error.code;
        if(trap.type!=TRAP_NONE)return VM_ERR_INVALID_OPCODE;
        if(vm->frame_count || vm->stack_size!=base+fn->result_count)return VM_ERR_TYPE_ERROR;
        if(fn->result_count && !result_tag_matches(fn->result_tag,vm->stack[base].tag))return VM_ERR_TYPE_ERROR;
        return VM_OK;
    }
}
static void vm_rmb_unwind(VmState *vm) {
    while(vm->stack_size)vm_release(&vm->heap,stack_pop(vm));
    for(uint32_t i=0;i<vm->frame_count;i++) {
        vm_release(&vm->heap,vm->frames[i].owned_callable);
        vm->frames[i].owned_callable=val_void();
    }
    vm->frame_count=0;vm->activation_floor=0;
    vm_clear_reference_activations(vm);effect_prune(vm,0);
}
static VmResult vm_execute_record_maps(const NvmModule *module,VmState *vm) {
    if(!vm || !module || module!=vm->module || module!=vm->root_module)return VM_ERR_TYPE_ERROR;
    VmRecordMapPublicPreparation *p=NULL;
    NvmArrayEligibilityResult prepared=vm_record_map_public_prepare(vm,&p);
    if(prepared.status!=NVM_ARRAY_ELIGIBLE)
        return prepared.status==NVM_ARRAY_MEMORY?VM_ERR_MEMORY:VM_ERR_TYPE_ERROR;
    VmResult status=VM_ERR_TYPE_ERROR;
    uint32_t entry=p->counts.carrier.entry,initializer=p->counts.carrier.initializer;
    if(!(module->header.flags&NVM_FLAG_HAS_MAIN) || entry>=p->counts.carrier.functions ||
       p->functions[entry].signature.arity || p->functions[entry].signature.result_count!=1 ||
       p->functions[entry].signature.result_tag!=TAG_INT)goto done;
    if(initializer!=UINT32_MAX && (initializer>=p->counts.carrier.functions ||
       p->functions[initializer].signature.arity ||
       !((p->functions[initializer].signature.result_count==0 && p->functions[initializer].signature.result_tag==TAG_VOID) ||
         (p->functions[initializer].signature.result_count==1 && p->functions[initializer].signature.result_tag==TAG_INT))))goto done;
    /* The managed C inputs are exclusively stable. Recheck the exact borrowed
     * identities before the sole prepaid stack reallocation, not after it. */
    if(vm->module!=module || vm->root_module!=module || vm->stack!=p->original_stack ||
       vm->stack_capacity!=p->original_stack_capacity || vm->stack_size || vm->frame_count ||
       vm->activation_floor || vm->callbacks || vm->linked_module_count || vm->handler_count ||
       vm->references.active || vm->callee_references.active || vm->reference_generation || vm->opcode_trace ||
       vm->heap.gc_running || vm_rm_private_dag(&vm->heap) ||
       !pthread_equal(vm->owner_thread,pthread_self()) || !vm->decoded_module_valid || !vm->dispatch_module_valid)
        goto done;
    status=stack_reserve(vm,VM_RM_ROOT_CAPACITY);
    if(status!=VM_OK)goto done;
    if(vm->stack_capacity!=p->rounded_stack_capacity){status=VM_ERR_TYPE_ERROR;goto done;}
    vm->last_error=VM_OK;vm->error_msg[0]='\0';
    VmRecordMapBorrowedInvocation view={p,vm,module,UINT32_MAX,0,true};
    if(initializer!=UINT32_MAX) {
        status=vm_rmb_call_root(&view,initializer);
        if(status==VM_OK)while(vm->stack_size)vm_release(&vm->heap,stack_pop(vm));
    }
    if(status==VM_OK)status=vm_rmb_call_root(&view,entry);
    if(status!=VM_OK) {
        if(vm->last_error==VM_OK)vm_error(vm,status,"I stopped the borrowed map invocation.");
        vm_rmb_unwind(vm);
    }
    view.active=false;view.preparation=NULL;view.vm=NULL;view.module=NULL;
done:
    vm_record_map_public_preparation_free(p);return status;
}

#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
VmResult vm_record_map_borrowed_execute_private(const NvmModule *module,VmState *vm) {
    return vm_execute_record_maps(module,vm);
}
#endif
