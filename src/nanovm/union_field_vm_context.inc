/* Closed ordinary-heap execution, distinct from private array/map policies. */
struct VmUnionFieldInvocation {
    VmUnionFieldPrivate *owner;
    VmState *vm;
    const NvmModule *module;
    uint32_t active_root,root_base;
    bool active;
};
static bool vm_uf_shell(const VmUnionFieldPrivate *p,NanoValue value) {
    if(value.tag==TAG_VOID || value.tag==TAG_INT || value.tag==TAG_U8 ||
       value.tag==TAG_FLOAT || value.tag==TAG_BOOL)return true;
    if(value.tag!=TAG_STRING && value.tag!=TAG_STRUCT && value.tag!=TAG_UNION)return false;
    if(!value.as.obj)return false;
    const VmHeapHeader *h=value.as.obj;
    if(h->obj_type!=value.tag || !h->ref_count || h->buffered>1 ||
       (h->colour!=VM_GC_BLACK && h->colour!=VM_GC_PURPLE) ||
       (h->buffered && h->colour!=VM_GC_PURPLE))return false;
    if(value.tag==TAG_STRING)return !h->buffered && h->colour==VM_GC_BLACK;
    if(value.tag==TAG_STRUCT) {
        const VmStruct *s=value.as.sval;
        return s->def_idx==0 && p->record_layout==1 && s->field_count==1 && s->fields && !s->field_names;
    }
    const VmUnion *u=value.as.uval;
    return u->def_idx==0 && p->union_layout==0 && u->variant<2 &&
        u->field_count==p->variants[u->variant].field_count && (!u->field_count || u->fields);
}
static bool vm_uf_storage(const VmUnionFieldPrivate *p) {
    if(!p || !p->prepared || !p->image)return false;
    const VmState *vm=&p->vm;const NvmModule *m=nvm_union_field_image_module_internal(p->image);
    return m && vm->module==m && vm->root_module==m &&
        pthread_equal(vm->owner_thread,pthread_self()) && !vm->verified &&
        !vm->heap.gc_running && !vm_rm_private_dag(&vm->heap) &&
        !vm->callbacks && !vm->linked_module_count && !vm->handler_count && !vm->opcode_trace &&
        !vm->activation_floor && !vm->references.active && !vm->callee_references.active && !vm->reference_generation &&
        !vm->halt_requested && vm->decoded_module_valid && vm->dispatch_module_valid &&
        !vm->dispatch_profile.fuse_load_local_field &&
        p->counts.functions && p->counts.functions<=256 && m->function_count==p->counts.functions && m->functions &&
        p->counts.instructions<=65536 && p->instructions &&
        vm->decoded_module.function_count==p->counts.functions && vm->decoded_module.functions &&
        vm->dispatch_module.function_count==p->counts.functions && vm->dispatch_module.functions &&
        vm->stack && vm->stack_capacity==VM_UF_ROOT_CAPACITY && vm->stack_size<=vm->stack_capacity &&
        vm->global_capacity==p->counts.globals && vm->global_count==p->counts.globals &&
        (!p->counts.globals || vm->globals) &&
        vm->module_constants.count==m->string_count && (!m->string_count || vm->module_constants.strings);
}
static bool vm_uf_idle(const VmUnionFieldPrivate *p) {
    if(!vm_uf_storage(p) || p->vm.stack_size || p->vm.frame_count)return false;
    for(uint32_t i=0;i<NVM_OWNED_MAX_FUNCTIONS-2;i++)
        if(p->vm.value_references[i].active)return false;
    for(uint32_t i=0;i<VM_MAX_FRAMES-NVM_OWNED_MAX_FUNCTIONS;i++)
        if(p->vm.effect_references[i])return false;
    return true;
}
/* Once per invocation, before activation: four-probe counts/function/variant
 * getters plus fixed named-field comparisons, at most256 function rows. No
 * decoding, module traversal, allocation or constructor-budget mutation. */
static bool vm_uf_invocation_tables(const VmUnionFieldPrivate *p) {
    NvmRecordArrayExecutionCounts c;
    if(!nvm_union_field_image_counts(p->image,&c))return false;
#define VM_UF_COUNT(x) do {if(c.x!=p->counts.x)return false;}while(0)
    VM_UF_COUNT(functions);VM_UF_COUNT(instructions);VM_UF_COUNT(globals);
    VM_UF_COUNT(initializer);VM_UF_COUNT(entry);VM_UF_COUNT(frame_limit);
    VM_UF_COUNT(strings);VM_UF_COUNT(sections);VM_UF_COUNT(metadata);
    VM_UF_COUNT(debug);VM_UF_COUNT(records);
#undef VM_UF_COUNT
    if(c.functions>256 || c.globals>256 || c.frame_limit!=VM_MAX_FRAMES || c.records!=1 ||
       p->record_layout!=1 || p->union_layout!=0)return false;
    for(uint32_t i=0;i<c.functions;i++) {
        NvmRecordArrayExecutionFunction original;
        const NvmRecordArrayExecutionFunction *f=&p->functions[i];
        if(!nvm_union_field_image_function(p->image,i,&original) ||
           !vm_rm_signature_same(&f->signature,&original.signature) ||
           !vm_rm_signature_same(&f->signature,&p->vm.module->functions[i]) ||
           f->instruction_start!=original.instruction_start || f->instruction_count!=original.instruction_count ||
           f->maximum_stack!=original.maximum_stack || f->parameter_tags_present!=original.parameter_tags_present)return false;
    }
    for(uint16_t i=0;i<2;i++) {
        NvmUnionVariantFact original;
        if(!nvm_union_field_image_variant(p->image,0,i,&original) ||
           original.layout!=p->variants[i].layout || original.name_idx!=p->variants[i].name_idx ||
           original.field_offset!=p->variants[i].field_offset || original.field_count!=p->variants[i].field_count)return false;
    }
    return true;
}
static bool vm_uf_context(const VmUnionFieldInvocation *v,const VmState *vm) {
    if(!v || !v->active || !v->owner || v->vm!=vm || vm!=&v->owner->vm ||
       !vm_uf_storage(v->owner) || v->module!=vm->module)return false;
    const VmUnionFieldPrivate *p=v->owner;
    if(!vm->frame_count || vm->frame_count>VM_MAX_FRAMES || vm->current_fn>=p->counts.functions ||
       v->active_root>=p->counts.functions || vm->frames[0].fn_idx!=v->active_root ||
       vm->frames[0].stack_base!=v->root_base)return false;
    for(uint32_t i=0;i<vm->frame_count;i++) {
        const VmCallFrame *f=&vm->frames[i];
        if(f->fn_idx>=p->counts.functions || f->module!=v->module || f->effect_owner || f->closure ||
           f->owned_callable.tag!=TAG_VOID || f->local_count!=p->functions[f->fn_idx].signature.local_count ||
           f->stack_base>vm->stack_size || f->local_count>vm->stack_size-f->stack_base ||
           !vm_rm_signature_same(&p->functions[f->fn_idx].signature,&v->module->functions[f->fn_idx]))return false;
    }
    const NvmFunctionEntry *fn=&p->functions[vm->current_fn].signature;
    return vm->frames[vm->frame_count-1].fn_idx==vm->current_fn &&
        vm->ip>=fn->code_offset && vm->ip-fn->code_offset<=fn->code_length;
}
static bool vm_uf_base_same(const NvmRecordArrayExecutionInstruction *a,const NvmRecordArrayExecutionInstruction *b) {
#define VM_UF_SAME(x) do {if(a->x!=b->x)return false;}while(0)
    VM_UF_SAME(function);VM_UF_SAME(pc);VM_UF_SAME(next_pc);VM_UF_SAME(callee);
    VM_UF_SAME(pops);VM_UF_SAME(pushes);VM_UF_SAME(obligations);VM_UF_SAME(opcode);
    VM_UF_SAME(recipe);VM_UF_SAME(successor_count);VM_UF_SAME(operand_count);
    for(unsigned i=0;i<MAX_OPERANDS;i++){VM_UF_SAME(operand_types[i]);VM_UF_SAME(operand_bits[i]);}
    for(unsigned i=0;i<2;i++)VM_UF_SAME(successors[i]);
#undef VM_UF_SAME
    return true;
}
static VmResult vm_uf_preflight(const VmUnionFieldInvocation *v,const VmDispatchInstruction *in) {
    if(!vm_uf_context(v,v?v->vm:NULL))return VM_ERR_TYPE_ERROR;
    VmState *vm=v->vm;const VmUnionFieldPrivate *p=v->owner;
    const NvmRecordArrayExecutionFunction *fn=&p->functions[vm->current_fn];
    const VmCallFrame *frame=&vm->frames[vm->frame_count-1];
    const VmDecodedFunction *decoded=&vm->decoded_module.functions[vm->current_fn];
    const VmDispatchFunction *dispatch=&vm->dispatch_module.functions[vm->current_fn];
    if(!in || !decoded->instruction_indices || !dispatch->offset_to_index ||
       in->byte_offset>=fn->signature.code_length || decoded->code_size!=fn->signature.code_length ||
       dispatch->code_size!=fn->signature.code_length)return VM_ERR_DECODE;
    uint32_t encoded=decoded->instruction_indices[in->byte_offset];
    if(!encoded || encoded>fn->instruction_count || dispatch->offset_to_index[in->byte_offset]!=encoded ||
       fn->instruction_start>p->counts.instructions || encoded>p->counts.instructions-fn->instruction_start)return VM_ERR_DECODE;
    const NvmUnionFieldExecutionInstruction *row=&p->instructions[fn->instruction_start+encoded-1];
    NvmUnionFieldExecutionInstruction original;
    /* One four-probe immutable getter per dispatched row. This is invocation
     * validation, not a claim of prepaid constructor or guest fuel accounting. */
    if(!nvm_union_field_image_instruction(p->image,fn->instruction_start+encoded-1,&original) ||
       !vm_uf_base_same(&row->base,&original.base) || !vm_uf_site_same(&row->site,&original.site))return VM_ERR_DECODE;
    const NvmRecordArrayExecutionInstruction *fact=&row->base;const NvmUnionFieldSite *site=&row->site;
    NvmRecordArrayExecutionInstruction recipe={0};
    if(!vm_uf_recipe(fact->opcode,&recipe) || recipe.recipe!=fact->recipe || recipe.obligations!=fact->obligations ||
       fact->function!=vm->current_fn || fact->pc!=in->byte_offset || fact->next_pc!=in->next_byte_offset ||
       fact->operand_count>MAX_OPERANDS || fact->successor_count>2 || !vm_rm_instruction_same(fact,&in->instruction) ||
       in->super_op!=VM_SUPER_NONE || in->super_operand || in->call_handle.resolved || in->call_handle.module ||
       in->call_handle.function || in->call_handle.function_index || in->call_target!=fact->callee ||
       in->next_index!=(encoded<fn->instruction_count?encoded:VM_DISPATCH_NO_INDEX))return VM_ERR_DECODE;
    uint32_t target=UINT32_MAX,branch=UINT32_MAX;
    if(fact->opcode==OP_JMP || fact->opcode==OP_JMP_TRUE || fact->opcode==OP_JMP_FALSE) {
        if(!fact->successor_count || fact->successors[0]>fn->signature.code_length)return VM_ERR_DECODE;
        uint32_t pc=fact->successors[0];target=fn->signature.code_offset+pc;
        if(pc<fn->signature.code_length) {
            uint32_t index=decoded->instruction_indices[pc];
            if(!index || index>fn->instruction_count)return VM_ERR_DECODE;
            branch=index-1;
        }
    }
    if(in->branch_target_offset!=target || in->branch_target!=branch)return VM_ERR_DECODE;
    if(site->state==UFS_NOT_ESTABLISHED)return VM_ERR_TYPE_ERROR;
    uint32_t operands=vm->stack_size-frame->stack_base-frame->local_count;
    if(operands>256)return VM_ERR_STACK_OVERFLOW;
    if(fact->pops>operands)return VM_ERR_STACK_UNDERFLOW;
    if(fact->pushes>vm->stack_capacity-(vm->stack_size-fact->pops))return VM_ERR_STACK_OVERFLOW;
    for(uint16_t i=0;i<fact->pops;i++)if(!vm_uf_shell(p,stack_peek(vm,i)))return VM_ERR_TYPE_ERROR;
    VmResult checked=vm_rmx_scalar_inputs(vm,fact->opcode);if(checked!=VM_OK)return checked;
    NanoValue retained=val_void();const DecodedInstruction *d=&in->instruction;
    switch(fact->opcode) {
    case OP_PUSH_STR:
        if(d->operands[0].u32>=vm->module_constants.count)return VM_ERR_DECODE;
        retained=val_string(vm->module_constants.strings[d->operands[0].u32]);break;
    case OP_DUP:retained=stack_peek(vm,0);break;
    case OP_LOAD_LOCAL:case OP_STORE_LOCAL:
        if(d->operands[0].u16>=frame->local_count)return VM_ERR_OUT_OF_BOUNDS;
        if(fact->opcode==OP_LOAD_LOCAL)retained=vm->stack[frame->stack_base+d->operands[0].u16];
        break;
    case OP_LOAD_GLOBAL:case OP_STORE_GLOBAL:
        if(d->operands[0].u32>=vm->global_capacity)return VM_ERR_OUT_OF_BOUNDS;
        if(fact->opcode==OP_LOAD_GLOBAL)retained=vm->globals[d->operands[0].u32];
        break;
    case OP_STRUCT_NEW:
        /* Shared NEW constructs zero fields. record_site rejects this against
         * the sole one-field union record, so it has no prepared live site. */
        return VM_ERR_TYPE_ERROR;
    case OP_STRUCT_LITERAL:case OP_AGG_PACK:
        if(site->state!=UFS_PROVED || site->nominal!=0 || site->member!=fact->pops)return VM_ERR_TYPE_ERROR;
        if(site->operation==UFS_VARIANT_PACK) {
            if(site->layout!=p->union_layout || site->variant>=2 || site->member!=p->variants[site->variant].field_count ||
               (site->member && stack_peek(vm,0).tag!=TAG_INT))return VM_ERR_TYPE_ERROR;
        } else if(site->operation!=UFS_RECORD_PACK ||
            site->layout!=p->record_layout || site->member!=1 || stack_peek(vm,0).tag!=TAG_UNION)return VM_ERR_TYPE_ERROR;
        break;
    case OP_STRUCT_GET:case OP_AGG_GET:case OP_STRUCT_SET:case OP_AGG_SET: {
        bool set=fact->opcode==OP_STRUCT_SET || fact->opcode==OP_AGG_SET;
        NanoValue receiver=stack_peek(vm,set?1:0);
        if(site->state!=UFS_PROVED || site->nominal!=0 || site->member!=0)return VM_ERR_TYPE_ERROR;
        if(receiver.tag==TAG_STRUCT && site->layout==p->record_layout) {
            if(set) {
                if(site->operation!=UFS_RECORD_SET || stack_peek(vm,0).tag!=TAG_UNION)return VM_ERR_TYPE_ERROR;
            } else {
                if(site->operation!=UFS_RECORD_GET)return VM_ERR_TYPE_ERROR;
                retained=receiver.as.sval->fields[0];
                if(retained.tag!=TAG_UNION)return VM_ERR_TYPE_ERROR;
            }
        } else if(!set && fact->opcode==OP_AGG_GET && receiver.tag==TAG_UNION && site->operation==UFS_PAYLOAD_GET &&
            site->layout==p->union_layout && site->variant==receiver.as.uval->variant && receiver.as.uval->field_count==1) {
            if(receiver.as.uval->fields[0].tag!=TAG_INT)return VM_ERR_TYPE_ERROR;
        } else return VM_ERR_TYPE_ERROR;
        break;
    }
    case OP_AGG_TAG:
        if(site->state!=UFS_PROVED || site->operation!=UFS_TAG || site->nominal!=0 ||
           site->layout!=p->union_layout || stack_peek(vm,0).tag!=TAG_UNION)return VM_ERR_TYPE_ERROR;
        break;
    case OP_CALL:case OP_TAIL_CALL: {
        if(fact->callee>=p->counts.functions)return VM_ERR_UNDEFINED_FUNCTION;
        const NvmRecordArrayExecutionFunction *target_fn=&p->functions[fact->callee];
        const NvmFunctionEntry *callee=&target_fn->signature;
        if(!vm_rm_signature_same(callee,&vm->module->functions[fact->callee]) || callee->arity!=fact->pops || callee->upvalue_count)return VM_ERR_DECODE;
        bool tail=fact->opcode==OP_TAIL_CALL;
        if(tail && (operands!=callee->arity || callee->result_count!=fn->signature.result_count || callee->result_tag!=fn->signature.result_tag))return VM_ERR_TYPE_ERROR;
        if(!tail && vm->frame_count==VM_MAX_FRAMES)return VM_ERR_CALL_DEPTH;
        uint32_t base=tail?frame->stack_base:vm->stack_size-callee->arity;
        uint32_t available=vm->stack_capacity-base;
        if(callee->local_count>available || target_fn->maximum_stack>available-callee->local_count)return VM_ERR_STACK_OVERFLOW;
        for(uint16_t i=0;i<callee->arity;i++) {
            NanoValue value=stack_peek(vm,callee->arity-1-i);
            if(vm->module->function_param_types && vm->module->function_param_types[fact->callee]) {
                uint8_t tag=vm->module->function_param_types[fact->callee][i];
                if(tag!=TAG_VOID && tag!=value.tag)return VM_ERR_TYPE_ERROR;
            }
            if(tail && val_is_heap_obj(value)) {
                uint32_t copies=0;
                for(uint16_t j=0;j<callee->arity;j++)if(val_is_heap_obj(stack_peek(vm,j)) && stack_peek(vm,j).as.obj==value.as.obj)copies++;
                if(((VmHeapHeader *)value.as.obj)->ref_count>UINT32_MAX-copies)return VM_ERR_MEMORY;
            }
        }
        break;
    }
    case OP_RET:
        if(operands!=fn->signature.result_count || (operands && stack_peek(vm,0).tag!=fn->signature.result_tag))return VM_ERR_TYPE_ERROR;
        break;
    case OP_HALT:
        if(site->state!=UFS_PROVED || site->operation!=UFS_HALT)return VM_ERR_TYPE_ERROR;
        break;
    case OP_ASSERT:case OP_JMP_TRUE:case OP_JMP_FALSE:
        if(stack_peek(vm,0).tag!=TAG_BOOL)return VM_ERR_TYPE_ERROR;
        break;
    case OP_ARR_NEW:case OP_ARR_LITERAL:case OP_ARR_GET:case OP_ARR_SET:case OP_ARR_PUSH:case OP_ARR_POP:case OP_ARR_LEN:case OP_ARR_SLICE:case OP_STR_SPLIT:
        return VM_ERR_TYPE_ERROR; /* No array origins in the frozen union proof. */
    default:break;
    }
    if(!vm_uf_shell(p,retained))return VM_ERR_TYPE_ERROR;
    return vm_rmx_retain_check(retained);
}
