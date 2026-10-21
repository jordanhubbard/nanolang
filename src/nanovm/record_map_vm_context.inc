/* Private map context only; old array owner tables are never aliased. */
#include "record_map_heap_private.h"
static bool vm_rmx_array_valid_depth(VmArray *a,uint8_t expected_leaf,uint8_t depth) {
    if(!depth || !a || a->header.obj_type!=TAG_ARRAY || !a->header.ref_count ||
       a->length>a->capacity || a->unboxed!=vm_array_type_unboxable(a->elem_type) ||
       (a->capacity && !(a->unboxed?a->packed!=NULL:a->elements!=NULL)))return false;
    if(expected_leaf && a->elem_type!=expected_leaf)return false;
    if(a->elem_type==TAG_STRING) {
        for(uint32_t i=0;i<a->length;i++) {
            NanoValue v=a->elements[i];
            if(v.tag!=TAG_STRING || !v.as.string || v.as.string->header.obj_type!=TAG_STRING ||
               !v.as.string->header.ref_count)return false;
        }
        return true;
    }
    if(vm_rm_leaf(a->elem_type))return true;
    if(depth==1 || expected_leaf || a->elem_type!=TAG_ARRAY)return false;
    for(uint32_t i=0;i<a->length;i++) {
        NanoValue child=a->elements[i];
        if(child.tag!=TAG_ARRAY || !child.as.array ||
           !vm_rmx_array_valid_depth(child.as.array,TAG_VOID,(uint8_t)(depth-1)))return false;
    }
    return true;
}
static bool vm_rmx_array_valid(VmArray *a,uint8_t expected_leaf) {
    return vm_rmx_array_valid_depth(a,expected_leaf,3);
}
static bool vm_rmx_array_exact(VmArray *a,uint8_t depth,uint8_t leaf) {
    if(!depth || depth>3 || !vm_rmx_array_valid_depth(a,depth==1?leaf:TAG_VOID,depth))return false;
    if(depth==1)return a->elem_type==leaf;
    if(a->elem_type!=TAG_ARRAY)return false;
    for(uint32_t i=0;i<a->length;i++)
        if(!vm_rmx_array_exact(a->elements[i].as.array,(uint8_t)(depth-1),leaf))return false;
    return true;
}
static bool vm_rmx_value_valid(const VmRecordMapPrivate *p,NanoValue value) {
    if(value.tag==TAG_VOID || vm_rm_leaf(value.tag)) {
        if(value.tag!=TAG_STRING)return true;
        return value.as.string && value.as.string->header.obj_type==TAG_STRING && value.as.string->header.ref_count &&
            !value.as.string->header.buffered;
    }
    if(value.tag==TAG_HASHMAP) {
        uint32_t count;return vm_rm_heap_len(value.as.hashmap,&count)==VM_RM_HEAP_OK;
    }
    if(val_is_heap_obj(value) && value.as.obj && ((VmHeapHeader *)value.as.obj)->buffered)return false;
    if(value.tag==TAG_ARRAY)return vm_rmx_array_valid(value.as.array,TAG_VOID);
    if(value.tag==TAG_STRUCT) {
        VmStruct *s=value.as.sval;
        return s && s->header.obj_type==TAG_STRUCT && s->header.ref_count &&
            s->def_idx<p->layout_count && p->layouts[s->def_idx].declaration.kind==NVM_V2_LAYOUT_STRUCT &&
            s->field_count==p->layouts[s->def_idx].declaration.fields && !s->field_names &&
            (!s->field_count || s->fields);
    }
    return false;
}
static bool vm_rmx_field_accepts(const VmRecordMapPrivate *p,uint32_t layout,uint16_t field,NanoValue value) {
    if(layout>=p->layout_count || field>=p->layouts[layout].declaration.fields || !vm_rmx_value_valid(p,value))return false;
    const VmRecordMapRuntimeField *expected=&p->fields[p->layouts[layout].field_start+field];
    if(value.tag!=expected->declaration.type_tag)return false;
    if(value.tag==TAG_STRUCT)return value.as.sval->def_idx==expected->declaration.nested_idx;
    if(value.tag==TAG_HASHMAP)return value.as.hashmap->key_type==expected->key && value.as.hashmap->val_type==expected->value;
    if(value.tag==TAG_ARRAY) {
        if(value.as.array->elem_type!=expected->element)return false;
        return vm_rmx_array_exact(value.as.array,expected->array_depth,
                                 expected->nested_leaf?expected->nested_leaf:expected->element);
    }
    return true;
}
/* These actual scalar handlers reject after popping and do not release a
 * wrong-tag heap operand. I reject while every operand remains a stack root.
 * I do not replace polymorphic truthiness/comparison or string fallback rules. */
static bool vm_rmx_context(const VmRecordMapInvocation *v,const VmState *vm) {
    if(!v || !v->active || !v->owner)return false;
    const VmRecordMapPrivate *p=v->owner;
    if(!p->prepared || !p->busy || v->epoch!=p->epoch || vm!=&p->vm || v->vm!=vm ||
       !pthread_equal(vm->owner_thread,pthread_self()) ||
       v->module!=nvm_record_map_image_borrow(p->image) || vm->module!=v->module || vm->root_module!=v->module ||
       vm->verified || !vm->heap.private_record_dag || vm->heap.gc_running || vm->callbacks ||
       vm->linked_module_count || vm->opcode_trace || vm->handler_count || vm->activation_floor ||
       vm->references.active || vm->callee_references.active || vm->reference_generation ||
       !vm->decoded_module_valid || !vm->dispatch_module_valid || vm->dispatch_profile.fuse_load_local_field ||
       !vm->frame_count || vm->frame_count>VM_MAX_FRAMES || vm->current_fn>=p->counts.functions ||
       vm->stack_capacity!=VM_RM_ROOT_CAPACITY || vm->stack_size>vm->stack_capacity ||
       v->active_root>=p->counts.functions || vm->frames[0].fn_idx!=v->active_root ||
       vm->frames[0].stack_base!=v->root_base)return false;
    for(uint32_t i=0;i<vm->frame_count;i++) {
        const VmCallFrame *f=&vm->frames[i];
        if(f->module!=v->module || f->fn_idx>=p->counts.functions || f->effect_owner || f->closure ||
           f->owned_callable.tag!=TAG_VOID || f->local_count!=p->functions[f->fn_idx].signature.local_count ||
           f->stack_base>vm->stack_size || f->local_count>vm->stack_size-f->stack_base)return false;
        if(!vm_rm_signature_same(&p->functions[f->fn_idx].signature,&vm->module->functions[f->fn_idx]))return false;
    }
    return true;
}
/* Runtime root/nominal preflight occurs before a handler pops or overwrites.
 * The real VM still performs its original scalar/error operations. */
static VmResult vm_rmx_preflight(VmRecordMapInvocation *v,const VmDispatchInstruction *instruction) {
    if(!v || !v->owner)return VM_ERR_TYPE_ERROR;
    VmRecordMapPrivate *p=v->owner;VmState *vm=&p->vm;
    if(!vm_rmx_context(v,vm))return VM_ERR_TYPE_ERROR;
    const VmCallFrame *frame=&vm->frames[vm->frame_count-1];
    const NvmRecordArrayExecutionFunction *fn=&p->functions[vm->current_fn];
    if(frame->module!=vm->module || frame->fn_idx!=vm->current_fn ||
       frame->local_count!=fn->signature.local_count || frame->effect_owner || frame->closure ||
       frame->owned_callable.tag!=TAG_VOID || frame->stack_base>vm->stack_size ||
       frame->local_count>vm->stack_size-frame->stack_base)return VM_ERR_TYPE_ERROR;
    if(instruction->byte_offset>=fn->signature.code_length)return VM_ERR_DECODE;
    uint32_t encoded=vm->decoded_module.functions[vm->current_fn].instruction_indices[instruction->byte_offset];
    if(!encoded || encoded>fn->instruction_count)return VM_ERR_DECODE;
    if(fn->instruction_start>p->counts.instructions || encoded>p->counts.instructions-fn->instruction_start)return VM_ERR_DECODE;
    const NvmRecordMapExecutionInstruction *map=&p->instructions[fn->instruction_start+encoded-1];
    const NvmRecordArrayExecutionInstruction *fact=&map->base;
    NvmRecordArrayExecutionInstruction expected={0};
    if(!vm_rm_recipe(fact->opcode,&expected) || fact->recipe!=expected.recipe || fact->obligations!=expected.obligations ||
       fact->function!=vm->current_fn || fact->pc!=instruction->byte_offset || fact->next_pc!=instruction->next_byte_offset ||
       instruction->call_target!=fact->callee || instruction->call_handle.resolved || instruction->call_handle.module ||
       instruction->call_handle.function || instruction->call_handle.function_index || instruction->super_operand)return VM_ERR_DECODE;
    uint8_t operation=NVM_RM_NONE;uint16_t result=0;
    switch(fact->opcode) {
    case OP_HM_NEW:operation=NVM_RM_NEW;result=1u<<TAG_HASHMAP;break;
    case OP_HM_SET:operation=NVM_RM_SET;result=1u<<TAG_HASHMAP;break;
    case OP_HM_DELETE:operation=NVM_RM_DELETE;result=1u<<TAG_HASHMAP;break;
    case OP_HM_GET:operation=NVM_RM_GET;result=(1u<<TAG_INT)|(1u<<TAG_VOID);break;
    case OP_HM_HAS:operation=NVM_RM_HAS;result=1u<<TAG_BOOL;break;
    case OP_HM_LEN:operation=NVM_RM_LEN;result=1u<<TAG_INT;break;
    default:break;
    }
    if(map->map_operation!=operation || map->result_tags!=result || map->key_tags!=(operation?(1u<<TAG_STRING):0) ||
       map->value_tags!=(operation?(1u<<TAG_INT):0))return VM_ERR_DECODE;
    if(instruction->next_index!=(encoded<fn->instruction_count?encoded:VM_DISPATCH_NO_INDEX))return VM_ERR_DECODE;
    uint32_t target=UINT32_MAX,branch=UINT32_MAX;
    if(fact->opcode==OP_JMP || fact->opcode==OP_JMP_TRUE || fact->opcode==OP_JMP_FALSE) {
        if(!fact->successor_count || fact->successors[0]>fn->signature.code_length)return VM_ERR_DECODE;
        uint32_t pc=fact->successors[0];target=fn->signature.code_offset+pc;
        if(pc<fn->signature.code_length) {
            uint32_t index=vm->decoded_module.functions[vm->current_fn].instruction_indices[pc];
            if(!index || index>fn->instruction_count)return VM_ERR_DECODE;
            branch=index-1;
        }
    }
    if(instruction->branch_target_offset!=target || instruction->branch_target!=branch)return VM_ERR_DECODE;
    const InstructionInfo *info=isa_get_info(fact->opcode);
    if(!info)return VM_ERR_DECODE;
    int pops=info->pop_count,pushes=info->push_count;
    switch(fact->opcode) {
    case OP_ARR_LITERAL:case OP_STRUCT_LITERAL:pops=instruction->instruction.operands[1].u16;pushes=1;break;
    case OP_AGG_PACK:pops=instruction->instruction.operands[3].u16;pushes=1;break;
    case OP_CALL:case OP_TAIL_CALL:
        if(fact->callee>=p->counts.functions || !vm_rm_signature_same(&p->functions[fact->callee].signature,
            &vm->module->functions[fact->callee]))return VM_ERR_DECODE;
        pops=p->functions[fact->callee].signature.arity;
        pushes=fact->opcode==OP_TAIL_CALL?0:p->functions[fact->callee].signature.result_count;break;
    case OP_RET:pops=fn->signature.result_count;pushes=0;break;
    default:break;
    }
    if(pops<0 || pushes<0 || fact->pops!=pops || fact->pushes!=pushes)return VM_ERR_DECODE;
    if(!vm_rm_instruction_same(fact,&instruction->instruction) || instruction->super_op!=VM_SUPER_NONE)return VM_ERR_DECODE;
    uint32_t operands=vm->stack_size-frame->stack_base-frame->local_count;
    if(operands>256)return VM_ERR_STACK_OVERFLOW;
    if(fact->pops>operands)return VM_ERR_STACK_UNDERFLOW;
    if(fact->pushes>vm->stack_capacity-(vm->stack_size-fact->pops))return VM_ERR_STACK_OVERFLOW;
    for(uint16_t i=0;i<fact->pops;i++)if(!vm_rmx_value_valid(p,stack_peek(vm,i)))return VM_ERR_TYPE_ERROR;
    const DecodedInstruction *in=&instruction->instruction;
    VmResult scalar_inputs=vm_rmx_scalar_inputs(vm,in->opcode);
    if(scalar_inputs!=VM_OK)return scalar_inputs;
    NanoValue retained=val_void();
    switch(in->opcode) {
    case OP_HM_NEW:
        if(in->operands[0].u8!=TAG_STRING || in->operands[1].u8!=TAG_INT)return VM_ERR_TYPE_ERROR;
        break;
    case OP_HM_SET:case OP_HM_GET:case OP_HM_HAS:case OP_HM_LEN:case OP_HM_DELETE: {
        uint16_t receiver=in->opcode==OP_HM_SET?2:in->opcode==OP_HM_LEN?0:1;
        if(stack_peek(vm,receiver).tag!=TAG_HASHMAP ||
           (in->opcode!=OP_HM_LEN && stack_peek(vm,receiver-1).tag!=TAG_STRING) ||
           (in->opcode==OP_HM_SET && stack_peek(vm,0).tag!=TAG_INT))return VM_ERR_TYPE_ERROR;
        break;
    }
    case OP_RET:
        if(operands!=fn->signature.result_count)return VM_ERR_TYPE_ERROR;
        if(fn->signature.result_count && !result_tag_matches(fn->signature.result_tag,stack_peek(vm,0).tag))return VM_ERR_TYPE_ERROR;
        break;
    case OP_PUSH_STR:
        if(in->operands[0].u32>=vm->module_constants.count)return VM_ERR_OUT_OF_BOUNDS;
        retained=val_string(vm->module_constants.strings[in->operands[0].u32]);break;
    case OP_DUP:retained=stack_peek(vm,0);break;
    case OP_LOAD_LOCAL:case OP_STORE_LOCAL:
        if(in->operands[0].u16>=frame->local_count)return VM_ERR_OUT_OF_BOUNDS;
        if(in->opcode==OP_LOAD_LOCAL)retained=vm->stack[frame->stack_base+in->operands[0].u16];
        break;
    case OP_LOAD_GLOBAL:case OP_STORE_GLOBAL:
        if(in->operands[0].u32>=vm->global_count)return VM_ERR_OUT_OF_BOUNDS;
        if(in->opcode==OP_LOAD_GLOBAL)retained=vm->globals[in->operands[0].u32];
        break;
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        if(!vm_rm_leaf(in->operands[0].u8) && in->operands[0].u8!=TAG_ARRAY)return VM_ERR_TYPE_ERROR;
        if(in->opcode==OP_ARR_LITERAL)
            for(uint16_t i=0;i<fact->pops;i++) {
                NanoValue child=stack_peek(vm,i);
                if(child.tag!=in->operands[0].u8 ||
                   (child.tag==TAG_ARRAY && (!child.as.array || !vm_rmx_array_valid_depth(child.as.array,TAG_VOID,2))))return VM_ERR_TYPE_ERROR;
            }
        break;
    case OP_ARR_PUSH:case OP_ARR_SET: {
        NanoValue receiver=stack_peek(vm,in->opcode==OP_ARR_PUSH?1:2),value=stack_peek(vm,0);
        if(receiver.tag!=TAG_ARRAY || value.tag!=receiver.as.array->elem_type ||
           (value.tag==TAG_ARRAY && (!value.as.array || !vm_rmx_array_valid_depth(value.as.array,TAG_VOID,2))))return VM_ERR_TYPE_ERROR;
        if(in->opcode==OP_ARR_PUSH)retained=value;
        break;
    }
    case OP_ARR_GET: {
        NanoValue receiver=stack_peek(vm,1),index=stack_peek(vm,0);
        if(receiver.tag!=TAG_ARRAY)return VM_ERR_TYPE_ERROR;
        if(index.tag==TAG_INT && index.as.i64>=0 && (uint64_t)index.as.i64<receiver.as.array->length)
            retained=vm_array_get(receiver.as.array,(uint32_t)index.as.i64);
        break;
    }
    case OP_ARR_POP:case OP_ARR_LEN:
        if(stack_peek(vm,0).tag!=TAG_ARRAY)return VM_ERR_TYPE_ERROR;
        break;
    case OP_ARR_SLICE:
        if(stack_peek(vm,2).tag!=TAG_ARRAY)return VM_ERR_TYPE_ERROR;
        break;
    case OP_STRUCT_NEW:case OP_STRUCT_LITERAL:case OP_AGG_PACK: {
        uint32_t ordinal=in->opcode==OP_AGG_PACK?in->operands[1].u32:in->operands[0].u32;
        if(ordinal>=p->counts.records || (in->opcode==OP_AGG_PACK &&
           (in->operands[0].u8!=AGG_RECORD || in->operands[2].u16)))return VM_ERR_TYPE_ERROR;
        uint32_t layout=p->record_layouts[ordinal];
        if(fact->pops!=p->layouts[layout].declaration.fields)return VM_ERR_TYPE_ERROR;
        for(uint16_t field=0;field<fact->pops;field++)
            if(!vm_rmx_field_accepts(p,layout,field,stack_peek(vm,fact->pops-1-field)))return VM_ERR_TYPE_ERROR;
        break;
    }
    case OP_STRUCT_GET:case OP_AGG_GET:case OP_STRUCT_SET:case OP_AGG_SET: {
        bool store=in->opcode==OP_STRUCT_SET || in->opcode==OP_AGG_SET;
        NanoValue receiver=stack_peek(vm,store?1:0);
        uint16_t field=in->operands[0].u16;
        if(receiver.tag!=TAG_STRUCT)return (in->opcode==OP_AGG_GET || in->opcode==OP_AGG_SET)?
            VM_ERR_OUT_OF_BOUNDS:VM_ERR_TYPE_ERROR;
        if(field>=receiver.as.sval->field_count)return VM_ERR_OUT_OF_BOUNDS;
        if(store) {
            if(!vm_rmx_field_accepts(p,receiver.as.sval->def_idx,field,stack_peek(vm,0)))return VM_ERR_TYPE_ERROR;
        } else retained=receiver.as.sval->fields[field];
        break;
    }
    case OP_TAIL_CALL: {
        if(fact->callee>=p->counts.functions)return VM_ERR_UNDEFINED_FUNCTION;
        const NvmRecordArrayExecutionFunction *target=&p->functions[fact->callee];
        const NvmFunctionEntry *callee=&target->signature;
        if(operands!=callee->arity || callee->result_count!=fn->signature.result_count ||
           callee->result_tag!=fn->signature.result_tag)return VM_ERR_TYPE_ERROR;
        uint32_t available=vm->stack_capacity-frame->stack_base;
        if(callee->local_count>available || target->maximum_stack>available-callee->local_count)
            return VM_ERR_STACK_OVERFLOW;
        for(uint16_t i=0;i<callee->arity;i++) {
            NanoValue value=stack_peek(vm,callee->arity-1-i);
            if(vm->module->function_param_types && vm->module->function_param_types[fact->callee]) {
                uint8_t tag=vm->module->function_param_types[fact->callee][i];
                if(tag!=TAG_VOID && value.tag!=tag)return VM_ERR_TYPE_ERROR;
            }
            /* The real handler retains all staged arguments before releasing
             * the old frame. Count repeated owners together, before mutation. */
            if(val_is_heap_obj(value)) {
                uint32_t copies=0;
                for(uint16_t j=0;j<callee->arity;j++) {
                    NanoValue other=stack_peek(vm,j);
                    if(val_is_heap_obj(other) && other.as.obj==value.as.obj)copies++;
                }
                if(!value.as.obj || ((VmHeapHeader *)value.as.obj)->ref_count>UINT32_MAX-copies)
                    return VM_ERR_MEMORY;
            }
        }
        break;
    }
    case OP_CALL: {
        if(fact->callee>=p->counts.functions)return VM_ERR_UNDEFINED_FUNCTION;
        const NvmFunctionEntry *callee=&p->functions[fact->callee].signature;
        if(vm->frame_count==VM_MAX_FRAMES)return VM_ERR_CALL_DEPTH;
        uint32_t available=vm->stack_capacity-(vm->stack_size-callee->arity);
        if(callee->local_count>available ||
           p->functions[fact->callee].maximum_stack>available-callee->local_count)return VM_ERR_STACK_OVERFLOW;
        if(vm->module->function_param_types && vm->module->function_param_types[fact->callee])
            for(uint16_t i=0;i<callee->arity;i++) {
                uint8_t tag=vm->module->function_param_types[fact->callee][i];
                if(tag!=TAG_VOID && stack_peek(vm,callee->arity-1-i).tag!=tag)return VM_ERR_TYPE_ERROR;
            }
        break;
    }
    default:break;
    }
    if(!vm_rmx_value_valid(p,retained))return VM_ERR_TYPE_ERROR;
    if(vm->frame_count>p->maximum_frames)p->maximum_frames=vm->frame_count;
    VmResult old_array=vm_record_array_retain_preflight(vm,(VmCallFrame *)frame,instruction);
    if(old_array!=VM_OK)return old_array;
    return vm_rmx_retain_check(retained);
}

static uint32_t vm_rmx_record_layout(const VmRecordMapInvocation *v,uint32_t ordinal) {
    /* The complete opcode preflight checked this ordinal before any pop. */
    assert(v && v->owner && ordinal<v->owner->counts.records);
    return v->owner->record_layouts[ordinal];
}
static VmResult vm_rmx_map_instruction(VmRecordMapInvocation *v,uint8_t opcode) {
    VmState *vm=v->vm;
    if(opcode==OP_HM_NEW) {
        VmHashMap *m=NULL;VmRmHeapStatus status=vm_rm_heap_new(&vm->heap,&m);
        if(status!=VM_RM_HEAP_OK)return vm_rmx_heap_status(status);
        stack_push(vm,val_hashmap(m));return VM_OK;
    }
    NanoValue value=val_void(),key=val_void();
    if(opcode==OP_HM_SET)value=stack_pop(vm);
    if(opcode!=OP_HM_LEN)key=stack_pop(vm);
    NanoValue map=stack_pop(vm),output=val_void();VmRmHeapStatus status;
    switch(opcode) {
    case OP_HM_SET:
        status=vm_rm_heap_set(&vm->heap,map.as.hashmap,key,value);
        if(status==VM_RM_HEAP_OK){output=map;map=val_void();}
        break;
    case OP_HM_DELETE:
        status=vm_rm_heap_delete(&vm->heap,map.as.hashmap,key);
        if(status==VM_RM_HEAP_OK){output=map;map=val_void();}
        break;
    case OP_HM_GET:status=vm_rm_heap_get(map.as.hashmap,key,&output);break;
    case OP_HM_HAS: {
        bool found=false;status=vm_rm_heap_has(map.as.hashmap,key,&found);
        if(status==VM_RM_HEAP_OK)output=val_bool(found);
        break;
    }
    case OP_HM_LEN: {
        uint32_t count=0;status=vm_rm_heap_len(map.as.hashmap,&count);
        if(status==VM_RM_HEAP_OK)output=val_int(count);
        break;
    }
    default:status=VM_RM_HEAP_INVALID;break;
    }
    vm_release(&vm->heap,value);vm_release(&vm->heap,key);vm_release(&vm->heap,map);
    if(status!=VM_RM_HEAP_OK)return vm_rmx_heap_status(status);
    stack_push(vm,output);return VM_OK;
}
