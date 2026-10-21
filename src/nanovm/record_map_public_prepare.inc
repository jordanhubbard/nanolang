/* Preparation only: no stack allocation, activation, dispatch, or selector. */
#include "record_map_public_prepare_private.h"
#include "../nanoisa/record_map_boundary_private.h"
_Static_assert(sizeof(VmRecordMapPublicPreparation)==14512,"public preparation owner ledger");
_Static_assert(sizeof(NvmRecordMapExecutionInstruction)==80,"instruction row ledger");
_Static_assert(sizeof(VmRecordMapRuntimeField)==20,"field row ledger");
#define VM_RMP_SCRATCH UINT64_C(1812)
#define VM_RMP_VIEW UINT64_C(40)
static bool vm_rmp_charge(VmRecordMapPublicPreparation *p,NvmArrayEligibilityStatus *status,uint64_t b,uint64_t w) {
    if(p->bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || b>NVM_RECORD_ARRAY_EXECUTION_BYTES-p->bytes ||
       p->work>NVM_RECORD_ARRAY_EXECUTION_STEPS || w>NVM_RECORD_ARRAY_EXECUTION_STEPS-p->work) {
        *status=NVM_ARRAY_LIMIT;return false;
    }
    p->bytes+=b;p->work+=w;return true;
}
static void *vm_rmp_zero(VmRecordMapPublicPreparation *p,NvmArrayEligibilityStatus *status,uint64_t count,size_t width) {
    if(!count)return NULL;
    if(!width || count>SIZE_MAX/width || count>UINT64_MAX/width){*status=NVM_ARRAY_LIMIT;return NULL;}
    if(!vm_rmp_charge(p,status,count*width,count*width))return NULL;
    void *out=calloc((size_t)count,width);
    if(!out)*status=NVM_ARRAY_MEMORY;
    return out;
}
void vm_record_map_public_preparation_free(VmRecordMapPublicPreparation *p) {
    if(!p)return;
    free(p->instructions);free(p->fields);nvm_record_map_boundary_free(p->boundary);free(p);
}
static bool vm_rmp_layouts(VmRecordMapPublicPreparation *p,NvmArrayEligibilityStatus *status) {
    NvmDeclarationCounts count;
    if(!nvm_record_map_execution_declaration_counts(p->plan,&count) || count.layouts>256 ||
       p->counts.carrier.records>256)return false;
    p->layout_count=count.layouts;
    if(!vm_rmp_charge(p,status,0,(uint64_t)count.layouts*24+(uint64_t)count.bindings*64u))return false;
    for(uint32_t i=0;i<count.layouts;i++) {
        VmRecordMapRuntimeLayout *layout=&p->layouts[i];
        if(!nvm_record_map_execution_layout(p->plan,i,&layout->declaration))return false;
        layout->field_start=p->field_count;
        if(layout->declaration.fields>65536u-p->field_count)return false;
        p->field_count+=layout->declaration.fields;
    }
    p->fields=vm_rmp_zero(p,status,p->field_count,sizeof *p->fields);
    if(p->field_count && !p->fields)return false;
    if(!vm_rmp_charge(p,status,0,(uint64_t)p->field_count*40))return false;
    for(uint32_t i=0;i<count.layouts;i++) {
        VmRecordMapRuntimeLayout *layout=&p->layouts[i];
        if(layout->declaration.kind!=NVM_V2_LAYOUT_STRUCT &&
           layout->declaration.kind!=NVM_V2_LAYOUT_UNION)return false;
        if(layout->declaration.flags!=(layout->declaration.kind==NVM_V2_LAYOUT_STRUCT?NVM_LAYOUT_COMPLETE:0))return false;
        for(uint16_t j=0;j<layout->declaration.fields;j++) {
            VmRecordMapRuntimeField *field=&p->fields[layout->field_start+j];
            if(!nvm_record_map_execution_field(p->plan,i,j,&field->declaration))return false;
            if(layout->declaration.kind!=NVM_V2_LAYOUT_STRUCT)continue;
            uint8_t tag=field->declaration.type_tag;
            if(tag==TAG_STRUCT) {
                uint32_t target=field->declaration.nested_idx;
                if(target>=count.layouts || p->layouts[target].declaration.kind!=NVM_V2_LAYOUT_STRUCT)return false;
            } else if((!vm_rm_leaf(tag) && tag!=TAG_ARRAY && tag!=TAG_HASHMAP) || field->declaration.nested_idx!=NVM_V2_NO_INDEX)return false;
        }
    }
    for(uint32_t i=0;i<count.bindings;i++) {
        NvmOrdinaryArrayBinding binding;NvmOrdinaryArrayType element;
        if(!nvm_record_map_execution_binding(p->plan,i,&binding) || binding.layout>=count.layouts ||
           binding.field>=p->layouts[binding.layout].declaration.fields ||
           !nvm_record_map_execution_type(p->plan,binding.element_type,&element))return false;
        uint8_t nested_leaf=TAG_VOID,depth=1;
        NvmOrdinaryArrayType leaf=element;
        while(leaf.tag==TAG_ARRAY) {
            if(++depth>3 || !nvm_record_map_execution_type(p->plan,leaf.referent,&leaf))return false;
            /* The bounded binding schedule includes each of the three levels. */
        }
        if(!vm_rm_leaf(leaf.tag) || leaf.referent!=NVM_V2_NO_INDEX)return false;
        if(depth>1)nested_leaf=leaf.tag;
        VmRecordMapRuntimeField *field=&p->fields[p->layouts[binding.layout].field_start+binding.field];
        if(field->declaration.type_tag!=TAG_ARRAY || field->element)return false;
        field->element=element.tag;field->nested_leaf=nested_leaf;field->array_depth=depth;
    }
    uint32_t records=0;
    for(uint32_t i=0;i<count.layouts;i++) {
        VmRecordMapRuntimeLayout *layout=&p->layouts[i];
        if(layout->declaration.kind!=NVM_V2_LAYOUT_STRUCT)continue;
        if(records>=p->counts.carrier.records)return false;
        NvmRecordArrayExecutionDescriptor descriptor;
        if(!nvm_record_map_execution_descriptor(p->plan,records,&descriptor) ||
           descriptor.ordinal!=records || descriptor.layout!=i || descriptor.fields!=layout->declaration.fields)return false;
        p->record_layouts[records++]=i;
        for(uint16_t j=0;j<layout->declaration.fields;j++) {
            VmRecordMapRuntimeField *field=&p->fields[layout->field_start+j];
            if(field->declaration.type_tag==TAG_ARRAY && !field->element)return false;
            if(field->declaration.type_tag==TAG_HASHMAP) {
                if(!nvm_record_map_execution_map_field(p->plan,i,j,&field->key,&field->value) ||
                   field->key!=TAG_STRING || field->value!=TAG_INT)return false;
            }
        }
    }
    if(records!=p->counts.carrier.records)return false;
    /* Explicit DFS: each nominal field edge is examined once. Rank decreases
     * along every physical record edge, so <=256 record nodes can be nested. */
    /* DFS scratch was prepaid before boundary proof. */
    uint8_t color[256]={0};
    struct {uint16_t layout,field;} stack[256];
    for(uint32_t root=0;root<count.layouts;root++) {
        if(color[root] || p->layouts[root].declaration.kind!=NVM_V2_LAYOUT_STRUCT)continue;
        uint32_t depth=1;stack[0].layout=(uint16_t)root;stack[0].field=0;color[root]=1;
        while(depth) {
            uint16_t index=stack[depth-1].layout;
            VmRecordMapRuntimeLayout *layout=&p->layouts[index];
            if(stack[depth-1].field==layout->declaration.fields) {
                color[index]=2;layout->rank++;depth--;continue;
            }
            VmRecordMapRuntimeField *field=&p->fields[layout->field_start+stack[depth-1].field];
            if(field->declaration.type_tag!=TAG_STRUCT){stack[depth-1].field++;continue;}
            uint32_t target=field->declaration.nested_idx;
            if(color[target]==1)return false;
            if(!color[target]) {
                if(depth==256)return false;
                color[target]=1;stack[depth].layout=(uint16_t)target;stack[depth++].field=0;
            } else {
                if(p->layouts[target].rank>layout->rank)layout->rank=p->layouts[target].rank;
                stack[depth-1].field++;
            }
        }
    }
    return true;
}

static bool vm_rmp_coverage(VmRecordMapPublicPreparation *p,const VmState *vm,NvmArrayEligibilityStatus *status) {
    const NvmModule *m=p->module;
    const VmDecodedModule *dm=&vm->decoded_module;
    const VmDispatchModule *dispatch=&vm->dispatch_module;
    if(!m || !m->functions || !dm->functions || !dispatch->functions || m!=vm->module || m!=vm->root_module ||
       dm->function_count!=p->counts.carrier.functions || dispatch->function_count!=p->counts.carrier.functions)return false;
    if(!vm_rmp_charge(p,status,0,256+(uint64_t)p->counts.carrier.instructions*128+
        (uint64_t)p->counts.carrier.functions*32))return false;
    uint32_t supported=0;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction expected={0};
        bool active=vm_rm_recipe((uint8_t)op,&expected)!=0;
        if(active!=(nvm_record_array_opcode_supported((uint8_t)op) || nvm_record_map_opcode((uint8_t)op)))return false;
        supported+=active;
    }
    if(supported!=99)return false;
    uint32_t total=0;
    for(uint32_t fi=0;fi<p->counts.carrier.functions;fi++) {
        const NvmRecordArrayExecutionFunction *fn=&p->functions[fi];
        const VmDecodedFunction *decoded=&dm->functions[fi];
        const VmDispatchFunction *body=&dispatch->functions[fi];
        if(!vm_rm_signature_same(&fn->signature,&m->functions[fi]) || fn->instruction_start!=total || total>p->counts.carrier.instructions ||
           fn->instruction_count>p->counts.carrier.instructions-total ||
           decoded->instruction_count!=fn->instruction_count || body->instruction_count!=fn->instruction_count ||
           decoded->code_size!=fn->signature.code_length || body->code_size!=fn->signature.code_length ||
           (fn->instruction_count && (!decoded->instructions || !body->instructions)) ||
           !decoded->boundaries || !decoded->instruction_indices || !body->offset_to_index ||
           !decoded->boundaries[decoded->code_size])return false;
        if(!vm_rmp_charge(p,status,0,10*((uint64_t)decoded->code_size+1)))return false;
        uint8_t bytes[256];
        uint32_t offset=fn->signature.code_offset,length=fn->signature.code_length;
        if(offset>m->code_size || length>m->code_size-offset || (length && !m->code))return false;
        for(uint32_t at=0;at<length;) {
            uint32_t n=length-at>sizeof bytes?sizeof bytes:length-at;
            if(!nvm_record_map_execution_bytes(p->plan,NVM_RA_SNAPSHOT_CODE,0,offset+at,n,bytes) ||
               memcmp(bytes,m->code+offset+at,n))return false;
            at+=n;
        }
        uint32_t expected_index=0;
        for(uint32_t pc=0;pc<=decoded->code_size;pc++) {
            bool start=expected_index<fn->instruction_count &&
                p->instructions[total+expected_index].base.pc==pc;
            uint32_t encoded=start?expected_index+1:0;
            if(decoded->boundaries[pc]!=(uint8_t)(start || pc==decoded->code_size) ||
               decoded->instruction_indices[pc]!=encoded || body->offset_to_index[pc]!=encoded)return false;
            expected_index+=start;
        }
        for(uint32_t i=0;i<fn->instruction_count;i++) {
            const NvmRecordArrayExecutionInstruction *fact=&p->instructions[total+i].base;
            const VmDecodedInstruction *in=&decoded->instructions[i];
            const VmDispatchInstruction *row=&body->instructions[i];
            NvmRecordArrayExecutionInstruction expected={0};
            const InstructionInfo *info=isa_get_info(fact->opcode);
            if(!info || !vm_rm_recipe(fact->opcode,&expected) || fact->recipe!=expected.recipe ||
               fact->obligations!=expected.obligations || fact->function!=fi ||
               fact->pc!=in->byte_offset || fact->next_pc!=in->next_byte_offset ||
               fact->pc!=row->byte_offset || fact->next_pc!=row->next_byte_offset ||
               !vm_rm_instruction_same(fact,&in->instruction) || !vm_rm_instruction_same(fact,&row->instruction) ||
               in->call_handle.resolved || row->call_handle.resolved || in->call_handle.module || row->call_handle.module ||
               in->call_handle.function || row->call_handle.function || in->call_handle.function_index || row->call_handle.function_index ||
               row->super_op!=VM_SUPER_NONE || row->super_operand)return false;
            const NvmRecordMapExecutionInstruction *map=&p->instructions[total+i];
            uint8_t operation=NVM_RM_NONE;uint16_t result=0;
            switch(fact->opcode) {
            case OP_HM_NEW:operation=NVM_RM_NEW;result=1u<<TAG_HASHMAP;break;
            case OP_HM_SET:operation=NVM_RM_SET;result=1u<<TAG_HASHMAP;break;
            case OP_HM_GET:operation=NVM_RM_GET;result=(1u<<TAG_INT)|(1u<<TAG_VOID);break;
            case OP_HM_HAS:operation=NVM_RM_HAS;result=1u<<TAG_BOOL;break;
            case OP_HM_LEN:operation=NVM_RM_LEN;result=1u<<TAG_INT;break;
            default:break;
            }
            if(map->map_operation!=operation || map->result_tags!=result ||
               map->key_tags!=(operation?(1u<<TAG_STRING):0) ||
               map->value_tags!=(operation?(1u<<TAG_INT):0))return false;
            uint32_t next=i+1<fn->instruction_count?i+1:VM_DISPATCH_NO_INDEX;
            if(row->next_index!=next)return false;
            int pops=info->pop_count,pushes=info->push_count;
            uint32_t callee=UINT32_MAX,resolved=UINT32_MAX,branch=UINT32_MAX,target=UINT32_MAX;
            uint8_t edges=0;uint32_t successors[2]={0,0};
            switch(fact->opcode) {
            case OP_ARR_LITERAL:case OP_STRUCT_LITERAL:pops=in->instruction.operands[1].u16;pushes=1;break;
            case OP_AGG_PACK:pops=in->instruction.operands[3].u16;pushes=1;break;
            case OP_CALL: case OP_TAIL_CALL:
                callee=in->instruction.operands[0].u32;
                if(callee>=p->counts.carrier.functions)return false;
                pops=m->functions[callee].arity;pushes=fact->opcode==OP_TAIL_CALL?0:m->functions[callee].result_count;resolved=callee;break;
            case OP_RET:pops=fn->signature.result_count;pushes=0;break;
            case OP_JMP:case OP_JMP_TRUE:case OP_JMP_FALSE: {
                int64_t dest=(int64_t)fact->pc+in->instruction.operands[0].i32;
                if(dest<0 || (uint64_t)dest>decoded->code_size)return false;
                successors[edges++]=(uint32_t)dest;
                resolved=fn->signature.code_offset+(uint32_t)dest;target=resolved;
                branch=dest==decoded->code_size?UINT32_MAX:body->offset_to_index[dest]-1;
                break;
            }
            default:break;
            }
            if(fact->opcode!=OP_JMP && fact->opcode!=OP_RET && fact->opcode!=OP_TAIL_CALL)successors[edges++]=fact->next_pc;
            if(pops<0 || pushes<0 || fact->pops!=pops || fact->pushes!=pushes || fact->callee!=callee ||
               fact->successor_count!=edges || in->resolved_target!=resolved || row->call_target!=callee ||
               row->branch_target!=branch || row->branch_target_offset!=target)return false;
            for(uint8_t j=0;j<edges;j++)if(fact->successors[j]!=successors[j])return false;
        }
        total+=fn->instruction_count;
    }
    return total==p->counts.carrier.instructions;
}

static bool vm_rmp_constants(VmRecordMapPublicPreparation *p,const VmState *vm,NvmArrayEligibilityStatus *status) {
    uint32_t count=p->counts.carrier.strings;
    if(vm->module_constants.count!=count || (count && !vm->module_constants.strings) ||
       !vm_rmp_charge(p,status,0,(uint64_t)count*32u))return false;
    uint8_t bytes[256];
    for(uint32_t i=0;i<count;i++) {
        uint32_t length;
        const VmString *str=vm->module_constants.strings[i];
        if(!nvm_record_map_execution_size(p->plan,NVM_RA_SNAPSHOT_STRING,i,&length) ||
           !str || str->length!=length || str->header.obj_type!=TAG_STRING || !str->header.ref_count ||
           str->header.buffered || str->header.colour!=VM_GC_BLACK ||
           (uint64_t)length+sizeof *str+1>SIZE_MAX)return false;
        if(!vm_rmp_charge(p,status,0,(uint64_t)length*2u))return false;
        uint32_t hash=UINT32_C(2166136261);
        for(uint32_t at=0;at<length;) {
            uint32_t n=length-at>sizeof bytes?sizeof bytes:length-at;
            if(!nvm_record_map_execution_bytes(p->plan,NVM_RA_SNAPSHOT_STRING,i,at,n,bytes))return false;
            for(uint32_t j=0;j<n;j++) {
                uint8_t actual=(uint8_t)str->data[at+j];
                if(bytes[j]!=actual)return false;
                hash^=actual;hash*=UINT32_C(16777619);
            }
            at+=n;
        }
        if(str->data[length] || str->hash!=hash)return false;
    }
    return true;
}
/* Caller precharges32F before this bounded function-row traversal. */
static bool vm_rmp_functions(VmRecordMapPublicPreparation *p,const VmState *vm,NvmArrayEligibilityStatus *status) {
    for(uint32_t i=0;i<p->counts.carrier.functions;i++) {
        NvmRecordArrayExecutionFunction *f=&p->functions[i];
        if(!nvm_record_map_execution_function(p->plan,i,f) || f->signature.local_count>256 ||
           f->signature.arity>f->signature.local_count || f->maximum_stack>256 ||
           !vm_rmp_charge(p,status,0,(uint64_t)f->signature.arity*4u))return false;
        const uint8_t *tags=vm->module->function_param_types?vm->module->function_param_types[i]:NULL;
        if((tags!=NULL)!=(f->parameter_tags_present!=0))return false;
        if(tags)for(uint16_t j=0;j<f->signature.arity;j++) {
            uint8_t tag;
            if(!nvm_record_map_execution_parameter(p->plan,i,j,&tag) || tag!=tags[j])return false;
        }
    }
    return true;
}
/* Positive-only internal reservation seam. The normal constructor supplies0/0;
 * additions are conservative live overlap, never a credit or trusted token. */
static NvmArrayEligibilityResult vm_rmp_prepare_reserved(const VmState *vm,VmRecordMapPublicPreparation **out,
    uint64_t extra_bytes,uint64_t extra_work) {
    NvmArrayEligibilityResult result={NVM_ARRAY_INVALID,0,0,"I require stable idle ordinary VM storage for preparation."};
    if(!vm || !out || !vm->module || vm->module!=vm->root_module ||
       !vm_rmo_disjoint(out,sizeof *out,vm,sizeof *vm) ||
       !vm_rmo_disjoint(out,sizeof *out,vm->module,sizeof *vm->module) ||
       !pthread_equal(vm->owner_thread,pthread_self()) || vm->stack_size || vm->frame_count ||
       vm->activation_floor || vm->references.active || vm->callee_references.active || vm->handler_count ||
       vm->linked_module_count || vm->callbacks || vm->opcode_trace || vm->heap.gc_running ||
       vm->heap.private_record_dag || !vm->decoded_module_valid || !vm->dispatch_module_valid ||
       (vm->stack_capacity && !vm->stack) || (uint64_t)vm->stack_capacity*sizeof(NanoValue)>SIZE_MAX)return result;
    if((uint64_t)vm->global_capacity*sizeof(NanoValue)>SIZE_MAX ||
       (uint64_t)vm->module_constants.count*sizeof(VmString *)>SIZE_MAX ||
       !vm_rmo_disjoint(out,sizeof *out,vm->stack,(size_t)vm->stack_capacity*sizeof(NanoValue)) ||
       !vm_rmo_disjoint(out,sizeof *out,vm->globals,(size_t)vm->global_capacity*sizeof(NanoValue)) ||
       !vm_rmo_disjoint(out,sizeof *out,vm->module_constants.strings,
           (size_t)vm->module_constants.count*sizeof(VmString *)))return result;
    uint64_t rounded=vm->stack_capacity,stack_bytes=0,stack_work=0;
    const uint64_t required=VM_RM_ROOT_CAPACITY;
    if(required>UINT32_MAX || required>SIZE_MAX/sizeof(NanoValue)){result.status=NVM_ARRAY_LIMIT;return result;}
    if(rounded<required) {
        if(!rounded)rounded=VM_STACK_INITIAL;
        while(rounded<required)rounded*=2u;
        if(rounded>UINT32_MAX || rounded>SIZE_MAX/sizeof(NanoValue))rounded=required;
        uint64_t old_bytes=(uint64_t)vm->stack_capacity*sizeof(NanoValue);
        uint64_t new_bytes=rounded*sizeof(NanoValue);
        if(old_bytes>UINT64_MAX-new_bytes){result.status=NVM_ARRAY_LIMIT;return result;}
        stack_bytes=old_bytes+new_bytes;stack_work=old_bytes+64u;
    }
    uint64_t fixed=sizeof(VmRecordMapPublicPreparation)+VM_RMP_VIEW+VM_RMP_SCRATCH;
    if(stack_bytes>UINT64_MAX-fixed || stack_work>UINT64_MAX-fixed-64u) {
        result.status=NVM_ARRAY_LIMIT;return result;
    }
    if(extra_bytes>UINT64_MAX-fixed-stack_bytes || extra_work>UINT64_MAX-fixed-64u-stack_work) {
        result.status=NVM_ARRAY_LIMIT;return result;
    }
    NvmRecordMapBoundaryPlan *boundary=NULL;
    result=nvm_prepare_record_map_boundary_reserved_private(vm->module,&boundary,
        fixed+stack_bytes+extra_bytes,fixed+64u+stack_work+extra_work);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    VmRecordMapPublicPreparation *p=calloc(1,sizeof *p);
    if(!p) {
        nvm_record_map_boundary_free(boundary);result.status=NVM_ARRAY_MEMORY;
        snprintf(result.message,sizeof result.message,"I could not allocate my borrowed-state preparation owner.");return result;
    }
    p->boundary=boundary;p->plan=nvm_record_map_boundary_execution(boundary);p->module=vm->module;
    p->original_stack=vm->stack;p->original_stack_capacity=vm->stack_capacity;p->rounded_stack_capacity=(uint32_t)rounded;
    NvmRecordMapBoundaryCounts bc;
    NvmArrayEligibilityStatus status=NVM_ARRAY_INVALID;
    if(!p->plan || !nvm_record_map_boundary_counts(boundary,&bc) ||
       !nvm_record_map_execution_counts(p->plan,&p->counts))goto failed;
    p->bytes=bc.peak_bytes_reserved;p->work=bc.work_reserved;
    if(p->counts.carrier.functions>256 || p->counts.carrier.records>256 ||
       p->counts.carrier.frame_limit!=VM_MAX_FRAMES ||
       p->counts.carrier.functions!=vm->module->function_count ||
       p->counts.carrier.strings!=vm->module->string_count ||
       !vm_rmp_charge(p,&status,0,(uint64_t)p->counts.carrier.functions*32u))goto failed;
    if(!vm_rmp_functions(p,vm,&status))goto failed;
    p->instructions=vm_rmp_zero(p,&status,p->counts.carrier.instructions,sizeof *p->instructions);
    if((p->counts.carrier.instructions && !p->instructions) ||
       !vm_rmp_charge(p,&status,0,(uint64_t)p->counts.carrier.instructions*4u))goto failed;
    for(uint32_t i=0;i<p->counts.carrier.instructions;i++)
        if(!nvm_record_map_execution_instruction(p->plan,i,&p->instructions[i]))goto failed;
    if(!vm_rmp_layouts(p,&status) || !vm_rmp_coverage(p,vm,&status) || !vm_rmp_constants(p,vm,&status))goto failed;
    VmRmImportState imported={0};VmRmImportReport report;
    VmRmImportStatus checked=vm_rmi_check_reserved(boundary,vm,&imported,&report,
        p->bytes-bc.peak_bytes_reserved,p->work-bc.work_reserved);
    free(imported.table);
    if(checked!=VM_RM_IMPORT_OK) {
        status=checked==VM_RM_IMPORT_LIMIT?NVM_ARRAY_LIMIT:checked==VM_RM_IMPORT_MEMORY?NVM_ARRAY_MEMORY:
            checked==VM_RM_IMPORT_UNRESOLVED?NVM_ARRAY_UNRESOLVED:NVM_ARRAY_INVALID;
        goto failed;
    }
    p->bytes=report.preparation_bytes;p->work=report.preparation_work;
    *out=p;
    snprintf(result.message,sizeof result.message,"I prepared immutable tables and checked ordinary imports without execution.");
    return result;
failed:
    vm_record_map_public_preparation_free(p);result.status=status;
    snprintf(result.message,sizeof result.message,"I could not prepare exact borrowed-state map tables and imports.");return result;
}

NvmArrayEligibilityResult vm_record_map_public_prepare(const VmState *vm,VmRecordMapPublicPreparation **out) {
    return vm_rmp_prepare_reserved(vm,out,0,0);
}
