/* Private constructor only; no dispatch, run, admission or heap-policy changes. */
#include "record_map_runtime_private.h"
#include "../nanoisa/record_map_structure_private.h"
typedef struct {
    NvmV2LayoutField declaration;
    uint8_t element,nested_leaf,array_depth,key,value;
} VmRecordMapRuntimeField;
typedef struct {
    NvmDeclarationLayout declaration;
    uint32_t field_start;
    uint16_t rank;
} VmRecordMapRuntimeLayout;
struct VmRecordMapPrivate {
    VmState vm;
    NvmRecordMapImage *image;
    const NvmRecordMapExecutionPlan *plan; /* image-owned borrow */
    NvmRecordArrayExecutionCounts counts;
    NvmRecordArrayExecutionFunction functions[256];
    NvmRecordMapExecutionInstruction *instructions;
    VmRecordMapRuntimeLayout layouts[256];
    uint32_t record_layouts[256];
    VmRecordMapRuntimeField *fields;
    uint32_t layout_count,field_count;
    uint64_t bytes,work,byte_limit,work_limit;
    NvmArrayEligibilityStatus preparation_status;
    bool prepared;
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
    uint64_t epoch;
    NanoValue result;
    VmResult last_status;
    uint32_t maximum_frames;
    bool busy,has_result;
#endif
};
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
struct VmRecordMapInvocation {
    VmRecordMapPrivate *owner;
    VmState *vm;
    const NvmModule *module;
    uint64_t epoch;
    uint32_t active_root,root_base;
    bool active;
};
#endif
#define VM_RM_ROOT_CAPACITY (VM_MAX_FRAMES*512u+1u)
_Static_assert(VM_MAX_FRAMES==NVM_RECORD_ARRAY_EXECUTION_FRAMES,"unchanged frame bound");
static bool vm_rm_charge(VmRecordMapPrivate *p,uint64_t b,uint64_t w) {
    if(p->bytes>p->byte_limit || b>p->byte_limit-p->bytes ||
       p->work>p->work_limit || w>p->work_limit-p->work) {
        p->preparation_status=NVM_ARRAY_LIMIT;return false;
    }
    p->bytes+=b;p->work+=w;return true;
}
static void *vm_rm_zero(VmRecordMapPrivate *p,uint64_t n,size_t width) {
    if(width && n>SIZE_MAX/width){p->preparation_status=NVM_ARRAY_LIMIT;return NULL;}
    if(!n)return NULL;
    if(!vm_rm_charge(p,n*width,n*width))return NULL;
    void *v=calloc((size_t)n,width);
    if(!v)p->preparation_status=NVM_ARRAY_MEMORY;
    return v;
}
static bool vm_rm_leaf(uint8_t tag) {
    return tag==TAG_INT || tag==TAG_U8 || tag==TAG_FLOAT || tag==TAG_BOOL || tag==TAG_STRING;
}
/* Independent physical declaration validation precedes the DAG heap policy.
 * I copy every global row, then validate every edge, including unused rows. */
static bool vm_rm_layouts(VmRecordMapPrivate *p) {
    NvmDeclarationCounts count;
    if(!nvm_record_map_execution_declaration_counts(p->plan,&count) || count.layouts>256 ||
       p->counts.records>256)return false;
    p->layout_count=count.layouts;
    if(!vm_rm_charge(p,0,(uint64_t)count.layouts*8+(uint64_t)count.bindings*8u))return false;
    for(uint32_t i=0;i<count.layouts;i++) {
        VmRecordMapRuntimeLayout *layout=&p->layouts[i];
        if(!nvm_record_map_execution_layout(p->plan,i,&layout->declaration))return false;
        layout->field_start=p->field_count;
        if(layout->declaration.fields>65536u-p->field_count)return false;
        p->field_count+=layout->declaration.fields;
    }
    p->fields=vm_rm_zero(p,p->field_count,sizeof *p->fields);
    if(p->field_count && !p->fields)return false;
    if(!vm_rm_charge(p,0,(uint64_t)p->field_count*16))return false;
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
            if(depth==3 && !vm_rm_charge(p,0,1))return false;
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
        if(records>=p->counts.records)return false;
        NvmRecordArrayExecutionDescriptor descriptor;
        if(!nvm_record_map_execution_descriptor(p->plan,records,&descriptor) ||
           descriptor.ordinal!=records || descriptor.layout!=i || descriptor.fields!=layout->declaration.fields)return false;
        p->record_layouts[records++]=i;
        for(uint16_t j=0;j<layout->declaration.fields;j++) {
            VmRecordMapRuntimeField *field=&p->fields[layout->field_start+j];
            if(field->declaration.type_tag==TAG_ARRAY && !field->element)return false;
            if(field->declaration.type_tag==TAG_HASHMAP) {
                if(!vm_rm_charge(p,0,NVM_RECORD_MAP_FIELD_QUERY_WORK) ||
                   !nvm_record_map_execution_map_field(p->plan,i,j,&field->key,&field->value) ||
                   field->key!=TAG_STRING || field->value!=TAG_INT)return false;
            }
        }
    }
    if(records!=p->counts.records)return false;
    /* Explicit DFS: each nominal field edge is examined once. Rank decreases
     * along every physical record edge, so <=256 record nodes can be nested. */
    if(!vm_rm_charge(p,256u+256u*2u*sizeof(uint16_t),256u+256u*2u*sizeof(uint16_t)))return false;
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

static int vm_rm_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
    r->obligations=NVM_RA_FIRST_ERROR_CLEANUP|NVM_RA_CHECK_TAGS;
    switch(op) {
    case OP_PUSH_I64: case OP_PUSH_U8: case OP_PUSH_F64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
        r->recipe=NVM_RA_ROOT_CONSTANT;r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;break;
    case OP_PUSH_STR:
        r->recipe=NVM_RA_ROOT_CONSTANT;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_NOP:r->recipe=NVM_RA_ROOT_SCALAR;r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;break;
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S:
    case OP_I64_REM_S: case OP_I64_NEG: case OP_F64_TO_BITS: case OP_F64_FROM_BITS:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: case OP_F64_NEG:
    case OP_CAST_INT: case OP_CAST_FLOAT: case OP_CAST_BOOL: case OP_CAST_U8: case OP_TYPE_CHECK:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT: case OP_AND: case OP_OR: case OP_NOT:
    case OP_STR_LEN: case OP_STR_CHAR_AT: case OP_ARR_LEN:
    case OP_STR_EQ: case OP_STR_CONTAINS: case OP_STR_STARTS_WITH: case OP_STR_ENDS_WITH:
        r->recipe=NVM_RA_ROOT_SCALAR;break;
    case OP_CAST_STRING: case OP_STR_CONCAT: case OP_STR_SUBSTR: case OP_STR_TRIM:
    case OP_STR_TO_LOWER: case OP_STR_TO_UPPER: case OP_STR_REPLACE:
    case OP_STR_FROM_INT: case OP_STR_FROM_FLOAT: case OP_STR_SPLIT:
        r->recipe=NVM_RA_ROOT_STRING;
        r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_LOAD_LOCAL: case OP_LOAD_GLOBAL:r->recipe=NVM_RA_ROOT_LOAD;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_STORE_LOCAL: case OP_STORE_GLOBAL:r->recipe=NVM_RA_ROOT_STORE;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_DUP:r->recipe=NVM_RA_ROOT_DUP;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_SWAP:r->recipe=NVM_RA_ROOT_SWAP;break;
    case OP_POP: case OP_ASSERT:r->recipe=NVM_RA_ROOT_DROP;break;
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:r->recipe=NVM_RA_ROOT_BRANCH;break;
    case OP_CALL:r->recipe=NVM_RA_ROOT_CALL;r->obligations|=NVM_RA_STAGE_OPERANDS;break;
    case OP_TAIL_CALL:r->recipe=NVM_RA_ROOT_TAIL;r->obligations|=NVM_RA_STAGE_OPERANDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_RET:r->recipe=NVM_RA_ROOT_RETURN;r->obligations|=NVM_RA_STAGE_OPERANDS;break;
    case OP_STRUCT_NEW: case OP_STRUCT_LITERAL: case OP_AGG_PACK:
        r->recipe=NVM_RA_ROOT_CONSTRUCT;
        r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_ARR_NEW: case OP_ARR_LITERAL:
        r->recipe=NVM_RA_ROOT_CONSTRUCT;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_STRUCT_GET: case OP_AGG_GET:
        r->recipe=NVM_RA_ROOT_GET;r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_GET:r->recipe=NVM_RA_ROOT_GET;r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_STRUCT_SET: case OP_AGG_SET:
        r->recipe=NVM_RA_ROOT_SET;r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_SET:r->recipe=NVM_RA_ROOT_SET;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_PUSH:r->recipe=NVM_RA_ROOT_ARRAY_PUSH;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_POP:r->recipe=NVM_RA_ROOT_ARRAY_POP;r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_SLICE:r->recipe=NVM_RA_ROOT_ARRAY_COPY;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_CHECK_BOUNDS;break;
    case OP_HM_NEW:case OP_HM_SET:case OP_HM_GET:case OP_HM_HAS:case OP_HM_LEN:
        r->recipe=NVM_RM_INVALID_BASE_RECIPE;
        r->obligations=NVM_RA_FIRST_ERROR_CLEANUP|NVM_RA_CHECK_TAGS|NVM_RA_STAGE_OPERANDS;
        if(op==OP_HM_NEW || op==OP_HM_SET)r->obligations|=NVM_RA_SAFEPOINT_BEFORE;
        if(op==OP_HM_SET || op==OP_HM_GET)r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;
        break;
    default:return 0;
    }
    return 1;
}
static bool vm_rm_operand(const DecodedInstruction *in,uint8_t index,uint64_t *out) {
    if(index>=in->operand_count)return false;
    switch(in->operand_types[index]) {
    case OPERAND_U8:*out=in->operands[index].u8;return true;
    case OPERAND_U16:*out=in->operands[index].u16;return true;
    case OPERAND_U32:*out=in->operands[index].u32;return true;
    case OPERAND_I32:*out=(uint32_t)in->operands[index].i32;return true;
    case OPERAND_I64:*out=(uint64_t)in->operands[index].i64;return true;
    case OPERAND_F64:memcpy(out,&in->operands[index].f64,sizeof *out);return true;
    default:return false;
    }
}
static bool vm_rm_instruction_same(const NvmRecordArrayExecutionInstruction *fact,
                                    const DecodedInstruction *in) {
    if(fact->opcode!=in->opcode || fact->operand_count!=in->operand_count ||
       fact->next_pc-fact->pc!=in->byte_length)return false;
    for(uint8_t i=0;i<fact->operand_count;i++) {
        uint64_t bits;
        if(fact->operand_types[i]!=(uint8_t)in->operand_types[i] ||
           !vm_rm_operand(in,i,&bits) || bits!=fact->operand_bits[i])return false;
    }
    return true;
}
static bool vm_rm_signature_same(const NvmFunctionEntry *a,const NvmFunctionEntry *b) {
    return a->name_idx==b->name_idx && a->arity==b->arity && a->code_offset==b->code_offset &&
        a->code_length==b->code_length && a->local_count==b->local_count &&
        a->upvalue_count==b->upvalue_count && a->result_tag==b->result_tag && a->result_count==b->result_count;
}
/* Complete defined-field agreement, including dead rows and every branch edge.
 * I run this before heap constants. This slice permits no invocation. */
static bool vm_rm_coverage(VmRecordMapPrivate *p) {
    const NvmModule *m=nvm_record_map_image_borrow(p->image);
    const VmDecodedModule *dm=&p->vm.decoded_module;
    const VmDispatchModule *dispatch=&p->vm.dispatch_module;
    if(!m || m!=p->vm.module || m!=p->vm.root_module ||
       dm->function_count!=p->counts.functions || dispatch->function_count!=p->counts.functions)return false;
    if(!vm_rm_charge(p,0,256+(uint64_t)p->counts.instructions*128+
        (uint64_t)p->counts.functions*32))return false;
    uint32_t supported=0;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction expected={0};
        bool active=vm_rm_recipe((uint8_t)op,&expected)!=0;
        if(active!=(nvm_record_array_opcode_supported((uint8_t)op) || nvm_record_map_opcode((uint8_t)op)))return false;
        supported+=active;
    }
    if(supported!=99)return false;
    uint32_t total=0;
    for(uint32_t fi=0;fi<p->counts.functions;fi++) {
        const NvmRecordArrayExecutionFunction *fn=&p->functions[fi];
        const VmDecodedFunction *decoded=&dm->functions[fi];
        const VmDispatchFunction *body=&dispatch->functions[fi];
        if(!vm_rm_signature_same(&fn->signature,&m->functions[fi]) || fn->instruction_start!=total ||
           decoded->instruction_count!=fn->instruction_count || body->instruction_count!=fn->instruction_count ||
           decoded->code_size!=fn->signature.code_length || body->code_size!=fn->signature.code_length ||
           !decoded->boundaries || !decoded->instruction_indices || !body->offset_to_index ||
           !decoded->boundaries[decoded->code_size])return false;
        if(!vm_rm_charge(p,0,(uint64_t)decoded->code_size+1))return false;
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
                if(callee>=p->counts.functions)return false;
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
    return total==p->counts.instructions;
}


typedef struct { uint64_t bytes,work;NvmArrayEligibilityStatus status; } VmRmBudget;
static bool vm_rm_budget_add(VmRmBudget *b,uint64_t bytes,uint64_t work) {
    if(b->bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES-b->bytes ||
       b->work>NVM_RECORD_ARRAY_EXECUTION_STEPS || work>NVM_RECORD_ARRAY_EXECUTION_STEPS-b->work) {
        b->status=NVM_ARRAY_LIMIT;return false;
    }
    b->bytes+=bytes;b->work+=work;return true;
}
static bool vm_rm_budget_memory(VmRmBudget *b,uint64_t n,uint64_t width) {
    if(width && n>UINT64_MAX/width){b->status=NVM_ARRAY_LIMIT;return false;}
    return vm_rm_budget_add(b,n*width,n*width);
}
static bool vm_rm_budget(const NvmRecordMapExecutionPlan *plan,VmRmBudget *b) {
    NvmRecordMapExecutionCounts all;NvmDeclarationCounts d;
    if(!vm_rm_budget_add(b,0,16) || !nvm_record_map_execution_counts(plan,&all) ||
       !nvm_record_map_execution_declaration_counts(plan,&d))return false;
    const NvmRecordArrayExecutionCounts *c=&all.carrier;
    if(!c->functions || c->functions>256 || c->globals>256 || c->records>256 ||
       c->instructions>65536 || c->strings>NVM_MAX_STRINGS || d.layouts>256 ||
       c->frame_limit!=VM_MAX_FRAMES || c->entry>=c->functions)return false;
    if(!vm_rm_budget_memory(b,1,sizeof(VmRecordMapPrivate)) ||
       !vm_rm_budget_memory(b,c->instructions,sizeof(NvmRecordMapExecutionInstruction)) ||
       !vm_rm_budget_memory(b,VM_RM_ROOT_CAPACITY,sizeof(NanoValue)) ||
       !vm_rm_budget_memory(b,c->globals,sizeof(NanoValue)) ||
       !vm_rm_budget_memory(b,c->functions,sizeof(VmDecodedFunction)+sizeof(VmDispatchFunction)) ||
       !vm_rm_budget_memory(b,1,256u+256u*2u*sizeof(uint16_t)))return false;
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
    if(!vm_rm_budget_memory(b,1,sizeof(VmRecordMapInvocation)))return false;
#endif
    uint32_t total=0;
    for(uint32_t i=0;i<c->functions;i++) {
        NvmRecordArrayExecutionFunction f;
        if(!vm_rm_budget_add(b,0,32) || !nvm_record_map_execution_function(plan,i,&f) ||
           f.signature.local_count>256 || f.maximum_stack>256 ||
           f.signature.arity>f.signature.local_count || f.signature.result_count>1 ||
           f.instruction_start!=total || f.instruction_count>c->instructions-total)return false;
        total+=f.instruction_count;
        uint64_t length=f.signature.code_length;
        if(length>16u*1024u*1024u || !vm_rm_budget_memory(b,length+1,9) ||
           !vm_rm_budget_memory(b,f.instruction_count,sizeof(VmDispatchInstruction)))return false;
        for(uint64_t cap=16;f.instruction_count;cap*=2) {
            if(!vm_rm_budget_memory(b,cap,sizeof(VmDecodedInstruction)))return false;
            if(cap>=f.instruction_count)break;
        }
        if(!vm_rm_budget_add(b,0,4*length+64u*(uint64_t)f.instruction_count+length+1))return false;
    }
    if(total!=c->instructions)return false;
    uint32_t fields=0;
    for(uint32_t i=0;i<d.layouts;i++) {
        NvmDeclarationLayout l;
        if(!vm_rm_budget_add(b,0,16) || !nvm_record_map_execution_layout(plan,i,&l) || l.fields>65536u-fields)return false;
        fields+=l.fields;
    }
    if(!vm_rm_budget_memory(b,fields,sizeof(VmRecordMapRuntimeField)) ||
       !vm_rm_budget_add(b,0,(uint64_t)fields*(16+NVM_RECORD_MAP_FIELD_QUERY_WORK)+
           (uint64_t)d.bindings*64+(uint64_t)d.layouts*8+256u+128u*(uint64_t)c->instructions+32u*c->functions))return false;
    /* All constants reserve independent allocation even for equal bytes. */
    if(!vm_rm_budget_memory(b,c->strings,sizeof(VmString *)))return false;
    uint64_t cap=256;
    if(!vm_rm_budget_memory(b,cap,sizeof(VmString *)))return false;
    while((uint64_t)c->strings*4>cap*3) {
        if(!vm_rm_budget_add(b,0,cap+c->strings))return false;
        cap*=2;if(!vm_rm_budget_memory(b,cap,sizeof(VmString *)))return false;
    }
    for(uint32_t s=0;s<c->strings;s++) {
        uint32_t length;
        if(!vm_rm_budget_add(b,0,8) || !nvm_record_map_execution_size(plan,NVM_RA_SNAPSHOT_STRING,s,&length) ||
           !vm_rm_budget_memory(b,1,(uint64_t)sizeof(VmString)+length+1) ||
           !vm_rm_budget_add(b,0,((uint64_t)length+1)*(s+8u)+16))return false;
    }
    return true;
}

void vm_record_map_private_destroy(VmRecordMapPrivate *p) {
    if(!p || !pthread_equal(p->vm.owner_thread,pthread_self()))return;
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
    if(p->busy)return;
    if(p->has_result)vm_release(&p->vm.heap,p->result);
#endif
    vm_destroy(&p->vm);free(p->fields);free(p->instructions);
    nvm_record_map_image_free(p->image);free(p);
}
static NvmArrayEligibilityResult vm_rm_create_reserved(const NvmRecordMapExecutionPlan *source,VmRecordMapPrivate **out,
        uint64_t extra_bytes,uint64_t extra_work) {
    NvmArrayEligibilityResult result={NVM_ARRAY_INVALID,0,0,"I require a stable private map plan and output."};
    if(!source || !out)return result;
    VmRmBudget budget={0,0,NVM_ARRAY_INVALID};
    if(!vm_rm_budget(source,&budget) || !vm_rm_budget_add(&budget,extra_bytes,extra_work)) {
        result.status=budget.status;return result;
    }
    NvmRecordMapImage *image=NULL;
    result=nvm_record_map_image_copy_reserved(source,budget.bytes,budget.work,&image);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    VmRecordMapPrivate *p=calloc(1,sizeof *p);
    if(!p){nvm_record_map_image_free(image);result.status=NVM_ARRAY_MEMORY;return result;}
    p->image=image;p->plan=nvm_record_map_image_plan(image);
    p->byte_limit=budget.bytes;p->work_limit=budget.work;p->bytes=sizeof *p;p->work=sizeof *p;
    p->preparation_status=NVM_ARRAY_INVALID;p->vm.owner_thread=pthread_self();
    p->vm.cop_in_fd=-1;p->vm.cop_out_fd=-1;p->vm.cop_pid=-1;
    p->vm.cop_sig_send_fd=-1;p->vm.cop_sig_recv_fd=-1;
    NvmRecordMapExecutionCounts all;
    if(!p->plan || !nvm_record_map_image_counts(image,&all))goto invalid;
    p->counts=all.carrier;
    const NvmModule *m=nvm_record_map_image_borrow(image);
    p->vm.module=m;p->vm.root_module=m;
    for(uint32_t i=0;i<p->counts.functions;i++)
        if(!nvm_record_map_execution_function(p->plan,i,&p->functions[i]))goto invalid;
    p->instructions=vm_rm_zero(p,p->counts.instructions,sizeof *p->instructions);
    if(p->counts.instructions && !p->instructions)goto invalid;
    for(uint32_t i=0;i<p->counts.instructions;i++)
        if(!nvm_record_map_image_instruction(image,i,&p->instructions[i]))goto invalid;
    if(!vm_rm_layouts(p))goto invalid;
    char error[VM_DECODE_ERROR_SIZE];
    if(!vm_decode_module(m,&p->vm.decoded_module,error)){p->preparation_status=NVM_ARRAY_MEMORY;goto invalid;}
    p->vm.decoded_module_valid=true;p->vm.dispatch_profile=vm_dispatch_profile_none();
    if(!vm_dispatch_build_module(&p->vm.decoded_module,p->vm.dispatch_profile,&p->vm.dispatch_module,error)) {
        p->preparation_status=NVM_ARRAY_MEMORY;goto invalid;
    }
    p->vm.dispatch_module_valid=true;
    if(!vm_rm_coverage(p))goto invalid;
    p->vm.stack=vm_rm_zero(p,VM_RM_ROOT_CAPACITY,sizeof *p->vm.stack);
    if(!p->vm.stack)goto invalid;
    p->vm.stack_capacity=VM_RM_ROOT_CAPACITY;
    p->vm.globals=vm_rm_zero(p,p->counts.globals,sizeof *p->vm.globals);
    if(p->counts.globals && !p->vm.globals)goto invalid;
    p->vm.global_count=p->counts.globals;p->vm.global_capacity=p->counts.globals;
    vm_heap_init(&p->vm.heap);
    /* Only constants exist. No selected DAG release policy or run entry yet. */
    if(!p->vm.heap.intern_buckets || !vm_module_constants_build(&p->vm,m,&p->vm.module_constants)) {
        p->preparation_status=NVM_ARRAY_MEMORY;goto invalid;
    }
    p->prepared=true;*out=p;
    snprintf(result.message,sizeof result.message,"I prepared independent private map storage without execution admission.");
    return result;
invalid:
    result.status=p->preparation_status;vm_record_map_private_destroy(p);return result;
}
bool vm_record_map_private_preparation(const VmRecordMapPrivate *p,VmRecordMapPreparation *out) {
    NvmRecordMapImageCost cost;
    if(!p || !out || !p->prepared || !nvm_record_map_image_cost(p->image,&cost))return false;
    VmRecordMapPreparation value={p->byte_limit,p->work_limit,cost.preparation_bytes_reserved,
        cost.preparation_work_reserved,sizeof *p,sizeof(VmRecordMapRuntimeField),sizeof(VmRecordMapRuntimeLayout),
        p->counts.functions,p->counts.instructions,p->field_count};
    *out=value;return true;
}
bool vm_record_map_private_instruction(const VmRecordMapPrivate *p,uint32_t i,NvmRecordMapExecutionInstruction *out) {
    if(!p || !p->prepared || !out || i>=p->counts.instructions)return false;
    *out=p->instructions[i];return true;
}

NvmArrayEligibilityResult vm_record_map_private_create(const NvmRecordMapExecutionPlan *source,VmRecordMapPrivate **out) {
    return vm_rm_create_reserved(source,out,0,0);
}
