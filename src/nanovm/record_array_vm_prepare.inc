/* Included only by the private VM build. This owner is never a public proof
 * bit: its copied image, plan, decoded/dispatch tables and VmState stay private. */
#include "record_array_runtime_private.h"
#include "../nanoisa/record_array_snapshot_private.h"
#include "../nanoisa/record_array_structure_private.h"

typedef struct {
    NvmV2LayoutField declaration;
    uint8_t element;
} VmRecordArrayField;
typedef struct {
    NvmDeclarationLayout declaration;
    uint32_t field_start;
    uint16_t rank;
} VmRecordArrayLayout;
struct VmRecordArrayPrivate {
    VmState vm;
    NvmRecordArrayExecutionPlan *plan;
    NvmRecordArrayImage *image;
    NvmRecordArrayExecutionCounts counts;
    NvmRecordArrayExecutionFunction functions[256];
    NvmRecordArrayExecutionInstruction *instructions;
    VmRecordArrayLayout layouts[256];
    uint32_t record_layouts[256];
    VmRecordArrayField *fields;
    uint32_t layout_count,field_count;
    uint64_t bytes,work,epoch;
    NanoValue result;
    NvmArrayEligibilityStatus preparation_status;
    VmResult last_status;
    uint32_t maximum_frames;
    bool prepared,busy,has_result;
};
_Static_assert(VM_MAX_FRAMES==NVM_RECORD_ARRAY_EXECUTION_FRAMES,
               "private ordinary frame depth must match the existing VM");
_Static_assert(VM_RECORD_ARRAY_ROOT_CAPACITY==VM_MAX_FRAMES*512u+1u,
               "private roots include all frame locals and operands plus return staging");

static bool vm_ra_charge(VmRecordArrayPrivate *p,uint64_t bytes,uint64_t work) {
    if(p->bytes>VM_RECORD_ARRAY_EXTRA_BYTES || p->work>VM_RECORD_ARRAY_EXTRA_STEPS ||
       bytes>VM_RECORD_ARRAY_EXTRA_BYTES-p->bytes || work>VM_RECORD_ARRAY_EXTRA_STEPS-p->work) {
        p->preparation_status=NVM_ARRAY_LIMIT;return false;
    }
    p->bytes+=bytes;p->work+=work;return true;
}
static void *vm_ra_zero(VmRecordArrayPrivate *p,uint64_t count,size_t width) {
    if(width && count>SIZE_MAX/width){p->preparation_status=NVM_ARRAY_LIMIT;return NULL;}
    uint64_t bytes=count*width;
    if(!bytes)return NULL;
    if(!vm_ra_charge(p,bytes,bytes))return NULL;
    void *out=calloc((size_t)count,width);
    if(!out)p->preparation_status=NVM_ARRAY_MEMORY;
    return out;
}
static bool vm_ra_leaf(uint8_t tag) {
    return tag==TAG_INT || tag==TAG_U8 || tag==TAG_FLOAT || tag==TAG_BOOL || tag==TAG_STRING;
}
/* Independent physical declaration validation precedes the DAG heap policy.
 * I copy every global row, then validate every edge, including unused rows. */
static bool vm_ra_layouts(VmRecordArrayPrivate *p) {
    NvmDeclarationCounts count;
    if(!nvm_record_array_execution_declaration_counts(p->plan,&count) || count.layouts>256 ||
       p->counts.records>256)return false;
    p->layout_count=count.layouts;
    if(!vm_ra_charge(p,0,(uint64_t)count.layouts*8+(uint64_t)count.bindings*8u))return false;
    for(uint32_t i=0;i<count.layouts;i++) {
        VmRecordArrayLayout *layout=&p->layouts[i];
        if(!nvm_record_array_execution_layout(p->plan,i,&layout->declaration))return false;
        layout->field_start=p->field_count;
        if(layout->declaration.fields>65536u-p->field_count)return false;
        p->field_count+=layout->declaration.fields;
    }
    p->fields=vm_ra_zero(p,p->field_count,sizeof *p->fields);
    if(p->field_count && !p->fields)return false;
    if(!vm_ra_charge(p,0,(uint64_t)p->field_count*16))return false;
    for(uint32_t i=0;i<count.layouts;i++) {
        VmRecordArrayLayout *layout=&p->layouts[i];
        if(layout->declaration.kind!=NVM_V2_LAYOUT_STRUCT &&
           layout->declaration.kind!=NVM_V2_LAYOUT_UNION)return false;
        if(layout->declaration.flags!=(layout->declaration.kind==NVM_V2_LAYOUT_STRUCT?NVM_LAYOUT_COMPLETE:0))return false;
        for(uint16_t j=0;j<layout->declaration.fields;j++) {
            VmRecordArrayField *field=&p->fields[layout->field_start+j];
            if(!nvm_record_array_execution_field(p->plan,i,j,&field->declaration))return false;
            if(layout->declaration.kind!=NVM_V2_LAYOUT_STRUCT)continue;
            uint8_t tag=field->declaration.type_tag;
            if(tag==TAG_STRUCT) {
                uint32_t target=field->declaration.nested_idx;
                if(target>=count.layouts || p->layouts[target].declaration.kind!=NVM_V2_LAYOUT_STRUCT)return false;
            } else if((!vm_ra_leaf(tag) && tag!=TAG_ARRAY) || field->declaration.nested_idx!=NVM_V2_NO_INDEX)return false;
        }
    }
    for(uint32_t i=0;i<count.bindings;i++) {
        NvmOrdinaryArrayBinding binding;NvmOrdinaryArrayType element;
        if(!nvm_record_array_execution_binding(p->plan,i,&binding) || binding.layout>=count.layouts ||
           binding.field>=p->layouts[binding.layout].declaration.fields ||
           !nvm_record_array_execution_type(p->plan,binding.element_type,&element) ||
           !vm_ra_leaf(element.tag) || element.referent!=NVM_V2_NO_INDEX)return false;
        VmRecordArrayField *field=&p->fields[p->layouts[binding.layout].field_start+binding.field];
        if(field->declaration.type_tag!=TAG_ARRAY || field->element)return false;
        field->element=element.tag;
    }
    uint32_t records=0;
    for(uint32_t i=0;i<count.layouts;i++) {
        VmRecordArrayLayout *layout=&p->layouts[i];
        if(layout->declaration.kind!=NVM_V2_LAYOUT_STRUCT)continue;
        if(records>=p->counts.records)return false;
        NvmRecordArrayExecutionDescriptor descriptor;
        if(!nvm_record_array_execution_descriptor(p->plan,records,&descriptor) ||
           descriptor.ordinal!=records || descriptor.layout!=i || descriptor.fields!=layout->declaration.fields)return false;
        p->record_layouts[records++]=i;
        for(uint16_t j=0;j<layout->declaration.fields;j++) {
            VmRecordArrayField *field=&p->fields[layout->field_start+j];
            if(field->declaration.type_tag==TAG_ARRAY && !field->element)return false;
        }
    }
    if(records!=p->counts.records)return false;
    /* Explicit DFS: each nominal field edge is examined once. Rank decreases
     * along every physical record edge, so <=256 record nodes can be nested. */
    if(!vm_ra_charge(p,256u+256u*2u*sizeof(uint16_t),256u+256u*2u*sizeof(uint16_t)))return false;
    uint8_t color[256]={0};
    struct {uint16_t layout,field;} stack[256];
    for(uint32_t root=0;root<count.layouts;root++) {
        if(color[root] || p->layouts[root].declaration.kind!=NVM_V2_LAYOUT_STRUCT)continue;
        uint32_t depth=1;stack[0].layout=(uint16_t)root;stack[0].field=0;color[root]=1;
        while(depth) {
            uint16_t index=stack[depth-1].layout;
            VmRecordArrayLayout *layout=&p->layouts[index];
            if(stack[depth-1].field==layout->declaration.fields) {
                color[index]=2;layout->rank++;depth--;continue;
            }
            VmRecordArrayField *field=&p->fields[layout->field_start+stack[depth-1].field];
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

/* My VM operation table is closed independently of future query changes.
 * Qualification compares all256 opcode decisions, not only executed rows. */
static int vm_ra_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
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
    default:return 0;
    }
    return 1;
}
static bool vm_ra_operand(const DecodedInstruction *in,uint8_t index,uint64_t *out) {
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
static bool vm_ra_instruction_same(const NvmRecordArrayExecutionInstruction *fact,
                                    const DecodedInstruction *in) {
    if(fact->opcode!=in->opcode || fact->operand_count!=in->operand_count ||
       fact->next_pc-fact->pc!=in->byte_length)return false;
    for(uint8_t i=0;i<fact->operand_count;i++) {
        uint64_t bits;
        if(fact->operand_types[i]!=(uint8_t)in->operand_types[i] ||
           !vm_ra_operand(in,i,&bits) || bits!=fact->operand_bits[i])return false;
    }
    return true;
}
static bool vm_ra_signature_same(const NvmFunctionEntry *a,const NvmFunctionEntry *b) {
    return a->name_idx==b->name_idx && a->arity==b->arity && a->code_offset==b->code_offset &&
        a->code_length==b->code_length && a->local_count==b->local_count &&
        a->upvalue_count==b->upvalue_count && a->result_tag==b->result_tag && a->result_count==b->result_count;
}
/* Complete defined-field agreement, including dead rows and every branch edge.
 * I run this before heap constants and before permitting a private invocation. */
static bool vm_ra_coverage(VmRecordArrayPrivate *p) {
    const NvmModule *m=nvm_record_array_image_borrow(p->image);
    const VmDecodedModule *dm=&p->vm.decoded_module;
    const VmDispatchModule *dispatch=&p->vm.dispatch_module;
    if(!m || m!=p->vm.module || m!=p->vm.root_module ||
       dm->function_count!=p->counts.functions || dispatch->function_count!=p->counts.functions)return false;
    if(!vm_ra_charge(p,0,256+(uint64_t)p->counts.instructions*128+
        (uint64_t)p->counts.functions*32))return false;
    uint32_t supported=0;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction expected={0};
        bool active=vm_ra_recipe((uint8_t)op,&expected)!=0;
        if(active!=nvm_record_array_opcode_supported((uint8_t)op))return false;
        supported+=active;
    }
    if(supported!=93)return false;
    uint32_t total=0;
    for(uint32_t fi=0;fi<p->counts.functions;fi++) {
        const NvmRecordArrayExecutionFunction *fn=&p->functions[fi];
        const VmDecodedFunction *decoded=&dm->functions[fi];
        const VmDispatchFunction *body=&dispatch->functions[fi];
        if(!vm_ra_signature_same(&fn->signature,&m->functions[fi]) || fn->instruction_start!=total ||
           decoded->instruction_count!=fn->instruction_count || body->instruction_count!=fn->instruction_count ||
           decoded->code_size!=fn->signature.code_length || body->code_size!=fn->signature.code_length ||
           !decoded->boundaries || !decoded->instruction_indices || !body->offset_to_index ||
           !decoded->boundaries[decoded->code_size])return false;
        if(!vm_ra_charge(p,0,(uint64_t)decoded->code_size+1))return false;
        uint32_t expected_index=0;
        for(uint32_t pc=0;pc<=decoded->code_size;pc++) {
            bool start=expected_index<fn->instruction_count &&
                p->instructions[total+expected_index].pc==pc;
            uint32_t encoded=start?expected_index+1:0;
            if(decoded->boundaries[pc]!=(uint8_t)(start || pc==decoded->code_size) ||
               decoded->instruction_indices[pc]!=encoded || body->offset_to_index[pc]!=encoded)return false;
            expected_index+=start;
        }
        for(uint32_t i=0;i<fn->instruction_count;i++) {
            const NvmRecordArrayExecutionInstruction *fact=&p->instructions[total+i];
            const VmDecodedInstruction *in=&decoded->instructions[i];
            const VmDispatchInstruction *row=&body->instructions[i];
            NvmRecordArrayExecutionInstruction expected={0};
            const InstructionInfo *info=isa_get_info(fact->opcode);
            if(!info || !vm_ra_recipe(fact->opcode,&expected) || fact->recipe!=expected.recipe ||
               fact->obligations!=expected.obligations || fact->function!=fi ||
               fact->pc!=in->byte_offset || fact->next_pc!=in->next_byte_offset ||
               fact->pc!=row->byte_offset || fact->next_pc!=row->next_byte_offset ||
               !vm_ra_instruction_same(fact,&in->instruction) || !vm_ra_instruction_same(fact,&row->instruction) ||
               in->call_handle.resolved || row->call_handle.resolved || in->call_handle.module || row->call_handle.module ||
               in->call_handle.function || row->call_handle.function || in->call_handle.function_index || row->call_handle.function_index ||
               row->super_op!=VM_SUPER_NONE || row->super_operand)return false;
            uint32_t next=i+1<fn->instruction_count?i+1:VM_DISPATCH_NO_INDEX;
            if(row->next_index!=next)return false;
            int pops=info->pop_count,pushes=info->push_count;
            uint32_t callee=UINT32_MAX,resolved=UINT32_MAX,branch=UINT32_MAX,target=UINT32_MAX;
            uint8_t edges=0;uint32_t successors[2]={0,0};
            switch(fact->opcode) {
            case OP_ARR_LITERAL:case OP_STRUCT_LITERAL:pops=in->instruction.operands[1].u16;pushes=1;break;
            case OP_AGG_PACK:pops=in->instruction.operands[3].u16;pushes=1;break;
            case OP_CALL:
                callee=in->instruction.operands[0].u32;
                if(callee>=p->counts.functions)return false;
                pops=m->functions[callee].arity;pushes=m->functions[callee].result_count;resolved=callee;break;
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
            if(fact->opcode!=OP_JMP && fact->opcode!=OP_RET)successors[edges++]=fact->next_pc;
            if(pops<0 || pushes<0 || fact->pops!=pops || fact->pushes!=pushes || fact->callee!=callee ||
               fact->successor_count!=edges || in->resolved_target!=resolved || row->call_target!=callee ||
               row->branch_target!=branch || row->branch_target_offset!=target)return false;
            for(uint8_t j=0;j<edges;j++)if(fact->successors[j]!=successors[j])return false;
        }
        total+=fn->instruction_count;
    }
    return total==p->counts.instructions;
}

/* Reserve decoder realloc overlap conservatively as the sum of every capacity,
 * plus all retained dispatch/offset maps. No allocation occurs in this pass. */
static bool vm_ra_decoder_budget(VmRecordArrayPrivate *p) {
    if(!vm_ra_charge(p,(uint64_t)p->counts.functions*
        (sizeof(VmDecodedFunction)+sizeof(VmDispatchFunction)),
        (uint64_t)p->counts.functions*(sizeof(VmDecodedFunction)+sizeof(VmDispatchFunction))))return false;
    for(uint32_t i=0;i<p->counts.functions;i++) {
        const NvmRecordArrayExecutionFunction *fn=&p->functions[i];
        uint64_t bytes=((uint64_t)fn->signature.code_length+1)*9+
            (uint64_t)fn->instruction_count*sizeof(VmDispatchInstruction);
        uint32_t capacity=0;
        while(capacity<fn->instruction_count) {
            capacity=capacity?capacity*2:16;
            bytes+=(uint64_t)capacity*sizeof(VmDecodedInstruction);
        }
        if(!vm_ra_charge(p,bytes,bytes+(uint64_t)fn->signature.code_length*4+
            (uint64_t)fn->instruction_count*64))return false;
    }
    return true;
}
/* Preparation constants include collision-chain comparison and intern-table
 * growth overlap. Runtime allocations have no invented total/fuel limit. */
static bool vm_ra_constants_budget(VmRecordArrayPrivate *p,const NvmModule *m) {
    uint64_t bytes=(uint64_t)m->string_count*sizeof(VmString *),work=bytes;
    uint32_t capacity=256;
    bytes+=(uint64_t)capacity*sizeof(VmString *);
    while((uint64_t)m->string_count*4>(uint64_t)capacity*3) {
        capacity*=2;bytes+=(uint64_t)capacity*sizeof(VmString *);
    }
    for(uint32_t i=0;i<m->string_count;i++) {
        bytes+=(uint64_t)sizeof(VmString)+m->string_lengths[i]+1;
        work+=((uint64_t)m->string_lengths[i]+1)*(i+8u)+16;
    }
    return vm_ra_charge(p,bytes,work+bytes);
}
NvmArrayEligibilityResult vm_record_array_private_create(const NvmModule *original,
                                                         VmRecordArrayPrivate **out) {
    NvmArrayEligibilityResult result={.status=NVM_ARRAY_INVALID};
    snprintf(result.message,sizeof result.message,"I require a stable module and private output.");
    if(!original || !out)return result;
    if(sizeof(VmRecordArrayPrivate)>VM_RECORD_ARRAY_EXTRA_BYTES ||
       sizeof(VmRecordArrayPrivate)>VM_RECORD_ARRAY_EXTRA_STEPS){result.status=NVM_ARRAY_LIMIT;return result;}
    VmRecordArrayPrivate *p=calloc(1,sizeof *p);
    if(!p){result.status=NVM_ARRAY_MEMORY;return result;}
    p->bytes=sizeof *p;p->work=sizeof *p;p->preparation_status=NVM_ARRAY_INVALID;
    p->result=val_void();p->vm.cop_in_fd=-1;p->vm.cop_out_fd=-1;p->vm.cop_pid=-1;
    p->vm.cop_sig_send_fd=-1;p->vm.cop_sig_recv_fd=-1;
    p->vm.owner_thread=pthread_self();
    result=nvm_prepare_record_array_execution(original,&p->plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)goto fail;
    if(!nvm_record_array_execution_counts(p->plan,&p->counts) || p->counts.functions>256 ||
       p->counts.globals>256 || p->counts.frame_limit!=VM_MAX_FRAMES ||
       p->counts.entry>=p->counts.functions)goto invalid;
    NvmRecordArrayImageCost image_cost;
    if(!nvm_record_array_image_bound(p->plan,&image_cost)) {p->preparation_status=NVM_ARRAY_LIMIT;goto invalid;}
    if(!vm_ra_charge(p,image_cost.bytes_reserved,image_cost.work_charged))goto invalid;
    result=nvm_record_array_image_copy(p->plan,&p->image);
    if(result.status!=NVM_ARRAY_ELIGIBLE)goto fail;
    const NvmModule *m=nvm_record_array_image_borrow(p->image);
    p->vm.module=m;p->vm.root_module=m;
    if(!vm_ra_charge(p,0,(uint64_t)p->counts.functions*32))goto invalid;
    for(uint32_t i=0;i<p->counts.functions;i++) {
        if(!nvm_record_array_execution_function(p->plan,i,&p->functions[i]) ||
           p->functions[i].signature.local_count>256 || p->functions[i].maximum_stack>256 ||
           p->functions[i].signature.arity>p->functions[i].signature.local_count ||
           p->functions[i].signature.result_count>1)goto invalid;
    }
    p->instructions=vm_ra_zero(p,p->counts.instructions,sizeof *p->instructions);
    if(p->counts.instructions && !p->instructions)goto invalid;
    if(!vm_ra_charge(p,0,(uint64_t)p->counts.instructions*sizeof *p->instructions))goto invalid;
    for(uint32_t i=0;i<p->counts.instructions;i++)
        if(!nvm_record_array_execution_instruction(p->plan,i,&p->instructions[i]))goto invalid;
    if(!vm_ra_layouts(p) || !vm_ra_decoder_budget(p))goto invalid;
    char error[VM_DECODE_ERROR_SIZE];
    if(!vm_decode_module(m,&p->vm.decoded_module,error)) {
        p->preparation_status=NVM_ARRAY_MEMORY;goto invalid;
    }
    p->vm.decoded_module_valid=true;
    p->vm.dispatch_profile=vm_dispatch_profile_none();
    if(!vm_dispatch_build_module(&p->vm.decoded_module,p->vm.dispatch_profile,&p->vm.dispatch_module,error)) {
        p->preparation_status=NVM_ARRAY_MEMORY;goto invalid;
    }
    p->vm.dispatch_module_valid=true;
    if(!vm_ra_coverage(p) || !vm_ra_constants_budget(p,m))goto invalid;
    p->vm.stack=vm_ra_zero(p,VM_RECORD_ARRAY_ROOT_CAPACITY,sizeof *p->vm.stack);
    if(!p->vm.stack)goto invalid;
    p->vm.stack_capacity=VM_RECORD_ARRAY_ROOT_CAPACITY;
    p->vm.globals=vm_ra_zero(p,p->counts.globals,sizeof *p->vm.globals);
    if(p->counts.globals && !p->vm.globals)goto invalid;
    p->vm.global_count=p->counts.globals;p->vm.global_capacity=p->counts.globals;
    vm_heap_init(&p->vm.heap);
    p->vm.heap.private_record_dag=true;
    if(!p->vm.heap.intern_buckets || !vm_module_constants_build(&p->vm,m,&p->vm.module_constants)) {
        p->preparation_status=NVM_ARRAY_MEMORY;goto invalid;
    }
    /* No mutable original pointer survives preparation. Ordinary verified and
     * admission caches remain false; only the scoped private runner can enter. */
    p->prepared=true;p->last_status=VM_OK;*out=p;return result;
invalid:
    result.status=p->preparation_status;
    snprintf(result.message,sizeof result.message,"I could not prepare exact private VM storage and correspondence.");
fail:
    vm_record_array_private_destroy(p);return result;
}
void vm_record_array_private_destroy(VmRecordArrayPrivate *p) {
    if(!p || p->busy || !pthread_equal(p->vm.owner_thread,pthread_self()))return;
    if(p->has_result)vm_release(&p->vm.heap,p->result);
    vm_destroy(&p->vm);
    free(p->fields);free(p->instructions);
    nvm_record_array_image_free(p->image);
    nvm_record_array_execution_free(p->plan);free(p);
}
static bool vm_ra_observation_value(const VmRecordArrayPrivate *p,VmRecordArrayRoot root,
                                    uint32_t index,const uint32_t *path,uint16_t depth,NanoValue *out) {
    if(!p || !p->prepared || p->busy || !pthread_equal(p->vm.owner_thread,pthread_self()) ||
       depth>VM_RECORD_ARRAY_PATH_MAX || (depth && !path))return false;
    NanoValue value;
    if(root==VM_RA_RESULT) {
        if(index || !p->has_result)return false;
        value=p->result;
    } else if(root==VM_RA_GLOBAL) {
        if(index>=p->vm.global_count)return false;
        value=p->vm.globals[index];
    } else return false;
    for(uint16_t i=0;i<depth;i++) {
        if(value.tag==TAG_STRUCT) {
            if(!value.as.sval || path[i]>=value.as.sval->field_count)return false;
            value=value.as.sval->fields[path[i]];
        } else if(value.tag==TAG_ARRAY) {
            if(!value.as.array || path[i]>=value.as.array->length)return false;
            value=vm_array_get(value.as.array,path[i]);
        } else return false;
    }
    *out=value;return true;
}
bool vm_record_array_private_observe(const VmRecordArrayPrivate *p,VmRecordArrayRoot root,
        uint32_t index,const uint32_t *path,uint16_t depth,VmRecordArrayObservation *out) {
    NanoValue value;
    if(!out || !vm_ra_observation_value(p,root,index,path,depth,&value))return false;
    VmRecordArrayObservation observed={.epoch=p->epoch,.tag=value.tag,.layout=UINT32_MAX};
    if(val_is_heap_obj(value)) {
        if(!value.as.obj)return false;
        /* This private identity is compared only within a single frozen epoch;
         * I expose no mutable pointer and never dereference a caller identity. */
        observed.identity=(uint64_t)(uintptr_t)value.as.obj;
        if(value.tag==TAG_STRING)observed.length=value.as.string->length;
        else if(value.tag==TAG_ARRAY) {
            observed.length=value.as.array->length;observed.element_tag=value.as.array->elem_type;
        } else if(value.tag==TAG_STRUCT) {
            observed.length=value.as.sval->field_count;observed.layout=value.as.sval->def_idx;
        } else return false;
    } else if(value.tag!=TAG_VOID)memcpy(&observed.scalar_bits,&value.as,sizeof observed.scalar_bits);
    *out=observed;return true;
}
bool vm_record_array_private_string(const VmRecordArrayPrivate *p,VmRecordArrayRoot root,
        uint32_t index,const uint32_t *path,uint16_t depth,uint32_t offset,uint32_t count,void *out) {
    NanoValue value;
    if((count && !out) || !vm_ra_observation_value(p,root,index,path,depth,&value) ||
       value.tag!=TAG_STRING || !value.as.string || offset>value.as.string->length ||
       count>value.as.string->length-offset)return false;
    if(count)memcpy(out,value.as.string->data+offset,count);
    return true;
}
bool vm_record_array_private_stats(const VmRecordArrayPrivate *p,VmRecordArrayPrivateStats *out) {
    if(!p || !out || p->busy || !pthread_equal(p->vm.owner_thread,pthread_self()))return false;
    VmRecordArrayPrivateStats result={0};
    result.preparation_bytes=p->bytes+p->counts.peak_bytes_reserved;
    result.preparation_steps=p->work+p->counts.work_reserved;
    result.epoch=p->epoch;result.heap_objects=p->vm.heap.stats.num_objects;
    result.heap_live_bytes=p->vm.heap.stats.allocated-p->vm.heap.stats.freed;
    result.active_frames=p->vm.frame_count;result.active_stack=p->vm.stack_size;
    result.maximum_frames=p->maximum_frames;result.last_status=p->last_status;result.has_result=p->has_result;
    *out=result;return true;
}
static bool vm_ra_value_valid(const VmRecordArrayPrivate *p,NanoValue value) {
    if(value.tag==TAG_VOID || vm_ra_leaf(value.tag)) {
        if(value.tag!=TAG_STRING)return true;
        return value.as.string && value.as.string->header.obj_type==TAG_STRING && value.as.string->header.ref_count;
    }
    if(value.tag==TAG_ARRAY) {
        VmArray *a=value.as.array;
        return a && a->header.obj_type==TAG_ARRAY && a->header.ref_count &&
            vm_ra_leaf(a->elem_type) && a->length<=a->capacity &&
            a->unboxed==vm_array_type_unboxable(a->elem_type) &&
            (!a->capacity || (a->unboxed?a->packed!=NULL:a->elements!=NULL));
    }
    if(value.tag==TAG_STRUCT) {
        VmStruct *s=value.as.sval;
        return s && s->header.obj_type==TAG_STRUCT && s->header.ref_count &&
            s->def_idx<p->layout_count && p->layouts[s->def_idx].declaration.kind==NVM_V2_LAYOUT_STRUCT &&
            s->field_count==p->layouts[s->def_idx].declaration.fields && !s->field_names &&
            (!s->field_count || s->fields);
    }
    return false;
}
static bool vm_ra_field_accepts(const VmRecordArrayPrivate *p,uint32_t layout,uint16_t field,NanoValue value) {
    if(layout>=p->layout_count || field>=p->layouts[layout].declaration.fields || !vm_ra_value_valid(p,value))return false;
    const VmRecordArrayField *expected=&p->fields[p->layouts[layout].field_start+field];
    if(value.tag!=expected->declaration.type_tag)return false;
    if(value.tag==TAG_STRUCT)return value.as.sval->def_idx==expected->declaration.nested_idx;
    if(value.tag==TAG_ARRAY)return value.as.array->elem_type==expected->element;
    return true;
}
static VmResult vm_ra_retain_check(NanoValue value) {
    if(val_is_heap_obj(value) && (!value.as.obj || ((VmHeapHeader *)value.as.obj)->ref_count==UINT32_MAX))return VM_ERR_MEMORY;
    return VM_OK;
}
static bool vm_ra_context(const VmRecordArrayPrivate *p,const VmState *vm) {
    return p && p->prepared && p->busy && vm==&p->vm && vm->module==nvm_record_array_image_borrow(p->image) &&
        vm->root_module==vm->module && !vm->verified && vm->heap.private_record_dag &&
        vm->decoded_module_valid && vm->dispatch_module_valid && !vm->dispatch_profile.fuse_load_local_field &&
        vm->frame_count && vm->frame_count<=VM_MAX_FRAMES && vm->current_fn<p->counts.functions &&
        vm->stack_capacity==VM_RECORD_ARRAY_ROOT_CAPACITY && vm->stack_size<=vm->stack_capacity;
}
/* Runtime root/nominal preflight occurs before a handler pops or overwrites.
 * The real VM still performs its original scalar/error operations. */
static VmResult vm_ra_preflight(VmRecordArrayPrivate *p,const VmDispatchInstruction *instruction) {
    VmState *vm=&p->vm;
    if(!vm_ra_context(p,vm))return VM_ERR_TYPE_ERROR;
    const VmCallFrame *frame=&vm->frames[vm->frame_count-1];
    const NvmRecordArrayExecutionFunction *fn=&p->functions[vm->current_fn];
    if(frame->module!=vm->module || frame->fn_idx!=vm->current_fn ||
       frame->local_count!=fn->signature.local_count || frame->effect_owner || frame->closure ||
       frame->owned_callable.tag!=TAG_VOID || frame->stack_base>vm->stack_size ||
       frame->local_count>vm->stack_size-frame->stack_base)return VM_ERR_TYPE_ERROR;
    uint32_t encoded=vm->decoded_module.functions[vm->current_fn].instruction_indices[instruction->byte_offset];
    if(!encoded || encoded>fn->instruction_count)return VM_ERR_DECODE;
    const NvmRecordArrayExecutionInstruction *fact=&p->instructions[fn->instruction_start+encoded-1];
    if(!vm_ra_instruction_same(fact,&instruction->instruction) || instruction->super_op!=VM_SUPER_NONE)return VM_ERR_DECODE;
    uint32_t operands=vm->stack_size-frame->stack_base-frame->local_count;
    if(operands>256)return VM_ERR_STACK_OVERFLOW;
    if(fact->pops>operands)return VM_ERR_STACK_UNDERFLOW;
    if(fact->pushes>vm->stack_capacity-(vm->stack_size-fact->pops))return VM_ERR_STACK_OVERFLOW;
    for(uint16_t i=0;i<fact->pops;i++)if(!vm_ra_value_valid(p,stack_peek(vm,i)))return VM_ERR_TYPE_ERROR;
    const DecodedInstruction *in=&instruction->instruction;
    NanoValue retained=val_void();
    switch(in->opcode) {
    case OP_PUSH_STR:
        if(in->operands[0].u32>=vm->module_constants.count)return VM_ERR_OUT_OF_BOUNDS;
        retained=val_string(vm->module_constants.strings[in->operands[0].u32]);break;
    case OP_DUP:retained=stack_peek(vm,0);break;
    case OP_LOAD_LOCAL:case OP_STORE_LOCAL:
        if(in->operands[0].u16>=frame->local_count)return VM_ERR_OUT_OF_BOUNDS;
        if(in->opcode==OP_LOAD_LOCAL)retained=vm->stack[frame->stack_base+in->operands[0].u16];break;
    case OP_LOAD_GLOBAL:case OP_STORE_GLOBAL:
        if(in->operands[0].u32>=vm->global_count)return VM_ERR_OUT_OF_BOUNDS;
        if(in->opcode==OP_LOAD_GLOBAL)retained=vm->globals[in->operands[0].u32];break;
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        if(!vm_ra_leaf(in->operands[0].u8))return VM_ERR_TYPE_ERROR;
        if(in->opcode==OP_ARR_LITERAL)
            for(uint16_t i=0;i<fact->pops;i++)if(stack_peek(vm,i).tag!=in->operands[0].u8)return VM_ERR_TYPE_ERROR;
        break;
    case OP_ARR_PUSH:case OP_ARR_SET: {
        NanoValue receiver=stack_peek(vm,in->opcode==OP_ARR_PUSH?1:2),value=stack_peek(vm,0);
        if(receiver.tag!=TAG_ARRAY || value.tag!=receiver.as.array->elem_type)return VM_ERR_TYPE_ERROR;
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
            if(!vm_ra_field_accepts(p,layout,field,stack_peek(vm,fact->pops-1-field)))return VM_ERR_TYPE_ERROR;
        break;
    }
    case OP_STRUCT_GET:case OP_AGG_GET:case OP_STRUCT_SET:case OP_AGG_SET: {
        bool store=in->opcode==OP_STRUCT_SET || in->opcode==OP_AGG_SET;
        NanoValue receiver=stack_peek(vm,store?1:0);
        uint16_t field=in->operands[0].u16;
        if(receiver.tag!=TAG_STRUCT)return VM_ERR_TYPE_ERROR;
        if(field>=receiver.as.sval->field_count)return VM_ERR_OUT_OF_BOUNDS;
        if(store) {
            if(!vm_ra_field_accepts(p,receiver.as.sval->def_idx,field,stack_peek(vm,0)))return VM_ERR_TYPE_ERROR;
        } else retained=receiver.as.sval->fields[field];
        break;
    }
    case OP_CALL: {
        if(fact->callee>=p->counts.functions)return VM_ERR_UNDEFINED_FUNCTION;
        const NvmFunctionEntry *callee=&p->functions[fact->callee].signature;
        if(vm->frame_count==VM_MAX_FRAMES)return VM_ERR_CALL_DEPTH;
        if(callee->local_count>vm->stack_capacity-(vm->stack_size-callee->arity))return VM_ERR_STACK_OVERFLOW;
        if(vm->module->function_param_types && vm->module->function_param_types[fact->callee])
            for(uint16_t i=0;i<callee->arity;i++) {
                uint8_t tag=vm->module->function_param_types[fact->callee][i];
                if(tag!=TAG_VOID && stack_peek(vm,callee->arity-1-i).tag!=tag)return VM_ERR_TYPE_ERROR;
            }
        break;
    }
    default:break;
    }
    if(!vm_ra_value_valid(p,retained))return VM_ERR_TYPE_ERROR;
    if(vm->frame_count>p->maximum_frames)p->maximum_frames=vm->frame_count;
    return vm_ra_retain_check(retained);
}
