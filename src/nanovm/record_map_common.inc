/* Shared immutable row/recipe helpers; no owner, observation API or heap policy. */
#include "../nanoisa/record_map_structure_private.h"
#include "record_map_heap_private.h"
typedef struct {
    NvmV2LayoutField declaration;
    uint8_t element,nested_leaf,array_depth,key,value;
} VmRecordMapRuntimeField;
typedef struct {
    NvmDeclarationLayout declaration;
    uint32_t field_start;
    uint16_t rank;
} VmRecordMapRuntimeLayout;
#define VM_RM_ROOT_CAPACITY (VM_MAX_FRAMES*512u+1u)
_Static_assert(VM_MAX_FRAMES==NVM_RECORD_ARRAY_EXECUTION_FRAMES,"unchanged frame bound");
static bool vm_rm_leaf(uint8_t tag) {
    return tag==TAG_INT || tag==TAG_U8 || tag==TAG_FLOAT || tag==TAG_BOOL || tag==TAG_STRING;
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
    case OP_HM_NEW:case OP_HM_SET:case OP_HM_GET:case OP_HM_HAS:case OP_HM_LEN:case OP_HM_DELETE:
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

static VmResult vm_rmx_retain_check(NanoValue value) {
    if(val_is_heap_obj(value) && (!value.as.obj || ((VmHeapHeader *)value.as.obj)->ref_count==UINT32_MAX))return VM_ERR_MEMORY;
    return VM_OK;
}
static VmResult vm_rmx_scalar_inputs(VmState *vm,uint8_t opcode) {
    uint8_t tag=TAG_VOID;uint16_t count=0;
    switch(opcode) {
    case OP_I64_ADD:case OP_I64_SUB:case OP_I64_MUL:case OP_I64_DIV_S:case OP_I64_REM_S:
    case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:
        tag=TAG_INT;count=2;break;
    case OP_I64_NEG:case OP_F64_FROM_BITS:tag=TAG_INT;count=1;break;
    case OP_F64_ADD:case OP_F64_SUB:case OP_F64_MUL:case OP_F64_DIV:
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE:
        tag=TAG_FLOAT;count=2;break;
    case OP_F64_NEG:case OP_F64_TO_BITS:tag=TAG_FLOAT;count=1;break;
    case OP_BOOL_AND:case OP_BOOL_OR:tag=TAG_BOOL;count=2;break;
    case OP_BOOL_NOT:tag=TAG_BOOL;count=1;break;
    case OP_CAST_U8:
        return stack_peek(vm,0).tag==TAG_INT || stack_peek(vm,0).tag==TAG_U8?VM_OK:VM_ERR_TYPE_ERROR;
    default:return VM_OK;
    }
    for(uint16_t i=0;i<count;i++)if(stack_peek(vm,i).tag!=tag)return VM_ERR_TYPE_ERROR;
    return VM_OK;
}
static VmResult vm_rmx_heap_status(VmRmHeapStatus status) {
    if(status==VM_RM_HEAP_OK)return VM_OK;
    if(status==VM_RM_HEAP_MEMORY || status==VM_RM_HEAP_RETAIN_OVERFLOW)return VM_ERR_MEMORY;
    return VM_ERR_TYPE_ERROR;
}
static bool vm_rmo_range(const void *p,size_t size,uintptr_t *begin,uintptr_t *end) {
    if(size&&!p)return false;
    uintptr_t start=(uintptr_t)p;
    if(size>UINTPTR_MAX-start)return false;
    *begin=start;*end=start+size;return true;
}
static bool vm_rmo_disjoint(const void *a,size_t an,const void *b,size_t bn) {
    uintptr_t ab,ae,bb,be;
    if(!vm_rmo_range(a,an,&ab,&ae)||!vm_rmo_range(b,bn,&bb,&be))return false;
    return !an||!bn||ae<=bb||be<=ab;
}
