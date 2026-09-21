#include "nvm2c_record_array_private.h"
#ifdef NANO_RECORD_ARRAY_GENERATED_PRIVATE
#include "record_array_generated_private.h"
#include "record_array_structure_private.h"
#include "../binary64_arithmetic_source.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
/* I stage the entire product. Reallocation counts old and replacement capacity
 * simultaneously; work charges every formatted/copied byte before publishing. */
typedef struct {
    char *text;size_t used,capacity;
    uint64_t live,peak,work;
    NvmArrayEligibilityStatus status;
} RgOutput;
static bool rg_charge(RgOutput *b,uint64_t bytes,uint64_t work) {
    if(b->status!=NVM_ARRAY_ELIGIBLE)return false;
    if(b->live>NRG_EXTRA_BYTES || b->work>NRG_EXTRA_STEPS ||
       bytes>NRG_EXTRA_BYTES-b->live || work>NRG_EXTRA_STEPS-b->work) {
        b->status=NVM_ARRAY_LIMIT;return false;
    }
    b->live+=bytes;b->work+=work;if(b->peak<b->live)b->peak=b->live;return true;
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf,2,3)))
#endif
static bool rg_write(RgOutput *b,const char *format,...) {
    if(b->status!=NVM_ARRAY_ELIGIBLE)return false;
    va_list args,copy;va_start(args,format);va_copy(copy,args);
    int n=vsnprintf(NULL,0,format,copy);va_end(copy);
    if(n<0 || (uint64_t)n>SIZE_MAX-b->used-1) { va_end(args);b->status=NVM_ARRAY_LIMIT;return false; }
    size_t need=b->used+(size_t)n+1;
    if(!rg_charge(b,0,(uint64_t)n*2+1)) { va_end(args);return false; }
    if(need>b->capacity) {
        size_t capacity=b->capacity?b->capacity:4096;
        while(capacity<need) {
            if(capacity>NRG_EXTRA_BYTES/2) { capacity=need;break; }
            capacity*=2;
        }
        if(!rg_charge(b,capacity,b->used)) { va_end(args);return false; }
        char *replacement=malloc(capacity);
        if(!replacement) { b->live-=capacity;b->status=NVM_ARRAY_MEMORY;va_end(args);return false; }
        if(b->used)memcpy(replacement,b->text,b->used);
        free(b->text);b->live-=b->capacity;b->text=replacement;b->capacity=capacity;
    }
    vsnprintf(b->text+b->used,b->capacity-b->used,format,args);va_end(args);
    b->used+=(size_t)n;return true;
}
static int rg_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
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

static bool rg_args(RgOutput *b,uint32_t count,uint32_t tag) {
    if(count==1)rg_write(b,"NmsValue a,r={0,0}; if(!nrg_peek(p,0,&a))return;\n");
    else rg_write(b,"NmsValue a,b,r={0,0}; if(!nrg_peek(p,1,&a)||!nrg_peek(p,0,&b))return;\n");
    if(tag && count==1)rg_write(b,"if(a.tag!=%u){nrg_fail(p,NRG_TYPE);return;}\n",tag);
    else if(tag)rg_write(b,"if(a.tag!=%u||b.tag!=%u){nrg_fail(p,NRG_TYPE);return;}\n",tag,tag);
    return b->status==NVM_ARRAY_ELIGIBLE;
}
static bool rg_operation(RgOutput *b,const NvmRecordArrayExecutionInstruction *r) {
    uint64_t arg=r->operand_bits[0];
    rg_write(b,"b%u:; {\n",r->pc);
    switch(r->opcode) {
    case OP_PUSH_I64:rg_write(b,"NmsValue v={UINT64_C(%" PRIu64 "),1}; if(!nrg_push_move(p,&v))return;\n",(uint64_t)(arg));break;
    case OP_PUSH_U8:rg_write(b,"NmsValue v={UINT64_C(%" PRIu64 "),2}; if(!nrg_push_move(p,&v))return;\n",(uint64_t)(arg));break;
    case OP_PUSH_F64:rg_write(b,"NmsValue v={UINT64_C(%" PRIu64 "),3}; if(!nrg_push_move(p,&v))return;\n",(uint64_t)(arg));break;
    case OP_PUSH_BOOL:rg_write(b,"NmsValue v={UINT64_C(%" PRIu64 "),4}; if(!nrg_push_move(p,&v))return;\n",(uint64_t)(arg!=0));break;
    case OP_PUSH_VOID:rg_write(b,"NmsValue v={UINT64_C(%" PRIu64 "),0}; if(!nrg_push_move(p,&v))return;\n",(uint64_t)(0));break;
    case OP_PUSH_STR:rg_write(b,"NmsValue v={UINT64_C(%" PRIu64 "),5}; if(!nrg_push_move(p,&v))return;\n",(uint64_t)(arg+1));break;
    case OP_NOP:break;
    case OP_I64_ADD:rg_args(b,2,1);rg_write(b,"r=(NmsValue){a.payload+b.payload,1}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_SUB:rg_args(b,2,1);rg_write(b,"r=(NmsValue){a.payload-b.payload,1}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_MUL:rg_args(b,2,1);rg_write(b,"r=(NmsValue){a.payload*b.payload,1}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_DIV_S:rg_args(b,2,1);rg_write(b,"r=(NmsValue){b.payload==0?0:(a.payload==UINT64_C(9223372036854775808)&&b.payload==UINT64_MAX)?a.payload:(uint64_t)((int64_t)a.payload/(int64_t)b.payload),1}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_REM_S:rg_args(b,2,1);rg_write(b,"r=(NmsValue){b.payload==0||(a.payload==UINT64_C(9223372036854775808)&&b.payload==UINT64_MAX)?0:(uint64_t)((int64_t)a.payload%%(int64_t)b.payload),1}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_NEG:rg_args(b,1,1);rg_write(b,"r=(NmsValue){UINT64_C(0)-a.payload,1}; if(!nrg_replace(p,1,&r))return;\n");break;
    case OP_F64_FROM_BITS:rg_args(b,1,1);rg_write(b,"r=(NmsValue){a.payload,3}; if(!nrg_replace(p,1,&r))return;\n");break;
    case OP_F64_TO_BITS:rg_args(b,1,3);rg_write(b,"r=(NmsValue){a.payload,1}; if(!nrg_replace(p,1,&r))return;\n");break;
    case OP_F64_ADD:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_bits(nano_rt_f64_add(rg_float(a.payload),rg_float(b.payload))),3}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_SUB:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_bits(nano_rt_f64_sub(rg_float(a.payload),rg_float(b.payload))),3}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_MUL:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_bits(nano_rt_f64_mul(rg_float(a.payload),rg_float(b.payload))),3}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_DIV:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_bits(nano_rt_f64_div(rg_float(a.payload),rg_float(b.payload))),3}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_NEG:rg_args(b,1,3);rg_write(b,"r=(NmsValue){rg_bits(-rg_float(a.payload)),3}; if(!nrg_replace(p,1,&r))return;\n");break;
    case OP_I64_EQ:rg_args(b,2,1);rg_write(b,"r=(NmsValue){(int64_t)a.payload==(int64_t)b.payload,4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_NE:rg_args(b,2,1);rg_write(b,"r=(NmsValue){(int64_t)a.payload!=(int64_t)b.payload,4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_LT_S:rg_args(b,2,1);rg_write(b,"r=(NmsValue){(int64_t)a.payload<(int64_t)b.payload,4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_LE_S:rg_args(b,2,1);rg_write(b,"r=(NmsValue){(int64_t)a.payload<=(int64_t)b.payload,4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_GT_S:rg_args(b,2,1);rg_write(b,"r=(NmsValue){(int64_t)a.payload>(int64_t)b.payload,4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_I64_GE_S:rg_args(b,2,1);rg_write(b,"r=(NmsValue){(int64_t)a.payload>=(int64_t)b.payload,4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_EQ:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_float(a.payload)==rg_float(b.payload),4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_NE:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_float(a.payload)!=rg_float(b.payload),4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_LT:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_float(a.payload)<rg_float(b.payload),4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_LE:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_float(a.payload)<=rg_float(b.payload),4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_GT:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_float(a.payload)>rg_float(b.payload),4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_F64_GE:rg_args(b,2,3);rg_write(b,"r=(NmsValue){rg_float(a.payload)>=rg_float(b.payload),4}; if(!nrg_replace(p,2,&r))return;\n");break;
    case OP_EQ:rg_args(b,2,0);rg_write(b,"r=(NmsValue){nrg_equal(p,&a,&b),4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_NE:rg_args(b,2,0);rg_write(b,"r=(NmsValue){!nrg_equal(p,&a,&b),4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_LT:rg_args(b,2,0);rg_write(b,"r=(NmsValue){nrg_order(p,&a,&b)<0,4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_LE:rg_args(b,2,0);rg_write(b,"r=(NmsValue){nrg_order(p,&a,&b)<=0,4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_GT:rg_args(b,2,0);rg_write(b,"r=(NmsValue){nrg_order(p,&a,&b)>0,4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_GE:rg_args(b,2,0);rg_write(b,"r=(NmsValue){nrg_order(p,&a,&b)>=0,4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_BOOL_AND:rg_args(b,2,4);rg_write(b,"r=(NmsValue){a.payload&&b.payload,4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_BOOL_OR:rg_args(b,2,4);rg_write(b,"r=(NmsValue){a.payload||b.payload,4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_AND:rg_args(b,2,0);rg_write(b,"r=(NmsValue){nrg_truth(p,&a)&&nrg_truth(p,&b),4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_OR:rg_args(b,2,0);rg_write(b,"r=(NmsValue){nrg_truth(p,&a)||nrg_truth(p,&b),4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_BOOL_NOT:rg_args(b,1,4);rg_write(b,"r=(NmsValue){!a.payload,4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,1,&r))return;\n");break;
    case OP_NOT:rg_args(b,1,0);rg_write(b,"r=(NmsValue){!nrg_truth(p,&a),4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,1,&r))return;\n");break;
    case OP_CAST_BOOL:rg_args(b,1,0);rg_write(b,"r=(NmsValue){nrg_truth(p,&a),4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,1,&r))return;\n");break;
    case OP_CAST_U8:rg_args(b,1,0);rg_write(b,"if(a.tag!=1&&a.tag!=2){nrg_fail(p,NRG_TYPE);return;} r=(NmsValue){(uint8_t)a.payload,2}; if(!nrg_replace(p,1,&r))return;\n");break;
    case OP_TYPE_CHECK:rg_args(b,1,0);rg_write(b,"r=(NmsValue){a.tag==%u,4}; if(!nrg_replace(p,1,&r))return;\n",(unsigned)arg);break;
    case OP_CAST_INT:rg_write(b,"if(!nrg_cast_int(p))return;\n");break;
    case OP_CAST_FLOAT:rg_write(b,"if(!nrg_cast_float(p))return;\n");break;
    case OP_CAST_STRING:rg_write(b,"if(!nrg_cast_string(p))return;\n");break;
    case OP_STR_FROM_INT:rg_write(b,"if(!nrg_format(p,1))return;\n");break;
    case OP_STR_FROM_FLOAT:rg_write(b,"if(!nrg_format(p,3))return;\n");break;
    case OP_STR_CONCAT:rg_write(b,"if(!nrg_concat(p))return;\n");break;
    case OP_STR_SUBSTR:rg_write(b,"if(!nrg_substring(p))return;\n");break;
    case OP_STR_TRIM:rg_write(b,"if(!nrg_trim(p))return;\n");break;
    case OP_STR_TO_LOWER:rg_write(b,"if(!nrg_case(p,false))return;\n");break;
    case OP_STR_TO_UPPER:rg_write(b,"if(!nrg_case(p,true))return;\n");break;
    case OP_STR_SPLIT:rg_write(b,"if(!nrg_split(p))return;\n");break;
    case OP_STR_REPLACE:rg_write(b,"if(!nrg_string_replace(p))return;\n");break;
    case OP_STR_LEN:rg_write(b,"if(!nrg_length(p,false))return;\n");break;
    case OP_ARR_LEN:rg_write(b,"if(!nrg_length(p,true))return;\n");break;
    case OP_STR_CHAR_AT:rg_write(b,"if(!nrg_character(p))return;\n");break;
    case OP_STR_CONTAINS:rg_write(b,"if(!nrg_predicate(p,0))return;\n");break;
    case OP_STR_STARTS_WITH:rg_write(b,"if(!nrg_predicate(p,1))return;\n");break;
    case OP_STR_ENDS_WITH:rg_write(b,"if(!nrg_predicate(p,2))return;\n");break;
    case OP_ARR_GET:rg_write(b,"if(!nrg_array_get(p))return;\n");break;
    case OP_ARR_SET:rg_write(b,"if(!nrg_array_set(p))return;\n");break;
    case OP_ARR_PUSH:rg_write(b,"if(!nrg_array_push(p))return;\n");break;
    case OP_ARR_POP:rg_write(b,"if(!nrg_array_pop(p))return;\n");break;
    case OP_ARR_SLICE:rg_write(b,"if(!nrg_array_slice(p))return;\n");break;
    case OP_POP:rg_write(b,"if(!nrg_drop(p))return;\n");break;
    case OP_DUP:rg_write(b,"if(!nrg_dup(p))return;\n");break;
    case OP_SWAP:rg_write(b,"if(!nrg_swap(p))return;\n");break;
    case OP_STR_EQ:rg_args(b,2,5);rg_write(b,"r=(NmsValue){nrg_equal(p,&a,&b),4}; if(nrg_status(p)!=NRG_OK||!nrg_replace(p,2,&r))return;\n");break;
    case OP_LOAD_LOCAL:case OP_LOAD_GLOBAL:case OP_STORE_LOCAL:case OP_STORE_GLOBAL:
        rg_write(b,"if(!nrg_%s(p,%s,%u))return;\n",r->recipe==NVM_RA_ROOT_LOAD?"load":"store",
            r->opcode==OP_LOAD_GLOBAL||r->opcode==OP_STORE_GLOBAL?"true":"false",(unsigned)arg);break;
    case OP_STRUCT_NEW:case OP_STRUCT_LITERAL:case OP_AGG_PACK:
        rg_write(b,"if(!nrg_record_new(p,%u))return;\n",(unsigned)(r->opcode==OP_AGG_PACK?r->operand_bits[1]:arg));break;
    case OP_STRUCT_GET:case OP_AGG_GET:case OP_STRUCT_SET:case OP_AGG_SET:
        rg_write(b,"if(!nrg_record_%s(p,%u,%s))return;\n",r->recipe==NVM_RA_ROOT_GET?"get":"set",(unsigned)arg,
            r->opcode==OP_AGG_GET||r->opcode==OP_AGG_SET?"true":"false");break;
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        rg_write(b,"if(!nrg_array_new(p,%u,%u,%s))return;\n",(unsigned)arg,
            r->opcode==OP_ARR_LITERAL?(unsigned)r->operand_bits[1]:0,r->opcode==OP_ARR_LITERAL?"true":"false");break;
    case OP_ASSERT:
        rg_write(b,"NmsValue a;if(!nrg_peek(p,0,&a))return;bool ok=nrg_truth(p,&a);if(!nrg_drop(p))return;if(!ok){nrg_fail(p,NRG_ASSERT);return;}\n");break;
    case OP_JMP:rg_write(b,"goto b%u;\n",r->successors[0]);break;
    case OP_JMP_TRUE:case OP_JMP_FALSE:
        rg_write(b,"NmsValue a;if(!nrg_peek(p,0,&a))return;bool take=nrg_truth(p,&a);if(!nrg_drop(p))return;if(%stake)goto b%u;goto b%u;\n",
            r->opcode==OP_JMP_FALSE?"!":"",(uint32_t)((int64_t)r->pc+(int32_t)arg),r->next_pc);break;
    case OP_CALL:rg_write(b,"nrg_call(p,%u,%u);return;\n",r->callee,r->next_pc);break;
    case OP_RET:rg_write(b,"nrg_return(p);return;\n");break;
    default:b->status=NVM_ARRAY_UNRESOLVED;return false;
    }
    if(r->opcode!=OP_JMP && r->opcode!=OP_JMP_TRUE && r->opcode!=OP_JMP_FALSE && r->opcode!=OP_CALL && r->opcode!=OP_RET)
        rg_write(b,"goto b%u;\n",r->next_pc);
    return rg_write(b,"}\n");
}
static bool rg_fact(RgOutput *b,const NvmRecordArrayExecutionPlan *plan,
    const NvmRecordArrayExecutionFunction *functions,uint32_t count,
    const NvmRecordArrayExecutionInstruction *r) {
    NvmRecordArrayExecutionInstruction expected={0};
    if(!rg_recipe(r->opcode,&expected) || r->recipe!=expected.recipe || r->obligations!=expected.obligations ||
       r->function>=count || r->next_pc<=r->pc || r->next_pc-r->pc>64 ||
       r->next_pc>functions[r->function].signature.code_length)return false;
    uint8_t bytes[64];DecodedInstruction decoded;
    uint32_t width=r->next_pc-r->pc;
    if(!rg_charge(b,0,256) || !nvm_record_array_execution_bytes(plan,NVM_RA_SNAPSHOT_CODE,0,
        functions[r->function].signature.code_offset+r->pc,width,bytes) ||
       isa_decode(bytes,width,&decoded)!=width || decoded.opcode!=r->opcode || decoded.operand_count!=r->operand_count)return false;
    for(uint8_t i=0;i<r->operand_count;i++) {
        uint64_t bits=0;
        switch(decoded.operand_types[i]) {
        case OPERAND_U8:bits=decoded.operands[i].u8;break;
        case OPERAND_U16:bits=decoded.operands[i].u16;break;
        case OPERAND_U32:bits=decoded.operands[i].u32;break;
        case OPERAND_I32:bits=(uint32_t)decoded.operands[i].i32;break;
        case OPERAND_I64:bits=(uint64_t)decoded.operands[i].i64;break;
        case OPERAND_F64:memcpy(&bits,&decoded.operands[i].f64,8);break;
        default:return false;
        }
        if(r->operand_types[i]!=(uint8_t)decoded.operand_types[i] || r->operand_bits[i]!=bits)return false;
    }
    const InstructionInfo *info=isa_get_info(r->opcode);
    int pops=info->pop_count,pushes=info->push_count;uint32_t callee=UINT32_MAX;
    if(r->opcode==OP_CALL) {
        callee=(uint32_t)r->operand_bits[0];if(callee>=count)return false;
        pops=functions[callee].signature.arity;pushes=functions[callee].signature.result_count;
    } else if(r->opcode==OP_RET) { pops=functions[r->function].signature.result_count;pushes=0; }
    else if(r->opcode==OP_ARR_LITERAL || r->opcode==OP_STRUCT_LITERAL) { pops=(int)r->operand_bits[1];pushes=1; }
    else if(r->opcode==OP_AGG_PACK) { pops=(int)r->operand_bits[3];pushes=1; }
    if(pops!=r->pops || pushes!=r->pushes || callee!=r->callee)return false;
    uint32_t successors[2];uint8_t edges=0;
    if(r->opcode==OP_JMP || r->opcode==OP_JMP_TRUE || r->opcode==OP_JMP_FALSE)
        successors[edges++]=(uint32_t)((int64_t)r->pc+(int32_t)r->operand_bits[0]);
    if(r->opcode!=OP_JMP && r->opcode!=OP_RET)successors[edges++]=r->next_pc;
    if(edges!=r->successor_count)return false;
    for(uint8_t i=0;i<edges;i++)if(successors[i]!=r->successors[i])return false;
    return true;
}
NvmArrayEligibilityResult nvm2c_record_array_private(const NvmModule *module,
    char **out,size_t *length,NvmRecordArrayGeneratedCost *cost) {
    NvmArrayEligibilityResult result={.status=NVM_ARRAY_INVALID};
    snprintf(result.message,sizeof result.message,"I require a complete private generated program and disjoint outputs.");
    if(!module || !out || !length || !cost)return result;
    NvmRecordArrayExecutionPlan *plan=NULL;
    result=nvm_prepare_record_array_execution(module,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    RgOutput b={.status=NVM_ARRAY_ELIGIBLE};
    NvmRecordArrayExecutionCounts counts;
    NvmDeclarationCounts declarations;
    NvmHeader header;
    NvmRecordArrayExecutionFunction functions[256];
    NvmRecordArrayExecutionDescriptor records[256];
    NvmDeclarationLayout layouts[256];
    uint32_t starts[256],record_starts[256],fields_count=0,record_fields=0;
    NrgField *fields=NULL;
    NvmRecordArrayExecutionInstruction *instructions=NULL;
    uint64_t fixed=sizeof b+sizeof counts+sizeof declarations+sizeof header+sizeof functions+sizeof records+
        sizeof layouts+sizeof starts+sizeof record_starts;
    if(!rg_charge(&b,fixed,fixed) || !nvm_record_array_execution_counts(plan,&counts) ||
       !nvm_record_array_execution_declaration_counts(plan,&declarations) ||
       !nvm_record_array_execution_header(plan,&header) ||
       !counts.functions || counts.functions>256 || counts.entry>=counts.functions ||
       (counts.initializer!=UINT32_MAX && counts.initializer>=counts.functions) || counts.globals>256 ||
       counts.records>256 || declarations.layouts>256 || counts.instructions>65536)goto invalid;
    uint32_t supported=0;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction recipe={0};
        bool active=rg_recipe((uint8_t)op,&recipe)!=0;
        if(active!=nvm_record_array_opcode_supported((uint8_t)op))goto invalid;
        supported+=active;
    }
    if(supported!=93)goto invalid;
    for(uint32_t i=0;i<counts.functions;i++)if(!nvm_record_array_execution_function(plan,i,&functions[i]))goto invalid;
    for(uint32_t i=0;i<declarations.layouts;i++) {
        if(!nvm_record_array_execution_layout(plan,i,&layouts[i]) || layouts[i].fields>65536-fields_count)goto invalid;
        starts[i]=fields_count;fields_count+=layouts[i].fields;
    }
    if(!rg_charge(&b,(uint64_t)(fields_count?fields_count:1)*sizeof *fields,(uint64_t)(fields_count?fields_count:1)*sizeof *fields+
        (uint64_t)declarations.bindings*32))goto invalid;
    fields=calloc(fields_count?fields_count:1,sizeof *fields);
    if(!fields) { b.status=NVM_ARRAY_MEMORY;goto invalid; }
    for(uint32_t i=0;i<declarations.layouts;i++)for(uint16_t j=0;j<layouts[i].fields;j++) {
        NvmV2LayoutField f;if(!nvm_record_array_execution_field(plan,i,j,&f))goto invalid;
        fields[starts[i]+j]=(NrgField){f.type_tag,f.nested_idx,0};
    }
    for(uint32_t i=0;i<declarations.bindings;i++) {
        NvmOrdinaryArrayBinding binding;NvmOrdinaryArrayType element;
        if(!nvm_record_array_execution_binding(plan,i,&binding) || binding.layout>=declarations.layouts ||
           binding.field>=layouts[binding.layout].fields ||
           !nvm_record_array_execution_type(plan,binding.element_type,&element))goto invalid;
        fields[starts[binding.layout]+binding.field].element=element.tag;
    }
    for(uint32_t i=0;i<counts.records;i++) {
        if(!nvm_record_array_execution_descriptor(plan,i,&records[i]) || records[i].ordinal!=i ||
           records[i].layout>=declarations.layouts || records[i].fields!=layouts[records[i].layout].fields)goto invalid;
        record_starts[i]=record_fields;record_fields+=records[i].fields;
    }
    uint64_t instruction_bytes=(uint64_t)(counts.instructions?counts.instructions:1)*sizeof *instructions;
    if(!rg_charge(&b,instruction_bytes,instruction_bytes))goto invalid;
    instructions=calloc(counts.instructions?counts.instructions:1,sizeof *instructions);
    if(!instructions) { b.status=NVM_ARRAY_MEMORY;goto invalid; }
    uint32_t total=0;
    for(uint32_t fi=0;fi<counts.functions;fi++) {
        NvmRecordArrayExecutionFunction *fn=&functions[fi];uint32_t pc=0;
        if(fn->instruction_start!=total || fn->instruction_count>counts.instructions-total)goto invalid;
        for(uint32_t j=0;j<fn->instruction_count;j++) {
            NvmRecordArrayExecutionInstruction *r=&instructions[total+j];
            if(!nvm_record_array_execution_instruction(plan,total+j,r) || r->function!=fi || r->pc!=pc ||
               !rg_fact(&b,plan,functions,counts.functions,r))goto invalid;
            pc=r->next_pc;
        }
        if(pc!=fn->signature.code_length)goto invalid;
        total+=fn->instruction_count;
    }
    if(total!=counts.instructions)goto invalid;
    for(uint32_t i=0;i<counts.instructions;i++) {
        const NvmRecordArrayExecutionInstruction *r=&instructions[i];
        const NvmRecordArrayExecutionFunction *f=&functions[r->function];
        for(uint8_t edge=0;edge<r->successor_count;edge++) {
            uint32_t target=r->successors[edge];
            if(target==f->signature.code_length)continue;
            uint32_t lo=f->instruction_start,hi=lo+f->instruction_count;
            while(lo<hi) {
                if(!rg_charge(&b,0,1))goto invalid;
                uint32_t middle=lo+(hi-lo)/2;
                if(instructions[middle].pc<target)lo=middle+1;else hi=middle;
            }
            if(lo==f->instruction_start+f->instruction_count || instructions[lo].pc!=target)goto invalid;
        }
    }

    rg_write(&b,"#define NANO_RECORD_ARRAY_GENERATED_PRIVATE 1\n#include \"record_array_generated_private.h\"\n#include <stdint.h>\n%s\n",nl_binary64_arithmetic_source);
    rg_write(&b,"static inline double rg_float(uint64_t bits){double x;volatile unsigned char *d=(volatile unsigned char *)&x;const volatile unsigned char *s=(const volatile unsigned char *)&bits;for(unsigned i=0;i<8;i++)d[i]=s[i];return x;}\n"
        "static inline uint64_t rg_bits(double x){uint64_t bits;volatile unsigned char *d=(volatile unsigned char *)&bits;const volatile unsigned char *s=(const volatile unsigned char *)&x;for(unsigned i=0;i<8;i++)d[i]=s[i];return bits;}\n");
    for(uint32_t fi=0;fi<counts.functions;fi++)rg_write(&b,"static void body_%u(NrgInstance *);\n",fi);
    for(uint32_t i=0;i<counts.strings;i++) {
        uint32_t size;if(!nvm_record_array_execution_size(plan,NVM_RA_SNAPSHOT_STRING,i,&size))goto invalid;
        rg_write(&b,"static const unsigned char literal_%u[%u]={",i,size?size:1);
        for(uint32_t j=0;j<size;j++) {
            uint8_t byte;if(!rg_charge(&b,0,1) || !nvm_record_array_execution_bytes(plan,NVM_RA_SNAPSHOT_STRING,i,j,1,&byte))goto invalid;
            rg_write(&b,"%u,",byte);
        }
        if(!size)rg_write(&b,"0");
        rg_write(&b,"};\n");
    }
    rg_write(&b,"static const NmsView literals[%u]={",counts.strings?counts.strings:1);
    for(uint32_t i=0;i<counts.strings;i++) {
        uint32_t size;if(!nvm_record_array_execution_size(plan,NVM_RA_SNAPSHOT_STRING,i,&size))goto invalid;
        rg_write(&b,"{literal_%u,%u},",i,size);
    }
    if(!counts.strings)rg_write(&b,"{0,0}");
    rg_write(&b,"};\n");
    rg_write(&b,"static const NmsRecordDescriptor records[%u]={",counts.records?counts.records:1);
    for(uint32_t i=0;i<counts.records;i++)rg_write(&b,"{%u,%u},",records[i].layout,records[i].fields);
    if(!counts.records)rg_write(&b,"{0,0}");
    rg_write(&b,"};\n");
    rg_write(&b,"static const uint32_t starts[%u]={",counts.records?counts.records:1);
    for(uint32_t i=0;i<counts.records;i++)rg_write(&b,"%u,",record_starts[i]);
    if(!counts.records)rg_write(&b,"0");
    rg_write(&b,"};\n");
    rg_write(&b,"static const NrgField fields[%u]={",record_fields?record_fields:1);
    for(uint32_t i=0;i<counts.records;i++)for(uint32_t j=0;j<records[i].fields;j++) {
        NrgField f=fields[starts[records[i].layout]+j];rg_write(&b,"{%u,%u,%u},",f.tag,f.nested_layout,f.element);
    }
    if(!record_fields)rg_write(&b,"{0,0,0}");
    rg_write(&b,"};\n");
    for(uint32_t i=0;i<counts.functions;i++)if(functions[i].parameter_tags_present) {
        uint32_t arity=functions[i].signature.arity;
        rg_write(&b,"static const uint8_t parameters_%u[%u]={",i,arity?arity:1);
        for(uint32_t j=0;j<arity;j++) {
            uint8_t tag;if(!nvm_record_array_execution_parameter(plan,i,(uint16_t)j,&tag))goto invalid;
            rg_write(&b,"%u,",tag);
        }
        if(!arity)rg_write(&b,"0");
        rg_write(&b,"};\n");
    }
    rg_write(&b,"static const NrgFunction functions[%u]={",counts.functions);
    for(uint32_t i=0;i<counts.functions;i++) {
        NvmRecordArrayExecutionFunction *f=&functions[i];
        rg_write(&b,"{%u,%u,%u,%u,%u,",f->signature.local_count,f->signature.arity,f->signature.result_count,f->signature.result_tag,f->maximum_stack);
        if(f->parameter_tags_present)rg_write(&b,"parameters_%u",i);else rg_write(&b,"0");rg_write(&b,",body_%u},",i);
    }
    rg_write(&b,"};\nstatic const NrgProgram program={NRG_ABI,sizeof(NmsValue),offsetof(NmsValue,tag),NRG_FRAMES,%u,%u,%u,%u,%u,%u,%u,%u,functions,literals,records,starts,fields};\n",
        counts.functions,counts.entry,counts.initializer,counts.globals,(unsigned)!!(header.flags&NVM_FLAG_HAS_MAIN),counts.strings,counts.records,record_fields);
    for(uint32_t fi=0;fi<counts.functions;fi++) {
        NvmRecordArrayExecutionFunction *fn=&functions[fi];
        rg_write(&b,"static void body_%u(NrgInstance *p){switch(nrg_resume(p)){",fi);
        for(uint32_t j=0;j<fn->instruction_count;j++) {
            uint32_t pc=instructions[fn->instruction_start+j].pc;rg_write(&b,"case %u:goto b%u;",pc,pc);
        }
        rg_write(&b,"case %u:goto b%u;default:nrg_fail(p,NRG_STATE);return;}\n",fn->signature.code_length,fn->signature.code_length);
        for(uint32_t j=0;j<fn->instruction_count;j++)if(!rg_operation(&b,&instructions[fn->instruction_start+j]))goto invalid;
        rg_write(&b,"b%u:;nrg_return(p);return;}\n",fn->signature.code_length);
    }
    rg_write(&b,"NrgStatus nrg_generated_create(NrgInstance **out){\n"
        "(void)&nano_rt_f64_add;(void)&nano_rt_f64_sub;(void)&nano_rt_f64_mul;(void)&nano_rt_f64_div;(void)&rg_float;(void)&rg_bits;\n"
        "if(program.abi!=1||program.value_size!=sizeof(NmsValue)||program.value_tag_offset!=offsetof(NmsValue,tag)||program.frame_limit!=1024||"
        "program.function_count!=%u||program.entry!=%u||program.initializer!=%u||program.global_count!=%u||program.has_main!=%u||program.literal_count!=%u||program.record_count!=%u||program.field_count!=%u||"
        "program.functions!=functions||program.literals!=literals||program.records!=records||program.field_starts!=starts||program.fields!=fields)return NRG_STATE;\n",
        counts.functions,counts.entry,counts.initializer,counts.globals,(unsigned)!!(header.flags&NVM_FLAG_HAS_MAIN),counts.strings,counts.records,record_fields);
    for(uint32_t i=0;i<counts.functions;i++) {
        NvmRecordArrayExecutionFunction *f=&functions[i];
        rg_write(&b,"if(functions[%u].locals!=%u||functions[%u].arity!=%u||functions[%u].result_count!=%u||functions[%u].result_tag!=%u||functions[%u].maximum_stack!=%u||functions[%u].body!=body_%u)return NRG_STATE;\n",
            i,f->signature.local_count,i,f->signature.arity,i,f->signature.result_count,i,f->signature.result_tag,i,f->maximum_stack,i,i);
        if(f->parameter_tags_present) {
            rg_write(&b,"if(functions[%u].parameters!=parameters_%u)return NRG_STATE;\n",i,i);
            for(uint32_t j=0;j<f->signature.arity;j++) {
                uint8_t tag;if(!nvm_record_array_execution_parameter(plan,i,(uint16_t)j,&tag))goto invalid;
                rg_write(&b,"if(parameters_%u[%u]!=%u)return NRG_STATE;\n",i,j,tag);
            }
        } else rg_write(&b,"if(functions[%u].parameters)return NRG_STATE;\n",i);
    }
    for(uint32_t i=0;i<counts.strings;i++) {
        uint32_t size;if(!nvm_record_array_execution_size(plan,NVM_RA_SNAPSHOT_STRING,i,&size))goto invalid;
        rg_write(&b,"if(literals[%u].data!=literal_%u||literals[%u].length!=%u)return NRG_STATE;\n",i,i,i,size);
    }
    for(uint32_t i=0;i<counts.records;i++) {
        rg_write(&b,"if(records[%u].global_layout_index!=%u||records[%u].field_count!=%u||starts[%u]!=%u)return NRG_STATE;\n",
            i,records[i].layout,i,records[i].fields,i,record_starts[i]);
        for(uint32_t j=0;j<records[i].fields;j++) {
            uint32_t k=record_starts[i]+j;NrgField f=fields[starts[records[i].layout]+j];
            rg_write(&b,"if(fields[%u].tag!=%u||fields[%u].nested_layout!=%u||fields[%u].element!=%u)return NRG_STATE;\n",k,f.tag,k,f.nested_layout,k,f.element);
        }
    }
    rg_write(&b,"return nrg_create(&program,out); }\n");
    if(b.status!=NVM_ARRAY_ELIGIBLE)goto invalid;
    *cost=(NvmRecordArrayGeneratedCost){counts.peak_bytes_reserved,counts.work_reserved,b.peak,b.work};
    *out=b.text;*length=b.used;b.text=NULL;
    free(fields);free(instructions);nvm_record_array_execution_free(plan);return result;
invalid:
    if(b.status==NVM_ARRAY_ELIGIBLE)b.status=NVM_ARRAY_INVALID;
    result.status=b.status;snprintf(result.message,sizeof result.message,"I could not complete private generated correspondence and emission.");
    free(b.text);free(fields);free(instructions);nvm_record_array_execution_free(plan);return result;
}
#endif
