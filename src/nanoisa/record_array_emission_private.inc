/* I share exact checked output accounting and opcode facts across emitters. */
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
