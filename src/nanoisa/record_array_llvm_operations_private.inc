/* I emit direct operations against borrowed value scratch and owned runtime roots.
 * This file is private to the LLVM text writer; it contains no bytecode dispatch
 * in a generated product. Every failure branch preserves operand roots. */
typedef struct { RgOutput *out; uint32_t serial; } RlBlock;
static uint32_t rl_next(RlBlock *s) {
    if(s->serial==UINT32_MAX) { s->out->status=NVM_ARRAY_LIMIT;return 0; }
    return s->serial++;
}
static void rl_guard(RlBlock *s,uint32_t condition,const char *failure) {
    uint32_t label=rl_next(s);
    rg_write(s->out," br i1 %%v%u, label %%l%u, label %%%s\nl%u:\n",condition,label,failure,label);
}
static void rl_check_status(RlBlock *s) {
    uint32_t status=rl_next(s),ok=rl_next(s);
    rg_write(s->out," %%v%u = call i32 @nrg_status(ptr %%p)\n %%v%u = icmp eq i32 %%v%u, 0\n",status,ok,status);
    rl_guard(s,ok,"error");
}
static void rl_call(RlBlock *s,const char *name,const char *arguments) {
    uint32_t value=rl_next(s);
    rg_write(s->out," %%v%u = call zeroext i1 @%s(ptr %%p%s)\n",value,name,arguments);
    rl_guard(s,value,"error");
}
static void rl_peek(RlBlock *s,uint32_t distance,char scratch,uint32_t *payload,uint32_t *tag) {
    uint32_t ok=rl_next(s);*payload=rl_next(s);*tag=rl_next(s);
    rg_write(s->out," %%v%u = call zeroext i1 @nrg_peek(ptr %%p, i32 %u, ptr %%%c)\n",ok,distance,scratch);
    rl_guard(s,ok,"error");
    rg_write(s->out," %%v%u = load i64, ptr %%%c_payload\n %%v%u = load i32, ptr %%%c_tag\n",*payload,scratch,*tag,scratch);
}
static void rl_tag(RlBlock *s,uint32_t actual,uint32_t expected) {
    uint32_t ok=rl_next(s);
    rg_write(s->out," %%v%u = icmp eq i32 %%v%u, %u\n",ok,actual,expected);rl_guard(s,ok,"type_error");
}
static void rl_replace(RlBlock *s,uint32_t count,uint32_t payload,uint32_t tag) {
    rg_write(s->out," store i64 %%v%u, ptr %%r_payload\n store i32 %u, ptr %%r_tag\n",payload,tag);
    char args[64];snprintf(args,sizeof args,", i32 %u, ptr %%r",count);rl_call(s,"nrg_replace",args);
}
static void rl_boolean(RlBlock *s,uint32_t count,uint32_t condition) {
    uint32_t payload=rl_next(s);rg_write(s->out," %%v%u = zext i1 %%v%u to i64\n",payload,condition);
    rl_replace(s,count,payload,4);
}
static uint32_t rl_truth(RlBlock *s,char scratch) {
    uint32_t result=rl_next(s);rg_write(s->out," %%v%u = call zeroext i1 @nrg_truth(ptr %%p, ptr %%%c)\n",result,scratch);
    return result;
}
static void rl_operation(RlBlock *s,const NvmRecordArrayExecutionInstruction *r) {
    RgOutput *b=s->out;uint32_t x=0,y=0,xt=0,yt=0,z=0,ok=0;uint64_t arg=r->operand_bits[0];
    char args[128];rg_write(b,"b%u:\n",r->pc);
    switch(r->opcode) {
    case OP_NOP:break;
    case OP_PUSH_I64:case OP_PUSH_U8:case OP_PUSH_F64:case OP_PUSH_BOOL:case OP_PUSH_VOID:case OP_PUSH_STR: {
        uint32_t tag=r->opcode==OP_PUSH_I64?1:r->opcode==OP_PUSH_U8?2:r->opcode==OP_PUSH_F64?3:r->opcode==OP_PUSH_BOOL?4:r->opcode==OP_PUSH_STR?5:0;
        uint64_t bits=r->opcode==OP_PUSH_VOID?0:r->opcode==OP_PUSH_BOOL?(arg!=0):r->opcode==OP_PUSH_STR?arg+1:arg;
        rg_write(b," store i64 %" PRIu64 ", ptr %%r_payload\n store i32 %u, ptr %%r_tag\n",bits,tag);
        rl_call(s,"nrg_push_move",", ptr %r");break;
    }
    case OP_I64_ADD:case OP_I64_SUB:case OP_I64_MUL:
        rl_peek(s,1,'a',&x,&xt);rl_peek(s,0,'b',&y,&yt);rl_tag(s,xt,1);rl_tag(s,yt,1);z=rl_next(s);
        rg_write(b," %%v%u = %s i64 %%v%u, %%v%u\n",z,r->opcode==OP_I64_ADD?"add":r->opcode==OP_I64_SUB?"sub":"mul",x,y);rl_replace(s,2,z,1);break;
    case OP_I64_DIV_S:case OP_I64_REM_S: {
        rl_peek(s,1,'a',&x,&xt);rl_peek(s,0,'b',&y,&yt);rl_tag(s,xt,1);rl_tag(s,yt,1);
        uint32_t zero=rl_next(s),minimum=rl_next(s),minus=rl_next(s),overflow=rl_next(s),exception=rl_next(s),safe=rl_next(s),quotient=rl_next(s),special=rl_next(s);z=rl_next(s);
        rg_write(b," %%v%u = icmp eq i64 %%v%u, 0\n %%v%u = icmp eq i64 %%v%u, 9223372036854775808\n %%v%u = icmp eq i64 %%v%u, -1\n %%v%u = and i1 %%v%u, %%v%u\n %%v%u = or i1 %%v%u, %%v%u\n %%v%u = select i1 %%v%u, i64 1, i64 %%v%u\n %%v%u = %s i64 %%v%u, %%v%u\n",zero,y,minimum,x,minus,y,overflow,minimum,minus,exception,zero,overflow,safe,exception,y,quotient,r->opcode==OP_I64_DIV_S?"sdiv":"srem",x,safe);
        rg_write(b," %%v%u = select i1 %%v%u, i64 0, i64 %s\n",special,zero,r->opcode==OP_I64_DIV_S?"9223372036854775808":"0");
        rg_write(b," %%v%u = select i1 %%v%u, i64 %%v%u, i64 %%v%u\n",z,exception,special,quotient);rl_replace(s,2,z,1);break;
    }
    case OP_I64_NEG:case OP_F64_FROM_BITS:case OP_F64_TO_BITS:
        rl_peek(s,0,'a',&x,&xt);rl_tag(s,xt,r->opcode==OP_F64_TO_BITS?3:1);z=x;
        if(r->opcode==OP_I64_NEG) { z=rl_next(s);rg_write(b," %%v%u = sub i64 0, %%v%u\n",z,x); }
        rl_replace(s,1,z,r->opcode==OP_F64_FROM_BITS?3:1);break;
    case OP_F64_ADD:case OP_F64_SUB:case OP_F64_MUL:case OP_F64_DIV:case OP_F64_NEG: {
        rl_peek(s,r->opcode==OP_F64_NEG?0:1,'a',&x,&xt);rl_tag(s,xt,3);
        uint32_t left=rl_next(s);rg_write(b," %%v%u = bitcast i64 %%v%u to double\n",left,x);
        if(r->opcode!=OP_F64_NEG) { rl_peek(s,0,'b',&y,&yt);rl_tag(s,yt,3); }
        uint32_t value=rl_next(s),divide_label=0,zero_label=0,join=0;
        if(r->opcode==OP_F64_DIV) {
            uint32_t absolute=rl_next(s),zero=rl_next(s);divide_label=rl_next(s);zero_label=rl_next(s);join=rl_next(s);
            rg_write(b," %%v%u = and i64 %%v%u, 9223372036854775807\n %%v%u = icmp eq i64 %%v%u, 0\n br i1 %%v%u, label %%l%u, label %%l%u\nl%u:\n br label %%l%u\nl%u:\n",absolute,y,zero,absolute,zero,zero_label,divide_label,zero_label,join,divide_label);
        }
        if(r->opcode==OP_F64_NEG)rg_write(b," %%v%u = fneg double %%v%u\n",value,left);
        else {
            uint32_t right=rl_next(s);rg_write(b," %%v%u = bitcast i64 %%v%u to double\n",right,y);
            rg_write(b," %%v%u = %s double %%v%u, %%v%u\n",value,r->opcode==OP_F64_ADD?"fadd":r->opcode==OP_F64_SUB?"fsub":r->opcode==OP_F64_MUL?"fmul":"fdiv",left,right);
        }
        z=rl_next(s);rg_write(b," %%v%u = bitcast double %%v%u to i64\n",z,value);
        if(r->opcode!=OP_F64_NEG) {
            uint32_t nan=rl_next(s),canon=rl_next(s);rg_write(b," %%v%u = fcmp uno double %%v%u, %%v%u\n %%v%u = select i1 %%v%u, i64 9221120237041090560, i64 %%v%u\n",nan,value,value,canon,nan,z);z=canon;
            if(r->opcode==OP_F64_DIV) {
                uint32_t answer=rl_next(s);rg_write(b," br label %%l%u\nl%u:\n %%v%u = phi i64 [ 0, %%l%u ], [ %%v%u, %%l%u ]\n",join,join,answer,zero_label,z,divide_label);z=answer;
            }
        }
        rl_replace(s,r->opcode==OP_F64_NEG?1:2,z,3);break;
    }
    case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE: {
        bool floating=r->opcode==OP_F64_EQ||r->opcode==OP_F64_NE||r->opcode==OP_F64_LT||r->opcode==OP_F64_LE||r->opcode==OP_F64_GT||r->opcode==OP_F64_GE;
        rl_peek(s,1,'a',&x,&xt);rl_peek(s,0,'b',&y,&yt);rl_tag(s,xt,floating?3:1);rl_tag(s,yt,floating?3:1);
        if(floating) { uint32_t left=rl_next(s),right=rl_next(s);rg_write(b," %%v%u = bitcast i64 %%v%u to double\n %%v%u = bitcast i64 %%v%u to double\n",left,x,right,y);x=left;y=right; }
        const char *predicate=r->opcode==OP_I64_EQ?"eq":r->opcode==OP_I64_NE?"ne":r->opcode==OP_I64_LT_S?"slt":r->opcode==OP_I64_LE_S?"sle":r->opcode==OP_I64_GT_S?"sgt":r->opcode==OP_I64_GE_S?"sge":r->opcode==OP_F64_EQ?"oeq":r->opcode==OP_F64_NE?"une":r->opcode==OP_F64_LT?"olt":r->opcode==OP_F64_LE?"ole":r->opcode==OP_F64_GT?"ogt":"oge";
        z=rl_next(s);rg_write(b," %%v%u = %s %s %s %%v%u, %%v%u\n",z,floating?"fcmp":"icmp",predicate,floating?"double":"i64",x,y);rl_boolean(s,2,z);break;
    }
    case OP_EQ:case OP_NE:case OP_STR_EQ:
        rl_peek(s,1,'a',&x,&xt);rl_peek(s,0,'b',&y,&yt);
        if(r->opcode==OP_STR_EQ) { rl_tag(s,xt,5);rl_tag(s,yt,5); }
        z=rl_next(s);rg_write(b," %%v%u = call zeroext i1 @nrg_equal(ptr %%p, ptr %%a, ptr %%b)\n",z);rl_check_status(s);
        if(r->opcode==OP_NE) { ok=rl_next(s);rg_write(b," %%v%u = xor i1 %%v%u, true\n",ok,z);z=ok; }
        rl_boolean(s,2,z);break;
    case OP_LT:case OP_LE:case OP_GT:case OP_GE:
        rl_peek(s,1,'a',&x,&xt);rl_peek(s,0,'b',&y,&yt);x=rl_next(s);z=rl_next(s);
        rg_write(b," %%v%u = call i32 @nrg_order(ptr %%p, ptr %%a, ptr %%b)\n",x);rl_check_status(s);
        rg_write(b," %%v%u = icmp %s i32 %%v%u, 0\n",z,r->opcode==OP_LT?"slt":r->opcode==OP_LE?"sle":r->opcode==OP_GT?"sgt":"sge",x);rl_boolean(s,2,z);break;
    case OP_BOOL_AND:case OP_BOOL_OR:
        rl_peek(s,1,'a',&x,&xt);rl_peek(s,0,'b',&y,&yt);rl_tag(s,xt,4);rl_tag(s,yt,4);
        xt=rl_next(s);yt=rl_next(s);z=rl_next(s);
        rg_write(b," %%v%u = icmp ne i64 %%v%u, 0\n %%v%u = icmp ne i64 %%v%u, 0\n %%v%u = %s i1 %%v%u, %%v%u\n",xt,x,yt,y,z,r->opcode==OP_BOOL_AND?"and":"or",xt,yt);rl_boolean(s,2,z);break;
    case OP_AND:case OP_OR: {
        rl_peek(s,1,'a',&x,&xt);rl_peek(s,0,'b',&y,&yt);x=rl_truth(s,'a');
        uint32_t short_label=rl_next(s),right_label=rl_next(s),join=rl_next(s);
        rg_write(b," br i1 %%v%u, label %%l%u, label %%l%u\nl%u:\n br label %%l%u\nl%u:\n",x,r->opcode==OP_AND?right_label:short_label,r->opcode==OP_AND?short_label:right_label,short_label,join,right_label);
        y=rl_truth(s,'b');rg_write(b," br label %%l%u\nl%u:\n",join,join);z=rl_next(s);
        rg_write(b," %%v%u = phi i1 [ %s, %%l%u ], [ %%v%u, %%l%u ]\n",z,r->opcode==OP_AND?"false":"true",short_label,y,right_label);
        rl_check_status(s);rl_boolean(s,2,z);break;
    }
    case OP_BOOL_NOT:case OP_NOT:case OP_CAST_BOOL:
        rl_peek(s,0,'a',&x,&xt);
        if(r->opcode==OP_BOOL_NOT) { rl_tag(s,xt,4);z=rl_next(s);rg_write(b," %%v%u = icmp eq i64 %%v%u, 0\n",z,x); }
        else { z=rl_truth(s,'a');if(r->opcode==OP_NOT) { ok=rl_next(s);rg_write(b," %%v%u = xor i1 %%v%u, true\n",ok,z);z=ok; } }
        rl_check_status(s);rl_boolean(s,1,z);break;
    case OP_CAST_U8:
        rl_peek(s,0,'a',&x,&xt);y=rl_next(s);yt=rl_next(s);ok=rl_next(s);z=rl_next(s);
        rg_write(b," %%v%u = icmp eq i32 %%v%u, 1\n %%v%u = icmp eq i32 %%v%u, 2\n %%v%u = or i1 %%v%u, %%v%u\n",y,xt,yt,xt,ok,y,yt);rl_guard(s,ok,"type_error");
        rg_write(b," %%v%u = and i64 %%v%u, 255\n",z,x);rl_replace(s,1,z,2);break;
    case OP_TYPE_CHECK:
        rl_peek(s,0,'a',&x,&xt);z=rl_next(s);rg_write(b," %%v%u = icmp eq i32 %%v%u, %u\n",z,xt,(uint32_t)arg);rl_boolean(s,1,z);break;
    case OP_CAST_INT:rl_call(s,"nrg_cast_int","");break;
    case OP_CAST_FLOAT:rl_call(s,"nrg_cast_float","");break;
    case OP_CAST_STRING:rl_call(s,"nrg_cast_string","");break;
    case OP_STR_FROM_INT:rl_call(s,"nrg_format",", i32 1");break;
    case OP_STR_FROM_FLOAT:rl_call(s,"nrg_format",", i32 3");break;
    case OP_STR_CONCAT:rl_call(s,"nrg_concat","");break;
    case OP_STR_SUBSTR:rl_call(s,"nrg_substring","");break;
    case OP_STR_TRIM:rl_call(s,"nrg_trim","");break;
    case OP_STR_TO_LOWER:rl_call(s,"nrg_case",", i1 zeroext false");break;
    case OP_STR_TO_UPPER:rl_call(s,"nrg_case",", i1 zeroext true");break;
    case OP_STR_SPLIT:rl_call(s,"nrg_split","");break;
    case OP_STR_REPLACE:rl_call(s,"nrg_string_replace","");break;
    case OP_STR_LEN:rl_call(s,"nrg_length",", i1 zeroext false");break;
    case OP_ARR_LEN:rl_call(s,"nrg_length",", i1 zeroext true");break;
    case OP_STR_CHAR_AT:rl_call(s,"nrg_character","");break;
    case OP_STR_CONTAINS:rl_call(s,"nrg_predicate",", i32 0");break;
    case OP_STR_STARTS_WITH:rl_call(s,"nrg_predicate",", i32 1");break;
    case OP_STR_ENDS_WITH:rl_call(s,"nrg_predicate",", i32 2");break;
    case OP_ARR_GET:rl_call(s,"nrg_array_get","");break;
    case OP_ARR_SET:rl_call(s,"nrg_array_set","");break;
    case OP_ARR_PUSH:rl_call(s,"nrg_array_push","");break;
    case OP_ARR_POP:rl_call(s,"nrg_array_pop","");break;
    case OP_ARR_SLICE:rl_call(s,"nrg_array_slice","");break;
    case OP_POP:rl_call(s,"nrg_drop","");break;
    case OP_DUP:rl_call(s,"nrg_dup","");break;
    case OP_SWAP:rl_call(s,"nrg_swap","");break;
    case OP_LOAD_LOCAL:case OP_LOAD_GLOBAL:case OP_STORE_LOCAL:case OP_STORE_GLOBAL:
        snprintf(args,sizeof args,", i1 zeroext %s, i32 %u",r->opcode==OP_LOAD_GLOBAL||r->opcode==OP_STORE_GLOBAL?"true":"false",(uint32_t)arg);
        rl_call(s,r->recipe==NVM_RA_ROOT_LOAD?"nrg_load":"nrg_store",args);break;
    case OP_STRUCT_NEW:case OP_STRUCT_LITERAL:case OP_AGG_PACK:
        snprintf(args,sizeof args,", i32 %u",(uint32_t)(r->opcode==OP_AGG_PACK?r->operand_bits[1]:arg));rl_call(s,"nrg_record_new",args);break;
    case OP_STRUCT_GET:case OP_AGG_GET:case OP_STRUCT_SET:case OP_AGG_SET:
        snprintf(args,sizeof args,", i32 %u, i1 zeroext %s",(uint32_t)arg,r->opcode==OP_AGG_GET||r->opcode==OP_AGG_SET?"true":"false");rl_call(s,r->recipe==NVM_RA_ROOT_GET?"nrg_record_get":"nrg_record_set",args);break;
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        snprintf(args,sizeof args,", i32 %u, i32 %u, i1 zeroext %s",(uint32_t)arg,r->opcode==OP_ARR_LITERAL?(uint32_t)r->operand_bits[1]:0,r->opcode==OP_ARR_LITERAL?"true":"false");rl_call(s,"nrg_array_new",args);break;
    case OP_ASSERT:
        rl_peek(s,0,'a',&x,&xt);z=rl_truth(s,'a');rl_call(s,"nrg_drop","");rl_guard(s,z,"assert_error");break;
    case OP_JMP:rg_write(b," br label %%b%u\n",r->successors[0]);return;
    case OP_JMP_TRUE:case OP_JMP_FALSE:
        rl_peek(s,0,'a',&x,&xt);z=rl_truth(s,'a');rl_call(s,"nrg_drop","");
        rg_write(b," br i1 %%v%u, label %%b%u, label %%b%u\n",z,r->opcode==OP_JMP_TRUE?r->successors[0]:r->next_pc,r->opcode==OP_JMP_TRUE?r->next_pc:r->successors[0]);return;
    case OP_CALL:rg_write(b," call void @nrg_call(ptr %%p, i32 %u, i32 %u)\n ret void\n",r->callee,r->next_pc);return;
    case OP_RET:rg_write(b," call void @nrg_return(ptr %%p)\n ret void\n");return;
    default:b->status=NVM_ARRAY_UNRESOLVED;return;
    }
    rg_write(b," br label %%b%u\n",r->next_pc);
}
