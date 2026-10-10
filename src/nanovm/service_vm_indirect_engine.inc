#ifndef SERVICE_EXECUTION_AUTH_ARGUMENT
#define SERVICE_EXECUTION_AUTH_ARGUMENT
#define SERVICE_EXECUTION_AUTH_CHECK(c) SERVICE_CONST(RUNTIME_OK)
#endif
#include "../nanoisa/service_indirect_dispatch.inc"
#include <limits.h>
#include <string.h>
static SERVICE_TYPE(RuntimeStatus) fvm_bad(SERVICE_TYPE(Runtime) *c) {
    return SERVICE_FN(runtime_fail)(c,SERVICE_CONST(RUNTIME_STATE));
}
static bool fvm_scalar(const SERVICE_TYPE(RuntimeView) *v,uint8_t tag) {
    return v->initialized && !v->owning && !v->formal && !v->type.mode &&
        v->type.category==SERVICE_CONST(CATEGORY_UNKNOWN) && v->type.tag==tag &&
        v->type.global_index==NVM_V2_NO_INDEX && v->type.catalog_ordinal==NVM_V2_NO_INDEX &&
        v->fields==(tag==TAG_VOID?0:1) && (tag!=TAG_BOOL || v->values[0]==0 || v->values[0]==1);
}
static int64_t fvm_bits(uint64_t bits) { int64_t result;memcpy(&result,&bits,sizeof result);return result; }
#define FVM_TRY(x) do { SERVICE_TYPE(RuntimeStatus) status_=(x); if(status_!=SERVICE_CONST(RUNTIME_OK))return status_; } while(0)
static SERVICE_TYPE(RuntimeStatus) fvm_numeric(SERVICE_TYPE(Runtime) *c,const SERVICE_TYPE(RuntimeFrameView) *f,uint8_t op) {
    bool unary=op==OP_NEG || op==OP_I64_NEG || op==OP_NOT;
    uint16_t count=unary?1:2;uint32_t roots[2];SERVICE_TYPE(RuntimeView) values[2];
    if(f->stack_count<count)return fvm_bad(c);
    for(uint16_t i=0;i<count;i++)
        if(!SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f->stack_count-count+i),&roots[i]) ||
           !SERVICE_FN(runtime_view)(c,roots[i],&values[i]))return fvm_bad(c);
    bool logic=op==OP_AND || op==OP_OR || op==OP_NOT;
    uint8_t input=logic?TAG_BOOL:TAG_INT;
    if((op==OP_EQ || op==OP_NE) && values[0].type.tag==TAG_BOOL)input=TAG_BOOL;
    for(uint16_t i=0;i<count;i++)if(!fvm_scalar(&values[i],input))
        return SERVICE_FN(runtime_fail)(c,SERVICE_CONST(RUNTIME_TYPE));
    int64_t a=values[0].values[0],b=unary?0:values[1].values[0],result=0;uint8_t tag=TAG_INT;
    switch(op) {
    case OP_ADD:case OP_I64_ADD:result=fvm_bits((uint64_t)a+(uint64_t)b);break;
    case OP_SUB:case OP_I64_SUB:result=fvm_bits((uint64_t)a-(uint64_t)b);break;
    case OP_MUL:case OP_I64_MUL:result=fvm_bits((uint64_t)a*(uint64_t)b);break;
    case OP_DIV:case OP_I64_DIV_S:result=!b?0:a==INT64_MIN&&b==-1?INT64_MIN:a/b;break;
    case OP_MOD:case OP_I64_REM_S:result=!b || (a==INT64_MIN&&b==-1)?0:a%b;break;
    case OP_NEG:case OP_I64_NEG:result=fvm_bits(UINT64_C(0)-(uint64_t)a);break;
    case OP_EQ:case OP_I64_EQ:result=a==b;tag=TAG_BOOL;break;
    case OP_NE:case OP_I64_NE:result=a!=b;tag=TAG_BOOL;break;
    case OP_LT:case OP_I64_LT_S:result=a<b;tag=TAG_BOOL;break;
    case OP_LE:case OP_I64_LE_S:result=a<=b;tag=TAG_BOOL;break;
    case OP_GT:case OP_I64_GT_S:result=a>b;tag=TAG_BOOL;break;
    case OP_GE:case OP_I64_GE_S:result=a>=b;tag=TAG_BOOL;break;
    case OP_AND:result=a&&b;tag=TAG_BOOL;break;
    case OP_OR:result=a||b;tag=TAG_BOOL;break;
    case OP_NOT:result=!a;tag=TAG_BOOL;break;
    default:return fvm_bad(c);
    }
    for(uint16_t i=0;i<count;i++)FVM_TRY(SERVICE_FN(runtime_drop)(c,roots[i]));
    FVM_TRY(SERVICE_FN(runtime_scalar)(c,roots[0],tag,result));
    return SERVICE_FN(runtime_frame_next)(c,0);
}
static SERVICE_TYPE(RuntimeStatus) fvm_step(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeFrameView) f,
                                     const SERVICE_TYPE(CodeInstruction) *in) {
    const DecodedInstruction *d=&in->decoded;uint8_t op=d->opcode,edge=0;
    uint32_t src=SERVICE_CONST(RUNTIME_NO_SLOT),dst=SERVICE_CONST(RUNTIME_NO_SLOT);
    SERVICE_TYPE(RuntimeView) value;
    switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_PUSH_STR:
        if(!SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_string_literal)(c,dst));break;
    case OP_STR_EQ:case OP_STR_LEN: {
        uint16_t count=op==OP_STR_EQ?2:1;uint32_t other=SERVICE_CONST(RUNTIME_NO_SLOT);
        if(f.stack_count<count || !SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-count),&src) ||
           (count==2 && !SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-1),&other)))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_string_operation)(c,op,src,other,src));break;
    }
    case OP_PUSH_I64:case OP_PUSH_BOOL:case OP_PUSH_VOID:
        if(!SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_scalar)(c,dst,op==OP_PUSH_I64?TAG_INT:op==OP_PUSH_BOOL?TAG_BOOL:TAG_VOID,
            op==OP_PUSH_I64?d->operands[0].i64:op==OP_PUSH_BOOL?d->operands[0].u8:0));break;
    case OP_FUNCREF:
        if(!SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_indirect_funcref)(c,dst));break;
    case OP_DUP:
        if(!f.stack_count || !SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-1),&src) ||
           !SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_copy)(c,src,dst));break;
    case OP_POP:case OP_FILE_DROP_STACK:
        if(!f.stack_count || !SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-1),&src) ||
           !SERVICE_FN(runtime_view)(c,src,&value))return fvm_bad(c);
        if(op==OP_POP && (value.owning || value.formal))return SERVICE_FN(runtime_fail)(c,SERVICE_CONST(RUNTIME_TYPE));
        FVM_TRY(SERVICE_FN(runtime_drop)(c,src));break;
    case OP_LOAD_LOCAL:case OP_OWN_MOVE_LOCAL:
        if(!SERVICE_FN(runtime_frame_local)(c,d->operands[0].u16,&src) ||
           !SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)f.stack_count,&dst) ||
           !SERVICE_FN(runtime_view)(c,src,&value))return fvm_bad(c);
        if(!value.initialized || value.formal || value.owning!=(op==OP_OWN_MOVE_LOCAL))
            return SERVICE_FN(runtime_fail)(c,SERVICE_CONST(RUNTIME_TYPE));
        FVM_TRY(op==OP_LOAD_LOCAL?SERVICE_FN(runtime_copy)(c,src,dst):SERVICE_FN(runtime_move)(c,src,dst));break;
    case OP_STORE_LOCAL:case OP_OWN_STORE_LOCAL:FVM_TRY(SERVICE_FN(runtime_frame_store)(c));break;
    case OP_REGION_BEGIN:FVM_TRY(SERVICE_FN(runtime_frame_region_begin)(c));break;
    case OP_REGION_END:FVM_TRY(SERVICE_FN(runtime_frame_region_end)(c));break;
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE:FVM_TRY(SERVICE_FN(runtime_frame_borrow)(c));break;
    case OP_FILE_END_BORROW:FVM_TRY(SERVICE_FN(runtime_frame_end_reference)(c));break;
    case OP_FILE_DROP_LOCAL:
        if(!SERVICE_FN(runtime_frame_local)(c,d->operands[0].u16,&src))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_drop)(c,src));break;
    case OP_FILE_RESULT_BRANCH: {
        SERVICE_TYPE(FlowArm) arm;
        if(!SERVICE_FN(runtime_frame_local)(c,d->operands[0].u16,&src))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_result_arm)(c,src,&arm));edge=arm==SERVICE_CONST(FLOW_ARM_ERROR);break;
    }
    case OP_FILE_RESULT_TAKE:
        if(!SERVICE_FN(runtime_frame_local)(c,d->operands[0].u16,&src) ||
           !SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_take)(c,src,d->operands[1].u8?SERVICE_CONST(FLOW_ARM_ERROR):SERVICE_CONST(FLOW_ARM_OK),dst));break;
    case OP_FILE_SERVICE: {
        uint32_t ordinal=in->catalog_ordinal,original,reference=SERVICE_CONST(RUNTIME_NO_SLOT);
        if(!SERVICE_FN(indirect_hosted_import)(SERVICE_FN(runtime_indirect_plan)(c),ordinal,&original) ||
           original!=d->operands[0].u32 || !SERVICE_FN(runtime_frame_scratch)(c,&dst))return fvm_bad(c);
#if SERVICE_ORDERED_ARGUMENTS
        const NlServicePlanMethod *method=DISPATCH_CATALOG_METHOD(SERVICE_FN(runtime_indirect_plan)(c),ordinal);
        if(!method || !method->param_count)return fvm_bad(c);
        bool borrowed=method->input_mode==NL_SERVICE_INPUT_EXCLUSIVE;
        size_t count=method->param_count-1-(borrowed?1:0);uint32_t inputs[2],output;
        if(count>2 || count>f.stack_count)return fvm_bad(c);
        for(size_t i=0;i<count;i++)
            if(!SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-count+i),&inputs[i]))return fvm_bad(c);
        if(borrowed && !SERVICE_FN(runtime_frame_reference)(c,d->operands[1].u16,&reference))return fvm_bad(c);
        FVM_TRY(SERVICE_ORDERED_CALL(c,d->operands[0].u32,reference,inputs,count,dst));
        if(!SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)(f.stack_count-count),&output))return fvm_bad(c);
#else
        uint32_t method=DISPATCH_METHOD_LOCAL(ordinal);
        bool consumes=method==1 || method==4 || (DISPATCH_ENDPOINT(SERVICE_FN(runtime_indirect_plan)(c),ordinal) && method==0);
        if(consumes && (!f.stack_count || !SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-1),&src)))return fvm_bad(c);
        if(method>=1 && method<=3 && !SERVICE_FN(runtime_frame_reference)(c,d->operands[1].u16,&reference))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_service)(c,d->operands[0].u32,reference,src,dst));
        uint32_t output;
        if(!SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)(f.stack_count-(consumes?1:0)),&output))return fvm_bad(c);
#endif
        FVM_TRY(SERVICE_FN(runtime_move)(c,dst,output));break;
    }
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        if(!f.stack_count || !SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-1),&src) ||
           !SERVICE_FN(runtime_view)(c,src,&value))return fvm_bad(c);
        if(!fvm_scalar(&value,TAG_BOOL))return SERVICE_FN(runtime_fail)(c,SERVICE_CONST(RUNTIME_TYPE));
        FVM_TRY(SERVICE_FN(runtime_drop)(c,src));
        if(op==OP_ASSERT) { if(!value.values[0])return SERVICE_FN(runtime_fail)(c,SERVICE_CONST(RUNTIME_ASSERT)); }
        else edge=(value.values[0]!=0)==(op==OP_JMP_TRUE);
        break;
    case OP_AGG_PACK:case OP_UNION_CONSTRUCT: {
        uint16_t variant=d->operands[op==OP_AGG_PACK?2:1].u16;
        uint16_t count=d->operands[op==OP_AGG_PACK?3:2].u16;
        uint32_t inputs[SERVICE_CONST(RUNTIME_FIELDS)];
        if(count>SERVICE_CONST(RUNTIME_FIELDS) || count>f.stack_count)return fvm_bad(c);
        for(uint16_t i=0;i<count;i++)
            if(!SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-count+i),&inputs[i]))return fvm_bad(c);
        if(count)dst=inputs[0];else if(!SERVICE_FN(runtime_frame_reserve)(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(SERVICE_FN(runtime_construct)(c,in->catalog_ordinal,variant,inputs,count,dst));break;
    }
    case OP_AGG_GET:case OP_UNION_FIELD:case OP_AGG_TAG:case OP_UNION_TAG:
        if(!f.stack_count || !SERVICE_FN(runtime_frame_operand)(c,(uint16_t)(f.stack_count-1),&src))return fvm_bad(c);
        if(op==OP_AGG_GET || op==OP_UNION_FIELD)FVM_TRY(SERVICE_FN(runtime_project)(c,src,d->operands[0].u16,src));
        else {
            SERVICE_TYPE(FlowArm) arm;
            if(!SERVICE_FN(runtime_view)(c,src,&value))return fvm_bad(c);
            if(value.owning || value.formal || value.type.category!=SERVICE_CONST(CATEGORY_SCALAR_RESULT))
                return SERVICE_FN(runtime_fail)(c,SERVICE_CONST(RUNTIME_TYPE));
            FVM_TRY(SERVICE_FN(runtime_result_arm)(c,src,&arm));FVM_TRY(SERVICE_FN(runtime_drop)(c,src));
            FVM_TRY(SERVICE_FN(runtime_scalar)(c,src,TAG_INT,arm==SERVICE_CONST(FLOW_ARM_ERROR)));
        }break;
    case OP_FILE_CALL_INDIRECT_REFS:case OP_CALL_INDIRECT:case OP_CALL:case OP_CALL_REF:case OP_FILE_CALL_REFS:return SERVICE_FN(runtime_frame_call)(c);
    case OP_RET:return SERVICE_FN(runtime_frame_return)(c);
    default:return fvm_numeric(c,&f,op);
    }
    return SERVICE_FN(runtime_frame_next)(c,edge);
}
#undef FVM_TRY
static SERVICE_TYPE(IndirectExecutionReport) file_vm_indirect_execute_serialized(const uint8_t *bytes,size_t size,
    const SERVICE_TYPE(IndirectOptions) *options,SERVICE_TYPE(RuntimeView) *out SERVICE_EXECUTION_AUTH_ARGUMENT) {
    if(!out)return fcd_refused(SERVICE_CONST(RUNTIME_INVALID),options);
    SERVICE_TYPE(Runtime) *c=NULL;
    SERVICE_TYPE(RuntimeStatus) status=SERVICE_FN(runtime_indirect_create)(bytes,size,SERVICE_CONST(RUNTIME_VM),options,&c);
    if(status!=SERVICE_CONST(RUNTIME_OK))return fcd_refused(status,options);
    const SERVICE_TYPE(IndirectHostedPlan) *plan=SERVICE_FN(runtime_indirect_plan)(c);
    if(!fcd_coverage(plan)) {
        (void)SERVICE_FN(runtime_indirect_destroy)(&c,NULL);
        return fcd_refused(SERVICE_CONST(RUNTIME_UNRESOLVED),options);
    }
    status=SERVICE_EXECUTION_AUTH_CHECK(c);
    if(status!=SERVICE_CONST(RUNTIME_OK)) {
        (void)SERVICE_FN(runtime_indirect_destroy)(&c,NULL);
        return fcd_refused(status,options);
    }
    status=SERVICE_FN(runtime_begin)(c);
    if(status==SERVICE_CONST(RUNTIME_OK))status=SERVICE_FN(runtime_frame_start)(c);
    while(status==SERVICE_CONST(RUNTIME_OK)) {
        SERVICE_TYPE(IndirectFrameView) current;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(CyclicVariant) variant;
        if(!SERVICE_FN(runtime_indirect_frame_view)(c,&current) || current.instruction_open ||
           current.frame.mode!=SERVICE_CONST(RUNTIME_VM) ||
           !SERVICE_FN(indirect_hosted_instruction)(plan,current.frame.function,(uint16_t)current.frame.instruction,&in) ||
           !SERVICE_FN(indirect_hosted_variant)(plan,current.frame.function,(uint16_t)current.frame.instruction,current.variant,&variant) ||
           !variant.body.reachable || variant.input.stack!=current.frame.stack_count) {
            status=fvm_bad(c);break;
        }
        status=SERVICE_FN(runtime_indirect_enter)(c);
        if(status!=SERVICE_CONST(RUNTIME_OK))break;
        status=fvm_step(c,current.frame,&in);
        if(status!=SERVICE_CONST(RUNTIME_OK))break;
        if(in.decoded.opcode==OP_RET && current.frame.depth==1) {
            uint32_t root;
            if(!SERVICE_FN(runtime_current_root)(c,&root))break;
            status=SERVICE_FN(runtime_frame_start)(c);
        }
    }
    if(status!=SERVICE_CONST(RUNTIME_OK))(void)SERVICE_FN(runtime_fail)(c,status);
    return SERVICE_FN(runtime_indirect_destroy)(&c,out);
}
