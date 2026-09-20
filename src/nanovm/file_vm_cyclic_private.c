#include "file_vm_cyclic_private.h"
#ifdef NVM_FILE_CYCLIC_VM_PRIVATE
#include "../nanoisa/file_cyclic_dispatch.inc"
#include <limits.h>
#include <string.h>
static NvmFileRuntimeStatus fvm_bad(NvmFileRuntime *c) {
    return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_STATE);
}
static bool fvm_scalar(const NvmFileRuntimeView *v,uint8_t tag) {
    return v->initialized && !v->owning && !v->formal && !v->type.mode &&
        v->type.category==NVM_FILE_CATEGORY_UNKNOWN && v->type.tag==tag &&
        v->type.global_index==NVM_V2_NO_INDEX && v->type.catalog_ordinal==NVM_V2_NO_INDEX &&
        v->fields==(tag==TAG_VOID?0:1) && (tag!=TAG_BOOL || v->values[0]==0 || v->values[0]==1);
}
static int64_t fvm_bits(uint64_t bits) { int64_t result;memcpy(&result,&bits,sizeof result);return result; }
#define FVM_TRY(x) do { NvmFileRuntimeStatus status_=(x); if(status_!=NVM_FILE_RUNTIME_OK)return status_; } while(0)
static NvmFileRuntimeStatus fvm_numeric(NvmFileRuntime *c,const NvmFileRuntimeFrameView *f,uint8_t op) {
    bool unary=op==OP_NEG || op==OP_I64_NEG || op==OP_NOT;
    uint16_t count=unary?1:2;uint32_t roots[2];NvmFileRuntimeView values[2];
    if(f->stack_count<count)return fvm_bad(c);
    for(uint16_t i=0;i<count;i++)
        if(!nvm_file_runtime_frame_operand(c,(uint16_t)(f->stack_count-count+i),&roots[i]) ||
           !nvm_file_runtime_view(c,roots[i],&values[i]))return fvm_bad(c);
    bool logic=op==OP_AND || op==OP_OR || op==OP_NOT;
    uint8_t input=logic?TAG_BOOL:TAG_INT;
    if((op==OP_EQ || op==OP_NE) && values[0].type.tag==TAG_BOOL)input=TAG_BOOL;
    for(uint16_t i=0;i<count;i++)if(!fvm_scalar(&values[i],input))
        return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);
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
    for(uint16_t i=0;i<count;i++)FVM_TRY(nvm_file_runtime_drop(c,roots[i]));
    FVM_TRY(nvm_file_runtime_scalar(c,roots[0],tag,result));
    return nvm_file_runtime_frame_next(c,0);
}
static NvmFileRuntimeStatus fvm_step(NvmFileRuntime *c,NvmFileRuntimeFrameView f,
                                     const NvmFileCodeInstruction *in) {
    const DecodedInstruction *d=&in->decoded;uint8_t op=d->opcode,edge=0;
    uint32_t src=NVM_FILE_RUNTIME_NO_SLOT,dst=NVM_FILE_RUNTIME_NO_SLOT;
    NvmFileRuntimeView value;
    switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_PUSH_I64:case OP_PUSH_BOOL:case OP_PUSH_VOID:
        if(!nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_scalar(c,dst,op==OP_PUSH_I64?TAG_INT:op==OP_PUSH_BOOL?TAG_BOOL:TAG_VOID,
            op==OP_PUSH_I64?d->operands[0].i64:op==OP_PUSH_BOOL?d->operands[0].u8:0));break;
    case OP_DUP:
        if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) ||
           !nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_copy(c,src,dst));break;
    case OP_POP:case OP_FILE_DROP_STACK:
        if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) ||
           !nvm_file_runtime_view(c,src,&value))return fvm_bad(c);
        if(op==OP_POP && (value.owning || value.formal))return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);
        FVM_TRY(nvm_file_runtime_drop(c,src));break;
    case OP_LOAD_LOCAL:case OP_OWN_MOVE_LOCAL:
        if(!nvm_file_runtime_frame_local(c,d->operands[0].u16,&src) ||
           !nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst) ||
           !nvm_file_runtime_view(c,src,&value))return fvm_bad(c);
        if(!value.initialized || value.formal || value.owning!=(op==OP_OWN_MOVE_LOCAL))
            return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);
        FVM_TRY(op==OP_LOAD_LOCAL?nvm_file_runtime_copy(c,src,dst):nvm_file_runtime_move(c,src,dst));break;
    case OP_STORE_LOCAL:case OP_OWN_STORE_LOCAL:FVM_TRY(nvm_file_runtime_frame_store(c));break;
    case OP_REGION_BEGIN:FVM_TRY(nvm_file_runtime_frame_region_begin(c));break;
    case OP_REGION_END:FVM_TRY(nvm_file_runtime_frame_region_end(c));break;
    case OP_BORROW_LOCAL_EXCLUSIVE:FVM_TRY(nvm_file_runtime_frame_borrow(c));break;
    case OP_FILE_END_BORROW:FVM_TRY(nvm_file_runtime_frame_end_reference(c));break;
    case OP_FILE_DROP_LOCAL:
        if(!nvm_file_runtime_frame_local(c,d->operands[0].u16,&src))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_drop(c,src));break;
    case OP_FILE_RESULT_BRANCH: {
        NvmFileFlowArm arm;
        if(!nvm_file_runtime_frame_local(c,d->operands[0].u16,&src))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_result_arm(c,src,&arm));edge=arm==NVM_FILE_FLOW_ARM_ERROR;break;
    }
    case OP_FILE_RESULT_TAKE:
        if(!nvm_file_runtime_frame_local(c,d->operands[0].u16,&src) ||
           !nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_take(c,src,d->operands[1].u8?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK,dst));break;
    case OP_FILE_SERVICE: {
        uint32_t ordinal=in->catalog_ordinal,original,reference=NVM_FILE_RUNTIME_NO_SLOT;
        if(!nvm_file_cyclic_hosted_import(nvm_file_runtime_cyclic_plan(c),ordinal,&original) ||
           original!=d->operands[0].u32 || !nvm_file_runtime_frame_scratch(c,&dst))return fvm_bad(c);
        bool consumes=ordinal==1 || ordinal==4;
        if(consumes && (!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src)))return fvm_bad(c);
        if(ordinal>=1 && ordinal<=3 && !nvm_file_runtime_frame_reference(c,d->operands[1].u16,&reference))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_service(c,d->operands[0].u32,reference,src,dst));
        uint32_t output;
        if(!nvm_file_runtime_frame_reserve(c,(uint16_t)(f.stack_count-(consumes?1:0)),&output))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_move(c,dst,output));break;
    }
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) ||
           !nvm_file_runtime_view(c,src,&value))return fvm_bad(c);
        if(!fvm_scalar(&value,TAG_BOOL))return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);
        FVM_TRY(nvm_file_runtime_drop(c,src));
        if(op==OP_ASSERT) { if(!value.values[0])return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_ASSERT); }
        else edge=(value.values[0]!=0)==(op==OP_JMP_TRUE);
        break;
    case OP_AGG_PACK:case OP_UNION_CONSTRUCT: {
        uint16_t variant=d->operands[op==OP_AGG_PACK?2:1].u16;
        uint16_t count=d->operands[op==OP_AGG_PACK?3:2].u16;
        uint32_t inputs[NVM_FILE_RUNTIME_FIELDS];
        if(count>NVM_FILE_RUNTIME_FIELDS || count>f.stack_count)return fvm_bad(c);
        for(uint16_t i=0;i<count;i++)
            if(!nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-count+i),&inputs[i]))return fvm_bad(c);
        if(count)dst=inputs[0];else if(!nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return fvm_bad(c);
        FVM_TRY(nvm_file_runtime_construct(c,in->catalog_ordinal,variant,inputs,count,dst));break;
    }
    case OP_AGG_GET:case OP_UNION_FIELD:case OP_AGG_TAG:case OP_UNION_TAG:
        if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src))return fvm_bad(c);
        if(op==OP_AGG_GET || op==OP_UNION_FIELD)FVM_TRY(nvm_file_runtime_project(c,src,d->operands[0].u16,src));
        else {
            NvmFileFlowArm arm;
            if(!nvm_file_runtime_view(c,src,&value))return fvm_bad(c);
            if(value.owning || value.formal || value.type.category!=NVM_FILE_CATEGORY_SCALAR_RESULT)
                return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);
            FVM_TRY(nvm_file_runtime_result_arm(c,src,&arm));FVM_TRY(nvm_file_runtime_drop(c,src));
            FVM_TRY(nvm_file_runtime_scalar(c,src,TAG_INT,arm==NVM_FILE_FLOW_ARM_ERROR));
        }break;
    case OP_CALL:case OP_CALL_REF:return nvm_file_runtime_frame_call(c);
    case OP_RET:return nvm_file_runtime_frame_return(c);
    default:return fvm_numeric(c,&f,op);
    }
    return nvm_file_runtime_frame_next(c,edge);
}
#undef FVM_TRY
NvmFileCyclicExecutionReport nvm_file_vm_cyclic_execute(const uint8_t *bytes,size_t size,
    const NvmFileCyclicOptions *options,NvmFileRuntimeView *out) {
    if(!out)return fcd_refused(NVM_FILE_RUNTIME_INVALID,options);
    NvmFileRuntime *c=NULL;
    NvmFileRuntimeStatus status=nvm_file_runtime_cyclic_create(bytes,size,NVM_FILE_RUNTIME_VM,options,&c);
    if(status!=NVM_FILE_RUNTIME_OK)return fcd_refused(status,options);
    const NvmFileCyclicHostedPlan *plan=nvm_file_runtime_cyclic_plan(c);
    if(!fcd_coverage(plan)) {
        (void)nvm_file_runtime_cyclic_destroy(&c,NULL);
        return fcd_refused(NVM_FILE_RUNTIME_UNRESOLVED,options);
    }
    status=nvm_file_runtime_begin(c);
    if(status==NVM_FILE_RUNTIME_OK)status=nvm_file_runtime_frame_start(c);
    while(status==NVM_FILE_RUNTIME_OK) {
        NvmFileCyclicFrameView current;NvmFileCodeInstruction in;NvmFileCyclicVariant variant;
        if(!nvm_file_runtime_cyclic_frame_view(c,&current) || current.instruction_open ||
           current.frame.mode!=NVM_FILE_RUNTIME_VM ||
           !nvm_file_cyclic_hosted_instruction(plan,current.frame.function,(uint16_t)current.frame.instruction,&in) ||
           !nvm_file_cyclic_hosted_variant(plan,current.frame.function,(uint16_t)current.frame.instruction,current.variant,&variant) ||
           !variant.body.reachable || variant.input.stack!=current.frame.stack_count) {
            status=fvm_bad(c);break;
        }
        status=nvm_file_runtime_cyclic_enter(c);
        if(status!=NVM_FILE_RUNTIME_OK)break;
        status=fvm_step(c,current.frame,&in);
        if(status!=NVM_FILE_RUNTIME_OK)break;
        if(in.decoded.opcode==OP_RET && current.frame.depth==1) {
            uint32_t root;
            if(!nvm_file_runtime_current_root(c,&root))break;
            status=nvm_file_runtime_frame_start(c);
        }
    }
    if(status!=NVM_FILE_RUNTIME_OK)(void)nvm_file_runtime_fail(c,status);
    return nvm_file_runtime_cyclic_destroy(&c,out);
}
#endif
