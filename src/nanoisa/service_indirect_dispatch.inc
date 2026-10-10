#include "service_dispatch_catalog.inc"
/* Shared explicit indirect adapter coverage; old aindirect authority is unchanged. */
#include "nvm_v2_sections.h"
static bool fcd_supported(uint8_t op) {
    switch(op) {
    case OP_PUSH_STR: case OP_STR_EQ: case OP_STR_LEN:
    case OP_FUNCREF: case OP_FILE_CALL_INDIRECT_REFS:case OP_CALL_INDIRECT:
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
    case OP_DUP: case OP_POP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
    case OP_REGION_BEGIN: case OP_REGION_END: case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE:
    case OP_FILE_SERVICE: case OP_FILE_RESULT_BRANCH: case OP_FILE_RESULT_TAKE:
    case OP_FILE_DROP_LOCAL: case OP_FILE_DROP_STACK: case OP_FILE_END_BORROW:
    case OP_CALL: case OP_CALL_REF:case OP_FILE_CALL_REFS: case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:
    case OP_RET: case OP_ASSERT:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S:
    case OP_I64_REM_S: case OP_I64_NEG:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S:
    case OP_I64_GT_S: case OP_I64_GE_S: case OP_AND: case OP_OR: case OP_NOT:
    case OP_AGG_PACK: case OP_UNION_CONSTRUCT: case OP_AGG_GET: case OP_UNION_FIELD:
    case OP_AGG_TAG: case OP_UNION_TAG:return true;
    default:return false;
    }
}
static bool fcd_fact(const SERVICE_TYPE(IndirectHostedPlan) *p,const SERVICE_TYPE(CodeInstruction) *in,
                     const SERVICE_TYPE(BodyInstruction) *fact) {
    uint8_t op=in->decoded.opcode;
    if(!fcd_supported(op))return false;
    if(!fact->reachable)return true; /* Opcode coverage still includes dead code. */
    SERVICE_TYPE(BodyCleanup) cleanup=SERVICE_CONST(BODY_CLEANUP_NORMAL);
    switch(op) {
    case OP_FILE_DROP_LOCAL:cleanup=SERVICE_CONST(BODY_CLEANUP_DROP_LOCAL);break;
    case OP_FILE_DROP_STACK:cleanup=SERVICE_CONST(BODY_CLEANUP_DROP_STACK);break;
    case OP_ASSERT:cleanup=SERVICE_CONST(BODY_CLEANUP_ASSERT);break;
    case OP_RET:cleanup=SERVICE_CONST(BODY_CLEANUP_RETURN);break;
    case OP_FILE_CALL_INDIRECT_REFS:case OP_CALL_INDIRECT:case OP_CALL:case OP_CALL_REF:case OP_FILE_CALL_REFS:cleanup=SERVICE_CONST(BODY_CLEANUP_CALL);break;
    case OP_FILE_SERVICE:cleanup=SERVICE_CONST(BODY_CLEANUP_SERVICE);break;
    default:break;
    }
    if(fact->cleanup!=cleanup || fact->refinement!=(op==OP_FILE_RESULT_BRANCH) ||
       fact->exit_checked!=(op==OP_RET))return false;
    uint32_t checks=SERVICE_CONST(FLOW_CHECK_CLEANUP),discharged=0;
    bool service=op==OP_FILE_SERVICE,call=isa_is_file_indirect_call(op) || op==OP_CALL || (op==OP_CALL_REF || op==OP_FILE_CALL_REFS);
    if(fact->has_obligation!=(service || call))return false;
    if(service || call) {
        const SERVICE_TYPE(FlowObligation) *o=&fact->obligation;
        if(o->site!=in->byte_offset || o->target!=(isa_is_file_indirect_call(op)?NVM_V2_NO_INDEX:in->decoded.operands[0].u32) ||
           o->kind!=(service?SERVICE_CONST(FLOW_SERVICE):SERVICE_CONST(FLOW_CALL)))return false;
        checks|=SERVICE_CONST(FLOW_CHECK_RESULT);
        if(service) {
            uint32_t original,ordinal=in->catalog_ordinal;
            if(!SERVICE_FN(indirect_hosted_import)(p,ordinal,&original) || original!=o->target)return false;
            const NlServicePlanMethod *method=DISPATCH_CATALOG_METHOD(p,ordinal);
            if(!method || o->required_rights!=method->required_rights ||
               o->acquired_rights!=method->acquired_rights)return false;
            checks|=SERVICE_CONST(FLOW_CHECK_BINDING)|SERVICE_CONST(FLOW_CHECK_INVOCATION)|
                    SERVICE_CONST(FLOW_CHECK_LIVENESS)|SERVICE_CONST(FLOW_CHECK_RIGHTS);
            if(DISPATCH_METHOD_LOCAL(ordinal)>=1 && DISPATCH_METHOD_LOCAL(ordinal)<=3)checks|=SERVICE_CONST(FLOW_CHECK_BORROW);
            if(DISPATCH_METHOD_LOCAL(ordinal)==1)checks|=SERVICE_CONST(FLOW_CHECK_BYTE);
            if(DISPATCH_ENDPOINT(p,ordinal) && DISPATCH_METHOD_LOCAL(ordinal)==0)checks|=FLOW_ENDPOINT_CHECK;
        } else { checks|=SERVICE_CONST(FLOW_CHECK_CALLEE);discharged=SERVICE_CONST(FLOW_CHECK_CALLEE); }
        if(o->checks!=checks)return false;
    }
    return fact->discharged_checks==discharged && fact->pending_checks==(checks&~discharged);
}
static bool fcd_coverage(const SERVICE_TYPE(IndirectHostedPlan) *p) {
    SERVICE_TYPE(IndirectHostedStartup) start;
    if(!SERVICE_FN(indirect_hosted_startup)(p,&start) || start.runtime_admitted)return false;
    for(uint32_t f=0;f<start.functions;f++) {
        SERVICE_TYPE(IndirectHostedFunction) fn;
        if(!SERVICE_FN(indirect_hosted_function)(p,f,&fn) || fn.entry_variant!=0)return false;
        for(uint16_t i=0;i<fn.code.instruction_count;i++) {
            SERVICE_TYPE(CodeInstruction) in;uint8_t count;
            if(!SERVICE_FN(indirect_hosted_instruction)(p,f,i,&in) || !fcd_supported(in.decoded.opcode) || in.successor_count>2 ||
               !SERVICE_FN(indirect_hosted_variant_count)(p,f,i,&count) ||
               count>SERVICE_CONST(CYCLIC_ALTERNATIVES) || (!i && !count))return false;
            for(uint8_t v=0;v<count;v++) {
                SERVICE_TYPE(CyclicVariant) fact;
                if(!SERVICE_FN(indirect_hosted_variant)(p,f,i,v,&fact) || !fact.body.reachable ||
                   !fcd_fact(p,&in,&fact.body) || fact.body.input_stack!=fact.input.stack ||
                   fact.body.output_stack!=fact.output.stack ||
                   (fact.edge_mask & ~((1u<<in.successor_count)-1u)))return false;
                for(uint8_t e=0;e<2;e++) {
                    if(!(fact.edge_mask&(1u<<e))) {
                        if(fact.edge_variants[e]!=SERVICE_CONST(CYCLIC_NO_VARIANT))return false;
                    } else {
                        uint8_t destination_count;
                        if(e>=in.successor_count || in.successors[e]>=fn.code.instruction_count ||
                           !SERVICE_FN(indirect_hosted_variant_count)(p,f,in.successors[e],&destination_count) ||
                           fact.edge_variants[e]>=destination_count)return false;
                    }
                }
                if(isa_is_file_indirect_call(in.decoded.opcode)) {
                    SERVICE_TYPE(IndirectFlowCall) call;
                    if(!SERVICE_FN(indirect_hosted_call)(p,f,i,v,&call) || !call.candidates ||
                       call.candidates!=call.checked_candidates || call.function!=f || call.pc!=in.byte_offset ||
                       call.common.target!=NVM_V2_NO_INDEX)return false;
                    for(uint32_t target=0;target<SERVICE_CONST(FLOW_FUNCTIONS);target++)if(call.candidates&(UINT64_C(1)<<target)) {
                        SERVICE_TYPE(IndirectHostedFunction) callee;
                        if(!SERVICE_FN(indirect_hosted_function)(p,target,&callee) || callee.entry_variant!=0 ||
                           callee.code.declaration.parameters!=call.common.parameters ||
                           callee.code.declaration.result_count!=call.common.result_count)return false;
                    }
                }
                if(in.decoded.opcode==OP_CALL || (in.decoded.opcode==OP_CALL_REF || in.decoded.opcode==OP_FILE_CALL_REFS)) {
                    SERVICE_TYPE(IndirectHostedFunction) callee;
                    if(!SERVICE_FN(indirect_hosted_function)(p,in.decoded.operands[0].u32,&callee) ||
                       callee.entry_variant!=0)return false;
                }
            }
        }
    }
    return true;
}
static inline SERVICE_TYPE(IndirectExecutionReport) fcd_refused(SERVICE_TYPE(RuntimeStatus) status,
                                               const SERVICE_TYPE(IndirectOptions) *options) {
    SERVICE_TYPE(IndirectExecutionReport) report={0};
    report.revision=SERVICE_CONST(INDIRECT_RUNTIME_REVISION);
    report.instruction_limit=options?options->instruction_limit:0;
    report.runtime.status=status;
    report.runtime.function=report.runtime.instruction=NVM_V2_NO_INDEX;
    return report;
}
