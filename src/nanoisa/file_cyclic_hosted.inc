/* I retain every cyclic alternative without granting executable authority.
 * Owning TU: file_flow.c, after the qualified hosted and cyclic query helpers. */
#include "file_cyclic_hosted.h"
struct NvmFileCyclicHostedPlan {
    NvmFileCyclicReport *query;
    uint8_t *bytes;
    NvmFileCyclicHostedStartup startup;
    NvmFileCyclicHostedFunction functions[NVM_FILE_FLOW_FUNCTIONS];
};
static bool file_ch_member(const NvmFileCyclicReport *r,const char *id,NvmFileFlowDeclaration *out) {
    if(!id){*out=scalar_type(TAG_VOID);return true;}
    if(!strcmp(id,"nsi:core/int")){*out=scalar_type(TAG_INT);return true;}
    if(!strcmp(id,"nsi:core/bool")){*out=scalar_type(TAG_BOOL);return true;}
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        const NlFilePlanType *type=nl_file_catalog_type(i);NvmFileNominalLayout layout;
        if(type && !strcmp(type->id,id) && nvm_file_cyclic_type(r,i,&layout)) {
            *out=(NvmFileFlowDeclaration){layout.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
                0,layout.global_index,layout.catalog_ordinal,layout.category};return true;
        }
    }
    return false;
}
/* Check the complete logical obligation inventory separately from any future
 * runtime dispatcher. I do not discharge pending service/lifetime checks. */
static bool file_ch_fact(const NvmFileCyclicReport *r,const NvmFileCodeInstruction *in,
                         const NvmFileCyclicVariant *variant) {
    const NvmFileBodyInstruction *fact=&variant->body;uint8_t op=in->decoded.opcode;
    NvmFileBodyCleanup cleanup=NVM_FILE_BODY_CLEANUP_NORMAL;
    uint16_t cleanup_local=NVM_FILE_FLOW_NO_REFERENCE;
    switch(op) {
    case OP_FILE_DROP_LOCAL:cleanup=NVM_FILE_BODY_CLEANUP_DROP_LOCAL;cleanup_local=in->decoded.operands[0].u16;break;
    case OP_FILE_DROP_STACK:cleanup=NVM_FILE_BODY_CLEANUP_DROP_STACK;break;
    case OP_ASSERT:cleanup=NVM_FILE_BODY_CLEANUP_ASSERT;break;
    case OP_RET:cleanup=NVM_FILE_BODY_CLEANUP_RETURN;break;
    case OP_CALL:case OP_CALL_REF:cleanup=NVM_FILE_BODY_CLEANUP_CALL;break;
    case OP_FILE_SERVICE:cleanup=NVM_FILE_BODY_CLEANUP_SERVICE;break;
    default:break;
    }
    bool call=op==OP_CALL || op==OP_CALL_REF,service=op==OP_FILE_SERVICE;
    if(!fact->reachable || fact->cleanup!=cleanup || fact->cleanup_local!=cleanup_local ||
       fact->refinement!=(op==OP_FILE_RESULT_BRANCH) || fact->exit_checked!=(op==OP_RET) ||
       fact->input_stack!=variant->input.stack || fact->output_stack!=variant->output.stack ||
       fact->has_obligation!=(call || service))return false;
    NvmFileFlowObligation expected={0};uint32_t discharged=0,checks=NVM_FILE_FLOW_CHECK_CLEANUP;
    if(call) {
        uint32_t target=in->decoded.operands[0].u32;
        if(target>=r->plan->count)return false;
        const NvmFileFlowFunction *fn=&r->plan->functions[target].declaration;
        expected.kind=NVM_FILE_FLOW_CALL;expected.site=in->byte_offset;expected.target=target;
        expected.parameters=fn->parameters;expected.result_count=fn->result_count;expected.result=fn->result;
        for(uint16_t i=0;i<fn->parameters;i++) {
            NvmFileFlowDeclaration d;
            if(!nvm_file_cyclic_local(r,target,i,&d))return false;
            expected.owned_inputs+=owned(d);expected.borrowed_inputs+=d.mode!=0;
        }
        expected.checks=checks|NVM_FILE_FLOW_CHECK_CALLEE|NVM_FILE_FLOW_CHECK_RESULT;
        discharged=NVM_FILE_FLOW_CHECK_CALLEE;checks=expected.checks;
    } else if(service) {
        const NlFilePlanMethod *method=nl_file_catalog_method(in->catalog_ordinal);uint32_t import;
        if(!method || !method->param_count ||
           !nvm_file_cyclic_import(r,in->catalog_ordinal,&import) || import!=in->decoded.operands[0].u32 ||
           !file_ch_member(r,method->params[method->param_count-1].type_id,&expected.result))return false;
        expected.kind=NVM_FILE_FLOW_SERVICE;expected.site=in->byte_offset;expected.target=import;
        expected.parameters=(uint16_t)(method->param_count-1);expected.result_count=1;
        expected.required_rights=method->required_rights;expected.acquired_rights=method->acquired_rights;
        expected.outcomes[0]=input_state(method->outcomes[0].input_state);
        expected.outcomes[1]=input_state(method->outcomes[1].input_state);
        checks|=NVM_FILE_FLOW_CHECK_BINDING|NVM_FILE_FLOW_CHECK_INVOCATION|
                NVM_FILE_FLOW_CHECK_LIVENESS|NVM_FILE_FLOW_CHECK_RIGHTS|NVM_FILE_FLOW_CHECK_RESULT;
        if(method->input_mode==NL_FILE_INPUT_EXCLUSIVE) {
            expected.borrowed_inputs=1;checks|=NVM_FILE_FLOW_CHECK_BORROW;
            if(in->catalog_ordinal==1)checks|=NVM_FILE_FLOW_CHECK_BYTE;
        } else if(method->input_mode==NL_FILE_INPUT_CONSUME)expected.owned_inputs=1;
        expected.checks=checks;
    }
    return fact->discharged_checks==discharged && fact->pending_checks==(checks&~discharged) &&
        obligation_same(fact->obligation,expected);
}
static NvmFileFlowStatus file_ch_start(NvmFileCyclicHostedPlan *p,const NvmV2Module *wire,const NvmModule *m) {
    NvmFileCodePlan *code=p->query->plan;
    if(code->count!=wire->functions.count || code->count!=m->function_count ||
       !(m->header.flags & NVM_FLAG_HAS_MAIN) || m->header.entry_point!=wire->entry_point ||
       wire->entry_point>=code->count)return NVM_FILE_FLOW_INVALID;
    p->startup.entry=wire->entry_point;p->startup.initializer=NVM_V2_NO_INDEX;p->startup.functions=code->count;
    const NvmFileFlowFunction *entry=&code->functions[p->startup.entry].declaration;
    if(entry->parameters || entry->result_count!=1 || entry->result.mode ||
       entry->result.global_index!=NVM_V2_NO_INDEX ||
       (entry->result.tag!=TAG_INT && entry->result.tag!=TAG_BOOL))return NVM_FILE_FLOW_UNRESOLVED;
    for(uint32_t f=0;f<code->count;f++) {
        uint32_t index=m->functions[f].name_idx;
        if(index>=m->string_count || !m->strings[index] || memchr(m->strings[index],0,m->string_lengths[index]))
            return NVM_FILE_FLOW_INVALID;
        if(p->startup.initializer==NVM_V2_NO_INDEX && !strcmp(m->strings[index],"__init__")) {
            const NvmFileFlowFunction *init=&code->functions[f].declaration;
            if(init->parameters || init->result_count || init->result.tag!=TAG_VOID ||
               init->result.mode || init->result.global_index!=NVM_V2_NO_INDEX)return NVM_FILE_FLOW_UNRESOLVED;
            p->startup.initializer=f;
        }
    }
    return NVM_FILE_FLOW_OK;
}
static bool file_ch_peak(NvmFileCyclicHostedFunction *fn,NvmFileCyclicStateInfo info) {
    if(info.locals!=fn->locals || info.stack>NVM_FILE_FLOW_STACK || info.owners>NVM_FILE_FLOW_OWNERS ||
       info.references>NVM_FILE_FLOW_REFERENCES || info.regions>NVM_FILE_FLOW_REFERENCES)return false;
    if(info.stack>fn->operand_peak)fn->operand_peak=info.stack;
    if(info.owners>fn->frame_owner_peak)fn->frame_owner_peak=info.owners;
    if(info.references>fn->frame_reference_peak)fn->frame_reference_peak=info.references;
    if(info.regions>fn->frame_region_peak)fn->frame_region_peak=info.regions;
    return true;
}
static NvmFileFlowStatus file_ch_bounds(NvmFileCyclicHostedPlan *p,const NvmV2Module *wire) {
    NvmFileCodePlan *code=p->query->plan;bool checked[NVM_FILE_FLOW_FUNCTIONS]={0};
    if(!file_cyclic_complete(p->query))return NVM_FILE_FLOW_INVALID;
    for(uint32_t rank=0;rank<code->count;rank++) {
        uint32_t f=code->function_order[rank];
        if(f>=code->count || checked[f])return NVM_FILE_FLOW_INVALID;
        NvmFileCyclicHostedFunction *fn=&p->functions[f];fn->code=code->functions[f];
        fn->declared_stack=wire->functions.items[f].max_stack;fn->locals=fn->code.declaration.locals;
        fn->staging_slots=1;fn->frames=1;fn->entry_variant=0;
        const FileCyclicSite *entry=file_cyclic_site(p->query,f,0);
        if(!entry || !entry->count || entry->nodes[0]->state.info.stack)return NVM_FILE_FLOW_INVALID;
        for(uint16_t i=0;i<fn->code.instruction_count;i++) {
            const NvmFileCodeInstruction *in=&code->instructions[code->starts[f]+i];
            const FileCyclicSite *site=file_cyclic_site(p->query,f,i);
            /* Structural query coverage applies even with zero reachable variants. */
            if(!site || !file_code_supported(in->decoded.opcode))return NVM_FILE_FLOW_UNRESOLVED;
            for(uint8_t v=0;v<site->count;v++) {
                const NvmFileCyclicVariant *fact=&site->nodes[v]->fact;
                if(!file_ch_fact(p->query,in,fact) || !file_ch_peak(fn,fact->input) ||
                   !file_ch_peak(fn,fact->output))return NVM_FILE_FLOW_INVALID;
                if(fact->body.has_obligation && fact->body.obligation.kind==NVM_FILE_FLOW_CALL) {
                    uint32_t staging=(uint32_t)fact->body.obligation.parameters+1;
                    if(staging>NVM_FILE_FLOW_LOCALS+1u)return NVM_FILE_FLOW_INVALID;
                    if(staging>fn->staging_slots)fn->staging_slots=(uint16_t)staging;
                }
            }
        }
        if(fn->declared_stack && fn->declared_stack<fn->operand_peak)return NVM_FILE_FLOW_INVALID;
        uint64_t frame=(uint64_t)fn->locals+fn->operand_peak+fn->staging_slots;
        fn->vm_value_slots=fn->native_value_slots=frame;
        for(uint16_t i=0;i<fn->code.instruction_count;i++) {
            const FileCyclicSite *site=file_cyclic_site(p->query,f,i);
            for(uint8_t v=0;v<site->count;v++) {
                const NvmFileBodyInstruction *body=&site->nodes[v]->fact.body;
                if(!body->has_obligation || body->obligation.kind!=NVM_FILE_FLOW_CALL)continue;
                const NvmFileFlowObligation *call=&body->obligation;
                if(call->target>=code->count || !checked[call->target] || call->borrowed_inputs>call->parameters)
                    return NVM_FILE_FLOW_INVALID;
                uint16_t supplied=(uint16_t)(call->parameters-call->borrowed_inputs);
                if(supplied>body->input_stack)return NVM_FILE_FLOW_INVALID;
                const NvmFileCyclicHostedFunction *callee=&p->functions[call->target];
                /* All operands/frame sizes are bounded above, and each callee
                 * has already passed the <=64*769 value/64 frame checks. */
                uint64_t vm=(uint64_t)fn->locals+body->input_stack-supplied+fn->staging_slots+callee->vm_value_slots;
                uint64_t native=frame+callee->native_value_slots;
                if(vm>fn->vm_value_slots)fn->vm_value_slots=vm;
                if(native>fn->native_value_slots)fn->native_value_slots=native;
                if((uint32_t)callee->frames+1>fn->frames)fn->frames=(uint16_t)(callee->frames+1);
            }
        }
        fn->reference_slots=(uint32_t)fn->frames*NVM_FILE_FLOW_REFERENCES;
        fn->region_slots=(uint32_t)fn->frames*NVM_FILE_FLOW_REFERENCES;
        if(fn->frames>NVM_FILE_FLOW_FUNCTIONS || fn->vm_value_slots>UINT64_C(64)*769 ||
           fn->native_value_slots>UINT64_C(64)*769)return NVM_FILE_FLOW_LIMIT;
        checked[f]=true;
    }
    uint32_t roots[2]={p->startup.entry,p->startup.initializer};
    for(unsigned i=0;i<2;i++)if(roots[i]!=NVM_V2_NO_INDEX) {
        const NvmFileCyclicHostedFunction *fn=&p->functions[roots[i]];
        if(fn->vm_value_slots>p->startup.vm_value_slots)p->startup.vm_value_slots=fn->vm_value_slots;
        if(fn->native_value_slots>p->startup.native_value_slots)p->startup.native_value_slots=fn->native_value_slots;
        if(fn->frames>p->startup.frames)p->startup.frames=fn->frames;
        if(fn->reference_slots>p->startup.reference_slots)p->startup.reference_slots=fn->reference_slots;
        if(fn->region_slots>p->startup.region_slots)p->startup.region_slots=fn->region_slots;
    }
    return NVM_FILE_FLOW_OK;
}
void nvm_file_cyclic_hosted_free(NvmFileCyclicHostedPlan *p) {
    if(p){nvm_file_cyclic_free(p->query);free(p->bytes);free(p);}
}
NvmFileFlowStatus nvm_file_cyclic_hosted_prepare(const uint8_t *data,size_t size,NvmFileCyclicHostedPlan **out) {
    if(!out)return NVM_FILE_FLOW_INVALID;
    NvmV2Header header;size_t bound;
    NvmFileFlowStatus status=file_hosted_preflight(data,size,&header,&bound);
    if(status!=NVM_FILE_FLOW_OK)return status;
    /* The unchanged preflight reserves a full FLOW budget. Replace that one
     * allowance with the full CYCLIC budget, retaining its unused old-plan
     * allowance conservatively, then account my own plan before allocating. */
    if(bound<NVM_FILE_FLOW_BYTES)return NVM_FILE_FLOW_INVALID;
    bound-=NVM_FILE_FLOW_BYTES;
    if(!file_hosted_add(&bound,1,NVM_FILE_CYCLIC_BYTES) ||
       !file_hosted_add(&bound,1,sizeof(NvmFileCyclicHostedPlan)))return NVM_FILE_FLOW_LIMIT;
    uint8_t *copy=malloc(size);if(!copy)return NVM_FILE_FLOW_MEMORY;
    memcpy(copy,data,size);
    NvmV2Module wire={0};NvmModule *module=NULL;NvmFileCyclicHostedPlan *plan=NULL;
    NvmV2Result decoded=nvm_v2_module_deserialize(copy,size,&wire);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    decoded=nvm_v2_to_nvm_module(&wire,&module);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    if(!module || module->code_size!=wire.code_size || !module->code ||
       memcmp(module->code,wire.code,wire.code_size)) {
        status=NVM_FILE_FLOW_UNRESOLVED;goto done;
    }
    plan=calloc(1,sizeof *plan);if(!plan){status=NVM_FILE_FLOW_MEMORY;goto done;}
    plan->startup.revision=NVM_FILE_CYCLIC_HOSTED_REVISION;
    plan->startup.features=header.feature_bits;plan->startup.input_bytes=size;plan->startup.allocation_bound=bound;
    status=nvm_file_cyclic_analyze(module,&plan->query);if(status!=NVM_FILE_FLOW_OK)goto done;
    if(plan->query->summary.revision!=NVM_FILE_CYCLIC_REVISION || plan->query->summary.runtime_admitted ||
       plan->query->summary.storage_peak>NVM_FILE_CYCLIC_BYTES){status=NVM_FILE_FLOW_INVALID;goto done;}
    plan->startup.query_storage_peak=plan->query->summary.storage_peak;
    size_t retained=sizeof *plan;
    if(!file_hosted_add(&retained,size,1) || !file_hosted_add(&retained,1,plan->startup.query_storage_peak) ||
       retained>bound){status=NVM_FILE_FLOW_LIMIT;goto done;}
    plan->startup.retained_bound=retained;
    status=file_ch_start(plan,&wire,module);if(status!=NVM_FILE_FLOW_OK)goto done;
    status=file_ch_bounds(plan,&wire);if(status!=NVM_FILE_FLOW_OK)goto done;
    plan->bytes=copy;copy=NULL;*out=plan;plan=NULL;
done:
    nvm_file_cyclic_hosted_free(plan);nvm_module_free(module);nvm_v2_module_free(&wire);free(copy);return status;
}
bool nvm_file_cyclic_hosted_query_summary(const NvmFileCyclicHostedPlan *p,NvmFileCyclicSummary *out) {
    return p && nvm_file_cyclic_summary(p->query,out);
}
bool nvm_file_cyclic_hosted_startup(const NvmFileCyclicHostedPlan *p,NvmFileCyclicHostedStartup *out) {
    if(!p || !out)return false;
    *out=p->startup;return true;
}
bool nvm_file_cyclic_hosted_function(const NvmFileCyclicHostedPlan *p,uint32_t f,NvmFileCyclicHostedFunction *out) {
    if(!p || !out || f>=p->startup.functions)return false;
    *out=p->functions[f];return true;
}
bool nvm_file_cyclic_hosted_function_order(const NvmFileCyclicHostedPlan *p,uint32_t rank,uint32_t *out) {
    return p && nvm_file_code_function_order(p->query->plan,rank,out);
}
bool nvm_file_cyclic_hosted_local(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t l,NvmFileFlowDeclaration *out) {
    return p && nvm_file_cyclic_local(p->query,f,l,out);
}
bool nvm_file_cyclic_hosted_instruction(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,NvmFileCodeInstruction *out) {
    return p && nvm_file_cyclic_instruction(p->query,f,i,out);
}
bool nvm_file_cyclic_hosted_component(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,uint16_t *out) {
    return p && nvm_file_cyclic_component(p->query,f,i,out);
}
bool nvm_file_cyclic_hosted_variant_count(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,uint8_t *out) {
    return p && nvm_file_cyclic_variant_count(p->query,f,i,out);
}
bool nvm_file_cyclic_hosted_variant(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,NvmFileCyclicVariant *out) {
    return p && nvm_file_cyclic_variant(p->query,f,i,v,out);
}
bool nvm_file_cyclic_hosted_input_local(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t l,NvmFileFlowValue *out) {
    return p && nvm_file_cyclic_input_local(p->query,f,i,v,l,out);
}
bool nvm_file_cyclic_hosted_input_stack(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t s,NvmFileFlowValue *out) {
    return p && nvm_file_cyclic_input_stack(p->query,f,i,v,s,out);
}
bool nvm_file_cyclic_hosted_input_reference(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t ref,NvmFileFlowReference *out) {
    return p && nvm_file_cyclic_input_reference(p->query,f,i,v,ref,out);
}
bool nvm_file_cyclic_hosted_input_region(const NvmFileCyclicHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t region,uint64_t *out) {
    return p && nvm_file_cyclic_input_region(p->query,f,i,v,region,out);
}
bool nvm_file_cyclic_hosted_type(const NvmFileCyclicHostedPlan *p,uint32_t ordinal,NvmFileNominalLayout *out) {
    return p && nvm_file_cyclic_type(p->query,ordinal,out);
}
bool nvm_file_cyclic_hosted_import(const NvmFileCyclicHostedPlan *p,uint32_t ordinal,uint32_t *out) {
    return p && nvm_file_cyclic_import(p->query,ordinal,out);
}
bool nvm_file_cyclic_hosted_bytes(const NvmFileCyclicHostedPlan *p,size_t offset,void *out,size_t count) {
    if(!p || (count && !out) || offset>p->startup.input_bytes || count>p->startup.input_bytes-offset)return false;
    if(count)memcpy(out,p->bytes+offset,count);
    return true;
}
