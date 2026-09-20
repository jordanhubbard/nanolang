/* I retain every indirect candidate without granting runtime authority. */
#include "file_indirect_hosted.h"
struct NvmFileIndirectHostedPlan {
    NvmFileIndirectFlow *query;
    uint64_t targets[NVM_FILE_CODE_INSTRUCTIONS];
    uint8_t *bytes;
    NvmFileIndirectHostedStartup startup;
    NvmFileIndirectHostedFunction functions[NVM_FILE_FLOW_FUNCTIONS];
};
static bool file_ih_fact(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,
                        const NvmFileCodeInstruction *in,const NvmFileCyclicVariant *v) {
    const NvmFileCyclicReport *r=p->query->ownership;
    if(in->decoded.opcode!=OP_CALL_INDIRECT)return file_ch_fact(r,in,v);
    const NvmFileBodyInstruction *b=&v->body;
    uint64_t bits=p->targets[r->plan->starts[f]+i];
    if(!bits || !b->reachable || b->cleanup!=NVM_FILE_BODY_CLEANUP_CALL ||
       b->cleanup_local!=NVM_FILE_FLOW_NO_REFERENCE || b->refinement || b->exit_checked ||
       !b->has_obligation || b->input_stack!=v->input.stack || b->output_stack!=v->output.stack ||
       !b->input_stack || b->discharged_checks!=NVM_FILE_FLOW_CHECK_CALLEE ||
       b->pending_checks!=(NVM_FILE_FLOW_CHECK_RESULT|NVM_FILE_FLOW_CHECK_CLEANUP))return false;
    for(uint32_t target=0;target<r->plan->count;target++)if(bits&(UINT64_C(1)<<target)) {
        const NvmFileFlowFunction *fn=&r->plan->functions[target].declaration;
        NvmFileFlowObligation expected={0};expected.kind=NVM_FILE_FLOW_CALL;
        expected.site=in->byte_offset;expected.target=NVM_V2_NO_INDEX;
        expected.parameters=fn->parameters;expected.result_count=fn->result_count;expected.result=fn->result;
        expected.checks=NVM_FILE_FLOW_CHECK_CALLEE|NVM_FILE_FLOW_CHECK_RESULT|NVM_FILE_FLOW_CHECK_CLEANUP;
        for(uint16_t local=0;local<fn->parameters;local++) {
            NvmFileFlowDeclaration d;
            if(!nvm_file_indirect_flow_local(p->query,target,local,&d) || d.mode)return false;
            expected.owned_inputs+=owned(d);
        }
        if(!obligation_same(b->obligation,expected))return false;
    }
    return true;
}
static bool file_ih_target_map(NvmFileIndirectHostedPlan *p) {
    const NvmFileIndirectTargets *t=p->query->targets;
    const NvmFileCodePlan *code=p->query->ownership->plan;
    if(t->summary.functions!=code->count || t->summary.instructions!=code->instruction_count)return false;
    for(uint32_t n=0;n<t->summary.calls;n++) {
        NvmFileIndirectCall call=t->calls[n];
        if(call.function>=code->count || call.instruction>=code->functions[call.function].instruction_count ||
           !call.candidates || (code->count<64 && (call.candidates>>code->count)))return false;
        uint32_t index=code->starts[call.function]+call.instruction;
        if(index>=NVM_FILE_CODE_INSTRUCTIONS || p->targets[index] ||
           code->instructions[index].decoded.opcode!=OP_CALL_INDIRECT ||
           code->instructions[index].byte_offset!=call.pc)return false;
        p->targets[index]=call.candidates;
    }
    for(uint32_t i=0;i<code->instruction_count;i++)
        if((code->instructions[i].decoded.opcode==OP_CALL_INDIRECT)!=!!p->targets[i])return false;
    return true;
}
static NvmFileFlowStatus file_ih_start(NvmFileIndirectHostedPlan *p,const NvmV2Module *wire,const NvmModule *m) {
    NvmFileCodePlan *code=p->query->ownership->plan;
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
static bool file_ih_peak(NvmFileIndirectHostedFunction *fn,NvmFileCyclicStateInfo info) {
    if(info.locals!=fn->locals || info.stack>NVM_FILE_FLOW_STACK || info.owners>NVM_FILE_FLOW_OWNERS ||
       info.references>NVM_FILE_FLOW_REFERENCES || info.regions>NVM_FILE_FLOW_REFERENCES)return false;
    if(info.stack>fn->operand_peak)fn->operand_peak=info.stack;
    if(info.owners>fn->frame_owner_peak)fn->frame_owner_peak=info.owners;
    if(info.references>fn->frame_reference_peak)fn->frame_reference_peak=info.references;
    if(info.regions>fn->frame_region_peak)fn->frame_region_peak=info.regions;
    return true;
}
static NvmFileFlowStatus file_ih_bounds(NvmFileIndirectHostedPlan *p,const NvmV2Module *wire) {
    NvmFileCodePlan *code=p->query->ownership->plan;bool checked[NVM_FILE_FLOW_FUNCTIONS]={0};
    if(!file_cyclic_complete(p->query->ownership))return NVM_FILE_FLOW_INVALID;
    for(uint32_t rank=0;rank<code->count;rank++) {
        uint32_t f=code->function_order[rank];
        if(f>=code->count || checked[f])return NVM_FILE_FLOW_INVALID;
        NvmFileIndirectHostedFunction *fn=&p->functions[f];fn->code=code->functions[f];
        fn->declared_stack=wire->functions.items[f].max_stack;fn->locals=fn->code.declaration.locals;
        fn->staging_slots=1;fn->frames=1;fn->entry_variant=0;
        const FileCyclicSite *entry=file_cyclic_site(p->query->ownership,f,0);
        if(!entry || !entry->count || entry->nodes[0]->state.info.stack)return NVM_FILE_FLOW_INVALID;
        for(uint16_t i=0;i<fn->code.instruction_count;i++) {
            const NvmFileCodeInstruction *in=&code->instructions[code->starts[f]+i];
            const FileCyclicSite *site=file_cyclic_site(p->query->ownership,f,i);
            /* Structural query coverage applies even with zero reachable variants. */
            if(!site || (!file_code_supported(in->decoded.opcode) && in->decoded.opcode!=OP_FUNCREF && in->decoded.opcode!=OP_CALL_INDIRECT))return NVM_FILE_FLOW_UNRESOLVED;
            for(uint8_t v=0;v<site->count;v++) {
                const NvmFileCyclicVariant *fact=&site->nodes[v]->fact;
                if(!file_ih_fact(p,f,i,in,fact) || !file_ih_peak(fn,fact->input) ||
                   !file_ih_peak(fn,fact->output))return NVM_FILE_FLOW_INVALID;
                if(fact->body.has_obligation && fact->body.obligation.kind==NVM_FILE_FLOW_CALL) {
                    uint32_t staging=(uint32_t)fact->body.obligation.parameters+1u+(in->decoded.opcode==OP_CALL_INDIRECT);
                    if(staging>NVM_FILE_FLOW_LOCALS+1u)return NVM_FILE_FLOW_INVALID;
                    if(staging>fn->staging_slots)fn->staging_slots=(uint16_t)staging;
                }
            }
        }
        if(fn->declared_stack && fn->declared_stack<fn->operand_peak)return NVM_FILE_FLOW_INVALID;
        uint64_t frame=(uint64_t)fn->locals+fn->operand_peak+fn->staging_slots;
        fn->vm_value_slots=fn->native_value_slots=frame;
        for(uint16_t i=0;i<fn->code.instruction_count;i++) {
            const FileCyclicSite *site=file_cyclic_site(p->query->ownership,f,i);
            for(uint8_t v=0;v<site->count;v++) {
                const NvmFileBodyInstruction *body=&site->nodes[v]->fact.body;
                if(!body->has_obligation || body->obligation.kind!=NVM_FILE_FLOW_CALL)continue;
                const NvmFileFlowObligation *call=&body->obligation;
                bool indirect=code->instructions[code->starts[f]+i].decoded.opcode==OP_CALL_INDIRECT;
                uint64_t candidates=indirect?p->targets[code->starts[f]+i]:
                    (call->target<code->count?UINT64_C(1)<<call->target:0);
                if(!candidates || call->borrowed_inputs>call->parameters ||
                   (indirect && (call->borrowed_inputs || call->target!=NVM_V2_NO_INDEX)))return NVM_FILE_FLOW_INVALID;
                uint32_t consumed=(uint32_t)call->parameters-call->borrowed_inputs+(unsigned)indirect;
                if(consumed>body->input_stack)return NVM_FILE_FLOW_INVALID;
                for(uint32_t target=0;target<code->count;target++)if(candidates&(UINT64_C(1)<<target)) {
                    if(!checked[target])return NVM_FILE_FLOW_INVALID;
                    const NvmFileIndirectHostedFunction *callee=&p->functions[target];
                    uint64_t vm=(uint64_t)fn->locals+body->input_stack-consumed+fn->staging_slots+callee->vm_value_slots;
                    uint64_t native=frame+callee->native_value_slots;
                    if(vm>fn->vm_value_slots)fn->vm_value_slots=vm;
                    if(native>fn->native_value_slots)fn->native_value_slots=native;
                    if((uint32_t)callee->frames+1>fn->frames)fn->frames=(uint16_t)(callee->frames+1);
                }
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
        const NvmFileIndirectHostedFunction *fn=&p->functions[roots[i]];
        if(fn->vm_value_slots>p->startup.vm_value_slots)p->startup.vm_value_slots=fn->vm_value_slots;
        if(fn->native_value_slots>p->startup.native_value_slots)p->startup.native_value_slots=fn->native_value_slots;
        if(fn->frames>p->startup.frames)p->startup.frames=fn->frames;
        if(fn->reference_slots>p->startup.reference_slots)p->startup.reference_slots=fn->reference_slots;
        if(fn->region_slots>p->startup.region_slots)p->startup.region_slots=fn->region_slots;
    }
    return NVM_FILE_FLOW_OK;
}
void nvm_file_indirect_hosted_free(NvmFileIndirectHostedPlan *p) {
    if(p){nvm_file_indirect_flow_free(p->query);free(p->bytes);free(p);}
}
NvmFileFlowStatus nvm_file_indirect_hosted_prepare(const uint8_t *data,size_t size,NvmFileIndirectHostedPlan **out) {
    if(!out)return NVM_FILE_FLOW_INVALID;
    NvmV2Header header;size_t bound;
    NvmFileFlowStatus status=file_hosted_preflight(data,size,&header,&bound);
    if(status!=NVM_FILE_FLOW_OK)return status;
    /* The unchanged preflight reserves a full FLOW budget. Replace that one
     * allowance with the full INDIRECT budget, retaining its unused old-plan
     * allowance conservatively, then account my own plan before allocating. */
    if(bound<NVM_FILE_FLOW_BYTES)return NVM_FILE_FLOW_INVALID;
    bound-=NVM_FILE_FLOW_BYTES;
    if(!file_hosted_add(&bound,1,NVM_FILE_INDIRECT_FLOW_BYTES) ||
       !file_hosted_add(&bound,1,sizeof(NvmFileIndirectHostedPlan)))return NVM_FILE_FLOW_LIMIT;
    uint8_t *copy=malloc(size);if(!copy)return NVM_FILE_FLOW_MEMORY;
    memcpy(copy,data,size);
    NvmV2Module wire={0};NvmModule *module=NULL;NvmFileIndirectHostedPlan *plan=NULL;
    NvmV2Result decoded=nvm_v2_module_deserialize(copy,size,&wire);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    decoded=nvm_v2_to_nvm_module(&wire,&module);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    if(!module || module->code_size!=wire.code_size || !module->code ||
       memcmp(module->code,wire.code,wire.code_size)) {
        status=NVM_FILE_FLOW_UNRESOLVED;goto done;
    }
    plan=calloc(1,sizeof *plan);if(!plan){status=NVM_FILE_FLOW_MEMORY;goto done;}
    plan->startup.revision=NVM_FILE_INDIRECT_HOSTED_REVISION;
    plan->startup.features=header.feature_bits;plan->startup.input_bytes=size;plan->startup.allocation_bound=bound;
    status=nvm_file_indirect_flow_analyze(module,&plan->query);if(status!=NVM_FILE_FLOW_OK)goto done;
    if(plan->query->summary.ownership.revision!=NVM_FILE_CYCLIC_REVISION || plan->query->summary.runtime_admitted || plan->query->summary.ownership.runtime_admitted ||
       plan->query->summary.storage_bound>NVM_FILE_INDIRECT_FLOW_BYTES){status=NVM_FILE_FLOW_INVALID;goto done;}
    plan->startup.query_storage_peak=plan->query->summary.storage_bound;
    size_t retained=sizeof *plan;
    if(!file_hosted_add(&retained,size,1) || !file_hosted_add(&retained,1,plan->startup.query_storage_peak) ||
       retained>bound){status=NVM_FILE_FLOW_LIMIT;goto done;}
    plan->startup.retained_bound=retained;
    if(!file_ih_target_map(plan)){status=NVM_FILE_FLOW_INVALID;goto done;}
    status=file_ih_start(plan,&wire,module);if(status!=NVM_FILE_FLOW_OK)goto done;
    status=file_ih_bounds(plan,&wire);if(status!=NVM_FILE_FLOW_OK)goto done;
    plan->bytes=copy;copy=NULL;*out=plan;plan=NULL;
done:
    nvm_file_indirect_hosted_free(plan);nvm_module_free(module);nvm_v2_module_free(&wire);free(copy);return status;
}
bool nvm_file_indirect_hosted_query_summary(const NvmFileIndirectHostedPlan *p,NvmFileIndirectFlowSummary *out) {
    return p && nvm_file_indirect_flow_summary(p->query,out);
}
bool nvm_file_indirect_hosted_startup(const NvmFileIndirectHostedPlan *p,NvmFileIndirectHostedStartup *out) {
    if(!p || !out)return false;
    *out=p->startup;return true;
}
bool nvm_file_indirect_hosted_function(const NvmFileIndirectHostedPlan *p,uint32_t f,NvmFileIndirectHostedFunction *out) {
    if(!p || !out || f>=p->startup.functions)return false;
    *out=p->functions[f];return true;
}
bool nvm_file_indirect_hosted_function_order(const NvmFileIndirectHostedPlan *p,uint32_t rank,uint32_t *out) {
    return p && nvm_file_code_function_order(p->query->ownership->plan,rank,out);
}
bool nvm_file_indirect_hosted_local(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t l,NvmFileFlowDeclaration *out) {
    return p && nvm_file_indirect_flow_local(p->query,f,l,out);
}
bool nvm_file_indirect_hosted_instruction(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,NvmFileCodeInstruction *out) {
    return p && nvm_file_indirect_flow_instruction(p->query,f,i,out);
}
bool nvm_file_indirect_hosted_component(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint16_t *out) {
    return p && nvm_file_indirect_flow_component(p->query,f,i,out);
}
bool nvm_file_indirect_hosted_variant_count(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint8_t *out) {
    return p && nvm_file_indirect_flow_variant_count(p->query,f,i,out);
}
bool nvm_file_indirect_hosted_variant(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,NvmFileCyclicVariant *out) {
    return p && nvm_file_indirect_flow_variant(p->query,f,i,v,out);
}
bool nvm_file_indirect_hosted_input_local(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t l,NvmFileFlowValue *out) {
    return p && nvm_file_indirect_flow_input_local(p->query,f,i,v,l,out);
}
bool nvm_file_indirect_hosted_input_stack(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t s,NvmFileFlowValue *out) {
    return p && nvm_file_indirect_flow_input_stack(p->query,f,i,v,s,out);
}
bool nvm_file_indirect_hosted_input_reference(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t ref,NvmFileFlowReference *out) {
    return p && nvm_file_indirect_flow_input_reference(p->query,f,i,v,ref,out);
}
bool nvm_file_indirect_hosted_input_region(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,uint16_t region,uint64_t *out) {
    return p && nvm_file_indirect_flow_input_region(p->query,f,i,v,region,out);
}
bool nvm_file_indirect_hosted_type(const NvmFileIndirectHostedPlan *p,uint32_t ordinal,NvmFileNominalLayout *out) {
    return p && nvm_file_indirect_flow_type(p->query,ordinal,out);
}
bool nvm_file_indirect_hosted_import(const NvmFileIndirectHostedPlan *p,uint32_t ordinal,uint32_t *out) {
    return p && nvm_file_indirect_flow_import(p->query,ordinal,out);
}
bool nvm_file_indirect_hosted_bytes(const NvmFileIndirectHostedPlan *p,size_t offset,void *out,size_t count) {
    if(!p || (count && !out) || offset>p->startup.input_bytes || count>p->startup.input_bytes-offset)return false;
    if(count)memcpy(out,p->bytes+offset,count);
    return true;
}
bool nvm_file_indirect_hosted_call(const NvmFileIndirectHostedPlan *p,uint32_t f,uint16_t i,uint8_t v,NvmFileIndirectFlowCall *out) {
    if(!p || !out || f>=p->startup.functions || i>=p->functions[f].code.instruction_count)return false;
    NvmFileCyclicVariant variant;
    uint32_t index=p->query->ownership->plan->starts[f]+i;
    if(!p->targets[index] || !nvm_file_indirect_flow_variant(p->query,f,i,v,&variant))return false;
    *out=(NvmFileIndirectFlowCall){f,p->query->ownership->plan->instructions[index].byte_offset,
        p->targets[index],p->targets[index],variant.body.obligation};
    return true;
}
