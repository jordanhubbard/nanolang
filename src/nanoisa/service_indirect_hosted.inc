/* I retain every indirect candidate without granting runtime authority. */
struct SERVICE_TYPE(IndirectHostedPlan) {
    SERVICE_TYPE(IndirectFlow) *query;
    uint64_t targets[SERVICE_CONST(CODE_INSTRUCTIONS)];
    uint8_t *bytes;
    SERVICE_TYPE(IndirectHostedStartup) startup;
    SERVICE_TYPE(IndirectHostedFunction) functions[SERVICE_CONST(FLOW_FUNCTIONS)];
};
static bool file_ih_fact(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,
                        const SERVICE_TYPE(CodeInstruction) *in,const SERVICE_TYPE(CyclicVariant) *v) {
    const SERVICE_TYPE(CyclicReport) *r=p->query->ownership;
    if(!isa_is_file_indirect_call(in->decoded.opcode))return file_ch_fact(r,in,v);
    const SERVICE_TYPE(BodyInstruction) *b=&v->body;
    uint64_t bits=p->targets[r->plan->starts[f]+i];
    if(!bits || !b->reachable || b->cleanup!=SERVICE_CONST(BODY_CLEANUP_CALL) ||
       b->cleanup_local!=SERVICE_CONST(FLOW_NO_REFERENCE) || b->refinement || b->exit_checked ||
       !b->has_obligation || b->input_stack!=v->input.stack || b->output_stack!=v->output.stack ||
       !b->input_stack || b->discharged_checks!=SERVICE_CONST(FLOW_CHECK_CALLEE) ||
       b->pending_checks!=(SERVICE_CONST(FLOW_CHECK_RESULT)|SERVICE_CONST(FLOW_CHECK_CLEANUP)))return false;
    for(uint32_t target=0;target<r->plan->count;target++)if(bits&(UINT64_C(1)<<target)) {
        const SERVICE_TYPE(FlowFunction) *fn=&r->plan->functions[target].declaration;
        SERVICE_TYPE(FlowObligation) expected={0};expected.kind=SERVICE_CONST(FLOW_CALL);
        expected.site=in->byte_offset;expected.target=NVM_V2_NO_INDEX;
        expected.parameters=fn->parameters;expected.result_count=fn->result_count;expected.result=fn->result;
        expected.checks=SERVICE_CONST(FLOW_CHECK_CALLEE)|SERVICE_CONST(FLOW_CHECK_RESULT)|SERVICE_CONST(FLOW_CHECK_CLEANUP);
        for(uint16_t local=0;local<fn->parameters;local++) {
            SERVICE_TYPE(FlowDeclaration) d;
            if(!SERVICE_FN(indirect_flow_local)(p->query,target,local,&d))return false;
            if(d.mode && in->decoded.opcode!=OP_FILE_CALL_INDIRECT_REFS)return false;
            if(d.mode ? in->call_references[local]>=SERVICE_CONST(FLOW_REFERENCES) :
               (in->decoded.opcode==OP_FILE_CALL_INDIRECT_REFS && in->call_references[local]!=SERVICE_CONST(FLOW_NO_REFERENCE)))return false;
            expected.borrowed_inputs+=d.mode!=0;
            expected.owned_inputs+=owned(d);
        }
        if(!obligation_same(b->obligation,expected))return false;
    }
    return true;
}
static bool file_ih_target_map(SERVICE_TYPE(IndirectHostedPlan) *p) {
    const SERVICE_TYPE(IndirectTargets) *t=p->query->targets;
    const SERVICE_TYPE(CodePlan) *code=p->query->ownership->plan;
    if(t->summary.functions!=code->count || t->summary.instructions!=code->instruction_count)return false;
    for(uint32_t n=0;n<t->summary.calls;n++) {
        SERVICE_TYPE(IndirectCall) call=t->calls[n];
        if(call.function>=code->count || call.instruction>=code->functions[call.function].instruction_count ||
           !call.candidates || (code->count<64 && (call.candidates>>code->count)))return false;
        uint32_t index=code->starts[call.function]+call.instruction;
        if(index>=SERVICE_CONST(CODE_INSTRUCTIONS) || p->targets[index] ||
           !isa_is_file_indirect_call(code->instructions[index].decoded.opcode) ||
           code->instructions[index].byte_offset!=call.pc)return false;
        p->targets[index]=call.candidates;
    }
    for(uint32_t i=0;i<code->instruction_count;i++)
        if((isa_is_file_indirect_call(code->instructions[i].decoded.opcode))!=!!p->targets[i])return false;
    return true;
}
static SERVICE_TYPE(FlowStatus) file_ih_start(SERVICE_TYPE(IndirectHostedPlan) *p,const NvmV2Module *wire,const NvmModule *m) {
    SERVICE_TYPE(CodePlan) *code=p->query->ownership->plan;
    if(code->count!=wire->functions.count || code->count!=m->function_count ||
       !(m->header.flags & NVM_FLAG_HAS_MAIN) || m->header.entry_point!=wire->entry_point ||
       wire->entry_point>=code->count)return SERVICE_CONST(FLOW_INVALID);
    p->startup.entry=wire->entry_point;p->startup.initializer=NVM_V2_NO_INDEX;p->startup.functions=code->count;
    const SERVICE_TYPE(FlowFunction) *entry=&code->functions[p->startup.entry].declaration;
    if(entry->parameters || entry->result_count!=1 || entry->result.mode ||
       entry->result.global_index!=NVM_V2_NO_INDEX ||
       (entry->result.tag!=TAG_INT && entry->result.tag!=TAG_BOOL))return SERVICE_CONST(FLOW_UNRESOLVED);
    for(uint32_t f=0;f<code->count;f++) {
        uint32_t index=m->functions[f].name_idx;
        if(index>=m->string_count || !m->strings[index] || memchr(m->strings[index],0,m->string_lengths[index]))
            return SERVICE_CONST(FLOW_INVALID);
        if(p->startup.initializer==NVM_V2_NO_INDEX && !strcmp(m->strings[index],"__init__")) {
            const SERVICE_TYPE(FlowFunction) *init=&code->functions[f].declaration;
            if(init->parameters || init->result_count || init->result.tag!=TAG_VOID ||
               init->result.mode || init->result.global_index!=NVM_V2_NO_INDEX)return SERVICE_CONST(FLOW_UNRESOLVED);
            p->startup.initializer=f;
        }
    }
    return SERVICE_CONST(FLOW_OK);
}
static bool file_ih_peak(SERVICE_TYPE(IndirectHostedFunction) *fn,SERVICE_TYPE(CyclicStateInfo) info) {
    if(info.locals!=fn->locals || info.stack>SERVICE_CONST(FLOW_STACK) || info.owners>SERVICE_CONST(FLOW_OWNERS) ||
       info.references>SERVICE_CONST(FLOW_REFERENCES) || info.regions>SERVICE_CONST(FLOW_REFERENCES))return false;
    if(info.stack>fn->operand_peak)fn->operand_peak=info.stack;
    if(info.owners>fn->frame_owner_peak)fn->frame_owner_peak=info.owners;
    if(info.references>fn->frame_reference_peak)fn->frame_reference_peak=info.references;
    if(info.regions>fn->frame_region_peak)fn->frame_region_peak=info.regions;
    return true;
}
static SERVICE_TYPE(FlowStatus) file_ih_bounds(SERVICE_TYPE(IndirectHostedPlan) *p,const NvmV2Module *wire) {
    SERVICE_TYPE(CodePlan) *code=p->query->ownership->plan;bool checked[SERVICE_CONST(FLOW_FUNCTIONS)]={0};
    if(!file_cyclic_complete(p->query->ownership))return SERVICE_CONST(FLOW_INVALID);
    for(uint32_t rank=0;rank<code->count;rank++) {
        uint32_t f=code->function_order[rank];
        if(f>=code->count || checked[f])return SERVICE_CONST(FLOW_INVALID);
        SERVICE_TYPE(IndirectHostedFunction) *fn=&p->functions[f];fn->code=code->functions[f];
        fn->declared_stack=wire->functions.items[f].max_stack;fn->locals=fn->code.declaration.locals;
        fn->staging_slots=1;fn->frames=1;fn->entry_variant=0;
        const FileCyclicSite *entry=file_cyclic_site(p->query->ownership,f,0);
        if(!entry || !entry->count || entry->nodes[0]->state.info.stack)return SERVICE_CONST(FLOW_INVALID);
        for(uint16_t i=0;i<fn->code.instruction_count;i++) {
            const SERVICE_TYPE(CodeInstruction) *in=&code->instructions[code->starts[f]+i];
            const FileCyclicSite *site=file_cyclic_site(p->query->ownership,f,i);
            /* Structural query coverage applies even with zero reachable variants. */
            if(!site || !file_code_supported_mode(in->decoded.opcode,true))return SERVICE_CONST(FLOW_UNRESOLVED);
            for(uint8_t v=0;v<site->count;v++) {
                const SERVICE_TYPE(CyclicVariant) *fact=&site->nodes[v]->fact;
                if(!file_ih_fact(p,f,i,in,fact) || !file_ih_peak(fn,fact->input) ||
                   !file_ih_peak(fn,fact->output))return SERVICE_CONST(FLOW_INVALID);
                if(fact->body.has_obligation && fact->body.obligation.kind==SERVICE_CONST(FLOW_CALL)) {
                    uint32_t staging=(uint32_t)fact->body.obligation.parameters+1u+(isa_is_file_indirect_call(in->decoded.opcode));
                    if(staging>SERVICE_CONST(FLOW_LOCALS)+2u)return SERVICE_CONST(FLOW_INVALID);
                    if(staging>fn->staging_slots)fn->staging_slots=(uint16_t)staging;
                }
            }
        }
        if(fn->declared_stack && fn->declared_stack<fn->operand_peak)return SERVICE_CONST(FLOW_INVALID);
        uint64_t frame=(uint64_t)fn->locals+fn->operand_peak+fn->staging_slots;
        fn->vm_value_slots=fn->native_value_slots=frame;
        for(uint16_t i=0;i<fn->code.instruction_count;i++) {
            const FileCyclicSite *site=file_cyclic_site(p->query->ownership,f,i);
            for(uint8_t v=0;v<site->count;v++) {
                const SERVICE_TYPE(BodyInstruction) *body=&site->nodes[v]->fact.body;
                if(!body->has_obligation || body->obligation.kind!=SERVICE_CONST(FLOW_CALL))continue;
                const SERVICE_TYPE(FlowObligation) *call=&body->obligation;
                bool indirect=isa_is_file_indirect_call(code->instructions[code->starts[f]+i].decoded.opcode);
                uint64_t candidates=indirect?p->targets[code->starts[f]+i]:
                    (call->target<code->count?UINT64_C(1)<<call->target:0);
                if(!candidates || call->borrowed_inputs>call->parameters ||
                   (indirect && call->target!=NVM_V2_NO_INDEX))return SERVICE_CONST(FLOW_INVALID);
                uint32_t consumed=(uint32_t)call->parameters-call->borrowed_inputs+(unsigned)indirect;
                if(consumed>body->input_stack)return SERVICE_CONST(FLOW_INVALID);
                for(uint32_t target=0;target<code->count;target++)if(candidates&(UINT64_C(1)<<target)) {
                    if(!checked[target])return SERVICE_CONST(FLOW_INVALID);
                    const SERVICE_TYPE(IndirectHostedFunction) *callee=&p->functions[target];
                    uint64_t vm=(uint64_t)fn->locals+body->input_stack-consumed+fn->staging_slots+callee->vm_value_slots;
                    uint64_t native=frame+callee->native_value_slots;
                    if(vm>fn->vm_value_slots)fn->vm_value_slots=vm;
                    if(native>fn->native_value_slots)fn->native_value_slots=native;
                    if((uint32_t)callee->frames+1>fn->frames)fn->frames=(uint16_t)(callee->frames+1);
                }
            }
        }
        fn->reference_slots=(uint32_t)fn->frames*SERVICE_CONST(FLOW_REFERENCES);
        fn->region_slots=(uint32_t)fn->frames*SERVICE_CONST(FLOW_REFERENCES);
        if(fn->frames>SERVICE_CONST(FLOW_FUNCTIONS) || fn->vm_value_slots>UINT64_C(64)*770 ||
           fn->native_value_slots>UINT64_C(64)*770)return SERVICE_CONST(FLOW_LIMIT);
        checked[f]=true;
    }
    uint32_t roots[2]={p->startup.entry,p->startup.initializer};
    for(unsigned i=0;i<2;i++)if(roots[i]!=NVM_V2_NO_INDEX) {
        const SERVICE_TYPE(IndirectHostedFunction) *fn=&p->functions[roots[i]];
        if(fn->vm_value_slots>p->startup.vm_value_slots)p->startup.vm_value_slots=fn->vm_value_slots;
        if(fn->native_value_slots>p->startup.native_value_slots)p->startup.native_value_slots=fn->native_value_slots;
        if(fn->frames>p->startup.frames)p->startup.frames=fn->frames;
        if(fn->reference_slots>p->startup.reference_slots)p->startup.reference_slots=fn->reference_slots;
        if(fn->region_slots>p->startup.region_slots)p->startup.region_slots=fn->region_slots;
    }
    return SERVICE_CONST(FLOW_OK);
}
void SERVICE_FN(indirect_hosted_free)(SERVICE_TYPE(IndirectHostedPlan) *p) {
    if(p){SERVICE_FN(indirect_flow_free)(p->query);free(p->bytes);free(p);}
}
SERVICE_TYPE(FlowStatus) SERVICE_FN(indirect_hosted_prepare)(const uint8_t *data,size_t size,SERVICE_TYPE(IndirectHostedPlan) **out) {
    if(!out)return SERVICE_CONST(FLOW_INVALID);
    NvmV2Header header;size_t bound;
    SERVICE_TYPE(FlowStatus) status=file_hosted_preflight(data,size,&header,&bound);
    if(status!=SERVICE_CONST(FLOW_OK))return status;
    /* The unchanged preflight reserves a full FLOW budget. Replace that one
     * allowance with the full INDIRECT budget, retaining its unused old-plan
     * allowance conservatively, then account my own plan before allocating. */
    if(bound<SERVICE_CONST(FLOW_BYTES))return SERVICE_CONST(FLOW_INVALID);
    bound-=SERVICE_CONST(FLOW_BYTES);
    if(!file_hosted_add(&bound,1,SERVICE_CONST(INDIRECT_FLOW_BYTES)) ||
       !file_hosted_add(&bound,1,sizeof(SERVICE_TYPE(IndirectHostedPlan))))return SERVICE_CONST(FLOW_LIMIT);
    uint8_t *copy=malloc(size);if(!copy)return SERVICE_CONST(FLOW_MEMORY);
    memcpy(copy,data,size);
    NvmV2Module wire={0};NvmModule *module=NULL;SERVICE_TYPE(IndirectHostedPlan) *plan=NULL;
    NvmV2Result decoded=nvm_v2_module_deserialize(copy,size,&wire);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    decoded=nvm_v2_to_nvm_module(&wire,&module);
    if(decoded!=NVM_V2_OK){status=file_hosted_codec_status(decoded);goto done;}
    if(!module || module->code_size!=wire.code_size || !module->code ||
       memcmp(module->code,wire.code,wire.code_size)) {
        status=SERVICE_CONST(FLOW_UNRESOLVED);goto done;
    }
    plan=calloc(1,sizeof *plan);if(!plan){status=SERVICE_CONST(FLOW_MEMORY);goto done;}
    plan->startup.revision=SERVICE_CONST(INDIRECT_HOSTED_REVISION);
    plan->startup.features=header.feature_bits;plan->startup.input_bytes=size;plan->startup.allocation_bound=bound;
    status=SERVICE_FN(indirect_flow_analyze)(module,&plan->query);if(status!=SERVICE_CONST(FLOW_OK))goto done;
    if(plan->query->summary.ownership.revision!=SERVICE_CONST(CYCLIC_REVISION) || plan->query->summary.runtime_admitted || plan->query->summary.ownership.runtime_admitted ||
       plan->query->summary.storage_bound>SERVICE_CONST(INDIRECT_FLOW_BYTES)){status=SERVICE_CONST(FLOW_INVALID);goto done;}
    plan->startup.query_storage_peak=plan->query->summary.storage_bound;
    size_t retained=sizeof *plan;
    if(!file_hosted_add(&retained,size,1) || !file_hosted_add(&retained,1,plan->startup.query_storage_peak) ||
       retained>bound){status=SERVICE_CONST(FLOW_LIMIT);goto done;}
    plan->startup.retained_bound=retained;
    if(!file_ih_target_map(plan)){status=SERVICE_CONST(FLOW_INVALID);goto done;}
    status=file_ih_start(plan,&wire,module);if(status!=SERVICE_CONST(FLOW_OK))goto done;
    status=file_ih_bounds(plan,&wire);if(status!=SERVICE_CONST(FLOW_OK))goto done;
    plan->bytes=copy;copy=NULL;*out=plan;plan=NULL;
done:
    SERVICE_FN(indirect_hosted_free)(plan);nvm_module_free(module);nvm_v2_module_free(&wire);free(copy);return status;
}
bool SERVICE_FN(indirect_hosted_query_summary)(const SERVICE_TYPE(IndirectHostedPlan) *p,SERVICE_TYPE(IndirectFlowSummary) *out) {
    return p && SERVICE_FN(indirect_flow_summary)(p->query,out);
}
bool SERVICE_FN(indirect_hosted_startup)(const SERVICE_TYPE(IndirectHostedPlan) *p,SERVICE_TYPE(IndirectHostedStartup) *out) {
    if(!p || !out)return false;
    *out=p->startup;return true;
}
bool SERVICE_FN(indirect_hosted_function)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,SERVICE_TYPE(IndirectHostedFunction) *out) {
    if(!p || !out || f>=p->startup.functions)return false;
    *out=p->functions[f];return true;
}
bool SERVICE_FN(indirect_hosted_function_order)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t rank,uint32_t *out) {
    return p && SERVICE_FN(code_function_order)(p->query->ownership->plan,rank,out);
}
bool SERVICE_FN(indirect_hosted_local)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t l,SERVICE_TYPE(FlowDeclaration) *out) {
    return p && SERVICE_FN(indirect_flow_local)(p->query,f,l,out);
}
bool SERVICE_FN(indirect_hosted_instruction)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,SERVICE_TYPE(CodeInstruction) *out) {
    return p && SERVICE_FN(indirect_flow_instruction)(p->query,f,i,out);
}
bool SERVICE_FN(indirect_hosted_component)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint16_t *out) {
    return p && SERVICE_FN(indirect_flow_component)(p->query,f,i,out);
}
bool SERVICE_FN(indirect_hosted_variant_count)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t *out) {
    return p && SERVICE_FN(indirect_flow_variant_count)(p->query,f,i,out);
}
bool SERVICE_FN(indirect_hosted_variant)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t v,SERVICE_TYPE(CyclicVariant) *out) {
    return p && SERVICE_FN(indirect_flow_variant)(p->query,f,i,v,out);
}
bool SERVICE_FN(indirect_hosted_input_local)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t v,uint16_t l,SERVICE_TYPE(FlowValue) *out) {
    return p && SERVICE_FN(indirect_flow_input_local)(p->query,f,i,v,l,out);
}
bool SERVICE_FN(indirect_hosted_input_stack)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t v,uint16_t s,SERVICE_TYPE(FlowValue) *out) {
    return p && SERVICE_FN(indirect_flow_input_stack)(p->query,f,i,v,s,out);
}
bool SERVICE_FN(indirect_hosted_input_reference)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t v,uint16_t ref,SERVICE_TYPE(FlowReference) *out) {
    return p && SERVICE_FN(indirect_flow_input_reference)(p->query,f,i,v,ref,out);
}
bool SERVICE_FN(indirect_hosted_input_region)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t v,uint16_t region,uint64_t *out) {
    return p && SERVICE_FN(indirect_flow_input_region)(p->query,f,i,v,region,out);
}
bool SERVICE_FN(indirect_hosted_type)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t ordinal,SERVICE_TYPE(NominalLayout) *out) {
    return p && SERVICE_FN(indirect_flow_type)(p->query,ordinal,out);
}
bool SERVICE_FN(indirect_hosted_import)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t ordinal,uint32_t *out) {
    return p && SERVICE_FN(indirect_flow_import)(p->query,ordinal,out);
}
bool SERVICE_FN(indirect_hosted_bytes)(const SERVICE_TYPE(IndirectHostedPlan) *p,size_t offset,void *out,size_t count) {
    if(!p || (count && !out) || offset>p->startup.input_bytes || count>p->startup.input_bytes-offset)return false;
    if(count)memcpy(out,p->bytes+offset,count);
    return true;
}
bool SERVICE_FN(indirect_hosted_call)(const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t v,SERVICE_TYPE(IndirectFlowCall) *out) {
    if(!p || !out || f>=p->startup.functions || i>=p->functions[f].code.instruction_count)return false;
    SERVICE_TYPE(CyclicVariant) variant;
    uint32_t index=p->query->ownership->plan->starts[f]+i;
    if(!p->targets[index] || !SERVICE_FN(indirect_flow_variant)(p->query,f,i,v,&variant))return false;
    *out=(SERVICE_TYPE(IndirectFlowCall)){f,p->query->ownership->plan->instructions[index].byte_offset,
        p->targets[index],p->targets[index],variant.body.obligation};
    return true;
}

/* I inspect the retained, validated constant section; no caller bytes survive
 * preparation. Counted strings preserve embedded zero bytes. */
bool SERVICE_FN(indirect_hosted_string)(const SERVICE_TYPE(IndirectHostedPlan) *p,
    uint32_t index,const uint8_t **out,size_t *length) {
    if(!p || !out || !length)return false;
    NvmV2Header h;
    if(nvm_v2_read_header(p->bytes,p->startup.input_bytes,&h)!=NVM_V2_OK)return false;
    for(uint32_t section=0;section<h.section_count;section++) {
        NvmV2SectionEntry e;
        if(nvm_v2_read_section(p->bytes,p->startup.input_bytes,&h,section,&e)!=NVM_V2_OK)return false;
        if(e.type!=NVM_V2_SECTION_CONSTANTS)continue;
        NvmV2Cursor c;nvm_v2_cursor_init(&c,p->bytes+e.offset,(size_t)e.size);
        uint32_t count;if(nvm_v2_u32(&c,&count)!=NVM_V2_OK || index>=count)return false;
        for(uint32_t i=0;i<=index;i++) {
            uint8_t tag,pad;uint16_t pad2;uint32_t size;const uint8_t *bytes;
            if(nvm_v2_u8(&c,&tag)!=NVM_V2_OK || nvm_v2_u8(&c,&pad)!=NVM_V2_OK ||
               nvm_v2_u16(&c,&pad2)!=NVM_V2_OK || nvm_v2_u32(&c,&size)!=NVM_V2_OK ||
               tag!=TAG_STRING || pad || pad2 || nvm_v2_take(&c,size,&bytes)!=NVM_V2_OK ||
               nvm_v2_align4(&c)!=NVM_V2_OK)return false;
            if(i==index){*out=bytes;*length=size;return true;}
        }
    }
    return false;
}
