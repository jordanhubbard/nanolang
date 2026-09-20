/* Private cyclic carrier. No decoded dispatch or public admission. */
static bool frc_identity(NlFileValue a,NlFileValue b) {
    return a.invocation==b.invocation && a.generation==b.generation && a.slot==b.slot;
}
static bool frc_zero(NlFileValue a) { return !a.invocation && !a.generation && !a.slot; }
static NvmFileRuntimeStatus frc_bad(NvmFileRuntime *c) {
    return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
}
static bool frc_payload(const FileRuntimeValue *v) {
    const NvmFileRuntimeView *p=&v->view;
    if(!p->initialized) {
        if(!frc_zero(v->owner) || v->reference || p->owning || p->formal || p->fields ||
           p->arm || p->type.tag || p->type.mode || p->type.global_index ||
           p->type.catalog_ordinal || p->type.category)return false;
        for(uint8_t i=0;i<NVM_FILE_RUNTIME_FIELDS;i++)if(p->values[i])return false;
        return true;
    }
    if(p->owning)return !p->formal && !p->fields && !v->reference;
    if(!frc_zero(v->owner))return false;
    if(p->formal)return p->type.mode==2 && !p->fields;
    if(v->reference || p->type.mode)return false;
    if(p->type.tag==TAG_INT)return p->fields==1 && p->arm==NVM_FILE_FLOW_ARM_NONE;
    if(p->type.tag==TAG_BOOL)return p->fields==1 && p->arm==NVM_FILE_FLOW_ARM_NONE &&
        (p->values[0]==0 || p->values[0]==1);
    if(p->type.tag==TAG_VOID)return !p->fields && p->arm==NVM_FILE_FLOW_ARM_NONE;
    const NlFilePlanType *type=nl_file_catalog_type(p->type.catalog_ordinal);
    if(!type)return false;
    if(p->type.category==NVM_FILE_CATEGORY_RECORD) {
        if(p->fields!=type->member_count || p->fields>NVM_FILE_RUNTIME_FIELDS ||
           p->arm!=NVM_FILE_FLOW_ARM_NONE)return false;
        for(uint8_t i=0;i<p->fields;i++)
            if(!strcmp(type->members[i].type_id,"nsi:core/bool") && p->values[i]!=0 && p->values[i]!=1)return false;
        return true;
    }
    if(p->type.category!=NVM_FILE_CATEGORY_SCALAR_RESULT ||
       (p->arm!=NVM_FILE_FLOW_ARM_OK && p->arm!=NVM_FILE_FLOW_ARM_ERROR))return false;
    if(p->arm==NVM_FILE_FLOW_ARM_ERROR) {
        if(p->fields!=NVM_FILE_RUNTIME_FIELDS)return false;
        for(uint8_t i=4;i<7;i++)if(p->values[i]!=0 && p->values[i]!=1)return false;
        return true;
    }
    /* Catalog success payloads: WriteCount(int), Unit, ReadByte(int,bool). */
    uint32_t ordinal=p->type.catalog_ordinal;
    if(ordinal==4)return p->fields==1;
    if(ordinal==6)return p->fields==2 && (p->values[1]==0 || p->values[1]==1);
    return (ordinal==5 || ordinal==7) && !p->fields;
}
static bool frc_value(NvmFileRuntime *c,uint32_t root,NvmFileFlowValue expected) {
    if(root>=c->storage.values)return false;
    const FileRuntimeValue *v=&c->values[root];
    if(!frc_payload(v) || v->view.initialized!=expected.initialized)return false;
    if(!expected.initialized)return !expected.owner;
    if(!frf_matches(v->view,expected.type))return false;
    NvmFileFlowArm actual=v->view.arm;
    if(v->view.owning) {
        bool open=v->view.type.category==NVM_FILE_CATEGORY_OPEN_RESULT;
        if(nl_file_value_validate(c->files,&v->owner,open)!=NL_FILE_VALUE_OK)return false;
        if(open) {
            NlFileOpenView view;
            if(nl_file_open_view(c->files,&v->owner,&view)!=NL_FILE_VALUE_OK)return false;
            actual=view.ok?NVM_FILE_FLOW_ARM_OK:NVM_FILE_FLOW_ARM_ERROR;
        }
        if(!expected.owner || expected.owner>NVM_FILE_FLOW_OWNERS)return false;
        uint32_t index=(uint32_t)expected.owner-1;
        if(!frc_zero(c->witness->owners[index]))return false;
        for(uint32_t i=0;i<NVM_FILE_FLOW_OWNERS;i++)
            if(!frc_zero(c->witness->owners[i]) && frc_identity(c->witness->owners[i],v->owner))return false;
        c->witness->owners[index]=v->owner;c->witness->roots[index]=root;
    } else if(expected.owner)return false;
    return expected.arm==NVM_FILE_FLOW_ARM_UNKNOWN?
        (actual==NVM_FILE_FLOW_ARM_OK || actual==NVM_FILE_FLOW_ARM_ERROR):actual==expected.arm;
}
static bool frc_reference_empty(const FileRuntimeReference *r) {
    return !r->live && !r->formal && !r->owner_root && !r->origin && !r->formal_root &&
        !r->region && !r->borrow.epoch && frc_zero(r->borrow.value);
}
static bool frc_frame(NvmFileRuntime *c,const FileRuntimeFrame *f,uint32_t region_end) {
    NvmFileCyclicVariant fact;NvmFileHostedFunction fn;
    if(!fr_function(c,f->function,&fn) || f->instruction>UINT16_MAX ||
       !nvm_file_cyclic_hosted_variant(c->cyclic_plan,f->function,(uint16_t)f->instruction,f->variant,&fact) ||
       !fact.body.reachable || fact.input.locals!=fn.locals || f->stack_count>fn.operand_peak ||
       (!f->waiting && f->stack_count!=fact.input.stack) ||
       (f->waiting && f->stack_count>fact.input.stack) ||
       !frf_range(c,f->locals_base,fn.locals) || !frf_range(c,f->stack_base,f->stack_count) ||
       !frf_empty_range(c,f->staging_base,fn.staging_slots) ||
       f->reference_base>c->storage.references || NVM_FILE_FLOW_REFERENCES>c->storage.references-f->reference_base ||
       f->region_base>region_end || region_end>c->region_count ||
       region_end-f->region_base!=fact.input.regions)return false;
    if(f->waiting) {
        NvmFileCodeInstruction in;
        if(!nvm_file_cyclic_hosted_instruction(c->cyclic_plan,f->function,(uint16_t)f->instruction,&in) ||
           (in.decoded.opcode!=OP_CALL && in.decoded.opcode!=OP_CALL_REF) || !fact.body.has_obligation ||
           (uint32_t)f->stack_count+fact.body.obligation.parameters-fact.body.obligation.borrowed_inputs!=fact.input.stack)
            return false;
    }
    memset(c->witness,0,sizeof *c->witness);
    for(uint16_t i=0;i<fn.locals;i++) {
        NvmFileFlowValue v;
        if(!nvm_file_cyclic_hosted_input_local(c->cyclic_plan,f->function,(uint16_t)f->instruction,f->variant,i,&v) ||
           !frc_value(c,f->locals_base+i,v))return false;
    }
    for(uint16_t i=0;i<f->stack_count;i++) {
        NvmFileFlowValue v;
        if(!nvm_file_cyclic_hosted_input_stack(c->cyclic_plan,f->function,(uint16_t)f->instruction,f->variant,i,&v) ||
           !frc_value(c,f->stack_base+i,v))return false;
    }
    for(uint16_t i=0;i<fact.input.regions;i++) {
        uint64_t abstract;
        if(!nvm_file_cyclic_hosted_input_region(c->cyclic_plan,f->function,(uint16_t)f->instruction,f->variant,i,&abstract) ||
           abstract!=UINT64_C(513)+i || !c->regions[f->region_base+i])return false;
    }
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++) {
        NvmFileFlowReference expected;const FileRuntimeReference *r=&c->references[f->reference_base+i];
        if(!nvm_file_cyclic_hosted_input_reference(c->cyclic_plan,f->function,(uint16_t)f->instruction,f->variant,i,&expected))return false;
        if(!expected.live){if(!frc_reference_empty(r))return false;continue;}
        if(!r->live || expected.formal!=r->formal || !fr_reference(c,f->reference_base+i) ||
           nl_file_value_borrow_validate(c->files,&r->borrow)!=NL_FILE_VALUE_OK ||
           !frc_identity(r->borrow.value,c->values[r->owner_root].owner))return false;
        const FileRuntimeReference *origin=&c->references[r->origin];
        if(!frc_identity(origin->borrow.value,r->borrow.value) || origin->borrow.epoch!=r->borrow.epoch)return false;
        if(expected.formal) {
            if(i>=fn.code.declaration.parameters || expected.local!=i || expected.owner!=UINT64_C(257)+i ||
               expected.identity!=expected.owner || expected.region || r->region ||
               r->formal_root!=f->locals_base+i || !c->values[r->formal_root].view.formal ||
               c->values[r->formal_root].reference!=f->reference_base+i || r->origin>=f->reference_base)return false;
        } else {
            if(!expected.owner || expected.owner>NVM_FILE_FLOW_OWNERS || expected.local>=fn.locals ||
               expected.identity!=UINT64_C(769)+i || expected.region<513 ||
               expected.region-513>=fact.input.regions || r->origin!=f->reference_base+i ||
               r->formal_root!=NVM_FILE_RUNTIME_NO_SLOT || r->owner_root!=f->locals_base+expected.local ||
               c->witness->roots[expected.owner-1]!=r->owner_root ||
               !frc_identity(c->witness->owners[expected.owner-1],r->borrow.value) ||
               r->region!=c->regions[f->region_base+(uint32_t)(expected.region-513)])return false;
        }
    }
    return true;
}
static NvmFileRuntimeStatus frc_validate(NvmFileRuntime *c,const FileRuntimeFrame *current) {
    if(c->kind!=FR_CYCLIC || !c->witness || !c->frame_count ||
       c->frame_count>c->storage.frames || c->region_count>c->storage.regions || !c->acquired || !c->files)return frc_bad(c);
    bool slots[NL_FILE_VALUE_SLOTS]={0};uint64_t rooted=0,borrowed=0;
    /* Every owning root, including waiting callers, has a unique live core slot.
     * The full generation/type check prevents stale same-slot aliases. */
    for(uint32_t root=0;root<c->storage.values;root++) {
        FileRuntimeValue *v=&c->values[root];
        if(!frc_payload(v))return frc_bad(c);
        if(v->view.initialized) {
            bool covered=false;
            for(uint16_t d=0;d<c->frame_count;d++) {
                const FileRuntimeFrame *f=d+1==c->frame_count?current:&c->frames[d];
                NvmFileHostedFunction fn;if(!fr_function(c,f->function,&fn))return frc_bad(c);
                if((root>=f->locals_base && root-f->locals_base<fn.locals) ||
                   (root>=f->stack_base && root-f->stack_base<f->stack_count))covered=true;
            }
            if(!covered)return frc_bad(c);
        }
        if(v->view.initialized && v->view.owning) {
            if(v->owner.slot>=NL_FILE_VALUE_SLOTS || slots[v->owner.slot])return frc_bad(c);
            NlFileValueStatus checked=nl_file_value_validate(c->files,&v->owner,
                v->view.type.category==NVM_FILE_CATEGORY_OPEN_RESULT);
            if(checked!=NL_FILE_VALUE_OK)return fr_core(c,checked);
            slots[v->owner.slot]=true;rooted|=UINT64_C(1)<<v->owner.slot;
        }
    }
    for(uint32_t i=0;i<c->region_count;i++) {
        if(!c->regions[i])return frc_bad(c);
        for(uint32_t j=0;j<i;j++)if(c->regions[j]==c->regions[i])return frc_bad(c);
    }
    for(uint32_t i=c->region_count;i<c->storage.regions;i++)if(c->regions[i])return frc_bad(c);
    for(uint32_t i=(uint32_t)c->frame_count*NVM_FILE_FLOW_REFERENCES;i<c->storage.references;i++)
        if(!frc_reference_empty(&c->references[i]))return frc_bad(c);
    for(uint32_t i=0;i<c->storage.references;i++) {
        const FileRuntimeReference *r=&c->references[i];
        if(r->live && !r->formal) {
            if(r->borrow.value.slot>=NL_FILE_VALUE_SLOTS ||
               (borrowed&(UINT64_C(1)<<r->borrow.value.slot)))return frc_bad(c);
            borrowed|=UINT64_C(1)<<r->borrow.value.slot;
        }
    }
    uint64_t core_owners,core_borrowed;
    NlFileValueStatus core=nl_file_values_live_slots(c->files,&core_owners,&core_borrowed);
    if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    if(rooted!=core_owners || borrowed!=core_borrowed)return frc_bad(c);
    uint64_t base=0;
    for(uint16_t d=0;d<c->frame_count;d++) {
        const FileRuntimeFrame *f=d+1==c->frame_count?current:&c->frames[d];
        FileRuntimeFrame layout;NvmFileHostedFunction fn;
        if(!frf_layout(c,f->function,base,d,&layout,&fn) || f->locals_base!=layout.locals_base ||
           f->stack_base!=layout.stack_base || f->staging_base!=layout.staging_base ||
           f->reference_base!=layout.reference_base || (d==0 && f->region_base))return frc_bad(c);
        if(d+1<c->frame_count) {
            NvmFileCodeInstruction call;
            if(!nvm_file_cyclic_hosted_instruction(c->cyclic_plan,f->function,(uint16_t)f->instruction,&call) ||
               call.decoded.operands[0].u32!=c->frames[d+1].function)return frc_bad(c);
        }
        base=(uint64_t)f->stack_base+(c->mode==NVM_FILE_RUNTIME_VM?f->stack_count:fn.operand_peak);
        uint32_t end=d+1==c->frame_count?c->region_count:c->frames[d+1].region_base;
        if((d+1<c->frame_count && !f->waiting) || !frc_frame(c,f,end))return frc_bad(c);
    }
    return NVM_FILE_RUNTIME_OK;
}
const NvmFileCyclicHostedPlan *nvm_file_runtime_cyclic_plan(const NvmFileRuntime *c) {
    return c && c->kind==FR_CYCLIC?c->cyclic_plan:NULL;
}
bool nvm_file_runtime_cyclic_frame_view(const NvmFileRuntime *c,NvmFileCyclicFrameView *out) {
    NvmFileRuntimeFrameView frame;
    if(!out || !c || c->kind!=FR_CYCLIC || !nvm_file_runtime_frame_view(c,&frame))return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    *out=(NvmFileCyclicFrameView){NVM_FILE_CYCLIC_RUNTIME_REVISION,frame,f->variant,f->instruction_open};return true;
}
NvmFileRuntimeStatus nvm_file_runtime_cyclic_enter(NvmFileRuntime *c) {
    if(!c || c->kind!=FR_CYCLIC)return NVM_FILE_RUNTIME_INVALID;
    NvmFileRuntimeStatus status=fr_ready(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    NvmFileHostedFunction fn;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    if(!frf_current(c,&fn,&in,&fact))return frc_bad(c);
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    if(f->instruction_open)return frc_bad(c);
    c->report.function=f->function;c->report.instruction=in.byte_offset;
    status=frc_validate(c,f);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(c->instructions_started>c->instruction_limit)return frc_bad(c);
    if(c->instructions_started==c->instruction_limit) {
        c->fuel_exhausted=true;return fr_error(c,NVM_FILE_RUNTIME_LIMIT,NL_FILE_VALUE_LIMIT);
    }
    c->instructions_started++;f->instruction_open=true;return NVM_FILE_RUNTIME_OK;
}
static NvmFileCyclicExecutionReport frc_report(const NvmFileRuntime *c,NvmFileRuntimeReport r) {
    return (NvmFileCyclicExecutionReport){NVM_FILE_CYCLIC_RUNTIME_REVISION,r,
        c?c->instruction_limit:0,c?c->instructions_started:0,c?c->fuel_exhausted:false};
}
bool nvm_file_runtime_cyclic_report(const NvmFileRuntime *c,NvmFileCyclicExecutionReport *out) {
    if(!c || !out || c->kind!=FR_CYCLIC || c->busy)return false;
    *out=frc_report(c,c->report);return true;
}
NvmFileCyclicExecutionReport nvm_file_runtime_cyclic_finish(NvmFileRuntime *c,NvmFileRuntimeView *out) {
    if(!c || c->kind!=FR_CYCLIC)return frc_report(NULL,fr_refused(NVM_FILE_RUNTIME_INVALID));
    return frc_report(c,fr_finish(c,out));
}
NvmFileCyclicExecutionReport nvm_file_runtime_cyclic_destroy(NvmFileRuntime **address,NvmFileRuntimeView *out) {
    if(!address || !*address || (*address)->kind!=FR_CYCLIC)return frc_report(NULL,fr_refused(NVM_FILE_RUNTIME_INVALID));
    NvmFileRuntime *c=*address;
    if(c->busy)return frc_report(c,fr_refused(NVM_FILE_RUNTIME_BUSY));
    NvmFileCyclicExecutionReport report=nvm_file_runtime_cyclic_finish(c,out);
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    nvm_file_cyclic_hosted_free(c->cyclic_plan);free(c);*address=NULL;return report;
}
bool nvm_file_runtime_cyclic_abi(uint32_t revision,size_t options,size_t report,size_t view,size_t frame) {
    return revision==NVM_FILE_CYCLIC_RUNTIME_REVISION && options==sizeof(NvmFileCyclicOptions) &&
        report==sizeof(NvmFileCyclicExecutionReport) && view==sizeof(NvmFileRuntimeView) &&
        frame==sizeof(NvmFileCyclicFrameView);
}

/* Revision 1: exact selected variants, charge-before-effects, invocation fuel,
 * checked call/return staging and cleanup-before-publication. */
bool nvm_file_runtime_cyclic_native_abi(uint32_t revision,size_t options,size_t report,
                                      size_t view,size_t frame,size_t value) {
    return revision==1u && frame==sizeof(NvmFileRuntimeFrameView) &&
        nvm_file_runtime_cyclic_abi(revision,options,report,value,view);
}
