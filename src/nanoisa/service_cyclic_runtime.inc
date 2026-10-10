/* Private cyclic carrier. No decoded dispatch or public admission. */
static bool frc_identity(CoreValue a,CoreValue b) {
    return RUNTIME_IDENTITY(a,b);
}
static bool frc_zero(CoreValue a) { return RUNTIME_ZERO(a); }
static SERVICE_TYPE(RuntimeStatus) frc_bad(SERVICE_TYPE(Runtime) *c) {
    return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
}
static bool frc_payload(const SERVICE_TYPE(Runtime) *c,const FileRuntimeValue *v) {
    const SERVICE_TYPE(RuntimeView) *p=&v->view;
    if(p->initialized && p->type.tag==TAG_FUNCTION)
        return c->kind==FR_INDIRECT && !p->owning && !p->formal && !p->fields &&
            p->arm==SERVICE_CONST(FLOW_ARM_NONE) && fr_same(p->type,fr_scalar_type(TAG_FUNCTION)) &&
            !v->reference && frc_zero(v->owner) && v->callable_plan==c->indirect_plan &&
            v->callable_target<c->startup.functions;
    if(v->callable_plan || v->callable_target)return false;
    if(!p->initialized) {
        if(!frc_zero(v->owner) || v->reference || p->owning || p->formal || p->fields ||
           p->arm || p->type.tag || p->type.mode || p->type.global_index ||
           p->type.catalog_ordinal || p->type.category)return false;
        for(uint8_t i=0;i<SERVICE_CONST(RUNTIME_FIELDS);i++)if(p->values[i])return false;
        return true;
    }
    if(p->owning)return RUNTIME_VALUE_MATCH(c,v->owner,p->type) && !p->formal && !p->fields && !v->reference;
    if(!frc_zero(v->owner))return false;
    if(p->formal)return (p->type.mode==1 || p->type.mode==2) && !p->fields;
    if(v->reference || p->type.mode)return false;
    if(p->type.tag==TAG_STRING)return p->fields==1 && p->arm==SERVICE_CONST(FLOW_ARM_NONE) &&
        fr_same(p->type,fr_scalar_type(TAG_STRING)) && fr_string_identity(c,p->values[0]);
    if(p->type.tag==TAG_INT)return p->fields==1 && p->arm==SERVICE_CONST(FLOW_ARM_NONE);
    if(p->type.tag==TAG_BOOL)return p->fields==1 && p->arm==SERVICE_CONST(FLOW_ARM_NONE) &&
        (p->values[0]==0 || p->values[0]==1);
    if(p->type.tag==TAG_VOID)return !p->fields && p->arm==SERVICE_CONST(FLOW_ARM_NONE);
    const NlServicePlanType *type=RUNTIME_CATALOG_TYPE(c,p->type.catalog_ordinal);
    if(!type)return false;
    if(p->type.category==SERVICE_CONST(CATEGORY_RECORD)) {
        if(p->fields!=type->member_count || p->fields>SERVICE_CONST(RUNTIME_FIELDS) ||
           p->arm!=SERVICE_CONST(FLOW_ARM_NONE))return false;
        for(uint8_t i=0;i<p->fields;i++) {
            const char *id=type->members[i].type_id;
            if(!id)return false;
            if(!strcmp(id,"nsi:core/bool")) {if(p->values[i]!=0 && p->values[i]!=1)return false;}
            else if(!strcmp(id,"nsi:core/string")) {if(!fr_string_identity(c,p->values[i]))return false;}
            else if(strcmp(id,"nsi:core/int"))return false;
        }
        return true;
    }
    if(p->type.category!=SERVICE_CONST(CATEGORY_SCALAR_RESULT) ||
       (p->arm!=SERVICE_CONST(FLOW_ARM_OK) && p->arm!=SERVICE_CONST(FLOW_ARM_ERROR)))return false;
    return RUNTIME_SCALAR_PAYLOAD(c,p);
}
static bool frc_value(SERVICE_TYPE(Runtime) *c,uint32_t root,SERVICE_TYPE(FlowValue) expected) {
    if(root>=c->storage.values)return false;
    const FileRuntimeValue *v=&c->values[root];
    if(!frc_payload(c,v) || v->view.initialized!=expected.initialized)return false;
    if(!expected.initialized)return !expected.owner;
    if(!frf_matches(v->view,expected.type))return false;
    SERVICE_TYPE(FlowArm) actual=v->view.arm;
    if(v->view.owning) {
        bool open=v->view.type.category==NOMINAL_CATEGORY_OWNER_RESULT;
        if(CORE_FN(value_validate)(c->files,&v->owner,open)!=CORE_STATUS(OK))return false;
        if(open) {
            CoreOpenView view;
            if(CORE_FN(open_view)(c->files,&v->owner,&view)!=CORE_STATUS(OK))return false;
            actual=view.ok?SERVICE_CONST(FLOW_ARM_OK):SERVICE_CONST(FLOW_ARM_ERROR);
        }
        if(!expected.owner || expected.owner>SERVICE_CONST(FLOW_OWNERS))return false;
        uint32_t index=(uint32_t)expected.owner-1;
        if(!frc_zero(c->witness->owners[index]))return false;
        for(uint32_t i=0;i<SERVICE_CONST(FLOW_OWNERS);i++)
            if(!frc_zero(c->witness->owners[i]) && frc_identity(c->witness->owners[i],v->owner))return false;
        c->witness->owners[index]=v->owner;c->witness->roots[index]=root;
    } else if(expected.owner)return false;
    return expected.arm==SERVICE_CONST(FLOW_ARM_UNKNOWN)?
        (actual==SERVICE_CONST(FLOW_ARM_OK) || actual==SERVICE_CONST(FLOW_ARM_ERROR)):actual==expected.arm;
}
static bool frc_reference_empty(const FileRuntimeReference *r) {
    return !r->live && !r->formal && !r->shared && !r->owner_root && !r->origin && !r->formal_root &&
        !r->region && !RUNTIME_BORROW_EPOCH(r->borrow) && frc_zero(RUNTIME_BORROW_VALUE(r->borrow));
}
static bool frc_frame(SERVICE_TYPE(Runtime) *c,const FileRuntimeFrame *f,uint32_t region_end) {
    SERVICE_TYPE(CyclicVariant) fact;SERVICE_TYPE(HostedFunction) fn;
    if(!fr_function(c,f->function,&fn) || f->instruction>UINT16_MAX ||
       !fr_variant(c,f->function,(uint16_t)f->instruction,f->variant,&fact) ||
       !fact.body.reachable || fact.input.locals!=fn.locals || f->stack_count>fn.operand_peak ||
       (!f->waiting && f->stack_count!=fact.input.stack) ||
       (f->waiting && f->stack_count>fact.input.stack) ||
       !frf_range(c,f->locals_base,fn.locals) || !frf_range(c,f->stack_base,f->stack_count) ||
       !frf_empty_range(c,f->staging_base,fn.staging_slots) ||
       f->reference_base>c->storage.references || SERVICE_CONST(FLOW_REFERENCES)>c->storage.references-f->reference_base ||
       f->region_base>region_end || region_end>c->region_count ||
       region_end-f->region_base!=fact.input.regions)return false;
    if(f->waiting) {
        SERVICE_TYPE(CodeInstruction) in;
        if(!fr_code(c,f->function,(uint16_t)f->instruction,&in) ||
           !fr_call_target(c,f,&in,f->selected_target) || !fact.body.has_obligation ||
           (uint32_t)f->stack_count+fact.body.obligation.parameters-fact.body.obligation.borrowed_inputs+
               (isa_is_file_indirect_call(in.decoded.opcode)?1u:0u)!=fact.input.stack)
            return false;
    }
    memset(c->witness,0,sizeof *c->witness);
    for(uint16_t i=0;i<fn.locals;i++) {
        SERVICE_TYPE(FlowValue) v;
        if(!fr_input_local(c,f->function,(uint16_t)f->instruction,f->variant,i,&v) ||
           !frc_value(c,f->locals_base+i,v))return false;
    }
    for(uint16_t i=0;i<f->stack_count;i++) {
        SERVICE_TYPE(FlowValue) v;
        if(!fr_input_stack(c,f->function,(uint16_t)f->instruction,f->variant,i,&v) ||
           !frc_value(c,f->stack_base+i,v))return false;
    }
    for(uint16_t i=0;i<fact.input.regions;i++) {
        uint64_t abstract;
        if(!fr_input_region(c,f->function,(uint16_t)f->instruction,f->variant,i,&abstract) ||
           abstract!=UINT64_C(513)+i || !c->regions[f->region_base+i])return false;
    }
    for(uint16_t i=0;i<SERVICE_CONST(FLOW_REFERENCES);i++) {
        SERVICE_TYPE(FlowReference) expected;const FileRuntimeReference *r=&c->references[f->reference_base+i];
        if(!fr_input_reference(c,f->function,(uint16_t)f->instruction,f->variant,i,&expected))return false;
        if(!expected.live){if(!frc_reference_empty(r))return false;continue;}
        if(!r->live || expected.formal!=r->formal || expected.shared!=r->shared || !fr_reference(c,f->reference_base+i) ||
           CORE_FN(value_borrow_validate)(c->files,&r->borrow)!=CORE_STATUS(OK) ||
           !frc_identity(RUNTIME_BORROW_VALUE(r->borrow),c->values[r->owner_root].owner))return false;
        const FileRuntimeReference *origin=&c->references[r->origin];
        if(!frc_identity(RUNTIME_BORROW_VALUE(origin->borrow),RUNTIME_BORROW_VALUE(r->borrow)) || RUNTIME_BORROW_EPOCH(origin->borrow)!=RUNTIME_BORROW_EPOCH(r->borrow))return false;
        if(expected.formal) {
            if(i>=fn.code.declaration.parameters || expected.local!=i || expected.owner!=UINT64_C(257)+i ||
               expected.identity!=expected.owner || expected.region || r->region ||
               r->formal_root!=f->locals_base+i || !c->values[r->formal_root].view.formal ||
               c->values[r->formal_root].reference!=f->reference_base+i || r->origin>=f->reference_base)return false;
        } else {
            if(!expected.owner || expected.owner>SERVICE_CONST(FLOW_OWNERS) || expected.local>=fn.locals ||
               expected.identity!=UINT64_C(769)+i || expected.region<513 ||
               expected.region-513>=fact.input.regions || (!r->shared && r->origin!=f->reference_base+i) ||
               r->formal_root!=SERVICE_CONST(RUNTIME_NO_SLOT) || r->owner_root!=f->locals_base+expected.local ||
               c->witness->roots[expected.owner-1]!=r->owner_root ||
               !frc_identity(c->witness->owners[expected.owner-1],RUNTIME_BORROW_VALUE(r->borrow)) ||
               r->region!=c->regions[f->region_base+(uint32_t)(expected.region-513)])return false;
        }
    }
    return true;
}
static SERVICE_TYPE(RuntimeStatus) frc_validate(SERVICE_TYPE(Runtime) *c,const FileRuntimeFrame *current) {
    if(c->kind==FR_ACYCLIC || !c->witness || !c->frame_count ||
       c->frame_count>c->storage.frames || c->region_count>c->storage.regions || !c->acquired || !c->files)return frc_bad(c);
    uint64_t rooted[RUNTIME_MAX_INSTANCES]={0},borrowed[RUNTIME_MAX_INSTANCES]={0};
    /* Every owning root, including waiting callers, has a unique live core slot.
     * The full generation/type check prevents stale same-slot aliases. */
    for(uint32_t root=0;root<c->storage.values;root++) {
        FileRuntimeValue *v=&c->values[root];
        if(!frc_payload(c,v))return frc_bad(c);
        if(v->view.initialized) {
            bool covered=false;
            for(uint16_t d=0;d<c->frame_count;d++) {
                const FileRuntimeFrame *f=d+1==c->frame_count?current:&c->frames[d];
                SERVICE_TYPE(HostedFunction) fn;if(!fr_function(c,f->function,&fn))return frc_bad(c);
                if((root>=f->locals_base && root-f->locals_base<fn.locals) ||
                   (root>=f->stack_base && root-f->stack_base<f->stack_count))covered=true;
            }
            if(!covered)return frc_bad(c);
        }
        if(v->view.initialized && v->view.owning) {
            uint32_t instance=RUNTIME_CORE_INDEX(v->owner),slot=RUNTIME_CORE_SLOT(v->owner);
            if(instance>=RUNTIME_INSTANCES(c) || slot>=64 || (rooted[instance]&(UINT64_C(1)<<slot)))return frc_bad(c);
            CoreStatus checked=CORE_FN(value_validate)(c->files,&v->owner,
                v->view.type.category==NOMINAL_CATEGORY_OWNER_RESULT);
            if(checked!=CORE_STATUS(OK))return fr_core(c,checked);
            rooted[instance]|=UINT64_C(1)<<slot;
        }
    }
    for(uint32_t i=0;i<c->region_count;i++) {
        if(!c->regions[i])return frc_bad(c);
        for(uint32_t j=0;j<i;j++)if(c->regions[j]==c->regions[i])return frc_bad(c);
    }
    for(uint32_t i=c->region_count;i<c->storage.regions;i++)if(c->regions[i])return frc_bad(c);
    for(uint32_t i=(uint32_t)c->frame_count*SERVICE_CONST(FLOW_REFERENCES);i<c->storage.references;i++)
        if(!frc_reference_empty(&c->references[i]))return frc_bad(c);
    for(uint32_t i=0;i<c->storage.references;i++) {
        const FileRuntimeReference *r=&c->references[i];
        if(r->live && !r->formal && r->origin==i) {
            CoreValue owner=RUNTIME_BORROW_VALUE(r->borrow);
            uint32_t instance=RUNTIME_CORE_INDEX(owner),slot=RUNTIME_CORE_SLOT(owner);
            if(instance>=RUNTIME_INSTANCES(c) || slot>=64 || (borrowed[instance]&(UINT64_C(1)<<slot)))return frc_bad(c);
            borrowed[instance]|=UINT64_C(1)<<slot;
        }
    }
    for(uint32_t i=0;i<RUNTIME_INSTANCES(c);i++) {
        uint64_t core_owners,core_borrowed;
        CoreStatus core=RUNTIME_LIVE_SLOTS(c,i,&core_owners,&core_borrowed);
        if(core!=CORE_STATUS(OK))return fr_core(c,core);
        if(rooted[i]!=core_owners || borrowed[i]!=core_borrowed)return frc_bad(c);
    }
    uint64_t base=0;
    for(uint16_t d=0;d<c->frame_count;d++) {
        const FileRuntimeFrame *f=d+1==c->frame_count?current:&c->frames[d];
        FileRuntimeFrame layout;SERVICE_TYPE(HostedFunction) fn;
        if(!frf_layout(c,f->function,base,d,&layout,&fn) || f->locals_base!=layout.locals_base ||
           f->stack_base!=layout.stack_base || f->staging_base!=layout.staging_base ||
           f->reference_base!=layout.reference_base || (d==0 && f->region_base))return frc_bad(c);
        if(d+1<c->frame_count) {
            SERVICE_TYPE(CodeInstruction) call;
            if(!fr_code(c,f->function,(uint16_t)f->instruction,&call) ||
               !fr_call_target(c,f,&call,c->frames[d+1].function) ||
               f->selected_target!=c->frames[d+1].function)return frc_bad(c);
        }
        base=(uint64_t)f->stack_base+(c->mode==SERVICE_CONST(RUNTIME_VM)?f->stack_count:fn.operand_peak);
        uint32_t end=d+1==c->frame_count?c->region_count:c->frames[d+1].region_base;
        if((d+1<c->frame_count && !f->waiting) || !frc_frame(c,f,end))return frc_bad(c);
    }
    return SERVICE_CONST(RUNTIME_OK);
}
const SERVICE_TYPE(CyclicHostedPlan) *SERVICE_FN(runtime_cyclic_plan)(const SERVICE_TYPE(Runtime) *c) {
    return c && c->kind==FR_CYCLIC?c->cyclic_plan:NULL;
}
bool SERVICE_FN(runtime_cyclic_frame_view)(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(CyclicFrameView) *out) {
    SERVICE_TYPE(RuntimeFrameView) frame;
    if(!out || !c || c->kind!=FR_CYCLIC || !SERVICE_FN(runtime_frame_view)(c,&frame))return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    *out=(SERVICE_TYPE(CyclicFrameView)){SERVICE_CONST(CYCLIC_RUNTIME_REVISION),frame,f->variant,f->instruction_open};return true;
}
static SERVICE_TYPE(RuntimeStatus) frc_enter(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(RuntimeStatus) status=fr_ready(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    SERVICE_TYPE(HostedFunction) fn;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    if(!frf_current(c,&fn,&in,&fact))return frc_bad(c);
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    if(f->instruction_open)return frc_bad(c);
    c->report.function=f->function;c->report.instruction=in.byte_offset;
    status=frc_validate(c,f);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(c->instructions_started>c->instruction_limit)return frc_bad(c);
    if(c->instructions_started==c->instruction_limit) {
        c->fuel_exhausted=true;return fr_error(c,SERVICE_CONST(RUNTIME_LIMIT),CORE_STATUS(LIMIT));
    }
    c->instructions_started++;f->instruction_open=true;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_cyclic_enter)(SERVICE_TYPE(Runtime) *c) {
    if(!c || c->kind!=FR_CYCLIC)return SERVICE_CONST(RUNTIME_INVALID);
    return frc_enter(c);
}
static SERVICE_TYPE(CyclicExecutionReport) frc_report(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeReport) r) {
    return (SERVICE_TYPE(CyclicExecutionReport)){SERVICE_CONST(CYCLIC_RUNTIME_REVISION),r,
        c?c->instruction_limit:0,c?c->instructions_started:0,c?c->fuel_exhausted:false};
}
bool SERVICE_FN(runtime_cyclic_report)(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(CyclicExecutionReport) *out) {
    if(!c || !out || c->kind!=FR_CYCLIC || c->busy)return false;
    *out=frc_report(c,c->report);return true;
}
SERVICE_TYPE(CyclicExecutionReport) SERVICE_FN(runtime_cyclic_finish)(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeView) *out) {
    if(!c || c->kind!=FR_CYCLIC)return frc_report(NULL,fr_refused(SERVICE_CONST(RUNTIME_INVALID)));
    return frc_report(c,fr_finish(c,out));
}
SERVICE_TYPE(CyclicExecutionReport) SERVICE_FN(runtime_cyclic_destroy)(SERVICE_TYPE(Runtime) **address,SERVICE_TYPE(RuntimeView) *out) {
    if(!address || !*address || (*address)->kind!=FR_CYCLIC)return frc_report(NULL,fr_refused(SERVICE_CONST(RUNTIME_INVALID)));
    SERVICE_TYPE(Runtime) *c=*address;
    if(c->busy)return frc_report(c,fr_refused(SERVICE_CONST(RUNTIME_BUSY)));
    SERVICE_TYPE(CyclicExecutionReport) report=SERVICE_FN(runtime_cyclic_finish)(c,out);
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    SERVICE_FN(cyclic_hosted_free)(c->cyclic_plan);free(c);*address=NULL;return report;
}
bool SERVICE_FN(runtime_cyclic_abi)(uint32_t revision,size_t options,size_t report,size_t view,size_t frame) {
    return revision==SERVICE_CONST(CYCLIC_RUNTIME_REVISION) && options==sizeof(SERVICE_TYPE(CyclicOptions)) &&
        report==sizeof(SERVICE_TYPE(CyclicExecutionReport)) && view==sizeof(SERVICE_TYPE(RuntimeView)) &&
        frame==sizeof(SERVICE_TYPE(CyclicFrameView));
}

/* Revision 1: exact selected variants, charge-before-effects, invocation fuel,
 * checked call/return staging and cleanup-before-publication. */
bool SERVICE_FN(runtime_cyclic_native_abi)(uint32_t revision,size_t options,size_t report,
                                      size_t view,size_t frame,size_t value) {
    return revision==1u && frame==sizeof(SERVICE_TYPE(RuntimeFrameView)) &&
        SERVICE_FN(runtime_cyclic_abi)(revision,options,report,value,view);
}
