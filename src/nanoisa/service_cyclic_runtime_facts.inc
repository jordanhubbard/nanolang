/* Owning translation unit: file_runtime.c. Copy common storage/declarations,
 * never convert cyclic authority into an old hosted plan. */
static bool fr_function(const SERVICE_TYPE(Runtime) *c,uint32_t f,SERVICE_TYPE(HostedFunction) *out) {
    if(c->kind==FR_ACYCLIC)return SERVICE_FN(hosted_function)(c->plan,f,out);
    if(c->kind==FR_INDIRECT) {
        SERVICE_TYPE(IndirectHostedFunction) v;
        if(!SERVICE_FN(indirect_hosted_function)(c->indirect_plan,f,&v) || v.entry_variant!=0)return false;
        *out=(SERVICE_TYPE(HostedFunction)){v.code,v.declared_stack,v.operand_peak,v.locals,
            v.staging_slots,v.frames,v.reference_slots,v.region_slots,v.vm_value_slots,v.native_value_slots};return true;
    }
    SERVICE_TYPE(CyclicHostedFunction) v;
    if(!SERVICE_FN(cyclic_hosted_function)(c->cyclic_plan,f,&v) || v.entry_variant!=0)return false;
    *out=(SERVICE_TYPE(HostedFunction)){v.code,v.declared_stack,v.operand_peak,v.locals,
        v.staging_slots,v.frames,v.reference_slots,v.region_slots,v.vm_value_slots,v.native_value_slots};return true;
}
static bool fr_local(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t local,SERVICE_TYPE(FlowDeclaration) *out) {
    if(c->kind==FR_INDIRECT)return SERVICE_FN(indirect_hosted_local)(c->indirect_plan,f,local,out);
    return c->kind==FR_ACYCLIC?SERVICE_FN(hosted_local)(c->plan,f,local,out):
        SERVICE_FN(cyclic_hosted_local)(c->cyclic_plan,f,local,out);
}
static bool fr_variant(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t i,uint8_t v,SERVICE_TYPE(CyclicVariant) *out) {
    return c->kind==FR_INDIRECT?SERVICE_FN(indirect_hosted_variant)(c->indirect_plan,f,i,v,out):
        SERVICE_FN(cyclic_hosted_variant)(c->cyclic_plan,f,i,v,out);
}
static bool fr_input_local(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t i,uint8_t v,uint16_t n,SERVICE_TYPE(FlowValue) *out) {
    return c->kind==FR_INDIRECT?SERVICE_FN(indirect_hosted_input_local)(c->indirect_plan,f,i,v,n,out):
        SERVICE_FN(cyclic_hosted_input_local)(c->cyclic_plan,f,i,v,n,out);
}
static bool fr_input_stack(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t i,uint8_t v,uint16_t n,SERVICE_TYPE(FlowValue) *out) {
    return c->kind==FR_INDIRECT?SERVICE_FN(indirect_hosted_input_stack)(c->indirect_plan,f,i,v,n,out):
        SERVICE_FN(cyclic_hosted_input_stack)(c->cyclic_plan,f,i,v,n,out);
}
static bool fr_input_reference(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t i,uint8_t v,uint16_t n,SERVICE_TYPE(FlowReference) *out) {
    return c->kind==FR_INDIRECT?SERVICE_FN(indirect_hosted_input_reference)(c->indirect_plan,f,i,v,n,out):
        SERVICE_FN(cyclic_hosted_input_reference)(c->cyclic_plan,f,i,v,n,out);
}
static bool fr_input_region(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t i,uint8_t v,uint16_t n,uint64_t *out) {
    return c->kind==FR_INDIRECT?SERVICE_FN(indirect_hosted_input_region)(c->indirect_plan,f,i,v,n,out):
        SERVICE_FN(cyclic_hosted_input_region)(c->cyclic_plan,f,i,v,n,out);
}
static bool fr_code(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t i,SERVICE_TYPE(CodeInstruction) *out) {
    return c->kind==FR_INDIRECT?SERVICE_FN(indirect_hosted_instruction)(c->indirect_plan,f,i,out):
        SERVICE_FN(cyclic_hosted_instruction)(c->cyclic_plan,f,i,out);
}
static bool fr_instruction(const SERVICE_TYPE(Runtime) *c,uint32_t f,uint16_t i,uint8_t variant,
                            SERVICE_TYPE(CodeInstruction) *in,SERVICE_TYPE(BodyInstruction) *fact) {
    if(c->kind==FR_ACYCLIC)return SERVICE_FN(hosted_instruction)(c->plan,f,i,in,fact);
    SERVICE_TYPE(CyclicVariant) v;
    if(!fr_code(c,f,i,in) ||
       !fr_variant(c,f,i,variant,&v))return false;
    *fact=v.body;return true;
}
static bool fr_import(const SERVICE_TYPE(Runtime) *c,uint32_t original,uint32_t *ordinal) {
    for(uint32_t i=0;i<RUNTIME_IMPORTS(c);i++) {
        uint32_t value;
        if(!(c->kind==FR_ACYCLIC?SERVICE_FN(hosted_import)(c->plan,i,&value):
             c->kind==FR_INDIRECT?SERVICE_FN(indirect_hosted_import)(c->indirect_plan,i,&value):
             SERVICE_FN(cyclic_hosted_import)(c->cyclic_plan,i,&value)))return false;
        if(value==original){*ordinal=i;return true;}
    }
    return false;
}
static bool frc_edge(const SERVICE_TYPE(Runtime) *c,const FileRuntimeFrame *f,uint8_t edge,uint8_t *out) {
    if(c->kind==FR_ACYCLIC){*out=0;return true;}
    SERVICE_TYPE(CyclicVariant) v;
    if(edge>=2 || !fr_variant(c,f->function,
       (uint16_t)f->instruction,f->variant,&v) || !(v.edge_mask&(1u<<edge)) ||
       v.edge_variants[edge]==SERVICE_CONST(CYCLIC_NO_VARIANT))return false;
    *out=v.edge_variants[edge];return true;
}
static bool fr_call_target(const SERVICE_TYPE(Runtime) *c,const FileRuntimeFrame *f,
                           const SERVICE_TYPE(CodeInstruction) *in,uint32_t target) {
    if(!isa_is_file_indirect_call(in->decoded.opcode))
        return (in->decoded.opcode==OP_CALL || in->decoded.opcode==OP_CALL_REF ||
                in->decoded.opcode==OP_FILE_CALL_REFS) && in->decoded.operands[0].u32==target;
    SERVICE_TYPE(IndirectFlowCall) call;
    return c->kind==FR_INDIRECT && target<SERVICE_CONST(FLOW_FUNCTIONS) &&
        SERVICE_FN(indirect_hosted_call)(c->indirect_plan,f->function,(uint16_t)f->instruction,f->variant,&call) &&
        call.candidates && call.candidates==call.checked_candidates &&
        (call.candidates&(UINT64_C(1)<<target)) && call.common.target==NVM_V2_NO_INDEX;
}
static SERVICE_TYPE(RuntimeStatus) frc_validate(SERVICE_TYPE(Runtime) *,const FileRuntimeFrame *);
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_cyclic_create)(const uint8_t *bytes,size_t size,
    SERVICE_TYPE(RuntimeMode) mode,const SERVICE_TYPE(CyclicOptions) *options,SERVICE_TYPE(Runtime) **out) {
    if(!out || !options || options->revision!=SERVICE_CONST(CYCLIC_RUNTIME_REVISION) ||
       options->instruction_limit>SERVICE_CONST(CYCLIC_FUEL_MAX) ||
       (mode!=SERVICE_CONST(RUNTIME_VM) && mode!=SERVICE_CONST(RUNTIME_NATIVE)))return SERVICE_CONST(RUNTIME_INVALID);
    SERVICE_TYPE(CyclicHostedPlan) *plan=NULL;
    SERVICE_TYPE(FlowStatus) checked=SERVICE_FN(cyclic_hosted_prepare)(bytes,size,&plan);
    if(checked!=SERVICE_CONST(FLOW_OK))return fr_prepare(checked);
    SERVICE_TYPE(CyclicHostedStartup) v;size_t core,total;
    if(!SERVICE_FN(cyclic_hosted_startup)(plan,&v) || !RUNTIME_BOUND(&core)) {
        SERVICE_FN(cyclic_hosted_free)(plan);return SERVICE_CONST(RUNTIME_INVALID);
    }
    uint64_t values=mode==SERVICE_CONST(RUNTIME_VM)?v.vm_value_slots:v.native_value_slots;
    total=v.retained_bound;
    if(!values || values>UINT32_MAX || !v.frames || !v.reference_slots || !v.region_slots ||
       !fr_add(&total,1,sizeof(SERVICE_TYPE(Runtime))) || !fr_add(&total,1,sizeof(FileRuntimeWitness)) ||
       !fr_add(&total,(size_t)values,sizeof(FileRuntimeValue)) ||
       !fr_add(&total,v.reference_slots,sizeof(FileRuntimeReference)) ||
       !fr_add(&total,v.region_slots,sizeof(uint64_t)) ||
       !fr_add(&total,v.frames,sizeof(FileRuntimeFrame)) || !fr_add(&total,1,core)) {
        SERVICE_FN(cyclic_hosted_free)(plan);return SERVICE_CONST(RUNTIME_LIMIT);
    }
    SERVICE_TYPE(Runtime) *c=calloc(1,sizeof *c);
    if(!c){SERVICE_FN(cyclic_hosted_free)(plan);return SERVICE_CONST(RUNTIME_MEMORY);}
    c->kind=FR_CYCLIC;c->cyclic_plan=plan;c->mode=mode;c->next_region=1;
    c->instruction_limit=options->instruction_limit;
    c->startup=(SERVICE_TYPE(HostedStartup)){v.entry,v.initializer,v.functions,v.features,
        v.vm_value_slots,v.native_value_slots,v.frames,v.reference_slots,v.region_slots,v.allocation_bound};
    c->storage=(SERVICE_TYPE(RuntimeStorage)){total,(uint32_t)values,v.reference_slots,v.region_slots,v.frames};
    c->witness=calloc(1,sizeof *c->witness);
    c->values=calloc(c->storage.values,sizeof *c->values);
    c->references=calloc(c->storage.references,sizeof *c->references);
    c->regions=calloc(c->storage.regions,sizeof *c->regions);
    c->frames=calloc(c->storage.frames,sizeof *c->frames);
    SERVICE_TYPE(RuntimeStatus) status=SERVICE_CONST(RUNTIME_MEMORY);
    if(!c->witness || !c->values || !c->references || !c->regions || !c->frames)goto fail;
    status=SERVICE_CONST(RUNTIME_INVALID);
    RUNTIME_LOAD_TYPES(c,SERVICE_FN(cyclic_hosted_type),plan);
    c->report.function=c->report.instruction=NVM_V2_NO_INDEX;
    c->current_root=v.initializer!=NVM_V2_NO_INDEX?v.initializer:v.entry;
    *out=c;return SERVICE_CONST(RUNTIME_OK);
fail:
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    SERVICE_FN(cyclic_hosted_free)(plan);free(c);return status;
}
