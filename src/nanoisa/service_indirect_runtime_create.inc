SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_indirect_create)(const uint8_t *bytes,size_t size,
    SERVICE_TYPE(RuntimeMode) mode,const SERVICE_TYPE(IndirectOptions) *options,SERVICE_TYPE(Runtime) **out) {
    if(!out || !options || options->revision!=SERVICE_CONST(INDIRECT_RUNTIME_REVISION) ||
       options->instruction_limit>SERVICE_CONST(INDIRECT_FUEL_MAX) ||
       (mode!=SERVICE_CONST(RUNTIME_VM) && mode!=SERVICE_CONST(RUNTIME_NATIVE)))return SERVICE_CONST(RUNTIME_INVALID);
    SERVICE_TYPE(IndirectHostedPlan) *plan=NULL;
    SERVICE_TYPE(FlowStatus) checked=SERVICE_FN(indirect_hosted_prepare)(bytes,size,&plan);
    if(checked!=SERVICE_CONST(FLOW_OK))return fr_prepare(checked);
    SERVICE_TYPE(IndirectHostedStartup) v;size_t core,total;
    if(!SERVICE_FN(indirect_hosted_startup)(plan,&v) || !CORE_FN(values_storage_bound)(&core)) {
        SERVICE_FN(indirect_hosted_free)(plan);return SERVICE_CONST(RUNTIME_INVALID);
    }
    uint64_t values=mode==SERVICE_CONST(RUNTIME_VM)?v.vm_value_slots:v.native_value_slots;
    total=v.retained_bound;
    if(!values || values>UINT32_MAX || !v.frames || !v.reference_slots || !v.region_slots ||
       !fr_add(&total,1,sizeof(SERVICE_TYPE(Runtime))) || !fr_add(&total,1,sizeof(FileRuntimeWitness)) ||
       !fr_add(&total,(size_t)values,sizeof(FileRuntimeValue)) ||
       !fr_add(&total,v.reference_slots,sizeof(FileRuntimeReference)) ||
       !fr_add(&total,v.region_slots,sizeof(uint64_t)) ||
       !fr_add(&total,v.frames,sizeof(FileRuntimeFrame)) || !fr_add(&total,1,core)) {
        SERVICE_FN(indirect_hosted_free)(plan);return SERVICE_CONST(RUNTIME_LIMIT);
    }
    SERVICE_TYPE(Runtime) *c=calloc(1,sizeof *c);
    if(!c){SERVICE_FN(indirect_hosted_free)(plan);return SERVICE_CONST(RUNTIME_MEMORY);}
    c->kind=FR_INDIRECT;c->indirect_plan=plan;c->mode=mode;c->next_region=1;
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
    for(uint32_t i=0;i<NOMINAL_TYPES;i++) {
        NominalLayout type;
        if(!SERVICE_FN(indirect_hosted_type)(plan,i,&type))goto fail;
        c->types[i]=(SERVICE_TYPE(FlowDeclaration)){type.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
            0,type.global_index,type.catalog_ordinal,type.category};
    }
    c->report.function=c->report.instruction=NVM_V2_NO_INDEX;
    c->current_root=v.initializer!=NVM_V2_NO_INDEX?v.initializer:v.entry;
    *out=c;return SERVICE_CONST(RUNTIME_OK);
fail:
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    SERVICE_FN(indirect_hosted_free)(plan);free(c);return status;
}
