NvmFileRuntimeStatus nvm_file_runtime_indirect_create(const uint8_t *bytes,size_t size,
    NvmFileRuntimeMode mode,const NvmFileIndirectOptions *options,NvmFileRuntime **out) {
    if(!out || !options || options->revision!=NVM_FILE_INDIRECT_RUNTIME_REVISION ||
       options->instruction_limit>NVM_FILE_INDIRECT_FUEL_MAX ||
       (mode!=NVM_FILE_RUNTIME_VM && mode!=NVM_FILE_RUNTIME_NATIVE))return NVM_FILE_RUNTIME_INVALID;
    NvmFileIndirectHostedPlan *plan=NULL;
    NvmFileFlowStatus checked=nvm_file_indirect_hosted_prepare(bytes,size,&plan);
    if(checked!=NVM_FILE_FLOW_OK)return fr_prepare(checked);
    NvmFileIndirectHostedStartup v;size_t core,total;
    if(!nvm_file_indirect_hosted_startup(plan,&v) || !nl_file_values_storage_bound(&core)) {
        nvm_file_indirect_hosted_free(plan);return NVM_FILE_RUNTIME_INVALID;
    }
    uint64_t values=mode==NVM_FILE_RUNTIME_VM?v.vm_value_slots:v.native_value_slots;
    total=v.retained_bound;
    if(!values || values>UINT32_MAX || !v.frames || !v.reference_slots || !v.region_slots ||
       !fr_add(&total,1,sizeof(NvmFileRuntime)) || !fr_add(&total,1,sizeof(FileRuntimeWitness)) ||
       !fr_add(&total,(size_t)values,sizeof(FileRuntimeValue)) ||
       !fr_add(&total,v.reference_slots,sizeof(FileRuntimeReference)) ||
       !fr_add(&total,v.region_slots,sizeof(uint64_t)) ||
       !fr_add(&total,v.frames,sizeof(FileRuntimeFrame)) || !fr_add(&total,1,core)) {
        nvm_file_indirect_hosted_free(plan);return NVM_FILE_RUNTIME_LIMIT;
    }
    NvmFileRuntime *c=calloc(1,sizeof *c);
    if(!c){nvm_file_indirect_hosted_free(plan);return NVM_FILE_RUNTIME_MEMORY;}
    c->kind=FR_INDIRECT;c->indirect_plan=plan;c->mode=mode;c->next_region=1;
    c->instruction_limit=options->instruction_limit;
    c->startup=(NvmFileHostedStartup){v.entry,v.initializer,v.functions,v.features,
        v.vm_value_slots,v.native_value_slots,v.frames,v.reference_slots,v.region_slots,v.allocation_bound};
    c->storage=(NvmFileRuntimeStorage){total,(uint32_t)values,v.reference_slots,v.region_slots,v.frames};
    c->witness=calloc(1,sizeof *c->witness);
    c->values=calloc(c->storage.values,sizeof *c->values);
    c->references=calloc(c->storage.references,sizeof *c->references);
    c->regions=calloc(c->storage.regions,sizeof *c->regions);
    c->frames=calloc(c->storage.frames,sizeof *c->frames);
    NvmFileRuntimeStatus status=NVM_FILE_RUNTIME_MEMORY;
    if(!c->witness || !c->values || !c->references || !c->regions || !c->frames)goto fail;
    status=NVM_FILE_RUNTIME_INVALID;
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        NvmFileNominalLayout type;
        if(!nvm_file_indirect_hosted_type(plan,i,&type))goto fail;
        c->types[i]=(NvmFileFlowDeclaration){type.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
            0,type.global_index,type.catalog_ordinal,type.category};
    }
    c->report.function=c->report.instruction=NVM_V2_NO_INDEX;
    c->current_root=v.initializer!=NVM_V2_NO_INDEX?v.initializer:v.entry;
    *out=c;return NVM_FILE_RUNTIME_OK;
fail:
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    nvm_file_indirect_hosted_free(plan);free(c);return status;
}
