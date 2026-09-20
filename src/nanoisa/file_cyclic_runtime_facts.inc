/* Owning translation unit: file_runtime.c. Copy common storage/declarations,
 * never convert cyclic authority into an old hosted plan. */
static bool fr_function(const NvmFileRuntime *c,uint32_t f,NvmFileHostedFunction *out) {
    if(c->kind==FR_ACYCLIC)return nvm_file_hosted_function(c->plan,f,out);
    NvmFileCyclicHostedFunction v;
    if(!nvm_file_cyclic_hosted_function(c->cyclic_plan,f,&v) || v.entry_variant!=0)return false;
    *out=(NvmFileHostedFunction){v.code,v.declared_stack,v.operand_peak,v.locals,
        v.staging_slots,v.frames,v.reference_slots,v.region_slots,v.vm_value_slots,v.native_value_slots};return true;
}
static bool fr_local(const NvmFileRuntime *c,uint32_t f,uint16_t local,NvmFileFlowDeclaration *out) {
    return c->kind==FR_ACYCLIC?nvm_file_hosted_local(c->plan,f,local,out):
        nvm_file_cyclic_hosted_local(c->cyclic_plan,f,local,out);
}
static bool fr_instruction(const NvmFileRuntime *c,uint32_t f,uint16_t i,uint8_t variant,
                            NvmFileCodeInstruction *in,NvmFileBodyInstruction *fact) {
    if(c->kind==FR_ACYCLIC)return nvm_file_hosted_instruction(c->plan,f,i,in,fact);
    NvmFileCyclicVariant v;
    if(!nvm_file_cyclic_hosted_instruction(c->cyclic_plan,f,i,in) ||
       !nvm_file_cyclic_hosted_variant(c->cyclic_plan,f,i,variant,&v))return false;
    *fact=v.body;return true;
}
static bool fr_import(const NvmFileRuntime *c,uint32_t original,uint32_t *ordinal) {
    if(c->kind==FR_ACYCLIC)return nvm_file_hosted_import(c->plan,original,ordinal);
    for(uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        uint32_t value;
        if(!nvm_file_cyclic_hosted_import(c->cyclic_plan,i,&value))return false;
        if(value==original){*ordinal=i;return true;}
    }
    return false;
}
static bool frc_edge(const NvmFileRuntime *c,const FileRuntimeFrame *f,uint8_t edge,uint8_t *out) {
    if(c->kind==FR_ACYCLIC){*out=0;return true;}
    NvmFileCyclicVariant v;
    if(edge>=2 || !nvm_file_cyclic_hosted_variant(c->cyclic_plan,f->function,
       (uint16_t)f->instruction,f->variant,&v) || !(v.edge_mask&(1u<<edge)) ||
       v.edge_variants[edge]==NVM_FILE_CYCLIC_NO_VARIANT)return false;
    *out=v.edge_variants[edge];return true;
}
static NvmFileRuntimeStatus frc_validate(NvmFileRuntime *,const FileRuntimeFrame *);
NvmFileRuntimeStatus nvm_file_runtime_cyclic_create(const uint8_t *bytes,size_t size,
    NvmFileRuntimeMode mode,const NvmFileCyclicOptions *options,NvmFileRuntime **out) {
    if(!out || !options || options->revision!=NVM_FILE_CYCLIC_RUNTIME_REVISION ||
       options->instruction_limit>NVM_FILE_CYCLIC_FUEL_MAX ||
       (mode!=NVM_FILE_RUNTIME_VM && mode!=NVM_FILE_RUNTIME_NATIVE))return NVM_FILE_RUNTIME_INVALID;
    NvmFileCyclicHostedPlan *plan=NULL;
    NvmFileFlowStatus checked=nvm_file_cyclic_hosted_prepare(bytes,size,&plan);
    if(checked!=NVM_FILE_FLOW_OK)return fr_prepare(checked);
    NvmFileCyclicHostedStartup v;size_t core,total;
    if(!nvm_file_cyclic_hosted_startup(plan,&v) || !nl_file_values_storage_bound(&core)) {
        nvm_file_cyclic_hosted_free(plan);return NVM_FILE_RUNTIME_INVALID;
    }
    uint64_t values=mode==NVM_FILE_RUNTIME_VM?v.vm_value_slots:v.native_value_slots;
    total=v.retained_bound;
    if(!values || values>UINT32_MAX || !v.frames || !v.reference_slots || !v.region_slots ||
       !fr_add(&total,1,sizeof(NvmFileRuntime)) || !fr_add(&total,1,sizeof(FileRuntimeWitness)) ||
       !fr_add(&total,(size_t)values,sizeof(FileRuntimeValue)) ||
       !fr_add(&total,v.reference_slots,sizeof(FileRuntimeReference)) ||
       !fr_add(&total,v.region_slots,sizeof(uint64_t)) ||
       !fr_add(&total,v.frames,sizeof(FileRuntimeFrame)) || !fr_add(&total,1,core)) {
        nvm_file_cyclic_hosted_free(plan);return NVM_FILE_RUNTIME_LIMIT;
    }
    NvmFileRuntime *c=calloc(1,sizeof *c);
    if(!c){nvm_file_cyclic_hosted_free(plan);return NVM_FILE_RUNTIME_MEMORY;}
    c->kind=FR_CYCLIC;c->cyclic_plan=plan;c->mode=mode;c->next_region=1;
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
        if(!nvm_file_cyclic_hosted_type(plan,i,&type))goto fail;
        c->types[i]=(NvmFileFlowDeclaration){type.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
            0,type.global_index,type.catalog_ordinal,type.category};
    }
    c->report.function=c->report.instruction=NVM_V2_NO_INDEX;
    c->current_root=v.initializer!=NVM_V2_NO_INDEX?v.initializer:v.entry;
    *out=c;return NVM_FILE_RUNTIME_OK;
fail:
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    nvm_file_cyclic_hosted_free(plan);free(c);return status;
}
