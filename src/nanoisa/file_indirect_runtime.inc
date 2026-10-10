/* Owning translation unit: file_runtime.c. No old plan gains indirect authority. */
const NvmFileIndirectHostedPlan *nvm_file_runtime_indirect_plan(const NvmFileRuntime *c) {
    return c && c->kind==FR_INDIRECT?c->indirect_plan:NULL;
}
bool nvm_file_runtime_indirect_frame_view(const NvmFileRuntime *c,NvmFileIndirectFrameView *out) {
    NvmFileRuntimeFrameView frame;
    if(!out || !c || c->kind!=FR_INDIRECT || !nvm_file_runtime_frame_view(c,&frame))return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    *out=(NvmFileIndirectFrameView){NVM_FILE_INDIRECT_RUNTIME_REVISION,frame,f->variant,f->instruction_open};return true;
}
NvmFileRuntimeStatus nvm_file_runtime_indirect_enter(NvmFileRuntime *c) {
    if(!c || c->kind!=FR_INDIRECT)return NVM_FILE_RUNTIME_INVALID;
    return frc_enter(c);
}
NvmFileRuntimeStatus nvm_file_runtime_indirect_funcref(NvmFileRuntime *c,uint32_t dst) {
    if(!c || c->kind!=FR_INDIRECT)return NVM_FILE_RUNTIME_INVALID;
    NvmFileHostedFunction fn,target;NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    NvmFileRuntimeStatus status=frf_ready(c,&fn,&in,&fact);if(status!=NVM_FILE_RUNTIME_OK)return status;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    if(in.decoded.opcode!=OP_FUNCREF || !fr_function(c,in.decoded.operands[0].u32,&target) ||
       f->stack_count>=fn.operand_peak || dst!=f->stack_base+f->stack_count || !fr_empty(c,dst))return frf_bad(c);
    c->values[dst].view=fr_view(fr_scalar_type(TAG_FUNCTION));
    c->values[dst].callable_plan=c->indirect_plan;
    c->values[dst].callable_target=in.decoded.operands[0].u32;
    return NVM_FILE_RUNTIME_OK;
}
static NvmFileIndirectExecutionReport fri_report(const NvmFileRuntime *c,NvmFileRuntimeReport report) {
    return (NvmFileIndirectExecutionReport){NVM_FILE_INDIRECT_RUNTIME_REVISION,report,
        c?c->instruction_limit:0,c?c->instructions_started:0,c?c->fuel_exhausted:false};
}
NvmFileIndirectExecutionReport nvm_file_runtime_indirect_finish(NvmFileRuntime *c,NvmFileRuntimeView *out) {
    if(!c || c->kind!=FR_INDIRECT)return fri_report(NULL,fr_refused(NVM_FILE_RUNTIME_INVALID));
    return fri_report(c,fr_finish(c,out));
}
NvmFileIndirectExecutionReport nvm_file_runtime_indirect_destroy(NvmFileRuntime **address,NvmFileRuntimeView *out) {
    if(!address || !*address || (*address)->kind!=FR_INDIRECT)return fri_report(NULL,fr_refused(NVM_FILE_RUNTIME_INVALID));
    NvmFileRuntime *c=*address;
    if(c->busy)return fri_report(c,fr_refused(NVM_FILE_RUNTIME_BUSY));
    NvmFileIndirectExecutionReport report=nvm_file_runtime_indirect_finish(c,out);
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    nvm_file_indirect_hosted_free(c->indirect_plan);free(c);*address=NULL;return report;
}

/* Revision 1 binds the generated indirect surface to these exact C layouts. */
bool nvm_file_runtime_indirect_native_abi(uint32_t revision,size_t options,size_t report,
                                        size_t view,size_t frame,size_t value) {
    return revision==1u && options==sizeof(NvmFileIndirectOptions) &&
        report==sizeof(NvmFileIndirectExecutionReport) && view==sizeof(NvmFileIndirectFrameView) &&
        frame==sizeof(NvmFileRuntimeFrameView) && value==sizeof(NvmFileRuntimeView);
}
