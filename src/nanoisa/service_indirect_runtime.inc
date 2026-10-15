/* Owning translation unit: file_runtime.c. No old plan gains indirect authority. */
const SERVICE_TYPE(IndirectHostedPlan) *SERVICE_FN(runtime_indirect_plan)(const SERVICE_TYPE(Runtime) *c) {
    return c && c->kind==FR_INDIRECT?c->indirect_plan:NULL;
}
bool SERVICE_FN(runtime_indirect_frame_view)(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(IndirectFrameView) *out) {
    SERVICE_TYPE(RuntimeFrameView) frame;
    if(!out || !c || c->kind!=FR_INDIRECT || !SERVICE_FN(runtime_frame_view)(c,&frame))return false;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    *out=(SERVICE_TYPE(IndirectFrameView)){SERVICE_CONST(INDIRECT_RUNTIME_REVISION),frame,f->variant,f->instruction_open};return true;
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_indirect_enter)(SERVICE_TYPE(Runtime) *c) {
    if(!c || c->kind!=FR_INDIRECT)return SERVICE_CONST(RUNTIME_INVALID);
    return frc_enter(c);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_indirect_funcref)(SERVICE_TYPE(Runtime) *c,uint32_t dst) {
    if(!c || c->kind!=FR_INDIRECT)return SERVICE_CONST(RUNTIME_INVALID);
    SERVICE_TYPE(HostedFunction) fn,target;SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    SERVICE_TYPE(RuntimeStatus) status=frf_ready(c,&fn,&in,&fact);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    const FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    if(in.decoded.opcode!=OP_FUNCREF || !fr_function(c,in.decoded.operands[0].u32,&target) ||
       f->stack_count>=fn.operand_peak || dst!=f->stack_base+f->stack_count || !fr_empty(c,dst))return frf_bad(c);
    c->values[dst].view=fr_view(fr_scalar_type(TAG_FUNCTION));
    c->values[dst].callable_plan=c->indirect_plan;
    c->values[dst].callable_target=in.decoded.operands[0].u32;
    return SERVICE_CONST(RUNTIME_OK);
}
static SERVICE_TYPE(IndirectExecutionReport) fri_report(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeReport) report) {
    return (SERVICE_TYPE(IndirectExecutionReport)){SERVICE_CONST(INDIRECT_RUNTIME_REVISION),report,
        c?c->instruction_limit:0,c?c->instructions_started:0,c?c->fuel_exhausted:false};
}
SERVICE_TYPE(IndirectExecutionReport) SERVICE_FN(runtime_indirect_finish)(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeView) *out) {
    if(!c || c->kind!=FR_INDIRECT)return fri_report(NULL,fr_refused(SERVICE_CONST(RUNTIME_INVALID)));
    return fri_report(c,fr_finish(c,out));
}
SERVICE_TYPE(IndirectExecutionReport) SERVICE_FN(runtime_indirect_destroy)(SERVICE_TYPE(Runtime) **address,SERVICE_TYPE(RuntimeView) *out) {
    if(!address || !*address || (*address)->kind!=FR_INDIRECT)return fri_report(NULL,fr_refused(SERVICE_CONST(RUNTIME_INVALID)));
    SERVICE_TYPE(Runtime) *c=*address;
    if(c->busy)return fri_report(c,fr_refused(SERVICE_CONST(RUNTIME_BUSY)));
    SERVICE_TYPE(IndirectExecutionReport) report=SERVICE_FN(runtime_indirect_finish)(c,out);
    free(c->frames);free(c->regions);free(c->references);free(c->values);free(c->witness);
    SERVICE_FN(indirect_hosted_free)(c->indirect_plan);free(c);*address=NULL;return report;
}

/* Revision 1 binds the generated indirect surface to these exact C layouts. */
bool SERVICE_FN(runtime_indirect_native_abi)(uint32_t revision,size_t options,size_t report,
                                        size_t view,size_t frame,size_t value) {
    return revision==1u && options==sizeof(SERVICE_TYPE(IndirectOptions)) &&
        report==sizeof(SERVICE_TYPE(IndirectExecutionReport)) && view==sizeof(SERVICE_TYPE(IndirectFrameView)) &&
        frame==sizeof(SERVICE_TYPE(RuntimeFrameView)) && value==sizeof(SERVICE_TYPE(RuntimeView));
}
