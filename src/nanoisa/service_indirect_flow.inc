struct SERVICE_TYPE(IndirectFlow) {
 SERVICE_TYPE(IndirectTargets) *targets;
 SERVICE_TYPE(CyclicReport) *ownership;
 SERVICE_TYPE(IndirectFlowSummary) summary;
};
typedef struct {
 const SERVICE_TYPE(IndirectTargets) *targets;
 uint64_t bits[SERVICE_CONST(CODE_INSTRUCTIONS)];
 uint32_t applications;
 SERVICE_TYPE(FlowState) saved,first;
 FileCyclicState canonical_first;
 FileCyclicWorkspace candidate;
} FileIndirectFlowScratch;
static bool fif_state_metadata_equal(const SERVICE_TYPE(FlowState) *a,const SERVICE_TYPE(FlowState) *b) {
 if(a->family!=b->family || a->reachable!=b->reachable ||
    a->cleanup_obligations!=b->cleanup_obligations || a->obligation_count!=b->obligation_count)return false;
 for(uint16_t i=0;i<a->obligation_count;i++)if(!obligation_same(a->obligations[i],b->obligations[i]))return false;
 return true;
}
static SERVICE_TYPE(FlowStatus) fif_transfer(void *context,SERVICE_TYPE(BodyReport) *body,SERVICE_TYPE(FlowState) *s,
                                      const SERVICE_TYPE(CodeInstruction) *in,SERVICE_TYPE(BodyInstruction) *fact) {
 FileIndirectFlowScratch *w=context;
 if(in->decoded.opcode==OP_FUNCREF) {
  if(in->decoded.operands[0].u32>=body->plan->count)return SERVICE_CONST(FLOW_INVALID);
  if(s->stack_count==SERVICE_CONST(FLOW_STACK))return SERVICE_CONST(FLOW_LIMIT);
  s->stack[s->stack_count++]=initial_value(scalar_type(TAG_FUNCTION),true,0);
  return SERVICE_CONST(FLOW_OK);
 }
 if(!isa_is_file_indirect_call(in->decoded.opcode))return file_body_transfer(body,s,in,fact);
 if(!s->stack_count)return SERVICE_CONST(FLOW_INVALID);
 SERVICE_TYPE(FlowValue) callable=s->stack[s->stack_count-1];
 if(!callable.initialized || !same(callable.type,scalar_type(TAG_FUNCTION)) ||
    callable.owner || callable.arm!=SERVICE_CONST(FLOW_ARM_NONE))return SERVICE_CONST(FLOW_INVALID);
 /* The instruction pointer belongs to this second checked plan. The target map
  * is indexed only after the original function/PC/index identities agree. */
 size_t index=(size_t)(in-body->plan->instructions);
 if(index>=body->plan->instruction_count || !w->bits[index])return SERVICE_CONST(FLOW_UNRESOLVED);
 const SERVICE_TYPE(CodePlan) *original=w->targets->plan;
 uint32_t caller=s->function;
 if(caller>=original->count || caller>=body->plan->count ||
    original->instruction_count!=body->plan->instruction_count ||
    original->starts[caller]!=body->plan->starts[caller] ||
    index<original->starts[caller] ||
    index-original->starts[caller]>=original->functions[caller].instruction_count ||
    original->functions[caller].instruction_count!=body->plan->functions[caller].instruction_count ||
    original->instructions[index].byte_offset!=in->byte_offset ||
    original->instructions[index].decoded.opcode!=in->decoded.opcode)return SERVICE_CONST(FLOW_INVALID);
 uint64_t candidates=w->bits[index];
 pop(s);w->saved=*s;
 uint64_t identity=s->declarations->next_identity,first_identity=identity;
 bool first=true;SERVICE_TYPE(FlowObligation) common={0};
 for(uint32_t f=0;f<body->plan->count;f++)if(candidates&(UINT64_C(1)<<f)) {
  if(!body->checked[f])return SERVICE_CONST(FLOW_UNRESOLVED);
  if(w->applications==SERVICE_CONST(INDIRECT_FLOW_APPLICATIONS))return SERVICE_CONST(FLOW_LIMIT);
  w->applications++;
  w->candidate.transfer=w->saved;s->declarations->next_identity=identity;
  const SERVICE_TYPE(FlowFunction) *fn=&body->plan->functions[f].declaration;
  uint16_t references[SERVICE_CONST(FLOW_LOCALS)];
  for(uint16_t i=0;i<fn->parameters;i++) {
   bool borrowed=s->declarations->locals[s->declarations->offsets[f]+i].mode!=0;
   if(borrowed && in->decoded.opcode!=OP_FILE_CALL_INDIRECT_REFS)return SERVICE_CONST(FLOW_UNRESOLVED);
   references[i]=borrowed?in->call_references[i]:SERVICE_CONST(FLOW_NO_REFERENCE);
  }
  SERVICE_TYPE(FlowStatus) status=SERVICE_FN(flow_call)(&w->candidate.transfer,in->byte_offset,f,references,fn->parameters);
  if(status!=SERVICE_CONST(FLOW_OK))return status;
  SERVICE_TYPE(FlowObligation) *call=NULL;
  for(uint16_t i=0;i<w->candidate.transfer.obligation_count;i++)
   if(w->candidate.transfer.obligations[i].site==in->byte_offset)call=&w->candidate.transfer.obligations[i];
  if(!call || call->kind!=SERVICE_CONST(FLOW_CALL) || call->target!=f)return SERVICE_CONST(FLOW_INVALID);
  call->target=NVM_V2_NO_INDEX;
  status=file_cyclic_canonicalize(&w->candidate);if(status!=SERVICE_CONST(FLOW_OK))return status;
  if(first) {
   w->first=w->candidate.transfer;w->canonical_first=w->candidate.canonical;
   common=*call;first_identity=s->declarations->next_identity;first=false;
  } else if(!file_cyclic_equal(&w->canonical_first,&w->candidate.canonical) ||
            !fif_state_metadata_equal(&w->first,&w->candidate.transfer) ||
            !obligation_same(common,*call))return SERVICE_CONST(FLOW_UNRESOLVED);
 }
 if(first)return SERVICE_CONST(FLOW_UNRESOLVED);
 *s=w->first;s->declarations->next_identity=first_identity;
 fact->cleanup=SERVICE_CONST(BODY_CLEANUP_CALL);fact->has_obligation=true;fact->obligation=common;
 fact->discharged_checks=SERVICE_CONST(FLOW_CHECK_CALLEE);
 fact->pending_checks|=common.checks&~fact->discharged_checks;
 return SERVICE_CONST(FLOW_OK);
}
static SERVICE_TYPE(FlowStatus) fif_target_status(SERVICE_TYPE(IndirectStatus) status) {
 switch(status) {
 case SERVICE_CONST(INDIRECT_DESCRIBED):return SERVICE_CONST(FLOW_OK);
 case SERVICE_CONST(INDIRECT_INVALID):return SERVICE_CONST(FLOW_INVALID);
 case SERVICE_CONST(INDIRECT_UNRESOLVED):return SERVICE_CONST(FLOW_UNRESOLVED);
 case SERVICE_CONST(INDIRECT_LIMIT):return SERVICE_CONST(FLOW_LIMIT);
 case SERVICE_CONST(INDIRECT_MEMORY):return SERVICE_CONST(FLOW_MEMORY);
 }
 return SERVICE_CONST(FLOW_INVALID);
}
SERVICE_TYPE(FlowStatus) SERVICE_FN(indirect_flow_analyze)(const NvmModule *m,SERVICE_TYPE(IndirectFlow) **out) {
 if(!out)return SERVICE_CONST(FLOW_INVALID);
 SERVICE_TYPE(IndirectTargets) *targets=NULL;
 SERVICE_TYPE(FlowStatus) status=fif_target_status(SERVICE_FN(indirect_targets)(m,&targets).status);
 if(status!=SERVICE_CONST(FLOW_OK))return status;
 size_t extra=sizeof(SERVICE_TYPE(IndirectFlow))+sizeof(FileIndirectFlowScratch),bytes,declaration_bytes;
 uint32_t instructions;
 status=file_cyclic_budget(m,extra,&instructions,&bytes,&declaration_bytes);
 if(status!=SERVICE_CONST(FLOW_OK)){SERVICE_FN(indirect_targets_free)(targets);return status;}
 if(targets->summary.storage_bound>SERVICE_CONST(INDIRECT_FLOW_BYTES)-bytes) {
  SERVICE_FN(indirect_targets_free)(targets);return SERVICE_CONST(FLOW_LIMIT);
 }
 SERVICE_TYPE(IndirectFlow) *r=calloc(1,sizeof *r);
 FileIndirectFlowScratch *w=calloc(1,sizeof *w);
 if(!r || !w){free(r);free(w);SERVICE_FN(indirect_targets_free)(targets);return SERVICE_CONST(FLOW_MEMORY);}
 r->targets=targets;w->targets=targets;
 for(uint32_t i=0;i<targets->summary.calls;i++) {
  SERVICE_TYPE(IndirectCall) call=targets->calls[i];
  uint32_t index=targets->plan->starts[call.function]+call.instruction;
  if(index>=instructions || targets->plan->instructions[index].byte_offset!=call.pc || w->bits[index]) {
   status=SERVICE_CONST(FLOW_INVALID);goto fail;
  }
  w->bits[index]=call.candidates;
 }
 FileCyclicExtension extension={extra,targets->graph,w,fif_transfer};
 status=file_cyclic_analyze_extended(m,&r->ownership,&extension);
 if(status!=SERVICE_CONST(FLOW_OK))goto fail;
 r->summary.targets=targets->summary;r->summary.ownership=r->ownership->summary;
 r->summary.candidate_applications=w->applications;
 r->summary.storage_bound=targets->summary.storage_bound+r->ownership->summary.storage_peak;
 if(r->summary.storage_bound>SERVICE_CONST(INDIRECT_FLOW_BYTES)){status=SERVICE_CONST(FLOW_LIMIT);goto fail;}
 free(w);*out=r;return SERVICE_CONST(FLOW_OK);
fail:
 free(w);SERVICE_FN(indirect_flow_free)(r);return status;
}
void SERVICE_FN(indirect_flow_free)(SERVICE_TYPE(IndirectFlow) *r) {
 if(r){SERVICE_FN(cyclic_free)(r->ownership);SERVICE_FN(indirect_targets_free)(r->targets);free(r);}
}
bool SERVICE_FN(indirect_flow_summary)(const SERVICE_TYPE(IndirectFlow) *r,SERVICE_TYPE(IndirectFlowSummary) *out) {
 if(!r || !out)return false;
 *out=r->summary;return true;
}
bool SERVICE_FN(indirect_flow_function)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,SERVICE_TYPE(CodeFunction) *out) {
 return r && SERVICE_FN(cyclic_function)(r->ownership,f,out);
}
bool SERVICE_FN(indirect_flow_local)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t l,SERVICE_TYPE(FlowDeclaration) *out) {
 return r && SERVICE_FN(cyclic_local)(r->ownership,f,l,out);
}
bool SERVICE_FN(indirect_flow_instruction)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,SERVICE_TYPE(CodeInstruction) *out) {
 return r && SERVICE_FN(cyclic_instruction)(r->ownership,f,i,out);
}
bool SERVICE_FN(indirect_flow_variant_count)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint8_t *out) {
 return r && SERVICE_FN(cyclic_variant_count)(r->ownership,f,i,out);
}
bool SERVICE_FN(indirect_flow_variant)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint8_t v,SERVICE_TYPE(CyclicVariant) *out) {
 return r && SERVICE_FN(cyclic_variant)(r->ownership,f,i,v,out);
}
bool SERVICE_FN(indirect_flow_input_local)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,SERVICE_TYPE(FlowValue) *out) {
 return r && SERVICE_FN(cyclic_input_local)(r->ownership,f,i,v,l,out);
}
bool SERVICE_FN(indirect_flow_input_stack)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,SERVICE_TYPE(FlowValue) *out) {
 return r && SERVICE_FN(cyclic_input_stack)(r->ownership,f,i,v,l,out);
}
bool SERVICE_FN(indirect_flow_input_reference)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,SERVICE_TYPE(FlowReference) *out) {
 return r && SERVICE_FN(cyclic_input_reference)(r->ownership,f,i,v,l,out);
}
bool SERVICE_FN(indirect_flow_input_region)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,uint64_t *out) {
 return r && SERVICE_FN(cyclic_input_region)(r->ownership,f,i,v,l,out);
}
bool SERVICE_FN(indirect_flow_call)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint8_t v,SERVICE_TYPE(IndirectFlowCall) *out) {
 if(!r || !out)return false;
 SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(CyclicVariant) variant;
 if(!SERVICE_FN(cyclic_instruction)(r->ownership,f,i,&in) || !isa_is_file_indirect_call(in.decoded.opcode) ||
    !SERVICE_FN(cyclic_variant)(r->ownership,f,i,v,&variant) || !variant.body.has_obligation)return false;
 for(uint32_t n=0;n<r->targets->summary.calls;n++) {
  const SERVICE_TYPE(IndirectCall) *call=&r->targets->calls[n];
  if(call->function==f && call->instruction==i && call->pc==in.byte_offset) {
   *out=(SERVICE_TYPE(IndirectFlowCall)){f,in.byte_offset,call->candidates,call->candidates,variant.body.obligation};return true;
  }
 }
 return false;
}
bool SERVICE_FN(indirect_flow_component)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t f,uint16_t i,uint16_t *out) {
 return r && SERVICE_FN(cyclic_component)(r->ownership,f,i,out);
}
bool SERVICE_FN(indirect_flow_type)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t i,SERVICE_TYPE(NominalLayout) *out) {
 return r && SERVICE_FN(cyclic_type)(r->ownership,i,out);
}
bool SERVICE_FN(indirect_flow_import)(const SERVICE_TYPE(IndirectFlow) *r,uint32_t i,uint32_t *out) {
 return r && SERVICE_FN(cyclic_import)(r->ownership,i,out);
}
