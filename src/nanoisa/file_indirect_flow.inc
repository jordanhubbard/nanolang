#include "file_indirect_flow.h"
struct NvmFileIndirectFlow {
 NvmFileIndirectTargets *targets;
 NvmFileCyclicReport *ownership;
 NvmFileIndirectFlowSummary summary;
};
typedef struct {
 const NvmFileIndirectTargets *targets;
 uint64_t bits[NVM_FILE_CODE_INSTRUCTIONS];
 uint32_t applications;
 NvmFileFlowState saved,first;
 FileCyclicState canonical_first;
 FileCyclicWorkspace candidate;
} FileIndirectFlowScratch;
static bool fif_state_metadata_equal(const NvmFileFlowState *a,const NvmFileFlowState *b) {
 if(a->family!=b->family || a->reachable!=b->reachable ||
    a->cleanup_obligations!=b->cleanup_obligations || a->obligation_count!=b->obligation_count)return false;
 for(uint16_t i=0;i<a->obligation_count;i++)if(!obligation_same(a->obligations[i],b->obligations[i]))return false;
 return true;
}
static NvmFileFlowStatus fif_transfer(void *context,NvmFileBodyReport *body,NvmFileFlowState *s,
                                      const NvmFileCodeInstruction *in,NvmFileBodyInstruction *fact) {
 FileIndirectFlowScratch *w=context;
 if(in->decoded.opcode==OP_FUNCREF) {
  if(in->decoded.operands[0].u32>=body->plan->count)return NVM_FILE_FLOW_INVALID;
  if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
  s->stack[s->stack_count++]=initial_value(scalar_type(TAG_FUNCTION),true,0);
  return NVM_FILE_FLOW_OK;
 }
 if(in->decoded.opcode!=OP_CALL_INDIRECT)return file_body_transfer(body,s,in,fact);
 if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
 NvmFileFlowValue callable=s->stack[s->stack_count-1];
 if(!callable.initialized || !same(callable.type,scalar_type(TAG_FUNCTION)) ||
    callable.owner || callable.arm!=NVM_FILE_FLOW_ARM_NONE)return NVM_FILE_FLOW_INVALID;
 /* The instruction pointer belongs to this second checked plan. The target map
  * is indexed only after the original function/PC/index identities agree. */
 size_t index=(size_t)(in-body->plan->instructions);
 if(index>=body->plan->instruction_count || !w->bits[index])return NVM_FILE_FLOW_UNRESOLVED;
 const NvmFileCodePlan *original=w->targets->plan;
 uint32_t caller=s->function;
 if(caller>=original->count || caller>=body->plan->count ||
    original->instruction_count!=body->plan->instruction_count ||
    original->starts[caller]!=body->plan->starts[caller] ||
    index<original->starts[caller] ||
    index-original->starts[caller]>=original->functions[caller].instruction_count ||
    original->functions[caller].instruction_count!=body->plan->functions[caller].instruction_count ||
    original->instructions[index].byte_offset!=in->byte_offset ||
    original->instructions[index].decoded.opcode!=OP_CALL_INDIRECT)return NVM_FILE_FLOW_INVALID;
 uint64_t candidates=w->bits[index];
 pop(s);w->saved=*s;
 uint64_t identity=s->declarations->next_identity,first_identity=identity;
 bool first=true;NvmFileFlowObligation common={0};
 for(uint32_t f=0;f<body->plan->count;f++)if(candidates&(UINT64_C(1)<<f)) {
  if(!body->checked[f])return NVM_FILE_FLOW_UNRESOLVED;
  if(w->applications==NVM_FILE_INDIRECT_FLOW_APPLICATIONS)return NVM_FILE_FLOW_LIMIT;
  w->applications++;
  w->candidate.transfer=w->saved;s->declarations->next_identity=identity;
  const NvmFileFlowFunction *fn=&body->plan->functions[f].declaration;
  uint16_t references[NVM_FILE_FLOW_LOCALS];
  for(uint16_t i=0;i<fn->parameters;i++) {
   if(s->declarations->locals[s->declarations->offsets[f]+i].mode)return NVM_FILE_FLOW_UNRESOLVED;
   references[i]=NVM_FILE_FLOW_NO_REFERENCE;
  }
  NvmFileFlowStatus status=nvm_file_flow_call(&w->candidate.transfer,in->byte_offset,f,references,fn->parameters);
  if(status!=NVM_FILE_FLOW_OK)return status;
  NvmFileFlowObligation *call=NULL;
  for(uint16_t i=0;i<w->candidate.transfer.obligation_count;i++)
   if(w->candidate.transfer.obligations[i].site==in->byte_offset)call=&w->candidate.transfer.obligations[i];
  if(!call || call->kind!=NVM_FILE_FLOW_CALL || call->target!=f)return NVM_FILE_FLOW_INVALID;
  call->target=NVM_V2_NO_INDEX;
  status=file_cyclic_canonicalize(&w->candidate);if(status!=NVM_FILE_FLOW_OK)return status;
  if(first) {
   w->first=w->candidate.transfer;w->canonical_first=w->candidate.canonical;
   common=*call;first_identity=s->declarations->next_identity;first=false;
  } else if(!file_cyclic_equal(&w->canonical_first,&w->candidate.canonical) ||
            !fif_state_metadata_equal(&w->first,&w->candidate.transfer) ||
            !obligation_same(common,*call))return NVM_FILE_FLOW_UNRESOLVED;
 }
 if(first)return NVM_FILE_FLOW_UNRESOLVED;
 *s=w->first;s->declarations->next_identity=first_identity;
 fact->cleanup=NVM_FILE_BODY_CLEANUP_CALL;fact->has_obligation=true;fact->obligation=common;
 fact->discharged_checks=NVM_FILE_FLOW_CHECK_CALLEE;
 fact->pending_checks|=common.checks&~fact->discharged_checks;
 return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus fif_target_status(NvmFileIndirectStatus status) {
 switch(status) {
 case NVM_FILE_INDIRECT_DESCRIBED:return NVM_FILE_FLOW_OK;
 case NVM_FILE_INDIRECT_INVALID:return NVM_FILE_FLOW_INVALID;
 case NVM_FILE_INDIRECT_UNRESOLVED:return NVM_FILE_FLOW_UNRESOLVED;
 case NVM_FILE_INDIRECT_LIMIT:return NVM_FILE_FLOW_LIMIT;
 case NVM_FILE_INDIRECT_MEMORY:return NVM_FILE_FLOW_MEMORY;
 }
 return NVM_FILE_FLOW_INVALID;
}
NvmFileFlowStatus nvm_file_indirect_flow_analyze(const NvmModule *m,NvmFileIndirectFlow **out) {
 if(!out)return NVM_FILE_FLOW_INVALID;
 NvmFileIndirectTargets *targets=NULL;
 NvmFileFlowStatus status=fif_target_status(nvm_file_indirect_targets(m,&targets).status);
 if(status!=NVM_FILE_FLOW_OK)return status;
 size_t extra=sizeof(NvmFileIndirectFlow)+sizeof(FileIndirectFlowScratch),bytes,declaration_bytes;
 uint32_t instructions;
 status=file_cyclic_budget(m,extra,&instructions,&bytes,&declaration_bytes);
 if(status!=NVM_FILE_FLOW_OK){nvm_file_indirect_targets_free(targets);return status;}
 if(targets->summary.storage_bound>NVM_FILE_INDIRECT_FLOW_BYTES-bytes) {
  nvm_file_indirect_targets_free(targets);return NVM_FILE_FLOW_LIMIT;
 }
 NvmFileIndirectFlow *r=calloc(1,sizeof *r);
 FileIndirectFlowScratch *w=calloc(1,sizeof *w);
 if(!r || !w){free(r);free(w);nvm_file_indirect_targets_free(targets);return NVM_FILE_FLOW_MEMORY;}
 r->targets=targets;w->targets=targets;
 for(uint32_t i=0;i<targets->summary.calls;i++) {
  NvmFileIndirectCall call=targets->calls[i];
  uint32_t index=targets->plan->starts[call.function]+call.instruction;
  if(index>=instructions || targets->plan->instructions[index].byte_offset!=call.pc || w->bits[index]) {
   status=NVM_FILE_FLOW_INVALID;goto fail;
  }
  w->bits[index]=call.candidates;
 }
 FileCyclicExtension extension={extra,targets->graph,w,fif_transfer};
 status=file_cyclic_analyze_extended(m,&r->ownership,&extension);
 if(status!=NVM_FILE_FLOW_OK)goto fail;
 r->summary.targets=targets->summary;r->summary.ownership=r->ownership->summary;
 r->summary.candidate_applications=w->applications;
 r->summary.storage_bound=targets->summary.storage_bound+r->ownership->summary.storage_peak;
 if(r->summary.storage_bound>NVM_FILE_INDIRECT_FLOW_BYTES){status=NVM_FILE_FLOW_LIMIT;goto fail;}
 free(w);*out=r;return NVM_FILE_FLOW_OK;
fail:
 free(w);nvm_file_indirect_flow_free(r);return status;
}
void nvm_file_indirect_flow_free(NvmFileIndirectFlow *r) {
 if(r){nvm_file_cyclic_free(r->ownership);nvm_file_indirect_targets_free(r->targets);free(r);}
}
bool nvm_file_indirect_flow_summary(const NvmFileIndirectFlow *r,NvmFileIndirectFlowSummary *out) {
 if(!r || !out)return false;
 *out=r->summary;return true;
}
bool nvm_file_indirect_flow_function(const NvmFileIndirectFlow *r,uint32_t f,NvmFileCodeFunction *out) {
 return r && nvm_file_cyclic_function(r->ownership,f,out);
}
bool nvm_file_indirect_flow_local(const NvmFileIndirectFlow *r,uint32_t f,uint16_t l,NvmFileFlowDeclaration *out) {
 return r && nvm_file_cyclic_local(r->ownership,f,l,out);
}
bool nvm_file_indirect_flow_instruction(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,NvmFileCodeInstruction *out) {
 return r && nvm_file_cyclic_instruction(r->ownership,f,i,out);
}
bool nvm_file_indirect_flow_variant_count(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint8_t *out) {
 return r && nvm_file_cyclic_variant_count(r->ownership,f,i,out);
}
bool nvm_file_indirect_flow_variant(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint8_t v,NvmFileCyclicVariant *out) {
 return r && nvm_file_cyclic_variant(r->ownership,f,i,v,out);
}
bool nvm_file_indirect_flow_input_local(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,NvmFileFlowValue *out) {
 return r && nvm_file_cyclic_input_local(r->ownership,f,i,v,l,out);
}
bool nvm_file_indirect_flow_input_stack(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,NvmFileFlowValue *out) {
 return r && nvm_file_cyclic_input_stack(r->ownership,f,i,v,l,out);
}
bool nvm_file_indirect_flow_input_reference(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,NvmFileFlowReference *out) {
 return r && nvm_file_cyclic_input_reference(r->ownership,f,i,v,l,out);
}
bool nvm_file_indirect_flow_input_region(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint8_t v,uint16_t l,uint64_t *out) {
 return r && nvm_file_cyclic_input_region(r->ownership,f,i,v,l,out);
}
bool nvm_file_indirect_flow_call(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint8_t v,NvmFileIndirectFlowCall *out) {
 if(!r || !out)return false;
 NvmFileCodeInstruction in;NvmFileCyclicVariant variant;
 if(!nvm_file_cyclic_instruction(r->ownership,f,i,&in) || in.decoded.opcode!=OP_CALL_INDIRECT ||
    !nvm_file_cyclic_variant(r->ownership,f,i,v,&variant) || !variant.body.has_obligation)return false;
 for(uint32_t n=0;n<r->targets->summary.calls;n++) {
  const NvmFileIndirectCall *call=&r->targets->calls[n];
  if(call->function==f && call->instruction==i && call->pc==in.byte_offset) {
   *out=(NvmFileIndirectFlowCall){f,in.byte_offset,call->candidates,call->candidates,variant.body.obligation};return true;
  }
 }
 return false;
}
bool nvm_file_indirect_flow_component(const NvmFileIndirectFlow *r,uint32_t f,uint16_t i,uint16_t *out) {
 return r && nvm_file_cyclic_component(r->ownership,f,i,out);
}
bool nvm_file_indirect_flow_type(const NvmFileIndirectFlow *r,uint32_t i,NvmFileNominalLayout *out) {
 return r && nvm_file_cyclic_type(r->ownership,i,out);
}
bool nvm_file_indirect_flow_import(const NvmFileIndirectFlow *r,uint32_t i,uint32_t *out) {
 return r && nvm_file_cyclic_import(r->ownership,i,out);
}
