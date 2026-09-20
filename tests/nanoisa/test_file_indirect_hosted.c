/* I prepare copied indirect facts only; no callable or File service executes. */
#define FILE_HOSTED_MAIN prior_file_hosted_fixture_main
#include "test_file_hosted.c"
#undef FILE_HOSTED_MAIN
#include "../../src/nanoisa/file_indirect_hosted.h"
#include "../../src/nanoisa/file_cyclic_hosted.h"
#ifdef HOSTED_INSTRUMENT
#define malloc file_test_malloc
#define calloc file_test_calloc
#define realloc file_test_realloc
#define free file_test_free
#include "../../src/nanoisa/file_flow.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
#endif
static NvmFileIndirectHostedPlan *ih_expect(const uint8_t *bytes,size_t size,NvmFileFlowStatus expected){
 NvmFileIndirectHostedPlan *p=(void *)(uintptr_t)1;
 NvmFileFlowStatus actual=nvm_file_indirect_hosted_prepare(bytes,size,&p);
 if(actual!=expected)fprintf(stderr,"indirect hosted expected %u actual %u\n",expected,actual);
 CHECK(actual==expected);if(actual!=NVM_FILE_FLOW_OK){CHECK(p==(void *)(uintptr_t)1);return NULL;}
 NvmFileIndirectHostedStartup s;CHECK(nvm_file_indirect_hosted_startup(p,&s));
 CHECK(s.revision==1 && !s.runtime_admitted && s.input_bytes==size && s.query_storage_peak<=NVM_FILE_INDIRECT_FLOW_BYTES);
 CHECK(s.retained_bound>=size+s.query_storage_peak && s.retained_bound<=s.allocation_bound && s.allocation_bound<=NVM_FILE_HOSTED_BYTES);return p;
}
static void ih_value(NvmFileFlowValue a,NvmFileFlowValue b){
 CHECK(a.type.tag==b.type.tag && a.type.mode==b.type.mode && a.type.global_index==b.type.global_index &&
 a.type.catalog_ordinal==b.type.catalog_ordinal && a.type.category==b.type.category &&
 a.initialized==b.initialized && a.owner==b.owner && a.arm==b.arm);
}
static void ih_info(NvmFileCyclicStateInfo a,NvmFileCyclicStateInfo b){CHECK(a.locals==b.locals && a.stack==b.stack && a.owners==b.owners && a.references==b.references && a.regions==b.regions);}
static void ih_body(NvmFileBodyInstruction a,NvmFileBodyInstruction b){
 CHECK(a.reachable==b.reachable && a.exit_checked==b.exit_checked && a.refinement==b.refinement &&
 a.has_obligation==b.has_obligation && a.input_stack==b.input_stack && a.output_stack==b.output_stack &&
 a.cleanup_local==b.cleanup_local && a.cleanup==b.cleanup && a.discharged_checks==b.discharged_checks && a.pending_checks==b.pending_checks);
 NvmFileFlowObligation x=a.obligation,y=b.obligation;
 CHECK(x.kind==y.kind && x.site==y.site && x.target==y.target && x.checks==y.checks &&
 x.required_rights==y.required_rights && x.acquired_rights==y.acquired_rights && x.parameters==y.parameters &&
 x.owned_inputs==y.owned_inputs && x.borrowed_inputs==y.borrowed_inputs && x.result_count==y.result_count &&
 x.outcomes[0]==y.outcomes[0] && x.outcomes[1]==y.outcomes[1]);
 ih_value((NvmFileFlowValue){x.result,false,0,0},(NvmFileFlowValue){y.result,false,0,0});
}
static unsigned ih_compare(NvmFileIndirectHostedPlan *p,NvmFileIndirectFlow *q){
 NvmFileIndirectFlowSummary ha,hb;CHECK(nvm_file_indirect_hosted_query_summary(p,&ha) && nvm_file_indirect_flow_summary(q,&hb));
 NvmFileCyclicSummary a=ha.ownership,b=hb.ownership;CHECK(ha.targets.calls==hb.targets.calls && ha.candidate_applications==hb.candidate_applications && !ha.runtime_admitted);
 CHECK(a.revision==b.revision && a.functions==b.functions && a.instructions==b.instructions && a.variants==b.variants &&
 a.transfers==b.transfers && a.edges==b.edges && a.storage_peak==b.storage_peak && !a.runtime_admitted && !b.runtime_admitted);
 uint64_t visited=0;unsigned seen_services=0,calls=0,borrowed=0;
 for(uint32_t rank=0;rank<a.functions;rank++){uint32_t f;CHECK(nvm_file_indirect_hosted_function_order(p,rank,&f));CHECK(f<a.functions && !(visited&(UINT64_C(1)<<f)));visited|=UINT64_C(1)<<f;}
 for(uint32_t f=0;f<a.functions;f++){
  NvmFileIndirectHostedFunction hf;NvmFileCodeFunction qf;CHECK(nvm_file_indirect_hosted_function(p,f,&hf) && nvm_file_indirect_flow_function(q,f,&qf));
  CHECK(hf.code.code_offset==qf.code_offset && hf.code.code_length==qf.code_length && hf.code.instruction_count==qf.instruction_count &&
        hf.locals==qf.declaration.locals && hf.code.declaration.parameters==qf.declaration.parameters && hf.entry_variant==0);
  ih_value((NvmFileFlowValue){hf.code.declaration.result,false,0,0},(NvmFileFlowValue){qf.declaration.result,false,0,0});
  uint16_t peak=0,owners=0,refs=0,regions=0;
  for(uint16_t l=0;l<hf.locals;l++){
   NvmFileFlowDeclaration hd,qd;CHECK(nvm_file_indirect_hosted_local(p,f,l,&hd) && nvm_file_indirect_flow_local(q,f,l,&qd));
   ih_value((NvmFileFlowValue){hd,false,0,0},(NvmFileFlowValue){qd,false,0,0});
   NvmFileFlowValue seed;CHECK(nvm_file_indirect_hosted_input_local(p,f,0,hf.entry_variant,l,&seed));
   CHECK(seed.initialized==(l<qf.declaration.parameters));
  }
  for(uint16_t i=0;i<qf.instruction_count;i++){
   NvmFileCodeInstruction hi,qi;uint8_t hn,qn;uint16_t hc,qc;
   CHECK(nvm_file_indirect_hosted_instruction(p,f,i,&hi) && nvm_file_indirect_flow_instruction(q,f,i,&qi));
   CHECK(hi.byte_offset==qi.byte_offset && hi.catalog_ordinal==qi.catalog_ordinal && hi.successor_count==qi.successor_count &&
         hi.successors[0]==qi.successors[0] && hi.successors[1]==qi.successors[1] && hi.decoded.opcode==qi.decoded.opcode &&
         hi.decoded.byte_length==qi.decoded.byte_length && !memcmp(hi.decoded.operands,qi.decoded.operands,sizeof hi.decoded.operands) &&
         !memcmp(hi.decoded.operand_types,qi.decoded.operand_types,sizeof hi.decoded.operand_types));
   CHECK(nvm_file_indirect_hosted_component(p,f,i,&hc) && nvm_file_indirect_flow_component(q,f,i,&qc) && hc==qc);
   CHECK(nvm_file_indirect_hosted_variant_count(p,f,i,&hn) && nvm_file_indirect_flow_variant_count(q,f,i,&qn) && hn==qn);
   for(uint8_t v=0;v<hn;v++){
    NvmFileCyclicVariant x,y;CHECK(nvm_file_indirect_hosted_variant(p,f,i,v,&x) && nvm_file_indirect_flow_variant(q,f,i,v,&y));
    ih_info(x.input,y.input);ih_info(x.output,y.output);ih_body(x.body,y.body);
    CHECK(x.edge_mask==y.edge_mask && x.edge_variants[0]==y.edge_variants[0] && x.edge_variants[1]==y.edge_variants[1]);
    NvmFileCyclicStateInfo infos[2]={x.input,x.output};for(unsigned side=0;side<2;side++){
     if(infos[side].stack>peak)peak=infos[side].stack;
     if(infos[side].owners>owners)owners=infos[side].owners;
     if(infos[side].references>refs)refs=infos[side].references;
     if(infos[side].regions>regions)regions=infos[side].regions;
    }
    for(uint16_t l=0;l<x.input.locals;l++){NvmFileFlowValue hv,qv;CHECK(nvm_file_indirect_hosted_input_local(p,f,i,v,l,&hv) && nvm_file_indirect_flow_input_local(q,f,i,v,l,&qv));ih_value(hv,qv);}
    for(uint16_t l=0;l<x.input.stack;l++){NvmFileFlowValue hv,qv;CHECK(nvm_file_indirect_hosted_input_stack(p,f,i,v,l,&hv) && nvm_file_indirect_flow_input_stack(q,f,i,v,l,&qv));ih_value(hv,qv);}
    for(uint16_t l=0;l<NVM_FILE_FLOW_REFERENCES;l++){
     NvmFileFlowReference hv,qv;CHECK(nvm_file_indirect_hosted_input_reference(p,f,i,v,l,&hv) && nvm_file_indirect_flow_input_reference(q,f,i,v,l,&qv));
     CHECK(hv.live==qv.live && hv.formal==qv.formal && hv.local==qv.local && hv.owner==qv.owner && hv.identity==qv.identity && hv.region==qv.region);
    }
    for(uint16_t l=0;l<x.input.regions;l++){uint64_t hv,qv;CHECK(nvm_file_indirect_hosted_input_region(p,f,i,v,l,&hv) && nvm_file_indirect_flow_input_region(q,f,i,v,l,&qv) && hv==qv);}
    for(unsigned edge=0;edge<2;edge++)if(x.edge_mask&(1u<<edge)){
     NvmFileCyclicVariant dest;CHECK(edge<hi.successor_count && nvm_file_indirect_hosted_variant(p,f,hi.successors[edge],x.edge_variants[edge],&dest));
     CHECK(dest.input.stack==x.output.stack && dest.input.owners==x.output.owners);
    }else CHECK(x.edge_variants[edge]==NVM_FILE_CYCLIC_NO_VARIANT);
    CHECK(x.body.pending_checks & NVM_FILE_FLOW_CHECK_CLEANUP);
    if(x.body.has_obligation){NvmFileFlowObligation o=x.body.obligation;CHECK(o.site==hi.byte_offset);
     if(hi.decoded.opcode==OP_CALL_INDIRECT){NvmFileIndirectFlowCall hc,qc;CHECK(nvm_file_indirect_hosted_call(p,f,i,v,&hc) && nvm_file_indirect_flow_call(q,f,i,v,&qc));CHECK(hc.candidates==qc.candidates && hc.checked_candidates==qc.checked_candidates && hc.candidates==hc.checked_candidates && hc.function==f && hc.pc==hi.byte_offset && hc.common.target==NVM_V2_NO_INDEX);}
     else CHECK(o.target==hi.decoded.operands[0].u32);
     if(o.kind==NVM_FILE_FLOW_CALL){calls++;borrowed+=o.borrowed_inputs!=0;CHECK(x.body.discharged_checks==NVM_FILE_FLOW_CHECK_CALLEE && x.body.pending_checks==(NVM_FILE_FLOW_CHECK_RESULT|NVM_FILE_FLOW_CHECK_CLEANUP));}
     else{uint32_t import;CHECK(nvm_file_indirect_hosted_import(p,hi.catalog_ordinal,&import) && import==o.target);seen_services|=1u<<hi.catalog_ordinal;
      const NlFilePlanMethod *method=nl_file_catalog_method(hi.catalog_ordinal);CHECK(method && o.required_rights==method->required_rights && o.acquired_rights==method->acquired_rights);
      CHECK(!x.body.discharged_checks && x.body.pending_checks==o.checks);
      if(hi.catalog_ordinal==4)CHECK(o.owned_inputs==1 && o.outcomes[0]==NVM_FILE_FLOW_INPUT_CONSUMED && o.outcomes[1]==NVM_FILE_FLOW_INPUT_CONSUMED);
      if(hi.catalog_ordinal>=1 && hi.catalog_ordinal<=3)CHECK(o.borrowed_inputs==1 && (o.checks & NVM_FILE_FLOW_CHECK_BORROW));
      if(hi.catalog_ordinal==1)CHECK(o.checks & NVM_FILE_FLOW_CHECK_BYTE);
     }
    }
   }
  }
  CHECK(hf.operand_peak==peak && hf.frame_owner_peak==owners && hf.frame_reference_peak==refs && hf.frame_region_peak==regions);
 }
 for(uint32_t i=0;i<8;i++){NvmFileNominalLayout x,y;CHECK(nvm_file_indirect_hosted_type(p,i,&x) && nvm_file_indirect_flow_type(q,i,&y));CHECK(x.global_index==y.global_index && x.catalog_ordinal==y.catalog_ordinal && x.source_ordinal==y.source_ordinal && x.layout_kind==y.layout_kind && x.ownership_flags==y.ownership_flags && x.category==y.category);}
 for(uint32_t i=0;i<5;i++){uint32_t x,y;CHECK(nvm_file_indirect_hosted_import(p,i,&x) && nvm_file_indirect_flow_import(q,i,&y) && x==y);}
 CHECK(seen_services!=31 || (calls==2 && borrowed==1));
 printf("I compared full hosted variants: service mask %u, calls %u, borrowed calls %u\n",seen_services,calls,borrowed);return seen_services;
}
static void ih_invalid_getters(NvmFileIndirectHostedPlan *p){
#define CH_BAD(type,expression) do{type out;memset(&out,0xa5,sizeof out);type before=out;CHECK(!(expression));CHECK(!memcmp(&out,&before,sizeof out));}while(0)
 CH_BAD(NvmFileIndirectHostedStartup,nvm_file_indirect_hosted_startup(NULL,&out));
 CH_BAD(NvmFileIndirectFlowSummary,nvm_file_indirect_hosted_query_summary(NULL,&out));
 CH_BAD(NvmFileIndirectHostedFunction,nvm_file_indirect_hosted_function(p,UINT32_MAX,&out));
 CH_BAD(uint32_t,nvm_file_indirect_hosted_function_order(p,UINT32_MAX,&out));
 CH_BAD(NvmFileFlowDeclaration,nvm_file_indirect_hosted_local(p,0,UINT16_MAX,&out));
 CH_BAD(NvmFileCodeInstruction,nvm_file_indirect_hosted_instruction(p,0,UINT16_MAX,&out));
 CH_BAD(uint16_t,nvm_file_indirect_hosted_component(p,UINT32_MAX,0,&out));
 CH_BAD(uint8_t,nvm_file_indirect_hosted_variant_count(p,0,UINT16_MAX,&out));
 CH_BAD(NvmFileCyclicVariant,nvm_file_indirect_hosted_variant(p,0,0,UINT8_MAX,&out));
 CH_BAD(NvmFileFlowValue,nvm_file_indirect_hosted_input_local(p,0,0,0,UINT16_MAX,&out));
 CH_BAD(NvmFileFlowValue,nvm_file_indirect_hosted_input_stack(p,0,0,0,0,&out));
 CH_BAD(NvmFileFlowReference,nvm_file_indirect_hosted_input_reference(p,0,0,0,256,&out));
 CH_BAD(uint64_t,nvm_file_indirect_hosted_input_region(p,0,0,0,0,&out));
 CH_BAD(NvmFileNominalLayout,nvm_file_indirect_hosted_type(p,UINT32_MAX,&out));
 CH_BAD(uint32_t,nvm_file_indirect_hosted_import(p,UINT32_MAX,&out));
 unsigned char bytes[4]={1,2,3,4};CHECK(!nvm_file_indirect_hosted_bytes(p,SIZE_MAX,bytes,sizeof bytes) && bytes[0]==1 && bytes[3]==4);
 CHECK(!nvm_file_indirect_hosted_bytes(p,0,NULL,1));CHECK(nvm_file_indirect_hosted_bytes(p,0,NULL,0));
#undef CH_BAD
}
static void ih_ref(Body *c,unsigned f){op(c,OP_FUNCREF);u32(c,f);}
static void ih_call(Body *c,unsigned n){op(c,OP_CALL_INDIRECT);u16(c,(uint16_t)n);u16(c,1);}
static NvmModule *ih_module(NvmFileNominalBindings *b,bool perm,bool reverse,unsigned larger,uint32_t *pc){
 NvmModule *m=bodymodule(b,perm);
 for(unsigned f=1;f<=4;f+=3){uint8_t tag=TAG_INT;size_t at=ownership_function_offset(m,f);
  m->functions[f].result_tag=TAG_INT;CHECK(nvm_set_function_param_types(m,f,&tag,1));
  desc(m->ownership_data+at+4,TAG_INT,0,NVM_V2_NO_INDEX);desc(m->ownership_data+at+12,TAG_INT,0,NVM_V2_NO_INDEX);
  Body c={0};if(f==larger){for(unsigned i=0;i<20;i++)integer(&c);for(unsigned i=0;i<20;i++)op(&c,OP_POP);}
  one(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);setbody(m,f,c);
 }
 desc(m->ownership_data+24+12+8*4,TAG_FUNCTION,0,NVM_V2_NO_INDEX);
 Body c={0};op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t other=branch(&c,OP_JMP_FALSE,0);
 ih_ref(&c,reverse?4:1);uint32_t join=branch(&c,OP_JMP,0);target(&c,other);ih_ref(&c,reverse?1:4);target(&c,join);one(&c,OP_STORE_LOCAL,4);
 uint32_t header=c.n;op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t done=branch(&c,OP_JMP_FALSE,0);
 integer(&c);one(&c,OP_LOAD_LOCAL,4);*pc=c.n;ih_call(&c,1);op(&c,OP_POP);
 uint32_t back=branch(&c,OP_JMP,0);wr32(c.bytes+back+1,(uint32_t)(int32_t)((int64_t)header-back));target(&c,done);retint(&c);setbody(m,0,c);return m;
}
static uint16_t ih_index(NvmFileIndirectHostedPlan *p,uint32_t pc){
 NvmFileIndirectHostedFunction f;CHECK(nvm_file_indirect_hosted_function(p,0,&f));
 for(uint16_t i=0;i<f.code.instruction_count;i++){NvmFileCodeInstruction in;CHECK(nvm_file_indirect_hosted_instruction(p,0,i,&in));if(in.byte_offset==pc)return i;}
 CHECK(false);return 0;
}
static void ih_relations(void){
 for(unsigned perm=0;perm<2;perm++)for(unsigned reverse=0;reverse<2;reverse++)for(unsigned larger=1;larger<=4;larger+=3){
  NvmFileNominalBindings b;uint32_t pc;NvmModule *m=ih_module(&b,perm!=0,reverse!=0,larger,&pc);
  NvmFileIndirectFlow *q=NULL;OK(nvm_file_indirect_flow_analyze(m,&q));size_t size;uint8_t *bytes=serialize(m,&size),*saved=malloc(size);CHECK(saved);memcpy(saved,bytes,size);
  NvmFileIndirectHostedPlan *p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);ih_compare(p,q);ih_invalid_getters(p);
  NvmFileIndirectHostedFunction caller,big,small;CHECK(nvm_file_indirect_hosted_function(p,0,&caller));CHECK(nvm_file_indirect_hosted_function(p,larger,&big));CHECK(nvm_file_indirect_hosted_function(p,larger==1?4:1,&small));
  CHECK(big.operand_peak==20 && big.vm_value_slots==22 && small.vm_value_slots==3);
  CHECK(caller.operand_peak==2 && caller.staging_slots==3 && caller.vm_value_slots==37 && caller.native_value_slots==39 && caller.frames==2);
  uint16_t site=ih_index(p,pc);NvmFileIndirectFlowCall call;CHECK(nvm_file_indirect_hosted_call(p,0,site,0,&call));CHECK(call.candidates==18 && call.checked_candidates==18 && call.common.parameters==1 && call.common.result.tag==TAG_INT);
  NvmFileIndirectFlowCall bad;memset(&bad,0xa5,sizeof bad);call=bad;CHECK(!nvm_file_indirect_hosted_call(p,0,site,255,&call) && !memcmp(&call,&bad,sizeof call));
  CHECK(!nvm_verify(m).ok);char error[128];CHECK(!nvm2c_emit(m,error,sizeof error));expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);
  NvmFileCyclicHostedPlan *old=(void *)(uintptr_t)1;CHECK(nvm_file_cyclic_hosted_prepare(bytes,size,&old)==NVM_FILE_FLOW_UNRESOLVED && old==(void *)(uintptr_t)1);
  memset(bytes,0,size);free(bytes);memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);
  ih_compare(p,q);uint8_t *copy=malloc(size);CHECK(copy && nvm_file_indirect_hosted_bytes(p,0,copy,size) && !memcmp(copy,saved,size));free(copy);free(saved);
  CHECK(nvm_file_indirect_hosted_call(p,0,site,0,&call) && call.candidates==18);nvm_file_indirect_flow_free(q);nvm_file_indirect_hosted_free(p);
 }
 /* I preserve all direct/borrow/service obligations in the distinct plan. */
 for(unsigned perm=0;perm<2;perm++){NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,perm!=0);setbody(m,0,lifecycle_code(b));size_t size;uint8_t *bytes=serialize(m,&size);
  NvmFileIndirectFlow *q=NULL;OK(nvm_file_indirect_flow_analyze(m,&q));NvmFileIndirectHostedPlan *p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);CHECK(ih_compare(p,q)==31);
  nvm_file_indirect_hosted_free(p);nvm_file_indirect_flow_free(q);free(bytes);nvm_module_free(m);
 }
}
static void ih_owned_candidates(void){
 for(unsigned perm=0;perm<2;perm++)for(unsigned result=0;result<2;result++){
  NvmFileNominalBindings b;uint32_t pc;NvmModule *m=ih_module(&b,perm!=0,perm!=0,4,&pc);
  uint8_t tag=result?TAG_UNION:TAG_STRUCT;uint32_t layout=b.layouts[result?3:0];
  for(unsigned f=1;f<=4;f+=3){size_t at=ownership_function_offset(m,f);m->functions[f].result_tag=tag;CHECK(nvm_set_function_param_types(m,f,&tag,1));
   desc(m->ownership_data+at+4,tag,0,layout);desc(m->ownership_data+at+12,tag,0,layout);Body c={0};one(&c,OP_OWN_MOVE_LOCAL,0);op(&c,OP_RET);setbody(m,f,c);
  }
  Body c={0};op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t other=branch(&c,OP_JMP_FALSE,0);ih_ref(&c,perm?4:1);uint32_t join=branch(&c,OP_JMP,0);target(&c,other);ih_ref(&c,perm?1:4);target(&c,join);one(&c,OP_STORE_LOCAL,4);
  service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);uint32_t failure=UINT32_MAX;
  if(!result){failure=branch(&c,OP_FILE_RESULT_BRANCH,1);take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);}
  uint16_t slot=result?1:0;uint32_t header=c.n;one(&c,OP_OWN_MOVE_LOCAL,slot);one(&c,OP_LOAD_LOCAL,4);pc=c.n;ih_call(&c,1);one(&c,OP_OWN_STORE_LOCAL,slot);
  op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t done=branch(&c,OP_JMP_FALSE,0);uint32_t back=branch(&c,OP_JMP,0);wr32(c.bytes+back+1,(uint32_t)(int32_t)((int64_t)header-back));target(&c,done);one(&c,OP_FILE_DROP_LOCAL,slot);retint(&c);
  if(!result){target(&c,failure);take_result(&c,1,1);op(&c,OP_POP);retint(&c);}setbody(m,0,c);
  size_t size;uint8_t *bytes=serialize(m,&size);NvmFileIndirectHostedPlan *p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);NvmFileIndirectFlow *q=NULL;OK(nvm_file_indirect_flow_analyze(m,&q));ih_compare(p,q);
  uint16_t site=ih_index(p,pc);NvmFileIndirectFlowCall call;CHECK(nvm_file_indirect_hosted_call(p,0,site,0,&call) && call.candidates==18 && call.common.owned_inputs==1 && !call.common.borrowed_inputs && call.common.result.tag==tag && call.common.result.global_index==layout);
  free(bytes);nvm_module_free(m);ih_compare(p,q);CHECK(nvm_file_indirect_hosted_call(p,0,site,0,&call) && call.common.owned_inputs==1);
  nvm_file_indirect_flow_free(q);nvm_file_indirect_hosted_free(p);
 }
}
static void ih_candidate_depth(void){
 for(unsigned larger=1;larger<=4;larger+=3){NvmFileNominalBindings b;uint32_t pc;NvmModule *m=ih_module(&b,false,false,larger,&pc);
  make_initializer(m,3);m->functions[3].name_idx=string(m,"deep_helper");m->functions[3].result_count=1;m->functions[3].result_tag=TAG_INT;
  desc(m->ownership_data+ownership_function_offset(m,3)+4,TAG_INT,0,NVM_V2_NO_INDEX);
  Body c={0};for(unsigned i=0;i<30;i++)integer(&c);for(unsigned i=0;i<30;i++)op(&c,OP_POP);retint(&c);setbody(m,3,c);
  c=(Body){0};for(unsigned i=0;i<20;i++)integer(&c);for(unsigned i=0;i<20;i++)op(&c,OP_POP);op(&c,OP_CALL);u32(&c,3);op(&c,OP_POP);one(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);setbody(m,larger,c);
  size_t size;uint8_t *bytes=serialize(m,&size);NvmFileIndirectHostedPlan *p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);NvmFileIndirectHostedFunction big,root;
  CHECK(nvm_file_indirect_hosted_function(p,larger,&big) && big.frames==2 && big.vm_value_slots==35 && big.native_value_slots==55);
  CHECK(nvm_file_indirect_hosted_function(p,0,&root) && root.frames==3 && root.vm_value_slots==50 && root.native_value_slots==72 && root.reference_slots==768 && root.region_slots==768);
  unsigned ranks[5];for(unsigned r=0;r<5;r++){uint32_t f;CHECK(nvm_file_indirect_hosted_function_order(p,r,&f));ranks[f]=r;}
  CHECK(ranks[3]<ranks[larger] && ranks[1]<ranks[0] && ranks[4]<ranks[0]);
  nvm_file_indirect_hosted_free(p);free(bytes);nvm_module_free(m);
 }
}
static void ih_startup_wire(void){
 NvmFileNominalBindings b;uint32_t pc;NvmModule *m=ih_module(&b,false,false,4,&pc);make_initializer(m,3);
 Body c={0};for(unsigned i=0;i<50;i++)integer(&c);for(unsigned i=0;i<50;i++)op(&c,OP_POP);op(&c,OP_RET);setbody(m,3,c);
 size_t size;uint8_t *bytes=serialize(m,&size);NvmFileIndirectHostedPlan *p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);NvmFileIndirectHostedStartup s;
 CHECK(nvm_file_indirect_hosted_startup(p,&s) && s.entry==0 && s.initializer==3 && s.vm_value_slots==53 && s.native_value_slots==53 && s.frames==2);nvm_file_indirect_hosted_free(p);
 NvmV2SectionEntry fs=section(bytes,size,NVM_V2_SECTION_FUNCTIONS);uint8_t *depth=bytes+fs.offset+4+28;
 depth[0]=1;rehash(bytes,size);ih_expect(bytes,size,NVM_FILE_FLOW_INVALID);depth[0]=2;rehash(bytes,size);p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);nvm_file_indirect_hosted_free(p);free(bytes);
 m->functions[1].name_idx=string(m,"__init__");bytes=serialize(m,&size);ih_expect(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);
 m->functions[1].name_idx=string(m,"candidate");m->functions[4].name_idx=string(m,"__init__");bytes=serialize(m,&size);p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);CHECK(nvm_file_indirect_hosted_startup(p,&s) && s.initializer==3);nvm_file_indirect_hosted_free(p);free(bytes);
 m->functions[4].name_idx=nvm_add_string(m,"bad\0name",8);bytes=serialize(m,&size);ih_expect(bytes,size,NVM_FILE_FLOW_INVALID);free(bytes);nvm_module_free(m);
 m=ih_module(&b,false,false,4,&pc);m->header.entry_point=1;bytes=serialize(m,&size);ih_expect(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);m->header.entry_point=0;
 bytes=serialize(m,&size);uint8_t *saved=malloc(size);CHECK(saved);memcpy(saved,bytes,size);
 ih_expect(NULL,0,NVM_FILE_FLOW_INVALID);CHECK(nvm_file_indirect_hosted_prepare(bytes,size,NULL)==NVM_FILE_FLOW_INVALID);ih_expect(bytes,size-1,NVM_FILE_FLOW_INVALID);
 uint8_t dummy=0;ih_expect(&dummy,(size_t)NVM_FILE_HOSTED_INPUT_BYTES+1,NVM_FILE_FLOW_LIMIT);
 NvmV2Header h;CHECK(nvm_v2_read_header(bytes,size,&h)==NVM_V2_OK);h.feature_bits|=NVM_V2_FEATURE_CALLBACKS;nvm_v2_write_header(bytes,&h);ih_expect(bytes,size,NVM_FILE_FLOW_UNRESOLVED);memcpy(bytes,saved,size);
 fs=section(bytes,size,NVM_V2_SECTION_FUNCTIONS);wr32(bytes+fs.offset,65);rehash(bytes,size);ih_expect(bytes,size,NVM_FILE_FLOW_LIMIT);memcpy(bytes,saved,size);
 NvmV2SectionEntry ss=section(bytes,size,NVM_V2_SECTION_SERVICE_BINDINGS);bytes[ss.offset]=1;rehash(bytes,size);ih_expect(bytes,size,NVM_FILE_FLOW_INVALID);free(bytes);free(saved);
 /* Every body is checked, even one unreachable from the selected entry. */
 c=(Body){0};one(&c,OP_LOAD_LOCAL,99);op(&c,OP_RET);setbody(m,4,c);bytes=serialize(m,&size);ih_expect(bytes,size,NVM_FILE_FLOW_INVALID);free(bytes);nvm_module_free(m);
}
static NvmModule *ih_arity(unsigned parameters){
 NvmFileNominalBindings b;uint32_t pc;NvmModule *m=ih_module(&b,false,false,4,&pc);
 /* The target has256 ordinary locals even for the253-argument wire control. */
 size_t old=ownership_function_offset(m,1),tail=old+12+8*m->functions[1].local_count;
 size_t size=m->ownership_size+8*(256-m->functions[1].local_count);uint8_t *data=calloc(1,size);CHECK(data);
 memcpy(data,m->ownership_data,old+12);memcpy(data+old+12+8*256,m->ownership_data+tail,m->ownership_size-tail);
 data[old]=0;data[old+1]=1;data[old+2]=(uint8_t)parameters;data[old+3]=(uint8_t)(parameters>>8);
 for(unsigned i=0;i<256;i++)desc(data+old+12+8*i,TAG_INT,0,NVM_V2_NO_INDEX);
 free(m->ownership_data);m->ownership_data=data;m->ownership_size=(uint32_t)size;m->functions[1].local_count=256;m->functions[1].arity=(uint16_t)parameters;
 uint8_t tags[256];memset(tags,TAG_INT,sizeof tags);CHECK(nvm_set_function_param_types(m,1,tags,(uint16_t)parameters));
 Body c={0};for(unsigned i=0;i<parameters;i++)integer(&c);ih_ref(&c,1);ih_call(&c,parameters);op(&c,OP_RET);setbody(m,0,c);return m;
}
static void ih_internal_and_arity(void){
 for(unsigned n=253;n<=256;n++){
  NvmModule *m=ih_arity(n);size_t size;uint8_t *bytes=serialize(m,&size);
  NvmFileIndirectHostedPlan *p=ih_expect(bytes,size,n==253?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_LIMIT);
  if(p){NvmFileIndirectHostedFunction f;CHECK(nvm_file_indirect_hosted_function(p,0,&f) && f.operand_peak==254 && f.staging_slots==255);
#ifdef HOSTED_INSTRUMENT
   /* These are isolated downstream arithmetic checks, not serialized255-arg
    * admission: the original256-instruction ceiling independently refuses it. */
   NvmFileCodePlan *code=p->query->ownership->plan;uint16_t i=254;CHECK(code->instructions[i].decoded.opcode==OP_CALL_INDIRECT);
   FileCyclicNode *node=p->query->ownership->sites[i].nodes[0];NvmFileCyclicVariant saved=node->fact;
   NvmFileFlowFunction savedfn=code->functions[1].declaration;
   NvmV2Module wire={0};CHECK(nvm_v2_module_deserialize(bytes,size,&wire)==NVM_V2_OK);
   node->fact.input.stack=node->fact.body.input_stack=256;node->fact.body.obligation.parameters=255;code->functions[1].declaration.parameters=255;
   OK(file_ih_bounds(p,&wire));CHECK(p->functions[0].staging_slots==257 && p->functions[0].operand_peak==256);
   node->fact.body.obligation.parameters=256;code->functions[1].declaration.parameters=256;
   CHECK(file_ih_bounds(p,&wire)==NVM_FILE_FLOW_INVALID);
   node->fact=saved;code->functions[1].declaration=savedfn;OK(file_ih_bounds(p,&wire));nvm_v2_module_free(&wire);
#endif
   nvm_file_indirect_hosted_free(p);
  }free(bytes);nvm_module_free(m);
 }
#ifdef HOSTED_INSTRUMENT
 NvmFileNominalBindings b;uint32_t pc;NvmModule *m=ih_module(&b,false,false,4,&pc);size_t size;uint8_t *bytes=serialize(m,&size);
 NvmFileIndirectHostedPlan *p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);uint16_t i=ih_index(p,pc);NvmFileIndirectTargets *targets=p->query->targets;
 NvmFileIndirectCall saved=targets->calls[0];uint32_t count=targets->summary.calls;
 memset(p->targets,0,sizeof p->targets);targets->summary.calls=0;CHECK(!file_ih_target_map(p));targets->summary.calls=count;
 targets->calls[0].pc++;memset(p->targets,0,sizeof p->targets);CHECK(!file_ih_target_map(p));targets->calls[0]=saved;
 targets->calls[0].candidates=0;memset(p->targets,0,sizeof p->targets);CHECK(!file_ih_target_map(p));targets->calls[0]=saved;
 memset(p->targets,0,sizeof p->targets);CHECK(file_ih_target_map(p));CHECK(!file_ih_target_map(p)); /* duplicate destination */
 NvmFileCodeInstruction *in=&p->query->ownership->plan->instructions[i];NvmFileCyclicVariant v=p->query->ownership->sites[i].nodes[0]->fact;
 CHECK(file_ih_fact(p,0,i,in,&v));v.body.obligation.parameters++;CHECK(!file_ih_fact(p,0,i,in,&v));
 v=p->query->ownership->sites[i].nodes[0]->fact;v.body.pending_checks=0;CHECK(!file_ih_fact(p,0,i,in,&v));
 NvmFileFlowFunction *second=&p->query->ownership->plan->functions[4].declaration;uint8_t tag=second->result.tag;second->result.tag=TAG_BOOL;
 v=p->query->ownership->sites[i].nodes[0]->fact;CHECK(!file_ih_fact(p,0,i,in,&v));second->result.tag=tag;CHECK(file_ih_fact(p,0,i,in,&v));
 nvm_file_indirect_hosted_free(p);free(bytes);nvm_module_free(m);
#endif
}
static void ih_allocations(void){
#ifdef HOSTED_INSTRUMENT
 NvmFileNominalBindings b;uint32_t pc;NvmModule *m=ih_module(&b,false,false,4,&pc);
 for(unsigned f=0;f<m->function_count;f++){Body c={0};c.n=m->functions[f].code_length;memcpy(c.bytes,m->code+m->functions[f].code_offset,c.n);for(unsigned i=0;i<100;i++)integer(&c);op(&c,OP_RET);setbody(m,f,c);}
 CHECK(m->code_size>4096);for(unsigned i=0;i<80;i++){char name[32];snprintf(name,sizeof name,"indirect-name-%u",i);(void)string(m,name);}
 size_t size;uint8_t *bytes=serialize(m,&size);NvmFileIndirectFlow *reference=NULL;OK(nvm_file_indirect_flow_analyze(m,&reference));
 size_t baseline=tracked_live,basebytes=tracked_bytes;
 NvmFileIndirectHostedPlan *good=ih_expect(bytes,size,NVM_FILE_FLOW_OK);NvmFileIndirectHostedStartup facts;CHECK(nvm_file_indirect_hosted_startup(good,&facts));nvm_file_indirect_hosted_free(good);
 unsigned failures[2]={0},recoveries[2]={0};
 for(unsigned transient=0;transient<2;transient++){
  bool completed=false;single_failure=transient!=0;
  for(int prefix=0;prefix<4096;prefix++){
   tracked_peak=tracked_bytes;failed_calls=0;allocation_budget=prefix;NvmFileIndirectHostedPlan *p=(NvmFileIndirectHostedPlan *)(uintptr_t)1;
   NvmFileFlowStatus got=nvm_file_indirect_hosted_prepare(bytes,size,&p);allocation_budget=-1;
   CHECK(tracked_peak-basebytes<=facts.allocation_bound && (!transient || failed_calls<=1));
   if(got==NVM_FILE_FLOW_OK){CHECK(transient || !failed_calls);NvmFileIndirectHostedStartup actual;CHECK(nvm_file_indirect_hosted_startup(p,&actual));
    CHECK(actual.allocation_bound==facts.allocation_bound && actual.retained_bound==facts.retained_bound && actual.query_storage_peak==facts.query_storage_peak);
    uint8_t *copy=malloc(size);CHECK(copy && nvm_file_indirect_hosted_bytes(p,0,copy,size) && !memcmp(copy,bytes,size));free(copy);
    if(failed_calls)ih_compare(p,reference);
    nvm_file_indirect_hosted_free(p);recoveries[transient]+=failed_calls!=0;
   }else{CHECK(failed_calls && p==(NvmFileIndirectHostedPlan *)(uintptr_t)1 && (got==NVM_FILE_FLOW_MEMORY || got==NVM_FILE_FLOW_UNRESOLVED));failures[transient]++;}
   CHECK(tracked_live==baseline && tracked_bytes==basebytes);
   if(!failed_calls){CHECK(got==NVM_FILE_FLOW_OK);completed=true;break;}
   p=ih_expect(bytes,size,NVM_FILE_FLOW_OK);nvm_file_indirect_hosted_free(p);CHECK(tracked_live==baseline && tracked_bytes==basebytes);
  }
  CHECK(completed && failures[transient]);
 }
 single_failure=false;printf("I checked indirect hosted allocation prefixes %u and transient refusals %u, complete recoveries %u/%u\n",failures[0],failures[1],recoveries[0],recoveries[1]);
 nvm_file_indirect_flow_free(reference);free(bytes);nvm_module_free(m);CHECK(!tracked_live && !tracked_bytes);
#endif
}
int main(void){
 CHECK(prior_file_hosted_fixture_main()==0);
 ih_relations();ih_owned_candidates();ih_candidate_depth();ih_startup_wire();ih_internal_and_arity();ih_allocations();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
 printf("PASS %u private indirect hosted checks; no runtime or service execution\n",checks);return 0;
}
