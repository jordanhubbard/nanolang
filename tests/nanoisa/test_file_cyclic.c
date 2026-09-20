/* I inspect query facts only. No cyclic bytecode or File service executes. */
#define FILE_CYCLIC_ALLOC_TEST
#define FILE_BODY_MAIN prior_file_body_fixture_main
#include "test_file_body.c"
#undef FILE_BODY_MAIN
#include "../../src/nanoisa/file_cyclic.h"

static void jump_to(Body *c,uint32_t pc,uint32_t destination){
 int64_t offset=(int64_t)destination-pc;CHECK(offset>=INT32_MIN && offset<=INT32_MAX);
 wr32(c->bytes+pc+(c->bytes[pc]==OP_FILE_RESULT_BRANCH?3:1),(uint32_t)(int32_t)offset);
}
static void boolean(Body *c){op(c,OP_PUSH_BOOL);op(c,1);}
static void back(Body *c,uint32_t pc){uint32_t at=branch(c,OP_JMP,0);jump_to(c,at,pc);}
static NvmFileCyclicReport *cyclic_expect(NvmModule *m,NvmFileFlowStatus expected){
 NvmFileCyclicReport *r=(NvmFileCyclicReport *)(uintptr_t)1;
 NvmFileFlowStatus actual=nvm_file_cyclic_analyze(m,&r);
 if(actual!=expected)fprintf(stderr,"cyclic expected %u actual %u\n",(unsigned)expected,(unsigned)actual);
 CHECK(actual==expected);
 if(expected!=NVM_FILE_FLOW_OK){CHECK(r==(NvmFileCyclicReport *)(uintptr_t)1);return NULL;}
 CHECK(r && r!=(NvmFileCyclicReport *)(uintptr_t)1);
 NvmFileCyclicSummary s;CHECK(nvm_file_cyclic_summary(r,&s));
 CHECK(s.revision==1 && !s.runtime_admitted && s.variants==s.transfers && s.storage_peak<=NVM_FILE_CYCLIC_BYTES);
 CHECK(s.transfers<=NVM_FILE_CYCLIC_PAIRS && s.edges<=NVM_FILE_CYCLIC_EDGES);
 return r;
}
static uint16_t cyclic_index(NvmFileCyclicReport *r,uint32_t f,uint32_t pc){
 NvmFileCodeFunction fn;CHECK(nvm_file_cyclic_function(r,f,&fn));
 for(uint16_t i=0;i<fn.instruction_count;i++){NvmFileCodeInstruction in;CHECK(nvm_file_cyclic_instruction(r,f,i,&in));if(in.byte_offset==pc)return i;}
 CHECK(false);return 0;
}
static void cyclic_release(NvmFileCyclicReport *r){nvm_file_cyclic_free(r);
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
}
static void same_value(NvmFileFlowValue a,NvmFileFlowValue b){
 CHECK(a.type.tag==b.type.tag && a.type.mode==b.type.mode && a.type.global_index==b.type.global_index &&
       a.type.catalog_ordinal==b.type.catalog_ordinal && a.type.category==b.type.category &&
       a.initialized==b.initialized && a.owner==b.owner && a.arm==b.arm);
}
static void same_info(NvmFileCyclicStateInfo a,NvmFileCyclicStateInfo b){
 CHECK(a.locals==b.locals && a.stack==b.stack && a.regions==b.regions && a.owners==b.owners && a.references==b.references);
}
static void same_body_fact(NvmFileBodyInstruction a,NvmFileBodyInstruction b){
 CHECK(a.reachable==b.reachable && a.exit_checked==b.exit_checked && a.refinement==b.refinement && a.has_obligation==b.has_obligation &&
 a.input_stack==b.input_stack && a.output_stack==b.output_stack && a.cleanup_local==b.cleanup_local && a.cleanup==b.cleanup &&
 a.discharged_checks==b.discharged_checks && a.pending_checks==b.pending_checks);
 if(a.has_obligation){NvmFileFlowObligation x=a.obligation,y=b.obligation;
 CHECK(x.kind==y.kind && x.site==y.site && x.target==y.target && x.checks==y.checks &&
 x.required_rights==y.required_rights && x.acquired_rights==y.acquired_rights && x.parameters==y.parameters &&
 x.owned_inputs==y.owned_inputs && x.borrowed_inputs==y.borrowed_inputs && x.result_count==y.result_count &&
 x.outcomes[0]==y.outcomes[0] && x.outcomes[1]==y.outcomes[1]);
 same_value((NvmFileFlowValue){x.result,false,0,0},(NvmFileFlowValue){y.result,false,0,0});}
}
/* I compare every observable state field and every edge, not only counts. */
static void equal_reports(NvmFileCyclicReport *a,NvmFileCyclicReport *b){
 NvmFileCyclicSummary sa,sb;CHECK(nvm_file_cyclic_summary(a,&sa) && nvm_file_cyclic_summary(b,&sb));
 CHECK(sa.functions==sb.functions && sa.instructions==sb.instructions && sa.variants==sb.variants && sa.transfers==sb.transfers && sa.edges==sb.edges && sa.storage_peak==sb.storage_peak);
 for(uint32_t f=0;f<sa.functions;f++){
  NvmFileCodeFunction fn;CHECK(nvm_file_cyclic_function(a,f,&fn));
  for(uint16_t i=0;i<fn.instruction_count;i++){
   uint8_t na,nb;uint16_t ca,cb;NvmFileCodeInstruction ia,ib;
   CHECK(nvm_file_cyclic_instruction(a,f,i,&ia) && nvm_file_cyclic_instruction(b,f,i,&ib));
   CHECK(ia.byte_offset==ib.byte_offset && ia.catalog_ordinal==ib.catalog_ordinal && ia.successor_count==ib.successor_count && ia.successors[0]==ib.successors[0] && ia.successors[1]==ib.successors[1]);
   CHECK(ia.decoded.opcode==ib.decoded.opcode && ia.decoded.byte_length==ib.decoded.byte_length && !memcmp(ia.decoded.operands,ib.decoded.operands,sizeof ia.decoded.operands));
   CHECK(nvm_file_cyclic_component(a,f,i,&ca) && nvm_file_cyclic_component(b,f,i,&cb) && ca==cb);
   CHECK(nvm_file_cyclic_variant_count(a,f,i,&na) && nvm_file_cyclic_variant_count(b,f,i,&nb) && na==nb);
   for(uint8_t v=0;v<na;v++){
    NvmFileCyclicVariant va,vb;CHECK(nvm_file_cyclic_variant(a,f,i,v,&va) && nvm_file_cyclic_variant(b,f,i,v,&vb));
    same_info(va.input,vb.input);same_info(va.output,vb.output);same_body_fact(va.body,vb.body);
    CHECK(va.edge_mask==vb.edge_mask && va.edge_variants[0]==vb.edge_variants[0] && va.edge_variants[1]==vb.edge_variants[1]);
    for(uint16_t l=0;l<va.input.locals;l++){NvmFileFlowValue x,y;CHECK(nvm_file_cyclic_input_local(a,f,i,v,l,&x) && nvm_file_cyclic_input_local(b,f,i,v,l,&y));same_value(x,y);}
    for(uint16_t l=0;l<va.input.stack;l++){NvmFileFlowValue x,y;CHECK(nvm_file_cyclic_input_stack(a,f,i,v,l,&x) && nvm_file_cyclic_input_stack(b,f,i,v,l,&y));same_value(x,y);}
    for(uint16_t l=0;l<NVM_FILE_FLOW_REFERENCES;l++){
     NvmFileFlowReference x,y;CHECK(nvm_file_cyclic_input_reference(a,f,i,v,l,&x) && nvm_file_cyclic_input_reference(b,f,i,v,l,&y));
     CHECK(x.live==y.live && x.formal==y.formal && x.local==y.local && x.owner==y.owner && x.identity==y.identity && x.region==y.region);
    }
    for(uint16_t l=0;l<va.input.regions;l++){uint64_t x,y;CHECK(nvm_file_cyclic_input_region(a,f,i,v,l,&x) && nvm_file_cyclic_input_region(b,f,i,v,l,&y) && x==y);}
    for(uint8_t e=0;e<ia.successor_count;e++)if(va.edge_mask&(1u<<e)){
     NvmFileCyclicVariant dest;CHECK(nvm_file_cyclic_variant(a,f,ia.successors[e],va.edge_variants[e],&dest));
     CHECK(dest.input.stack==va.output.stack && dest.input.owners==va.output.owners);
    }
   }
  }
 }
 for(uint32_t k=0;k<8;k++){NvmFileNominalLayout x,y;CHECK(nvm_file_cyclic_type(a,k,&x) && nvm_file_cyclic_type(b,k,&y));CHECK(x.global_index==y.global_index && x.catalog_ordinal==y.catalog_ordinal && x.source_ordinal==y.source_ordinal && x.layout_kind==y.layout_kind && x.ownership_flags==y.ownership_flags && x.category==y.category);}
 for(uint32_t k=0;k<5;k++){uint32_t x,y;CHECK(nvm_file_cyclic_import(a,k,&x) && nvm_file_cyclic_import(b,k,&y) && x==y);}
}
static Body scalar_loop(bool init,bool load,uint32_t *header){
 Body c={0};if(init){integer(&c);one(&c,OP_STORE_LOCAL,4);}*header=c.n;
 boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);integer(&c);one(&c,OP_STORE_LOCAL,4);back(&c,*header);
 target(&c,leave);if(load){one(&c,OP_LOAD_LOCAL,4);op(&c,OP_RET);}else retint(&c);return c;
}
static void initial_alternatives(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);uint32_t header;Body c=scalar_loop(false,false,&header);setbody(m,0,c);
 NvmFileCyclicReport *r=cyclic_expect(m,NVM_FILE_FLOW_OK);uint16_t i=cyclic_index(r,0,header);uint8_t count;CHECK(nvm_file_cyclic_variant_count(r,0,i,&count) && count==2);
 unsigned initialized=0;for(uint8_t v=0;v<count;v++){NvmFileFlowValue x;CHECK(nvm_file_cyclic_input_local(r,0,i,v,4,&x));initialized+=x.initialized;}
 CHECK(initialized==1);cyclic_release(r);
 c=scalar_loop(false,true,&header);setbody(m,0,c);CHECK(!cyclic_expect(m,NVM_FILE_FLOW_INVALID));
 c=scalar_loop(true,true,&header);setbody(m,0,c);r=cyclic_expect(m,NVM_FILE_FLOW_OK);
 CHECK(nvm_file_cyclic_variant_count(r,0,cyclic_index(r,0,header),&count) && count==1);
 NvmFileCyclicReport *second=cyclic_expect(m,NVM_FILE_FLOW_OK);equal_reports(r,second);
 memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);equal_reports(r,second);
 NvmFileFlowValue sentinel={0},before=sentinel;CHECK(!nvm_file_cyclic_input_local(r,UINT32_MAX,0,0,0,&sentinel) && !memcmp(&sentinel,&before,sizeof before));
 nvm_file_cyclic_free(second);cyclic_release(r);
}
static Body owner_loop(NvmFileNominalBindings b,unsigned mode,uint32_t *header){
 Body c={0};service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);uint32_t failed=branch(&c,OP_FILE_RESULT_BRANCH,1);
 take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);
 if(mode==1 || mode==2){op(&c,OP_REGION_BEGIN);op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);}
 *header=c.n;boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 uint32_t replacement_error=UINT32_MAX;
 if(mode==1){service(&c,b,3,20);op(&c,OP_POP);integer(&c);op(&c,OP_CALL_REF);u32(&c,2);u16(&c,20);op(&c,OP_POP);}
 else {
  one(&c,OP_OWN_MOVE_LOCAL,0);service(&c,b,4,UINT16_MAX);op(&c,OP_POP);
  service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);replacement_error=branch(&c,OP_FILE_RESULT_BRANCH,1);
  take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);
 }
 back(&c,*header);target(&c,leave);
 if(mode==1 || mode==2){one(&c,OP_FILE_END_BORROW,20);op(&c,OP_REGION_END);}
 one(&c,OP_OWN_MOVE_LOCAL,0);service(&c,b,4,UINT16_MAX);op(&c,OP_POP);retint(&c);
 if(replacement_error!=UINT32_MAX){target(&c,replacement_error);take_result(&c,1,1);op(&c,OP_POP);retint(&c);}
 target(&c,failed);take_result(&c,1,1);op(&c,OP_POP);retint(&c);return c;
}
static void owner_relations(void){
 for(unsigned permutation=0;permutation<2;permutation++)for(unsigned mode=0;mode<3;mode++){
  NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,permutation!=0);uint32_t header;setbody(m,0,owner_loop(b,mode,&header));
  NvmFileCyclicReport *r=cyclic_expect(m,mode==2?NVM_FILE_FLOW_INVALID:NVM_FILE_FLOW_OK);
  if(r){NvmFileCyclicReport *repeat=cyclic_expect(m,NVM_FILE_FLOW_OK);equal_reports(r,repeat);nvm_file_cyclic_free(repeat);
   uint16_t at=cyclic_index(r,0,header);uint8_t count;CHECK(nvm_file_cyclic_variant_count(r,0,at,&count) && count==1);
   NvmFileFlowValue owner;NvmFileFlowReference ref;NvmFileCyclicVariant fact;
   CHECK(nvm_file_cyclic_input_local(r,0,at,0,0,&owner) && owner.owner==1);
   CHECK(nvm_file_cyclic_input_reference(r,0,at,0,20,&ref) && ref.live==(mode==1));
   CHECK(nvm_file_cyclic_variant(r,0,at,0,&fact) && fact.input.owners==1);
   if(mode==1)CHECK(ref.owner==owner.owner && ref.identity==789 && ref.region==513 && fact.input.references==1);
   NvmFileCodePlan *old=(NvmFileCodePlan *)(uintptr_t)1;CHECK(nvm_file_code_prepare(m,&old)==NVM_FILE_FLOW_UNRESOLVED && old==(NvmFileCodePlan *)(uintptr_t)1);
   NvmFileBodyReport *body=(NvmFileBodyReport *)(uintptr_t)1;CHECK(nvm_file_body_analyze(m,&body)==NVM_FILE_FLOW_UNRESOLVED && body==(NvmFileBodyReport *)(uintptr_t)1);
   CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);cyclic_release(r);
  }nvm_module_free(m);
 }
}
/* Distinct initialization patterns produce exactly16 and17 alternatives at one
 * real decoded join; no production constant or interning helper is altered. */
static NvmModule *alternatives(unsigned n,uint32_t *join){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);
 for(unsigned l=4;l<=8;l++)desc(m->ownership_data+24+12+8*l,TAG_INT,0,NVM_V2_NO_INDEX);
 Body c={0};uint32_t ends[17];CHECK(n<=17);
 for(unsigned mask=0;mask<n;mask++){
  uint32_t next=UINT32_MAX;if(mask+1<n){boolean(&c);next=branch(&c,OP_JMP_FALSE,0);}
  for(unsigned bit=0;bit<5;bit++)if(mask&(1u<<bit)){integer(&c);one(&c,OP_STORE_LOCAL,(uint16_t)(4+bit));}
  ends[mask]=branch(&c,OP_JMP,0);if(next!=UINT32_MAX)target(&c,next);
 }
 *join=c.n;op(&c,OP_NOP);retint(&c);for(unsigned j=0;j<n;j++)jump_to(&c,ends[j],*join);setbody(m,0,c);return m;
}
static void alternative_boundary(void){
 uint32_t join;NvmModule *m=alternatives(16,&join);NvmFileCyclicReport *r=cyclic_expect(m,NVM_FILE_FLOW_OK);
 uint8_t count;uint16_t at=cyclic_index(r,0,join);CHECK(nvm_file_cyclic_variant_count(r,0,at,&count) && count==16);
 uint32_t seen=0;for(uint8_t v=0;v<count;v++){unsigned mask=0;for(unsigned bit=0;bit<5;bit++){NvmFileFlowValue x;CHECK(nvm_file_cyclic_input_local(r,0,at,v,(uint16_t)(4+bit),&x));if(x.initialized)mask|=1u<<bit;}CHECK(mask<16 && !(seen&(1u<<mask)));seen|=1u<<mask;}
 CHECK(seen==65535);cyclic_release(r);nvm_module_free(m);
 m=alternatives(17,&join);CHECK(!cyclic_expect(m,NVM_FILE_FLOW_LIMIT));nvm_module_free(m);
}
static void owner_empty_and_lower_callee(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);Body c={0};uint32_t header=c.n;
 boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);uint32_t error=branch(&c,OP_FILE_RESULT_BRANCH,1);
 take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);op(&c,OP_REGION_BEGIN);
 op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);service(&c,b,3,20);op(&c,OP_POP);
 one(&c,OP_FILE_END_BORROW,20);op(&c,OP_REGION_END);uint32_t drop=c.n;one(&c,OP_FILE_DROP_LOCAL,0);back(&c,header);
 target(&c,error);take_result(&c,1,1);op(&c,OP_POP);back(&c,header);target(&c,leave);retint(&c);setbody(m,0,c);
 NvmFileCyclicReport *r=cyclic_expect(m,NVM_FILE_FLOW_OK);uint8_t count;CHECK(nvm_file_cyclic_variant_count(r,0,cyclic_index(r,0,header),&count) && count==1);
 NvmFileCyclicVariant fact;CHECK(nvm_file_cyclic_variant(r,0,cyclic_index(r,0,drop),0,&fact));
 CHECK(fact.body.cleanup==NVM_FILE_BODY_CLEANUP_DROP_LOCAL && fact.input.owners==1 && fact.output.owners==0);cyclic_release(r);
 c=(Body){0};op(&c,OP_CALL);u32(&c,0);op(&c,OP_RET);setbody(m,3,c);
 r=cyclic_expect(m,NVM_FILE_FLOW_OK);CHECK(nvm_file_cyclic_variant(r,3,0,0,&fact) && fact.body.has_obligation && fact.body.obligation.target==0 && (fact.body.discharged_checks & NVM_FILE_FLOW_CHECK_CALLEE));cyclic_release(r);nvm_module_free(m);
}
static void acyclic_equivalence(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,true);setbody(m,0,lifecycle_code(b));
 NvmFileBodyReport *old=NULL;OK(nvm_file_body_analyze(m,&old));NvmFileCyclicReport *r=cyclic_expect(m,NVM_FILE_FLOW_OK);
 for(uint32_t f=0;f<5;f++){NvmFileCodeFunction fn;CHECK(nvm_file_body_function(old,f,&fn));
  for(uint16_t i=0;i<fn.instruction_count;i++){
   NvmFileCodeInstruction in;NvmFileBodyInstruction oldfact;CHECK(nvm_file_body_instruction(old,f,i,&in,&oldfact));
   uint8_t count;CHECK(nvm_file_cyclic_variant_count(r,f,i,&count) && count==(oldfact.reachable?1:0));
   if(count){NvmFileCyclicVariant fact;CHECK(nvm_file_cyclic_variant(r,f,i,0,&fact));same_body_fact(oldfact,fact.body);}
  }
 }
 nvm_file_body_free(old);cyclic_release(r);
 NvmFileFlowDeclarations *d=NULL;OK(nvm_file_flow_declarations(m,&d));NvmFileFlowState *state_before=opened(d,b);
 OK(nvm_file_flow_drop_local(state_before,0));NvmFileFlowCounts before=counts(state_before);CHECK(before.cleanup_obligations==1);
#ifdef FLOW_INSTRUMENT
 uint64_t original_next=d->next_identity;
#endif
 r=cyclic_expect(m,NVM_FILE_FLOW_OK);NvmFileFlowCounts after=counts(state_before);
 CHECK(after.cleanup_obligations==before.cleanup_obligations && after.obligations==before.obligations && after.owners==before.owners);
#ifdef FLOW_INSTRUMENT
 CHECK(original_next==d->next_identity);
#endif
 nvm_file_flow_state_free(state_before);nvm_file_flow_declarations_free(d);cyclic_release(r);nvm_module_free(m);
}
static void nested_cycles(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);Body c={0};integer(&c);one(&c,OP_STORE_LOCAL,4);
 uint32_t outer=c.n;boolean(&c);uint32_t exit=branch(&c,OP_JMP_FALSE,0);
 uint32_t inner=c.n;boolean(&c);uint32_t done=branch(&c,OP_JMP_FALSE,0);back(&c,inner);
 target(&c,done);back(&c,outer);target(&c,exit);one(&c,OP_LOAD_LOCAL,4);op(&c,OP_RET);setbody(m,0,c);
 NvmFileCyclicReport *r=cyclic_expect(m,NVM_FILE_FLOW_OK);uint16_t a,bcomponent;
 CHECK(nvm_file_cyclic_component(r,0,cyclic_index(r,0,outer),&a) && nvm_file_cyclic_component(r,0,cyclic_index(r,0,inner),&bcomponent) && a==bcomponent);
 cyclic_release(r);nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};uint32_t header=c.n;boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);op(&c,OP_REGION_BEGIN);back(&c,header);target(&c,leave);retint(&c);setbody(m,0,c);
 /* An exit already exposes an unbalanced region; it must refuse rather than
  * waiting for a variant limit or discarding that region on the backedge. */
 CHECK(!cyclic_expect(m,NVM_FILE_FLOW_INVALID));nvm_module_free(m);
}
static void query_refusals_and_limits(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);Body c={0};
 uint32_t code_size=m->code_size;m->code_size=NVM_FILE_CODE_BYTES+1;CHECK(!cyclic_expect(m,NVM_FILE_FLOW_LIMIT));m->code_size=code_size;
 uint32_t functions=m->function_count;m->function_count=NVM_FILE_FLOW_FUNCTIONS+1;CHECK(!cyclic_expect(m,NVM_FILE_FLOW_LIMIT));m->function_count=functions;
 uint16_t locals=m->functions[0].local_count;m->functions[0].local_count=NVM_FILE_FLOW_LOCALS+1;CHECK(!cyclic_expect(m,NVM_FILE_FLOW_LIMIT));m->functions[0].local_count=locals;
 op(&c,OP_JMP);u32(&c,0);setbody(m,0,c);CHECK(!cyclic_expect(m,NVM_FILE_FLOW_INVALID));nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};op(&c,OP_CALL);u32(&c,0);op(&c,OP_RET);setbody(m,0,c);CHECK(!cyclic_expect(m,NVM_FILE_FLOW_UNRESOLVED));nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};retint(&c);op(&c,OP_CALL_INDIRECT);u32(&c,0);op(&c,OP_RET);setbody(m,0,c);CHECK(!cyclic_expect(m,NVM_FILE_FLOW_UNRESOLVED));nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};retint(&c);one(&c,OP_LOAD_LOCAL,UINT16_MAX);op(&c,OP_RET);setbody(m,0,c);CHECK(!cyclic_expect(m,NVM_FILE_FLOW_INVALID));nvm_module_free(m);
 m=bodymodule(&b,false);c=(Body){0};for(unsigned i=0;i<254;i++)op(&c,OP_NOP);retint(&c);setbody(m,0,c);
 NvmFileCyclicReport *r=cyclic_expect(m,NVM_FILE_FLOW_OK);cyclic_release(r);
 c=(Body){0};for(unsigned i=0;i<255;i++)op(&c,OP_NOP);retint(&c);setbody(m,0,c);CHECK(!cyclic_expect(m,NVM_FILE_FLOW_LIMIT));nvm_module_free(m);
}
static void decoded_owner_swap(void){
 for(unsigned held=0;held<2;held++){
  NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);
  desc(m->ownership_data+24+12+8*4,TAG_STRUCT,0,b.layouts[0]);
  Body c={0};service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);
  uint32_t first_error=branch(&c,OP_FILE_RESULT_BRANCH,1);
  take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);
  service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);
  uint32_t second_error=branch(&c,OP_FILE_RESULT_BRANCH,1);
  take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,9);
  if(held){op(&c,OP_REGION_BEGIN);op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);}
  uint32_t header=c.n;boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
  uint32_t move[6];
  move[0]=c.n;one(&c,OP_OWN_MOVE_LOCAL,0);move[1]=c.n;one(&c,OP_OWN_STORE_LOCAL,4);
  move[2]=c.n;one(&c,OP_OWN_MOVE_LOCAL,9);move[3]=c.n;one(&c,OP_OWN_STORE_LOCAL,0);
  move[4]=c.n;one(&c,OP_OWN_MOVE_LOCAL,4);move[5]=c.n;one(&c,OP_OWN_STORE_LOCAL,9);
  back(&c,header);target(&c,leave);
  if(held){one(&c,OP_FILE_END_BORROW,20);op(&c,OP_REGION_END);}
  one(&c,OP_FILE_DROP_LOCAL,0);one(&c,OP_FILE_DROP_LOCAL,9);retint(&c);
  target(&c,second_error);take_result(&c,1,1);op(&c,OP_POP);one(&c,OP_FILE_DROP_LOCAL,0);retint(&c);
  target(&c,first_error);take_result(&c,1,1);op(&c,OP_POP);retint(&c);setbody(m,0,c);
  NvmFileCyclicReport *r=cyclic_expect(m,held?NVM_FILE_FLOW_INVALID:NVM_FILE_FLOW_OK);
  if(r){uint8_t count;CHECK(nvm_file_cyclic_variant_count(r,0,cyclic_index(r,0,header),&count) && count==1);
   for(unsigned j=0;j<6;j++){NvmFileCyclicVariant fact;uint16_t at=cyclic_index(r,0,move[j]);
    CHECK(nvm_file_cyclic_variant(r,0,at,0,&fact) && fact.input.owners==2 && fact.output.owners==2 && !fact.input.references);
    CHECK(fact.input.stack==(j%2) && fact.output.stack==1-(j%2));
    if(j%2){NvmFileFlowValue operand;CHECK(nvm_file_cyclic_input_stack(r,0,at,0,0,&operand) && operand.owner==2);}
   }
   NvmFileFlowValue a,bvalue,temp;uint16_t at=cyclic_index(r,0,header);
   CHECK(nvm_file_cyclic_input_local(r,0,at,0,0,&a) && nvm_file_cyclic_input_local(r,0,at,0,9,&bvalue) && nvm_file_cyclic_input_local(r,0,at,0,4,&temp));
   CHECK(a.owner==1 && bvalue.owner==2 && !temp.initialized && !temp.owner);cyclic_release(r);
  }
  nvm_module_free(m);
 }
}
static void canonical_relation_controls(void){
#ifdef FLOW_INSTRUMENT
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);NvmFileFlowDeclarations *d=NULL;OK(nvm_file_flow_declarations(m,&d));
 NvmFileFlowState *s=state(d,0);FileCyclicWorkspace *w=calloc(1,sizeof *w);FileCyclicState *saved=calloc(1,sizeof *saved);CHECK(w && saved);
 s->locals[0].initialized=s->locals[9].initialized=true;s->locals[0].owner=7000;s->locals[9].owner=9000;
 s->region_count=1;s->regions[0]=333;s->references[20]=(NvmFileFlowReference){true,false,0,7000,444,333};
 w->transfer=*s;OK(file_cyclic_canonicalize(w));*saved=w->canonical;
 CHECK(saved->locals[0].owner==1 && saved->locals[9].owner==2 && saved->references[20].owner==1);
 s->locals[0].owner=2222;s->locals[9].owner=1111;s->references[20].owner=2222;s->references[20].identity=555;
 w->transfer=*s;OK(file_cyclic_canonicalize(w));CHECK(file_cyclic_equal(saved,&w->canonical));
 CHECK(nvm_file_flow_take(s,0)==NVM_FILE_FLOW_INVALID);
 s->locals[9].owner=2222;w->transfer=*s;CHECK(file_cyclic_canonicalize(w)==NVM_FILE_FLOW_INVALID);
 s->locals[9].owner=1111;s->references[20].owner=1111;w->transfer=*s;CHECK(file_cyclic_canonicalize(w)==NVM_FILE_FLOW_INVALID);
 s->references[20].owner=2222;s->references[21]=s->references[20];s->references[21].identity=666;
 w->transfer=*s;CHECK(file_cyclic_canonicalize(w)==NVM_FILE_FLOW_INVALID);
 free(saved);free(w);nvm_file_flow_state_free(s);nvm_file_flow_declarations_free(d);nvm_module_free(m);CHECK(!live);
#endif
}
static void cyclic_allocations(void){
#ifdef FLOW_INSTRUMENT
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);uint32_t header;setbody(m,0,owner_loop(b,0,&header));
 allocations=0;NvmFileCyclicReport *r=cyclic_expect(m,NVM_FILE_FLOW_OK);unsigned total=allocations;cyclic_release(r);CHECK(total>20 && total<2048);
 for(unsigned fail=0;fail<total;fail++){
  budget=(int)fail;CHECK(!cyclic_expect(m,NVM_FILE_FLOW_MEMORY));budget=-1;CHECK(!live);
  r=cyclic_expect(m,NVM_FILE_FLOW_OK);cyclic_release(r);
 }
 for(unsigned fail=1;fail<=total;fail++){
  allocations=0;transient_allocation=fail;CHECK(!cyclic_expect(m,NVM_FILE_FLOW_MEMORY));transient_allocation=0;CHECK(!live);
  r=cyclic_expect(m,NVM_FILE_FLOW_OK);cyclic_release(r);
 }
 nvm_module_free(m);
 /* Exact budget arithmetic and unreachable maximum counters are white-box
  * query-only controls, not claims of constructing a maximal wire program. */
 NvmFileCyclicReport counted={0};counted.summary.storage_peak=NVM_FILE_CYCLIC_BYTES-sizeof(FileCyclicNode);
 CHECK(file_cyclic_room(&counted,1,sizeof(FileCyclicNode)) && counted.summary.storage_peak==NVM_FILE_CYCLIC_BYTES);
 CHECK(!file_cyclic_room(&counted,1,1));counted.summary.storage_peak=0;CHECK(!file_cyclic_room(&counted,SIZE_MAX,2));
 m=bodymodule(&b,false);NvmFileCodePlan *plan=NULL;OK(file_code_prepare_mode(m,&plan,false));
 counted=(NvmFileCyclicReport){0};counted.plan=plan;counted.sites=calloc(plan->instruction_count,sizeof *counted.sites);CHECK(counted.sites);
 FileCyclicWorkspace *w=calloc(1,sizeof *w);CHECK(w);uint8_t out=99;
 counted.summary.variants=NVM_FILE_CYCLIC_PAIRS;CHECK(file_cyclic_intern(&counted,w,0,0,&out)==NVM_FILE_FLOW_LIMIT && out==99);
 counted.summary.variants=0;w->tail=NVM_FILE_CYCLIC_FUNCTION_PAIRS;CHECK(file_cyclic_intern(&counted,w,0,0,&out)==NVM_FILE_FLOW_LIMIT && out==99);
 /* The exact transfer/edge guards are not normally reachable before tighter
  * memory limits. I set only report counters, then analyze real decoded code. */
 size_t declarations=sizeof(NvmFileFlowDeclarations);for(uint32_t f=0;f<plan->count;f++)declarations+=plan->functions[f].declaration.locals*sizeof(NvmFileFlowDeclaration);
 NvmFileFlowDeclarations *scratch=malloc(declarations);CHECK(scratch);memcpy(scratch,plan->declarations,declarations);
 for(unsigned guard=0;guard<2;guard++){
  memset(w,0,sizeof *w);w->body.plan=plan;counted.summary=(NvmFileCyclicSummary){0};
  if(guard==0)counted.summary.transfers=NVM_FILE_CYCLIC_PAIRS;else counted.summary.edges=NVM_FILE_CYCLIC_EDGES;
  CHECK(file_cyclic_function_analyze(&counted,w,scratch,0)==NVM_FILE_FLOW_LIMIT);
  CHECK(counted.summary.variants==1 && counted.sites[0].count==1 && !counted.sites[0].nodes[0]->processed);
  if(guard==0)CHECK(counted.summary.transfers==NVM_FILE_CYCLIC_PAIRS && !counted.summary.edges);
  else CHECK(counted.summary.edges==NVM_FILE_CYCLIC_EDGES && counted.summary.transfers==1);
  flow_free(counted.sites[0].nodes[0]);counted.sites[0].nodes[0]=NULL;counted.sites[0].count=0;
 }
 free(scratch);free(w);free(counted.sites);nvm_file_code_free(plan);nvm_module_free(m);CHECK(!live);
 printf("I checked %u allocation prefixes and %u transient query failures\n",total,total);
#endif
}
int main(void){
 CHECK(prior_file_body_fixture_main()==0);
 initial_alternatives();owner_relations();alternative_boundary();owner_empty_and_lower_callee();acyclic_equivalence();nested_cycles();query_refusals_and_limits();decoded_owner_swap();canonical_relation_controls();cyclic_allocations();
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
 printf("PASS %u private cyclic query checks; no pending module execution\n",checks);return 0;
}
