/* I prepare bytes and compare facts only; no cyclic File module executes. */
#define FILE_HOSTED_MAIN prior_file_hosted_fixture_main
#include "test_file_hosted.c"
#undef FILE_HOSTED_MAIN
#include "../../src/nanoisa/file_cyclic_hosted.h"

static void ch_boolean(Body *c){op(c,OP_PUSH_BOOL);op(c,1);}
static void ch_back(Body *c,uint32_t pc){uint32_t at=branch(c,OP_JMP,0);CHECK(pc<=at);wr32(c->bytes+at+1,(uint32_t)(int32_t)((int64_t)pc-at));}
static Body ch_scalar_loop(uint32_t *header){
 Body c={0};*header=c.n;ch_boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 integer(&c);one(&c,OP_STORE_LOCAL,4);ch_back(&c,*header);target(&c,leave);retint(&c);return c;
}
static Body ch_owner_loop(NvmFileNominalBindings b,uint32_t *header){
 Body c={0};service(&c,b,0,UINT16_MAX);one(&c,OP_OWN_STORE_LOCAL,1);uint32_t failed=branch(&c,OP_FILE_RESULT_BRANCH,1);
 take_result(&c,1,0);one(&c,OP_OWN_STORE_LOCAL,0);op(&c,OP_REGION_BEGIN);
 op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);*header=c.n;
 ch_boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 integer(&c);service(&c,b,1,20);op(&c,OP_POP);service(&c,b,2,20);op(&c,OP_POP);service(&c,b,3,20);op(&c,OP_POP);
 integer(&c);op(&c,OP_CALL_REF);u32(&c,2);u16(&c,20);op(&c,OP_POP);ch_back(&c,*header);
 target(&c,leave);one(&c,OP_FILE_END_BORROW,20);op(&c,OP_REGION_END);
 one(&c,OP_OWN_MOVE_LOCAL,0);op(&c,OP_CALL);u32(&c,1);service(&c,b,4,UINT16_MAX);op(&c,OP_POP);retint(&c);
 target(&c,failed);take_result(&c,1,1);op(&c,OP_POP);retint(&c);return c;
}
static NvmFileCyclicHostedPlan *ch_expect(const uint8_t *bytes,size_t size,NvmFileFlowStatus expected){
 NvmFileCyclicHostedPlan *p=(NvmFileCyclicHostedPlan *)(uintptr_t)1;
 NvmFileFlowStatus got=nvm_file_cyclic_hosted_prepare(bytes,size,&p);
 if(got!=expected)fprintf(stderr,"cyclic hosted expected %u actual %u\n",(unsigned)expected,(unsigned)got);
 CHECK(got==expected);
 if(expected!=NVM_FILE_FLOW_OK){CHECK(p==(NvmFileCyclicHostedPlan *)(uintptr_t)1);return NULL;}
 CHECK(p && p!=(NvmFileCyclicHostedPlan *)(uintptr_t)1);NvmFileCyclicHostedStartup s;
 CHECK(nvm_file_cyclic_hosted_startup(p,&s) && s.revision==1 && !s.runtime_admitted && s.input_bytes==size);
 CHECK(s.query_storage_peak<=NVM_FILE_CYCLIC_BYTES && s.retained_bound>=size+s.query_storage_peak &&
       s.retained_bound<=s.allocation_bound && s.allocation_bound<=NVM_FILE_HOSTED_BYTES);
 return p;
}
static void ch_value(NvmFileFlowValue a,NvmFileFlowValue b){
 CHECK(a.type.tag==b.type.tag && a.type.mode==b.type.mode && a.type.global_index==b.type.global_index &&
 a.type.catalog_ordinal==b.type.catalog_ordinal && a.type.category==b.type.category &&
 a.initialized==b.initialized && a.owner==b.owner && a.arm==b.arm);
}
static void ch_info(NvmFileCyclicStateInfo a,NvmFileCyclicStateInfo b){CHECK(a.locals==b.locals && a.stack==b.stack && a.owners==b.owners && a.references==b.references && a.regions==b.regions);}
static void ch_body(NvmFileBodyInstruction a,NvmFileBodyInstruction b){
 CHECK(a.reachable==b.reachable && a.exit_checked==b.exit_checked && a.refinement==b.refinement &&
 a.has_obligation==b.has_obligation && a.input_stack==b.input_stack && a.output_stack==b.output_stack &&
 a.cleanup_local==b.cleanup_local && a.cleanup==b.cleanup && a.discharged_checks==b.discharged_checks && a.pending_checks==b.pending_checks);
 NvmFileFlowObligation x=a.obligation,y=b.obligation;
 CHECK(x.kind==y.kind && x.site==y.site && x.target==y.target && x.checks==y.checks &&
 x.required_rights==y.required_rights && x.acquired_rights==y.acquired_rights && x.parameters==y.parameters &&
 x.owned_inputs==y.owned_inputs && x.borrowed_inputs==y.borrowed_inputs && x.result_count==y.result_count &&
 x.outcomes[0]==y.outcomes[0] && x.outcomes[1]==y.outcomes[1]);
 ch_value((NvmFileFlowValue){x.result,false,0,0},(NvmFileFlowValue){y.result,false,0,0});
}
static unsigned ch_compare(NvmFileCyclicHostedPlan *p,NvmFileCyclicReport *q){
 NvmFileCyclicSummary a,b;CHECK(nvm_file_cyclic_hosted_query_summary(p,&a) && nvm_file_cyclic_summary(q,&b));
 CHECK(a.revision==b.revision && a.functions==b.functions && a.instructions==b.instructions && a.variants==b.variants &&
 a.transfers==b.transfers && a.edges==b.edges && a.storage_peak==b.storage_peak && !a.runtime_admitted && !b.runtime_admitted);
 uint64_t visited=0;unsigned seen_services=0,calls=0,borrowed=0;
 for(uint32_t rank=0;rank<a.functions;rank++){uint32_t f;CHECK(nvm_file_cyclic_hosted_function_order(p,rank,&f));CHECK(f<a.functions && !(visited&(UINT64_C(1)<<f)));visited|=UINT64_C(1)<<f;}
 for(uint32_t f=0;f<a.functions;f++){
  NvmFileCyclicHostedFunction hf;NvmFileCodeFunction qf;CHECK(nvm_file_cyclic_hosted_function(p,f,&hf) && nvm_file_cyclic_function(q,f,&qf));
  CHECK(hf.code.code_offset==qf.code_offset && hf.code.code_length==qf.code_length && hf.code.instruction_count==qf.instruction_count &&
        hf.locals==qf.declaration.locals && hf.code.declaration.parameters==qf.declaration.parameters && hf.entry_variant==0);
  ch_value((NvmFileFlowValue){hf.code.declaration.result,false,0,0},(NvmFileFlowValue){qf.declaration.result,false,0,0});
  uint16_t peak=0,owners=0,refs=0,regions=0;
  for(uint16_t l=0;l<hf.locals;l++){
   NvmFileFlowDeclaration hd,qd;CHECK(nvm_file_cyclic_hosted_local(p,f,l,&hd) && nvm_file_cyclic_local(q,f,l,&qd));
   ch_value((NvmFileFlowValue){hd,false,0,0},(NvmFileFlowValue){qd,false,0,0});
   NvmFileFlowValue seed;CHECK(nvm_file_cyclic_hosted_input_local(p,f,0,hf.entry_variant,l,&seed));
   CHECK(seed.initialized==(l<qf.declaration.parameters));
  }
  for(uint16_t i=0;i<qf.instruction_count;i++){
   NvmFileCodeInstruction hi,qi;uint8_t hn,qn;uint16_t hc,qc;
   CHECK(nvm_file_cyclic_hosted_instruction(p,f,i,&hi) && nvm_file_cyclic_instruction(q,f,i,&qi));
   CHECK(hi.byte_offset==qi.byte_offset && hi.catalog_ordinal==qi.catalog_ordinal && hi.successor_count==qi.successor_count &&
         hi.successors[0]==qi.successors[0] && hi.successors[1]==qi.successors[1] && hi.decoded.opcode==qi.decoded.opcode &&
         hi.decoded.byte_length==qi.decoded.byte_length && !memcmp(hi.decoded.operands,qi.decoded.operands,sizeof hi.decoded.operands) &&
         !memcmp(hi.decoded.operand_types,qi.decoded.operand_types,sizeof hi.decoded.operand_types));
   CHECK(nvm_file_cyclic_hosted_component(p,f,i,&hc) && nvm_file_cyclic_component(q,f,i,&qc) && hc==qc);
   CHECK(nvm_file_cyclic_hosted_variant_count(p,f,i,&hn) && nvm_file_cyclic_variant_count(q,f,i,&qn) && hn==qn);
   for(uint8_t v=0;v<hn;v++){
    NvmFileCyclicVariant x,y;CHECK(nvm_file_cyclic_hosted_variant(p,f,i,v,&x) && nvm_file_cyclic_variant(q,f,i,v,&y));
    ch_info(x.input,y.input);ch_info(x.output,y.output);ch_body(x.body,y.body);
    CHECK(x.edge_mask==y.edge_mask && x.edge_variants[0]==y.edge_variants[0] && x.edge_variants[1]==y.edge_variants[1]);
    NvmFileCyclicStateInfo infos[2]={x.input,x.output};for(unsigned side=0;side<2;side++){
     if(infos[side].stack>peak)peak=infos[side].stack;
     if(infos[side].owners>owners)owners=infos[side].owners;
     if(infos[side].references>refs)refs=infos[side].references;
     if(infos[side].regions>regions)regions=infos[side].regions;
    }
    for(uint16_t l=0;l<x.input.locals;l++){NvmFileFlowValue hv,qv;CHECK(nvm_file_cyclic_hosted_input_local(p,f,i,v,l,&hv) && nvm_file_cyclic_input_local(q,f,i,v,l,&qv));ch_value(hv,qv);}
    for(uint16_t l=0;l<x.input.stack;l++){NvmFileFlowValue hv,qv;CHECK(nvm_file_cyclic_hosted_input_stack(p,f,i,v,l,&hv) && nvm_file_cyclic_input_stack(q,f,i,v,l,&qv));ch_value(hv,qv);}
    for(uint16_t l=0;l<NVM_FILE_FLOW_REFERENCES;l++){
     NvmFileFlowReference hv,qv;CHECK(nvm_file_cyclic_hosted_input_reference(p,f,i,v,l,&hv) && nvm_file_cyclic_input_reference(q,f,i,v,l,&qv));
     CHECK(hv.live==qv.live && hv.formal==qv.formal && hv.local==qv.local && hv.owner==qv.owner && hv.identity==qv.identity && hv.region==qv.region);
    }
    for(uint16_t l=0;l<x.input.regions;l++){uint64_t hv,qv;CHECK(nvm_file_cyclic_hosted_input_region(p,f,i,v,l,&hv) && nvm_file_cyclic_input_region(q,f,i,v,l,&qv) && hv==qv);}
    for(unsigned edge=0;edge<2;edge++)if(x.edge_mask&(1u<<edge)){
     NvmFileCyclicVariant dest;CHECK(edge<hi.successor_count && nvm_file_cyclic_hosted_variant(p,f,hi.successors[edge],x.edge_variants[edge],&dest));
     CHECK(dest.input.stack==x.output.stack && dest.input.owners==x.output.owners);
    }else CHECK(x.edge_variants[edge]==NVM_FILE_CYCLIC_NO_VARIANT);
    CHECK(x.body.pending_checks & NVM_FILE_FLOW_CHECK_CLEANUP);
    if(x.body.has_obligation){NvmFileFlowObligation o=x.body.obligation;CHECK(o.site==hi.byte_offset && o.target==hi.decoded.operands[0].u32);
     if(o.kind==NVM_FILE_FLOW_CALL){calls++;borrowed+=o.borrowed_inputs!=0;CHECK(x.body.discharged_checks==NVM_FILE_FLOW_CHECK_CALLEE && x.body.pending_checks==(NVM_FILE_FLOW_CHECK_RESULT|NVM_FILE_FLOW_CHECK_CLEANUP));}
     else{uint32_t import;CHECK(nvm_file_cyclic_hosted_import(p,hi.catalog_ordinal,&import) && import==o.target);seen_services|=1u<<hi.catalog_ordinal;
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
 for(uint32_t i=0;i<8;i++){NvmFileNominalLayout x,y;CHECK(nvm_file_cyclic_hosted_type(p,i,&x) && nvm_file_cyclic_type(q,i,&y));CHECK(x.global_index==y.global_index && x.catalog_ordinal==y.catalog_ordinal && x.source_ordinal==y.source_ordinal && x.layout_kind==y.layout_kind && x.ownership_flags==y.ownership_flags && x.category==y.category);}
 for(uint32_t i=0;i<5;i++){uint32_t x,y;CHECK(nvm_file_cyclic_hosted_import(p,i,&x) && nvm_file_cyclic_import(q,i,&y) && x==y);}
 CHECK(seen_services!=31 || (calls==2 && borrowed==1));
 printf("I compared full hosted variants: service mask %u, calls %u, borrowed calls %u\n",seen_services,calls,borrowed);return seen_services;
}
static void ch_invalid_getters(NvmFileCyclicHostedPlan *p){
#define CH_BAD(type,expression) do{type out;memset(&out,0xa5,sizeof out);type before=out;CHECK(!(expression));CHECK(!memcmp(&out,&before,sizeof out));}while(0)
 CH_BAD(NvmFileCyclicHostedStartup,nvm_file_cyclic_hosted_startup(NULL,&out));
 CH_BAD(NvmFileCyclicSummary,nvm_file_cyclic_hosted_query_summary(NULL,&out));
 CH_BAD(NvmFileCyclicHostedFunction,nvm_file_cyclic_hosted_function(p,UINT32_MAX,&out));
 CH_BAD(uint32_t,nvm_file_cyclic_hosted_function_order(p,UINT32_MAX,&out));
 CH_BAD(NvmFileFlowDeclaration,nvm_file_cyclic_hosted_local(p,0,UINT16_MAX,&out));
 CH_BAD(NvmFileCodeInstruction,nvm_file_cyclic_hosted_instruction(p,0,UINT16_MAX,&out));
 CH_BAD(uint16_t,nvm_file_cyclic_hosted_component(p,UINT32_MAX,0,&out));
 CH_BAD(uint8_t,nvm_file_cyclic_hosted_variant_count(p,0,UINT16_MAX,&out));
 CH_BAD(NvmFileCyclicVariant,nvm_file_cyclic_hosted_variant(p,0,0,UINT8_MAX,&out));
 CH_BAD(NvmFileFlowValue,nvm_file_cyclic_hosted_input_local(p,0,0,0,UINT16_MAX,&out));
 CH_BAD(NvmFileFlowValue,nvm_file_cyclic_hosted_input_stack(p,0,0,0,0,&out));
 CH_BAD(NvmFileFlowReference,nvm_file_cyclic_hosted_input_reference(p,0,0,0,256,&out));
 CH_BAD(uint64_t,nvm_file_cyclic_hosted_input_region(p,0,0,0,0,&out));
 CH_BAD(NvmFileNominalLayout,nvm_file_cyclic_hosted_type(p,UINT32_MAX,&out));
 CH_BAD(uint32_t,nvm_file_cyclic_hosted_import(p,UINT32_MAX,&out));
 unsigned char bytes[4]={1,2,3,4};CHECK(!nvm_file_cyclic_hosted_bytes(p,SIZE_MAX,bytes,sizeof bytes) && bytes[0]==1 && bytes[3]==4);
 CHECK(!nvm_file_cyclic_hosted_bytes(p,0,NULL,1));CHECK(nvm_file_cyclic_hosted_bytes(p,0,NULL,0));
#undef CH_BAD
}
static void ch_relations_and_independence(void){
 for(unsigned perm=0;perm<2;perm++)for(unsigned shape=0;shape<2;shape++){
  NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,perm!=0);uint32_t header;
  setbody(m,0,shape?ch_owner_loop(b,&header):ch_scalar_loop(&header));
  NvmFileCyclicReport *q=NULL;OK(nvm_file_cyclic_analyze(m,&q));size_t size;uint8_t *bytes=serialize(m,&size),*saved=malloc(size);CHECK(saved);memcpy(saved,bytes,size);
  NvmFileCyclicHostedPlan *p=ch_expect(bytes,size,NVM_FILE_FLOW_OK);CHECK(ch_compare(p,q)==(shape?31u:2u));ch_invalid_getters(p);
  NvmFileCyclicHostedFunction fn;CHECK(nvm_file_cyclic_hosted_function(p,0,&fn));uint16_t at=UINT16_MAX;
  for(uint16_t i=0;i<fn.code.instruction_count;i++){NvmFileCodeInstruction in;CHECK(nvm_file_cyclic_hosted_instruction(p,0,i,&in));if(in.byte_offset==header)at=i;}
  CHECK(at!=UINT16_MAX);uint8_t count;CHECK(nvm_file_cyclic_hosted_variant_count(p,0,at,&count) && count==(shape?1:2));
  if(!shape){NvmFileFlowValue first,backedge;CHECK(nvm_file_cyclic_hosted_input_local(p,0,0,0,4,&first) && !first.initialized);CHECK(nvm_file_cyclic_hosted_input_local(p,0,0,1,4,&backedge) && backedge.initialized);}
  else{NvmFileFlowValue owner;NvmFileFlowReference ref;uint64_t region;CHECK(nvm_file_cyclic_hosted_input_local(p,0,at,0,0,&owner) && owner.owner==1);
   CHECK(nvm_file_cyclic_hosted_input_reference(p,0,at,0,20,&ref) && ref.live && ref.owner==owner.owner && ref.identity==789);
   CHECK(nvm_file_cyclic_hosted_input_region(p,0,at,0,0,&region) && region==ref.region && region==513);
   CHECK(fn.vm_value_slots==19 && fn.native_value_slots==20 && fn.frames==2 && fn.staging_slots==3);
  }
  expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);CHECK(!nvm_verify(m).ok);char error[128];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
  memset(bytes,0,size);free(bytes);memset(m->code,0,m->code_size);memset(m->ownership_data,0,m->ownership_size);nvm_module_free(m);
  ch_compare(p,q);uint8_t *copied=malloc(size);CHECK(copied);CHECK(nvm_file_cyclic_hosted_bytes(p,0,copied,size) && !memcmp(copied,saved,size));
  free(copied);free(saved);nvm_file_cyclic_free(q);nvm_file_cyclic_hosted_free(p);
 }
}
static void ch_storage_and_stack(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);uint32_t header;setbody(m,0,ch_owner_loop(b,&header));
 make_initializer(m,3);m->functions[3].name_idx=string(m,"caller");m->functions[3].result_count=1;m->functions[3].result_tag=TAG_INT;
 desc(m->ownership_data+ownership_function_offset(m,3)+4,TAG_INT,0,NVM_V2_NO_INDEX);
 Body c={0};for(unsigned i=0;i<5;i++)integer(&c);op(&c,OP_CALL);u32(&c,0);for(unsigned i=0;i<6;i++)op(&c,OP_POP);retint(&c);setbody(m,3,c);m->header.entry_point=3;
 size_t size;uint8_t *bytes=serialize(m,&size);NvmFileCyclicHostedPlan *p=ch_expect(bytes,size,NVM_FILE_FLOW_OK);
 NvmFileCyclicHostedStartup s;NvmFileCyclicHostedFunction f;CHECK(nvm_file_cyclic_hosted_startup(p,&s) && s.entry==3 && s.frames==3 && s.vm_value_slots==27 && s.native_value_slots==29 && s.reference_slots==768 && s.region_slots==768);
 CHECK(nvm_file_cyclic_hosted_function(p,3,&f) && f.operand_peak==6 && f.declared_stack==0 && f.staging_slots==1);
 uint32_t order[5];for(unsigned i=0;i<5;i++)CHECK(nvm_file_cyclic_hosted_function_order(p,i,&order[i]));unsigned main_rank=5,caller_rank=5;
 for(unsigned i=0;i<5;i++){if(order[i]==0)main_rank=i;if(order[i]==3)caller_rank=i;}CHECK(main_rank<caller_rank);
 nvm_file_cyclic_hosted_free(p);
 NvmV2SectionEntry functions=section(bytes,size,NVM_V2_SECTION_FUNCTIONS);uint8_t *depth=bytes+functions.offset+4+3*32+28;
 depth[0]=5;rehash(bytes,size);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_INVALID));
 depth[0]=6;rehash(bytes,size);p=ch_expect(bytes,size,NVM_FILE_FLOW_OK);nvm_file_cyclic_hosted_free(p);
 depth[0]=depth[1]=255;rehash(bytes,size);p=ch_expect(bytes,size,NVM_FILE_FLOW_OK);CHECK(nvm_file_cyclic_hosted_function(p,3,&f) && f.declared_stack==65535 && f.operand_peak==6 && f.vm_value_slots==27);nvm_file_cyclic_hosted_free(p);free(bytes);nvm_module_free(m);
 m=bodymodule(&b,false);make_initializer(m,3);c=(Body){0};uint32_t top=c.n;ch_boolean(&c);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 for(unsigned i=0;i<20;i++)integer(&c);
 for(unsigned i=0;i<20;i++)op(&c,OP_POP);
 ch_back(&c,top);target(&c,leave);op(&c,OP_RET);setbody(m,3,c);
 bytes=serialize(m,&size);p=ch_expect(bytes,size,NVM_FILE_FLOW_OK);CHECK(nvm_file_cyclic_hosted_startup(p,&s) && s.initializer==3 && s.entry==0 && s.vm_value_slots==23 && s.native_value_slots==23);
 nvm_file_cyclic_hosted_free(p);free(bytes);nvm_module_free(m);
}
static void ch_wire_and_refusals(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);uint32_t header;setbody(m,0,ch_scalar_loop(&header));size_t size;uint8_t *bytes=serialize(m,&size),*saved=malloc(size);CHECK(saved);memcpy(saved,bytes,size);
 CHECK(!ch_expect(NULL,0,NVM_FILE_FLOW_INVALID));CHECK(nvm_file_cyclic_hosted_prepare(bytes,size,NULL)==NVM_FILE_FLOW_INVALID);
 CHECK(!ch_expect(bytes,size-1,NVM_FILE_FLOW_INVALID));uint8_t dummy=0;CHECK(!ch_expect(&dummy,(size_t)NVM_FILE_HOSTED_INPUT_BYTES+1,NVM_FILE_FLOW_LIMIT));
 NvmV2SectionEntry functions=section(bytes,size,NVM_V2_SECTION_FUNCTIONS);wr32(bytes+functions.offset,65);rehash(bytes,size);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_LIMIT));memcpy(bytes,saved,size);
 NvmV2SectionEntry services=section(bytes,size,NVM_V2_SECTION_SERVICE_BINDINGS);bytes[services.offset]=1;rehash(bytes,size);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_INVALID));memcpy(bytes,saved,size);
 NvmV2Header h;CHECK(nvm_v2_read_header(bytes,size,&h)==NVM_V2_OK);h.feature_bits|=NVM_V2_FEATURE_CALLBACKS;nvm_v2_write_header(bytes,&h);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_UNRESOLVED));memcpy(bytes,saved,size);
 bytes[functions.offset+4+30]=1;rehash(bytes,size);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_UNRESOLVED));free(saved);free(bytes);
 Body c={0};retint(&c);one(&c,OP_LOAD_LOCAL,UINT16_MAX);op(&c,OP_RET);setbody(m,0,c);bytes=serialize(m,&size);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_INVALID));free(bytes);
 c=(Body){0};retint(&c);op(&c,OP_CALL_INDIRECT);u32(&c,0);op(&c,OP_RET);setbody(m,0,c);bytes=serialize(m,&size);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_UNRESOLVED));free(bytes);
 c=(Body){0};for(unsigned i=0;i<255;i++)op(&c,OP_NOP);retint(&c);setbody(m,0,c);bytes=serialize(m,&size);CHECK(!ch_expect(bytes,size,NVM_FILE_FLOW_LIMIT));free(bytes);nvm_module_free(m);
}
static void ch_alternative_limits(void){
 for(unsigned n=16;n<=17;n++){
  NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);
  for(unsigned l=4;l<=8;l++)desc(m->ownership_data+24+12+8*l,TAG_INT,0,NVM_V2_NO_INDEX);
  Body c={0};uint32_t ends[17];
  for(unsigned mask=0;mask<n;mask++){
   uint32_t next=UINT32_MAX;if(mask+1<n){ch_boolean(&c);next=branch(&c,OP_JMP_FALSE,0);}
   for(unsigned bit=0;bit<5;bit++)if(mask&(1u<<bit)){integer(&c);one(&c,OP_STORE_LOCAL,(uint16_t)(4+bit));}
   ends[mask]=branch(&c,OP_JMP,0);if(next!=UINT32_MAX)target(&c,next);
  }
  uint32_t join=c.n;op(&c,OP_NOP);retint(&c);for(unsigned j=0;j<n;j++)wr32(c.bytes+ends[j]+1,join-ends[j]);setbody(m,0,c);
  size_t size;uint8_t *bytes=serialize(m,&size);NvmFileCyclicHostedPlan *p=ch_expect(bytes,size,n==16?NVM_FILE_FLOW_OK:NVM_FILE_FLOW_LIMIT);
  if(p){NvmFileCyclicReport *q=NULL;OK(nvm_file_cyclic_analyze(m,&q));ch_compare(p,q);nvm_file_cyclic_free(q);
   NvmFileCyclicHostedFunction fn;CHECK(nvm_file_cyclic_hosted_function(p,0,&fn));bool found=false;
   for(uint16_t i=0;i<fn.code.instruction_count;i++){NvmFileCodeInstruction in;CHECK(nvm_file_cyclic_hosted_instruction(p,0,i,&in));if(in.byte_offset==join){
    uint8_t count;CHECK(nvm_file_cyclic_hosted_variant_count(p,0,i,&count) && count==16);uint32_t seen=0;
    for(uint8_t v=0;v<count;v++){unsigned mask=0;for(unsigned bit=0;bit<5;bit++){NvmFileFlowValue value;CHECK(nvm_file_cyclic_hosted_input_local(p,0,i,v,(uint16_t)(4+bit),&value));if(value.initialized)mask|=1u<<bit;}
     CHECK(mask<16 && !(seen&(1u<<mask)));seen|=1u<<mask;
    }CHECK(seen==65535);found=true;
   }}CHECK(found);nvm_file_cyclic_hosted_free(p);
  }free(bytes);nvm_module_free(m);
 }
}
static void ch_allocations(void){
#ifdef HOSTED_INSTRUMENT
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);uint32_t header;setbody(m,0,ch_owner_loop(b,&header));
 for(unsigned f=0;f<m->function_count;f++){Body c={0};c.n=m->functions[f].code_length;memcpy(c.bytes,m->code+m->functions[f].code_offset,c.n);for(unsigned i=0;i<145;i++)integer(&c);op(&c,OP_RET);setbody(m,f,c);}
 CHECK(m->code_size>4096);for(unsigned i=0;i<80;i++){char name[32];snprintf(name,sizeof name,"cyclic-name-%u",i);(void)string(m,name);}
 size_t size;uint8_t *bytes=serialize(m,&size);NvmFileCyclicReport *reference=NULL;OK(nvm_file_cyclic_analyze(m,&reference));
 size_t baseline=tracked_live,basebytes=tracked_bytes;
 NvmFileCyclicHostedPlan *good=ch_expect(bytes,size,NVM_FILE_FLOW_OK);NvmFileCyclicHostedStartup facts;CHECK(nvm_file_cyclic_hosted_startup(good,&facts));nvm_file_cyclic_hosted_free(good);
 unsigned failures[2]={0},recoveries[2]={0};
 for(unsigned transient=0;transient<2;transient++){
  bool completed=false;single_failure=transient!=0;
  for(int prefix=0;prefix<4096;prefix++){
   tracked_peak=tracked_bytes;failed_calls=0;allocation_budget=prefix;NvmFileCyclicHostedPlan *p=(NvmFileCyclicHostedPlan *)(uintptr_t)1;
   NvmFileFlowStatus got=nvm_file_cyclic_hosted_prepare(bytes,size,&p);allocation_budget=-1;
   CHECK(tracked_peak-basebytes<=facts.allocation_bound && (!transient || failed_calls<=1));
   if(got==NVM_FILE_FLOW_OK){CHECK(transient || !failed_calls);NvmFileCyclicHostedStartup actual;CHECK(nvm_file_cyclic_hosted_startup(p,&actual));
    CHECK(actual.allocation_bound==facts.allocation_bound && actual.retained_bound==facts.retained_bound && actual.query_storage_peak==facts.query_storage_peak);
    uint8_t *copy=malloc(size);CHECK(copy && nvm_file_cyclic_hosted_bytes(p,0,copy,size) && !memcmp(copy,bytes,size));free(copy);
    if(failed_calls)ch_compare(p,reference);
    nvm_file_cyclic_hosted_free(p);recoveries[transient]+=failed_calls!=0;
   }else{CHECK(failed_calls && p==(NvmFileCyclicHostedPlan *)(uintptr_t)1 && (got==NVM_FILE_FLOW_MEMORY || got==NVM_FILE_FLOW_UNRESOLVED));failures[transient]++;}
   CHECK(tracked_live==baseline && tracked_bytes==basebytes);
   if(!failed_calls){CHECK(got==NVM_FILE_FLOW_OK);completed=true;break;}
   p=ch_expect(bytes,size,NVM_FILE_FLOW_OK);nvm_file_cyclic_hosted_free(p);CHECK(tracked_live==baseline && tracked_bytes==basebytes);
  }
  CHECK(completed && failures[transient]);
 }
 single_failure=false;printf("I checked cyclic hosted allocation prefixes %u and transient refusals %u, complete recoveries %u/%u\n",failures[0],failures[1],recoveries[0],recoveries[1]);
 nvm_file_cyclic_free(reference);free(bytes);nvm_module_free(m);CHECK(!tracked_live && !tracked_bytes);
#endif
}
int main(void){
 CHECK(prior_file_hosted_fixture_main()==0);
 ch_relations_and_independence();ch_storage_and_stack();ch_wire_and_refusals();ch_alternative_limits();ch_allocations();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
 printf("PASS %u private cyclic hosted checks; no runtime or service execution\n",checks);return 0;
}
