/* I query serialized bytes only; no hosted File operation is dispatched. */
#include <stdlib.h>
#include "file_hosted_alloc.h"
#ifdef HOSTED_INSTRUMENT
#define malloc file_test_malloc
#define calloc file_test_calloc
#define realloc file_test_realloc
#define free file_test_free
#endif
#define FILE_BODY_MAIN prior_file_body_fixture_main
#include "test_file_body.c"
#undef FILE_BODY_MAIN
#undef malloc
#undef calloc
#undef realloc
#undef free
#include "../../src/nanoisa/file_hosted.h"
#ifdef HOSTED_INSTRUMENT
#include "file_hosted_alloc.h"
typedef struct {void *pointer;size_t size;} Allocation;
static Allocation tracked[8192];
static size_t tracked_bytes,tracked_peak,tracked_live,failed_calls;
static int allocation_budget=-1;
static bool single_failure;
static bool deny_allocation(void){if(!allocation_budget){failed_calls++;if(single_failure)allocation_budget=-1;return true;}if(allocation_budget>0)allocation_budget--;return false;}
static void record_allocation(void *p,size_t n){if(!p)return;for(unsigned i=0;i<8192;i++)if(!tracked[i].pointer){tracked[i]=(Allocation){p,n};tracked_live++;tracked_bytes+=n;if(tracked_bytes>tracked_peak)tracked_peak=tracked_bytes;return;}CHECK(false);}
static void forget_allocation(void *p){for(unsigned i=0;i<8192;i++)if(tracked[i].pointer==p && p){CHECK(tracked_live && tracked_bytes>=tracked[i].size);tracked_live--;tracked_bytes-=tracked[i].size;tracked[i]=(Allocation){0};return;}}
void *file_test_malloc(size_t n){if(deny_allocation())return NULL;void *p=malloc(n);record_allocation(p,n);return p;}
void *file_test_calloc(size_t n,size_t width){if(n && width>SIZE_MAX/n)return NULL;if(deny_allocation())return NULL;void *p=calloc(n,width);record_allocation(p,n*width);return p;}
void file_test_free(void *p){forget_allocation(p);free(p);}
void *file_test_realloc(void *p,size_t n){
 if(deny_allocation())return NULL;
 /* Allocate/copy/free models the permitted worst old+new realloc peak, without
  * observing a freed pointer or changing failure-preserves-old semantics. */
 if(!p){void *next=malloc(n);record_allocation(next,n);return next;}
 size_t old=0;bool found=false;for(unsigned i=0;i<8192;i++)if(tracked[i].pointer==p){old=tracked[i].size;found=true;break;}
 CHECK(found);void *next=malloc(n);if(!next)return NULL;record_allocation(next,n);memcpy(next,p,old<n?old:n);file_test_free(p);return next;
}
#endif
static uint8_t *serialize(NvmModule *m,size_t *size){
 NvmV2Module wire={0};CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
 CHECK(nvm_v2_module_serialize(&wire,NULL,0,size)==NVM_V2_OK);uint8_t *bytes=malloc(*size);CHECK(bytes);
 CHECK(nvm_v2_module_serialize(&wire,bytes,*size,size)==NVM_V2_OK);nvm_v2_module_free(&wire);return bytes;
}
static NvmV2SectionEntry section(const uint8_t *bytes,size_t size,uint32_t type){NvmV2Header h;CHECK(nvm_v2_read_header(bytes,size,&h)==NVM_V2_OK);for(uint32_t i=0;i<h.section_count;i++){NvmV2SectionEntry e;CHECK(nvm_v2_read_section(bytes,size,&h,i,&e)==NVM_V2_OK);if(e.type==type)return e;}CHECK(false);return (NvmV2SectionEntry){0};}
static void rehash(uint8_t *bytes,size_t size){NvmV2Header h;CHECK(nvm_v2_read_header(bytes,size,&h)==NVM_V2_OK);h.checksum=nvm_crc32(bytes+h.header_size,(uint32_t)(size-h.header_size));nvm_v2_write_header(bytes,&h);}
static void expect_hosted(const uint8_t *bytes,size_t size,NvmFileFlowStatus expected){
 NvmFileHostedPlan *p=(NvmFileHostedPlan *)(uintptr_t)1;
#ifdef HOSTED_INSTRUMENT
 size_t baseline=tracked_live,basebytes=tracked_bytes;
#endif
 CHECK(nvm_file_hosted_prepare(bytes,size,&p)==expected);
 if(expected==NVM_FILE_FLOW_OK){CHECK(p && p!=(NvmFileHostedPlan *)(uintptr_t)1);nvm_file_hosted_free(p);}else CHECK(p==(NvmFileHostedPlan *)(uintptr_t)1);
#ifdef HOSTED_INSTRUMENT
 CHECK(tracked_live==baseline && tracked_bytes==basebytes);
#endif
}
static void make_initializer(NvmModule *m,unsigned f){
 size_t at=24;for(unsigned i=0;i<f;i++)at+=12+8*m->functions[i].local_count;
 NvmFunctionEntry *fn=&m->functions[f];fn->name_idx=string(m,"__init__");fn->arity=0;fn->result_count=0;fn->result_tag=TAG_VOID;
 CHECK(nvm_set_function_param_types(m,f,NULL,0));m->ownership_data[at+2]=m->ownership_data[at+3]=0;
 desc(m->ownership_data+at+4,TAG_VOID,0,NVM_V2_NO_INDEX);
 for(uint16_t i=0;i<fn->local_count;i++)desc(m->ownership_data+at+12+8*i,TAG_INT,0,NVM_V2_NO_INDEX);
 Body c={0};op(&c,OP_RET);setbody(m,f,c);
}
static void exact_startup_and_bounds(void){
 for(unsigned permutation=0;permutation<2;permutation++){
  NvmFileNominalBindings bindings;NvmModule *m=bodymodule(&bindings,permutation!=0);size_t size;uint8_t *bytes=serialize(m,&size);
  NvmFileHostedPlan *p=NULL;OK(nvm_file_hosted_prepare(bytes,size,&p));NvmFileHostedStartup startup;
  CHECK(nvm_file_hosted_startup(p,&startup));CHECK(startup.entry==0 && startup.initializer==NVM_V2_NO_INDEX && startup.functions==5);
  CHECK(startup.vm_value_slots==14 && startup.native_value_slots==14 && startup.frames==1 && startup.reference_slots==256 && startup.region_slots==256);
  CHECK(startup.allocation_bound<=NVM_FILE_HOSTED_BYTES && startup.allocation_bound>=NVM_FILE_FLOW_BYTES);
  NvmFileHostedFunction f;CHECK(nvm_file_hosted_function(p,0,&f) && f.declared_stack==0 && f.operand_peak==1 && f.locals==12 && f.staging_slots==1);
  CHECK(nvm_file_hosted_function(p,2,&f) && f.code.declaration.parameters==2 && f.vm_value_slots==4);
  NvmFileFlowDeclaration type;CHECK(nvm_file_hosted_local(p,2,0,&type) && type.mode==2 && type.global_index==bindings.layouts[0]);
  NvmFileCodeInstruction in={0},oldin=in;NvmFileBodyInstruction fact={0},oldfact=fact;
  CHECK(!nvm_file_hosted_instruction(p,5,0,&in,&fact) && !memcmp(&in,&oldin,sizeof in) && !memcmp(&fact,&oldfact,sizeof fact));
  NvmFileHostedFunction old=f;CHECK(!nvm_file_hosted_function(p,5,&f) && !memcmp(&old,&f,sizeof f));
  CHECK(!nvm_verify(m).ok);char error[128];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
  memset(bytes,0,size);free(bytes);nvm_module_free(m);CHECK(nvm_file_hosted_startup(p,&startup) && startup.entry==0);nvm_file_hosted_free(p);
  m=bodymodule(&bindings,permutation!=0);setbody(m,0,lifecycle_code(bindings));bytes=serialize(m,&size);p=NULL;OK(nvm_file_hosted_prepare(bytes,size,&p));CHECK(nvm_file_hosted_startup(p,&startup));
  CHECK(startup.vm_value_slots==19 && startup.native_value_slots==20 && startup.frames==2 && startup.reference_slots==512);
  CHECK(nvm_file_hosted_function(p,0,&f) && f.staging_slots==3 && f.operand_peak==1);
  NvmFileCodeFunction code=f.code;unsigned calls=0,services=0;
  for(uint16_t i=0;i<code.instruction_count;i++){CHECK(nvm_file_hosted_instruction(p,0,i,&in,&fact));if(fact.has_obligation){CHECK(fact.pending_checks & NVM_FILE_FLOW_CHECK_CLEANUP);if(fact.obligation.kind==NVM_FILE_FLOW_CALL){calls++;CHECK(fact.discharged_checks==NVM_FILE_FLOW_CHECK_CALLEE);}else{services++;CHECK(fact.discharged_checks==0 && (fact.pending_checks & fact.obligation.checks)==fact.obligation.checks);}}}
  CHECK(calls==2 && services==5);nvm_file_hosted_free(p);free(bytes);nvm_module_free(m);
 }
}
static void initializer_controls(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);make_initializer(m,3);m->functions[4].name_idx=string(m,"__init__");size_t size;uint8_t *bytes=serialize(m,&size);
 NvmFileHostedPlan *p=NULL;OK(nvm_file_hosted_prepare(bytes,size,&p));NvmFileHostedStartup s;CHECK(nvm_file_hosted_startup(p,&s) && s.initializer==3 && s.entry==0);nvm_file_hosted_free(p);free(bytes);
 m->functions[1].name_idx=string(m,"__init__");bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);nvm_module_free(m);
 m=bodymodule(&b,false);m->functions[0].name_idx=string(m,"__init__");bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);nvm_module_free(m);
 m=bodymodule(&b,false);const char name[]="__init__\0hidden";m->functions[3].name_idx=nvm_add_string(m,name,sizeof name-1);bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);free(bytes);nvm_module_free(m);
 m=bodymodule(&b,false);m->header.flags&=~NVM_FLAG_HAS_MAIN;bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);free(bytes);nvm_module_free(m);
}
static size_t ownership_function_offset(const NvmModule *m,unsigned f){
 size_t at=24;for(unsigned i=0;i<f;i++)at+=12+8*m->functions[i].local_count;return at;
}
static void startup_supplement(void){
 NvmFileNominalBindings b;size_t size;uint8_t *bytes;Body c={0};
 NvmModule *m=bodymodule(&b,false);
 m->functions[0].result_tag=TAG_BOOL;desc(m->ownership_data+28,TAG_BOOL,0,NVM_V2_NO_INDEX);
 op(&c,OP_PUSH_BOOL);op(&c,1);op(&c,OP_RET);setbody(m,0,c);
 bytes=serialize(m,&size);NvmFileHostedPlan *p=NULL;OK(nvm_file_hosted_prepare(bytes,size,&p));
 NvmFileHostedFunction entry;CHECK(nvm_file_hosted_function(p,0,&entry));
 CHECK(entry.code.declaration.result.tag==TAG_BOOL && entry.operand_peak==1);
 nvm_file_hosted_free(p);free(bytes);nvm_module_free(m);
 /* A valid borrowed helper body is not a valid hosted entry signature. */
 m=bodymodule(&b,false);m->header.entry_point=3;bytes=serialize(m,&size);
 expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);nvm_module_free(m);
 /* Independently reject a zero-argument VOID entry. */
 m=bodymodule(&b,false);make_initializer(m,3);m->functions[3].name_idx=string(m,"void_entry");m->header.entry_point=3;
 bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);nvm_module_free(m);
 /* Selected initializers must not publish even an ignored scalar result. */
 m=bodymodule(&b,false);make_initializer(m,3);size_t at=ownership_function_offset(m,3);
 m->functions[3].result_count=1;m->functions[3].result_tag=TAG_INT;
 desc(m->ownership_data+at+4,TAG_INT,0,NVM_V2_NO_INDEX);c=(Body){0};retint(&c);setbody(m,3,c);
 bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);nvm_module_free(m);
 /* This owner-returning initializer has no parameters: the result itself is
  * the refusal, not the borrowed/owning parameter rule. Its body is logical. */
 m=bodymodule(&b,false);make_initializer(m,3);at=ownership_function_offset(m,3);
 m->functions[3].result_count=1;m->functions[3].result_tag=TAG_UNION;
 desc(m->ownership_data+at+4,TAG_UNION,0,b.layouts[3]);
 c=(Body){0};service(&c,b,0,UINT16_MAX);op(&c,OP_RET);setbody(m,3,c);
 bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);
 m->functions[3].name_idx=string(m,"owner_entry");m->header.entry_point=3;
 bytes=serialize(m,&size);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);free(bytes);nvm_module_free(m);
 /* Entry and initializer execute sequentially. The initializer's larger
  * frame must dominate, without summing the two invocation requirements. */
 m=bodymodule(&b,false);make_initializer(m,3);c=(Body){0};
 for(unsigned i=0;i<20;i++)integer(&c);
 for(unsigned i=0;i<20;i++)op(&c,OP_POP);
 op(&c,OP_RET);setbody(m,3,c);bytes=serialize(m,&size);p=NULL;
 OK(nvm_file_hosted_prepare(bytes,size,&p));NvmFileHostedStartup startup;NvmFileHostedFunction init;
 CHECK(nvm_file_hosted_startup(p,&startup) && startup.entry==0 && startup.initializer==3);
 CHECK(nvm_file_hosted_function(p,0,&entry) && nvm_file_hosted_function(p,3,&init));
 CHECK(entry.vm_value_slots==14 && entry.native_value_slots==14);
 CHECK(init.operand_peak==20 && init.locals==2 && init.staging_slots==1);
 CHECK(init.vm_value_slots==23 && init.native_value_slots==23);
 CHECK(startup.vm_value_slots==23 && startup.native_value_slots==23 && startup.frames==1);
 CHECK(startup.reference_slots==256 && startup.region_slots==256);
 nvm_file_hosted_free(p);free(bytes);nvm_module_free(m);
}
static void wire_controls(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);Body c={0};integer(&c);integer(&c);op(&c,OP_ADD);op(&c,OP_RET);setbody(m,0,c);size_t size;uint8_t *bytes=serialize(m,&size);
 NvmV2SectionEntry f=section(bytes,size,NVM_V2_SECTION_FUNCTIONS);uint8_t *depth=bytes+f.offset+4+28;
 depth[0]=1;rehash(bytes,size);expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);
 depth[0]=2;rehash(bytes,size);expect_hosted(bytes,size,NVM_FILE_FLOW_OK);depth[0]=255;depth[1]=255;rehash(bytes,size);NvmFileHostedPlan *p=NULL;OK(nvm_file_hosted_prepare(bytes,size,&p));NvmFileHostedFunction facts;CHECK(nvm_file_hosted_function(p,0,&facts) && facts.declared_stack==65535 && facts.operand_peak==2 && facts.vm_value_slots==15);nvm_file_hosted_free(p);
 depth[0]=depth[1]=0;rehash(bytes,size);expect_hosted(bytes,size,NVM_FILE_FLOW_OK);
 uint8_t *original=malloc(size);CHECK(original);memcpy(original,bytes,size);
 NvmV2Header h;CHECK(nvm_v2_read_header(bytes,size,&h)==NVM_V2_OK);uint32_t features=h.feature_bits;
 const uint32_t required[]={NVM_V2_FEATURE_FFI,NVM_V2_FEATURE_RETAINED_LAYOUTS,NVM_V2_FEATURE_OWNERSHIP,NVM_V2_FEATURE_SERVICE_BINDINGS};
 for(unsigned i=0;i<4;i++){h.feature_bits=features & ~required[i];nvm_v2_write_header(bytes,&h);expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);}
 h.feature_bits=features|UINT32_C(0x80000000);nvm_v2_write_header(bytes,&h);expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);memcpy(bytes,original,size);
 h.feature_bits=features|NVM_V2_FEATURE_CAPTURE_BINDINGS;nvm_v2_write_header(bytes,&h);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);memcpy(bytes,original,size);
 h.feature_bits=features|NVM_V2_FEATURE_CALLBACKS;nvm_v2_write_header(bytes,&h);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);memcpy(bytes,original,size);
 NvmV2SectionEntry service_section=section(bytes,size,NVM_V2_SECTION_SERVICE_BINDINGS);bytes[service_section.offset]=1;rehash(bytes,size);expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);memcpy(bytes,original,size);
 bytes[3]=1;expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);memcpy(bytes,original,size);
 expect_hosted(bytes,size-1,NVM_FILE_FLOW_INVALID);expect_hosted(NULL,0,NVM_FILE_FLOW_INVALID);
 uint8_t dummy=0;expect_hosted(&dummy,(size_t)NVM_FILE_HOSTED_INPUT_BYTES+1,NVM_FILE_FLOW_LIMIT);
 NvmV2SectionEntry constants=section(bytes,size,NVM_V2_SECTION_CONSTANTS);wr32(bytes+constants.offset,UINT32_MAX);rehash(bytes,size);expect_hosted(bytes,size,NVM_FILE_FLOW_INVALID);memcpy(bytes,original,size);
 NvmV2SectionEntry functions=section(bytes,size,NVM_V2_SECTION_FUNCTIONS);wr32(bytes+functions.offset,65);rehash(bytes,size);expect_hosted(bytes,size,NVM_FILE_FLOW_LIMIT);memcpy(bytes,original,size);
 /* Allocating reader's invalid reserved function flags stay conservatively
  * unresolved, the same classification used for its legacy OOM ambiguities. */
 bytes[functions.offset+4+30]=1;rehash(bytes,size);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);
 free(original);free(bytes);nvm_module_free(m);
}
static void allocation_prefixes(void){
#ifdef HOSTED_INSTRUMENT
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);setbody(m,0,lifecycle_code(b));
 /* Unreachable, structurally valid instructions force CODE beyond the bridge's
  * initial 4096 bytes. They are decoded but never interpreted or executed. */
 for(unsigned f=0;f<m->function_count;f++){
  Body padded={0};padded.n=m->functions[f].code_length;memcpy(padded.bytes,m->code+m->functions[f].code_offset,padded.n);
  for(unsigned i=0;i<150;i++)integer(&padded);
  op(&padded,OP_RET);setbody(m,f,padded);
 }
 CHECK(m->code_size>4096);
 for(unsigned i=0;i<80;i++){char name[32];snprintf(name,sizeof name,"unused-name-%u",i);(void)string(m,name);}
 for(unsigned i=0;i<20;i++){char key[32];snprintf(key,sizeof key,"metadata-%u",i);CHECK(nvm_add_metadata(m,string(m,key),string(m,"value")));}
 for(unsigned i=0;i<300;i++)CHECK(nvm_add_debug_entry(m,0,i+1,1));
 size_t size;uint8_t *bytes=serialize(m,&size);size_t baseline=tracked_live,basebytes=tracked_bytes;bool success=false;unsigned memory=0,unresolved=0;
 NvmFileHostedPlan *good=NULL;OK(nvm_file_hosted_prepare(bytes,size,&good));NvmFileHostedStartup facts;CHECK(nvm_file_hosted_startup(good,&facts));nvm_file_hosted_free(good);
 for(int prefix=0;prefix<4096;prefix++){
  tracked_peak=tracked_bytes;failed_calls=0;allocation_budget=prefix;NvmFileHostedPlan *p=(NvmFileHostedPlan *)(uintptr_t)1;NvmFileFlowStatus status=nvm_file_hosted_prepare(bytes,size,&p);allocation_budget=-1;
  CHECK(tracked_peak-basebytes<=facts.allocation_bound);
  if(status==NVM_FILE_FLOW_OK){CHECK(!failed_calls);nvm_file_hosted_free(p);success=true;CHECK(tracked_live==baseline && tracked_bytes==basebytes);break;}
  CHECK(failed_calls && p==(NvmFileHostedPlan *)(uintptr_t)1);
  CHECK(status==NVM_FILE_FLOW_MEMORY || status==NVM_FILE_FLOW_UNRESOLVED);memory+=status==NVM_FILE_FLOW_MEMORY;unresolved+=status==NVM_FILE_FLOW_UNRESOLVED;
  CHECK(tracked_live==baseline && tracked_bytes==basebytes);expect_hosted(bytes,size,NVM_FILE_FLOW_OK);
 }
 CHECK(success && memory && unresolved);
 unsigned transient_refusals=0,transient_successes=0;success=false;single_failure=true;
 for(int prefix=0;prefix<4096;prefix++){
  tracked_peak=tracked_bytes;failed_calls=0;allocation_budget=prefix;
  NvmFileHostedPlan *p=(NvmFileHostedPlan *)(uintptr_t)1;
  NvmFileFlowStatus status=nvm_file_hosted_prepare(bytes,size,&p);allocation_budget=-1;
  CHECK(tracked_peak-basebytes<=facts.allocation_bound);
  if(status==NVM_FILE_FLOW_OK){
   NvmFileHostedStartup recovered;CHECK(nvm_file_hosted_startup(p,&recovered));
   CHECK(!memcmp(&recovered,&facts,sizeof facts));
   for(unsigned f=0;f<m->function_count;f++){NvmFileHostedFunction fn;CHECK(nvm_file_hosted_function(p,f,&fn));CHECK(fn.code.code_length==m->functions[f].code_length);}
   nvm_file_hosted_free(p);transient_successes+=failed_calls!=0;
  }else{
   CHECK(failed_calls==1 && p==(NvmFileHostedPlan *)(uintptr_t)1);
   CHECK(status==NVM_FILE_FLOW_MEMORY || status==NVM_FILE_FLOW_UNRESOLVED);transient_refusals++;
  }
  CHECK(tracked_live==baseline && tracked_bytes==basebytes);
  if(!failed_calls){success=true;break;}
  expect_hosted(bytes,size,NVM_FILE_FLOW_OK);
 }
 single_failure=false;CHECK(success && transient_refusals);
 printf("I retain %u transient refusals and %u complete recovered plans after one allocation failure\n",transient_refusals,transient_successes);
 free(bytes);nvm_module_free(m);CHECK(!tracked_live && !tracked_bytes);
 printf("I retain %u precise MEMORY and %u ambiguous UNRESOLVED allocation prefixes\n",memory,unresolved);
#endif
}
#ifndef FILE_HOSTED_MAIN
#define FILE_HOSTED_MAIN main
#endif
int FILE_HOSTED_MAIN(void){exact_startup_and_bounds();initializer_controls();startup_supplement();wire_controls();allocation_prefixes();
#ifdef HOSTED_INSTRUMENT
 CHECK(!tracked_live && !tracked_bytes);
#endif
 printf("PASS %u private serialized File hosted-plan checks; no handler or public admission\n",checks);return 0;}
