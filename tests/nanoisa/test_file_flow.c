#include "../../src/nanoisa/file_flow.h"
#include "../../src/nanoisa/service_bindings_module.h"
#include "../../src/nanoisa/ownership_contracts.h"
#include "../../src/nanoisa/retained_layouts.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanoisa/nvm2c.h"
#include "../../src/nsi_file_catalog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#define OK(x) CHECK((x)==NVM_FILE_FLOW_OK)
#ifdef FLOW_INSTRUMENT
static int budget=-1;static unsigned live,allocations;
static void *flow_malloc(size_t n){allocations++;if(!budget)return NULL;if(budget>0)budget--;void *p=malloc(n);if(p)live++;return p;}
static void *flow_calloc(size_t n,size_t z){if(n && z>SIZE_MAX/n)return NULL;void *p=flow_malloc(n*z);if(p)memset(p,0,n*z);return p;}
static void flow_free(void *p){if(p){CHECK(live);live--;}free(p);}
#define malloc flow_malloc
#define calloc flow_calloc
#define free flow_free
#define member_type nominal_member_type
#include "../../src/nanoisa/service_file_nominal_plan.c"
#undef member_type
#include "../../src/nanoisa/file_flow.c"
#undef malloc
#undef calloc
#undef free
#endif
static void wr32(uint8_t *p,uint32_t x){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(x>>(8*i));}
static uint32_t string(NvmModule *m,const char *s){uint32_t n=nvm_add_string(m,s,(uint32_t)strlen(s));CHECK(n!=UINT32_MAX);return n;}
static void desc(uint8_t *p,uint8_t tag,uint8_t mode,uint32_t layout){p[0]=tag;p[1]=mode;p[2]=p[3]=0;wr32(p+4,layout);}
static NvmModule *fixture(bool permuted,NvmFileNominalBindings *b){
 AsmResult result={0};NvmModule *m=asm_assemble(".function main 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n.entry main\n",&result);CHECK(m);
 for(unsigned i=0;i<8;i++)b->layouts[i]=i+1;
 if(permuted){const uint32_t x[]={2,1,3,8,7,6,5,4};memcpy(b->layouts,x,sizeof x);}
 uint32_t mod=string(m,nl_file_catalog_interface());
 for(unsigned i=0;i<5;i++){
  unsigned ordinal=permuted?4-i:i;const NlFilePlanMethod *method=nl_file_catalog_method(ordinal);
  uint8_t tags[]={TAG_STRUCT,TAG_INT};uint16_t count=ordinal==0?0:ordinal==1?2:1;
  uint32_t index=nvm_add_import(m,mod,string(m,method->id),count,TAG_UNION,tags);CHECK(index==i);
  m->imports[index].kind=NVM_IMPORT_SERVICE;b->imports[ordinal]=index;
 }
 NvmV2Layout layouts[9]={0};NvmV2LayoutField fields[8][7]={0};layouts[0].kind=NVM_V2_LAYOUT_TUPLE;layouts[0].name_idx=NVM_V2_NO_INDEX;
 for(unsigned i=0;i<8;i++){
  const NlFilePlanType *t=nl_file_catalog_type(i);CHECK(t);NvmV2Layout *l=&layouts[b->layouts[i]];
  l->kind=i<3?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION;l->name_idx=string(m,t->id);l->field_count=(uint16_t)t->member_count;l->fields=fields[i];
  for(size_t j=0;j<t->member_count;j++){
   const char *id=t->members[j].type_id;uint8_t tag=TAG_VOID;uint32_t nested=NVM_V2_NO_INDEX;
   if(id){if(!strcmp(id,"nsi:core/int"))tag=TAG_INT;else if(!strcmp(id,"nsi:core/bool"))tag=TAG_BOOL;else for(unsigned k=0;k<8;k++)if(!strcmp(id,nl_file_catalog_type(k)->id)){tag=k<3?TAG_STRUCT:TAG_UNION;nested=b->layouts[k];}}
   fields[i][j]=(NvmV2LayoutField){tag,nested,string(m,t->members[j].id)};
  }
 }
 m->struct_count=3;m->union_count=5;NvmV2Layouts table={layouts,9};CHECK(nvm_retain_layouts(m,&table)==NVM_V2_OK);
 m->functions[0].arity=2;m->functions[0].local_count=4;uint8_t params[]={TAG_INT,TAG_STRUCT};CHECK(nvm_set_function_param_types(m,0,params,2));
 m->ownership_size=68;m->ownership_data=calloc(1,68);CHECK(m->ownership_data);
 uint8_t *o=m->ownership_data;wr32(o,1);wr32(o+4,9);
 for(unsigned i=0;i<8;i++)o[8+b->layouts[i]]=(i==0||i==3)?3:1;
 wr32(o+20,1);o[24]=4;o[26]=2;desc(o+28,TAG_INT,0,NVM_V2_NO_INDEX);desc(o+36,TAG_INT,0,NVM_V2_NO_INDEX);
 desc(o+44,TAG_STRUCT,2,b->layouts[0]);desc(o+52,TAG_UNION,0,b->layouts[3]);desc(o+60,TAG_FLOAT,0,NVM_V2_NO_INDEX);
 m->service_data=malloc(120);CHECK(m->service_data);size_t n=0;CHECK(nvm_file_nominal_encode(b,m->service_data,120,&n)==NVM_SERVICE_OK);m->service_size=(uint32_t)n;return m;
}
/* The bytecode body remains an ordinary placeholder. I query declarations and
 * explicit logical transitions; I never execute this synthetic module. */
static NvmModule *module(NvmFileNominalBindings *b,bool permute){
 NvmModule *m=fixture(permute,b);
 for(unsigned i=1;i<5;i++){NvmFunctionEntry f=m->functions[0];f.arity=0;f.local_count=0;CHECK(nvm_add_function(m,&f)==i);}
 const uint16_t locals[]={12,1,2,2,1},arity[]={0,1,2,2,1};
 const int types[5][12]={{0,3,1,7,-1,2,4,5,6,0,3,-2},{0},{0,-1},{0,0},{3}};
 const int results[]={-1,0,4,-1,3};
 size_t size=24;for(unsigned i=0;i<5;i++)size+=12+8*locals[i];
 free(m->ownership_data);m->ownership_data=calloc(1,size);CHECK(m->ownership_data);m->ownership_size=(uint32_t)size;
 uint8_t *o=m->ownership_data;wr32(o,1);wr32(o+4,9);for(unsigned i=0;i<8;i++)o[8+b->layouts[i]]=(i==0||i==3)?3:1;wr32(o+20,5);size_t p=24;
 for(unsigned f=0;f<5;f++){
  m->functions[f].arity=arity[f];m->functions[f].local_count=locals[f];m->functions[f].result_count=1;
  int r=results[f];uint8_t rt=r<0?TAG_INT:r<3?TAG_STRUCT:TAG_UNION;m->functions[f].result_tag=rt;
  o[p]=(uint8_t)locals[f];o[p+1]=(uint8_t)(locals[f]>>8);o[p+2]=(uint8_t)arity[f];desc(o+p+4,rt,0,r<0?NVM_V2_NO_INDEX:b->layouts[r]);p+=12;
  uint8_t tags[12];for(unsigned l=0;l<locals[f];l++){
   int t=types[f][l];uint8_t tag=t==-1?TAG_INT:t==-2?TAG_BOOL:t<3?TAG_STRUCT:TAG_UNION;
   uint8_t mode=(f==2 && l==0)||f==3?2:0;tags[l]=tag;desc(o+p,tag,mode,t<0?NVM_V2_NO_INDEX:b->layouts[t]);p+=8;
  }
  CHECK(nvm_set_function_param_types(m,f,tags,arity[f]));
 }
 CHECK(p==size);return m;
}
static NvmFileFlowState *state(NvmFileFlowDeclarations *d,unsigned f){NvmFileFlowState *s=NULL;OK(nvm_file_flow_state(d,f,&s));CHECK(s);return s;}
static NvmFileFlowValue local(NvmFileFlowState *s,unsigned l){NvmFileFlowValue v;CHECK(nvm_file_flow_local(s,l,&v));return v;}
static NvmFileFlowCounts counts(NvmFileFlowState *s){NvmFileFlowCounts c;CHECK(nvm_file_flow_counts(s,&c));return c;}
static void finish(NvmFileFlowState *s){OK(nvm_file_flow_push_scalar(s,TAG_INT));OK(nvm_file_flow_can_exit(s));}
static void error_value(NvmFileFlowState *s){for(unsigned i=0;i<7;i++)OK(nvm_file_flow_push_scalar(s,i<4?TAG_INT:TAG_BOOL));OK(nvm_file_flow_construct(s,1,0));}
static NvmFileFlowState *opened(NvmFileFlowDeclarations *d,NvmFileNominalBindings b){
 NvmFileFlowState *s=state(d,0),*a=NULL,*e=NULL;
 OK(nvm_file_flow_service(s,1,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE));CHECK(counts(s).owners==1);
 CHECK(nvm_file_flow_dup(s)==NVM_FILE_FLOW_INVALID);OK(nvm_file_flow_put(s,1));
 CHECK(nvm_file_flow_take_result(s,1,NVM_FILE_FLOW_ARM_OK)==NVM_FILE_FLOW_INVALID);
 OK(nvm_file_flow_refine(s,1,&a,&e));OK(nvm_file_flow_take_result(e,1,NVM_FILE_FLOW_ARM_ERROR));CHECK(!counts(e).owners);OK(nvm_file_flow_pop(e));finish(e);nvm_file_flow_state_free(e);
 OK(nvm_file_flow_take_result(a,1,NVM_FILE_FLOW_ARM_OK));OK(nvm_file_flow_put(a,0));nvm_file_flow_state_free(s);return a;
}
static void lifecycle(NvmFileFlowDeclarations *d,NvmFileNominalBindings b){
 NvmFileFlowState *s=opened(d,b);uint64_t id=local(s,0).owner;
 CHECK(nvm_file_flow_load(s,0)==NVM_FILE_FLOW_INVALID);CHECK(nvm_file_flow_can_exit(s)==NVM_FILE_FLOW_INVALID);
 OK(nvm_file_flow_region_begin(s));OK(nvm_file_flow_borrow(s,0,20));CHECK(nvm_file_flow_borrow(s,0,21)==NVM_FILE_FLOW_INVALID);CHECK(nvm_file_flow_take(s,0)==NVM_FILE_FLOW_INVALID);
 OK(nvm_file_flow_push_scalar(s,TAG_BOOL));CHECK(nvm_file_flow_service(s,2,b.imports[1],20)==NVM_FILE_FLOW_INVALID);OK(nvm_file_flow_pop(s));
 OK(nvm_file_flow_push_scalar(s,TAG_INT));OK(nvm_file_flow_service(s,2,b.imports[1],20));OK(nvm_file_flow_pop(s));
 OK(nvm_file_flow_service(s,3,b.imports[2],20));OK(nvm_file_flow_pop(s));OK(nvm_file_flow_service(s,4,b.imports[3],20));OK(nvm_file_flow_pop(s));CHECK(local(s,0).owner==id);
 NvmFileFlowObligation event;CHECK(nvm_file_flow_obligation(s,1,&event));CHECK(event.checks & NVM_FILE_FLOW_CHECK_BYTE);CHECK(event.outcomes[0]==NVM_FILE_FLOW_INPUT_PRESERVED && event.outcomes[1]==NVM_FILE_FLOW_INPUT_PRESERVED);
 uint16_t refs[]={20,NVM_FILE_FLOW_NO_REFERENCE};OK(nvm_file_flow_push_scalar(s,TAG_INT));OK(nvm_file_flow_call(s,5,2,refs,2));OK(nvm_file_flow_pop(s));CHECK(local(s,0).owner==id);
 uint16_t duplicate[]={20,20};CHECK(nvm_file_flow_call(s,6,3,duplicate,2)==NVM_FILE_FLOW_INVALID);
 OK(nvm_file_flow_region_end(s));CHECK(nvm_file_flow_service(s,7,b.imports[3],20)==NVM_FILE_FLOW_INVALID);
 OK(nvm_file_flow_move(s,0,9));CHECK(!local(s,0).initialized && local(s,9).owner==id);OK(nvm_file_flow_take(s,9));
 uint16_t ownerarg[]={NVM_FILE_FLOW_NO_REFERENCE};OK(nvm_file_flow_call(s,8,1,ownerarg,1));OK(nvm_file_flow_put(s,0));CHECK(local(s,0).owner!=id);
 OK(nvm_file_flow_take(s,0));OK(nvm_file_flow_service(s,9,b.imports[4],NVM_FILE_FLOW_NO_REFERENCE));CHECK(!counts(s).owners);OK(nvm_file_flow_store(s,3));
 CHECK(nvm_file_flow_obligation(s,6,&event));CHECK(event.outcomes[0]==NVM_FILE_FLOW_INPUT_CONSUMED && event.outcomes[1]==NVM_FILE_FLOW_INPUT_CONSUMED);
 NvmFileFlowState *a=NULL,*e=NULL;OK(nvm_file_flow_refine(s,3,&a,&e));OK(nvm_file_flow_take_result(a,3,NVM_FILE_FLOW_ARM_OK));OK(nvm_file_flow_pop(a));finish(a);
 OK(nvm_file_flow_take_result(e,3,NVM_FILE_FLOW_ARM_ERROR));OK(nvm_file_flow_field(e,5));OK(nvm_file_flow_pop(e));finish(e);
 nvm_file_flow_state_free(a);nvm_file_flow_state_free(e);nvm_file_flow_state_free(s);
 s=state(d,2);CHECK(!counts(s).owners && counts(s).references==1);CHECK(nvm_file_flow_service(s,10,b.imports[1],0)==NVM_FILE_FLOW_INVALID);OK(nvm_file_flow_load(s,1));OK(nvm_file_flow_service(s,10,b.imports[1],0));OK(nvm_file_flow_can_exit(s));CHECK(nvm_file_flow_end_borrow(s,0)==NVM_FILE_FLOW_INVALID);
 nvm_file_flow_state_free(s);
}
static void branches(NvmFileFlowDeclarations *d,NvmFileNominalBindings b){
 NvmFileFlowState *s=state(d,0),*a=NULL,*e=NULL;bool changed=true;
 OK(nvm_file_flow_push_scalar(s,TAG_INT));OK(nvm_file_flow_construct(s,4,0));OK(nvm_file_flow_store(s,6));
 OK(nvm_file_flow_refine(s,6,&a,&e));CHECK(counts(a).reachable && !counts(e).reachable);CHECK(nvm_file_flow_push_scalar(e,TAG_INT)==NVM_FILE_FLOW_UNRESOLVED);
 OK(nvm_file_flow_join(a,e,&changed));CHECK(!changed);OK(nvm_file_flow_join(e,a,&changed));CHECK(changed && counts(e).reachable);nvm_file_flow_state_free(a);nvm_file_flow_state_free(e);
 OK(nvm_file_flow_clone(s,&a));error_value(a);OK(nvm_file_flow_construct(a,4,1));OK(nvm_file_flow_store(a,6));
 OK(nvm_file_flow_join(s,a,&changed));CHECK(changed && local(s,6).arm==NVM_FILE_FLOW_ARM_UNKNOWN);OK(nvm_file_flow_join(s,a,&changed));CHECK(!changed);nvm_file_flow_state_free(a);
 OK(nvm_file_flow_clone(s,&a));OK(nvm_file_flow_clone(s,&e));OK(nvm_file_flow_service(a,20,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE));OK(nvm_file_flow_put(a,1));OK(nvm_file_flow_service(e,20,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE));OK(nvm_file_flow_put(e,1));CHECK(local(a,1).owner!=local(e,1).owner);changed=true;CHECK(nvm_file_flow_join(a,e,&changed)==NVM_FILE_FLOW_UNRESOLVED && changed);
 nvm_file_flow_state_free(a);nvm_file_flow_state_free(e);nvm_file_flow_state_free(s);
 s=opened(d,b);OK(nvm_file_flow_region_begin(s));OK(nvm_file_flow_borrow(s,0,20));OK(nvm_file_flow_clone(s,&a));OK(nvm_file_flow_clone(s,&e));
 OK(nvm_file_flow_service(a,30,b.imports[2],20));OK(nvm_file_flow_pop(a));OK(nvm_file_flow_service(e,30,b.imports[3],20));OK(nvm_file_flow_pop(e));
 changed=false;CHECK(nvm_file_flow_join(a,e,&changed)==NVM_FILE_FLOW_UNRESOLVED && !changed);
 CHECK(nvm_file_flow_service(a,30,b.imports[3],20)==NVM_FILE_FLOW_UNRESOLVED);CHECK(!counts(a).stack);
 OK(nvm_file_flow_service(e,31,b.imports[2],20));OK(nvm_file_flow_pop(e));
 nvm_file_flow_state_free(a);nvm_file_flow_state_free(e);nvm_file_flow_state_free(s);
 s=state(d,0);a=state(d,0);CHECK(nvm_file_flow_join(s,a,&changed)==NVM_FILE_FLOW_UNRESOLVED);nvm_file_flow_state_free(a);
 OK(nvm_file_flow_service(s,40,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE));OK(nvm_file_flow_put(s,1));OK(nvm_file_flow_refine(s,1,&a,&e));OK(nvm_file_flow_move(a,1,10));CHECK(local(a,10).arm==NVM_FILE_FLOW_ARM_UNKNOWN);CHECK(nvm_file_flow_take_result(a,10,NVM_FILE_FLOW_ARM_OK)==NVM_FILE_FLOW_INVALID);
 nvm_file_flow_state_free(a);nvm_file_flow_state_free(e);nvm_file_flow_state_free(s);
}
static void limits(NvmFileFlowDeclarations *d,NvmFileNominalBindings b){
 NvmFileFlowState *s=state(d,0);for(unsigned i=0;i<NVM_FILE_FLOW_STACK;i++)OK(nvm_file_flow_push_scalar(s,TAG_INT));CHECK(nvm_file_flow_push_scalar(s,TAG_INT)==NVM_FILE_FLOW_LIMIT);CHECK(nvm_file_flow_service(s,1,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE)==NVM_FILE_FLOW_LIMIT);CHECK(counts(s).stack==256 && !counts(s).owners);nvm_file_flow_state_free(s);
 s=state(d,0);for(unsigned i=0;i<256;i++){OK(nvm_file_flow_service(s,i,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE));}CHECK(counts(s).owners==256);CHECK(nvm_file_flow_service(s,256,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE)==NVM_FILE_FLOW_LIMIT);for(unsigned i=0;i<256;i++)OK(nvm_file_flow_drop_stack(s));CHECK(counts(s).cleanup_obligations==256);CHECK(nvm_file_flow_service(s,256,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE)==NVM_FILE_FLOW_LIMIT);CHECK(!counts(s).owners);finish(s);nvm_file_flow_state_free(s);
 s=opened(d,b);for(unsigned i=0;i<256;i++)OK(nvm_file_flow_region_begin(s));CHECK(nvm_file_flow_region_begin(s)==NVM_FILE_FLOW_LIMIT);OK(nvm_file_flow_borrow(s,0,255));CHECK(nvm_file_flow_drop_local(s,0)==NVM_FILE_FLOW_INVALID);for(unsigned i=0;i<256;i++)OK(nvm_file_flow_region_end(s));OK(nvm_file_flow_drop_local(s,0));finish(s);nvm_file_flow_state_free(s);
#ifdef FLOW_INSTRUMENT
 s=state(d,0);uint64_t saved=d->next_identity;d->next_identity=UINT64_MAX;
 CHECK(nvm_file_flow_region_begin(s)==NVM_FILE_FLOW_LIMIT);CHECK(nvm_file_flow_service(s,1,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE)==NVM_FILE_FLOW_LIMIT);NvmFileFlowState *out=s;CHECK(nvm_file_flow_state(d,0,&out)==NVM_FILE_FLOW_LIMIT && out==s);CHECK(!counts(s).owners && !counts(s).stack);d->next_identity=saved;nvm_file_flow_state_free(s);
#endif
}
static void allocations_test(NvmModule *m,NvmFileFlowDeclarations *d,NvmFileNominalBindings b){
#ifdef FLOW_INSTRUMENT
 unsigned baseline=live;NvmFileFlowDeclarations *sentinel=d;
 for(int n=0;n<2;n++){budget=n;CHECK(nvm_file_flow_declarations(m,&sentinel)==NVM_FILE_FLOW_MEMORY && sentinel==d && live==baseline);}budget=-1;
 NvmFileFlowState *s=state(d,0),*a=s,*e=s;unsigned held=live;budget=0;CHECK(nvm_file_flow_state(d,0,&a)==NVM_FILE_FLOW_MEMORY && a==s && live==held);CHECK(nvm_file_flow_clone(s,&a)==NVM_FILE_FLOW_MEMORY && a==s && live==held);budget=-1;
 OK(nvm_file_flow_service(s,1,b.imports[0],NVM_FILE_FLOW_NO_REFERENCE));OK(nvm_file_flow_put(s,1));
 for(int n=0;n<2;n++){budget=n;CHECK(nvm_file_flow_refine(s,1,&a,&e)==NVM_FILE_FLOW_MEMORY && a==s && e==s && live==held);CHECK(local(s,1).arm==NVM_FILE_FLOW_ARM_UNKNOWN);}budget=-1;
 unsigned before=allocations;budget=0;OK(nvm_file_flow_take(s,1));OK(nvm_file_flow_put(s,10));OK(nvm_file_flow_drop_local(s,10));finish(s);CHECK(allocations==before);budget=-1;nvm_file_flow_state_free(s);CHECK(live==baseline);
#else
 (void)m;(void)d;(void)b;
#endif
}
int main(void){
 for(unsigned permutation=0;permutation<2;permutation++){
  NvmFileNominalBindings b;NvmModule *m=module(&b,permutation!=0);NvmFileFlowDeclarations *d=NULL;OK(nvm_file_flow_declarations(m,&d));
  NvmFileFlowFunction f;CHECK(nvm_file_flow_function(d,2,&f) && f.parameters==2 && f.result.catalog_ordinal==4);
  NvmFileFlowDeclaration item;CHECK(nvm_file_flow_declaration(d,2,0,&item) && item.mode==2 && item.global_index==b.layouts[0]);
  bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
  allocations_test(m,d,b);lifecycle(d,b);branches(d,b);limits(d,b);
  NvmFileFlowDeclarations *out=d;uint16_t save=m->functions[0].local_count;m->functions[0].local_count=257;CHECK(nvm_file_flow_declarations(m,&out)==NVM_FILE_FLOW_LIMIT && out==d);m->functions[0].local_count=save;
  m->functions[0].upvalue_count=1;CHECK(nvm_file_flow_declarations(m,&out)==NVM_FILE_FLOW_UNRESOLVED && out==d);m->functions[0].upvalue_count=0;
  /* The copied declarations/state survive all source and caller-plan lifetimes. */
  NvmFileFlowState *s=opened(d,b);nvm_module_free(m);nvm_file_flow_declarations_free(d);OK(nvm_file_flow_take(s,0));OK(nvm_file_flow_service(s,90,b.imports[4],NVM_FILE_FLOW_NO_REFERENCE));OK(nvm_file_flow_pop(s));finish(s);nvm_file_flow_state_free(s);
 }
#ifdef FLOW_INSTRUMENT
 CHECK(!live);
#endif
 printf("PASS %u private File flow checks; no CODE, public admission or host service execution\n",checks);return 0;
}
