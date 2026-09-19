#include "../../src/nanoisa/service_file_nominal.h"
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
#define CHECK(x) do{checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#ifdef NOMINAL_INSTRUMENT
static int allocation_budget=-1;static unsigned allocations,live;
static void *nominal_malloc(size_t n){allocations++;if(allocation_budget==0)return NULL;if(allocation_budget>0)allocation_budget--;void *p=malloc(n);if(p)live++;return p;}
static void nominal_free(void *p){if(p){CHECK(live>0);live--;}free(p);}
#define malloc nominal_malloc
#define free nominal_free
#include "../../src/nanoisa/service_file_nominal_plan.c"
#undef malloc
#undef free
#endif
static void wr32(uint8_t *p,uint32_t x){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(x>>(8*i));}
static uint32_t string(NvmModule *m,const char *s){uint32_t n=nvm_add_string(m,s,(uint32_t)strlen(s));CHECK(n!=UINT32_MAX);return n;}
static void raw(void){
 NvmFileNominalBindings b={{0,UINT32_C(0x01020304),UINT32_MAX-1,UINT32_C(0x80000000),7},{9,8,7,6,5,4,3,2}},got;
 static const uint8_t gold[120]={2,0,1,0,5,0,0,0,8,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,1,0,0,0,4,3,2,1,2,0,0,0,254,255,255,255,3,0,0,0,0,0,0,128,4,0,0,0,7,0,0,0,
 0,0,0,0,9,0,0,0,1,0,0,0,8,0,0,0,2,0,0,0,7,0,0,0,3,0,0,0,6,0,0,0,
 4,0,0,0,5,0,0,0,5,0,0,0,4,0,0,0,6,0,0,0,3,0,0,0,7,0,0,0,2,0,0,0};
 uint8_t out[128];size_t n=17;memset(out,0xa5,sizeof out);
 CHECK(nvm_file_nominal_check(&b)==NVM_SERVICE_OK);
 CHECK(nvm_file_nominal_decode(gold,120,&got)==NVM_SERVICE_OK && !memcmp(&got,&b,sizeof b));
 CHECK(nvm_file_nominal_encode(&b,out,sizeof out,&n)==NVM_SERVICE_OK && n==120 && !memcmp(out,gold,120));
 for(size_t i=120;i<128;i++)CHECK(out[i]==0xa5);
 CHECK(nvm_file_nominal_encode(&b,NULL,SIZE_MAX,&n)==NVM_SERVICE_OK && n==120);
 for(size_t z=0;z<120;z++){
  got=b;CHECK(nvm_file_nominal_decode(gold,z,&got)==NVM_SERVICE_SIZE && !memcmp(&got,&b,sizeof b));
  memset(out,0xa5,sizeof out);n=17;CHECK(nvm_file_nominal_encode(&b,out,z,&n)==NVM_SERVICE_SIZE && n==17);
  for(size_t j=0;j<128;j++)CHECK(out[j]==0xa5);
 }
 got=b;CHECK(nvm_file_nominal_decode(gold,SIZE_MAX,&got)==NVM_SERVICE_SIZE && !memcmp(&got,&b,sizeof b));
 CHECK(nvm_file_nominal_decode(NULL,120,&got)==NVM_SERVICE_ARGUMENT);
 CHECK(nvm_file_nominal_decode(gold,120,NULL)==NVM_SERVICE_ARGUMENT);
 CHECK(nvm_file_nominal_check(NULL)==NVM_SERVICE_ARGUMENT);
 CHECK(nvm_file_nominal_encode(NULL,out,120,&n)==NVM_SERVICE_ARGUMENT);
 CHECK(nvm_file_nominal_encode(&b,out,120,NULL)==NVM_SERVICE_ARGUMENT);
 for(size_t i=0;i<120;i++){
  uint8_t bad[120];memcpy(bad,gold,120);bad[i]^=0x40;got=b;
  bool index=i>=16 && (i-16)%8>=4;
  NvmServiceResult r=nvm_file_nominal_decode(bad,120,&got);
  if(index){CHECK(r==NVM_SERVICE_OK);CHECK(nvm_file_nominal_encode(&got,out,128,&n)==NVM_SERVICE_OK && !memcmp(out,bad,120));}
  else CHECK(r!=NVM_SERVICE_OK && !memcmp(&got,&b,sizeof b));
 }
 for(unsigned group=0;group<2;group++)for(unsigned i=0;i<(group?8:5);i++){
  NvmFileNominalBindings bad=b;uint32_t *a=group?bad.layouts:bad.imports;a[i]=UINT32_MAX;
  CHECK(nvm_file_nominal_check(&bad)==NVM_SERVICE_INDEX);
  for(unsigned j=0;j<i;j++){bad=b;a=group?bad.layouts:bad.imports;a[i]=a[j];CHECK(nvm_file_nominal_check(&bad)==NVM_SERVICE_INDEX);}
 }
 union {NvmFileNominalBindings b;uint8_t bytes[128];} alias;alias.b=b;
 CHECK(nvm_file_nominal_encode(&alias.b,alias.bytes,128,&n)==NVM_SERVICE_OK && !memcmp(alias.bytes,gold,120));
 CHECK(nvm_file_nominal_decode(alias.bytes,120,&alias.b)==NVM_SERVICE_OK && !memcmp(&alias.b,&b,sizeof b));
 uint8_t local[120];memcpy(local,gold,120);CHECK(nvm_file_nominal_decode(local,120,&got)==NVM_SERVICE_OK);memset(local,0,120);CHECK(!memcmp(&got,&b,sizeof b));
 NvmServiceBindings old;CHECK(nvm_service_bindings_decode(gold,120,&old)==NVM_SERVICE_SIZE);
}
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
static void reject(NvmModule *m){NvmFileNominalPlan *out=(NvmFileNominalPlan *)(void *)m;CHECK(nvm_file_nominal_plan(m,&out)!=NVM_FILE_NOMINAL_DESCRIBED);CHECK(out==(NvmFileNominalPlan *)(void *)m);}
static void query(bool permutation){
 NvmFileNominalBindings b;NvmModule *m=fixture(permutation,&b);NvmFileNominalPlan *plan=NULL;
 CHECK(nvm_file_nominal_plan(m,&plan)==NVM_FILE_NOMINAL_DESCRIBED && plan);CHECK(nvm_file_nominal_layout_count(plan)==9);
 NvmFileNominalLayout row,saved;memset(&saved,0xa5,sizeof saved);
 for(unsigned i=0;i<8;i++){
  CHECK(nvm_file_nominal_type(plan,i,&row));CHECK(row.global_index==b.layouts[i] && row.catalog_ordinal==i && row.ownership_flags==((i==0||i==3)?3:1));
  NvmFileNominalLayout by_source;CHECK(nvm_file_nominal_source(plan,row.layout_kind,row.source_ordinal,&by_source));CHECK(by_source.global_index==row.global_index);
 }
 CHECK(nvm_file_nominal_layout(plan,0,&row) && row.category==NVM_FILE_CATEGORY_UNKNOWN && row.catalog_ordinal==NVM_V2_NO_INDEX);
 for(unsigned i=0;i<5;i++){uint32_t index=99;CHECK(nvm_file_nominal_import(plan,i,&index) && index==b.imports[i]);}
 row=saved;CHECK(!nvm_file_nominal_layout(plan,9,&row) && !memcmp(&row,&saved,sizeof row));CHECK(!nvm_file_nominal_type(plan,8,&row) && !memcmp(&row,&saved,sizeof row));
 CHECK(!nvm_file_nominal_source(plan,NVM_V2_LAYOUT_ENUM,0,&row) && !memcmp(&row,&saved,sizeof row));
 CHECK(!nvm_file_nominal_source(plan,NVM_V2_LAYOUT_STRUCT,99,&row));uint32_t index=99;CHECK(!nvm_file_nominal_import(plan,5,&index) && index==99);
 CHECK(!nvm_file_nominal_layout(NULL,0,&row));CHECK(!nvm_file_nominal_layout(plan,0,NULL));CHECK(nvm_file_nominal_layout_count(NULL)==0);
 bool needs=true;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);NvmLayoutAuthority authority=NVM_LAYOUT_AUTHORITY_UNKNOWN;CHECK(nvm_ownership_layout_authority(m,b.layouts[0],&authority)!=NVM_V2_OK && authority==NVM_LAYOUT_AUTHORITY_UNKNOWN);
 CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);CHECK(nvm_service_bindings_validate(m)==NVM_V2_OK);
 NvmV2Module bridge={0};CHECK(nvm_v2_from_nvm_module(m,&bridge)==NVM_V2_OK);CHECK(bridge.service_data==m->service_data && bridge.ownership_data==m->ownership_data);CHECK(nvm_v2_service_bindings_validate(&bridge)==NVM_V2_OK);NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&bridge,&copy)==NVM_V2_OK && copy && copy->service_data!=m->service_data && copy->ownership_data!=m->ownership_data);CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));nvm_module_free(copy);nvm_v2_module_free(&bridge);
 uint8_t *service=m->service_data;uint32_t service_size=m->service_size;m->service_data=NULL;m->service_size=0;CHECK(nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK);m->service_data=service;m->service_size=service_size;
 for(uint32_t i=0;i<m->ownership_size;i++){m->ownership_data[i]^=0x80;reject(m);m->ownership_data[i]^=0x80;}
 for(uint32_t i=0;i<m->layout_size;i++){m->layout_data[i]^=0x80;reject(m);m->layout_data[i]^=0x80;}
 for(uint32_t i=0;i<m->service_size;i++){m->service_data[i]^=0x40;reject(m);m->service_data[i]^=0x40;}
 for(uint32_t n=0;n<68;n++){m->ownership_size=n;reject(m);}m->ownership_size=68;
 uint32_t layout_size=m->layout_size;for(uint32_t n=0;n<layout_size;n++){m->layout_size=n;reject(m);}m->layout_size=layout_size;
 m->struct_count++;reject(m);m->struct_count--;m->function_count++;reject(m);m->function_count--;
 m->functions[0].arity++;reject(m);m->functions[0].arity--;m->functions[0].local_count++;reject(m);m->functions[0].local_count--;
 m->functions[0].result_count=2;reject(m);m->functions[0].result_count=1;m->function_param_types[0][0]=TAG_FLOAT;reject(m);m->function_param_types[0][0]=TAG_INT;
 m->ownership_data[45]=1;reject(m);m->ownership_data[45]=2;m->ownership_data[53]=2;reject(m);m->ownership_data[53]=0;
 m->ownership_data[0]=2;reject(m);m->ownership_data[0]=1;
 uint32_t imported=b.imports[1];m->imports[imported].kind=NVM_IMPORT_FFI;reject(m);m->imports[imported].kind=NVM_IMPORT_SERVICE;
 m->import_param_types[imported][1]=TAG_U8;reject(m);m->import_param_types[imported][1]=TAG_INT;
 m->module_ref_count=1;reject(m);m->module_ref_count=0;m->callback_contract_count=1;reject(m);m->callback_contract_count=0;
 for(uint32_t i=0;i<m->string_count;i++)if(m->string_lengths[i] && !strncmp(m->strings[i],"nsi:",4)){m->strings[i][0]='x';reject(m);m->strings[i][0]='n';}
 uint8_t count_save[4];memcpy(count_save,m->layout_data,4);wr32(m->layout_data,65537);NvmFileNominalPlan *sentinel=plan;CHECK(nvm_file_nominal_plan(m,&sentinel)==NVM_FILE_NOMINAL_LIMIT && sentinel==plan);memcpy(m->layout_data,count_save,4);
#ifdef NOMINAL_INSTRUMENT
 unsigned old_live=live,old_allocations=allocations;allocation_budget=0;sentinel=plan;
 CHECK(nvm_file_nominal_plan(m,&sentinel)==NVM_FILE_NOMINAL_MEMORY && sentinel==plan && live==old_live && allocations==old_allocations+1);
 allocation_budget=1;NvmFileNominalPlan *again=NULL;CHECK(nvm_file_nominal_plan(m,&again)==NVM_FILE_NOMINAL_DESCRIBED && live==old_live+1);nvm_file_nominal_plan_free(again);CHECK(live==old_live);allocation_budget=-1;
#endif
 nvm_module_free(m);for(unsigned i=0;i<8;i++){CHECK(nvm_file_nominal_type(plan,i,&row));CHECK(row.global_index==b.layouts[i]);}nvm_file_nominal_plan_free(plan);
}
int main(void){raw();query(false);query(true);CHECK(nl_file_catalog_type(8)==NULL);nvm_file_nominal_plan_free(NULL);
#ifdef NOMINAL_INSTRUMENT
 CHECK(live==0);
#endif
 printf("PASS %u private File nominal checks; old authority and consumers refuse\n",checks);return 0;}
