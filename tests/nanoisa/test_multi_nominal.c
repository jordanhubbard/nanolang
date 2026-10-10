/* I exercise mixed/repeated catalog identity through metadata transport. */
#include "../../src/nanoisa/service_bindings_module.h"
#include "../../src/nanoisa/assembler.h"
#include "../../src/nanoisa/retained_layouts.h"
#include "../../src/nanoisa/ownership_contracts.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanoisa/nvm2c.h"
#include "../../src/nanovm/vm.h"
#include "../../src/nsi_file_catalog.h"
#include "../../src/nsi_websocket_plan.h"
#include "../../src/nanoisa/services_flow.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;char **g_argv;
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#ifdef SERVICE_ALLOC_TEST
static int budget=-1;
void *service_test_malloc(size_t n){if(!budget)return NULL;if(budget>0)budget--;return malloc(n);}
void *service_test_calloc(size_t n,size_t z){if(!budget)return NULL;if(budget>0)budget--;return calloc(n,z);}
void *service_test_realloc(void *p,size_t n){if(!budget)return NULL;if(budget>0)budget--;return realloc(p,n);}
#endif
static void wr32(uint8_t *p,uint32_t v){for(unsigned j=0;j<4;j++)p[j]=(uint8_t)(v>>(8*j));}
static uint32_t rd32(const uint8_t *p){return p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static unsigned types(const NvmServiceInstance *v){return v->catalog==1?8:v->catalog==2?9:7;}
static const NlServicePlanType *type(const NvmServiceInstance *v,unsigned j){return v->catalog==1?nl_file_catalog_type(j):v->catalog==2?nl_socket_catalog_type(j):nl_websocket_catalog_type(j);}
static uint32_t str(NvmModule *m,const char *s){return nvm_add_string(m,s,(uint32_t)strlen(s));}
static void descriptor(uint8_t *p,uint8_t tag,uint8_t mode,uint32_t layout){p[0]=tag;p[1]=mode;wr32(p+4,layout);}
static NvmModule *fixture_catalogs(bool reverse,NvmMultiNominalBindings *b,const uint32_t *catalogs,unsigned instances){
 AsmResult result={0};NvmModule *m=asm_assemble(".function main 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n.entry main\n",&result);CHECK(m);
 memset(b,0,sizeof *b);b->count=instances;unsigned total=catalogs[0]==3?0:1;
 CHECK(instances<=5);
 NvmV2Layout layouts[46]={0};NvmV2LayoutField fields[5][9][11]={0};
 layouts[0].kind=NVM_V2_LAYOUT_TUPLE;layouts[0].name_idx=NVM_V2_NO_INDEX;
 for(unsigned i=0;i<instances;i++){
  NvmServiceInstance *v=&b->instances[i];v->catalog=catalogs[i];
  for(unsigned j=0;j<9;j++)v->layouts[j]=j<types(v)?total+j:UINT32_MAX;
  if(reverse){uint32_t x=v->layouts[0];v->layouts[0]=v->layouts[1];v->layouts[1]=x;}
  total+=types(v);
  uint32_t module=str(m,v->catalog==1?nl_file_catalog_interface():v->catalog==2?nl_socket_catalog_interface():nl_websocket_catalog_interface());
  unsigned methods=v->catalog==3?4:5;
  for(unsigned j=methods;j<5;j++)v->imports[j]=UINT32_MAX;
  for(unsigned j=0;j<methods;j++){
   unsigned k=reverse?methods-1-j:j;
   const NlServicePlanMethod *method=v->catalog==1?nl_file_catalog_method(k):v->catalog==2?nl_socket_catalog_method(k):nl_websocket_catalog_method(k);
   uint8_t tags[3];unsigned count=(unsigned)method->param_count-1;
   for(unsigned n=0;n<count;n++)tags[n]=!strcmp(method->params[n].type_id,"nsi:core/int")?TAG_INT:!strcmp(method->params[n].type_id,"nsi:core/string")?TAG_STRING:TAG_STRUCT;
   uint32_t index=nvm_add_import(m,module,str(m,method->id),(uint16_t)count,TAG_UNION,tags);
   CHECK(index!=UINT32_MAX);m->imports[index].kind=NVM_IMPORT_SERVICE;v->imports[k]=index;
  }
  for(unsigned j=0;j<types(v);j++){
   const NlServicePlanType *t=type(v,j);NvmV2Layout *l=&layouts[v->layouts[j]];
   l->kind=t->kind==NL_NSI_TYPE_VARIANT?NVM_V2_LAYOUT_UNION:NVM_V2_LAYOUT_STRUCT;
   l->name_idx=str(m,t->id);l->field_count=(uint16_t)t->member_count;l->fields=fields[i][j];
   if(l->kind==NVM_V2_LAYOUT_STRUCT)m->struct_count++;else m->union_count++;
   for(unsigned k=0;k<t->member_count;k++){
    const char *id=t->members[k].type_id;uint8_t tag=TAG_VOID;uint32_t nested=NVM_V2_NO_INDEX;
    if(id){if(!strcmp(id,"nsi:core/int"))tag=TAG_INT;else if(!strcmp(id,"nsi:core/bool"))tag=TAG_BOOL;else if(!strcmp(id,"nsi:core/string"))tag=TAG_STRING;
     else for(unsigned n=0;n<types(v);n++)if(!strcmp(id,type(v,n)->id)){tag=type(v,n)->kind==NL_NSI_TYPE_VARIANT?TAG_UNION:TAG_STRUCT;nested=v->layouts[n];}}
    fields[i][j][k]=(NvmV2LayoutField){tag,nested,str(m,t->members[k].id)};
   }
  }
 }
 NvmV2Layouts table={layouts,total};CHECK(nvm_retain_layouts(m,&table)==NVM_V2_OK);
 unsigned locals=instances>=3?3:1,parameters=instances>=3?2:0;
 m->functions[0].arity=(uint8_t)parameters;m->functions[0].local_count=(uint16_t)locals;
 uint8_t params[]={TAG_STRUCT,TAG_STRUCT};CHECK(nvm_set_function_param_types(m,0,params,(uint16_t)parameters));
 unsigned functions=(8+total+3)&~3u;
 m->ownership_size=functions+16+8*locals;m->ownership_data=calloc(1,m->ownership_size);CHECK(m->ownership_data);uint8_t *o=m->ownership_data;
 wr32(o,1);wr32(o+4,total);
 for(unsigned i=0;i<instances;i++)for(unsigned j=0;j<types(&b->instances[i]);j++)o[8+b->instances[i].layouts[j]]=(j==0||j==3)?3:1;
 wr32(o+functions,1);o[functions+4]=(uint8_t)locals;o[functions+6]=(uint8_t)parameters;descriptor(o+functions+8,TAG_INT,0,NVM_V2_NO_INDEX);
 for(unsigned i=0;i<locals;i++)descriptor(o+functions+16+8*i,TAG_STRUCT,i<parameters?(uint8_t)(i+1):0,b->instances[i].layouts[0]);
 size_t size=0;CHECK(nvm_multi_nominal_encode(b,NULL,0,&size)==NVM_SERVICE_OK);m->service_data=malloc(size);CHECK(m->service_data);
 CHECK(nvm_multi_nominal_encode(b,m->service_data,size,&size)==NVM_SERVICE_OK);m->service_size=(uint32_t)size;return m;
}
static NvmModule *fixture(bool reverse,NvmMultiNominalBindings *b){
 const uint32_t catalogs[]={1,2,1};return fixture_catalogs(reverse,b,catalogs,3);
}
static void raw(void){
 NvmMultiNominalBindings b={0},out;b.count=64;
 for(unsigned i=0;i<64;i++){b.instances[i].catalog=i%3+1;for(unsigned j=0;j<5;j++)b.instances[i].imports[j]=j<(b.instances[i].catalog==3?4u:5u)?5*i+j:UINT32_MAX;
  for(unsigned j=0;j<9;j++)b.instances[i].layouts[j]=j<types(&b.instances[i])?9*i+j:UINT32_MAX;}
 uint8_t wire[NVM_MULTI_NOMINAL_MAX_BYTES],copy[sizeof wire];size_t n=0;
 CHECK(nvm_multi_nominal_encode(&b,wire,sizeof wire,&n)==NVM_SERVICE_OK && n==sizeof wire);
 CHECK(nvm_multi_nominal_decode(wire,n,&out)==NVM_SERVICE_OK && !memcmp(&b,&out,sizeof b));
 for(unsigned count=1;count<=64;count++){
  b.count=count;size_t z=0;CHECK(nvm_multi_nominal_encode(&b,copy,sizeof copy,&z)==NVM_SERVICE_OK && z==16+64*count);
  CHECK(nvm_multi_nominal_decode(copy,z,&out)==NVM_SERVICE_OK && out.count==count);
  CHECK(!memcmp(b.instances,out.instances,count*sizeof b.instances[0]));
 }
 b.count=64;

 for(size_t k=0;k<n;k++){memset(&out,0xa5,sizeof out);NvmMultiNominalBindings old=out;CHECK(nvm_multi_nominal_decode(wire,k,&out)!=NVM_SERVICE_OK && !memcmp(&out,&old,sizeof out));}
 for(size_t k=0;k<n;k++){
  memcpy(copy,wire,n);copy[k]^=0x80;memset(&out,0xa5,sizeof out);NvmMultiNominalBindings old=out;
  NvmServiceResult r=nvm_multi_nominal_decode(copy,n,&out);
  if(r!=NVM_SERVICE_OK)CHECK(!memcmp(&old,&out,sizeof out));else{uint8_t again[sizeof wire];size_t z=0;CHECK(nvm_multi_nominal_encode(&out,again,sizeof again,&z)==NVM_SERVICE_OK && z==n && !memcmp(copy,again,n));}
 }
 size_t size=999;memset(copy,0xa5,sizeof copy);CHECK(nvm_multi_nominal_encode(&b,copy,n-1,&size)==NVM_SERVICE_SIZE && size==999);
 for(unsigned i=0;i<sizeof copy;i++)CHECK(copy[i]==0xa5);
 b.instances[63].imports[0]=b.instances[0].imports[0];CHECK(nvm_multi_nominal_check(&b)==NVM_SERVICE_INDEX);
 b.instances[63].imports[0]=315;b.instances[63].layouts[0]=b.instances[0].layouts[0];CHECK(nvm_multi_nominal_check(&b)==NVM_SERVICE_INDEX);
 b.count=0;CHECK(nvm_multi_nominal_check(&b)==NVM_SERVICE_COUNT);b.count=65;CHECK(nvm_multi_nominal_check(&b)==NVM_SERVICE_COUNT);
 union {NvmMultiNominalBindings b;uint8_t bytes[NVM_MULTI_NOMINAL_MAX_BYTES];} alias;
 CHECK(nvm_multi_nominal_decode(wire,n,&alias.b)==NVM_SERVICE_OK);
 CHECK(nvm_multi_nominal_encode(&alias.b,alias.bytes,sizeof alias.bytes,&size)==NVM_SERVICE_OK && !memcmp(alias.bytes,wire,n));
 CHECK(nvm_multi_nominal_decode(alias.bytes,n,&alias.b)==NVM_SERVICE_OK && alias.b.count==64);
}
static void refuse(NvmModule *m){NvmMultiNominalPlan *out=(void *)&checks;CHECK(nvm_multi_nominal_plan(m,&out)!=NVM_MULTI_NOMINAL_DESCRIBED && out==(void *)&checks);CHECK(nvm_service_bindings_validate(m)!=NVM_V2_OK);}
static void consumers(NvmModule *m){
 CHECK(!nvm_verify(m).ok);char error[256];CHECK(nvm2c_emit(m,error,sizeof error)==NULL);
 VmState vm;vm_init(&vm,m);NanoValue value=val_int(123);CHECK(vm_invoke(&vm,0,NULL,0,&value)!=VM_OK && value.tag==TAG_INT && value.as.i64==123);vm_destroy(&vm);
}
static void module(bool reverse){
 NvmMultiNominalBindings b;NvmModule *m=fixture(reverse,&b);NvmMultiNominalPlan *p=NULL;
 CHECK(nvm_multi_nominal_plan(m,&p)==NVM_MULTI_NOMINAL_DESCRIBED);CHECK(nvm_multi_nominal_instance_count(p)==3);
 for(unsigned i=0;i<3;i++)for(unsigned j=0;j<types(&b.instances[i]);j++){
  NvmMultiNominalLayout row;CHECK(nvm_multi_nominal_type(p,i,j,&row));CHECK(row.global_index==b.instances[i].layouts[j] && row.instance==i && row.catalog==b.instances[i].catalog && row.catalog_ordinal==j);
 }
 for(unsigned i=0;i<3;i++)for(unsigned j=0;j<5;j++){uint32_t index=UINT32_MAX;CHECK(nvm_multi_nominal_import(p,i,j,&index) && index==b.instances[i].imports[j]);}
 NvmMultiNominalLayout unchanged={0},invalid=unchanged;uint32_t index=123;
 CHECK(!nvm_multi_nominal_type(p,3,0,&invalid) && !memcmp(&invalid,&unchanged,sizeof invalid));
 CHECK(!nvm_multi_nominal_type(p,0,8,&invalid) && !memcmp(&invalid,&unchanged,sizeof invalid));
 CHECK(!nvm_multi_nominal_import(p,0,5,&index) && index==123);
 CHECK(nvm_service_bindings_validate(m)==NVM_V2_OK);consumers(m);
 /* Same-shaped earlier File owner is structurally legal, but not this Result's instance. */
 uint32_t result=b.instances[2].layouts[3];size_t pos=4;
 for(uint32_t i=0;i<result;i++)pos+=8+12*(m->layout_data[pos+2]|((unsigned)m->layout_data[pos+3]<<8));
 uint32_t owner=rd32(m->layout_data+pos+12);wr32(m->layout_data+pos+12,b.instances[0].layouts[0]);refuse(m);wr32(m->layout_data+pos+12,owner);
 unsigned at=8+b.instances[1].layouts[0];m->ownership_data[at]=1;refuse(m);m->ownership_data[at]=3;
 m->ownership_data[61]=3;refuse(m);m->ownership_data[61]=2;
 m->imports[0].kind=NVM_IMPORT_FFI;refuse(m);m->imports[0].kind=NVM_IMPORT_SERVICE;
 m->import_count--;refuse(m);m->import_count++;
 NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);CHECK(nvm_v2_service_bindings_validate(&v)==NVM_V2_OK);
 size_t n=0;CHECK(nvm_v2_module_serialize(&v,NULL,0,&n)==NVM_V2_OK);uint8_t *wire=malloc(n);CHECK(wire);CHECK(nvm_v2_module_serialize(&v,wire,n,&n)==NVM_V2_OK);
 NvmV2Header header;CHECK(nvm_v2_read_header(wire,n,&header)==NVM_V2_OK);
 uint8_t *bad=malloc(n);CHECK(bad);
 for(uint32_t k=0;k<header.section_count;k++){
  NvmV2SectionEntry section;CHECK(nvm_v2_read_section(wire,n,&header,k,&section)==NVM_V2_OK);
  if(section.type==NVM_V2_SECTION_SERVICE_BINDINGS){
   const unsigned offsets[]={0,2,4,8,12,16,20,22,24};
   for(unsigned j=0;j<sizeof offsets/sizeof offsets[0];j++){
    memcpy(bad,wire,n);bad[section.offset+offsets[j]]^=0x80;
    NvmV2Header h=header;h.checksum=nvm_crc32(bad+NVM_V2_HEADER_SIZE,(uint32_t)(n-NVM_V2_HEADER_SIZE));nvm_v2_write_header(bad,&h);
    NvmV2Module rejected={0};CHECK(nvm_v2_module_deserialize(bad,n,&rejected)!=NVM_V2_OK);nvm_v2_module_free(&rejected);
   }
  }
 }
 free(bad);
 const char *path=getenv("MULTI_MODULE_WIRE");if(path){FILE *f=fopen(path,"wb");CHECK(f && fwrite(wire,1,n,f)==n);CHECK(fclose(f)==0);}
 NvmV2Module decoded={0};CHECK(nvm_v2_module_deserialize(wire,n,&decoded)==NVM_V2_OK);NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
 CHECK(copy->service_size==m->service_size && !memcmp(copy->service_data,m->service_data,m->service_size));
 CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
 CHECK(copy->layout_size==m->layout_size && !memcmp(copy->layout_data,m->layout_data,m->layout_size));consumers(copy);
 nvm_v2_module_free(&decoded);nvm_v2_module_free(&v);free(wire);nvm_module_free(copy);nvm_module_free(m);
 NvmMultiNominalLayout row;CHECK(nvm_multi_nominal_type(p,2,0,&row) && row.instance==2);nvm_multi_nominal_plan_free(p);
}
static void websocket(bool reverse,const uint32_t *catalogs,unsigned count){
 NvmMultiNominalBindings b;NvmModule *m=fixture_catalogs(reverse,&b,catalogs,count);
 NvmMultiNominalPlan *p=NULL;CHECK(nvm_multi_nominal_plan(m,&p)==NVM_MULTI_NOMINAL_DESCRIBED);
 unsigned ws=0;while(catalogs[ws]!=3)ws++;
 for(unsigned i=0;i<count;i++){
  for(unsigned j=0;j<types(&b.instances[i]);j++){
   NvmMultiNominalLayout row;CHECK(nvm_multi_nominal_type(p,i,j,&row));
   CHECK(row.instance==i && row.catalog==catalogs[i] && row.global_index==b.instances[i].layouts[j]);
  }
  for(unsigned j=0;j<(catalogs[i]==3?4u:5u);j++){
   uint32_t index=UINT32_MAX;CHECK(nvm_multi_nominal_import(p,i,j,&index) && index==b.instances[i].imports[j]);
  }
 }
 uint32_t index=123;CHECK(!nvm_multi_nominal_import(p,ws,4,&index) && index==123);
 NvmMultiNominalLayout row={0},old=row;CHECK(!nvm_multi_nominal_type(p,ws,7,&row) && !memcmp(&old,&row,sizeof row));
 NvmServicesNominalPlan *flow=(void *)&checks;
 CHECK(nvm_services_nominal_plan(m,&flow)==NVM_MULTI_NOMINAL_INVALID && flow==(void *)&checks);
 NvmServicesFlowDeclarations *declarations=(void *)&checks;
 CHECK(nvm_services_flow_declarations(m,&declarations)!=NVM_SERVICES_FLOW_OK && declarations==(void *)&checks);
 CHECK(nvm_service_bindings_validate(m)==NVM_V2_OK);consumers(m);
 NvmV2Module v={0},decoded={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);
 size_t size=0;CHECK(nvm_v2_module_serialize(&v,NULL,0,&size)==NVM_V2_OK);
 uint8_t *bytes=malloc(size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&v,bytes,size,&size)==NVM_V2_OK);
 CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
 NvmModule *copy=NULL;CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
 CHECK(copy->service_size==m->service_size && !memcmp(copy->service_data,m->service_data,m->service_size));
 CHECK(copy->layout_size==m->layout_size && !memcmp(copy->layout_data,m->layout_data,m->layout_size));
 CHECK(copy->ownership_size==m->ownership_size && !memcmp(copy->ownership_data,m->ownership_data,m->ownership_size));
 consumers(copy);nvm_module_free(copy);nvm_v2_module_free(&decoded);nvm_v2_module_free(&v);free(bytes);
 /* I reject padding that invents an extra method or layout. */
 for(unsigned slot=0;slot<3;slot++){
  uint32_t *value=slot?&b.instances[ws].layouts[6+slot]:&b.instances[ws].imports[4];
  *value=123;CHECK(nvm_multi_nominal_check(&b)==NVM_SERVICE_RESERVED);*value=UINT32_MAX;
 }
 uint8_t *tag=&m->import_param_types[b.instances[ws].imports[0]][0];CHECK(*tag==TAG_STRING);
 *tag=TAG_INT;refuse(m);*tag=TAG_STRING;
 size_t pos=4;uint32_t message=b.instances[ws].layouts[2];
 for(uint32_t i=0;i<message;i++)pos+=8+12*(m->layout_data[pos+2]|((unsigned)m->layout_data[pos+3]<<8));
 CHECK(m->layout_data[pos+20]==TAG_STRING);m->layout_data[pos+20]=TAG_INT;refuse(m);m->layout_data[pos+20]=TAG_STRING;
 if(count>1){
  unsigned last=count-1;uint32_t result=b.instances[last].layouts[3];pos=4;
  for(uint32_t i=0;i<result;i++)pos+=8+12*(m->layout_data[pos+2]|((unsigned)m->layout_data[pos+3]<<8));
  uint32_t owner=rd32(m->layout_data+pos+12);wr32(m->layout_data+pos+12,b.instances[0].layouts[0]);refuse(m);wr32(m->layout_data+pos+12,owner);
 }
 CHECK(nvm_multi_nominal_check(&b)==NVM_SERVICE_OK);
 nvm_multi_nominal_plan_free(p);nvm_module_free(m);
}
#ifdef SERVICE_ALLOC_TEST
static void allocations(bool websocket){
 const uint32_t catalogs[]={3,3,3,3,3};
 NvmMultiNominalBindings b;NvmModule *m=websocket?fixture_catalogs(false,&b,catalogs,5):fixture(false,&b);NvmMultiNominalPlan *p=(void *)&checks;
 budget=0;CHECK(nvm_multi_nominal_plan(m,&p)==NVM_MULTI_NOMINAL_MEMORY && p==(void *)&checks);budget=-1;
 bool success=false;
 for(int n=0;n<128;n++){NvmV2Module v={0};budget=n;NvmV2Result r=nvm_v2_from_nvm_module(m,&v);budget=-1;
  nvm_v2_module_free(&v);if(r==NVM_V2_OK){success=true;break;}}
 CHECK(success);
 NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);
 size_t size=0;CHECK(nvm_v2_module_serialize(&v,NULL,0,&size)==NVM_V2_OK);uint8_t *wire=malloc(size);CHECK(wire);
 CHECK(nvm_v2_module_serialize(&v,wire,size,&size)==NVM_V2_OK);
 success=false;
 for(int n=0;n<256;n++){NvmV2Module decoded={0};budget=n;NvmV2Result r=nvm_v2_module_deserialize(wire,size,&decoded);budget=-1;
  nvm_v2_module_free(&decoded);if(r==NVM_V2_OK){success=true;break;}}
 CHECK(success);success=false;
 for(int n=0;n<256;n++){NvmModule *copy=NULL;budget=n;NvmV2Result r=nvm_v2_to_nvm_module(&v,&copy);budget=-1;
  if(r==NVM_V2_OK){CHECK(copy);nvm_module_free(copy);success=true;break;}CHECK(!copy);}
 CHECK(success);free(wire);nvm_v2_module_free(&v);nvm_module_free(m);
}
#endif
int main(void){raw();module(false);module(true);
 const uint32_t one[]={3},mixed[]={1,2,3,3},five[]={3,3,3,3,3};
 for(unsigned reverse=0;reverse<2;reverse++){
  websocket(reverse,one,1);websocket(reverse,mixed,4);websocket(reverse,five,5);
 }
#ifdef SERVICE_ALLOC_TEST
 allocations(false);allocations(true);
#endif
 printf("PASS %u mixed/repeated service transport checks; no host execution\n",checks);return 0;}
