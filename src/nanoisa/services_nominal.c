#include "services_nominal.h"
#include "nvm_v2_sections.h"
#include "../nsi_websocket_plan.h"
NvmMultiNominalStatus nvm_services_nominal_plan(const NvmModule *m,NvmServicesNominalPlan **out) {
 return nvm_multi_nominal_plan(m,out);
}
static NvmServicesNominalLayout row(NvmMultiNominalLayout v) {
 NvmServicesCategory category=NVM_SERVICES_CATEGORY_UNKNOWN;
 uint32_t ordinal=NVM_V2_NO_INDEX;
 if(v.instance!=NVM_V2_NO_INDEX){
  ordinal=v.instance*9+v.catalog_ordinal;
  category=v.catalog_ordinal==0?NVM_SERVICES_CATEGORY_OWNER:
   v.catalog_ordinal==3?NVM_SERVICES_CATEGORY_OWNER_RESULT:
   v.layout_kind==NVM_V2_LAYOUT_STRUCT?NVM_SERVICES_CATEGORY_RECORD:NVM_SERVICES_CATEGORY_SCALAR_RESULT;
 }
 return (NvmServicesNominalLayout){v.global_index,ordinal,v.source_ordinal,v.layout_kind,v.ownership_flags,category};
}
bool nvm_services_nominal_layout(const NvmServicesNominalPlan *p,uint32_t i,NvmServicesNominalLayout *out) {
 NvmMultiNominalLayout v;if(!out || !nvm_multi_nominal_layout(p,i,&v))return false;
 *out=row(v);return true;
}
bool nvm_services_nominal_type(const NvmServicesNominalPlan *p,uint32_t i,NvmServicesNominalLayout *out) {
 NvmMultiNominalLayout v;if(!out || !nvm_multi_nominal_type(p,i/9,i%9,&v))return false;
 *out=row(v);return true;
}
bool nvm_services_nominal_source(const NvmServicesNominalPlan *p,uint8_t kind,uint32_t source,NvmServicesNominalLayout *out) {
 if(!out || (kind!=NVM_V2_LAYOUT_STRUCT && kind!=NVM_V2_LAYOUT_UNION))return false;
 for(uint32_t i=0;i<nvm_multi_nominal_layout_count(p);i++) {
  NvmMultiNominalLayout v;
  if(nvm_multi_nominal_layout(p,i,&v) && v.layout_kind==kind && v.source_ordinal==source){*out=row(v);return true;}
 }
 return false;
}
bool nvm_services_nominal_import(const NvmServicesNominalPlan *p,uint32_t i,uint32_t *out) {
 return nvm_multi_nominal_import(p,i/5,i%5,out);
}
static uint32_t catalog(const NvmServicesNominalPlan *p,uint32_t instance) {
 NvmMultiNominalLayout v;return nvm_multi_nominal_type(p,instance,0,&v)?v.catalog:0;
}
const NlServicePlanType *nvm_services_catalog_type(const NvmServicesNominalPlan *p,uint32_t i) {
 uint32_t c=catalog(p,i/9);
 return c==1?nl_file_catalog_type(i%9):c==2?nl_socket_catalog_type(i%9):c==3?nl_websocket_catalog_type(i%9):NULL;
}
const NlServicePlanMethod *nvm_services_catalog_method(const NvmServicesNominalPlan *p,uint32_t i) {
 uint32_t c=catalog(p,i/5);
 return c==1?nl_file_catalog_method(i%5):c==2?nl_socket_catalog_method(i%5):c==3?nl_websocket_catalog_method(i%5):NULL;
}
uint32_t nvm_services_catalog_types(const NvmServicesNominalPlan *p,uint32_t type) {
 uint32_t c=catalog(p,type/9);return nvm_multi_nominal_catalog_types(c);
}
bool nvm_services_catalog_endpoint(const NvmServicesNominalPlan *p,uint32_t type) {
 return type%9==8 && catalog(p,type/9)==2;
}
