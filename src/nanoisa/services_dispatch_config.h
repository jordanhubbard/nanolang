#include "services_flow_config.h"
#include "services_indirect_runtime.h"
#define SERVICE_PUBLIC_SUPPORTED 1
#define SERVICE_PUBLIC_INSTANCE_GRANTS 1
#define SERVICE_TYPE_TEXT "NvmServices"
#define SERVICE_CONST_TEXT "NVM_SERVICES_"
#define SERVICE_FN_TEXT "nvm_services_"
#define SERVICE_EMITTER_TEXT "nvm2c_services_"
#define SERVICE_PUBLIC_HEADER_TEXT "services_indirect_native_public.h"
#define SERVICE_NAME_TEXT "mixed services"
#define SERVICE_INSTALL_TEXT "services"
static inline uint32_t services_dispatch_instances(const NvmServicesIndirectHostedPlan *p) {
    NvmServicesNominalLayout type;uint32_t count=0;
    while(count<64 && nvm_services_indirect_hosted_type(p,count*9,&type))count++;
    return count;
}
static bool services_dispatch_tcp(const NvmServicesIndirectHostedPlan *p,uint32_t id) {
    NvmServicesNominalLayout type;
    return nvm_services_indirect_hosted_type(p,id/5*9+8,&type);
}
#define DISPATCH_METHOD_LOCAL(id) ((id)%5)
#define DISPATCH_ENDPOINT(p,id) services_dispatch_tcp(p,id)
#define DISPATCH_CATALOG_METHOD(p,id) (services_dispatch_tcp(p,id)?nl_socket_catalog_method((id)%5):nl_file_catalog_method((id)%5))
#define DISPATCH_TYPE_CAPACITY(p) (services_dispatch_instances(p)*9)
#define DISPATCH_TYPE_OPTIONAL(id) ((id)%9==8)
#define DISPATCH_IMPORT_COUNT(p) (services_dispatch_instances(p)*5)
