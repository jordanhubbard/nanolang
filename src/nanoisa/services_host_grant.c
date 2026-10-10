#include "services_host_grant_internal.h"
#include "file_host_grant_internal.h"
#include <stdlib.h>
#include <string.h>

static const unsigned services_host_identity=NVM_SERVICES_HOST_ABI;
struct NvmServicesHostGrant {
    const void *runtime_identity;
    unsigned abi,catalog;
    bool live;
    size_t count;
    NvmServicesHostPolicy policies[NVM_SERVICES_HOST_INSTANCES];
};
NvmServicesHostStatus nvm_services_host_enter_query(void) {
    return nvm_file_host_enter_query()==NVM_FILE_HOST_OK?NVM_SERVICES_HOST_OK:NVM_SERVICES_HOST_BUSY;
}
void nvm_services_host_leave(void) { nvm_file_host_leave(); }
static NvmServicesHostStatus services_host_check(const NvmServicesHostGrant *grant,unsigned abi,unsigned catalog) {
    if(!grant)return NVM_SERVICES_HOST_INVALID;
    const void *identity;
    memcpy(&identity,grant,sizeof identity);
    if(identity!=&services_host_identity)return NVM_SERVICES_HOST_UNRESOLVED;
    if(abi!=NVM_SERVICES_HOST_ABI ||
       catalog!=NVM_SERVICES_HOST_CATALOG || grant->abi!=abi || grant->catalog!=catalog ||
       !grant->count || grant->count>NVM_SERVICES_HOST_INSTANCES)return NVM_SERVICES_HOST_UNRESOLVED;
    return grant->live?NVM_SERVICES_HOST_OK:NVM_SERVICES_HOST_STATE;
}
NvmServicesHostStatus nvm_services_host_enter(const NvmServicesHostGrant *grant,unsigned abi,unsigned catalog) {
    NvmServicesHostStatus status=nvm_services_host_enter_query();
    if(status!=NVM_SERVICES_HOST_OK)return status;
    status=services_host_check(grant,abi,catalog);
    if(status!=NVM_SERVICES_HOST_OK)nvm_services_host_leave();
    return status;
}
NvmServicesHostStatus nvm_services_host_grant_create(const NvmServicesHostPolicy *policies,size_t count,NvmServicesHostGrant **out) {
    NvmServicesHostStatus status=nvm_services_host_enter_query();
    if(status!=NVM_SERVICES_HOST_OK)return status;
    if(!policies || !count || count>NVM_SERVICES_HOST_INSTANCES || !out || *out)status=NVM_SERVICES_HOST_INVALID;
    else for(size_t i=0;i<count;i++)if(policies[i].catalog!=NVM_SERVICES_HOST_FILE && policies[i].catalog!=NVM_SERVICES_HOST_TCP)
        status=NVM_SERVICES_HOST_INVALID;
    if(status==NVM_SERVICES_HOST_OK) {
        NvmServicesHostGrant *grant=calloc(1,sizeof *grant);
        if(!grant)status=NVM_SERVICES_HOST_MEMORY;
        else {
            grant->runtime_identity=&services_host_identity;grant->abi=NVM_SERVICES_HOST_ABI;
            grant->catalog=NVM_SERVICES_HOST_CATALOG;grant->live=true;grant->count=count;
            for(size_t i=0;i<count;i++)grant->policies[i]=policies[i];
            *out=grant;
        }
    }
    nvm_services_host_leave();return status;
}
NvmServicesHostStatus nvm_services_host_grant_revoke_instance(NvmServicesHostGrant *grant,size_t instance) {
    NvmServicesHostStatus status=nvm_services_host_enter_query();
    if(status!=NVM_SERVICES_HOST_OK)return status;
    status=services_host_check(grant,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG);
    if(status==NVM_SERVICES_HOST_OK || status==NVM_SERVICES_HOST_STATE) {
        if(instance>=grant->count)status=NVM_SERVICES_HOST_INVALID;
        else {grant->policies[instance].allowed=false;status=NVM_SERVICES_HOST_OK;}
    }
    nvm_services_host_leave();return status;
}
NvmServicesHostStatus nvm_services_host_grant_revoke(NvmServicesHostGrant *grant) {
    NvmServicesHostStatus status=nvm_services_host_enter_query();
    if(status!=NVM_SERVICES_HOST_OK)return status;
    status=services_host_check(grant,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG);
    if(status==NVM_SERVICES_HOST_OK || status==NVM_SERVICES_HOST_STATE){grant->live=false;status=NVM_SERVICES_HOST_OK;}
    nvm_services_host_leave();return status;
}
NvmServicesHostStatus nvm_services_host_grant_destroy(NvmServicesHostGrant **inout) {
    NvmServicesHostStatus status=nvm_services_host_enter_query();
    if(status!=NVM_SERVICES_HOST_OK)return status;
    if(!inout)status=NVM_SERVICES_HOST_INVALID;
    else if(*inout) {
        status=services_host_check(*inout,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG);
        if(status==NVM_SERVICES_HOST_OK || status==NVM_SERVICES_HOST_STATE){free(*inout);*inout=NULL;status=NVM_SERVICES_HOST_OK;}
    }
    nvm_services_host_leave();return status;
}
NvmServicesHostStatus nvm_services_host_authorize(const NvmServicesHostGrant *grant,const NvmServicesIndirectHostedPlan *plan) {
    NvmServicesHostStatus status=services_host_check(grant,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG);
    if(status!=NVM_SERVICES_HOST_OK)return status;
    if(!plan)return NVM_SERVICES_HOST_INVALID;
    NvmServicesNominalLayout type;
    for(uint32_t i=0;i<grant->count;i++) {
        if(!nvm_services_indirect_hosted_type(plan,i*9,&type))return NVM_SERVICES_HOST_UNRESOLVED;
        NvmServicesHostCatalog catalog=nvm_services_indirect_hosted_type(plan,i*9+8,&type)?NVM_SERVICES_HOST_TCP:NVM_SERVICES_HOST_FILE;
        if(catalog!=grant->policies[i].catalog)return NVM_SERVICES_HOST_UNRESOLVED;
        if(!grant->policies[i].allowed)return NVM_SERVICES_HOST_STATE;
    }
    if(nvm_services_indirect_hosted_type(plan,(uint32_t)grant->count*9,&type))return NVM_SERVICES_HOST_UNRESOLVED;
    return NVM_SERVICES_HOST_OK;
}
