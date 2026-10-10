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
    NvmServicesHostConfig policies[NVM_SERVICES_HOST_INSTANCES];
    char helpers[];
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
static NvmServicesHostStatus services_host_create_config(const NvmServicesHostConfig *policies,size_t count,NvmServicesHostGrant **out) {
    if(!policies || !count || count>NVM_SERVICES_HOST_INSTANCES || !out || *out)return NVM_SERVICES_HOST_INVALID;
    size_t lengths[NVM_SERVICES_HOST_INSTANCES]={0},extra=0;
    for(size_t i=0;i<count;i++) {
        const NvmServicesHostConfig *p=&policies[i];
        if(p->revision!=NVM_SERVICES_HOST_POLICY_REVISION)return NVM_SERVICES_HOST_INVALID;
        if(p->catalog==NVM_SERVICES_HOST_WEBSOCKET) {
            if(p->max_timeout_ms>60000 || (p->allow_lookup && !p->resolver_helper))return NVM_SERVICES_HOST_INVALID;
            if(p->resolver_helper) {
                size_t n=0;while(n<4096 && p->resolver_helper[n])n++;
                if(!n || n==4096 || p->resolver_helper[0]!='/')return NVM_SERVICES_HOST_INVALID;
                lengths[i]=n+1;extra+=n+1;
            }
        } else if((p->catalog!=NVM_SERVICES_HOST_FILE && p->catalog!=NVM_SERVICES_HOST_TCP) ||
                  p->allow_lookup || p->max_timeout_ms || p->resolver_helper)return NVM_SERVICES_HOST_INVALID;
    }
    /* My bounded count and path lengths cap extra at 64*4096 bytes. */
    NvmServicesHostGrant *grant=calloc(1,sizeof *grant+extra);
    if(!grant)return NVM_SERVICES_HOST_MEMORY;
    grant->runtime_identity=&services_host_identity;grant->abi=NVM_SERVICES_HOST_ABI;
    grant->catalog=NVM_SERVICES_HOST_CATALOG;grant->live=true;grant->count=count;
    size_t at=0;
    for(size_t i=0;i<count;i++) {
        grant->policies[i]=policies[i];
        if(lengths[i]) {
            memcpy(grant->helpers+at,policies[i].resolver_helper,lengths[i]);
            grant->policies[i].resolver_helper=grant->helpers+at;at+=lengths[i];
        }
    }
    *out=grant;return NVM_SERVICES_HOST_OK;
}
NvmServicesHostStatus nvm_services_host_grant_create_config(const NvmServicesHostConfig *policies,size_t count,NvmServicesHostGrant **out) {
    NvmServicesHostStatus status=nvm_services_host_enter_query();
    if(status!=NVM_SERVICES_HOST_OK)return status;
    status=services_host_create_config(policies,count,out);
    nvm_services_host_leave();return status;
}
NvmServicesHostStatus nvm_services_host_grant_create(const NvmServicesHostPolicy *policies,size_t count,NvmServicesHostGrant **out) {
    NvmServicesHostStatus status=nvm_services_host_enter_query();
    if(status!=NVM_SERVICES_HOST_OK)return status;
    if(!policies || !count || count>NVM_SERVICES_HOST_INSTANCES || !out || *out)status=NVM_SERVICES_HOST_INVALID;
    else {
        NvmServicesHostConfig configs[NVM_SERVICES_HOST_INSTANCES]={0};
        for(size_t i=0;i<count;i++) {
            if(policies[i].catalog!=NVM_SERVICES_HOST_FILE && policies[i].catalog!=NVM_SERVICES_HOST_TCP) {
                status=NVM_SERVICES_HOST_INVALID;break;
            }
            configs[i]=(NvmServicesHostConfig){.revision=NVM_SERVICES_HOST_POLICY_REVISION,
                .catalog=policies[i].catalog,.allowed=policies[i].allowed};
        }
        if(status==NVM_SERVICES_HOST_OK)status=services_host_create_config(configs,count,out);
    }
    nvm_services_host_leave();return status;
}
NvmServicesHostStatus nvm_services_host_websocket_policy(const NvmServicesHostGrant *grant,size_t instance,NlWsTransportPolicy *out) {
    NvmServicesHostStatus status=services_host_check(grant,NVM_SERVICES_HOST_ABI,NVM_SERVICES_HOST_CATALOG);
    if(status!=NVM_SERVICES_HOST_OK)return status;
    if(!out || instance>=grant->count)return NVM_SERVICES_HOST_INVALID;
    const NvmServicesHostConfig *p=&grant->policies[instance];
    if(p->catalog!=NVM_SERVICES_HOST_WEBSOCKET)return NVM_SERVICES_HOST_UNRESOLVED;
    *out=(NlWsTransportPolicy){p->allowed,p->allow_lookup,p->resolver_helper,p->max_timeout_ms};
    return NVM_SERVICES_HOST_OK;
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
        if(!nvm_services_indirect_hosted_type(plan,i*9,&type) ||
           !nvm_services_indirect_hosted_type(plan,i*9+6,&type))return NVM_SERVICES_HOST_UNRESOLVED;
        NvmServicesHostCatalog catalog=nvm_services_indirect_hosted_type(plan,i*9+8,&type)?NVM_SERVICES_HOST_TCP:
            nvm_services_indirect_hosted_type(plan,i*9+7,&type)?NVM_SERVICES_HOST_FILE:NVM_SERVICES_HOST_WEBSOCKET;
        if(catalog!=grant->policies[i].catalog)return NVM_SERVICES_HOST_UNRESOLVED;
        if(!grant->policies[i].allowed)return NVM_SERVICES_HOST_STATE;
    }
    if(nvm_services_indirect_hosted_type(plan,(uint32_t)grant->count*9,&type))return NVM_SERVICES_HOST_UNRESOLVED;
    return NVM_SERVICES_HOST_OK;
}
