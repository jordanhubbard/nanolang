#include "service_policy.h"
#include <string.h>
#include "../nanoisa/nvm_v2_sections.h"
#include "../nanoisa/service_file_nominal.h"
#include "../nanoisa/service_socket_nominal.h"
#include "../nanoisa/service_multi_nominal.h"
#include "../nanoisa/websocket_codec.h"
#include "../nanoisa/service_websocket_nominal.h"

bool nl_service_policy_read(const uint8_t *bytes,size_t size,bool files,bool tcp,bool websocket,NlServicePolicy *out) {
    if(!bytes || !size || !out)return false;
    NvmV2Module wire={0};
    if(nvm_v2_module_deserialize(bytes,size,&wire)!=NVM_V2_OK) {
        nvm_v2_module_free(&wire);
        if(nvm_websocket_deserialize(bytes,size,&wire)!=NVM_V2_OK)return false;
    }
    NlServicePolicy policy={0};
    NvmMultiNominalBindings mixed={0};
    if(wire.service_size==NVM_WEBSOCKET_NOMINAL_BYTES) {
        NvmWebSocketNominalBindings binding;
        if(nvm_websocket_nominal_decode(wire.service_data,wire.service_size,&binding)==NVM_SERVICE_OK)policy.profile=4;
    } else if(wire.service_size==NVM_FILE_NOMINAL_BYTES) {
        NvmFileNominalBindings binding;
        if(nvm_file_nominal_decode(wire.service_data,wire.service_size,&binding)==NVM_SERVICE_OK)policy.profile=1;
    } else if(wire.service_size==NVM_SOCKET_NOMINAL_BYTES) {
        NvmSocketNominalBindings binding;
        if(nvm_socket_nominal_decode(wire.service_data,wire.service_size,&binding)==NVM_SERVICE_OK)policy.profile=2;
    } else if(nvm_multi_nominal_decode(wire.service_data,wire.service_size,&mixed)==NVM_SERVICE_OK)policy.profile=3;
    nvm_v2_module_free(&wire);
    if(!policy.profile)return false;
    policy.count=policy.profile==3?mixed.count:1;
    if(policy.profile==4) {
        policy.requires_websocket=true;policy.allowed=websocket;
        *out=policy;return true;
    }
    policy.allowed=true;
    for(size_t i=0;i<policy.count;i++) {
        unsigned catalog=policy.profile==3?mixed.instances[i].catalog:policy.profile;
        if(catalog!=1 && catalog!=2 && catalog!=3)return false;
        bool allowed=catalog==1?files:catalog==2?tcp:websocket;
        policy.instances[i]=(NvmServicesHostPolicy){(NvmServicesHostCatalog)catalog,allowed};
        policy.requires_websocket|=catalog==3;policy.requires_file|=catalog==1;policy.requires_tcp|=catalog==2;policy.allowed&=allowed;
    }
    *out=policy;return true;
}

NvmServicesHostStatus nl_service_policy_grant(const NlServicePolicy *policy,
    const NvmWebSocketHostPolicy *websocket,NvmServicesHostGrant **out) {
    if(!policy || policy->profile!=3 || !policy->count || policy->count>NVM_SERVICES_HOST_INSTANCES)
        return NVM_SERVICES_HOST_INVALID;
    if(websocket && (websocket->revision!=NVM_WEBSOCKET_HOST_POLICY_REVISION ||
       websocket->max_timeout_ms>60000 || websocket->allow_lookup!=(websocket->resolver_helper!=NULL) ||
       (websocket->resolver_helper && (websocket->resolver_helper[0]!='/' || strlen(websocket->resolver_helper)>=4096))))
        return NVM_SERVICES_HOST_INVALID;
    NvmServicesHostConfig configs[NVM_SERVICES_HOST_INSTANCES]={0};
    for(size_t i=0;i<policy->count;i++) {
        configs[i]=(NvmServicesHostConfig){.revision=NVM_SERVICES_HOST_POLICY_REVISION,
            .catalog=policy->instances[i].catalog,.allowed=policy->instances[i].allowed};
        if(configs[i].catalog==NVM_SERVICES_HOST_WEBSOCKET) {
            configs[i].allowed=configs[i].allowed && websocket && websocket->allow_connections;
            if(websocket) {
                configs[i].allow_lookup=websocket->allow_lookup;
                configs[i].max_timeout_ms=websocket->max_timeout_ms;
                configs[i].resolver_helper=websocket->resolver_helper;
            }
        }
    }
    return nvm_services_host_grant_create_config(configs,policy->count,out);
}
