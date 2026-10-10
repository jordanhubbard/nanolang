#include "service_policy.h"
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
        if(catalog!=1 && catalog!=2)return false;
        bool allowed=catalog==1?files:tcp;
        policy.instances[i]=(NvmServicesHostPolicy){(NvmServicesHostCatalog)catalog,allowed};
        policy.requires_file|=catalog==1;policy.requires_tcp|=catalog==2;policy.allowed&=allowed;
    }
    *out=policy;return true;
}
