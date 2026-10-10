#include "websocket_host_grant_internal.h"
#include "file_host_grant_internal.h"
#include <stdlib.h>
#include <string.h>

/* I share one process-local public gate with File/TCP/mixed execution. */
static const unsigned websocket_host_identity=NVM_WEBSOCKET_HOST_ABI;
struct NvmWebSocketHostGrant {
    const void *runtime_identity;
    unsigned abi,catalog;
    bool live;
    NlWsTransportPolicy policy;
    char helper[4096];
};
NvmWebSocketHostStatus nvm_websocket_host_enter_query(void) {
    return nvm_file_host_enter_query()==NVM_FILE_HOST_OK?NVM_WEBSOCKET_HOST_OK:NVM_WEBSOCKET_HOST_BUSY;
}
void nvm_websocket_host_leave(void) { nvm_file_host_leave(); }
static NvmWebSocketHostStatus websocket_host_check(const NvmWebSocketHostGrant *g,unsigned abi,unsigned catalog) {
    if(!g)return NVM_WEBSOCKET_HOST_INVALID;
    if(g->runtime_identity!=&websocket_host_identity || abi!=NVM_WEBSOCKET_HOST_ABI ||
       catalog!=NVM_WEBSOCKET_HOST_CATALOG || g->abi!=abi || g->catalog!=catalog)return NVM_WEBSOCKET_HOST_UNRESOLVED;
    return g->live?NVM_WEBSOCKET_HOST_OK:NVM_WEBSOCKET_HOST_STATE;
}
NvmWebSocketHostStatus nvm_websocket_host_enter(const NvmWebSocketHostGrant *g,unsigned abi,unsigned catalog) {
    NvmWebSocketHostStatus status=nvm_websocket_host_enter_query();if(status!=NVM_WEBSOCKET_HOST_OK)return status;
    status=websocket_host_check(g,abi,catalog);if(status!=NVM_WEBSOCKET_HOST_OK)nvm_websocket_host_leave();return status;
}
NvmWebSocketHostStatus nvm_websocket_host_policy(const NvmWebSocketHostGrant *g,NlWsTransportPolicy *out) {
    NvmWebSocketHostStatus status=websocket_host_check(g,NVM_WEBSOCKET_HOST_ABI,NVM_WEBSOCKET_HOST_CATALOG);
    if(status!=NVM_WEBSOCKET_HOST_OK)return status;
    if(!out)return NVM_WEBSOCKET_HOST_INVALID;
    *out=g->policy;return NVM_WEBSOCKET_HOST_OK;
}
NvmWebSocketHostStatus nvm_websocket_host_grant_create(const NvmWebSocketHostPolicy *policy,NvmWebSocketHostGrant **out) {
    NvmWebSocketHostStatus status=nvm_websocket_host_enter_query();if(status!=NVM_WEBSOCKET_HOST_OK)return status;
    size_t length=0;
    if(!out || !policy || policy->revision!=NVM_WEBSOCKET_HOST_POLICY_REVISION || policy->max_timeout_ms>60000 ||
       (policy->allow_lookup && !policy->resolver_helper)) {status=NVM_WEBSOCKET_HOST_INVALID;goto done;}
    if(policy->resolver_helper) {
        while(length<4096 && policy->resolver_helper[length])length++;
        if(!length || length==4096 || policy->resolver_helper[0]!='/'){status=NVM_WEBSOCKET_HOST_INVALID;goto done;}
    }
    NvmWebSocketHostGrant *g=calloc(1,sizeof *g);
    if(!g){status=NVM_WEBSOCKET_HOST_MEMORY;goto done;}
    g->runtime_identity=&websocket_host_identity;g->abi=NVM_WEBSOCKET_HOST_ABI;g->catalog=NVM_WEBSOCKET_HOST_CATALOG;g->live=true;
    g->policy=(NlWsTransportPolicy){policy->allow_connections,policy->allow_lookup,NULL,policy->max_timeout_ms};
    if(policy->resolver_helper){memcpy(g->helper,policy->resolver_helper,length+1);g->policy.resolver_helper=g->helper;}
    *out=g;
done:
    nvm_websocket_host_leave();return status;
}
NvmWebSocketHostStatus nvm_websocket_host_grant_revoke(NvmWebSocketHostGrant *g) {
    NvmWebSocketHostStatus status=nvm_websocket_host_enter_query();if(status!=NVM_WEBSOCKET_HOST_OK)return status;
    status=websocket_host_check(g,NVM_WEBSOCKET_HOST_ABI,NVM_WEBSOCKET_HOST_CATALOG);
    if(status==NVM_WEBSOCKET_HOST_OK || status==NVM_WEBSOCKET_HOST_STATE){g->live=false;status=NVM_WEBSOCKET_HOST_OK;}
    nvm_websocket_host_leave();return status;
}
NvmWebSocketHostStatus nvm_websocket_host_grant_destroy(NvmWebSocketHostGrant **inout) {
    NvmWebSocketHostStatus status=nvm_websocket_host_enter_query();if(status!=NVM_WEBSOCKET_HOST_OK)return status;
    if(!inout)status=NVM_WEBSOCKET_HOST_INVALID;
    else if(*inout){status=websocket_host_check(*inout,NVM_WEBSOCKET_HOST_ABI,NVM_WEBSOCKET_HOST_CATALOG);
        if(status==NVM_WEBSOCKET_HOST_OK || status==NVM_WEBSOCKET_HOST_STATE){free(*inout);*inout=NULL;status=NVM_WEBSOCKET_HOST_OK;}}
    nvm_websocket_host_leave();return status;
}
