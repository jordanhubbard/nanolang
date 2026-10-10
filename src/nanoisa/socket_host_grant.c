#include "socket_host_grant_internal.h"
#include "file_host_grant_internal.h"
#include <stdbool.h>
#include <stdlib.h>

/* I use the existing process-local public gate, including grant mutation and
 * nonexecuting queries. My identity and policy never alias File authority. */
static const unsigned socket_host_identity=NVM_SOCKET_HOST_ABI;
#define SOCKET_HOST_TCP_POLICY 1u
struct NvmSocketHostGrant {
    const void *runtime_identity;
    unsigned abi,catalog,policy;
    bool live;
};
NvmSocketHostStatus nvm_socket_host_enter_query(void) {
    NvmFileHostStatus status=nvm_file_host_enter_query();
    return status==NVM_FILE_HOST_OK?NVM_SOCKET_HOST_OK:NVM_SOCKET_HOST_BUSY;
}
void nvm_socket_host_leave(void) { nvm_file_host_leave(); }
static NvmSocketHostStatus socket_host_check(const NvmSocketHostGrant *grant,unsigned abi,unsigned catalog) {
    if(!grant)return NVM_SOCKET_HOST_INVALID;
    if(grant->runtime_identity!=&socket_host_identity || abi!=NVM_SOCKET_HOST_ABI ||
       catalog!=NVM_SOCKET_HOST_CATALOG || grant->abi!=abi || grant->catalog!=catalog ||
       grant->policy!=SOCKET_HOST_TCP_POLICY)return NVM_SOCKET_HOST_UNRESOLVED;
    return grant->live?NVM_SOCKET_HOST_OK:NVM_SOCKET_HOST_STATE;
}
NvmSocketHostStatus nvm_socket_host_enter(const NvmSocketHostGrant *grant,unsigned abi,unsigned catalog) {
    NvmSocketHostStatus status=nvm_socket_host_enter_query();
    if(status!=NVM_SOCKET_HOST_OK)return status;
    status=socket_host_check(grant,abi,catalog);
    if(status!=NVM_SOCKET_HOST_OK)nvm_socket_host_leave();
    return status;
}
NvmSocketHostStatus nvm_socket_host_grant_create_tcp_connections(NvmSocketHostGrant **out) {
    NvmSocketHostStatus status=nvm_socket_host_enter_query();
    if(status!=NVM_SOCKET_HOST_OK)return status;
    if(!out){nvm_socket_host_leave();return NVM_SOCKET_HOST_INVALID;}
    NvmSocketHostGrant *grant=malloc(sizeof *grant);
    if(!grant){nvm_socket_host_leave();return NVM_SOCKET_HOST_MEMORY;}
    *grant=(NvmSocketHostGrant){&socket_host_identity,NVM_SOCKET_HOST_ABI,NVM_SOCKET_HOST_CATALOG,SOCKET_HOST_TCP_POLICY,true};
    *out=grant;nvm_socket_host_leave();return NVM_SOCKET_HOST_OK;
}
NvmSocketHostStatus nvm_socket_host_grant_revoke(NvmSocketHostGrant *grant) {
    NvmSocketHostStatus status=nvm_socket_host_enter_query();
    if(status!=NVM_SOCKET_HOST_OK)return status;
    status=socket_host_check(grant,NVM_SOCKET_HOST_ABI,NVM_SOCKET_HOST_CATALOG);
    if(status==NVM_SOCKET_HOST_OK || status==NVM_SOCKET_HOST_STATE){grant->live=false;status=NVM_SOCKET_HOST_OK;}
    nvm_socket_host_leave();return status;
}
NvmSocketHostStatus nvm_socket_host_grant_destroy(NvmSocketHostGrant **inout) {
    NvmSocketHostStatus status=nvm_socket_host_enter_query();
    if(status!=NVM_SOCKET_HOST_OK)return status;
    if(!inout)status=NVM_SOCKET_HOST_INVALID;
    else if(*inout){
        status=socket_host_check(*inout,NVM_SOCKET_HOST_ABI,NVM_SOCKET_HOST_CATALOG);
        if(status==NVM_SOCKET_HOST_OK || status==NVM_SOCKET_HOST_STATE){free(*inout);*inout=NULL;status=NVM_SOCKET_HOST_OK;}
    }
    nvm_socket_host_leave();return status;
}
