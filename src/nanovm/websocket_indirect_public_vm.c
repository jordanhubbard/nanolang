#include "../nanoisa/websocket_indirect_public_internal.h"
#include "../nanoisa/websocket_dispatch_config.h"
#include "service_vm_indirect_engine.inc"
NvmWebSocketIndirectExecutionReport nvm_websocket_execute_indirect_bytes(NvmWebSocketHostGrant *grant,
    const uint8_t *bytes,size_t size,const NvmWebSocketIndirectOptions *options,NvmWebSocketScalar *out) {
    NvmWebSocketHostStatus entered=nvm_websocket_host_enter(grant,NVM_WEBSOCKET_HOST_ABI,NVM_WEBSOCKET_HOST_CATALOG);
    if(entered!=NVM_WEBSOCKET_HOST_OK)return nvm_websocket_indirect_public_refused(
        nvm_websocket_public_grant_status(entered),entered==NVM_WEBSOCKET_HOST_BUSY?NULL:options);
    NvmWebSocketIndirectExecutionReport report;
    if(!nvm_websocket_indirect_public_options(options) || !out)
        report=nvm_websocket_indirect_public_refused(NVM_WEBSOCKET_RUNTIME_INVALID,options);
    else {
        NvmWebSocketIndirectOptions copied=*options;NvmWebSocketRuntimeView view={0};NlWsTransportPolicy policy;
        NvmWebSocketHostStatus status=nvm_websocket_host_policy(grant,&policy);
        if(status!=NVM_WEBSOCKET_HOST_OK)report=nvm_websocket_indirect_public_refused(nvm_websocket_public_grant_status(status),&copied);
        else {report=file_vm_indirect_execute_serialized(bytes,size,&copied,&view,&policy);
            report.runtime=nvm_websocket_public_publish(report.runtime,&view,out);}
    }
    nvm_websocket_host_leave();return report;
}
