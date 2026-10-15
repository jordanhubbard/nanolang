#include "websocket_vm_indirect_private.h"
#ifdef NVM_WEBSOCKET_INDIRECT_VM_PRIVATE
#include "../nanoisa/websocket_dispatch_config.h"
#include "service_vm_indirect_engine.inc"
NvmWebSocketIndirectExecutionReport nvm_websocket_vm_indirect_execute(const uint8_t *bytes,size_t size,
    const NvmWebSocketIndirectOptions *options,NvmWebSocketRuntimeView *out,const NlWsTransportPolicy *policy) {
    return file_vm_indirect_execute_serialized(bytes,size,options,out,policy);
}
#endif
