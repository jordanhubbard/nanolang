#include "socket_vm_indirect_private.h"
#ifdef NVM_SOCKET_INDIRECT_VM_PRIVATE
#include "../nanoisa/socket_dispatch_config.h"
#include "service_vm_indirect_engine.inc"
NvmSocketIndirectExecutionReport nvm_socket_vm_indirect_execute(const uint8_t *bytes,size_t size,
    const NvmSocketIndirectOptions *options,NvmSocketRuntimeView *out) {
    return file_vm_indirect_execute_serialized(bytes,size,options,out);
}
#endif
