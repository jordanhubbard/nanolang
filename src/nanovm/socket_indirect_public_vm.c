#include "../nanoisa/socket_indirect_public_internal.h"
#include "socket_vm_indirect_engine.inc"

NvmSocketIndirectExecutionReport nvm_socket_execute_indirect_bytes(NvmSocketHostGrant *grant,
    const uint8_t *bytes, size_t size, const NvmSocketIndirectOptions *options,
    NvmSocketScalar *out) {
    NvmSocketHostStatus entered = nvm_socket_host_enter(grant, NVM_SOCKET_HOST_ABI,
                                                  NVM_SOCKET_HOST_CATALOG);
    if (entered != NVM_SOCKET_HOST_OK)
        return nvm_socket_indirect_public_refused(nvm_socket_public_grant_status(entered),
                                             entered == NVM_SOCKET_HOST_BUSY ? NULL : options);
    NvmSocketIndirectExecutionReport report;
    if (!nvm_socket_indirect_public_options(options) || !out)
        report = nvm_socket_indirect_public_refused(NVM_SOCKET_RUNTIME_INVALID, options);
    else {
        NvmSocketIndirectOptions copied = *options;
        NvmSocketRuntimeView view = {0};
        report = file_vm_indirect_execute_serialized(bytes, size, &copied, &view);
        report.runtime = nvm_socket_public_publish(report.runtime, &view, out);
    }
    nvm_socket_host_leave();
    return report;
}
