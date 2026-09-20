#include "../nanoisa/file_cyclic_public_internal.h"
#include "file_vm_cyclic_engine.inc"

NvmFileCyclicExecutionReport nvm_file_execute_cyclic_bytes(NvmFileHostGrant *grant,
    const uint8_t *bytes, size_t size, const NvmFileCyclicOptions *options,
    NvmFileScalar *out) {
    NvmFileHostStatus entered = nvm_file_host_enter(grant, NVM_FILE_HOST_ABI,
                                                  NVM_FILE_HOST_CATALOG);
    if (entered != NVM_FILE_HOST_OK)
        return nvm_file_cyclic_public_refused(nvm_file_public_grant_status(entered),
                                             entered == NVM_FILE_HOST_BUSY ? NULL : options);
    NvmFileCyclicExecutionReport report;
    if (!nvm_file_cyclic_public_options(options) || !out)
        report = nvm_file_cyclic_public_refused(NVM_FILE_RUNTIME_INVALID, options);
    else {
        NvmFileCyclicOptions copied = *options;
        NvmFileRuntimeView view = {0};
        report = file_vm_cyclic_execute_serialized(bytes, size, &copied, &view);
        report.runtime = nvm_file_public_publish(report.runtime, &view, out);
    }
    nvm_file_host_leave();
    return report;
}
