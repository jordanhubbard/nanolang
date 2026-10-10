#include "../nanoisa/file_indirect_public_internal.h"
#include "file_vm_indirect_engine.inc"

NvmFileIndirectExecutionReport nvm_file_execute_indirect_bytes(NvmFileHostGrant *grant,
    const uint8_t *bytes, size_t size, const NvmFileIndirectOptions *options,
    NvmFileScalar *out) {
    NvmFileHostStatus entered = nvm_file_host_enter(grant, NVM_FILE_HOST_ABI,
                                                  NVM_FILE_HOST_CATALOG);
    if (entered != NVM_FILE_HOST_OK)
        return nvm_file_indirect_public_refused(nvm_file_public_grant_status(entered),
                                             entered == NVM_FILE_HOST_BUSY ? NULL : options);
    NvmFileIndirectExecutionReport report;
    if (!nvm_file_indirect_public_options(options) || !out)
        report = nvm_file_indirect_public_refused(NVM_FILE_RUNTIME_INVALID, options);
    else {
        NvmFileIndirectOptions copied = *options;
        NvmFileRuntimeView view = {0};
        report = file_vm_indirect_execute_serialized(bytes, size, &copied, &view);
        report.runtime = nvm_file_public_publish(report.runtime, &view, out);
    }
    nvm_file_host_leave();
    return report;
}
