#include "../nanoisa/file_public_internal.h"
#include "file_vm_engine.inc"

NvmFileRuntimeReport nvm_file_execute_bytes(NvmFileHostGrant *grant,
    const uint8_t *bytes, size_t size, NvmFileScalar *out) {
    NvmFileHostStatus entered = nvm_file_host_enter(grant, NVM_FILE_HOST_ABI,
                                                  NVM_FILE_HOST_CATALOG);
    if (entered != NVM_FILE_HOST_OK)
        return nvm_file_public_refused(nvm_file_public_grant_status(entered));
    NvmFileRuntimeView view = {0};
    NvmFileRuntimeReport report;
    if (!out) report = nvm_file_public_refused(NVM_FILE_RUNTIME_INVALID);
    else {
        report = file_vm_execute_serialized(bytes, size, &view);
        report = nvm_file_public_publish(report, &view, out);
    }
    nvm_file_host_leave();
    return report;
}
