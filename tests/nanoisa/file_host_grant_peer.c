/* I model a second C99 adapter TU, not a generated or executing File program. */
#include "file_host_grant_internal.h"
NvmFileHostStatus file_host_peer_enter(NvmFileHostGrant *grant) {
    NvmFileHostStatus status = nvm_file_host_enter(grant, NVM_FILE_HOST_ABI,
                                                 NVM_FILE_HOST_CATALOG);
    if (status == NVM_FILE_HOST_OK) nvm_file_host_leave();
    return status;
}
