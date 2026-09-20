#include "file_host_grant_internal.h"
#include <stdbool.h>
#include <stdatomic.h>
#include <stdlib.h>

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
#error "I require C11 for my owning File host gate."
#endif

static atomic_flag file_host_gate = ATOMIC_FLAG_INIT;
static const unsigned file_host_identity = NVM_FILE_HOST_ABI;
#define FILE_HOST_TEMPORARY_POLICY 1u

struct NvmFileHostGrant {
    const void *runtime_identity;
    unsigned abi, catalog, policy;
    bool live;
};

NvmFileHostStatus nvm_file_host_enter_query(void) {
    return atomic_flag_test_and_set_explicit(&file_host_gate, memory_order_acquire)
        ? NVM_FILE_HOST_BUSY : NVM_FILE_HOST_OK;
}

void nvm_file_host_leave(void) {
    atomic_flag_clear_explicit(&file_host_gate, memory_order_release);
}

/* The caller already owns the gate. I never inspect an unacquired grant. */
static NvmFileHostStatus file_host_check(const NvmFileHostGrant *grant,
                                       unsigned abi, unsigned catalog) {
    if (!grant) return NVM_FILE_HOST_INVALID;
    if (grant->runtime_identity != &file_host_identity ||
        abi != NVM_FILE_HOST_ABI || catalog != NVM_FILE_HOST_CATALOG ||
        grant->abi != abi || grant->catalog != catalog ||
        grant->policy != FILE_HOST_TEMPORARY_POLICY)
        return NVM_FILE_HOST_UNRESOLVED;
    if (!grant->live) return NVM_FILE_HOST_STATE;
    return NVM_FILE_HOST_OK;
}

NvmFileHostStatus nvm_file_host_enter(const NvmFileHostGrant *grant,
                                    unsigned abi, unsigned catalog) {
    NvmFileHostStatus status = nvm_file_host_enter_query();
    if (status != NVM_FILE_HOST_OK) return status;
    status = file_host_check(grant, abi, catalog);
    if (status != NVM_FILE_HOST_OK) nvm_file_host_leave();
    return status;
}

NvmFileHostStatus nvm_file_host_grant_create_temporary_files(NvmFileHostGrant **out) {
    NvmFileHostGrant *grant;
    NvmFileHostStatus status = nvm_file_host_enter_query();
    if (status != NVM_FILE_HOST_OK) return status;
    if (!out) {
        nvm_file_host_leave();
        return NVM_FILE_HOST_INVALID;
    }
    grant = malloc(sizeof(*grant));
    if (!grant) {
        nvm_file_host_leave();
        return NVM_FILE_HOST_MEMORY;
    }
    grant->runtime_identity = &file_host_identity;
    grant->abi = NVM_FILE_HOST_ABI;
    grant->catalog = NVM_FILE_HOST_CATALOG;
    grant->policy = FILE_HOST_TEMPORARY_POLICY;
    grant->live = true;
    *out = grant;
    nvm_file_host_leave();
    return NVM_FILE_HOST_OK;
}

NvmFileHostStatus nvm_file_host_grant_revoke(NvmFileHostGrant *grant) {
    NvmFileHostStatus status = nvm_file_host_enter_query();
    if (status != NVM_FILE_HOST_OK) return status;
    status = file_host_check(grant, NVM_FILE_HOST_ABI, NVM_FILE_HOST_CATALOG);
    if (status == NVM_FILE_HOST_OK || status == NVM_FILE_HOST_STATE) {
        grant->live = false;
        status = NVM_FILE_HOST_OK;
    }
    nvm_file_host_leave();
    return status;
}

NvmFileHostStatus nvm_file_host_grant_destroy(NvmFileHostGrant **inout) {
    NvmFileHostStatus status = nvm_file_host_enter_query();
    if (status != NVM_FILE_HOST_OK) return status;
    if (!inout) status = NVM_FILE_HOST_INVALID;
    else if (*inout) {
        status = file_host_check(*inout, NVM_FILE_HOST_ABI, NVM_FILE_HOST_CATALOG);
        if (status == NVM_FILE_HOST_OK || status == NVM_FILE_HOST_STATE) {
            free(*inout);
            *inout = NULL;
            status = NVM_FILE_HOST_OK;
        }
    }
    nvm_file_host_leave();
    return status;
}
