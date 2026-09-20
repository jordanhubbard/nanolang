#ifndef NANOISA_FILE_HOST_GRANT_INTERNAL_H
#define NANOISA_FILE_HOST_GRANT_INTERNAL_H
#include "file_host_grant.h"

/* Trusted adapter operations only. OK retains the single owning runtime gate;
 * every other status leaves no acquisition. Only that successful calling thread
 * may leave, exactly once, after complete cleanup. No callbacks, owner transfer,
 * or cross-runtime grant exchange. These are not public execution entrypoints.
 * Direct private query/core callers still require external serialization. */
#ifdef __cplusplus
extern "C" {
#endif
NvmFileHostStatus nvm_file_host_enter_query(void);
NvmFileHostStatus nvm_file_host_enter(const NvmFileHostGrant *grant,
                                    unsigned abi, unsigned catalog);
void nvm_file_host_leave(void);
#ifdef __cplusplus
}
#endif
#endif
