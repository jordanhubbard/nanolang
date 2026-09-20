#ifndef NANOISA_FILE_HOST_GRANT_H
#define NANOISA_FILE_HOST_GRANT_H

/* I expose policy ownership, not File execution or a transferable certificate.
 * Pointer lifetimes and disjoint caller output storage follow ordinary C rules.
 * No atomic types or internal grant layout cross this C99-compatible header. */
#define NVM_FILE_HOST_ABI 1u
#define NVM_FILE_HOST_CATALOG 1u

typedef struct NvmFileHostGrant NvmFileHostGrant;
typedef enum {
    NVM_FILE_HOST_OK = 0,
    NVM_FILE_HOST_INVALID = 1,
    NVM_FILE_HOST_MEMORY = 2,
    NVM_FILE_HOST_STATE = 3,
    NVM_FILE_HOST_UNRESOLVED = 4,
    NVM_FILE_HOST_BUSY = 5
} NvmFileHostStatus;

#ifdef __cplusplus
extern "C" {
#endif
/* All operations refuse BUSY before inspecting arguments while the shared gate
 * is held. Failure preserves caller storage. Creation allocates only policy;
 * it opens no stream and success transfers one grant to *out. */
NvmFileHostStatus nvm_file_host_grant_create_temporary_files(NvmFileHostGrant **out);
/* Repeated revoke is OK. A revoked object remains allocated until destroy. */
NvmFileHostStatus nvm_file_host_grant_revoke(NvmFileHostGrant *grant);
/* NULL inout is INVALID; *inout == NULL is OK. Success frees then clears *inout.
 * No other call may use a freed pointer, including one copied before destroy. */
NvmFileHostStatus nvm_file_host_grant_destroy(NvmFileHostGrant **inout);
#ifdef __cplusplus
}
#endif
#endif
