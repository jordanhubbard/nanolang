#ifndef NANOISA_SOCKET_HOST_GRANT_H
#define NANOISA_SOCKET_HOST_GRANT_H

/* I expose policy ownership, not TCP execution or a transferable certificate.
 * Pointer lifetimes and disjoint caller output storage follow ordinary C rules.
 * No atomic types or internal grant layout cross this C99-compatible header. */
#define NVM_SOCKET_HOST_ABI 1u
#define NVM_SOCKET_HOST_CATALOG 2u

typedef struct NvmSocketHostGrant NvmSocketHostGrant;
typedef enum {
    NVM_SOCKET_HOST_OK = 0,
    NVM_SOCKET_HOST_INVALID = 1,
    NVM_SOCKET_HOST_MEMORY = 2,
    NVM_SOCKET_HOST_STATE = 3,
    NVM_SOCKET_HOST_UNRESOLVED = 4,
    NVM_SOCKET_HOST_BUSY = 5
} NvmSocketHostStatus;

#ifdef __cplusplus
extern "C" {
#endif
/* All operations refuse BUSY before inspecting arguments while the shared gate
 * is held. Failure preserves caller storage. Creation allocates only policy;
 * it opens no socket and success transfers one grant to *out. */
NvmSocketHostStatus nvm_socket_host_grant_create_tcp_connections(NvmSocketHostGrant **out);
/* I grant outbound IPv4/IPv6 TCP acquisition for the checked net catalog.
 * I do not grant listeners, arbitrary foreign calls or source admission. */
/* Repeated revoke is OK. A revoked object remains allocated until destroy. */
NvmSocketHostStatus nvm_socket_host_grant_revoke(NvmSocketHostGrant *grant);
/* NULL inout is INVALID; *inout == NULL is OK. Success frees then clears *inout.
 * No other call may use a freed pointer, including one copied before destroy. */
NvmSocketHostStatus nvm_socket_host_grant_destroy(NvmSocketHostGrant **inout);
#ifdef __cplusplus
}
#endif
#endif
