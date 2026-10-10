#ifndef NANOISA_WEBSOCKET_HOST_GRANT_H
#define NANOISA_WEBSOCKET_HOST_GRANT_H

/* I expose policy ownership, not WebSocket execution or a transferable certificate.
 * Pointer lifetimes and disjoint caller output storage follow ordinary C rules.
 * No atomic types or internal grant layout cross this C99-compatible header. */
#define NVM_WEBSOCKET_HOST_ABI 1u
#define NVM_WEBSOCKET_HOST_CATALOG 3u
#include <stdbool.h>
#include <stdint.h>
#define NVM_WEBSOCKET_HOST_POLICY_REVISION 1u
typedef struct {
    uint32_t revision;
    bool allow_connections,allow_lookup;
    uint32_t max_timeout_ms;
    const char *resolver_helper;
} NvmWebSocketHostPolicy;

typedef struct NvmWebSocketHostGrant NvmWebSocketHostGrant;
typedef enum {
    NVM_WEBSOCKET_HOST_OK = 0,
    NVM_WEBSOCKET_HOST_INVALID = 1,
    NVM_WEBSOCKET_HOST_MEMORY = 2,
    NVM_WEBSOCKET_HOST_STATE = 3,
    NVM_WEBSOCKET_HOST_UNRESOLVED = 4,
    NVM_WEBSOCKET_HOST_BUSY = 5
} NvmWebSocketHostStatus;

#ifdef __cplusplus
extern "C" {
#endif
/* All operations refuse BUSY before inspecting arguments while the shared gate
 * is held. Failure preserves caller storage. Creation allocates only policy;
 * it opens no websocket and success transfers one grant to *out. */
/* I copy separate connection/lookup authority, the deadline ceiling (0..60000)
 * and an optional absolute resolver path. Lookup authority requires that path.
 * Creation grants no source admission and performs no network or DNS work. */
NvmWebSocketHostStatus nvm_websocket_host_grant_create(const NvmWebSocketHostPolicy *,NvmWebSocketHostGrant **out);
/* Repeated revoke is OK. A revoked object remains allocated until destroy. */
NvmWebSocketHostStatus nvm_websocket_host_grant_revoke(NvmWebSocketHostGrant *grant);
/* NULL inout is INVALID; *inout == NULL is OK. Success frees then clears *inout.
 * No other call may use a freed pointer, including one copied before destroy. */
NvmWebSocketHostStatus nvm_websocket_host_grant_destroy(NvmWebSocketHostGrant **inout);
#ifdef __cplusplus
}
#endif
#endif
