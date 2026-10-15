#ifndef NANOISA_WEBSOCKET_HOST_GRANT_INTERNAL_H
#define NANOISA_WEBSOCKET_HOST_GRANT_INTERNAL_H
#include "websocket_host_grant.h"
#include "../nsi_websocket_transport.h"

/* Trusted adapter operations only. OK retains the single owning runtime gate;
 * every other status leaves no acquisition. Only that successful calling thread
 * may leave, exactly once, after complete cleanup. No callbacks, owner transfer,
 * or cross-runtime grant exchange. These are not public execution entrypoints.
 * Direct private query/core callers still require external serialization. */
#ifdef __cplusplus
extern "C" {
#endif
NvmWebSocketHostStatus nvm_websocket_host_enter_query(void);
NvmWebSocketHostStatus nvm_websocket_host_enter(const NvmWebSocketHostGrant *grant,
                                    unsigned abi, unsigned catalog);
/* I copy policy only while the caller owns the gate; path lifetime is the grant. */
NvmWebSocketHostStatus nvm_websocket_host_policy(const NvmWebSocketHostGrant *,NlWsTransportPolicy *);
void nvm_websocket_host_leave(void);
#ifdef __cplusplus
}
#endif
#endif
