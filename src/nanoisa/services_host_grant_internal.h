#ifndef NANOISA_SERVICES_HOST_GRANT_INTERNAL_H
#define NANOISA_SERVICES_HOST_GRANT_INTERNAL_H
#include "services_host_grant.h"
#include "services_indirect_hosted.h"
#include "../nsi_websocket_transport.h"

/* Trusted adapter operations only. OK retains the single owning runtime gate;
 * every other status leaves no acquisition. Only that successful calling thread
 * may leave, exactly once, after complete cleanup. No callbacks, owner transfer,
 * or cross-runtime grant exchange. These are not public execution entrypoints.
 * Direct private query/core callers still require external serialization. */
#ifdef __cplusplus
extern "C" {
#endif
NvmServicesHostStatus nvm_services_host_enter_query(void);
NvmServicesHostStatus nvm_services_host_enter(const NvmServicesHostGrant *grant,
                                    unsigned abi, unsigned catalog);
/* I inspect only the retained checked plan while my successful caller holds
 * the shared gate. This query grants no acquisition by itself. */
NvmServicesHostStatus nvm_services_host_authorize(const NvmServicesHostGrant *,const NvmServicesIndirectHostedPlan *);
/* I export only the exact WebSocket instance while the caller owns the gate.
 * The resolver path belongs to the grant. Denied/revoked instance permission
 * stays false; whole-grant revocation refuses this query and preserves *out. */
NvmServicesHostStatus nvm_services_host_websocket_policy(const NvmServicesHostGrant *,size_t,NlWsTransportPolicy *out);
void nvm_services_host_leave(void);
#ifdef __cplusplus
}
#endif
#endif
