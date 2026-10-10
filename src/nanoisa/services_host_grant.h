#ifndef NANOISA_SERVICES_HOST_GRANT_H
#define NANOISA_SERVICES_HOST_GRANT_H

/* I expose policy ownership, not resource execution or a transferable certificate.
 * Pointer lifetimes and disjoint caller output storage follow ordinary C rules.
 * No atomic types or internal grant layout cross this C99-compatible header. */
#define NVM_SERVICES_HOST_ABI 1u
#define NVM_SERVICES_HOST_CATALOG 3u
#define NVM_SERVICES_HOST_INSTANCES 64u
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
/* I bind each policy to its zero-based position in the checked instance table. */
typedef enum { NVM_SERVICES_HOST_FILE=1, NVM_SERVICES_HOST_TCP=2, NVM_SERVICES_HOST_WEBSOCKET=3 } NvmServicesHostCatalog;
typedef struct { NvmServicesHostCatalog catalog; bool allowed; } NvmServicesHostPolicy;
#define NVM_SERVICES_HOST_POLICY_REVISION 1u
typedef struct {
    uint32_t revision;
    NvmServicesHostCatalog catalog;
    bool allowed, allow_lookup;
    uint32_t max_timeout_ms;
    const char *resolver_helper;
} NvmServicesHostConfig;

typedef struct NvmServicesHostGrant NvmServicesHostGrant;
typedef enum {
    NVM_SERVICES_HOST_OK = 0,
    NVM_SERVICES_HOST_INVALID = 1,
    NVM_SERVICES_HOST_MEMORY = 2,
    NVM_SERVICES_HOST_STATE = 3,
    NVM_SERVICES_HOST_UNRESOLVED = 4,
    NVM_SERVICES_HOST_BUSY = 5
} NvmServicesHostStatus;

#ifdef __cplusplus
extern "C" {
#endif
/* All operations refuse BUSY before inspecting arguments while the shared gate
 * is held. Failure preserves caller storage. Creation allocates only policy;
 * it acquires no resource and success transfers one grant to *out. */
/* I copy 1..64 policies. *out must initially be NULL. File permits temporary
 * files; TCP permits outbound IPv4/IPv6 connections. A false policy denies that
 * instance. Every declared instance must be allowed before execution begins. */
NvmServicesHostStatus nvm_services_host_grant_create(const NvmServicesHostPolicy *,size_t,NvmServicesHostGrant **out);
/* I copy explicit WebSocket connection/lookup authority, a 0..60000 millisecond
 * deadline ceiling and an optional absolute resolver path of at most 4095 bytes.
 * Lookup requires a resolver path. File/TCP entries require zero lookup,
 * deadline and path fields. My older constructor remains File/TCP-only.
 * Each declared instance must be allowed before execution; lookup can remain
 * denied independently. Creation performs no resource or resolver operations. */
NvmServicesHostStatus nvm_services_host_grant_create_config(const NvmServicesHostConfig *,size_t,NvmServicesHostGrant **out);
NvmServicesHostStatus nvm_services_host_grant_revoke_instance(NvmServicesHostGrant *,size_t);
/* I grant neither listeners nor arbitrary foreign calls. Catalog positions
 * must match the exact checked table; another table shape confers no authority. */
/* Repeated revoke is OK. A revoked object remains allocated until destroy. */
NvmServicesHostStatus nvm_services_host_grant_revoke(NvmServicesHostGrant *grant);
/* NULL inout is INVALID; *inout == NULL is OK. Success frees then clears *inout.
 * No other call may use a freed pointer, including one copied before destroy. */
NvmServicesHostStatus nvm_services_host_grant_destroy(NvmServicesHostGrant **inout);
#ifdef __cplusplus
}
#endif
#endif
