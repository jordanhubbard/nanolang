#ifndef NL_SERVICE_POLICY_H
#define NL_SERVICE_POLICY_H
#include "../nanoisa/services_host_grant.h"
#include <stdint.h>
/* I describe a serialized module's requested catalogs, not an execution proof.
 * Public execution/emission still validates its complete checked plan. */
typedef struct {
    /* I use profile 4 for WebSocket; profile 3 remains mixed File/TCP. */
    unsigned profile;
    size_t count;
    NvmServicesHostPolicy instances[NVM_SERVICES_HOST_INSTANCES];
    bool allowed, requires_file, requires_tcp, requires_websocket;
} NlServicePolicy;
/* Failure preserves out. Host opt-ins independently permit each catalog. */
bool nl_service_policy_read(const uint8_t *,size_t,bool,bool,bool,NlServicePolicy *);
#endif
