#ifndef NANOISA_WEBSOCKET_PUBLIC_H
#define NANOISA_WEBSOCKET_PUBLIC_H
#include "websocket_host_grant.h"
#include "websocket_runtime.h"
/* I publish a scalar only after terminal cleanup. Input/output storage remains
 * valid and disjoint for the complete call; no connection or context escapes. */
typedef struct { uint8_t tag; int64_t value; } NvmWebSocketScalar;
#endif
