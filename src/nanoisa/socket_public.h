#ifndef NANOISA_SOCKET_PUBLIC_H
#define NANOISA_SOCKET_PUBLIC_H
#include "socket_host_grant.h"
#include "socket_runtime.h"
/* I publish a scalar only after terminal cleanup. Input/output storage remains
 * valid and disjoint for the complete call; no connection or context escapes. */
typedef struct { uint8_t tag; int64_t value; } NvmSocketScalar;
#endif
