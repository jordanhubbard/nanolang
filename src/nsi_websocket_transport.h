#ifndef NL_NSI_WEBSOCKET_TRANSPORT_H
#define NL_NSI_WEBSOCKET_TRANSPORT_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct NlWsTransport NlWsTransport;
typedef enum {
    NL_WS_TRANSPORT_OK, NL_WS_TRANSPORT_ARGUMENT, NL_WS_TRANSPORT_RIGHTS,
    NL_WS_TRANSPORT_MEMORY, NL_WS_TRANSPORT_LIMIT, NL_WS_TRANSPORT_TIMEOUT,
    NL_WS_TRANSPORT_IO, NL_WS_TRANSPORT_PROTOCOL, NL_WS_TRANSPORT_CRYPTO,
    NL_WS_TRANSPORT_CLOSED
} NlWsTransportStatus;
typedef struct {
    NlWsTransportStatus status;
    int host_errno,resolver_error,supervisor_status,close_code,cleanup_errno;
    bool cleanup_failed,closure_unknown,terminal;
    size_t bytes;
} NlWsTransportResult;
typedef struct {
    bool allow_network,allow_lookup;
    const char *resolver_helper;
    unsigned max_timeout_ms;
} NlWsTransportPolicy;
typedef struct { bool binary; unsigned char *bytes; size_t length; } NlWsMessage;

/* I expose a private serialized host transport, not a source handle. The caller
 * owns a successfully returned connection and must close it exactly once.
 * Caller objects/buffers are valid and disjoint; outputs cannot discard live
 * owners. All outputs remain unchanged on failure. Policy/path storage need
 * remain valid only during connect; the timeout ceiling is copied. */
NlWsTransportResult nl_ws_transport_connect(const void *url,size_t length,
    const NlWsTransportPolicy *,unsigned timeout_ms,NlWsTransport **out);
NlWsTransportResult nl_ws_transport_send(NlWsTransport *,bool binary,
    const void *bytes,size_t length,unsigned timeout_ms);
/* I publish a fresh independent allocation, including for empty messages.
 * Allocation refusal retains the pending decoder event for a later retry. */
NlWsTransportResult nl_ws_transport_receive(NlWsTransport *,unsigned timeout_ms,NlWsMessage *out);
void nl_ws_message_free(NlWsMessage *);
/* I free the connection on every result, including invalid deadlines and failed
 * close handshakes. No subsequent operation may use its pointer. */
NlWsTransportResult nl_ws_transport_close(NlWsTransport *,unsigned timeout_ms);
/* I stop I/O without destroying the caller's ownership obligation. */
NlWsTransportResult nl_ws_transport_abort(NlWsTransport *);
bool nl_ws_transport_connected(const NlWsTransport *);
const char *nl_ws_transport_error(const NlWsTransport *);
#endif
