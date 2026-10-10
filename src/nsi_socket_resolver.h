#ifndef NL_NSI_SOCKET_RESOLVER_H
#define NL_NSI_SOCKET_RESOLVER_H
#include "nsi_socket.h"

typedef enum {
    NL_LOOKUP_COMPLETE, NL_LOOKUP_ARGUMENT, NL_LOOKUP_SYSTEM,
    NL_LOOKUP_TIMEOUT, NL_LOOKUP_PROTOCOL, NL_LOOKUP_CHILD
} NlSocketLookupStatus;
typedef struct {
    NlSocketResolveResult resolver;
    NlSocketLookupStatus supervision;
    int supervisor_errno;
} NlSocketLookupResult;

/* I select an explicit NANOLANG_RESOLVER, then NANOLANG_ROOT/bin/nano-resolver,
 * otherwise nano-resolver beside the running executable. Configured paths must
 * be absolute; an invalid override refuses without fallback. I never search
 * PATH or the working directory. The host must trust the selected path and its
 * environment. Failure leaves the output untouched. */
bool nl_socket_resolver_path(char *out,size_t capacity);

/* I execute an absolute, trusted helper path without a shell or PATH search.
 * The caller serializes process creation and service access and must not reap
 * this worker from a signal handler or another thread. This is deadline
 * supervision, not a sandbox: the helper inherits the caller's host authority.
 * Numeric addresses and denied lookup use no child. DNS requires a timeout of
 * 1..60000 milliseconds. I publish only a complete validated response from a
 * successfully exited child. Failure preserves *out. Killing and reaping a
 * timed-out worker still depends on operating-system process cleanup progress.
 * The helper path must remain stable and all input/output storage disjoint. */
NlSocketLookupResult nl_socket_resolve_tcp_supervised(NlSocketService *,
    const char *host, size_t length, uint16_t port, bool allow_lookup,
    const char *helper, unsigned timeout_ms, NlSocketResolution *out);
#endif
