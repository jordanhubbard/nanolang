#ifndef NL_NSI_SOCKET_H
#define NL_NSI_SOCKET_H

#include "nsi_cap.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* I expose only a private serialized socket adapter. These tokens grant
 * no source, import, NSI-dispatch or foreign-call admission. Creation is also
 * serialized, and acquisition must not race process fork/exec. */
typedef struct NlSocketService NlSocketService;
typedef struct { uint64_t context_id; NlCap cap; } NlSocketToken;
typedef struct { NlSocketToken endpoints[2]; } NlSocketPair;
typedef enum { NL_SOCKET_IPV4 = 4, NL_SOCKET_IPV6 = 6 } NlSocketFamily;
typedef struct {
    NlSocketFamily family;
    uint8_t address[16]; /* I store network-order bytes; IPv4 uses the first four. */
    uint16_t port;       /* I accept a nonzero host-order port. */
    uint32_t scope_id;   /* I accept an interface index only for IPv6. */
} NlSocketAddress;
typedef enum {
    NL_SOCKET_OK = 0, NL_SOCKET_EOF, NL_SOCKET_WOULD_BLOCK,
    NL_SOCKET_INTERRUPTED, NL_SOCKET_ARGUMENT, NL_SOCKET_DISPOSED,
    NL_SOCKET_TOKEN, NL_SOCKET_RIGHTS, NL_SOCKET_CAPACITY, NL_SOCKET_LIMIT,
    NL_SOCKET_MEMORY, NL_SOCKET_IO
} NlSocketStatus;
typedef struct {
    NlSocketStatus status;
    int host_errno, cleanup_errno;
    size_t bytes;
    unsigned close_attempts, closed_count;
    bool eof, consumed, cleanup_failed, closure_unknown;
    bool connect_pending;
} NlSocketResult;

/* I publish outputs only on success. Receive additionally publishes byte0 on
 * EOF; would-block/interruption leave it unchanged. Transfer permits out==token.
 * Receive refuses byte-output overlap with the input token before host I/O.
 * Valid output storage must not discard unrelated live owners. */
NlSocketResult nl_socket_service_create(NlSocketService **out);
NlSocketResult nl_socket_acquire_pair(NlSocketService *, uint32_t left_rights,
                                     uint32_t right_rights, NlSocketPair *out);
/* My trusted host caller supplies network authority. OK publishes one owner;
 * connect_pending distinguishes an unfinished connect from a ready connection.
 * Address/output overlap refuses. IPv4 requires zero tail bytes and scope.
 * I retain neither the address nor a caller buffer. */
NlSocketResult nl_socket_acquire_tcp(NlSocketService *, const NlSocketAddress *,
                                    uint32_t rights, NlSocketToken *out);
/* I poll at most once with zero timeout, then read SO_ERROR only on readiness.
 * A terminal failure remains latched until close/disposal. Pending data calls
 * return WOULD_BLOCK without host I/O. Completion needs no READ/WRITE right. */
NlSocketResult nl_socket_finish_connect(NlSocketService *, const NlSocketToken *);
NlSocketResult nl_socket_send_byte(NlSocketService *, const NlSocketToken *, uint8_t);
NlSocketResult nl_socket_receive_byte(NlSocketService *, const NlSocketToken *, uint8_t *out);
NlSocketResult nl_socket_transfer(NlSocketService *, const NlSocketToken *, NlSocketToken *out);
/* An accepted close retires authority before exactly one host close attempt.
 * Error means closure_unknown, not a retryable numeric descriptor. */
NlSocketResult nl_socket_consume_close(NlSocketService *, const NlSocketToken *);
/* Disposal attempts every remaining close, retains the first error and reports
 * any earlier unknown closure. Destruction frees storage; no later use is valid. */
NlSocketResult nl_socket_service_dispose(NlSocketService *);
NlSocketResult nl_socket_service_destroy(NlSocketService *);

#endif
