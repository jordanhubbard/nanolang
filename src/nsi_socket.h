#ifndef NL_NSI_SOCKET_H
#define NL_NSI_SOCKET_H

#include "nsi_cap.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* I expose only a private serialized local-socket adapter. These tokens grant
 * no source, import, NSI-dispatch or foreign-call admission. Creation is also
 * serialized, and acquisition must not race process fork/exec. */
typedef struct NlSocketService NlSocketService;
typedef struct { uint64_t context_id; NlCap cap; } NlSocketToken;
typedef struct { NlSocketToken endpoints[2]; } NlSocketPair;
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
} NlSocketResult;

/* I publish outputs only on success. Receive additionally publishes byte0 on
 * EOF; would-block/interruption leave it unchanged. Transfer permits out==token.
 * Receive refuses byte-output overlap with the input token before host I/O.
 * Valid output storage must not discard unrelated live owners. */
NlSocketResult nl_socket_service_create(NlSocketService **out);
NlSocketResult nl_socket_acquire_pair(NlSocketService *, uint32_t left_rights,
                                     uint32_t right_rights, NlSocketPair *out);
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
