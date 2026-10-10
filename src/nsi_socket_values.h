#ifndef NL_NSI_SOCKET_VALUES_H
#define NL_NSI_SOCKET_VALUES_H
#include "nsi_socket.h"

/* I expose a private serialized value lifetime, not source/dispatch authority.
 * Valid caller objects must be disjoint from opaque context storage. */
typedef struct NlSocketValues NlSocketValues;
#define NL_SOCKET_VALUE_SLOTS 64u
typedef enum {
    NL_SOCKET_VALUE_OK, NL_SOCKET_VALUE_ARGUMENT, NL_SOCKET_VALUE_STALE,
    NL_SOCKET_VALUE_TYPE, NL_SOCKET_VALUE_BORROWED, NL_SOCKET_VALUE_LIMIT,
    NL_SOCKET_VALUE_MEMORY, NL_SOCKET_VALUE_DISPOSED, NL_SOCKET_VALUE_STATE,
    NL_SOCKET_VALUE_STATUS_COUNT
} NlSocketValueStatus;
typedef struct { uint64_t invocation, generation; uint32_t slot; } NlSocketValue;
typedef struct { NlSocketValue value; uint64_t epoch; } NlSocketValueBorrow;
typedef struct { bool ok, pending; NlSocketResult error; } NlSocketConnectView;
typedef enum {
    NL_SOCKET_VALUE_CONNECT, NL_SOCKET_VALUE_SEND, NL_SOCKET_VALUE_RECEIVE,
    NL_SOCKET_VALUE_CLOSE
} NlSocketScalarKind;
typedef struct {
    NlSocketScalarKind kind;
    bool ok;
    NlSocketResult detail;
    int64_t value;
    bool eof;
} NlSocketScalarResult;
typedef struct {
    NlSocketValueStatus execution;
    uint64_t cleanup_failures;
    NlSocketResult first_cleanup, next_cleanup;
} NlSocketValuesFinish;

/* I leave outputs unchanged on checked API failure. Zero denotes empty value
 * and borrow storage. Creation requires *out==NULL. */
bool nl_socket_values_storage_bound(size_t *out);
NlSocketValueStatus nl_socket_values_create(NlSocketValues **out);
/* I publish an owned ConnectResult only on VALUE_OK. Host failure is its Error
 * arm; exhausted value capacity acquires nothing. Address/output must be disjoint. */
NlSocketValueStatus nl_socket_values_connect(NlSocketValues *, const NlSocketAddress *, NlSocketValue *out);
NlSocketValueStatus nl_socket_connect_view(NlSocketValues *, const NlSocketValue *, NlSocketConnectView *out);
/* Move and take-Ok require disjoint source/empty output; success clears source
 * and invalidates old copies, retaining the same private adapter owner. */
NlSocketValueStatus nl_socket_value_move(NlSocketValues *, NlSocketValue *, NlSocketValue *out);
NlSocketValueStatus nl_socket_connect_take_ok(NlSocketValues *, NlSocketValue *, NlSocketValue *out);
NlSocketValueStatus nl_socket_connect_take_error(NlSocketValues *, NlSocketValue *, NlSocketResult *out);
NlSocketValueStatus nl_socket_value_borrow(NlSocketValues *, const NlSocketValue *, NlSocketValueBorrow *out);
NlSocketValueStatus nl_socket_value_end_borrow(NlSocketValues *, NlSocketValueBorrow *);
/* I require one exact live exclusive borrow. Host Results retain the owner. */
NlSocketValueStatus nl_socket_value_finish_connect(NlSocketValues *, const NlSocketValueBorrow *, NlSocketScalarResult *out);
NlSocketValueStatus nl_socket_value_send_byte(NlSocketValues *, const NlSocketValueBorrow *, int64_t, NlSocketScalarResult *out);
NlSocketValueStatus nl_socket_value_receive_byte(NlSocketValues *, const NlSocketValueBorrow *, NlSocketScalarResult *out);
NlSocketValueStatus nl_socket_value_close(NlSocketValues *, NlSocketValue *, NlSocketScalarResult *out);
/* Drop accepts a connection or either Result arm. Empty is an idempotent no-op. */
NlSocketValueStatus nl_socket_value_drop(NlSocketValues *, NlSocketValue *);
/* Pure runtime validation queries create no host authority. */
NlSocketValueStatus nl_socket_value_validate(NlSocketValues *, const NlSocketValue *, bool connect_result);
NlSocketValueStatus nl_socket_value_borrow_validate(NlSocketValues *, const NlSocketValueBorrow *);
NlSocketValueStatus nl_socket_values_live_slots(NlSocketValues *, uint64_t *owners, uint64_t *borrowed);
bool nl_socket_values_report(const NlSocketValues *, NlSocketValuesFinish *out);
/* Finish is terminal and reclaims borrowed/unhandled owners too. I cache its
 * first execution status and cleanup failures. Destroy returns that report;
 * invalid status on a live context refuses destruction, allowing correction. */
NlSocketValuesFinish nl_socket_values_finish(NlSocketValues *, NlSocketValueStatus);
NlSocketValuesFinish nl_socket_values_destroy(NlSocketValues *, NlSocketValueStatus);
#endif
