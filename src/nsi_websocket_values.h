#ifndef NL_NSI_WEBSOCKET_VALUES_H
#define NL_NSI_WEBSOCKET_VALUES_H
#include "nsi_websocket_transport.h"

/* I expose private serialized invocation values. Caller storage is valid and
 * disjoint from opaque contexts. Tokens are runtime identities, not source
 * integers or a security boundary against arbitrary native memory access. */
#define NL_WS_VALUE_SLOTS 64u
typedef struct NlWsValues NlWsValues;
typedef enum {
    NL_WS_VALUE_OK, NL_WS_VALUE_ARGUMENT, NL_WS_VALUE_STALE, NL_WS_VALUE_TYPE,
    NL_WS_VALUE_BORROWED, NL_WS_VALUE_LIMIT, NL_WS_VALUE_MEMORY,
    NL_WS_VALUE_DISPOSED, NL_WS_VALUE_STATE, NL_WS_VALUE_STATUS_COUNT
} NlWsValueStatus;
typedef struct { uint64_t invocation,generation; uint32_t slot; } NlWsValue;
typedef struct { NlWsValue value; uint64_t epoch; } NlWsValueBorrow;
typedef struct { bool ok; NlWsTransportResult error; } NlWsConnectView;
typedef struct {
    NlWsValueStatus execution;
    uint64_t cleanup_failures;
    NlWsTransportResult first_cleanup,next_cleanup;
} NlWsValuesFinish;

/* I copy policy and resolver path at creation. Empty value/borrow storage is
 * all zero; creation requires *out==NULL. Checked API refusal preserves output.
 * A VALUE_OK host operation publishes its independent transport result, whose
 * status may be Error. Connection errors retain a ConnectResult until consumed. */
NlWsValueStatus nl_ws_values_create(const NlWsTransportPolicy *,NlWsValues **out);
NlWsValueStatus nl_ws_values_connect(NlWsValues *,const void *,size_t,int64_t,NlWsValue *out);
NlWsValueStatus nl_ws_connect_view(NlWsValues *,const NlWsValue *,NlWsConnectView *out);
NlWsValueStatus nl_ws_connect_take_ok(NlWsValues *,NlWsValue *,NlWsValue *out);
NlWsValueStatus nl_ws_connect_take_error(NlWsValues *,NlWsValue *,NlWsTransportResult *out);
NlWsValueStatus nl_ws_value_move(NlWsValues *,NlWsValue *,NlWsValue *out);
NlWsValueStatus nl_ws_value_borrow(NlWsValues *,const NlWsValue *,NlWsValueBorrow *out);
NlWsValueStatus nl_ws_value_end_borrow(NlWsValues *,NlWsValueBorrow *);
NlWsValueStatus nl_ws_value_send(NlWsValues *,const NlWsValueBorrow *,bool,
    const void *,size_t,int64_t,NlWsTransportResult *out);
/* Message output must be empty. I publish independent bytes only on transport
 * OK; callers free them with nl_ws_message_free, even after values destruction. */
NlWsValueStatus nl_ws_value_receive(NlWsValues *,const NlWsValueBorrow *,int64_t,
    NlWsMessage *message,NlWsTransportResult *out);
NlWsValueStatus nl_ws_value_close(NlWsValues *,NlWsValue *,int64_t,NlWsTransportResult *out);
NlWsValueStatus nl_ws_value_drop(NlWsValues *,NlWsValue *);
NlWsValueStatus nl_ws_value_validate(NlWsValues *,const NlWsValue *,bool connect_result);
NlWsValueStatus nl_ws_value_borrow_validate(NlWsValues *,const NlWsValueBorrow *);
NlWsValueStatus nl_ws_values_live_slots(NlWsValues *,uint64_t *owners,uint64_t *borrowed);
bool nl_ws_values_report(const NlWsValues *,NlWsValuesFinish *out);
/* Implicit cleanup aborts I/O without waiting for a peer handshake, including
 * borrowed owners. Finish caches its report; invalid execution status refuses
 * finish/destruction. Explicit close consumes before attempting host cleanup. */
NlWsValuesFinish nl_ws_values_finish(NlWsValues *,NlWsValueStatus);
NlWsValuesFinish nl_ws_values_destroy(NlWsValues *,NlWsValueStatus);
#endif
