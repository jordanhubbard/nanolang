#include "websocket_flow_config.h"
#define CoreValues NlWsValues
#define CoreValue NlWsValue
#define CoreBorrow NlWsValueBorrow
#define CoreStatus NlWsValueStatus
#define CoreFinish NlWsValuesFinish
#define CoreResult NlWsTransportResult

#define CoreOpenView NlWsConnectView
#define CORE_STATUS(name) NL_WS_VALUE_##name
#define RUNTIME_HAS_ENDPOINT 0
#define RUNTIME_ENDPOINT_ORDINAL UINT32_MAX
#define RUNTIME_VALUE_ADAPTER "websocket_runtime_values.inc"
#define RUNTIME_SERVICE_ADAPTER "websocket_runtime_service.inc"
#define CORE_FN(name) RUNTIME_CORE_##name
#define RUNTIME_CORE_open_take_error nl_ws_connect_take_error
#define RUNTIME_CORE_open_take_ok nl_ws_connect_take_ok
#define RUNTIME_CORE_open_view nl_ws_connect_view
#define RUNTIME_CORE_value_borrow nl_ws_value_borrow
#define RUNTIME_CORE_value_borrow_validate nl_ws_value_borrow_validate
#define RUNTIME_CORE_value_drop nl_ws_value_drop
#define RUNTIME_CORE_value_end_borrow nl_ws_value_end_borrow
#define RUNTIME_CORE_value_move nl_ws_value_move
#define RUNTIME_CORE_value_validate nl_ws_value_validate
#define RUNTIME_CORE_values_destroy nl_ws_values_destroy
#define RUNTIME_CORE_values_live_slots nl_ws_values_live_slots
#define RUNTIME_CORE_values_report nl_ws_values_report

#define RUNTIME_CONTEXT_MEMBERS NlWsTransportPolicy policy; char resolver_helper[4096]; bool policy_set;
#define RUNTIME_IMPORTS(c) NVM_WEBSOCKET_NOMINAL_METHODS
#define RUNTIME_CORE_BUDGET (32u*1024u*1024u)
#define RUNTIME_BOUND(out) fr_ws_bound(out)
#define RUNTIME_BEGIN(c) ((c)->policy_set?nl_ws_values_create_bounded(&(c)->policy,RUNTIME_CORE_BUDGET,&(c)->files):NL_WS_VALUE_ARGUMENT)
#define RUNTIME_SCALAR_PAYLOAD(c,p) fr_ws_payload(c,p)
static bool fr_ws_bound(size_t *out){*out=RUNTIME_CORE_BUDGET;return true;}
