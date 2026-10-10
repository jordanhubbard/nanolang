/* I instantiate only this immutable catalog's nominal and flow identities. */
#include "websocket_flow.h"
#include "service_websocket_nominal_config.h"
#define FLOW_FN(name) nvm_websocket_flow_##name
#define FLOW_TYPE(name) NvmWebSocketFlow##name
#define FLOW_CONST(name) NVM_WEBSOCKET_FLOW_##name
#define FLOW_ENDPOINT_TYPE UINT32_MAX
#define FLOW_ENDPOINT_CHECK 0u
#define FLOW_TIMEOUT_CHECK NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT
#define CODE_FN(name) nvm_websocket_code_##name
#define CODE_TYPE(name) NvmWebSocketCode##name
#define CODE_CONST(name) NVM_WEBSOCKET_CODE_##name
#define BODY_FN(name) nvm_websocket_body_##name
#define BODY_TYPE(name) NvmWebSocketBody##name
#define BODY_CONST(name) NVM_WEBSOCKET_BODY_##name
#define SERVICE_FN(name) nvm_websocket_##name
#define SERVICE_TYPE(name) NvmWebSocket##name
#define SERVICE_CONST(name) NVM_WEBSOCKET_##name
