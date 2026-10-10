#include "websocket_flow_config.h"
#include "websocket_indirect_runtime.h"
#define SERVICE_HAS_ENDPOINT 0
#define SERVICE_PUBLIC_SUPPORTED 1
#define SERVICE_TYPE_TEXT "NvmWebSocket"
#define SERVICE_CONST_TEXT "NVM_WEBSOCKET_"
#define SERVICE_FN_TEXT "nvm_websocket_"
#define SERVICE_EMITTER_TEXT "nvm2c_websocket_"
#define SERVICE_PUBLIC_HEADER_TEXT "websocket_indirect_native_public.h"
#define SERVICE_NAME_TEXT "WebSocket"
#define SERVICE_INSTALL_TEXT "websocket"

#define SERVICE_ORDERED_ARGUMENTS 1
#define SERVICE_EXPLICIT_POLICY 1
#define SERVICE_EXECUTION_AUTH_ARGUMENT ,const NlWsTransportPolicy *policy
#define SERVICE_EXECUTION_AUTH_CHECK(c) nvm_websocket_runtime_policy(c,policy)
