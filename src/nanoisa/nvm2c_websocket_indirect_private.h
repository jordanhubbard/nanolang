#ifndef NANOISA_NVM2C_WEBSOCKET_INDIRECT_PRIVATE_H
#define NANOISA_NVM2C_WEBSOCKET_INDIRECT_PRIVATE_H
#include "websocket_indirect_native_abi.h"
#ifdef NVM_WEBSOCKET_INDIRECT_NATIVE_PRIVATE
/* Explicit private providers only; no host effects during emission. Immutable
 * input and disjoint output/diagnostics; success publishes malloc-owned C11. */
NvmWebSocketRuntimeStatus nvm2c_websocket_indirect_private_emit(const uint8_t *,size_t,char **,char *,size_t);
NvmWebSocketIndirectExecutionReport nvm_websocket_native_indirect_execute(const NvmWebSocketIndirectOptions *,NvmWebSocketRuntimeView *,const NlWsTransportPolicy *);
#endif
#endif
