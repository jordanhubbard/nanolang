#ifndef NANOISA_WEBSOCKET_INDIRECT_RUNTIME_H
#define NANOISA_WEBSOCKET_INDIRECT_RUNTIME_H
#include "websocket_indirect_hosted.h"
#include "websocket_runtime_frames.h"
#include "websocket_indirect_report.h"
typedef struct {
    uint32_t revision;
    NvmWebSocketRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmWebSocketIndirectFrameView;
NvmWebSocketRuntimeStatus nvm_websocket_runtime_indirect_create(const uint8_t *,size_t,
    NvmWebSocketRuntimeMode,const NvmWebSocketIndirectOptions *,NvmWebSocketRuntime **);
const NvmWebSocketIndirectHostedPlan *nvm_websocket_runtime_indirect_plan(const NvmWebSocketRuntime *);
bool nvm_websocket_runtime_indirect_frame_view(const NvmWebSocketRuntime *,NvmWebSocketIndirectFrameView *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_indirect_enter(NvmWebSocketRuntime *);
/* Exact current FUNCREF only. The callable retains invocation-plan identity;
 * no integer-to-callable constructor or cross-context import exists. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_indirect_funcref(NvmWebSocketRuntime *,uint32_t dst);
NvmWebSocketIndirectExecutionReport nvm_websocket_runtime_indirect_finish(NvmWebSocketRuntime *,NvmWebSocketRuntimeView *);
NvmWebSocketIndirectExecutionReport nvm_websocket_runtime_indirect_destroy(NvmWebSocketRuntime **,NvmWebSocketRuntimeView *);
bool nvm_websocket_runtime_indirect_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
