#ifndef NANOISA_WEBSOCKET_CYCLIC_RUNTIME_H
#define NANOISA_WEBSOCKET_CYCLIC_RUNTIME_H
#include "websocket_cyclic_hosted.h"
#include "websocket_runtime_frames.h"
#include "websocket_cyclic_report.h"
/* Trusted carrier protocol. Granted wrappers own serialization; direct callers
 * still require external serialization and disjoint valid storage. */
typedef struct {
    uint32_t revision;
    NvmWebSocketRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmWebSocketCyclicFrameView;
NvmWebSocketRuntimeStatus nvm_websocket_runtime_cyclic_create(const uint8_t *,size_t,
    NvmWebSocketRuntimeMode,const NvmWebSocketCyclicOptions *,NvmWebSocketRuntime **);
/* Borrowed until context destruction; NULL for an acyclic context. */
const NvmWebSocketCyclicHostedPlan *nvm_websocket_runtime_cyclic_plan(const NvmWebSocketRuntime *);
bool nvm_websocket_runtime_cyclic_frame_view(const NvmWebSocketRuntime *,NvmWebSocketCyclicFrameView *);
/* Validate the exact input variant and charge once before any effects. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_cyclic_enter(NvmWebSocketRuntime *);
bool nvm_websocket_runtime_cyclic_report(const NvmWebSocketRuntime *,NvmWebSocketCyclicExecutionReport *);
NvmWebSocketCyclicExecutionReport nvm_websocket_runtime_cyclic_finish(NvmWebSocketRuntime *,NvmWebSocketRuntimeView *);
NvmWebSocketCyclicExecutionReport nvm_websocket_runtime_cyclic_destroy(NvmWebSocketRuntime **,NvmWebSocketRuntimeView *);
bool nvm_websocket_runtime_cyclic_abi(uint32_t,size_t,size_t,size_t,size_t);
/* Separate semantic contract for generated cyclic operations; old native ABI is unchanged. */
bool nvm_websocket_runtime_cyclic_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
