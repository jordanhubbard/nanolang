#ifndef NANOISA_SOCKET_INDIRECT_RUNTIME_H
#define NANOISA_SOCKET_INDIRECT_RUNTIME_H
#include "socket_indirect_hosted.h"
#include "socket_runtime_frames.h"
#include "socket_indirect_report.h"
typedef struct {
    uint32_t revision;
    NvmSocketRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmSocketIndirectFrameView;
NvmSocketRuntimeStatus nvm_socket_runtime_indirect_create(const uint8_t *,size_t,
    NvmSocketRuntimeMode,const NvmSocketIndirectOptions *,NvmSocketRuntime **);
const NvmSocketIndirectHostedPlan *nvm_socket_runtime_indirect_plan(const NvmSocketRuntime *);
bool nvm_socket_runtime_indirect_frame_view(const NvmSocketRuntime *,NvmSocketIndirectFrameView *);
NvmSocketRuntimeStatus nvm_socket_runtime_indirect_enter(NvmSocketRuntime *);
/* Exact current FUNCREF only. The callable retains invocation-plan identity;
 * no integer-to-callable constructor or cross-context import exists. */
NvmSocketRuntimeStatus nvm_socket_runtime_indirect_funcref(NvmSocketRuntime *,uint32_t dst);
NvmSocketIndirectExecutionReport nvm_socket_runtime_indirect_finish(NvmSocketRuntime *,NvmSocketRuntimeView *);
NvmSocketIndirectExecutionReport nvm_socket_runtime_indirect_destroy(NvmSocketRuntime **,NvmSocketRuntimeView *);
bool nvm_socket_runtime_indirect_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
