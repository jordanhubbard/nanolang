#ifndef NANOISA_SOCKET_CYCLIC_RUNTIME_H
#define NANOISA_SOCKET_CYCLIC_RUNTIME_H
#include "socket_cyclic_hosted.h"
#include "socket_runtime_frames.h"
#include "socket_cyclic_report.h"
/* Trusted carrier protocol. Granted wrappers own serialization; direct callers
 * still require external serialization and disjoint valid storage. */
typedef struct {
    uint32_t revision;
    NvmSocketRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmSocketCyclicFrameView;
NvmSocketRuntimeStatus nvm_socket_runtime_cyclic_create(const uint8_t *,size_t,
    NvmSocketRuntimeMode,const NvmSocketCyclicOptions *,NvmSocketRuntime **);
/* Borrowed until context destruction; NULL for an acyclic context. */
const NvmSocketCyclicHostedPlan *nvm_socket_runtime_cyclic_plan(const NvmSocketRuntime *);
bool nvm_socket_runtime_cyclic_frame_view(const NvmSocketRuntime *,NvmSocketCyclicFrameView *);
/* Validate the exact input variant and charge once before any effects. */
NvmSocketRuntimeStatus nvm_socket_runtime_cyclic_enter(NvmSocketRuntime *);
bool nvm_socket_runtime_cyclic_report(const NvmSocketRuntime *,NvmSocketCyclicExecutionReport *);
NvmSocketCyclicExecutionReport nvm_socket_runtime_cyclic_finish(NvmSocketRuntime *,NvmSocketRuntimeView *);
NvmSocketCyclicExecutionReport nvm_socket_runtime_cyclic_destroy(NvmSocketRuntime **,NvmSocketRuntimeView *);
bool nvm_socket_runtime_cyclic_abi(uint32_t,size_t,size_t,size_t,size_t);
/* Separate semantic contract for generated cyclic operations; old native ABI is unchanged. */
bool nvm_socket_runtime_cyclic_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
