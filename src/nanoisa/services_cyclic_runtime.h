#ifndef NANOISA_SERVICES_CYCLIC_RUNTIME_H
#define NANOISA_SERVICES_CYCLIC_RUNTIME_H
#include "services_cyclic_hosted.h"
#include "services_runtime_frames.h"
#include "services_cyclic_report.h"
/* Trusted carrier protocol. Granted wrappers own serialization; direct callers
 * still require external serialization and disjoint valid storage. */
typedef struct {
    uint32_t revision;
    NvmServicesRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmServicesCyclicFrameView;
NvmServicesRuntimeStatus nvm_services_runtime_cyclic_create(const uint8_t *,size_t,
    NvmServicesRuntimeMode,const NvmServicesCyclicOptions *,NvmServicesRuntime **);
/* Borrowed until context destruction; NULL for an acyclic context. */
const NvmServicesCyclicHostedPlan *nvm_services_runtime_cyclic_plan(const NvmServicesRuntime *);
bool nvm_services_runtime_cyclic_frame_view(const NvmServicesRuntime *,NvmServicesCyclicFrameView *);
/* Validate the exact input variant and charge once before any effects. */
NvmServicesRuntimeStatus nvm_services_runtime_cyclic_enter(NvmServicesRuntime *);
bool nvm_services_runtime_cyclic_report(const NvmServicesRuntime *,NvmServicesCyclicExecutionReport *);
NvmServicesCyclicExecutionReport nvm_services_runtime_cyclic_finish(NvmServicesRuntime *,NvmServicesRuntimeView *);
NvmServicesCyclicExecutionReport nvm_services_runtime_cyclic_destroy(NvmServicesRuntime **,NvmServicesRuntimeView *);
bool nvm_services_runtime_cyclic_abi(uint32_t,size_t,size_t,size_t,size_t);
/* Separate semantic contract for generated cyclic operations; old native ABI is unchanged. */
bool nvm_services_runtime_cyclic_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
