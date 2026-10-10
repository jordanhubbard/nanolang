#ifndef NANOISA_SERVICES_INDIRECT_RUNTIME_H
#define NANOISA_SERVICES_INDIRECT_RUNTIME_H
#include "services_indirect_hosted.h"
#include "services_runtime_frames.h"
#include "services_indirect_report.h"
typedef struct {
    uint32_t revision;
    NvmServicesRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmServicesIndirectFrameView;
NvmServicesRuntimeStatus nvm_services_runtime_indirect_create(const uint8_t *,size_t,
    NvmServicesRuntimeMode,const NvmServicesIndirectOptions *,NvmServicesRuntime **);
const NvmServicesIndirectHostedPlan *nvm_services_runtime_indirect_plan(const NvmServicesRuntime *);
bool nvm_services_runtime_indirect_frame_view(const NvmServicesRuntime *,NvmServicesIndirectFrameView *);
NvmServicesRuntimeStatus nvm_services_runtime_indirect_enter(NvmServicesRuntime *);
/* Exact current FUNCREF only. The callable retains invocation-plan identity;
 * no integer-to-callable constructor or cross-context import exists. */
NvmServicesRuntimeStatus nvm_services_runtime_indirect_funcref(NvmServicesRuntime *,uint32_t dst);
NvmServicesIndirectExecutionReport nvm_services_runtime_indirect_finish(NvmServicesRuntime *,NvmServicesRuntimeView *);
NvmServicesIndirectExecutionReport nvm_services_runtime_indirect_destroy(NvmServicesRuntime **,NvmServicesRuntimeView *);
bool nvm_services_runtime_indirect_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
