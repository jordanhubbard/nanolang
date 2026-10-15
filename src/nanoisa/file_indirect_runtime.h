#ifndef NANOISA_FILE_INDIRECT_RUNTIME_H
#define NANOISA_FILE_INDIRECT_RUNTIME_H
#include "file_indirect_hosted.h"
#include "file_runtime_frames.h"
#include "file_indirect_report.h"
typedef struct {
    uint32_t revision;
    NvmFileRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmFileIndirectFrameView;
NvmFileRuntimeStatus nvm_file_runtime_indirect_create(const uint8_t *,size_t,
    NvmFileRuntimeMode,const NvmFileIndirectOptions *,NvmFileRuntime **);
const NvmFileIndirectHostedPlan *nvm_file_runtime_indirect_plan(const NvmFileRuntime *);
bool nvm_file_runtime_indirect_frame_view(const NvmFileRuntime *,NvmFileIndirectFrameView *);
NvmFileRuntimeStatus nvm_file_runtime_indirect_enter(NvmFileRuntime *);
/* Exact current FUNCREF only. The callable retains invocation-plan identity;
 * no integer-to-callable constructor or cross-context import exists. */
NvmFileRuntimeStatus nvm_file_runtime_indirect_funcref(NvmFileRuntime *,uint32_t dst);
NvmFileIndirectExecutionReport nvm_file_runtime_indirect_finish(NvmFileRuntime *,NvmFileRuntimeView *);
NvmFileIndirectExecutionReport nvm_file_runtime_indirect_destroy(NvmFileRuntime **,NvmFileRuntimeView *);
bool nvm_file_runtime_indirect_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
