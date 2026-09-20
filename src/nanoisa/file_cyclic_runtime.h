#ifndef NANOISA_FILE_CYCLIC_RUNTIME_H
#define NANOISA_FILE_CYCLIC_RUNTIME_H
#include "file_cyclic_hosted.h"
#include "file_runtime_frames.h"
#define NVM_FILE_CYCLIC_RUNTIME_REVISION 1u
#define NVM_FILE_CYCLIC_FUEL_MAX UINT64_C(1000000)
#define NVM_FILE_CYCLIC_FUEL_DEFAULT UINT64_C(100000)
/* Source-private carrier protocol only: no dispatcher, public selector or
 * installed ABI. External serialization and disjoint outputs remain required. */
typedef struct { uint32_t revision; uint64_t instruction_limit; } NvmFileCyclicOptions;
typedef struct {
    uint32_t revision;
    NvmFileRuntimeReport runtime;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted;
} NvmFileCyclicExecutionReport;
typedef struct {
    uint32_t revision;
    NvmFileRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmFileCyclicFrameView;
NvmFileRuntimeStatus nvm_file_runtime_cyclic_create(const uint8_t *,size_t,
    NvmFileRuntimeMode,const NvmFileCyclicOptions *,NvmFileRuntime **);
/* Borrowed until context destruction; NULL for an acyclic context. */
const NvmFileCyclicHostedPlan *nvm_file_runtime_cyclic_plan(const NvmFileRuntime *);
bool nvm_file_runtime_cyclic_frame_view(const NvmFileRuntime *,NvmFileCyclicFrameView *);
/* Validate the exact input variant and charge once before any effects. */
NvmFileRuntimeStatus nvm_file_runtime_cyclic_enter(NvmFileRuntime *);
bool nvm_file_runtime_cyclic_report(const NvmFileRuntime *,NvmFileCyclicExecutionReport *);
NvmFileCyclicExecutionReport nvm_file_runtime_cyclic_finish(NvmFileRuntime *,NvmFileRuntimeView *);
NvmFileCyclicExecutionReport nvm_file_runtime_cyclic_destroy(NvmFileRuntime **,NvmFileRuntimeView *);
bool nvm_file_runtime_cyclic_abi(uint32_t,size_t,size_t,size_t,size_t);
/* Separate semantic contract for generated cyclic operations; old native ABI is unchanged. */
bool nvm_file_runtime_cyclic_native_abi(uint32_t,size_t,size_t,size_t,size_t,size_t);
#endif
