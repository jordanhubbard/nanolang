#ifndef NANOISA_FILE_RUNTIME_H
#define NANOISA_FILE_RUNTIME_H
#include "file_hosted.h"
#include "../nsi_file_values.h"

/* Private carrier primitives, not a CODE dispatcher or public execution route.
 * Every operation requires external serialization with the private File cores,
 * including construction/destruction. No thread-safety claim. All caller output
 * storage is disjoint from contexts, input spans and other output objects. */
#define NVM_FILE_RUNTIME_BYTES (64u*1024u*1024u)
#define NVM_FILE_RUNTIME_NO_SLOT UINT32_MAX
#define NVM_FILE_RUNTIME_FIELDS 7u
typedef enum { NVM_FILE_RUNTIME_VM, NVM_FILE_RUNTIME_NATIVE } NvmFileRuntimeMode;
typedef enum {
    NVM_FILE_RUNTIME_OK, NVM_FILE_RUNTIME_INVALID, NVM_FILE_RUNTIME_UNRESOLVED,
    NVM_FILE_RUNTIME_LIMIT, NVM_FILE_RUNTIME_MEMORY, NVM_FILE_RUNTIME_STATE,
    NVM_FILE_RUNTIME_TYPE, NVM_FILE_RUNTIME_STALE, NVM_FILE_RUNTIME_BORROWED,
    NVM_FILE_RUNTIME_ASSERT, NVM_FILE_RUNTIME_CLEANUP, NVM_FILE_RUNTIME_BUSY
} NvmFileRuntimeStatus;
typedef struct {
    bool initialized, owning, formal;
    NvmFileFlowDeclaration type;
    NvmFileFlowArm arm;
    uint8_t fields;
    int64_t values[NVM_FILE_RUNTIME_FIELDS];
} NvmFileRuntimeView; /* No File handle, capability, stream or borrow epoch. */
typedef struct {
    NvmFileRuntimeStatus status;
    bool acquired;
    NlFileValueStatus core_status;
    uint32_t function, instruction;
    NlFileValuesFinish cleanup;
} NvmFileRuntimeReport;
typedef struct {
    size_t allocation_bound;
    uint32_t values, references, regions;
    uint16_t frames;
} NvmFileRuntimeStorage;
typedef struct NvmFileRuntime NvmFileRuntime;
/* Preparation and arena allocation only; begin creates the fresh File core.
 * Failure preserves *out. Success owns the plan and borrows no input bytes. */
NvmFileRuntimeStatus nvm_file_runtime_create(const uint8_t *,size_t,NvmFileRuntimeMode,NvmFileRuntime **out);
bool nvm_file_runtime_storage(const NvmFileRuntime *,NvmFileRuntimeStorage *);
const NvmFileHostedPlan *nvm_file_runtime_plan(const NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_begin(NvmFileRuntime *);
/* The later matched dispatcher supplies actual instruction sites. This carrier
 * API does not prove its caller followed the checked CFG or discharged masks. */
NvmFileRuntimeStatus nvm_file_runtime_site(NvmFileRuntime *,uint32_t function,uint16_t instruction);
NvmFileRuntimeStatus nvm_file_runtime_fail(NvmFileRuntime *,NvmFileRuntimeStatus);
bool nvm_file_runtime_current_root(const NvmFileRuntime *,uint32_t *);
bool nvm_file_runtime_view(const NvmFileRuntime *,uint32_t,NvmFileRuntimeView *);
/* Destinations must be empty. No operation silently overwrites an owner. */
NvmFileRuntimeStatus nvm_file_runtime_scalar(NvmFileRuntime *,uint32_t,uint8_t,int64_t);
NvmFileRuntimeStatus nvm_file_runtime_copy(NvmFileRuntime *,uint32_t,uint32_t);
NvmFileRuntimeStatus nvm_file_runtime_move(NvmFileRuntime *,uint32_t,uint32_t);
NvmFileRuntimeStatus nvm_file_runtime_drop(NvmFileRuntime *,uint32_t);
NvmFileRuntimeStatus nvm_file_runtime_construct(NvmFileRuntime *,uint32_t ordinal,uint16_t variant,const uint32_t *,size_t,uint32_t);
NvmFileRuntimeStatus nvm_file_runtime_project(NvmFileRuntime *,uint32_t,uint16_t,uint32_t);
NvmFileRuntimeStatus nvm_file_runtime_result_arm(NvmFileRuntime *,uint32_t,NvmFileFlowArm *);
NvmFileRuntimeStatus nvm_file_runtime_take(NvmFileRuntime *,uint32_t,NvmFileFlowArm,uint32_t);
NvmFileRuntimeStatus nvm_file_runtime_region_begin(NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_region_end(NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_borrow(NvmFileRuntime *,uint32_t owner,uint32_t reference);
NvmFileRuntimeStatus nvm_file_runtime_bind_formal(NvmFileRuntime *,uint32_t source_reference,uint32_t formal_root,uint32_t formal_reference);
NvmFileRuntimeStatus nvm_file_runtime_end_reference(NvmFileRuntime *,uint32_t);
/* Exact checked module import, with original core receiver/domain semantics.
 * input is an INT root for write or File root for close; otherwise NO_SLOT.
 * reference is exclusive for write/rewind/read; otherwise NO_SLOT. */
NvmFileRuntimeStatus nvm_file_runtime_service(NvmFileRuntime *,uint32_t import,uint32_t reference,uint32_t input,uint32_t output);
/* Enforce initializer/entry sequencing and scalar staging, not instruction CFG.
 * VOID initializer uses NO_SLOT; entry uses its exact scalar result root. */
NvmFileRuntimeStatus nvm_file_runtime_complete_root(NvmFileRuntime *,uint32_t result);
/* Finish drains all roots before core disposal; only a complete clean invocation
 * publishes scalar output. Failed/repeated calls preserve first error/report.
 * A BUSY no-acquisition return does not mutate the active report or free memory. */
NvmFileRuntimeReport nvm_file_runtime_finish(NvmFileRuntime *,NvmFileRuntimeView *);
NvmFileRuntimeReport nvm_file_runtime_destroy(NvmFileRuntime **,NvmFileRuntimeView *);
#endif
