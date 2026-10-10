#ifndef NANOISA_SERVICES_RUNTIME_H
#define NANOISA_SERVICES_RUNTIME_H
#include "services_hosted.h"
#include "../nsi_services_values.h"

/* Private carrier primitives, not a CODE dispatcher or public execution route.
 * Every operation requires external serialization with the private File/TCP cores,
 * including construction/destruction. No thread-safety claim. All caller output
 * storage is disjoint from contexts, input spans and other output objects. */
#define NVM_SERVICES_RUNTIME_BYTES (64u*1024u*1024u)
#define NVM_SERVICES_RUNTIME_NO_SLOT UINT32_MAX
#define NVM_SERVICES_RUNTIME_FIELDS 11u
typedef enum { NVM_SERVICES_RUNTIME_VM, NVM_SERVICES_RUNTIME_NATIVE } NvmServicesRuntimeMode;
typedef enum {
    NVM_SERVICES_RUNTIME_OK, NVM_SERVICES_RUNTIME_INVALID, NVM_SERVICES_RUNTIME_UNRESOLVED,
    NVM_SERVICES_RUNTIME_LIMIT, NVM_SERVICES_RUNTIME_MEMORY, NVM_SERVICES_RUNTIME_STATE,
    NVM_SERVICES_RUNTIME_TYPE, NVM_SERVICES_RUNTIME_STALE, NVM_SERVICES_RUNTIME_BORROWED,
    NVM_SERVICES_RUNTIME_ASSERT, NVM_SERVICES_RUNTIME_CLEANUP, NVM_SERVICES_RUNTIME_BUSY
} NvmServicesRuntimeStatus;
typedef struct {
    bool initialized, owning, formal;
    NvmServicesFlowDeclaration type;
    NvmServicesFlowArm arm;
    uint8_t fields;
    int64_t values[NVM_SERVICES_RUNTIME_FIELDS];
} NvmServicesRuntimeView; /* No File/TCP handle, capability, stream or borrow epoch. */
typedef struct {
    NvmServicesRuntimeStatus status;
    bool acquired;
    NlServicesValueStatus core_status;
    uint32_t function, instruction;
    NlServicesFinish cleanup;
} NvmServicesRuntimeReport;
typedef struct {
    size_t allocation_bound;
    uint32_t values, references, regions;
    uint16_t frames;
} NvmServicesRuntimeStorage;
typedef struct NvmServicesRuntime NvmServicesRuntime;
/* Preparation and arena allocation only; begin creates the fresh per-instance cores.
 * Failure preserves *out. Success owns the plan and borrows no input bytes. */
NvmServicesRuntimeStatus nvm_services_runtime_create(const uint8_t *,size_t,NvmServicesRuntimeMode,NvmServicesRuntime **out);
bool nvm_services_runtime_storage(const NvmServicesRuntime *,NvmServicesRuntimeStorage *);
const NvmServicesHostedPlan *nvm_services_runtime_plan(const NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_begin(NvmServicesRuntime *);
/* The later matched dispatcher supplies actual instruction sites. This carrier
 * API does not prove its caller followed the checked CFG or discharged masks. */
NvmServicesRuntimeStatus nvm_services_runtime_site(NvmServicesRuntime *,uint32_t function,uint16_t instruction);
NvmServicesRuntimeStatus nvm_services_runtime_fail(NvmServicesRuntime *,NvmServicesRuntimeStatus);
bool nvm_services_runtime_current_root(const NvmServicesRuntime *,uint32_t *);
bool nvm_services_runtime_view(const NvmServicesRuntime *,uint32_t,NvmServicesRuntimeView *);
/* Destinations must be empty. No operation silently overwrites an owner. */
NvmServicesRuntimeStatus nvm_services_runtime_scalar(NvmServicesRuntime *,uint32_t,uint8_t,int64_t);
/* I copy counted immutable bytes into bounded invocation storage. Strings are
 * at most 1 MiB, with at most 4096 distinct strings per invocation; repeated
 * equal strings share storage. Copies and returns retain bytes until finish.
 * My view stores an invocation-local identity, never a host pointer. */
NvmServicesRuntimeStatus nvm_services_runtime_string(NvmServicesRuntime *,uint32_t,const void *,size_t);
NvmServicesRuntimeStatus nvm_services_runtime_string_literal(NvmServicesRuntime *,uint32_t);
NvmServicesRuntimeStatus nvm_services_runtime_string_operation(NvmServicesRuntime *,uint8_t,uint32_t,uint32_t,uint32_t);
bool nvm_services_runtime_string_read(const NvmServicesRuntime *,uint32_t,void *,size_t,size_t *);
NvmServicesRuntimeStatus nvm_services_runtime_copy(NvmServicesRuntime *,uint32_t,uint32_t);
NvmServicesRuntimeStatus nvm_services_runtime_move(NvmServicesRuntime *,uint32_t,uint32_t);
NvmServicesRuntimeStatus nvm_services_runtime_drop(NvmServicesRuntime *,uint32_t);
NvmServicesRuntimeStatus nvm_services_runtime_construct(NvmServicesRuntime *,uint32_t ordinal,uint16_t variant,const uint32_t *,size_t,uint32_t);
NvmServicesRuntimeStatus nvm_services_runtime_project(NvmServicesRuntime *,uint32_t,uint16_t,uint32_t);
NvmServicesRuntimeStatus nvm_services_runtime_result_arm(NvmServicesRuntime *,uint32_t,NvmServicesFlowArm *);
NvmServicesRuntimeStatus nvm_services_runtime_take(NvmServicesRuntime *,uint32_t,NvmServicesFlowArm,uint32_t);
NvmServicesRuntimeStatus nvm_services_runtime_region_begin(NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_region_end(NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_borrow(NvmServicesRuntime *,uint32_t owner,uint32_t reference);
NvmServicesRuntimeStatus nvm_services_runtime_borrow_shared(NvmServicesRuntime *,uint32_t owner,uint32_t reference);
NvmServicesRuntimeStatus nvm_services_runtime_bind_formal(NvmServicesRuntime *,uint32_t source_reference,uint32_t formal_root,uint32_t formal_reference);
NvmServicesRuntimeStatus nvm_services_runtime_end_reference(NvmServicesRuntime *,uint32_t);
/* Exact checked module import, with original core receiver/domain semantics.
 * input is Endpoint for TCP begin-connect, INT for byte writes, or the exact
 * instance owner for close; otherwise NO_SLOT.
 * reference is exclusive for methods 1..3 of that instance; otherwise NO_SLOT. */
NvmServicesRuntimeStatus nvm_services_runtime_service(NvmServicesRuntime *,uint32_t import,uint32_t reference,uint32_t input,uint32_t output);
/* Enforce initializer/entry sequencing and scalar staging, not instruction CFG.
 * VOID initializer uses NO_SLOT; entry uses its exact scalar result root. */
NvmServicesRuntimeStatus nvm_services_runtime_complete_root(NvmServicesRuntime *,uint32_t result);
/* Finish drains all roots before core disposal; only a complete clean invocation
 * publishes scalar output. Failed/repeated calls preserve first error/report.
 * A BUSY no-acquisition return does not mutate the active report or free memory. */
NvmServicesRuntimeReport nvm_services_runtime_finish(NvmServicesRuntime *,NvmServicesRuntimeView *);
NvmServicesRuntimeReport nvm_services_runtime_destroy(NvmServicesRuntime **,NvmServicesRuntimeView *);
#endif
