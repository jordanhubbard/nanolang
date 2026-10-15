#ifndef NANOISA_SOCKET_RUNTIME_H
#define NANOISA_SOCKET_RUNTIME_H
#include "socket_hosted.h"
#include "../nsi_socket_values.h"

/* Private carrier primitives, not a CODE dispatcher or public execution route.
 * Every operation requires external serialization with the private Socket cores,
 * including construction/destruction. No thread-safety claim. All caller output
 * storage is disjoint from contexts, input spans and other output objects. */
#define NVM_SOCKET_RUNTIME_BYTES (64u*1024u*1024u)
#define NVM_SOCKET_RUNTIME_NO_SLOT UINT32_MAX
#define NVM_SOCKET_RUNTIME_FIELDS 11u
typedef enum { NVM_SOCKET_RUNTIME_VM, NVM_SOCKET_RUNTIME_NATIVE } NvmSocketRuntimeMode;
typedef enum {
    NVM_SOCKET_RUNTIME_OK, NVM_SOCKET_RUNTIME_INVALID, NVM_SOCKET_RUNTIME_UNRESOLVED,
    NVM_SOCKET_RUNTIME_LIMIT, NVM_SOCKET_RUNTIME_MEMORY, NVM_SOCKET_RUNTIME_STATE,
    NVM_SOCKET_RUNTIME_TYPE, NVM_SOCKET_RUNTIME_STALE, NVM_SOCKET_RUNTIME_BORROWED,
    NVM_SOCKET_RUNTIME_ASSERT, NVM_SOCKET_RUNTIME_CLEANUP, NVM_SOCKET_RUNTIME_BUSY
} NvmSocketRuntimeStatus;
typedef struct {
    bool initialized, owning, formal;
    NvmSocketFlowDeclaration type;
    NvmSocketFlowArm arm;
    uint8_t fields;
    int64_t values[NVM_SOCKET_RUNTIME_FIELDS];
} NvmSocketRuntimeView; /* No Socket handle, capability, stream or borrow epoch. */
typedef struct {
    NvmSocketRuntimeStatus status;
    bool acquired;
    NlSocketValueStatus core_status;
    uint32_t function, instruction;
    NlSocketValuesFinish cleanup;
} NvmSocketRuntimeReport;
typedef struct {
    size_t allocation_bound;
    uint32_t values, references, regions;
    uint16_t frames;
} NvmSocketRuntimeStorage;
typedef struct NvmSocketRuntime NvmSocketRuntime;
/* Preparation and arena allocation only; begin creates the fresh Socket core.
 * Failure preserves *out. Success owns the plan and borrows no input bytes. */
NvmSocketRuntimeStatus nvm_socket_runtime_create(const uint8_t *,size_t,NvmSocketRuntimeMode,NvmSocketRuntime **out);
bool nvm_socket_runtime_storage(const NvmSocketRuntime *,NvmSocketRuntimeStorage *);
const NvmSocketHostedPlan *nvm_socket_runtime_plan(const NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_begin(NvmSocketRuntime *);
/* The later matched dispatcher supplies actual instruction sites. This carrier
 * API does not prove its caller followed the checked CFG or discharged masks. */
NvmSocketRuntimeStatus nvm_socket_runtime_site(NvmSocketRuntime *,uint32_t function,uint16_t instruction);
NvmSocketRuntimeStatus nvm_socket_runtime_fail(NvmSocketRuntime *,NvmSocketRuntimeStatus);
bool nvm_socket_runtime_current_root(const NvmSocketRuntime *,uint32_t *);
bool nvm_socket_runtime_view(const NvmSocketRuntime *,uint32_t,NvmSocketRuntimeView *);
/* Destinations must be empty. No operation silently overwrites an owner. */
NvmSocketRuntimeStatus nvm_socket_runtime_scalar(NvmSocketRuntime *,uint32_t,uint8_t,int64_t);
/* I copy counted immutable bytes into bounded invocation storage. Strings are
 * at most 1 MiB, with at most 4096 distinct strings per invocation; repeated
 * equal strings share storage. Copies and returns retain bytes until finish.
 * My view stores an invocation-local identity, never a host pointer. */
NvmSocketRuntimeStatus nvm_socket_runtime_string(NvmSocketRuntime *,uint32_t,const void *,size_t);
NvmSocketRuntimeStatus nvm_socket_runtime_string_literal(NvmSocketRuntime *,uint32_t);
NvmSocketRuntimeStatus nvm_socket_runtime_string_operation(NvmSocketRuntime *,uint8_t,uint32_t,uint32_t,uint32_t);
bool nvm_socket_runtime_string_read(const NvmSocketRuntime *,uint32_t,void *,size_t,size_t *);
NvmSocketRuntimeStatus nvm_socket_runtime_copy(NvmSocketRuntime *,uint32_t,uint32_t);
NvmSocketRuntimeStatus nvm_socket_runtime_move(NvmSocketRuntime *,uint32_t,uint32_t);
NvmSocketRuntimeStatus nvm_socket_runtime_drop(NvmSocketRuntime *,uint32_t);
NvmSocketRuntimeStatus nvm_socket_runtime_construct(NvmSocketRuntime *,uint32_t ordinal,uint16_t variant,const uint32_t *,size_t,uint32_t);
NvmSocketRuntimeStatus nvm_socket_runtime_project(NvmSocketRuntime *,uint32_t,uint16_t,uint32_t);
NvmSocketRuntimeStatus nvm_socket_runtime_result_arm(NvmSocketRuntime *,uint32_t,NvmSocketFlowArm *);
NvmSocketRuntimeStatus nvm_socket_runtime_take(NvmSocketRuntime *,uint32_t,NvmSocketFlowArm,uint32_t);
NvmSocketRuntimeStatus nvm_socket_runtime_region_begin(NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_region_end(NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_borrow(NvmSocketRuntime *,uint32_t owner,uint32_t reference);
NvmSocketRuntimeStatus nvm_socket_runtime_borrow_shared(NvmSocketRuntime *,uint32_t owner,uint32_t reference);
NvmSocketRuntimeStatus nvm_socket_runtime_bind_formal(NvmSocketRuntime *,uint32_t source_reference,uint32_t formal_root,uint32_t formal_reference);
NvmSocketRuntimeStatus nvm_socket_runtime_end_reference(NvmSocketRuntime *,uint32_t);
/* Exact checked module import, with original core receiver/domain semantics.
 * input is Endpoint for begin-connect, INT for send or Conn for close; otherwise NO_SLOT.
 * reference is exclusive for send/finish-connect/receive; otherwise NO_SLOT. */
NvmSocketRuntimeStatus nvm_socket_runtime_service(NvmSocketRuntime *,uint32_t import,uint32_t reference,uint32_t input,uint32_t output);
/* Enforce initializer/entry sequencing and scalar staging, not instruction CFG.
 * VOID initializer uses NO_SLOT; entry uses its exact scalar result root. */
NvmSocketRuntimeStatus nvm_socket_runtime_complete_root(NvmSocketRuntime *,uint32_t result);
/* Finish drains all roots before core disposal; only a complete clean invocation
 * publishes scalar output. Failed/repeated calls preserve first error/report.
 * A BUSY no-acquisition return does not mutate the active report or free memory. */
NvmSocketRuntimeReport nvm_socket_runtime_finish(NvmSocketRuntime *,NvmSocketRuntimeView *);
NvmSocketRuntimeReport nvm_socket_runtime_destroy(NvmSocketRuntime **,NvmSocketRuntimeView *);
#endif
