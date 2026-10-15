#ifndef NANOISA_WEBSOCKET_RUNTIME_H
#define NANOISA_WEBSOCKET_RUNTIME_H
#include "websocket_hosted.h"
#include "../nsi_websocket_values.h"

/* Private carrier primitives, not a CODE dispatcher or public execution route.
 * Every operation requires external serialization with the private WebSocket cores,
 * including construction/destruction. No thread-safety claim. All caller output
 * storage is disjoint from contexts, input spans and other output objects. */
#define NVM_WEBSOCKET_RUNTIME_BYTES (64u*1024u*1024u)
#define NVM_WEBSOCKET_RUNTIME_NO_SLOT UINT32_MAX
#define NVM_WEBSOCKET_RUNTIME_FIELDS 9u
typedef enum { NVM_WEBSOCKET_RUNTIME_VM, NVM_WEBSOCKET_RUNTIME_NATIVE } NvmWebSocketRuntimeMode;
typedef enum {
    NVM_WEBSOCKET_RUNTIME_OK, NVM_WEBSOCKET_RUNTIME_INVALID, NVM_WEBSOCKET_RUNTIME_UNRESOLVED,
    NVM_WEBSOCKET_RUNTIME_LIMIT, NVM_WEBSOCKET_RUNTIME_MEMORY, NVM_WEBSOCKET_RUNTIME_STATE,
    NVM_WEBSOCKET_RUNTIME_TYPE, NVM_WEBSOCKET_RUNTIME_STALE, NVM_WEBSOCKET_RUNTIME_BORROWED,
    NVM_WEBSOCKET_RUNTIME_ASSERT, NVM_WEBSOCKET_RUNTIME_CLEANUP, NVM_WEBSOCKET_RUNTIME_BUSY
} NvmWebSocketRuntimeStatus;
typedef struct {
    bool initialized, owning, formal;
    NvmWebSocketFlowDeclaration type;
    NvmWebSocketFlowArm arm;
    uint8_t fields;
    int64_t values[NVM_WEBSOCKET_RUNTIME_FIELDS];
} NvmWebSocketRuntimeView; /* No WebSocket handle, capability, stream or borrow epoch. */
typedef struct {
    NvmWebSocketRuntimeStatus status;
    bool acquired;
    NlWsValueStatus core_status;
    uint32_t function, instruction;
    NlWsValuesFinish cleanup;
} NvmWebSocketRuntimeReport;
typedef struct {
    size_t allocation_bound;
    uint32_t values, references, regions;
    uint16_t frames;
} NvmWebSocketRuntimeStorage;
typedef struct NvmWebSocketRuntime NvmWebSocketRuntime;
/* Preparation and arena allocation only; begin creates the fresh WebSocket core.
 * Failure preserves *out. Success owns the plan and borrows no input bytes. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_create(const uint8_t *,size_t,NvmWebSocketRuntimeMode,NvmWebSocketRuntime **out);
bool nvm_websocket_runtime_storage(const NvmWebSocketRuntime *,NvmWebSocketRuntimeStorage *);
const NvmWebSocketHostedPlan *nvm_websocket_runtime_plan(const NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_begin(NvmWebSocketRuntime *);
/* The later matched dispatcher supplies actual instruction sites. This carrier
 * API does not prove its caller followed the checked CFG or discharged masks. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_site(NvmWebSocketRuntime *,uint32_t function,uint16_t instruction);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_fail(NvmWebSocketRuntime *,NvmWebSocketRuntimeStatus);
bool nvm_websocket_runtime_current_root(const NvmWebSocketRuntime *,uint32_t *);
bool nvm_websocket_runtime_view(const NvmWebSocketRuntime *,uint32_t,NvmWebSocketRuntimeView *);
/* Destinations must be empty. No operation silently overwrites an owner. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_scalar(NvmWebSocketRuntime *,uint32_t,uint8_t,int64_t);
/* I copy counted immutable bytes into bounded invocation storage. Strings are
 * at most 1 MiB, with at most 4096 distinct strings per invocation; repeated
 * equal strings share storage. Copies and returns retain bytes until finish.
 * My view stores an invocation-local identity, never a host pointer. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_string(NvmWebSocketRuntime *,uint32_t,const void *,size_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_string_literal(NvmWebSocketRuntime *,uint32_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_string_operation(NvmWebSocketRuntime *,uint8_t,uint32_t,uint32_t,uint32_t);
bool nvm_websocket_runtime_string_read(const NvmWebSocketRuntime *,uint32_t,void *,size_t,size_t *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_copy(NvmWebSocketRuntime *,uint32_t,uint32_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_move(NvmWebSocketRuntime *,uint32_t,uint32_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_drop(NvmWebSocketRuntime *,uint32_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_construct(NvmWebSocketRuntime *,uint32_t ordinal,uint16_t variant,const uint32_t *,size_t,uint32_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_project(NvmWebSocketRuntime *,uint32_t,uint16_t,uint32_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_result_arm(NvmWebSocketRuntime *,uint32_t,NvmWebSocketFlowArm *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_take(NvmWebSocketRuntime *,uint32_t,NvmWebSocketFlowArm,uint32_t);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_region_begin(NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_region_end(NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_borrow(NvmWebSocketRuntime *,uint32_t owner,uint32_t reference);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_borrow_shared(NvmWebSocketRuntime *,uint32_t owner,uint32_t reference);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_bind_formal(NvmWebSocketRuntime *,uint32_t source_reference,uint32_t formal_root,uint32_t formal_reference);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_end_reference(NvmWebSocketRuntime *,uint32_t);
/* I copy explicit host policy before begin; preparation grants no authority. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_policy(NvmWebSocketRuntime *,const NlWsTransportPolicy *);
/* I consume ordered copied arguments, plus the Connection for consuming close.
 * Exclusive receivers use reference; other calls require NO_SLOT. Timeout is
 * always the last argument. Host errors publish Result and preserve borrows. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_service(NvmWebSocketRuntime *,uint32_t import,uint32_t reference,const uint32_t *inputs,size_t count,uint32_t output);
/* Enforce initializer/entry sequencing and scalar staging, not instruction CFG.
 * VOID initializer uses NO_SLOT; entry uses its exact scalar result root. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_complete_root(NvmWebSocketRuntime *,uint32_t result);
/* Finish drains all roots before core disposal; only a complete clean invocation
 * publishes scalar output. Failed/repeated calls preserve first error/report.
 * A BUSY no-acquisition return does not mutate the active report or free memory. */
NvmWebSocketRuntimeReport nvm_websocket_runtime_finish(NvmWebSocketRuntime *,NvmWebSocketRuntimeView *);
NvmWebSocketRuntimeReport nvm_websocket_runtime_destroy(NvmWebSocketRuntime **,NvmWebSocketRuntimeView *);
#endif
