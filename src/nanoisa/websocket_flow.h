#ifndef NANOISA_WEBSOCKET_FLOW_H
#define NANOISA_WEBSOCKET_FLOW_H
#include "service_websocket_nominal.h"

/* Private logical transfer states only. I neither decode CODE nor certify a
 * module, grant host rights, execute a service, or change public admission.
 * Calls sharing declarations/state objects require external serialization.
 * Output storage must be disjoint from module/declaration/state storage.
 * Caller references are balanced; freed pointers must not be reused. */
#define NVM_WEBSOCKET_FLOW_FUNCTIONS 64u
#define NVM_WEBSOCKET_FLOW_LOCALS 256u
#define NVM_WEBSOCKET_FLOW_STACK 256u
#define NVM_WEBSOCKET_FLOW_REFERENCES 256u
#define NVM_WEBSOCKET_FLOW_OWNERS 256u
#define NVM_WEBSOCKET_FLOW_OBLIGATIONS 256u
#define NVM_WEBSOCKET_FLOW_NO_REFERENCE UINT16_MAX
#define NVM_WEBSOCKET_FLOW_BYTES (16u * 1024u * 1024u)
typedef enum {
    NVM_WEBSOCKET_FLOW_OK, NVM_WEBSOCKET_FLOW_INVALID, NVM_WEBSOCKET_FLOW_UNRESOLVED,
    NVM_WEBSOCKET_FLOW_LIMIT, NVM_WEBSOCKET_FLOW_MEMORY
} NvmWebSocketFlowStatus;
typedef struct NvmWebSocketFlowDeclarations NvmWebSocketFlowDeclarations;
typedef struct NvmWebSocketFlowState NvmWebSocketFlowState;
typedef struct {
    uint8_t tag, mode;
    uint32_t global_index, catalog_ordinal;
    NvmWebSocketCategory category;
} NvmWebSocketFlowDeclaration;
typedef struct {
    uint16_t parameters, locals;
    uint8_t result_count;
    NvmWebSocketFlowDeclaration result;
} NvmWebSocketFlowFunction;
typedef enum {
    NVM_WEBSOCKET_FLOW_ARM_NONE, NVM_WEBSOCKET_FLOW_ARM_UNKNOWN,
    NVM_WEBSOCKET_FLOW_ARM_OK, NVM_WEBSOCKET_FLOW_ARM_ERROR
} NvmWebSocketFlowArm;
typedef struct {
    NvmWebSocketFlowDeclaration type;
    bool initialized;
    uint64_t owner; /* Symbolic identity, never a runtime handle/right. */
    NvmWebSocketFlowArm arm;
} NvmWebSocketFlowValue;
typedef struct {
    bool live, formal;
    uint16_t local;
    uint64_t owner, identity, region;
    bool shared;
} NvmWebSocketFlowReference;
typedef struct {
    uint16_t stack, regions, owners, references;
    uint64_t cleanup_obligations;
    uint16_t obligations;
    bool reachable;
} NvmWebSocketFlowCounts;

/* Pending requirements, not discharged runtime permissions. A logical site is
 * not an instruction PC until a separately checked decoder establishes it. */
typedef enum { NVM_WEBSOCKET_FLOW_SERVICE, NVM_WEBSOCKET_FLOW_CALL } NvmWebSocketFlowObligationKind;
typedef enum { NVM_WEBSOCKET_FLOW_INPUT_NONE, NVM_WEBSOCKET_FLOW_INPUT_PRESERVED,
               NVM_WEBSOCKET_FLOW_INPUT_CONSUMED } NvmWebSocketFlowInputState;
enum {
    NVM_WEBSOCKET_FLOW_CHECK_BINDING=1u, NVM_WEBSOCKET_FLOW_CHECK_INVOCATION=2u,
    NVM_WEBSOCKET_FLOW_CHECK_LIVENESS=4u, NVM_WEBSOCKET_FLOW_CHECK_RIGHTS=8u,
    NVM_WEBSOCKET_FLOW_CHECK_BORROW=16u, NVM_WEBSOCKET_FLOW_CHECK_BYTE=32u,
    NVM_WEBSOCKET_FLOW_CHECK_CALLEE=64u, NVM_WEBSOCKET_FLOW_CHECK_CLEANUP=128u,
    NVM_WEBSOCKET_FLOW_CHECK_RESULT=256u, NVM_WEBSOCKET_FLOW_CHECK_TIMEOUT=1024u
};
typedef struct {
    NvmWebSocketFlowObligationKind kind;
    uint32_t site, target, checks, required_rights, acquired_rights;
    uint16_t parameters, owned_inputs, borrowed_inputs;
    uint8_t result_count;
    NvmWebSocketFlowDeclaration result;
    NvmWebSocketFlowInputState outcomes[2]; /* Service Ok/Error input disposition. */
} NvmWebSocketFlowObligation;

/* I copy exact validated metadata; input arrays/bytes must remain immutable
 * during this call. Failures preserve *out. Successful objects borrow nothing
 * from the input module. Unsupported declarations are explicit UNRESOLVED. */
NvmWebSocketFlowStatus nvm_websocket_flow_declarations(const NvmModule *, NvmWebSocketFlowDeclarations **out);
void nvm_websocket_flow_declarations_free(NvmWebSocketFlowDeclarations *);
bool nvm_websocket_flow_function(const NvmWebSocketFlowDeclarations *, uint32_t, NvmWebSocketFlowFunction *);
bool nvm_websocket_flow_declaration(const NvmWebSocketFlowDeclarations *, uint32_t, uint16_t,
                             NvmWebSocketFlowDeclaration *);
/* A state holds its own declaration reference; freeing the caller's declaration
 * reference or destroying the input module does not invalidate the state. */
NvmWebSocketFlowStatus nvm_websocket_flow_state(NvmWebSocketFlowDeclarations *, uint32_t, NvmWebSocketFlowState **out);
void nvm_websocket_flow_state_free(NvmWebSocketFlowState *);
bool nvm_websocket_flow_local(const NvmWebSocketFlowState *, uint16_t, NvmWebSocketFlowValue *);
bool nvm_websocket_flow_stack(const NvmWebSocketFlowState *, uint16_t, NvmWebSocketFlowValue *);
bool nvm_websocket_flow_reference(const NvmWebSocketFlowState *, uint16_t, NvmWebSocketFlowReference *);
bool nvm_websocket_flow_counts(const NvmWebSocketFlowState *, NvmWebSocketFlowCounts *);
bool nvm_websocket_flow_obligation(const NvmWebSocketFlowState *, uint16_t, NvmWebSocketFlowObligation *);
NvmWebSocketFlowStatus nvm_websocket_flow_clone(const NvmWebSocketFlowState *, NvmWebSocketFlowState **out);
/* Outputs must be distinct; both remain untouched on any failure. */
NvmWebSocketFlowStatus nvm_websocket_flow_refine(const NvmWebSocketFlowState *, uint16_t local,
                                     NvmWebSocketFlowState **ok, NvmWebSocketFlowState **error);
NvmWebSocketFlowStatus nvm_websocket_flow_join(NvmWebSocketFlowState *, const NvmWebSocketFlowState *, bool *changed);

/* Every failed transition preserves state. Scalars are exact INT/BOOL/VOID/STRING.
 * Generic loads/stores/dup/drop cannot copy, overwrite or lose an owner. */
NvmWebSocketFlowStatus nvm_websocket_flow_push_scalar(NvmWebSocketFlowState *, uint8_t);
NvmWebSocketFlowStatus nvm_websocket_flow_load(NvmWebSocketFlowState *, uint16_t);
NvmWebSocketFlowStatus nvm_websocket_flow_store(NvmWebSocketFlowState *, uint16_t);
NvmWebSocketFlowStatus nvm_websocket_flow_dup(NvmWebSocketFlowState *);
NvmWebSocketFlowStatus nvm_websocket_flow_pop(NvmWebSocketFlowState *);
NvmWebSocketFlowStatus nvm_websocket_flow_clear_copy(NvmWebSocketFlowState *, uint16_t);
/* I construct exact WebSocketError/Message and passive Result values. Record
 * variant is zero; a Result arm consumes one exact payload, including VOID. */
NvmWebSocketFlowStatus nvm_websocket_flow_construct(NvmWebSocketFlowState *, uint32_t catalog, uint16_t variant);
NvmWebSocketFlowStatus nvm_websocket_flow_field(NvmWebSocketFlowState *, uint16_t field);
NvmWebSocketFlowStatus nvm_websocket_flow_take_result(NvmWebSocketFlowState *, uint16_t local, NvmWebSocketFlowArm);
/* I check the catalog's ordered arguments. Send/receive use an exclusive
 * reference; connect/close use NO_REFERENCE. Close consumes Connection followed
 * by timeout. Timeout ranges remain explicit runtime obligations. */
NvmWebSocketFlowStatus nvm_websocket_flow_service(NvmWebSocketFlowState *, uint32_t site, uint32_t import,
                                      uint16_t reference);
/* Exact per-parameter reference vector: mode1 uses a live shared slot,
 * mode2 uses a live exclusive slot;
 * mode0 uses NO_REFERENCE and takes a value from the ordered stack suffix.
 * Success records a pending callee-body/cleanup obligation, not a body proof. */
NvmWebSocketFlowStatus nvm_websocket_flow_call(NvmWebSocketFlowState *, uint32_t site, uint32_t function,
                                   const uint16_t *references, uint16_t count);
NvmWebSocketFlowStatus nvm_websocket_flow_move(NvmWebSocketFlowState *, uint16_t from, uint16_t to);
NvmWebSocketFlowStatus nvm_websocket_flow_take(NvmWebSocketFlowState *, uint16_t);
NvmWebSocketFlowStatus nvm_websocket_flow_put(NvmWebSocketFlowState *, uint16_t);
/* Explicit logical drop records cleanup, without calling a host destructor. */
NvmWebSocketFlowStatus nvm_websocket_flow_drop_local(NvmWebSocketFlowState *, uint16_t);
NvmWebSocketFlowStatus nvm_websocket_flow_drop_stack(NvmWebSocketFlowState *);
NvmWebSocketFlowStatus nvm_websocket_flow_region_begin(NvmWebSocketFlowState *);
NvmWebSocketFlowStatus nvm_websocket_flow_region_end(NvmWebSocketFlowState *);
NvmWebSocketFlowStatus nvm_websocket_flow_borrow(NvmWebSocketFlowState *, uint16_t local, uint16_t reference);
NvmWebSocketFlowStatus nvm_websocket_flow_borrow_shared(NvmWebSocketFlowState *,uint16_t local,uint16_t reference);
NvmWebSocketFlowStatus nvm_websocket_flow_end_borrow(NvmWebSocketFlowState *, uint16_t reference);
/* I check the exact declared result on the stack and complete owner/region
 * obligations without consuming it. Borrowed formals stay caller-owned.
 * A successful exit query is not a checked function-body or call certificate. */
NvmWebSocketFlowStatus nvm_websocket_flow_can_exit(const NvmWebSocketFlowState *);
#endif
