#ifndef NANOISA_SOCKET_FLOW_H
#define NANOISA_SOCKET_FLOW_H
#include "service_socket_nominal.h"

/* Private logical transfer states only. I neither decode CODE nor certify a
 * module, grant host rights, execute a service, or change public admission.
 * Calls sharing declarations/state objects require external serialization.
 * Output storage must be disjoint from module/declaration/state storage.
 * Caller references are balanced; freed pointers must not be reused. */
#define NVM_SOCKET_FLOW_FUNCTIONS 64u
#define NVM_SOCKET_FLOW_LOCALS 256u
#define NVM_SOCKET_FLOW_STACK 256u
#define NVM_SOCKET_FLOW_REFERENCES 256u
#define NVM_SOCKET_FLOW_OWNERS 256u
#define NVM_SOCKET_FLOW_OBLIGATIONS 256u
#define NVM_SOCKET_FLOW_NO_REFERENCE UINT16_MAX
#define NVM_SOCKET_FLOW_BYTES (16u * 1024u * 1024u)
typedef enum {
    NVM_SOCKET_FLOW_OK, NVM_SOCKET_FLOW_INVALID, NVM_SOCKET_FLOW_UNRESOLVED,
    NVM_SOCKET_FLOW_LIMIT, NVM_SOCKET_FLOW_MEMORY
} NvmSocketFlowStatus;
typedef struct NvmSocketFlowDeclarations NvmSocketFlowDeclarations;
typedef struct NvmSocketFlowState NvmSocketFlowState;
typedef struct {
    uint8_t tag, mode;
    uint32_t global_index, catalog_ordinal;
    NvmSocketCategory category;
} NvmSocketFlowDeclaration;
typedef struct {
    uint16_t parameters, locals;
    uint8_t result_count;
    NvmSocketFlowDeclaration result;
} NvmSocketFlowFunction;
typedef enum {
    NVM_SOCKET_FLOW_ARM_NONE, NVM_SOCKET_FLOW_ARM_UNKNOWN,
    NVM_SOCKET_FLOW_ARM_OK, NVM_SOCKET_FLOW_ARM_ERROR
} NvmSocketFlowArm;
typedef struct {
    NvmSocketFlowDeclaration type;
    bool initialized;
    uint64_t owner; /* Symbolic identity, never a runtime handle/right. */
    NvmSocketFlowArm arm;
} NvmSocketFlowValue;
typedef struct {
    bool live, formal;
    uint16_t local;
    uint64_t owner, identity, region;
    bool shared;
} NvmSocketFlowReference;
typedef struct {
    uint16_t stack, regions, owners, references;
    uint64_t cleanup_obligations;
    uint16_t obligations;
    bool reachable;
} NvmSocketFlowCounts;

/* Pending requirements, not discharged runtime permissions. A logical site is
 * not an instruction PC until a separately checked decoder establishes it. */
typedef enum { NVM_SOCKET_FLOW_SERVICE, NVM_SOCKET_FLOW_CALL } NvmSocketFlowObligationKind;
typedef enum { NVM_SOCKET_FLOW_INPUT_NONE, NVM_SOCKET_FLOW_INPUT_PRESERVED,
               NVM_SOCKET_FLOW_INPUT_CONSUMED } NvmSocketFlowInputState;
enum {
    NVM_SOCKET_FLOW_CHECK_BINDING=1u, NVM_SOCKET_FLOW_CHECK_INVOCATION=2u,
    NVM_SOCKET_FLOW_CHECK_LIVENESS=4u, NVM_SOCKET_FLOW_CHECK_RIGHTS=8u,
    NVM_SOCKET_FLOW_CHECK_BORROW=16u, NVM_SOCKET_FLOW_CHECK_BYTE=32u,
    NVM_SOCKET_FLOW_CHECK_CALLEE=64u, NVM_SOCKET_FLOW_CHECK_CLEANUP=128u,
    NVM_SOCKET_FLOW_CHECK_RESULT=256u, NVM_SOCKET_FLOW_CHECK_ENDPOINT=512u
};
typedef struct {
    NvmSocketFlowObligationKind kind;
    uint32_t site, target, checks, required_rights, acquired_rights;
    uint16_t parameters, owned_inputs, borrowed_inputs;
    uint8_t result_count;
    NvmSocketFlowDeclaration result;
    NvmSocketFlowInputState outcomes[2]; /* Service Ok/Error input disposition. */
} NvmSocketFlowObligation;

/* I copy exact validated metadata; input arrays/bytes must remain immutable
 * during this call. Failures preserve *out. Successful objects borrow nothing
 * from the input module. Unsupported declarations are explicit UNRESOLVED. */
NvmSocketFlowStatus nvm_socket_flow_declarations(const NvmModule *, NvmSocketFlowDeclarations **out);
void nvm_socket_flow_declarations_free(NvmSocketFlowDeclarations *);
bool nvm_socket_flow_function(const NvmSocketFlowDeclarations *, uint32_t, NvmSocketFlowFunction *);
bool nvm_socket_flow_declaration(const NvmSocketFlowDeclarations *, uint32_t, uint16_t,
                             NvmSocketFlowDeclaration *);
/* A state holds its own declaration reference; freeing the caller's declaration
 * reference or destroying the input module does not invalidate the state. */
NvmSocketFlowStatus nvm_socket_flow_state(NvmSocketFlowDeclarations *, uint32_t, NvmSocketFlowState **out);
void nvm_socket_flow_state_free(NvmSocketFlowState *);
bool nvm_socket_flow_local(const NvmSocketFlowState *, uint16_t, NvmSocketFlowValue *);
bool nvm_socket_flow_stack(const NvmSocketFlowState *, uint16_t, NvmSocketFlowValue *);
bool nvm_socket_flow_reference(const NvmSocketFlowState *, uint16_t, NvmSocketFlowReference *);
bool nvm_socket_flow_counts(const NvmSocketFlowState *, NvmSocketFlowCounts *);
bool nvm_socket_flow_obligation(const NvmSocketFlowState *, uint16_t, NvmSocketFlowObligation *);
NvmSocketFlowStatus nvm_socket_flow_clone(const NvmSocketFlowState *, NvmSocketFlowState **out);
/* Outputs must be distinct; both remain untouched on any failure. */
NvmSocketFlowStatus nvm_socket_flow_refine(const NvmSocketFlowState *, uint16_t local,
                                     NvmSocketFlowState **ok, NvmSocketFlowState **error);
NvmSocketFlowStatus nvm_socket_flow_join(NvmSocketFlowState *, const NvmSocketFlowState *, bool *changed);

/* Every failed transition preserves state. Scalars are exact INT/BOOL/VOID.
 * Generic loads/stores/dup/drop cannot copy, overwrite or lose an owner. */
NvmSocketFlowStatus nvm_socket_flow_push_scalar(NvmSocketFlowState *, uint8_t);
NvmSocketFlowStatus nvm_socket_flow_load(NvmSocketFlowState *, uint16_t);
NvmSocketFlowStatus nvm_socket_flow_store(NvmSocketFlowState *, uint16_t);
NvmSocketFlowStatus nvm_socket_flow_dup(NvmSocketFlowState *);
NvmSocketFlowStatus nvm_socket_flow_pop(NvmSocketFlowState *);
NvmSocketFlowStatus nvm_socket_flow_clear_copy(NvmSocketFlowState *, uint16_t);
/* Exact SocketError/ReadByte/Endpoint/scalar-Result constructors only. Record variant is0;
 * each selected scalar-Result arm consumes one exact payload, including VOID. */
NvmSocketFlowStatus nvm_socket_flow_construct(NvmSocketFlowState *, uint32_t catalog, uint16_t variant);
NvmSocketFlowStatus nvm_socket_flow_field(NvmSocketFlowState *, uint16_t field);
NvmSocketFlowStatus nvm_socket_flow_take_result(NvmSocketFlowState *, uint16_t local, NvmSocketFlowArm);
/* I require a live exclusive reference for send/completion/receive. Send also
 * consumes an INT. Begin-connect consumes Endpoint and records a pending domain
 * check; close consumes Conn. These two use NO_REFERENCE. */
NvmSocketFlowStatus nvm_socket_flow_service(NvmSocketFlowState *, uint32_t site, uint32_t import,
                                      uint16_t reference);
/* Exact per-parameter reference vector: mode1 uses a live shared slot,
 * mode2 uses a live exclusive slot;
 * mode0 uses NO_REFERENCE and takes a value from the ordered stack suffix.
 * Success records a pending callee-body/cleanup obligation, not a body proof. */
NvmSocketFlowStatus nvm_socket_flow_call(NvmSocketFlowState *, uint32_t site, uint32_t function,
                                   const uint16_t *references, uint16_t count);
NvmSocketFlowStatus nvm_socket_flow_move(NvmSocketFlowState *, uint16_t from, uint16_t to);
NvmSocketFlowStatus nvm_socket_flow_take(NvmSocketFlowState *, uint16_t);
NvmSocketFlowStatus nvm_socket_flow_put(NvmSocketFlowState *, uint16_t);
/* Explicit logical drop records cleanup, without calling a host destructor. */
NvmSocketFlowStatus nvm_socket_flow_drop_local(NvmSocketFlowState *, uint16_t);
NvmSocketFlowStatus nvm_socket_flow_drop_stack(NvmSocketFlowState *);
NvmSocketFlowStatus nvm_socket_flow_region_begin(NvmSocketFlowState *);
NvmSocketFlowStatus nvm_socket_flow_region_end(NvmSocketFlowState *);
NvmSocketFlowStatus nvm_socket_flow_borrow(NvmSocketFlowState *, uint16_t local, uint16_t reference);
NvmSocketFlowStatus nvm_socket_flow_borrow_shared(NvmSocketFlowState *,uint16_t local,uint16_t reference);
NvmSocketFlowStatus nvm_socket_flow_end_borrow(NvmSocketFlowState *, uint16_t reference);
/* I check the exact declared result on the stack and complete owner/region
 * obligations without consuming it. Borrowed formals stay caller-owned.
 * A successful exit query is not a checked function-body or call certificate. */
NvmSocketFlowStatus nvm_socket_flow_can_exit(const NvmSocketFlowState *);
#endif
