#ifndef NANOISA_SERVICES_FLOW_H
#define NANOISA_SERVICES_FLOW_H
#include "services_nominal.h"

/* Private logical transfer states only. I neither decode CODE nor certify a
 * module, grant host rights, execute a service, or change public admission.
 * Calls sharing declarations/state objects require external serialization.
 * Output storage must be disjoint from module/declaration/state storage.
 * Caller references are balanced; freed pointers must not be reused. */
#define NVM_SERVICES_FLOW_FUNCTIONS 64u
#define NVM_SERVICES_FLOW_LOCALS 256u
#define NVM_SERVICES_FLOW_STACK 256u
#define NVM_SERVICES_FLOW_REFERENCES 256u
#define NVM_SERVICES_FLOW_OWNERS 256u
#define NVM_SERVICES_FLOW_OBLIGATIONS 256u
#define NVM_SERVICES_FLOW_NO_REFERENCE UINT16_MAX
#define NVM_SERVICES_FLOW_BYTES (16u * 1024u * 1024u)
typedef enum {
    NVM_SERVICES_FLOW_OK, NVM_SERVICES_FLOW_INVALID, NVM_SERVICES_FLOW_UNRESOLVED,
    NVM_SERVICES_FLOW_LIMIT, NVM_SERVICES_FLOW_MEMORY
} NvmServicesFlowStatus;
typedef struct NvmServicesFlowDeclarations NvmServicesFlowDeclarations;
typedef struct NvmServicesFlowState NvmServicesFlowState;
typedef struct {
    uint8_t tag, mode;
    uint32_t global_index, catalog_ordinal; /* instance*9+type; scalar uses NO_INDEX */
    NvmServicesCategory category;
} NvmServicesFlowDeclaration;
typedef struct {
    uint16_t parameters, locals;
    uint8_t result_count;
    NvmServicesFlowDeclaration result;
} NvmServicesFlowFunction;
typedef enum {
    NVM_SERVICES_FLOW_ARM_NONE, NVM_SERVICES_FLOW_ARM_UNKNOWN,
    NVM_SERVICES_FLOW_ARM_OK, NVM_SERVICES_FLOW_ARM_ERROR
} NvmServicesFlowArm;
typedef struct {
    NvmServicesFlowDeclaration type;
    bool initialized;
    uint64_t owner; /* Symbolic identity, never a runtime handle/right. */
    NvmServicesFlowArm arm;
} NvmServicesFlowValue;
typedef struct {
    bool live, formal;
    uint16_t local;
    uint64_t owner, identity, region;
    bool shared;
} NvmServicesFlowReference;
typedef struct {
    uint16_t stack, regions, owners, references;
    uint64_t cleanup_obligations;
    uint16_t obligations;
    bool reachable;
} NvmServicesFlowCounts;

/* Pending requirements, not discharged runtime permissions. A logical site is
 * not an instruction PC until a separately checked decoder establishes it. */
typedef enum { NVM_SERVICES_FLOW_SERVICE, NVM_SERVICES_FLOW_CALL } NvmServicesFlowObligationKind;
typedef enum { NVM_SERVICES_FLOW_INPUT_NONE, NVM_SERVICES_FLOW_INPUT_PRESERVED,
               NVM_SERVICES_FLOW_INPUT_CONSUMED } NvmServicesFlowInputState;
enum {
    NVM_SERVICES_FLOW_CHECK_BINDING=1u, NVM_SERVICES_FLOW_CHECK_INVOCATION=2u,
    NVM_SERVICES_FLOW_CHECK_LIVENESS=4u, NVM_SERVICES_FLOW_CHECK_RIGHTS=8u,
    NVM_SERVICES_FLOW_CHECK_BORROW=16u, NVM_SERVICES_FLOW_CHECK_BYTE=32u,
    NVM_SERVICES_FLOW_CHECK_CALLEE=64u, NVM_SERVICES_FLOW_CHECK_CLEANUP=128u,
    NVM_SERVICES_FLOW_CHECK_RESULT=256u, NVM_SERVICES_FLOW_CHECK_ENDPOINT=512u
};
typedef struct {
    NvmServicesFlowObligationKind kind;
    uint32_t site, target, checks, required_rights, acquired_rights;
    uint16_t parameters, owned_inputs, borrowed_inputs;
    uint8_t result_count;
    NvmServicesFlowDeclaration result;
    NvmServicesFlowInputState outcomes[2]; /* Service Ok/Error input disposition. */
} NvmServicesFlowObligation;

/* I copy exact validated metadata; input arrays/bytes must remain immutable
 * during this call. Failures preserve *out. Successful objects borrow nothing
 * from the input module. Unsupported declarations are explicit UNRESOLVED. */
NvmServicesFlowStatus nvm_services_flow_declarations(const NvmModule *, NvmServicesFlowDeclarations **out);
void nvm_services_flow_declarations_free(NvmServicesFlowDeclarations *);
bool nvm_services_flow_function(const NvmServicesFlowDeclarations *, uint32_t, NvmServicesFlowFunction *);
bool nvm_services_flow_declaration(const NvmServicesFlowDeclarations *, uint32_t, uint16_t,
                             NvmServicesFlowDeclaration *);
/* A state holds its own declaration reference; freeing the caller's declaration
 * reference or destroying the input module does not invalidate the state. */
NvmServicesFlowStatus nvm_services_flow_state(NvmServicesFlowDeclarations *, uint32_t, NvmServicesFlowState **out);
void nvm_services_flow_state_free(NvmServicesFlowState *);
bool nvm_services_flow_local(const NvmServicesFlowState *, uint16_t, NvmServicesFlowValue *);
bool nvm_services_flow_stack(const NvmServicesFlowState *, uint16_t, NvmServicesFlowValue *);
bool nvm_services_flow_reference(const NvmServicesFlowState *, uint16_t, NvmServicesFlowReference *);
bool nvm_services_flow_counts(const NvmServicesFlowState *, NvmServicesFlowCounts *);
bool nvm_services_flow_obligation(const NvmServicesFlowState *, uint16_t, NvmServicesFlowObligation *);
NvmServicesFlowStatus nvm_services_flow_clone(const NvmServicesFlowState *, NvmServicesFlowState **out);
/* Outputs must be distinct; both remain untouched on any failure. */
NvmServicesFlowStatus nvm_services_flow_refine(const NvmServicesFlowState *, uint16_t local,
                                     NvmServicesFlowState **ok, NvmServicesFlowState **error);
NvmServicesFlowStatus nvm_services_flow_join(NvmServicesFlowState *, const NvmServicesFlowState *, bool *changed);

/* Every failed transition preserves state. Scalars are exact INT/BOOL/VOID.
 * Generic loads/stores/dup/drop cannot copy, overwrite or lose an owner. */
NvmServicesFlowStatus nvm_services_flow_push_scalar(NvmServicesFlowState *, uint8_t);
NvmServicesFlowStatus nvm_services_flow_load(NvmServicesFlowState *, uint16_t);
NvmServicesFlowStatus nvm_services_flow_store(NvmServicesFlowState *, uint16_t);
NvmServicesFlowStatus nvm_services_flow_dup(NvmServicesFlowState *);
NvmServicesFlowStatus nvm_services_flow_pop(NvmServicesFlowState *);
NvmServicesFlowStatus nvm_services_flow_clear_copy(NvmServicesFlowState *, uint16_t);
/* Exact instance-specific error/ReadByte/Endpoint/scalar-Result constructors only. Record variant is0;
 * each selected scalar-Result arm consumes one exact payload, including VOID. */
NvmServicesFlowStatus nvm_services_flow_construct(NvmServicesFlowState *, uint32_t type_id, uint16_t variant);
NvmServicesFlowStatus nvm_services_flow_field(NvmServicesFlowState *, uint16_t field);
NvmServicesFlowStatus nvm_services_flow_take_result(NvmServicesFlowState *, uint16_t local, NvmServicesFlowArm);
/* I select the exact instance's catalog method. Byte I/O/progress methods use
 * its live exclusive owner reference; byte writes also consume INT. File temp
 * consumes no value; TCP begin-connect consumes that instance's Endpoint and
 * records a pending domain check. Close consumes that instance's owner.
 * Acquiring/consuming methods use NO_REFERENCE. */
NvmServicesFlowStatus nvm_services_flow_service(NvmServicesFlowState *, uint32_t site, uint32_t import,
                                      uint16_t reference);
/* Exact per-parameter reference vector: mode1 uses a live shared slot,
 * mode2 uses a live exclusive slot;
 * mode0 uses NO_REFERENCE and takes a value from the ordered stack suffix.
 * Success records a pending callee-body/cleanup obligation, not a body proof. */
NvmServicesFlowStatus nvm_services_flow_call(NvmServicesFlowState *, uint32_t site, uint32_t function,
                                   const uint16_t *references, uint16_t count);
NvmServicesFlowStatus nvm_services_flow_move(NvmServicesFlowState *, uint16_t from, uint16_t to);
NvmServicesFlowStatus nvm_services_flow_take(NvmServicesFlowState *, uint16_t);
NvmServicesFlowStatus nvm_services_flow_put(NvmServicesFlowState *, uint16_t);
/* Explicit logical drop records cleanup, without calling a host destructor. */
NvmServicesFlowStatus nvm_services_flow_drop_local(NvmServicesFlowState *, uint16_t);
NvmServicesFlowStatus nvm_services_flow_drop_stack(NvmServicesFlowState *);
NvmServicesFlowStatus nvm_services_flow_region_begin(NvmServicesFlowState *);
NvmServicesFlowStatus nvm_services_flow_region_end(NvmServicesFlowState *);
NvmServicesFlowStatus nvm_services_flow_borrow(NvmServicesFlowState *, uint16_t local, uint16_t reference);
NvmServicesFlowStatus nvm_services_flow_borrow_shared(NvmServicesFlowState *,uint16_t local,uint16_t reference);
NvmServicesFlowStatus nvm_services_flow_end_borrow(NvmServicesFlowState *, uint16_t reference);
/* I check the exact declared result on the stack and complete owner/region
 * obligations without consuming it. Borrowed formals stay caller-owned.
 * A successful exit query is not a checked function-body or call certificate. */
NvmServicesFlowStatus nvm_services_flow_can_exit(const NvmServicesFlowState *);
#endif
