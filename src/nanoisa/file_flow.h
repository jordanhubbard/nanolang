#ifndef NANOISA_FILE_FLOW_H
#define NANOISA_FILE_FLOW_H
#include "service_file_nominal.h"

/* Private logical transfer states only. I neither decode CODE nor certify a
 * module, grant host rights, execute a service, or change public admission.
 * Calls sharing declarations/state objects require external serialization.
 * Output storage must be disjoint from module/declaration/state storage.
 * Caller references are balanced; freed pointers must not be reused. */
#define NVM_FILE_FLOW_FUNCTIONS 64u
#define NVM_FILE_FLOW_LOCALS 256u
#define NVM_FILE_FLOW_STACK 256u
#define NVM_FILE_FLOW_REFERENCES 256u
#define NVM_FILE_FLOW_OWNERS 256u
#define NVM_FILE_FLOW_OBLIGATIONS 256u
#define NVM_FILE_FLOW_NO_REFERENCE UINT16_MAX
#define NVM_FILE_FLOW_BYTES (16u * 1024u * 1024u)
typedef enum {
    NVM_FILE_FLOW_OK, NVM_FILE_FLOW_INVALID, NVM_FILE_FLOW_UNRESOLVED,
    NVM_FILE_FLOW_LIMIT, NVM_FILE_FLOW_MEMORY
} NvmFileFlowStatus;
typedef struct NvmFileFlowDeclarations NvmFileFlowDeclarations;
typedef struct NvmFileFlowState NvmFileFlowState;
typedef struct {
    uint8_t tag, mode;
    uint32_t global_index, catalog_ordinal;
    NvmFileCategory category;
} NvmFileFlowDeclaration;
typedef struct {
    uint16_t parameters, locals;
    uint8_t result_count;
    NvmFileFlowDeclaration result;
} NvmFileFlowFunction;
typedef enum {
    NVM_FILE_FLOW_ARM_NONE, NVM_FILE_FLOW_ARM_UNKNOWN,
    NVM_FILE_FLOW_ARM_OK, NVM_FILE_FLOW_ARM_ERROR
} NvmFileFlowArm;
typedef struct {
    NvmFileFlowDeclaration type;
    bool initialized;
    uint64_t owner; /* Symbolic identity, never a runtime handle/right. */
    NvmFileFlowArm arm;
} NvmFileFlowValue;
typedef struct {
    bool live, formal;
    uint16_t local;
    uint64_t owner, identity, region;
} NvmFileFlowReference;
typedef struct {
    uint16_t stack, regions, owners, references;
    uint64_t cleanup_obligations;
    uint16_t obligations;
    bool reachable;
} NvmFileFlowCounts;

/* Pending requirements, not discharged runtime permissions. A logical site is
 * not an instruction PC until a separately checked decoder establishes it. */
typedef enum { NVM_FILE_FLOW_SERVICE, NVM_FILE_FLOW_CALL } NvmFileFlowObligationKind;
typedef enum { NVM_FILE_FLOW_INPUT_NONE, NVM_FILE_FLOW_INPUT_PRESERVED,
               NVM_FILE_FLOW_INPUT_CONSUMED } NvmFileFlowInputState;
enum {
    NVM_FILE_FLOW_CHECK_BINDING=1u, NVM_FILE_FLOW_CHECK_INVOCATION=2u,
    NVM_FILE_FLOW_CHECK_LIVENESS=4u, NVM_FILE_FLOW_CHECK_RIGHTS=8u,
    NVM_FILE_FLOW_CHECK_BORROW=16u, NVM_FILE_FLOW_CHECK_BYTE=32u,
    NVM_FILE_FLOW_CHECK_CALLEE=64u, NVM_FILE_FLOW_CHECK_CLEANUP=128u,
    NVM_FILE_FLOW_CHECK_RESULT=256u
};
typedef struct {
    NvmFileFlowObligationKind kind;
    uint32_t site, target, checks, required_rights, acquired_rights;
    uint16_t parameters, owned_inputs, borrowed_inputs;
    uint8_t result_count;
    NvmFileFlowDeclaration result;
    NvmFileFlowInputState outcomes[2]; /* Service Ok/Error input disposition. */
} NvmFileFlowObligation;

/* I copy exact validated metadata; input arrays/bytes must remain immutable
 * during this call. Failures preserve *out. Successful objects borrow nothing
 * from the input module. Unsupported declarations are explicit UNRESOLVED. */
NvmFileFlowStatus nvm_file_flow_declarations(const NvmModule *, NvmFileFlowDeclarations **out);
void nvm_file_flow_declarations_free(NvmFileFlowDeclarations *);
bool nvm_file_flow_function(const NvmFileFlowDeclarations *, uint32_t, NvmFileFlowFunction *);
bool nvm_file_flow_declaration(const NvmFileFlowDeclarations *, uint32_t, uint16_t,
                             NvmFileFlowDeclaration *);
/* A state holds its own declaration reference; freeing the caller's declaration
 * reference or destroying the input module does not invalidate the state. */
NvmFileFlowStatus nvm_file_flow_state(NvmFileFlowDeclarations *, uint32_t, NvmFileFlowState **out);
void nvm_file_flow_state_free(NvmFileFlowState *);
bool nvm_file_flow_local(const NvmFileFlowState *, uint16_t, NvmFileFlowValue *);
bool nvm_file_flow_stack(const NvmFileFlowState *, uint16_t, NvmFileFlowValue *);
bool nvm_file_flow_reference(const NvmFileFlowState *, uint16_t, NvmFileFlowReference *);
bool nvm_file_flow_counts(const NvmFileFlowState *, NvmFileFlowCounts *);
bool nvm_file_flow_obligation(const NvmFileFlowState *, uint16_t, NvmFileFlowObligation *);
NvmFileFlowStatus nvm_file_flow_clone(const NvmFileFlowState *, NvmFileFlowState **out);
/* Outputs must be distinct; both remain untouched on any failure. */
NvmFileFlowStatus nvm_file_flow_refine(const NvmFileFlowState *, uint16_t local,
                                     NvmFileFlowState **ok, NvmFileFlowState **error);
NvmFileFlowStatus nvm_file_flow_join(NvmFileFlowState *, const NvmFileFlowState *, bool *changed);

/* Every failed transition preserves state. Scalars are exact INT/BOOL/VOID.
 * Generic loads/stores/dup/drop cannot copy, overwrite or lose an owner. */
NvmFileFlowStatus nvm_file_flow_push_scalar(NvmFileFlowState *, uint8_t);
NvmFileFlowStatus nvm_file_flow_load(NvmFileFlowState *, uint16_t);
NvmFileFlowStatus nvm_file_flow_store(NvmFileFlowState *, uint16_t);
NvmFileFlowStatus nvm_file_flow_dup(NvmFileFlowState *);
NvmFileFlowStatus nvm_file_flow_pop(NvmFileFlowState *);
NvmFileFlowStatus nvm_file_flow_clear_copy(NvmFileFlowState *, uint16_t);
/* Exact FileError/ReadByte/scalar-Result constructors only. Record variant is0;
 * each selected scalar-Result arm consumes one exact payload, including VOID. */
NvmFileFlowStatus nvm_file_flow_construct(NvmFileFlowState *, uint32_t catalog, uint16_t variant);
NvmFileFlowStatus nvm_file_flow_field(NvmFileFlowState *, uint16_t field);
NvmFileFlowStatus nvm_file_flow_take_result(NvmFileFlowState *, uint16_t local, NvmFileFlowArm);
/* Exclusive services take a reference slot (and write's INT on stack). Temp
 * takes no values; close consumes File from stack. Other modes require NO_REFERENCE. */
NvmFileFlowStatus nvm_file_flow_service(NvmFileFlowState *, uint32_t site, uint32_t import,
                                      uint16_t reference);
/* Exact per-parameter reference vector: mode2 uses a live exclusive slot;
 * mode0 uses NO_REFERENCE and takes a value from the ordered stack suffix.
 * Success records a pending callee-body/cleanup obligation, not a body proof. */
NvmFileFlowStatus nvm_file_flow_call(NvmFileFlowState *, uint32_t site, uint32_t function,
                                   const uint16_t *references, uint16_t count);
NvmFileFlowStatus nvm_file_flow_move(NvmFileFlowState *, uint16_t from, uint16_t to);
NvmFileFlowStatus nvm_file_flow_take(NvmFileFlowState *, uint16_t);
NvmFileFlowStatus nvm_file_flow_put(NvmFileFlowState *, uint16_t);
/* Explicit logical drop records cleanup, without calling a host destructor. */
NvmFileFlowStatus nvm_file_flow_drop_local(NvmFileFlowState *, uint16_t);
NvmFileFlowStatus nvm_file_flow_drop_stack(NvmFileFlowState *);
NvmFileFlowStatus nvm_file_flow_region_begin(NvmFileFlowState *);
NvmFileFlowStatus nvm_file_flow_region_end(NvmFileFlowState *);
NvmFileFlowStatus nvm_file_flow_borrow(NvmFileFlowState *, uint16_t local, uint16_t reference);
NvmFileFlowStatus nvm_file_flow_end_borrow(NvmFileFlowState *, uint16_t reference);
/* I check the exact declared result on the stack and complete owner/region
 * obligations without consuming it. Borrowed formals stay caller-owned.
 * A successful exit query is not a checked function-body or call certificate. */
NvmFileFlowStatus nvm_file_flow_can_exit(const NvmFileFlowState *);
#endif
