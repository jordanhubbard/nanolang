#ifndef NANOISA_OWNED_ARRAY_AUTHORITY_H
#define NANOISA_OWNED_ARRAY_AUTHORITY_H
#include "owned_array_origins.h"
/* Private complete query, never runtime permission or a caller-built proof. */
typedef struct NvmOwnedArrayPlan NvmOwnedArrayPlan;
typedef enum { NVM_OWNER_AUTH_PREPARED, NVM_OWNER_AUTH_UNRESOLVED,
    NVM_OWNER_AUTH_INVALID, NVM_OWNER_AUTH_LIMIT, NVM_OWNER_AUTH_MEMORY
} NvmOwnerAuthorityStatus;
typedef struct { NvmOwnerAuthorityStatus status; uint32_t function, pc; const char *message; } NvmOwnerAuthorityResult;
typedef struct { uint8_t tag, mode; uint32_t global_layout; bool owner; } NvmOwnerDeclaration;
typedef struct {
    uint16_t local_count, parameter_count, max_stack, result_fields;
    NvmOwnerDeclaration result, parameters[8];
} NvmOwnerSignature;
typedef struct { uint32_t functions, instructions, persisted_cells, visits, work; } NvmOwnerAuthorityCounts;
/* Construction borrows immutable module storage only during this call. The
 * result owns facts/transport. Failure preserves every caller output. */
NvmOwnerAuthorityResult nvm_prepare_owned_array_authority(const NvmModule *, NvmOwnedArrayPlan **);
void nvm_owned_array_plan_free(NvmOwnedArrayPlan *);
bool nvm_owned_array_plan_counts(const NvmOwnedArrayPlan *, NvmOwnerAuthorityCounts *);
bool nvm_owned_array_plan_signature(const NvmOwnedArrayPlan *, uint32_t, NvmOwnerSignature *);
bool nvm_owned_array_plan_local(const NvmOwnedArrayPlan *, uint32_t, uint16_t, NvmOwnerDeclaration *);
bool nvm_owned_array_plan_layout(const NvmOwnedArrayPlan *, uint32_t, NvmOwnedArrayLayoutFact *);
bool nvm_owned_array_plan_source(const NvmOwnedArrayPlan *, uint32_t, uint32_t *);
bool nvm_owned_array_plan_transport(const NvmOwnedArrayPlan *, NvmOwnedArrayTransport *);
bool nvm_owned_array_plan_obligation(const NvmOwnedArrayPlan *, uint32_t, NvmOwnerOriginObligation *);
#endif
