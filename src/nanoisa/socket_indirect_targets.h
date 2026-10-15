#ifndef NANOISA_SOCKET_INDIRECT_TARGETS_H
#define NANOISA_SOCKET_INDIRECT_TARGETS_H
#include "socket_code.h"
#define NVM_SOCKET_INDIRECT_VISITS 262144u
#define NVM_SOCKET_INDIRECT_BYTES (16u * 1024u * 1024u)
typedef struct NvmSocketIndirectTargets NvmSocketIndirectTargets;
typedef enum { NVM_SOCKET_INDIRECT_DESCRIBED, NVM_SOCKET_INDIRECT_INVALID,
 NVM_SOCKET_INDIRECT_UNRESOLVED, NVM_SOCKET_INDIRECT_LIMIT,
 NVM_SOCKET_INDIRECT_MEMORY } NvmSocketIndirectStatus;
typedef struct {
 NvmSocketIndirectStatus status;
 uint32_t function,pc; /* Original function-relative byte PC, or NO_INDEX. */
 const char *message; /* Static storage. */
} NvmSocketIndirectResult;
typedef struct {
 uint32_t functions,instructions,calls,visits;
 size_t storage_bound;
} NvmSocketIndirectSummary;
typedef struct {
 uint32_t function,pc;
 uint16_t instruction,parameters;
 uint64_t candidates; /* Bit f is original same-module function f. */
 uint8_t result_count;
 NvmSocketFlowDeclaration result;
} NvmSocketIndirectCall;
/* Private descriptive target facts only, NEVER body/ownership/runtime authority.
 * Inputs are valid immutable storage during preparation. The report owns all
 * facts and survives input destruction. Outputs must be disjoint and valid.
 * Only DESCRIBED publishes. Failure and invalid getters preserve outputs.
 * Calls require external serialization, as do the underlying ISA queries. */
NvmSocketIndirectResult nvm_socket_indirect_targets(const NvmModule *,NvmSocketIndirectTargets **);
void nvm_socket_indirect_targets_free(NvmSocketIndirectTargets *);
bool nvm_socket_indirect_targets_summary(const NvmSocketIndirectTargets *,NvmSocketIndirectSummary *);
bool nvm_socket_indirect_targets_call(const NvmSocketIndirectTargets *,uint32_t,NvmSocketIndirectCall *);
bool nvm_socket_indirect_targets_parameter(const NvmSocketIndirectTargets *,uint32_t,uint16_t,NvmSocketFlowDeclaration *);
#endif
