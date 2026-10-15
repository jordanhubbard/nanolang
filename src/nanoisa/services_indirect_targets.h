#ifndef NANOISA_SERVICES_INDIRECT_TARGETS_H
#define NANOISA_SERVICES_INDIRECT_TARGETS_H
#include "services_code.h"
#define NVM_SERVICES_INDIRECT_VISITS 262144u
#define NVM_SERVICES_INDIRECT_BYTES (16u * 1024u * 1024u)
typedef struct NvmServicesIndirectTargets NvmServicesIndirectTargets;
typedef enum { NVM_SERVICES_INDIRECT_DESCRIBED, NVM_SERVICES_INDIRECT_INVALID,
 NVM_SERVICES_INDIRECT_UNRESOLVED, NVM_SERVICES_INDIRECT_LIMIT,
 NVM_SERVICES_INDIRECT_MEMORY } NvmServicesIndirectStatus;
typedef struct {
 NvmServicesIndirectStatus status;
 uint32_t function,pc; /* Original function-relative byte PC, or NO_INDEX. */
 const char *message; /* Static storage. */
} NvmServicesIndirectResult;
typedef struct {
 uint32_t functions,instructions,calls,visits;
 size_t storage_bound;
} NvmServicesIndirectSummary;
typedef struct {
 uint32_t function,pc;
 uint16_t instruction,parameters;
 uint64_t candidates; /* Bit f is original same-module function f. */
 uint8_t result_count;
 NvmServicesFlowDeclaration result;
} NvmServicesIndirectCall;
/* Private descriptive target facts only, NEVER body/ownership/runtime authority.
 * Inputs are valid immutable storage during preparation. The report owns all
 * facts and survives input destruction. Outputs must be disjoint and valid.
 * Only DESCRIBED publishes. Failure and invalid getters preserve outputs.
 * Calls require external serialization, as do the underlying ISA queries. */
NvmServicesIndirectResult nvm_services_indirect_targets(const NvmModule *,NvmServicesIndirectTargets **);
void nvm_services_indirect_targets_free(NvmServicesIndirectTargets *);
bool nvm_services_indirect_targets_summary(const NvmServicesIndirectTargets *,NvmServicesIndirectSummary *);
bool nvm_services_indirect_targets_call(const NvmServicesIndirectTargets *,uint32_t,NvmServicesIndirectCall *);
bool nvm_services_indirect_targets_parameter(const NvmServicesIndirectTargets *,uint32_t,uint16_t,NvmServicesFlowDeclaration *);
#endif
