#ifndef NANOISA_FILE_INDIRECT_TARGETS_H
#define NANOISA_FILE_INDIRECT_TARGETS_H
#include "file_code.h"
#define NVM_FILE_INDIRECT_VISITS 262144u
#define NVM_FILE_INDIRECT_BYTES (16u * 1024u * 1024u)
typedef struct NvmFileIndirectTargets NvmFileIndirectTargets;
typedef enum { NVM_FILE_INDIRECT_DESCRIBED, NVM_FILE_INDIRECT_INVALID,
 NVM_FILE_INDIRECT_UNRESOLVED, NVM_FILE_INDIRECT_LIMIT,
 NVM_FILE_INDIRECT_MEMORY } NvmFileIndirectStatus;
typedef struct {
 NvmFileIndirectStatus status;
 uint32_t function,pc; /* Original function-relative byte PC, or NO_INDEX. */
 const char *message; /* Static storage. */
} NvmFileIndirectResult;
typedef struct {
 uint32_t functions,instructions,calls,visits;
 size_t storage_bound;
} NvmFileIndirectSummary;
typedef struct {
 uint32_t function,pc;
 uint16_t instruction,parameters;
 uint64_t candidates; /* Bit f is original same-module function f. */
 uint8_t result_count;
 NvmFileFlowDeclaration result;
} NvmFileIndirectCall;
/* Private descriptive target facts only, NEVER body/ownership/runtime authority.
 * Inputs are valid immutable storage during preparation. The report owns all
 * facts and survives input destruction. Outputs must be disjoint and valid.
 * Only DESCRIBED publishes. Failure and invalid getters preserve outputs.
 * Calls require external serialization, as do the underlying ISA queries. */
NvmFileIndirectResult nvm_file_indirect_targets(const NvmModule *,NvmFileIndirectTargets **);
void nvm_file_indirect_targets_free(NvmFileIndirectTargets *);
bool nvm_file_indirect_targets_summary(const NvmFileIndirectTargets *,NvmFileIndirectSummary *);
bool nvm_file_indirect_targets_call(const NvmFileIndirectTargets *,uint32_t,NvmFileIndirectCall *);
bool nvm_file_indirect_targets_parameter(const NvmFileIndirectTargets *,uint32_t,uint16_t,NvmFileFlowDeclaration *);
#endif
