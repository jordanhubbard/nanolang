#ifndef NANOISA_WEBSOCKET_INDIRECT_TARGETS_H
#define NANOISA_WEBSOCKET_INDIRECT_TARGETS_H
#include "websocket_code.h"
#define NVM_WEBSOCKET_INDIRECT_VISITS 262144u
#define NVM_WEBSOCKET_INDIRECT_BYTES (16u * 1024u * 1024u)
typedef struct NvmWebSocketIndirectTargets NvmWebSocketIndirectTargets;
typedef enum { NVM_WEBSOCKET_INDIRECT_DESCRIBED, NVM_WEBSOCKET_INDIRECT_INVALID,
 NVM_WEBSOCKET_INDIRECT_UNRESOLVED, NVM_WEBSOCKET_INDIRECT_LIMIT,
 NVM_WEBSOCKET_INDIRECT_MEMORY } NvmWebSocketIndirectStatus;
typedef struct {
 NvmWebSocketIndirectStatus status;
 uint32_t function,pc; /* Original function-relative byte PC, or NO_INDEX. */
 const char *message; /* Static storage. */
} NvmWebSocketIndirectResult;
typedef struct {
 uint32_t functions,instructions,calls,visits;
 size_t storage_bound;
} NvmWebSocketIndirectSummary;
typedef struct {
 uint32_t function,pc;
 uint16_t instruction,parameters;
 uint64_t candidates; /* Bit f is original same-module function f. */
 uint8_t result_count;
 NvmWebSocketFlowDeclaration result;
} NvmWebSocketIndirectCall;
/* Private descriptive target facts only, NEVER body/ownership/runtime authority.
 * Inputs are valid immutable storage during preparation. The report owns all
 * facts and survives input destruction. Outputs must be disjoint and valid.
 * Only DESCRIBED publishes. Failure and invalid getters preserve outputs.
 * Calls require external serialization, as do the underlying ISA queries. */
NvmWebSocketIndirectResult nvm_websocket_indirect_targets(const NvmModule *,NvmWebSocketIndirectTargets **);
void nvm_websocket_indirect_targets_free(NvmWebSocketIndirectTargets *);
bool nvm_websocket_indirect_targets_summary(const NvmWebSocketIndirectTargets *,NvmWebSocketIndirectSummary *);
bool nvm_websocket_indirect_targets_call(const NvmWebSocketIndirectTargets *,uint32_t,NvmWebSocketIndirectCall *);
bool nvm_websocket_indirect_targets_parameter(const NvmWebSocketIndirectTargets *,uint32_t,uint16_t,NvmWebSocketFlowDeclaration *);
#endif
