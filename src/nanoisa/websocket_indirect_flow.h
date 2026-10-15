#ifndef NANOISA_WEBSOCKET_INDIRECT_FLOW_H
#define NANOISA_WEBSOCKET_INDIRECT_FLOW_H
#include "websocket_cyclic.h"
#include "websocket_indirect_targets.h"
#define NVM_WEBSOCKET_INDIRECT_FLOW_BYTES (32u * 1024u * 1024u)
#define NVM_WEBSOCKET_INDIRECT_FLOW_APPLICATIONS 262144u
typedef struct NvmWebSocketIndirectFlow NvmWebSocketIndirectFlow;
typedef struct {
 NvmWebSocketIndirectSummary targets;
 NvmWebSocketCyclicSummary ownership;
 uint32_t candidate_applications;
 size_t storage_bound;
 bool runtime_admitted; /* Always false. */
} NvmWebSocketIndirectFlowSummary;
typedef struct {
 uint32_t function,pc;
 uint64_t candidates,checked_candidates;
 NvmWebSocketFlowObligation common; /* target=NO_INDEX; all candidates are required. */
} NvmWebSocketIndirectFlowCall;
/* I prepare a distinct private ownership report, never an executable plan.
 * Calls are serialized; valid input storage remains immutable throughout both
 * checked passes. Outputs are valid and disjoint. Failures preserve outputs;
 * success owns all facts independently of input lifetime. No inner old report
 * is exposed or accepted as runtime authority. */
NvmWebSocketFlowStatus nvm_websocket_indirect_flow_analyze(const NvmModule *,NvmWebSocketIndirectFlow **);
void nvm_websocket_indirect_flow_free(NvmWebSocketIndirectFlow *);
bool nvm_websocket_indirect_flow_summary(const NvmWebSocketIndirectFlow *,NvmWebSocketIndirectFlowSummary *);
bool nvm_websocket_indirect_flow_function(const NvmWebSocketIndirectFlow *,uint32_t,NvmWebSocketCodeFunction *);
bool nvm_websocket_indirect_flow_local(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,NvmWebSocketFlowDeclaration *);
bool nvm_websocket_indirect_flow_instruction(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,NvmWebSocketCodeInstruction *);
bool nvm_websocket_indirect_flow_variant_count(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint8_t *);
bool nvm_websocket_indirect_flow_variant(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmWebSocketCyclicVariant *);
bool nvm_websocket_indirect_flow_input_local(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowValue *);
bool nvm_websocket_indirect_flow_input_stack(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowValue *);
bool nvm_websocket_indirect_flow_input_reference(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmWebSocketFlowReference *);
bool nvm_websocket_indirect_flow_input_region(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_websocket_indirect_flow_call(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmWebSocketIndirectFlowCall *);
bool nvm_websocket_indirect_flow_component(const NvmWebSocketIndirectFlow *,uint32_t,uint16_t,uint16_t *);
bool nvm_websocket_indirect_flow_type(const NvmWebSocketIndirectFlow *,uint32_t,NvmWebSocketNominalLayout *);
bool nvm_websocket_indirect_flow_import(const NvmWebSocketIndirectFlow *,uint32_t,uint32_t *);
#endif
