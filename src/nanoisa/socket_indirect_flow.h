#ifndef NANOISA_SOCKET_INDIRECT_FLOW_H
#define NANOISA_SOCKET_INDIRECT_FLOW_H
#include "socket_cyclic.h"
#include "socket_indirect_targets.h"
#define NVM_SOCKET_INDIRECT_FLOW_BYTES (32u * 1024u * 1024u)
#define NVM_SOCKET_INDIRECT_FLOW_APPLICATIONS 262144u
typedef struct NvmSocketIndirectFlow NvmSocketIndirectFlow;
typedef struct {
 NvmSocketIndirectSummary targets;
 NvmSocketCyclicSummary ownership;
 uint32_t candidate_applications;
 size_t storage_bound;
 bool runtime_admitted; /* Always false. */
} NvmSocketIndirectFlowSummary;
typedef struct {
 uint32_t function,pc;
 uint64_t candidates,checked_candidates;
 NvmSocketFlowObligation common; /* target=NO_INDEX; all candidates are required. */
} NvmSocketIndirectFlowCall;
/* I prepare a distinct private ownership report, never an executable plan.
 * Calls are serialized; valid input storage remains immutable throughout both
 * checked passes. Outputs are valid and disjoint. Failures preserve outputs;
 * success owns all facts independently of input lifetime. No inner old report
 * is exposed or accepted as runtime authority. */
NvmSocketFlowStatus nvm_socket_indirect_flow_analyze(const NvmModule *,NvmSocketIndirectFlow **);
void nvm_socket_indirect_flow_free(NvmSocketIndirectFlow *);
bool nvm_socket_indirect_flow_summary(const NvmSocketIndirectFlow *,NvmSocketIndirectFlowSummary *);
bool nvm_socket_indirect_flow_function(const NvmSocketIndirectFlow *,uint32_t,NvmSocketCodeFunction *);
bool nvm_socket_indirect_flow_local(const NvmSocketIndirectFlow *,uint32_t,uint16_t,NvmSocketFlowDeclaration *);
bool nvm_socket_indirect_flow_instruction(const NvmSocketIndirectFlow *,uint32_t,uint16_t,NvmSocketCodeInstruction *);
bool nvm_socket_indirect_flow_variant_count(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint8_t *);
bool nvm_socket_indirect_flow_variant(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmSocketCyclicVariant *);
bool nvm_socket_indirect_flow_input_local(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmSocketFlowValue *);
bool nvm_socket_indirect_flow_input_stack(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmSocketFlowValue *);
bool nvm_socket_indirect_flow_input_reference(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmSocketFlowReference *);
bool nvm_socket_indirect_flow_input_region(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_socket_indirect_flow_call(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmSocketIndirectFlowCall *);
bool nvm_socket_indirect_flow_component(const NvmSocketIndirectFlow *,uint32_t,uint16_t,uint16_t *);
bool nvm_socket_indirect_flow_type(const NvmSocketIndirectFlow *,uint32_t,NvmSocketNominalLayout *);
bool nvm_socket_indirect_flow_import(const NvmSocketIndirectFlow *,uint32_t,uint32_t *);
#endif
