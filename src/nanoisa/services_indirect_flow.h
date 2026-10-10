#ifndef NANOISA_SERVICES_INDIRECT_FLOW_H
#define NANOISA_SERVICES_INDIRECT_FLOW_H
#include "services_cyclic.h"
#include "services_indirect_targets.h"
#define NVM_SERVICES_INDIRECT_FLOW_BYTES (32u * 1024u * 1024u)
#define NVM_SERVICES_INDIRECT_FLOW_APPLICATIONS 262144u
typedef struct NvmServicesIndirectFlow NvmServicesIndirectFlow;
typedef struct {
 NvmServicesIndirectSummary targets;
 NvmServicesCyclicSummary ownership;
 uint32_t candidate_applications;
 size_t storage_bound;
 bool runtime_admitted; /* Always false. */
} NvmServicesIndirectFlowSummary;
typedef struct {
 uint32_t function,pc;
 uint64_t candidates,checked_candidates;
 NvmServicesFlowObligation common; /* target=NO_INDEX; all candidates are required. */
} NvmServicesIndirectFlowCall;
/* I prepare a distinct private ownership report, never an executable plan.
 * Calls are serialized; valid input storage remains immutable throughout both
 * checked passes. Outputs are valid and disjoint. Failures preserve outputs;
 * success owns all facts independently of input lifetime. No inner old report
 * is exposed or accepted as runtime authority. */
NvmServicesFlowStatus nvm_services_indirect_flow_analyze(const NvmModule *,NvmServicesIndirectFlow **);
void nvm_services_indirect_flow_free(NvmServicesIndirectFlow *);
bool nvm_services_indirect_flow_summary(const NvmServicesIndirectFlow *,NvmServicesIndirectFlowSummary *);
bool nvm_services_indirect_flow_function(const NvmServicesIndirectFlow *,uint32_t,NvmServicesCodeFunction *);
bool nvm_services_indirect_flow_local(const NvmServicesIndirectFlow *,uint32_t,uint16_t,NvmServicesFlowDeclaration *);
bool nvm_services_indirect_flow_instruction(const NvmServicesIndirectFlow *,uint32_t,uint16_t,NvmServicesCodeInstruction *);
bool nvm_services_indirect_flow_variant_count(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint8_t *);
bool nvm_services_indirect_flow_variant(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmServicesCyclicVariant *);
bool nvm_services_indirect_flow_input_local(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowValue *);
bool nvm_services_indirect_flow_input_stack(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowValue *);
bool nvm_services_indirect_flow_input_reference(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmServicesFlowReference *);
bool nvm_services_indirect_flow_input_region(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_services_indirect_flow_call(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmServicesIndirectFlowCall *);
bool nvm_services_indirect_flow_component(const NvmServicesIndirectFlow *,uint32_t,uint16_t,uint16_t *);
bool nvm_services_indirect_flow_type(const NvmServicesIndirectFlow *,uint32_t,NvmServicesNominalLayout *);
bool nvm_services_indirect_flow_import(const NvmServicesIndirectFlow *,uint32_t,uint32_t *);
#endif
