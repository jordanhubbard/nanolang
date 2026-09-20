#ifndef NANOISA_FILE_INDIRECT_FLOW_H
#define NANOISA_FILE_INDIRECT_FLOW_H
#include "file_cyclic.h"
#include "file_indirect_targets.h"
#define NVM_FILE_INDIRECT_FLOW_BYTES (32u * 1024u * 1024u)
#define NVM_FILE_INDIRECT_FLOW_APPLICATIONS 262144u
typedef struct NvmFileIndirectFlow NvmFileIndirectFlow;
typedef struct {
 NvmFileIndirectSummary targets;
 NvmFileCyclicSummary ownership;
 uint32_t candidate_applications;
 size_t storage_bound;
 bool runtime_admitted; /* Always false. */
} NvmFileIndirectFlowSummary;
typedef struct {
 uint32_t function,pc;
 uint64_t candidates,checked_candidates;
 NvmFileFlowObligation common; /* target=NO_INDEX; all candidates are required. */
} NvmFileIndirectFlowCall;
/* I prepare a distinct private ownership report, never an executable plan.
 * Calls are serialized; valid input storage remains immutable throughout both
 * checked passes. Outputs are valid and disjoint. Failures preserve outputs;
 * success owns all facts independently of input lifetime. No inner old report
 * is exposed or accepted as runtime authority. */
NvmFileFlowStatus nvm_file_indirect_flow_analyze(const NvmModule *,NvmFileIndirectFlow **);
void nvm_file_indirect_flow_free(NvmFileIndirectFlow *);
bool nvm_file_indirect_flow_summary(const NvmFileIndirectFlow *,NvmFileIndirectFlowSummary *);
bool nvm_file_indirect_flow_function(const NvmFileIndirectFlow *,uint32_t,NvmFileCodeFunction *);
bool nvm_file_indirect_flow_local(const NvmFileIndirectFlow *,uint32_t,uint16_t,NvmFileFlowDeclaration *);
bool nvm_file_indirect_flow_instruction(const NvmFileIndirectFlow *,uint32_t,uint16_t,NvmFileCodeInstruction *);
bool nvm_file_indirect_flow_variant_count(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint8_t *);
bool nvm_file_indirect_flow_variant(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmFileCyclicVariant *);
bool nvm_file_indirect_flow_input_local(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmFileFlowValue *);
bool nvm_file_indirect_flow_input_stack(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmFileFlowValue *);
bool nvm_file_indirect_flow_input_reference(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,NvmFileFlowReference *);
bool nvm_file_indirect_flow_input_region(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint8_t,uint16_t,uint64_t *);
bool nvm_file_indirect_flow_call(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint8_t,NvmFileIndirectFlowCall *);
bool nvm_file_indirect_flow_component(const NvmFileIndirectFlow *,uint32_t,uint16_t,uint16_t *);
bool nvm_file_indirect_flow_type(const NvmFileIndirectFlow *,uint32_t,NvmFileNominalLayout *);
bool nvm_file_indirect_flow_import(const NvmFileIndirectFlow *,uint32_t,uint32_t *);
#endif
