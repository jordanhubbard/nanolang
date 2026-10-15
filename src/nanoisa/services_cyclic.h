#ifndef NANOISA_SERVICES_CYCLIC_H
#define NANOISA_SERVICES_CYCLIC_H
#include "services_body.h"

#define NVM_SERVICES_CYCLIC_REVISION 1u
#define NVM_SERVICES_CYCLIC_ALTERNATIVES 16u
#define NVM_SERVICES_CYCLIC_FUNCTION_PAIRS 4096u
#define NVM_SERVICES_CYCLIC_PAIRS 65536u
#define NVM_SERVICES_CYCLIC_EDGES 131072u
#define NVM_SERVICES_CYCLIC_BYTES (16u * 1024u * 1024u)
#define NVM_SERVICES_CYCLIC_NO_VARIANT UINT8_MAX

typedef struct NvmServicesCyclicReport NvmServicesCyclicReport;
typedef struct {
    uint32_t revision, functions, instructions, variants, transfers, edges;
    size_t storage_peak;
    bool runtime_admitted; /* Always false: no hosted, scalar or target authority. */
} NvmServicesCyclicSummary;
typedef struct {
    uint16_t locals, stack, regions, owners, references;
} NvmServicesCyclicStateInfo;
typedef struct {
    NvmServicesCyclicStateInfo input, output;
    NvmServicesBodyInstruction body;
    uint8_t edge_mask, edge_variants[2];
    /* Edges index the decoded instruction's original successors. An absent
     * refined edge has its bit clear and NO_VARIANT, never assumed reachable. */
} NvmServicesCyclicVariant;

/* Private non-admitting query. Calls require external serialization. Inputs
 * remain immutable during analysis; outputs are valid and disjoint. Success
 * owns all facts, failure leaves *out unchanged. No input pointers are retained.
 * Accessors copy facts and leave output unchanged on an invalid index.
 * Local/stack owners are canonical labels 1..256. Formal reference owner
 * anchors are 257+parameter, regions 513+depth and epochs 769+reference slot.
 * These are proof relations, NEVER physical Services identities or runtime epochs. */
NvmServicesFlowStatus nvm_services_cyclic_analyze(const NvmModule *, NvmServicesCyclicReport **);
void nvm_services_cyclic_free(NvmServicesCyclicReport *);
bool nvm_services_cyclic_summary(const NvmServicesCyclicReport *, NvmServicesCyclicSummary *);
bool nvm_services_cyclic_function(const NvmServicesCyclicReport *, uint32_t, NvmServicesCodeFunction *);
bool nvm_services_cyclic_local(const NvmServicesCyclicReport *, uint32_t, uint16_t, NvmServicesFlowDeclaration *);
bool nvm_services_cyclic_instruction(const NvmServicesCyclicReport *, uint32_t, uint16_t, NvmServicesCodeInstruction *);
bool nvm_services_cyclic_component(const NvmServicesCyclicReport *, uint32_t, uint16_t, uint16_t *);
bool nvm_services_cyclic_variant_count(const NvmServicesCyclicReport *, uint32_t, uint16_t, uint8_t *);
bool nvm_services_cyclic_variant(const NvmServicesCyclicReport *, uint32_t, uint16_t, uint8_t, NvmServicesCyclicVariant *);
bool nvm_services_cyclic_input_local(const NvmServicesCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmServicesFlowValue *);
bool nvm_services_cyclic_input_stack(const NvmServicesCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmServicesFlowValue *);
bool nvm_services_cyclic_input_reference(const NvmServicesCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmServicesFlowReference *);
bool nvm_services_cyclic_input_region(const NvmServicesCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, uint64_t *);
bool nvm_services_cyclic_type(const NvmServicesCyclicReport *, uint32_t, NvmServicesNominalLayout *);
bool nvm_services_cyclic_import(const NvmServicesCyclicReport *, uint32_t, uint32_t *);
#endif
