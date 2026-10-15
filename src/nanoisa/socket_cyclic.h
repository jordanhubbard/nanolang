#ifndef NANOISA_SOCKET_CYCLIC_H
#define NANOISA_SOCKET_CYCLIC_H
#include "socket_body.h"

#define NVM_SOCKET_CYCLIC_REVISION 1u
#define NVM_SOCKET_CYCLIC_ALTERNATIVES 16u
#define NVM_SOCKET_CYCLIC_FUNCTION_PAIRS 4096u
#define NVM_SOCKET_CYCLIC_PAIRS 65536u
#define NVM_SOCKET_CYCLIC_EDGES 131072u
#define NVM_SOCKET_CYCLIC_BYTES (16u * 1024u * 1024u)
#define NVM_SOCKET_CYCLIC_NO_VARIANT UINT8_MAX

typedef struct NvmSocketCyclicReport NvmSocketCyclicReport;
typedef struct {
    uint32_t revision, functions, instructions, variants, transfers, edges;
    size_t storage_peak;
    bool runtime_admitted; /* Always false: no hosted, scalar or target authority. */
} NvmSocketCyclicSummary;
typedef struct {
    uint16_t locals, stack, regions, owners, references;
} NvmSocketCyclicStateInfo;
typedef struct {
    NvmSocketCyclicStateInfo input, output;
    NvmSocketBodyInstruction body;
    uint8_t edge_mask, edge_variants[2];
    /* Edges index the decoded instruction's original successors. An absent
     * refined edge has its bit clear and NO_VARIANT, never assumed reachable. */
} NvmSocketCyclicVariant;

/* Private non-admitting query. Calls require external serialization. Inputs
 * remain immutable during analysis; outputs are valid and disjoint. Success
 * owns all facts, failure leaves *out unchanged. No input pointers are retained.
 * Accessors copy facts and leave output unchanged on an invalid index.
 * Local/stack owners are canonical labels 1..256. Formal reference owner
 * anchors are 257+parameter, regions 513+depth and epochs 769+reference slot.
 * These are proof relations, NEVER physical Socket identities or runtime epochs. */
NvmSocketFlowStatus nvm_socket_cyclic_analyze(const NvmModule *, NvmSocketCyclicReport **);
void nvm_socket_cyclic_free(NvmSocketCyclicReport *);
bool nvm_socket_cyclic_summary(const NvmSocketCyclicReport *, NvmSocketCyclicSummary *);
bool nvm_socket_cyclic_function(const NvmSocketCyclicReport *, uint32_t, NvmSocketCodeFunction *);
bool nvm_socket_cyclic_local(const NvmSocketCyclicReport *, uint32_t, uint16_t, NvmSocketFlowDeclaration *);
bool nvm_socket_cyclic_instruction(const NvmSocketCyclicReport *, uint32_t, uint16_t, NvmSocketCodeInstruction *);
bool nvm_socket_cyclic_component(const NvmSocketCyclicReport *, uint32_t, uint16_t, uint16_t *);
bool nvm_socket_cyclic_variant_count(const NvmSocketCyclicReport *, uint32_t, uint16_t, uint8_t *);
bool nvm_socket_cyclic_variant(const NvmSocketCyclicReport *, uint32_t, uint16_t, uint8_t, NvmSocketCyclicVariant *);
bool nvm_socket_cyclic_input_local(const NvmSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmSocketFlowValue *);
bool nvm_socket_cyclic_input_stack(const NvmSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmSocketFlowValue *);
bool nvm_socket_cyclic_input_reference(const NvmSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmSocketFlowReference *);
bool nvm_socket_cyclic_input_region(const NvmSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, uint64_t *);
bool nvm_socket_cyclic_type(const NvmSocketCyclicReport *, uint32_t, NvmSocketNominalLayout *);
bool nvm_socket_cyclic_import(const NvmSocketCyclicReport *, uint32_t, uint32_t *);
#endif
