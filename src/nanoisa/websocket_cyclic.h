#ifndef NANOISA_WEBSOCKET_CYCLIC_H
#define NANOISA_WEBSOCKET_CYCLIC_H
#include "websocket_body.h"

#define NVM_WEBSOCKET_CYCLIC_REVISION 1u
#define NVM_WEBSOCKET_CYCLIC_ALTERNATIVES 16u
#define NVM_WEBSOCKET_CYCLIC_FUNCTION_PAIRS 4096u
#define NVM_WEBSOCKET_CYCLIC_PAIRS 65536u
#define NVM_WEBSOCKET_CYCLIC_EDGES 131072u
#define NVM_WEBSOCKET_CYCLIC_BYTES (16u * 1024u * 1024u)
#define NVM_WEBSOCKET_CYCLIC_NO_VARIANT UINT8_MAX

typedef struct NvmWebSocketCyclicReport NvmWebSocketCyclicReport;
typedef struct {
    uint32_t revision, functions, instructions, variants, transfers, edges;
    size_t storage_peak;
    bool runtime_admitted; /* Always false: no hosted, scalar or target authority. */
} NvmWebSocketCyclicSummary;
typedef struct {
    uint16_t locals, stack, regions, owners, references;
} NvmWebSocketCyclicStateInfo;
typedef struct {
    NvmWebSocketCyclicStateInfo input, output;
    NvmWebSocketBodyInstruction body;
    uint8_t edge_mask, edge_variants[2];
    /* Edges index the decoded instruction's original successors. An absent
     * refined edge has its bit clear and NO_VARIANT, never assumed reachable. */
} NvmWebSocketCyclicVariant;

/* Private non-admitting query. Calls require external serialization. Inputs
 * remain immutable during analysis; outputs are valid and disjoint. Success
 * owns all facts, failure leaves *out unchanged. No input pointers are retained.
 * Accessors copy facts and leave output unchanged on an invalid index.
 * Local/stack owners are canonical labels 1..256. Formal reference owner
 * anchors are 257+parameter, regions 513+depth and epochs 769+reference slot.
 * These are proof relations, NEVER physical WebSocket identities or runtime epochs. */
NvmWebSocketFlowStatus nvm_websocket_cyclic_analyze(const NvmModule *, NvmWebSocketCyclicReport **);
void nvm_websocket_cyclic_free(NvmWebSocketCyclicReport *);
bool nvm_websocket_cyclic_summary(const NvmWebSocketCyclicReport *, NvmWebSocketCyclicSummary *);
bool nvm_websocket_cyclic_function(const NvmWebSocketCyclicReport *, uint32_t, NvmWebSocketCodeFunction *);
bool nvm_websocket_cyclic_local(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, NvmWebSocketFlowDeclaration *);
bool nvm_websocket_cyclic_instruction(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, NvmWebSocketCodeInstruction *);
bool nvm_websocket_cyclic_component(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, uint16_t *);
bool nvm_websocket_cyclic_variant_count(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, uint8_t *);
bool nvm_websocket_cyclic_variant(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, uint8_t, NvmWebSocketCyclicVariant *);
bool nvm_websocket_cyclic_input_local(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmWebSocketFlowValue *);
bool nvm_websocket_cyclic_input_stack(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmWebSocketFlowValue *);
bool nvm_websocket_cyclic_input_reference(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmWebSocketFlowReference *);
bool nvm_websocket_cyclic_input_region(const NvmWebSocketCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, uint64_t *);
bool nvm_websocket_cyclic_type(const NvmWebSocketCyclicReport *, uint32_t, NvmWebSocketNominalLayout *);
bool nvm_websocket_cyclic_import(const NvmWebSocketCyclicReport *, uint32_t, uint32_t *);
#endif
