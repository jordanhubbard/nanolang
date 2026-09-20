#ifndef NANOISA_FILE_CYCLIC_H
#define NANOISA_FILE_CYCLIC_H
#include "file_body.h"

#define NVM_FILE_CYCLIC_REVISION 1u
#define NVM_FILE_CYCLIC_ALTERNATIVES 16u
#define NVM_FILE_CYCLIC_FUNCTION_PAIRS 4096u
#define NVM_FILE_CYCLIC_PAIRS 65536u
#define NVM_FILE_CYCLIC_EDGES 131072u
#define NVM_FILE_CYCLIC_BYTES (16u * 1024u * 1024u)
#define NVM_FILE_CYCLIC_NO_VARIANT UINT8_MAX

typedef struct NvmFileCyclicReport NvmFileCyclicReport;
typedef struct {
    uint32_t revision, functions, instructions, variants, transfers, edges;
    size_t storage_peak;
    bool runtime_admitted; /* Always false: no hosted, scalar or target authority. */
} NvmFileCyclicSummary;
typedef struct {
    uint16_t locals, stack, regions, owners, references;
} NvmFileCyclicStateInfo;
typedef struct {
    NvmFileCyclicStateInfo input, output;
    NvmFileBodyInstruction body;
    uint8_t edge_mask, edge_variants[2];
    /* Edges index the decoded instruction's original successors. An absent
     * refined edge has its bit clear and NO_VARIANT, never assumed reachable. */
} NvmFileCyclicVariant;

/* Private non-admitting query. Calls require external serialization. Inputs
 * remain immutable during analysis; outputs are valid and disjoint. Success
 * owns all facts, failure leaves *out unchanged. No input pointers are retained.
 * Accessors copy facts and leave output unchanged on an invalid index.
 * Local/stack owners are canonical labels 1..256. Formal reference owner
 * anchors are 257+parameter, regions 513+depth and epochs 769+reference slot.
 * These are proof relations, NEVER physical File identities or runtime epochs. */
NvmFileFlowStatus nvm_file_cyclic_analyze(const NvmModule *, NvmFileCyclicReport **);
void nvm_file_cyclic_free(NvmFileCyclicReport *);
bool nvm_file_cyclic_summary(const NvmFileCyclicReport *, NvmFileCyclicSummary *);
bool nvm_file_cyclic_function(const NvmFileCyclicReport *, uint32_t, NvmFileCodeFunction *);
bool nvm_file_cyclic_local(const NvmFileCyclicReport *, uint32_t, uint16_t, NvmFileFlowDeclaration *);
bool nvm_file_cyclic_instruction(const NvmFileCyclicReport *, uint32_t, uint16_t, NvmFileCodeInstruction *);
bool nvm_file_cyclic_component(const NvmFileCyclicReport *, uint32_t, uint16_t, uint16_t *);
bool nvm_file_cyclic_variant_count(const NvmFileCyclicReport *, uint32_t, uint16_t, uint8_t *);
bool nvm_file_cyclic_variant(const NvmFileCyclicReport *, uint32_t, uint16_t, uint8_t, NvmFileCyclicVariant *);
bool nvm_file_cyclic_input_local(const NvmFileCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmFileFlowValue *);
bool nvm_file_cyclic_input_stack(const NvmFileCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmFileFlowValue *);
bool nvm_file_cyclic_input_reference(const NvmFileCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, NvmFileFlowReference *);
bool nvm_file_cyclic_input_region(const NvmFileCyclicReport *, uint32_t, uint16_t, uint8_t, uint16_t, uint64_t *);
bool nvm_file_cyclic_type(const NvmFileCyclicReport *, uint32_t, NvmFileNominalLayout *);
bool nvm_file_cyclic_import(const NvmFileCyclicReport *, uint32_t, uint32_t *);
#endif
