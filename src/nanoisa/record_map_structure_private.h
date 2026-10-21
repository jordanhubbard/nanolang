#ifndef NANOISA_RECORD_MAP_STRUCTURE_PRIVATE_H
#define NANOISA_RECORD_MAP_STRUCTURE_PRIVATE_H
#include "record_array_structure_private.h"
#include "ownership_map_declarations.h"
/* Internal constructor: each successful output has its own destructor. */
NvmMapDeclarationResult nvm_prepare_map_declarations_split(const NvmModule *,
    NvmMapDeclarationPlan **,NvmOwnershipDeclarationPlan **);
NvmArrayEligibilityResult nvm_record_map_structure_prepare(const NvmModule *,
    NvmRecordArrayBudget *,NvmRecordArrayStructure **,NvmMapDeclarationPlan **);
static inline bool nvm_record_map_opcode(uint8_t op) {
    return op==OP_HM_NEW || op==OP_HM_SET || op==OP_HM_GET ||
           op==OP_HM_HAS || op==OP_HM_LEN;
}
#endif
