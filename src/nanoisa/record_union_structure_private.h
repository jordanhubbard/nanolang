#ifndef NANOISA_RECORD_UNION_STRUCTURE_PRIVATE_H
#define NANOISA_RECORD_UNION_STRUCTURE_PRIVATE_H
#include "record_array_structure_private.h"
#include "ownership_union_fields_private.h"
/* Owning verifier/analysis TUs only. Neither pointer is executable authority.
 * carrier->declarations is always NULL; old record projection/publication must
 * never consume this carrier. Analysis explicitly selects the union context.
 * Constructor validates the module and commits output only on success. Both
 * children belong to this wrapper; a transfer clears its source pointer before
 * destruction. No getter exposes a carrier and no entry accepts a supplied
 * wrapper as proof. Input storage is stable and outputs disjoint during calls. */
typedef struct {
    NvmRecordArrayStructure *carrier;
    NvmUnionFieldDeclarations *declarations;
} NvmRecordUnionStructure;
NvmArrayEligibilityResult nvm_record_union_structure_prepare(
    const NvmModule *,NvmRecordArrayBudget *,NvmRecordUnionStructure **);
void nvm_record_union_structure_free(NvmRecordUnionStructure *);
#endif
