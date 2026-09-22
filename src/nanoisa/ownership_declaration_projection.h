#ifndef NANOISA_OWNERSHIP_DECLARATION_PROJECTION_H
#define NANOISA_OWNERSHIP_DECLARATION_PROJECTION_H
#include "ordinary_array_authority.h"
#include "nvm_v2_sections.h"
#include "ownership_contracts.h"
/* I expose copied complete declarations, never executable authority.
 * Input storage stays valid/immutable during preparation; output is disjoint.
 * Failure preserves outputs; success owns all numeric facts independently. */
typedef struct NvmOwnershipDeclarationPlan NvmOwnershipDeclarationPlan;
typedef enum {
    NVM_DECL_PREPARED, NVM_DECL_UNKNOWN, NVM_DECL_INVALID,
    NVM_DECL_LIMIT, NVM_DECL_MEMORY
} NvmDeclarationStatus;
typedef struct { NvmDeclarationStatus status; const char *message; } NvmDeclarationResult;
typedef struct { uint32_t layouts, types, bindings, unions, variants; } NvmDeclarationCounts;
typedef struct { uint8_t kind, flags; uint16_t fields; uint32_t name; } NvmDeclarationLayout;
NvmDeclarationResult nvm_prepare_ownership_declarations(const NvmModule *,NvmOwnershipDeclarationPlan **);
/* I read retained V2 facts directly using the same private declaration grammar.
 * Exact signature selectors and constant indices are never rebuilt. Other V2
 * sections are outside this declaration query: success grants no admission.
 * Existing private limits (256 layouts/65536 fields/16MiB/1M work) apply.
 * Unsupported declaration shapes remain UNKNOWN/INVALID; revision2 is not
 * enabled by this entry. Failure preserves *out. */
NvmDeclarationResult nvm_prepare_ownership_declarations_v2(const NvmV2Module *,NvmOwnershipDeclarationPlan **);
void nvm_ownership_declarations_free(NvmOwnershipDeclarationPlan *);
bool nvm_ownership_declarations_counts(const NvmOwnershipDeclarationPlan *,NvmDeclarationCounts *);
bool nvm_ownership_declarations_layout(const NvmOwnershipDeclarationPlan *,uint32_t,NvmDeclarationLayout *);
bool nvm_ownership_declarations_field(const NvmOwnershipDeclarationPlan *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_ownership_declarations_type(const NvmOwnershipDeclarationPlan *,uint32_t,NvmOrdinaryArrayType *);
bool nvm_ownership_declarations_binding(const NvmOwnershipDeclarationPlan *,uint32_t,NvmOrdinaryArrayBinding *);
bool nvm_ownership_declarations_variant(const NvmOwnershipDeclarationPlan *,uint32_t,uint16_t,NvmUnionVariantFact *);
#endif
