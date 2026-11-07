#ifndef NANOISA_OWNERSHIP_UNION_FIELDS_PRIVATE_H
#define NANOISA_OWNERSHIP_UNION_FIELDS_PRIVATE_H
#include "ownership_declaration_projection.h"
/* Descriptive numeric copy only: never execution authority. Input module and
 * backing storage remain readable/exclusively stable during preparation.
 * Writable outputs are disjoint from all input/owner storage. Getters borrow
 * the owner during the call, copy facts, and preserve output on failure.
 * String indices retain their source index namespace, not string storage.
 * Each getter costs at most four logical probes, including misses; callers
 * with budgets precharge explicitly. No hidden mutable query budget. */
typedef struct NvmUnionFieldDeclarations NvmUnionFieldDeclarations;
typedef enum { NVM_UFD_PREPARED, NVM_UFD_UNKNOWN, NVM_UFD_INVALID,
               NVM_UFD_LIMIT, NVM_UFD_MEMORY } NvmUnionFieldDeclarationStatus;
typedef struct {
    NvmUnionFieldDeclarationStatus status;
    const char *message;
    uint64_t bytes_reserved, work_reserved;
} NvmUnionFieldDeclarationResult;
NvmUnionFieldDeclarationResult nvm_union_field_declarations_prepare(const NvmModule *,NvmUnionFieldDeclarations **);
void nvm_union_field_declarations_free(NvmUnionFieldDeclarations *);
bool nvm_union_field_declarations_layout(const NvmUnionFieldDeclarations *,uint32_t,NvmDeclarationLayout *);
bool nvm_union_field_declarations_field(const NvmUnionFieldDeclarations *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_union_field_declarations_variant(const NvmUnionFieldDeclarations *,uint32_t,uint16_t,NvmUnionVariantFact *);
bool nvm_union_field_declarations_nominal(const NvmUnionFieldDeclarations *,uint8_t,uint32_t,uint32_t *);
#endif
