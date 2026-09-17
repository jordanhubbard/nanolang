#ifndef NANOISA_OWNERSHIP_CONTRACTS_H
#define NANOISA_OWNERSHIP_CONTRACTS_H
#include "nvm_format.h"
#include "nvm_format_v2.h"

#define NVM_OWNERSHIP_VERSION 1u
#define NVM_LAYOUT_COMPLETE 1u
#define NVM_LAYOUT_RESOURCE 2u

/* I validate declarations, not instruction lifetimes. Resource/reference
 * declarations set requires_verifier; executable consumers must refuse them
 * unless the shared executable ownership verifier admits their exact subset. */
NvmV2Result nvm_ownership_contracts_validate(const NvmModule *module,
                                            bool *requires_verifier);
#endif
