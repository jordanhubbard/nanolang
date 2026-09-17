#ifndef NANOISA_OWNERSHIP_CONTRACTS_H
#define NANOISA_OWNERSHIP_CONTRACTS_H
#include "nvm_format.h"
#include "nvm_format_v2.h"

#define NVM_OWNERSHIP_VERSION 1u
#define NVM_OWNERSHIP_PATH_VERSION 2u
#define NVM_OWNERSHIP_MAX_PATHS 256u
#define NVM_OWNERSHIP_MAX_PATH_DEPTH 32u
#define NVM_LAYOUT_COMPLETE 1u
#define NVM_LAYOUT_RESOURCE 2u

/* I validate declarations, not instruction lifetimes. Resource/reference
 * declarations set requires_verifier; executable consumers must refuse them
 * unless the shared executable ownership verifier admits their exact subset. */
NvmV2Result nvm_ownership_contracts_validate(const NvmModule *module,
                                            bool *requires_verifier);
/* I read immutable numeric paths without allocating. Layout/authority checks
 * remain the verifier's responsibility; this reader checks transport shape.
 * I leave outputs unchanged on failure. */
NvmV2Result nvm_ownership_path(const NvmModule *module, uint32_t index,
                               uint16_t *fields, uint16_t capacity, uint16_t *count);
#endif
