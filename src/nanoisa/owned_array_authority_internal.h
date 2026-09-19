#ifndef NANOISA_OWNED_ARRAY_AUTHORITY_INTERNAL_H
#define NANOISA_OWNED_ARRAY_AUTHORITY_INTERNAL_H
#include "owned_array_authority.h"
/* Internal owning query output. No consumer accepts an externally supplied
 * instance as state or executable authority; prepare always recomputes it. */
typedef struct {
    NvmOwnedArrayOrigins *origins;
    NvmOwnedArrayLayouts *layouts;
    NvmOwnerAuthorityCounts counts;
    NvmOwnerSignature signatures[8];
    NvmOwnerDeclaration locals[8][256];
    NvmOwnerOriginObligation obligations[4096];
} NvmOwnerLifetimeFacts;
NvmOwnerAuthorityResult nvm_analyze_owned_array_lifetimes(const NvmModule *, NvmOwnerLifetimeFacts **);
void nvm_owned_array_lifetime_facts_free(NvmOwnerLifetimeFacts *);
#endif
