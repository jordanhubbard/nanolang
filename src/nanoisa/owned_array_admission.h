#ifndef NANOISA_OWNED_ARRAY_ADMISSION_H
#define NANOISA_OWNED_ARRAY_ADMISSION_H
#include "owned_array_authority.h"
/* Routing is not authority. A selected/invalid route never falls back. Service
 * declarations retain priority at every consumer. */
typedef enum { NVM_OWNER_ARRAY_NOT_SELECTED, NVM_OWNER_ARRAY_SELECTED,
    NVM_OWNER_ARRAY_INVALID } NvmOwnedArrayRoute;
NvmOwnedArrayRoute nvm_owned_array_route(const NvmModule *);
/* Fresh complete authority plus explicit hosted runtime policy. PREPARED here
 * means admitted by this wrapper; the original query remains non-admitting.
 * Failure preserves *out. No caller supplies a prebuilt certificate. */
NvmOwnerAuthorityResult nvm_owned_array_admit(const NvmModule *, NvmOwnedArrayPlan **out);
#endif
