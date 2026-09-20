#ifndef NANOISA_OWNERSHIP_LAYOUTS_PRIVATE_H
#define NANOISA_OWNERSHIP_LAYOUTS_PRIVATE_H
#include "nvm_v2_sections.h"
#define NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_LAYOUTS 256u
#define NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_FIELDS 65536u
#define NVM_OWNERSHIP_LAYOUTS_PRIVATE_MAX_BYTES (16u * 1024u * 1024u)
/* I decode declaration structure only. This private profile additionally allows
 * ARRAY/NO_INDEX leaves in forward all-record DAGs. It supplies no element type,
 * ordinary/resource classification, reference eligibility or execution grant.
 * Input is valid immutable storage during the call; output is valid/disjoint.
 * Success owns copied numeric fields, independent of the input lifetime.
 * Failure preserves *out. The legacy result enum conflates truncation and OOM;
 * callers must not infer a precise allocation diagnosis from TRUNCATED alone.
 * Existing public layout decoding retains its original profile and semantics. */
NvmV2Result nvm_ownership_layouts_private_decode(const uint8_t *,size_t,NvmV2Layouts *);
/* I additionally validate scalar-union nodes beside forward record DAGs.
 * Only the complete mixed declaration query selects this private grammar. */
NvmV2Result nvm_ownership_mixed_layouts_private_decode(const uint8_t *,size_t,NvmV2Layouts *);
#endif
