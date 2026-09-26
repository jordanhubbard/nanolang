#ifndef NANOISA_OWNERSHIP_LAYOUTS_PRIVATE_H
#define NANOISA_OWNERSHIP_LAYOUTS_PRIVATE_H
#include "nvm_v2_sections.h"
#include "isa.h"
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
/* I preserve the legacy result and failure-atomic layout output, but separately
 * report an observed allocation refusal. The optional detail sink is set false
 * at entry, true only for failed items/fields/forward-workspace allocations.
 * It is disjoint from input and output storage. No allocation order changes. */
NvmV2Result nvm_ownership_mixed_layouts_private_decode_detailed(const uint8_t *,size_t,
                                                             NvmV2Layouts *,bool *allocation_failed);
/* I share exact tag/kind matching with the private typed declaration readers. */
static inline int nvm_ownership_nominal_layout_kind(uint8_t tag) {
    switch(tag) {
    case TAG_STRUCT:return NVM_V2_LAYOUT_STRUCT;
    case TAG_TUPLE:return NVM_V2_LAYOUT_TUPLE;
    case TAG_UNION:return NVM_V2_LAYOUT_UNION;
    case TAG_ENUM:return NVM_V2_LAYOUT_ENUM;
    default:return -1;
    }
}
/* I validate every typed table, including prior-only tables, with the common
 * bounded iterative storage-DAG reader. This grants no execution authority. */
NvmV2Result nvm_ownership_typed_layouts_private_decode_detailed(const uint8_t *,size_t,
                                                             NvmV2Layouts *,bool *);
#endif
