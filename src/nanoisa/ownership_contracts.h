#ifndef NANOISA_OWNERSHIP_CONTRACTS_H
#define NANOISA_OWNERSHIP_CONTRACTS_H
#include "nvm_format.h"
#include "nvm_format_v2.h"

#define NVM_OWNERSHIP_VERSION 1u
#define NVM_OWNERSHIP_PATH_VERSION 2u
#define NVM_OWNERSHIP_EXTENSION_VERSION 3u
/* I retain aggregate union declarations; execution needs selected transfers. */
#define NVM_OWNERSHIP_UNION_GRAPH_VERSION 4u
#define NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS 1u
#define NVM_OWNERSHIP_EXTENSION_ARRAY_FIELDS 2u
#define NVM_OWNERSHIP_EXTENSION_REVISION_1 1u
#define NVM_OWNERSHIP_MAX_EXTENSIONS 2u
#define NVM_OWNED_MAX_FUNCTIONS 8u
#define NVM_OWNERSHIP_MAX_PATHS 256u
#define NVM_OWNERSHIP_MAX_PATH_DEPTH 32u
#define NVM_LAYOUT_COMPLETE 1u
#define NVM_LAYOUT_RESOURCE 2u
#define NVM_OWNERSHIP_MAX_UNIONS 256u
#define NVM_OWNERSHIP_MAX_VARIANTS 256u

typedef struct {
    uint32_t layout;
    uint32_t name_idx;
    uint16_t field_offset;
    uint16_t field_count;
} NvmUnionVariantFact;

/* I validate declarations, not instruction lifetimes. Resource/reference
 * declarations set requires_verifier; executable consumers must refuse them
 * unless the shared executable ownership verifier admits their exact subset. */
NvmV2Result nvm_ownership_contracts_validate(const NvmModule *module,
                                            bool *requires_verifier);
typedef enum {
    NVM_LAYOUT_AUTHORITY_UNKNOWN, NVM_LAYOUT_AUTHORITY_ORDINARY,
    NVM_LAYOUT_AUTHORITY_RESOURCE
} NvmLayoutAuthority;
/* I query only after validating the complete declaration payload. With no
 * payload I return UNKNOWN (the caller separately resolves a retained index).
 * With a payload I also check its index. Failure leaves *out unchanged. This
 * declaration classification supplies no instruction-flow or storage admission. */
NvmV2Result nvm_ownership_layout_authority(const NvmModule *, uint32_t, NvmLayoutAuthority *);
/* I validate once, then publish exactly count declaration classifications.
 * A present payload must have that exact count; absent metadata yields UNKNOWN.
 * The caller supplies count entries (or NULL for zero). Every failure leaves
 * the entire output unchanged; layout resolution remains a separate duty. */
NvmV2Result nvm_ownership_layout_authorities(const NvmModule *, uint32_t, NvmLayoutAuthority *);

/* I read immutable numeric paths without allocating. Layout/authority checks
 * remain the verifier's responsibility; this reader checks transport shape.
 * I leave outputs unchanged on failure. */
NvmV2Result nvm_ownership_path(const NvmModule *module, uint32_t index,
                               uint16_t *fields, uint16_t capacity, uint16_t *count);
/* I return one exact concrete-union variant slice from the UNION_VARIANTS
 * extension in version 3 (scalars) or version 4 (aggregate child layouts).
 * Union ordinals follow retained UNION layout order. Failure leaves *out
 * unchanged. This declaration query alone grants no executable authority. */
NvmV2Result nvm_ownership_union_variant(const NvmModule *module,
                                        uint32_t union_ordinal,
                                        uint16_t variant,
                                        NvmUnionVariantFact *out);
#endif
