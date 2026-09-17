#ifndef NANOISA_REFERENCE_PLACES_H
#define NANOISA_REFERENCE_PLACES_H

#include "nvm_v2_sections.h"

/* I describe a checked place, not a NanoValue or a wire instruction. A runtime
 * invocation distinguishes recursive frames; a verifier supplies its own
 * symbolic invocation identity. The field array is borrowed and immutable. */
typedef enum {
    NVM_REFERENCE_SHARED = 1,
    NVM_REFERENCE_EXCLUSIVE = 2
} NvmReferenceMode;

typedef struct {
    uint64_t invocation;
    uint16_t local;
    uint32_t root_layout;
    const uint16_t *fields;
    uint16_t field_count;
    uint32_t referent_layout;
    NvmReferenceMode mode;
} NvmReferencePlace;

/* I require complete authoritative layouts, not the count-only bridge
 * placeholders. I require the authoritative local layout separately: a producer cannot
 * relabel a root to claim a different referent. I accept only a record path
 * ending at a fixed record of scalar numeric/bool fields in this slice. */
bool nvm_reference_place_valid(const NvmV2Layouts *layouts,
                               uint32_t authoritative_root_layout,
                               const NvmReferencePlace *place);

/* These queries operate on already validated places. Structurally invalid descriptors
 * conservatively conflict. Root identity never depends on a claimed layout.
 * Empty paths cover the whole root. No query allocates or changes a place. */
bool nvm_reference_places_overlap(const NvmReferencePlace *a,
                                  const NvmReferencePlace *b);
bool nvm_reference_holds_conflict(const NvmReferencePlace *a,
                                 const NvmReferencePlace *b);
/* I check direct owner access, not access through an authorized reference.
 * A read conflicts with an exclusive hold; a write/move with either mode. */
bool nvm_reference_owner_access_conflicts(const NvmReferencePlace *hold,
                                          const NvmReferencePlace *access,
                                          bool write_or_move);
#endif
