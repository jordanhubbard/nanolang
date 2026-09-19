#ifndef NANOISA_OWNED_ARRAY_LAYOUTS_H
#define NANOISA_OWNED_ARRAY_LAYOUTS_H
#include "managed_record_plan.h"

#define NVM_OWNED_ARRAY_MAX_PATHS 65536u
#define NVM_OWNED_ARRAY_MAX_PATH_DEPTH 32u
/* Private descriptive facts only. ARRAY leaves remain pending element-origin
 * proof; this query grants no affine state, scalar proof or runtime admission. */
typedef struct NvmOwnedArrayLayouts NvmOwnedArrayLayouts;
typedef struct {
    uint32_t layouts, source_records, managed_records, leaf_paths;
} NvmOwnedArrayLayoutCounts;
typedef struct {
    uint32_t global_layout, source_record, managed_record, field_count;
    uint32_t path_start, path_count;
    uint8_t flags, owner_depth;
    bool has_array, has_string, whole_root_borrow_unsuitable;
} NvmOwnedArrayLayoutFact;
typedef struct {
    uint32_t root_layout, terminal_layout;
    uint16_t length;
    uint8_t tag;
    uint16_t fields[NVM_OWNED_ARRAY_MAX_PATH_DEPTH];
} NvmOwnedArrayLeafPath;
typedef struct {
    const uint8_t *layouts, *ownership;
    uint32_t layout_size, ownership_size;
} NvmOwnedArrayTransport;
/* I borrow immutable module storage only during construction. The opaque result
 * owns every fact/transport byte. Failure preserves *out and all query outputs.
 * Returned transport pointers remain valid only for the result's lifetime. */
NvmRecordPlanResult nvm_describe_owned_array_layouts(const NvmModule *, NvmOwnedArrayLayouts **out);
void nvm_owned_array_layouts_free(NvmOwnedArrayLayouts *);
bool nvm_owned_array_layout_counts(const NvmOwnedArrayLayouts *, NvmOwnedArrayLayoutCounts *out);
bool nvm_owned_array_layout_fact(const NvmOwnedArrayLayouts *, uint32_t global, NvmOwnedArrayLayoutFact *out);
bool nvm_owned_array_source_layout(const NvmOwnedArrayLayouts *, uint32_t source, uint32_t *out_global);
bool nvm_owned_array_managed_layout(const NvmOwnedArrayLayouts *, uint32_t managed, uint32_t *out_global);
bool nvm_owned_array_leaf_path(const NvmOwnedArrayLayouts *, uint32_t path, NvmOwnedArrayLeafPath *out);
bool nvm_owned_array_layout_field(const NvmOwnedArrayLayouts *, uint32_t global, uint16_t field, NvmV2LayoutField *out);
bool nvm_owned_array_transport(const NvmOwnedArrayLayouts *, NvmOwnedArrayTransport *out);
#endif
