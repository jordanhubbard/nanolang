#include "reference_places.h"
#include "isa.h"

static bool descriptor_valid(const NvmReferencePlace *p) {
    return p && p->invocation != 0 &&
           (!p->field_count || p->fields) &&
           (p->mode == NVM_REFERENCE_SHARED ||
            p->mode == NVM_REFERENCE_EXCLUSIVE);
}

static bool scalar(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_U8 || tag == TAG_FLOAT || tag == TAG_BOOL;
}

static bool record_valid(const NvmV2Layout *layout) {
    return layout->kind == NVM_V2_LAYOUT_STRUCT &&
           (!layout->field_count || layout->fields);
}

bool nvm_reference_place_valid(const NvmV2Layouts *layouts,
                               uint32_t authoritative_root_layout,
                               const NvmReferencePlace *place) {
    if (!descriptor_valid(place) || !layouts || !layouts->items ||
        place->root_layout != authoritative_root_layout ||
        authoritative_root_layout >= layouts->count) return false;
    uint32_t current = authoritative_root_layout;
    for (uint16_t i = 0; i < place->field_count; i++) {
        const NvmV2Layout *layout = &layouts->items[current];
        if (!record_valid(layout) || place->fields[i] >= layout->field_count)
            return false;
        const NvmV2LayoutField *field = &layout->fields[place->fields[i]];
        /* v2 layout edges are strictly backward. I retain that invariant
         * even when this API is given producer-owned, not decoded, tables. */
        if (field->type_tag != TAG_STRUCT || field->nested_idx >= current)
            return false;
        current = field->nested_idx;
    }
    if (current != place->referent_layout) return false;
    const NvmV2Layout *referent = &layouts->items[current];
    if (!record_valid(referent)) return false;
    for (uint16_t i = 0; i < referent->field_count; i++) {
        const NvmV2LayoutField *field = &referent->fields[i];
        if (!scalar(field->type_tag) || field->nested_idx != NVM_V2_NO_INDEX)
            return false;
    }
    return true;
}

bool nvm_reference_places_overlap(const NvmReferencePlace *a,
                                  const NvmReferencePlace *b) {
    if (!descriptor_valid(a) || !descriptor_valid(b)) return true;
    if (a->invocation != b->invocation || a->local != b->local) return false;
    uint16_t common = a->field_count < b->field_count
                      ? a->field_count : b->field_count;
    for (uint16_t i = 0; i < common; i++)
        if (a->fields[i] != b->fields[i]) return false;
    return true;
}

bool nvm_reference_holds_conflict(const NvmReferencePlace *a,
                                 const NvmReferencePlace *b) {
    if (!descriptor_valid(a) || !descriptor_valid(b)) return true;
    return (a->mode == NVM_REFERENCE_EXCLUSIVE ||
            b->mode == NVM_REFERENCE_EXCLUSIVE) &&
           nvm_reference_places_overlap(a, b);
}

bool nvm_reference_owner_access_conflicts(const NvmReferencePlace *hold,
                                          const NvmReferencePlace *access,
                                          bool write_or_move) {
    if (!descriptor_valid(hold) || !descriptor_valid(access)) return true;
    return (write_or_move || hold->mode == NVM_REFERENCE_EXCLUSIVE) &&
           nvm_reference_places_overlap(hold, access);
}
