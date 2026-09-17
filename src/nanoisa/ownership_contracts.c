#include "ownership_contracts.h"
#include "retained_layouts.h"
#include "reference_places.h"
#include "isa.h"

static bool scalar(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_U8 || tag == TAG_FLOAT || tag == TAG_BOOL;
}

static NvmV2Result check_layouts(const NvmV2Layouts *layouts, const uint8_t *flags,
                                bool *needs) {
    for (uint32_t i = 0; i < layouts->count; i++) {
        unsigned flag = flags[i];
        if (flag & ~(NVM_LAYOUT_COMPLETE | NVM_LAYOUT_RESOURCE))
            return NVM_V2_ERR_RESERVED_FLAGS;
        if ((flag & NVM_LAYOUT_RESOURCE) && !(flag & NVM_LAYOUT_COMPLETE))
            return NVM_V2_ERR_SECTION_TYPE;
        if (!(flag & NVM_LAYOUT_COMPLETE)) continue;
        const NvmV2Layout *layout = &layouts->items[i];
        /* My first complete contracts cover finite scalar/record trees.
         * Other shapes retain their layouts without this completeness claim. */
        if (layout->kind != NVM_V2_LAYOUT_STRUCT) return NVM_V2_ERR_SECTION_TYPE;
        for (uint16_t j = 0; j < layout->field_count; j++) {
            const NvmV2LayoutField *field = &layout->fields[j];
            if (scalar(field->type_tag)) {
                if (field->nested_idx != NVM_V2_NO_INDEX) return NVM_V2_ERR_SECTION_TYPE;
            } else if (field->type_tag == TAG_STRUCT && field->nested_idx < i) {
                unsigned nested = flags[field->nested_idx];
                if (!(nested & NVM_LAYOUT_COMPLETE)) return NVM_V2_ERR_SECTION_TYPE;
                if ((nested & NVM_LAYOUT_RESOURCE) && !(flag & NVM_LAYOUT_RESOURCE))
                    return NVM_V2_ERR_SECTION_TYPE;
            } else return NVM_V2_ERR_SECTION_TYPE;
        }
        if (flag & NVM_LAYOUT_RESOURCE) *needs = true;
    }
    return NVM_V2_OK;
}

static NvmV2Result descriptor(NvmV2Cursor *cursor, const NvmV2Layouts *layouts,
                               const uint8_t *flags, bool parameter,
                               int signature_tag, bool *needs) {
    uint8_t tag, mode;
    uint16_t reserved;
    uint32_t layout;
    NvmV2Result result;
    if ((result = nvm_v2_u8(cursor, &tag)) != NVM_V2_OK ||
        (result = nvm_v2_u8(cursor, &mode)) != NVM_V2_OK ||
        (result = nvm_v2_u16(cursor, &reserved)) != NVM_V2_OK ||
        (result = nvm_v2_u32(cursor, &layout)) != NVM_V2_OK) return result;
    if (reserved) return NVM_V2_ERR_RESERVED_FLAGS;
    if (tag >= TAG_COUNT || mode > NVM_REFERENCE_EXCLUSIVE ||
        (!parameter && mode) || (signature_tag >= 0 && tag != signature_tag))
        return NVM_V2_ERR_SECTION_TYPE;
    if (layout != NVM_V2_NO_INDEX) {
        if (layout >= layouts->count) return NVM_V2_ERR_INDEX_RANGE;
        if (tag != TAG_STRUCT || !(flags[layout] & NVM_LAYOUT_COMPLETE) ||
            layouts->items[layout].kind != NVM_V2_LAYOUT_STRUCT)
            return NVM_V2_ERR_SECTION_TYPE;
    }
    if (mode) {
        if (layout == NVM_V2_NO_INDEX || !(flags[layout] & NVM_LAYOUT_RESOURCE))
            return NVM_V2_ERR_SECTION_TYPE;
        NvmReferencePlace place = {1, 0, layout, NULL, 0, layout, (NvmReferenceMode)mode};
        if (!nvm_reference_place_valid(layouts, layout, &place))
            return NVM_V2_ERR_SECTION_TYPE;
        *needs = true;
    }
    return NVM_V2_OK;
}

NvmV2Result nvm_ownership_contracts_validate(const NvmModule *module,
                                            bool *requires_verifier) {
    if (!module || !requires_verifier) return NVM_V2_ERR_INDEX_RANGE;
    *requires_verifier = false;
    if (!module->ownership_size)
        return module->ownership_data ? NVM_V2_ERR_SECTION_RANGE : NVM_V2_OK;
    if (!module->ownership_data || !module->layout_size ||
        !nvm_retained_layouts_valid(module)) return NVM_V2_ERR_SECTION_TYPE;
    NvmV2Layouts layouts = {0};
    NvmV2Result result = nvm_v2_layouts_decode(module->layout_data,
                                             module->layout_size, &layouts);
    if (result != NVM_V2_OK) return result;
    NvmV2Cursor cursor;
    nvm_v2_cursor_init(&cursor, module->ownership_data, module->ownership_size);
    uint32_t version, count;
    const uint8_t *flags;
    bool needs = false;
    if ((result = nvm_v2_u32(&cursor, &version)) != NVM_V2_OK ||
        (result = nvm_v2_u32(&cursor, &count)) != NVM_V2_OK) goto done;
    if (version != NVM_OWNERSHIP_VERSION) { result = NVM_V2_ERR_FORMAT_VERSION; goto done; }
    if (count != layouts.count) { result = NVM_V2_ERR_INDEX_RANGE; goto done; }
    if ((result = nvm_v2_take(&cursor, count, &flags)) != NVM_V2_OK ||
        (result = nvm_v2_align4(&cursor)) != NVM_V2_OK ||
        (result = check_layouts(&layouts, flags, &needs)) != NVM_V2_OK ||
        (result = nvm_v2_u32(&cursor, &count)) != NVM_V2_OK) goto done;
    if (count != module->function_count || (count && !module->functions)) {
        result = NVM_V2_ERR_INDEX_RANGE; goto done;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint16_t locals, params;
        const NvmFunctionEntry *function = &module->functions[i];
        if ((result = nvm_v2_u16(&cursor, &locals)) != NVM_V2_OK ||
            (result = nvm_v2_u16(&cursor, &params)) != NVM_V2_OK) goto done;
        if (locals != function->local_count || params != function->arity || params > locals ||
            function->result_count > 1) { result = NVM_V2_ERR_INDEX_RANGE; goto done; }
        int return_tag = function->result_count ? function->result_tag : TAG_VOID;
        if ((result = descriptor(&cursor, &layouts, flags, false, return_tag, &needs))
            != NVM_V2_OK) goto done;
        for (uint16_t local = 0; local < locals; local++) {
            int tag = -1;
            if (local < params) tag = module->function_param_types &&
                module->function_param_types[i] ? module->function_param_types[i][local] : TAG_VOID;
            if ((result = descriptor(&cursor, &layouts, flags, local < params, tag, &needs))
                != NVM_V2_OK) goto done;
        }
    }
    if (cursor.pos != cursor.size) { result = NVM_V2_ERR_SECTION_RANGE; goto done; }
    *requires_verifier = needs;
done:
    nvm_v2_layouts_free(&layouts);
    return result;
}
