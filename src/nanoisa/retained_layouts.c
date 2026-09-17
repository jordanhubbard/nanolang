#include "retained_layouts.h"
#include <stdlib.h>

bool nvm_layouts_have_facts(const NvmV2Layouts *layouts) {
    if (!layouts || (layouts->count && !layouts->items)) return false;
    for (uint32_t i = 0; i < layouts->count; i++)
        if (layouts->items[i].name_idx != NVM_V2_NO_INDEX ||
            layouts->items[i].field_count) return true;
    return false;
}

static bool name_valid(uint32_t index, const NvmModule *module) {
    return index == NVM_V2_NO_INDEX || index < module->string_count;
}

static bool table_valid(const NvmModule *module, const NvmV2Layouts *layouts) {
    uint32_t records = 0, enums = 0, unions = 0;
    for (uint32_t i = 0; i < layouts->count; i++) {
        const NvmV2Layout *layout = &layouts->items[i];
        if (!name_valid(layout->name_idx, module)) return false;
        switch (layout->kind) {
        case NVM_V2_LAYOUT_STRUCT: records++; break;
        case NVM_V2_LAYOUT_ENUM: enums++; break;
        case NVM_V2_LAYOUT_UNION: unions++; break;
        case NVM_V2_LAYOUT_TUPLE: break;
        default: return false;
        }
        for (uint16_t j = 0; j < layout->field_count; j++)
            if (!name_valid(layout->fields[j].name_idx, module)) return false;
    }
    return records == module->struct_count && enums == module->enum_count &&
           unions == module->union_count;
}

bool nvm_retained_layouts_valid(const NvmModule *module) {
    if (!module) return false;
    if (!module->layout_size) return module->layout_data == NULL;
    if (!module->layout_data) return false;
    NvmV2Layouts layouts = {0};
    NvmV2Result result = nvm_v2_layouts_decode(module->layout_data,
                                             module->layout_size, &layouts);
    if (result != NVM_V2_OK) return false;
    bool valid = nvm_v2_layouts_encoded_size(&layouts) == module->layout_size &&
                 table_valid(module, &layouts);
    nvm_v2_layouts_free(&layouts);
    return valid;
}

NvmV2Result nvm_retain_layouts(NvmModule *module, const NvmV2Layouts *layouts) {
    if (!module || !layouts || (layouts->count && !layouts->items))
        return NVM_V2_ERR_INDEX_RANGE;
    /* The encoder expects addressable fields. I check that precondition
     * before using it; the decoder then checks the canonical table rules. */
    for (uint32_t i = 0; i < layouts->count; i++)
        if (layouts->items[i].field_count && !layouts->items[i].fields)
            return NVM_V2_ERR_INDEX_RANGE;
    size_t size = nvm_v2_layouts_encoded_size(layouts);
    if (size > UINT32_MAX) return NVM_V2_ERR_INDEX_RANGE;
    uint8_t *bytes = malloc(size);
    if (!bytes) return NVM_V2_ERR_TRUNCATED;
    NvmV2Result result = nvm_v2_layouts_encode(layouts, bytes, size);
    if (result != NVM_V2_OK) { free(bytes); return result; }
    NvmModule trial = *module;
    trial.layout_data = bytes;
    trial.layout_size = (uint32_t)size;
    if (!nvm_retained_layouts_valid(&trial)) {
        free(bytes);
        return NVM_V2_ERR_INDEX_RANGE;
    }
    free(module->layout_data);
    module->layout_data = bytes;
    module->layout_size = (uint32_t)size;
    return NVM_V2_OK;
}
