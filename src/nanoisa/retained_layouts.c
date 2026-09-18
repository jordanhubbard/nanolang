#include "retained_layouts.h"
#include "ownership_contracts.h"
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

/* My non-admitting record view reuses the retained table codec. I preflight
 * bytes without allocation so codec allocation failure is distinguishable
 * from a truncated input, despite the codec's older shared error enum. */
#include "managed_record_plan.h"
#include "isa.h"

static NvmRecordPlanResult record_result(NvmRecordPlanStatus status,
                                         uint32_t layout, uint32_t field,
                                         const char *message) {
    NvmRecordPlanResult result = {status, layout, field, message};
    return result;
}
static NvmRecordPlanResult record_preflight(const NvmModule *module) {
    NvmV2Cursor cursor;
    nvm_v2_cursor_init(&cursor, module->layout_data, module->layout_size);
    uint32_t count, records = 0, enums = 0, unions = 0, total_fields = 0;
    if (nvm_v2_u32(&cursor, &count) != NVM_V2_OK)
        return record_result(NVM_RECORD_INVALID, 0, 0, "I require a complete layout count.");
    if (count > NVM_RECORD_PLAN_MAX_LAYOUTS)
        return record_result(NVM_RECORD_LIMIT, 0, 0, "I reached my record layout limit.");
    for (uint32_t i = 0; i < count; i++) {
        uint8_t kind, reserved;
        uint16_t fields;
        uint32_t name;
        if (nvm_v2_u8(&cursor, &kind) != NVM_V2_OK ||
            nvm_v2_u8(&cursor, &reserved) != NVM_V2_OK ||
            nvm_v2_u16(&cursor, &fields) != NVM_V2_OK ||
            nvm_v2_u32(&cursor, &name) != NVM_V2_OK || reserved ||
            kind > NVM_V2_LAYOUT_KIND_MAX || !name_valid(name, module))
            return record_result(NVM_RECORD_INVALID, i, 0, "I require canonical layout headers and names.");
        records += kind == NVM_V2_LAYOUT_STRUCT;
        enums += kind == NVM_V2_LAYOUT_ENUM;
        unions += kind == NVM_V2_LAYOUT_UNION;
        if (fields > NVM_RECORD_PLAN_MAX_FIELDS - total_fields)
            return record_result(NVM_RECORD_LIMIT, i, 0, "I reached my retained field limit.");
        total_fields += fields;
        for (uint16_t j = 0; j < fields; j++) {
            uint8_t tag, a, b, c;
            uint32_t nested, field_name;
            if (nvm_v2_u8(&cursor, &tag) != NVM_V2_OK ||
                nvm_v2_u8(&cursor, &a) != NVM_V2_OK ||
                nvm_v2_u8(&cursor, &b) != NVM_V2_OK ||
                nvm_v2_u8(&cursor, &c) != NVM_V2_OK ||
                nvm_v2_u32(&cursor, &nested) != NVM_V2_OK ||
                nvm_v2_u32(&cursor, &field_name) != NVM_V2_OK ||
                a || b || c || tag >= TAG_COUNT ||
                (nested != NVM_V2_NO_INDEX && nested >= i) || !name_valid(field_name, module))
                return record_result(NVM_RECORD_INVALID, i, j, "I require canonical closed fields and names.");
        }
    }
    if (cursor.pos != cursor.size || records != module->struct_count ||
        enums != module->enum_count || unions != module->union_count)
        return record_result(NVM_RECORD_INVALID, 0, 0, "I require exact retained bytes and per-kind counts.");
    return record_result(NVM_RECORD_DESCRIBED, 0, 0, "I validated only retained byte structure.");
}
void nvm_record_plan_free(NvmRecordPlan *plan) {
    if (!plan) return;
    nvm_v2_layouts_free(&plan->layouts);
    free(plan->record_to_layout);
    free(plan->layout_to_record);
    free(plan);
}
NvmRecordPlanResult nvm_describe_managed_records(const NvmModule *module,
                                                NvmRecordPlan **out) {
    if (!module || !out || (!module->layout_data != !module->layout_size) ||
        (!module->ownership_data != !module->ownership_size))
        return record_result(NVM_RECORD_INVALID, 0, 0, "I require a consistent module and plan output.");
    if (!module->layout_size)
        return record_result(NVM_RECORD_UNRESOLVED, 0, 0, "I require retained layout facts, not count-only placeholders.");
    NvmRecordPlanResult result = record_preflight(module);
    if (result.status != NVM_RECORD_DESCRIBED) return result;
    NvmV2Layouts layouts = {0};
    NvmV2Result decoded = nvm_v2_layouts_decode(module->layout_data, module->layout_size, &layouts);
    if (decoded != NVM_V2_OK)
        return record_result(decoded == NVM_V2_ERR_TRUNCATED ? NVM_RECORD_MEMORY : NVM_RECORD_INVALID,
                             0, 0, "I could not own the preflighted retained fields.");
    for (uint32_t i = 0; i < layouts.count; i++) {
        const NvmV2Layout *layout = &layouts.items[i];
        if (layout->kind != NVM_V2_LAYOUT_STRUCT) continue;
        for (uint16_t j = 0; j < layout->field_count; j++) {
            const NvmV2LayoutField *field = &layout->fields[j];
            bool supported = field->type_tag == TAG_VOID || field->type_tag == TAG_INT ||
                field->type_tag == TAG_U8 || field->type_tag == TAG_FLOAT ||
                field->type_tag == TAG_BOOL || field->type_tag == TAG_STRING ||
                field->type_tag == TAG_ARRAY || field->type_tag == TAG_ENUM ||
                field->type_tag == TAG_STRUCT;
            if (!supported) {
                result = record_result(NVM_RECORD_UNRESOLVED, i, j, "I have not described this record field kind.");
                goto fail;
            }
            uint32_t nested = field->nested_idx;
            bool valid = field->type_tag == TAG_STRUCT ?
                nested != NVM_V2_NO_INDEX && layouts.items[nested].kind == NVM_V2_LAYOUT_STRUCT :
                nested == NVM_V2_NO_INDEX ||
                (field->type_tag == TAG_ENUM && layouts.items[nested].kind == NVM_V2_LAYOUT_ENUM);
            if (!valid) {
                result = record_result(NVM_RECORD_INVALID, i, j, "I require nested kind identity to match the field tag.");
                goto fail;
            }
        }
    }
    NvmRecordAuthority authority = NVM_RECORD_AUTHORITY_UNKNOWN;
    if (module->ownership_size) {
        NvmLayoutAuthority declared[NVM_RECORD_PLAN_MAX_LAYOUTS];
        if (nvm_ownership_layout_authorities(module, layouts.count, declared) != NVM_V2_OK) {
            /* The legacy validator conflates some invalid/allocation outcomes. */
            result = record_result(NVM_RECORD_UNRESOLVED, 0, 0, "I could not establish checked ownership declarations.");
            goto fail;
        }
        for (uint32_t i = 0; i < layouts.count; i++) {
            if (layouts.items[i].kind == NVM_V2_LAYOUT_STRUCT &&
                declared[i] != NVM_LAYOUT_AUTHORITY_ORDINARY) {
                result = record_result(NVM_RECORD_UNRESOLVED, i, 0, "I require explicit ordinary authority for every record.");
                goto fail;
            }
        }
        authority = NVM_RECORD_AUTHORITY_ORDINARY;
    }
    NvmRecordPlan *plan = calloc(1, sizeof *plan);
    if (!plan) { result = record_result(NVM_RECORD_MEMORY, 0, 0, "I could not allocate my record plan."); goto fail; }
    plan->record_count = module->struct_count;
    if (module->struct_count) plan->record_to_layout = calloc(module->struct_count, sizeof(uint32_t));
    if (layouts.count) plan->layout_to_record = calloc(layouts.count, sizeof(uint32_t));
    if ((module->struct_count && !plan->record_to_layout) || (layouts.count && !plan->layout_to_record)) {
        nvm_record_plan_free(plan);
        result = record_result(NVM_RECORD_MEMORY, 0, 0, "I could not allocate my exact record maps."); goto fail;
    }
    for (uint32_t i = 0, record = 0; i < layouts.count; i++) {
        plan->layout_to_record[i] = NVM_V2_NO_INDEX;
        if (layouts.items[i].kind == NVM_V2_LAYOUT_STRUCT) {
            plan->record_to_layout[record] = i;
            plan->layout_to_record[i] = record++;
        }
    }
    plan->layouts = layouts;
    plan->authority = authority;
    *out = plan;
    return record_result(NVM_RECORD_DESCRIBED, 0, 0, "I described record identities without executable admission.");
fail:
    nvm_v2_layouts_free(&layouts);
    return result;
}
