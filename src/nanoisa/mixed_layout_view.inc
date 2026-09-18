/* Private descriptive reader. I reuse only retained byte preflight, not authority. */
#include "mixed_layout_view.h"
#include <string.h>

static bool mixed_ownership_read(const NvmModule *m, uint32_t layout_count,
                                 const NvmV2Layouts *layouts, uint8_t *flags) {
    if (!m->ownership_size) return true;
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, m->ownership_data, m->ownership_size);
    uint32_t version, count;
    const uint8_t *raw;
    if (nvm_v2_u32(&c, &version) != NVM_V2_OK ||
        (version != NVM_OWNERSHIP_VERSION && version != NVM_OWNERSHIP_PATH_VERSION) ||
        nvm_v2_u32(&c, &count) != NVM_V2_OK || count != layout_count ||
        nvm_v2_take(&c, count, &raw) != NVM_V2_OK) return false;
    memcpy(flags, raw, count);
    for (uint32_t i = 0; i < count; i++)
        if ((flags[i] & ~(NVM_LAYOUT_COMPLETE | NVM_LAYOUT_RESOURCE)) ||
            ((flags[i] & NVM_LAYOUT_RESOURCE) && !(flags[i] & NVM_LAYOUT_COMPLETE)))
            return false;
    if (nvm_v2_align4(&c) != NVM_V2_OK || nvm_v2_u32(&c, &count) != NVM_V2_OK ||
        count != m->function_count || (count && !m->functions)) return false;
    for (uint32_t i = 0; i < count; i++) {
        uint16_t locals, params;
        const NvmFunctionEntry *f = &m->functions[i];
        if (nvm_v2_u16(&c, &locals) != NVM_V2_OK ||
            nvm_v2_u16(&c, &params) != NVM_V2_OK || locals != f->local_count ||
            params != f->arity || params > locals || f->result_count > 1) return false;
        for (uint32_t j = 0; j <= locals; j++) {
            uint8_t tag, mode;
            uint16_t reserved;
            uint32_t layout;
            int expected = j == 0 ? (f->result_count ? f->result_tag : TAG_VOID) : -1;
            bool parameter = j > 0 && j <= params;
            if (parameter) expected = m->function_param_types && m->function_param_types[i] ?
                m->function_param_types[i][j - 1] : TAG_VOID;
            if (nvm_v2_u8(&c, &tag) != NVM_V2_OK || nvm_v2_u8(&c, &mode) != NVM_V2_OK ||
                nvm_v2_u16(&c, &reserved) != NVM_V2_OK ||
                nvm_v2_u32(&c, &layout) != NVM_V2_OK || reserved || tag >= TAG_COUNT ||
                mode > 2 || (!parameter && mode) || (expected >= 0 && tag != expected)) return false;
            if (layout != NVM_V2_NO_INDEX &&
                (layout >= layout_count || tag != TAG_STRUCT || !(flags[layout] & NVM_LAYOUT_COMPLETE) ||
                 (layouts && layouts->items[layout].kind != NVM_V2_LAYOUT_STRUCT))) return false;
            if (mode && (layout == NVM_V2_NO_INDEX || !(flags[layout] & NVM_LAYOUT_RESOURCE)))
                return false;
        }
    }
    if (version == NVM_OWNERSHIP_PATH_VERSION) {
        if (nvm_v2_u32(&c, &count) != NVM_V2_OK || count > NVM_OWNERSHIP_MAX_PATHS) return false;
        for (uint32_t i = 0; i < count; i++) {
            uint16_t n, reserved, field;
            if (nvm_v2_u16(&c, &n) != NVM_V2_OK || nvm_v2_u16(&c, &reserved) != NVM_V2_OK ||
                reserved || !n || n > NVM_OWNERSHIP_MAX_PATH_DEPTH) return false;
            for (uint16_t j = 0; j < n; j++)
                if (nvm_v2_u16(&c, &field) != NVM_V2_OK) return false;
            if (nvm_v2_align4(&c) != NVM_V2_OK) return false;
        }
    }
    return c.pos == c.size;
}

void nvm_mixed_layout_view_free(NvmMixedLayoutView *v) {
    if (!v) return;
    nvm_v2_layouts_free(&v->layouts);
    free(v->ownership);
    free(v);
}

/* I resolve only acyclic dependencies; a visiting child is invalid, never ordinary. */
static bool mixed_classify(NvmMixedLayoutView *v, uint32_t i, const uint8_t *flags,
                           uint8_t *colors, uint8_t *scalar_trees) {
    if (colors[i] == 2) return true;
    if (colors[i] == 1) return false;
    colors[i] = 1;
    const NvmV2Layout *l = &v->layouts.items[i];
    bool complete = (flags[i] & NVM_LAYOUT_COMPLETE) != 0;
    bool resource = (flags[i] & NVM_LAYOUT_RESOURCE) != 0;
    bool unknown = !complete, pending = false, scalar_tree = true;
    if (complete && l->kind != NVM_V2_LAYOUT_STRUCT) return false;
    if (l->kind != NVM_V2_LAYOUT_STRUCT) {
        colors[i] = 2;
        return true; /* Unsupported non-record declarations remain UNKNOWN. */
    }
    for (uint16_t j = 0; j < l->field_count; j++) {
        const NvmV2LayoutField *f = &l->fields[j];
        uint32_t child = f->nested_idx;
        if (f->type_tag == TAG_STRUCT) {
            if (child == NVM_V2_NO_INDEX || v->layouts.items[child].kind != NVM_V2_LAYOUT_STRUCT ||
                !mixed_classify(v, child, flags, colors, scalar_trees)) return false;
            if (complete && (flags[child] & NVM_LAYOUT_RESOURCE) && !resource) return false;
            unknown |= v->classes[child] == NVM_MIXED_UNKNOWN;
            pending |= v->classes[child] == NVM_MIXED_PENDING_ARRAY_PROOF;
            scalar_tree &= scalar_trees[child] != 0;
        } else {
            if (child != NVM_V2_NO_INDEX &&
                (f->type_tag != TAG_ENUM || v->layouts.items[child].kind != NVM_V2_LAYOUT_ENUM)) return false;
            bool scalar = f->type_tag == TAG_INT || f->type_tag == TAG_U8 ||
                          f->type_tag == TAG_FLOAT || f->type_tag == TAG_BOOL;
            scalar_tree &= scalar;
            pending |= f->type_tag == TAG_ARRAY;
            unknown |= !scalar && f->type_tag != TAG_STRING && f->type_tag != TAG_ARRAY;
        }
    }
    if (resource && (!scalar_tree || unknown || pending)) return false;
    scalar_trees[i] = scalar_tree && complete && !unknown;
    v->classes[i] = unknown ? NVM_MIXED_UNKNOWN : resource ? NVM_MIXED_RESOURCE :
        pending ? NVM_MIXED_PENDING_ARRAY_PROOF : NVM_MIXED_ORDINARY_STRUCTURAL;
    colors[i] = 2;
    return true;
}

NvmRecordPlanResult nvm_describe_mixed_layouts(const NvmModule *m, NvmMixedLayoutView **out) {
    if (!m || !out || (!m->layout_data != !m->layout_size) ||
        (!m->ownership_data != !m->ownership_size))
        return record_result(NVM_RECORD_INVALID, 0, 0, "I require consistent mixed descriptor inputs.");
    if (!m->layout_size)
        return record_result(NVM_RECORD_UNRESOLVED, 0, 0, "I require retained mixed layout facts.");
    if (m->ownership_size > NVM_MIXED_MAX_OWNERSHIP_BYTES)
        return record_result(NVM_RECORD_LIMIT, 0, 0, "I reached my mixed ownership byte budget.");
    bool forward;
    NvmRecordPlanResult r = record_preflight(m, &forward);
    if (r.status != NVM_RECORD_DESCRIBED) return r;
    NvmV2Cursor c;
    uint32_t count;
    nvm_v2_cursor_init(&c, m->layout_data, m->layout_size);
    if (nvm_v2_u32(&c, &count) != NVM_V2_OK)
        return record_result(NVM_RECORD_INVALID, 0, 0, "I require a complete mixed layout count.");
    uint8_t flags[NVM_RECORD_PLAN_MAX_LAYOUTS] = {0};
    if (!mixed_ownership_read(m, count, NULL, flags))
        return record_result(NVM_RECORD_INVALID, 0, 0, "I require exact ownership transport before allocation.");
    NvmMixedLayoutView *v = calloc(1, sizeof *v);
    if (!v) return record_result(NVM_RECORD_MEMORY, 0, 0, "I could not allocate my mixed descriptor view.");
    NvmV2Result decoded = nvm_v2_layouts_decode(m->layout_data, m->layout_size, &v->layouts);
    if (decoded != NVM_V2_OK) {
        r = record_result(decoded == NVM_V2_ERR_TRUNCATED ? NVM_RECORD_MEMORY : NVM_RECORD_INVALID,
                          0, 0, "I could not materialize my checked retained layouts.");
        goto fail;
    }
    if (!mixed_ownership_read(m, count, &v->layouts, flags)) goto invalid;
    bool resources = false;
    for (uint32_t i = 0; i < count; i++) resources |= (flags[i] & NVM_LAYOUT_RESOURCE) != 0;
    if (resources && forward) goto invalid;
    uint8_t colors[NVM_RECORD_PLAN_MAX_LAYOUTS] = {0};
    uint8_t scalar_trees[NVM_RECORD_PLAN_MAX_LAYOUTS] = {0};
    for (uint32_t i = 0; i < count; i++)
        if (!mixed_classify(v, i, flags, colors, scalar_trees)) goto invalid;
    for (uint32_t i = 0; i < NVM_RECORD_PLAN_MAX_LAYOUTS; i++) {
        v->source_to_global[i] = v->global_to_source[i] = NVM_V2_NO_INDEX;
        v->managed_to_global[i] = v->global_to_managed[i] = NVM_V2_NO_INDEX;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (v->layouts.items[i].kind != NVM_V2_LAYOUT_STRUCT) continue;
        v->global_to_source[i] = v->record_count;
        v->source_to_global[v->record_count++] = i;
        if (v->classes[i] == NVM_MIXED_ORDINARY_STRUCTURAL) {
            v->global_to_managed[i] = v->managed_count;
            v->managed_to_global[v->managed_count++] = i;
        }
    }
    if (m->ownership_size) {
        v->ownership = malloc(m->ownership_size);
        if (!v->ownership) {
            r = record_result(NVM_RECORD_MEMORY, 0, 0, "I could not retain my exact ownership descriptors.");
            goto fail;
        }
        memcpy(v->ownership, m->ownership_data, m->ownership_size);
        v->ownership_size = m->ownership_size;
    }
    *out = v;
    return record_result(NVM_RECORD_DESCRIBED, 0, 0, "I described mixed identities without executable authority.");
invalid:
    r = record_result(NVM_RECORD_INVALID, 0, 0, "I require consistent acyclic mixed declaration facts.");
fail:
    nvm_mixed_layout_view_free(v);
    return r;
}
