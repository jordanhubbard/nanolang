#include "capture_bindings.h"
#include "nvm_v2_sections.h"
#include <stdlib.h>
#include <string.h>

void nvm_capture_bindings_free(NvmCaptureBindings *bindings) {
    if (!bindings) return;
    free(bindings->functions);
    free(bindings->sites);
    memset(bindings, 0, sizeof(*bindings));
}

bool nvm_capture_source(const NvmCaptureSite *site, uint16_t index,
    uint8_t *kind, uint8_t *mode, uint16_t *slot) {
    if (!site || !site->sources || index >= site->capture_count ||
        !kind || !mode || !slot) return false;
    const uint8_t *source = site->sources + (size_t)index * 4;
    *kind = source[0];
    *mode = source[1];
    *slot = (uint16_t)((uint16_t)source[2] | (uint16_t)source[3] << 8);
    return true;
}

static bool modes_valid(const uint8_t *modes, uint16_t count) {
    for (uint16_t i = 0; i < count; ++i)
        if (modes[i] > NVM_CAPTURE_SHARED) return false;
    return true;
}

NvmCaptureResult nvm_capture_bindings_decode(const uint8_t *data, size_t size,
    const NvmModule *module, size_t limit, NvmCaptureBindings *out) {
    if (!data || !module || !out ||
        (module->function_count && !module->functions)) return NVM_CAPTURE_INVALID;
    NvmV2Cursor cursor;
    nvm_v2_cursor_init(&cursor, data, size);
    uint16_t version, reserved;
    uint32_t functions, sites;
    if (nvm_v2_u16(&cursor, &version) != NVM_V2_OK ||
        nvm_v2_u16(&cursor, &reserved) != NVM_V2_OK ||
        nvm_v2_u32(&cursor, &functions) != NVM_V2_OK ||
        nvm_v2_u32(&cursor, &sites) != NVM_V2_OK ||
        version != NVM_CAPTURE_BINDINGS_VERSION || reserved ||
        functions != module->function_count) return NVM_CAPTURE_INVALID;

    /* I establish minimum wire sizes before multiplying or allocating. */
    size_t remaining = size - cursor.pos;
    if (functions > remaining / 8) return NVM_CAPTURE_INVALID;
    remaining -= (size_t)functions * 8;
    if (sites > remaining / 16) return NVM_CAPTURE_INVALID;
    if (functions > limit / sizeof(NvmCaptureFunction)) return NVM_CAPTURE_LIMIT;
    size_t function_bytes = (size_t)functions * sizeof(NvmCaptureFunction);
    if (sites > (limit - function_bytes) / sizeof(NvmCaptureSite))
        return NVM_CAPTURE_LIMIT;
    size_t site_bytes = (size_t)sites * sizeof(NvmCaptureSite);
    NvmCaptureBindings parsed = {0};
    parsed.function_count = functions;
    parsed.site_count = sites;
    parsed.allocation_bytes = function_bytes + site_bytes;
    if (functions) parsed.functions = calloc(functions, sizeof(*parsed.functions));
    if (sites) parsed.sites = calloc(sites, sizeof(*parsed.sites));
    if ((functions && !parsed.functions) || (sites && !parsed.sites)) {
        nvm_capture_bindings_free(&parsed);
        return NVM_CAPTURE_MEMORY;
    }

    for (uint32_t i = 0; i < functions; ++i) {
        NvmCaptureFunction *function = &parsed.functions[i];
        uint32_t index;
        const NvmFunctionEntry *declared = &module->functions[i];
        if (nvm_v2_u32(&cursor, &index) != NVM_V2_OK || index != i ||
            nvm_v2_u16(&cursor, &function->local_count) != NVM_V2_OK ||
            nvm_v2_u16(&cursor, &function->upvalue_count) != NVM_V2_OK ||
            function->local_count != declared->local_count ||
            function->upvalue_count != declared->upvalue_count ||
            declared->arity > declared->local_count ||
            nvm_v2_take(&cursor, function->local_count, &function->local_modes) != NVM_V2_OK ||
            nvm_v2_take(&cursor, function->upvalue_count, &function->upvalue_modes) != NVM_V2_OK ||
            !modes_valid(function->local_modes, function->local_count) ||
            !modes_valid(function->upvalue_modes, function->upvalue_count)) goto invalid;
    }
    for (uint32_t i = 0; i < sites; ++i) {
        NvmCaptureSite *site = &parsed.sites[i];
        if (nvm_v2_u32(&cursor, &site->owner) != NVM_V2_OK ||
            nvm_v2_u32(&cursor, &site->instruction_offset) != NVM_V2_OK ||
            nvm_v2_u32(&cursor, &site->target) != NVM_V2_OK ||
            nvm_v2_u16(&cursor, &site->capture_count) != NVM_V2_OK ||
            nvm_v2_u16(&cursor, &reserved) != NVM_V2_OK || reserved ||
            site->owner >= functions || site->target >= functions) goto invalid;
        if (i && (site->owner < parsed.sites[i - 1].owner ||
            (site->owner == parsed.sites[i - 1].owner &&
             site->instruction_offset <= parsed.sites[i - 1].instruction_offset))) goto invalid;
        const NvmFunctionEntry *owner = &module->functions[site->owner];
        const NvmCaptureFunction *source = &parsed.functions[site->owner];
        const NvmCaptureFunction *target = &parsed.functions[site->target];
        /* Five bytes are necessary for CLOSURE_BIND, but the future instruction
         * validator must establish the exact boundary/opcode/site index. */
        if (owner->code_offset > module->code_size ||
            owner->code_length > module->code_size - owner->code_offset ||
            site->instruction_offset > owner->code_length ||
            owner->code_length - site->instruction_offset < 5 ||
            site->capture_count != target->upvalue_count ||
            nvm_v2_take(&cursor, (size_t)site->capture_count * 4,
                        &site->sources) != NVM_V2_OK) goto invalid;
        for (uint16_t j = 0; j < site->capture_count; ++j) {
            uint8_t kind, mode;
            uint16_t slot;
            if (!nvm_capture_source(site, j, &kind, &mode, &slot) ||
                kind > NVM_CAPTURE_UPVALUE || mode > NVM_CAPTURE_SHARED ||
                mode != target->upvalue_modes[j]) goto invalid;
            const uint8_t *modes = kind == NVM_CAPTURE_LOCAL ? source->local_modes : source->upvalue_modes;
            uint16_t count = kind == NVM_CAPTURE_LOCAL ? source->local_count : source->upvalue_count;
            if (slot >= count || mode != modes[slot]) goto invalid;
        }
    }
    if (!nvm_v2_cursor_exhausted(&cursor)) goto invalid;
    *out = parsed;
    return NVM_CAPTURE_OK;
invalid:
    nvm_capture_bindings_free(&parsed);
    return NVM_CAPTURE_INVALID;
}
