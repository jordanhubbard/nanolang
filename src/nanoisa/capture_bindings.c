#include "capture_bindings.h"
#include "isa.h"
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

static bool capture_extent(size_t *total, size_t amount, size_t limit) {
    if (*total > limit || amount > limit - *total) return false;
    *total += amount;
    return true;
}

static void capture_word(uint8_t **cursor, uint16_t value) {
    *(*cursor)++ = (uint8_t)value;
    *(*cursor)++ = (uint8_t)(value >> 8);
}

static void capture_wide(uint8_t **cursor, uint32_t value) {
    capture_word(cursor, (uint16_t)value);
    capture_word(cursor, (uint16_t)(value >> 16));
}

NvmCaptureResult nvm_capture_bindings_encode(const NvmCaptureBindings *bindings,
    const NvmModule *module, size_t limit, uint8_t **data, size_t *size) {
    if (!bindings || !module || !data || !size ||
        bindings->function_count != module->function_count ||
        (bindings->function_count && (!bindings->functions || !module->functions)) ||
        (bindings->site_count && !bindings->sites)) return NVM_CAPTURE_INVALID;
    if (bindings->function_count > limit / sizeof(NvmCaptureFunction))
        return NVM_CAPTURE_LIMIT;
    size_t tables = (size_t)bindings->function_count * sizeof(NvmCaptureFunction);
    if (bindings->site_count > (limit - tables) / sizeof(NvmCaptureSite))
        return NVM_CAPTURE_LIMIT;
    tables += (size_t)bindings->site_count * sizeof(NvmCaptureSite);
    size_t bytes = 0, available = limit - tables;
    if (!capture_extent(&bytes, 12, available)) return NVM_CAPTURE_LIMIT;
    for (uint32_t i = 0; i < bindings->function_count; ++i) {
        const NvmCaptureFunction *function = &bindings->functions[i];
        if ((function->local_count && !function->local_modes) ||
            (function->upvalue_count && !function->upvalue_modes))
            return NVM_CAPTURE_INVALID;
        if (!capture_extent(&bytes, 8, available) ||
            !capture_extent(&bytes, function->local_count, available) ||
            !capture_extent(&bytes, function->upvalue_count, available))
            return NVM_CAPTURE_LIMIT;
    }
    for (uint32_t i = 0; i < bindings->site_count; ++i) {
        const NvmCaptureSite *site = &bindings->sites[i];
        if (site->capture_count && !site->sources) return NVM_CAPTURE_INVALID;
        if (!capture_extent(&bytes, 16, available) ||
            !capture_extent(&bytes, (size_t)site->capture_count * 4, available))
            return NVM_CAPTURE_LIMIT;
    }
    uint8_t *payload = malloc(bytes);
    if (!payload) return NVM_CAPTURE_MEMORY;
    uint8_t *cursor = payload;
    capture_word(&cursor, NVM_CAPTURE_BINDINGS_VERSION);
    capture_word(&cursor, 0);
    capture_wide(&cursor, bindings->function_count);
    capture_wide(&cursor, bindings->site_count);
    for (uint32_t i = 0; i < bindings->function_count; ++i) {
        const NvmCaptureFunction *function = &bindings->functions[i];
        capture_wide(&cursor, i);
        capture_word(&cursor, function->local_count);
        capture_word(&cursor, function->upvalue_count);
        if (function->local_count) memcpy(cursor, function->local_modes, function->local_count);
        cursor += function->local_count;
        if (function->upvalue_count) memcpy(cursor, function->upvalue_modes, function->upvalue_count);
        cursor += function->upvalue_count;
    }
    for (uint32_t i = 0; i < bindings->site_count; ++i) {
        const NvmCaptureSite *site = &bindings->sites[i];
        capture_wide(&cursor, site->owner);
        capture_wide(&cursor, site->instruction_offset);
        capture_wide(&cursor, site->target);
        capture_word(&cursor, site->capture_count);
        capture_word(&cursor, 0);
        size_t sources = (size_t)site->capture_count * 4;
        if (sources) memcpy(cursor, site->sources, sources);
        cursor += sources;
    }
    NvmCaptureBindings validated = {0};
    NvmCaptureResult result = nvm_capture_bindings_decode(payload, bytes, module,
        limit - bytes, &validated);
    nvm_capture_bindings_free(&validated);
    if (result != NVM_CAPTURE_OK) { free(payload); return result; }
    *data = payload;
    *size = bytes;
    return NVM_CAPTURE_OK;
}

NvmCaptureResult nvm_capture_bindings_verify_code(const NvmCaptureBindings *bindings,
    const NvmModule *module, size_t work_limit) {
    if (!bindings || !module || bindings->function_count != module->function_count ||
        (bindings->function_count && (!bindings->functions || !module->functions)) ||
        (bindings->site_count && !bindings->sites) ||
        (module->code_size && !module->code)) return NVM_CAPTURE_INVALID;
    uint32_t next_site = 0;
    for (uint32_t i = 0; i < bindings->function_count; ++i) {
        const NvmFunctionEntry *function = &module->functions[i];
        const NvmCaptureFunction *binding = &bindings->functions[i];
        if (function->code_offset > module->code_size ||
            function->code_length > module->code_size - function->code_offset ||
            binding->local_count != function->local_count ||
            binding->upvalue_count != function->upvalue_count)
            return NVM_CAPTURE_INVALID;
        if (!work_limit || function->code_length > work_limit - 1)
            return NVM_CAPTURE_LIMIT;
        work_limit -= (size_t)function->code_length + 1;
        uint32_t position = 0;
        while (position < function->code_length) {
            DecodedInstruction instruction;
            uint32_t width = isa_decode(module->code + function->code_offset + position,
                function->code_length - position, &instruction);
            if (!width) return NVM_CAPTURE_INVALID;
            switch (instruction.opcode) {
            case OP_CALL: case OP_TAIL_CALL: case OP_FUNCREF: {
                /* I cannot provide an environment through a raw function index. */
                uint32_t target = instruction.operands[0].u32;
                if (target >= bindings->function_count ||
                    bindings->functions[target].upvalue_count)
                    return NVM_CAPTURE_INVALID;
                break;
            }
            case OP_CLOSURE_NEW:
                return NVM_CAPTURE_INVALID;
            case OP_BIND_INIT_LOCAL: case OP_BIND_CLEAR_LOCAL:
            case OP_LOAD_LOCAL: case OP_STORE_LOCAL: {
                uint16_t slot = instruction.operands[0].u16;
                if (slot >= binding->local_count ||
                    (instruction.opcode == OP_STORE_LOCAL &&
                     binding->local_modes[slot] != NVM_CAPTURE_SHARED))
                    return NVM_CAPTURE_INVALID;
                break;
            }
            case OP_LOAD_UPVALUE: case OP_STORE_UPVALUE: {
                uint16_t slot = instruction.operands[1].u16;
                if (instruction.operands[0].u16 || slot >= binding->upvalue_count ||
                    (instruction.opcode == OP_STORE_UPVALUE &&
                     binding->upvalue_modes[slot] != NVM_CAPTURE_SHARED))
                    return NVM_CAPTURE_INVALID;
                break;
            }
            case OP_CLOSURE_BIND:
                if (next_site >= bindings->site_count ||
                    instruction.operands[0].u32 != next_site ||
                    bindings->sites[next_site].owner != i ||
                    bindings->sites[next_site].instruction_offset != position)
                    return NVM_CAPTURE_INVALID;
                ++next_site;
                break;
            default:
                break;
            }
            position += width;
        }
        if (next_site < bindings->site_count && bindings->sites[next_site].owner <= i)
            return NVM_CAPTURE_INVALID;
    }
    return next_site == bindings->site_count ? NVM_CAPTURE_OK : NVM_CAPTURE_INVALID;
}

NvmCaptureResult nvm_capture_bindings_validate_module(const NvmModule *module) {
    if (!module) return NVM_CAPTURE_INVALID;
    if (!nvm_capture_bindings_present(module)) return NVM_CAPTURE_OK;
    if (!module->capture_data || !module->capture_size) return NVM_CAPTURE_INVALID;
    if (module->capture_size > NVM_CAPTURE_TRANSPORT_BYTES) return NVM_CAPTURE_LIMIT;
    NvmCaptureBindings bindings = {0};
    NvmCaptureResult result = nvm_capture_bindings_decode(module->capture_data,
        module->capture_size, module, NVM_CAPTURE_TRANSPORT_BYTES - module->capture_size, &bindings);
    if (result == NVM_CAPTURE_OK)
        result = nvm_capture_bindings_verify_code(&bindings, module, NVM_CAPTURE_TRANSPORT_WORK);
    nvm_capture_bindings_free(&bindings);
    return result;
}
