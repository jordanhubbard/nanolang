/* I expose the production shared declaration validator, not an imitation. */
#include "../src/nanoisa/ownership_contracts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t read32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static bool take(size_t *cursor, size_t length, size_t size, size_t *start) {
    if (length > size - *cursor) return false;
    *start = *cursor; *cursor += length;
    return true;
}
static void analyze(const unsigned char *bytes, size_t size) {
    if (size < 28) { puts("99:0"); return; }
    uint32_t layout_size = read32(bytes), own_size = read32(bytes + 4);
    uint32_t strings = read32(bytes + 8), functions = read32(bytes + 12);
    size_t cursor = 28;
    if (strings > (size - cursor) / 4) { puts("99:0"); return; }
    NvmModule model = {0};
    model.struct_count = read32(bytes + 16); model.enum_count = read32(bytes + 20);
    model.union_count = read32(bytes + 24);
    char **names = calloc(strings ? strings : 1, sizeof(*names));
    uint32_t *lengths = calloc(strings ? strings : 1, sizeof(*lengths));
    NvmFunctionEntry *entries = NULL; uint8_t **params = NULL;
    bool ok = names && lengths;
    for (uint32_t i = 0; ok && i < strings; ++i) {
        size_t at;
        if (!take(&cursor, 4, size, &at)) { ok = false; break; }
        uint32_t length = read32(bytes + at);
        if (!take(&cursor, length, size, &at)) { ok = false; break; }
        names[i] = malloc((size_t)length + 1);
        if (!names[i]) { ok = false; break; }
        memcpy(names[i], bytes + at, length); names[i][length] = 0; lengths[i] = length;
    }
    if (ok && functions > (size - cursor) / 32) ok = false;
    if (ok) {
        entries = calloc(functions ? functions : 1, sizeof(*entries));
        params = calloc(functions ? functions : 1, sizeof(*params));
        if (!entries || !params) ok = false;
    }
    for (uint32_t i = 0; ok && i < functions; ++i) {
        size_t at;
        if (!take(&cursor, 32, size, &at)) { ok = false; break; }
        const unsigned char *p = bytes + at;
        uint32_t arity = read32(p + 8), locals = read32(p + 12), upvalues = read32(p + 16);
        uint32_t results = read32(p + 20), result_tag = read32(p + 24), types = read32(p + 28);
        if (arity > UINT16_MAX || locals > UINT16_MAX || upvalues > UINT16_MAX || results > UINT8_MAX ||
            result_tag > UINT8_MAX || (types && types != arity) || !take(&cursor, types, size, &at)) { ok = false; break; }
        entries[i].code_offset = read32(p); entries[i].code_length = read32(p + 4);
        entries[i].arity = (uint16_t)arity; entries[i].local_count = (uint16_t)locals;
        entries[i].upvalue_count = (uint16_t)upvalues; entries[i].result_count = (uint8_t)results;
        entries[i].result_tag = (uint8_t)result_tag;
        params[i] = types ? (uint8_t *)(bytes + at) : NULL;
    }
    size_t layout_at = 0, own_at = 0;
    if (ok && (!take(&cursor, layout_size, size, &layout_at) ||
               !take(&cursor, own_size, size, &own_at) || size - cursor > UINT32_MAX)) ok = false;
    if (ok) {
        model.strings = names; model.string_lengths = lengths; model.string_count = strings;
        model.functions = entries; model.function_param_types = params; model.function_count = functions;
        model.layout_data = layout_size ? (uint8_t *)(bytes + layout_at) : NULL; model.layout_size = layout_size;
        model.ownership_data = own_size ? (uint8_t *)(bytes + own_at) : NULL; model.ownership_size = own_size;
        model.code_size = (uint32_t)(size - cursor); model.code = model.code_size ? (uint8_t *)(bytes + cursor) : NULL;
        bool needs = true;
        NvmV2Result result = nvm_ownership_contracts_validate(&model, &needs);
        printf("%u:%u\n", (unsigned)result, needs ? 1u : 0u);
    } else puts("99:0");
    if (names) for (uint32_t i = 0; i < strings; ++i) free(names[i]);
    free(names); free(lengths); free(entries); free(params);
}
int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        FILE *input = fopen(argv[i], "rb");
        if (!input || fseek(input, 0, SEEK_END)) return 2;
        long size = ftell(input);
        if (size < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(size ? (size_t)size : 1);
        if (!bytes) return 2;
        if (fread(bytes, 1, (size_t)size, input) != (size_t)size || fclose(input)) return 2;
        analyze(bytes, (size_t)size); free(bytes);
    }
    return 0;
}
