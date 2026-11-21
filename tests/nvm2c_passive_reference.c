/* I call the actual production passive validator on explicit bounded inputs. */
#include "../src/nanoisa/passive.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t read32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static bool analyze(const unsigned char *bytes, size_t size) {
    if (size < 8) return false;
    uint32_t payload = read32(bytes), count = read32(bytes + 4);
    if (count > (size - 8) / 28) return false;
    NvmFunctionEntry *functions = calloc(count ? count : 1, sizeof(*functions));
    uint8_t **params = calloc(count ? count : 1, sizeof(*params));
    if (!functions || !params) { free(functions); free(params); return false; }
    bool ok = true;
    size_t cursor = 8;
    for (uint32_t i = 0; i < count; ++i) {
        if (size - cursor < 28) { ok = false; break; }
        const unsigned char *p = bytes + cursor;
        uint32_t arity = read32(p + 8), locals = read32(p + 12);
        uint32_t upvalues = read32(p + 16), results = read32(p + 20), tags = read32(p + 24);
        cursor += 28;
        if (arity > UINT16_MAX || locals > UINT16_MAX || upvalues > UINT16_MAX || results > UINT8_MAX ||
            (tags && tags != arity) || tags > size - cursor) { ok = false; break; }
        functions[i].code_offset = read32(p); functions[i].code_length = read32(p + 4);
        functions[i].arity = (uint16_t)arity; functions[i].local_count = (uint16_t)locals;
        functions[i].upvalue_count = (uint16_t)upvalues; functions[i].result_count = (uint8_t)results;
        params[i] = tags ? (uint8_t *)(bytes + cursor) : NULL;
        cursor += tags;
    }
    if (ok && (payload > size - cursor || size - cursor - payload > UINT32_MAX)) ok = false;
    bool result = false;
    if (ok) {
        NvmModule module = {0};
        module.functions = functions; module.function_count = count; module.function_param_types = params;
        module.passive_data = payload ? (uint8_t *)(bytes + cursor) : NULL; module.passive_size = payload;
        cursor += payload;
        module.code_size = (uint32_t)(size - cursor);
        module.code = module.code_size ? (uint8_t *)(bytes + cursor) : NULL;
        result = nvm_passive_valid(&module);
    }
    free(functions); free(params);
    return result;
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
        printf("%u\n", analyze(bytes, (size_t)size));
        free(bytes);
    }
    return 0;
}
