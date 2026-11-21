/* I expose the existing private analyzer, without duplicating its decisions. */
#include "../src/nanoisa/passive.c"
#include <stdio.h>
#include <string.h>

static uint32_t read32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static bool analyze(const unsigned char *bytes, size_t size) {
    if (size < 8) return false;
    uint32_t root = read32(bytes), count = read32(bytes + 4);
    if (count > (size - 8) / 24) return false;
    NvmFunctionEntry *functions = calloc(count ? count : 1, sizeof(*functions));
    uint8_t *state = calloc(count ? count : 1, 1);
    if (!functions || !state) { free(functions); free(state); return false; }
    bool ok = true;
    for (uint32_t i = 0; i < count; ++i) {
        const unsigned char *p = bytes + 8 + (size_t)i * 24;
        if (read32(p + 8) > UINT16_MAX || read32(p + 12) > UINT16_MAX ||
            read32(p + 16) > UINT16_MAX || read32(p + 20) > UINT8_MAX) { ok = false; break; }
        functions[i].code_offset = read32(p);
        functions[i].code_length = read32(p + 4);
        functions[i].arity = (uint16_t)read32(p + 8);
        functions[i].local_count = (uint16_t)read32(p + 12);
        functions[i].upvalue_count = (uint16_t)read32(p + 16);
        functions[i].result_count = (uint8_t)read32(p + 20);
    }
    size_t begin = 8 + (size_t)count * 24;
    if (size - begin > UINT32_MAX) ok = false;
    NvmModule module = {0};
    module.functions = functions; module.function_count = count;
    module.code = (uint8_t *)(bytes + begin); module.code_size = (uint32_t)(size - begin);
    ClosedCalls calls = {&module, state, 0};
    bool result = ok && closed_function(&calls, root);
    free(functions); free(state);
    return result;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--allowed")) {
        for (unsigned version = 1; version <= 2; ++version)
            for (unsigned op = 0; op < 256; ++op) printf("%u\n", allowed((uint8_t)op, version));
        return 0;
    }
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
