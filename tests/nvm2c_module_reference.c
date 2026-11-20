/* I expose the existing whole-module decoder and primary scan as test oracles. */
#include "../src/nanoisa/nvm_v2_sections.h"
#include "../src/nanoisa/isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--widths")) {
        unsigned char code[64] = {0};
        for (int i = 0; i < 256; ++i) {
            code[0] = (unsigned char)i;
            DecodedInstruction instruction;
            printf("%u\n", isa_decode(code, sizeof code, &instruction));
        }
        return 0;
    }
    bool scan = argc > 1 && !strcmp(argv[1], "--scan");
    for (int i = scan ? 2 : 1; i < argc; ++i) {
        FILE *input = fopen(argv[i], "rb");
        if (!input || fseek(input, 0, SEEK_END)) return 2;
        long length = ftell(input);
        if (length < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(length ? (size_t)length : 1);
        if (!bytes) return 2;
        if (fread(bytes, 1, (size_t)length, input) != (size_t)length) return 2;
        if (fclose(input)) return 2;
        if (scan) printf("%d\n", isa_code_has_file_instructions(bytes, (size_t)length));
        else {
            NvmV2Module module = {0};
            printf("%u\n", (unsigned)nvm_v2_module_deserialize(bytes, (size_t)length, &module));
            nvm_v2_module_free(&module);
        }
        free(bytes);
    }
    return 0;
}
