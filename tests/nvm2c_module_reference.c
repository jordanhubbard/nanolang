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
    bool passive = argc > 1 && !strcmp(argv[1], "--passive");
    bool claims = argc > 1 && !strcmp(argv[1], "--claims");
    for (int i = scan || passive || claims ? 2 : 1; i < argc; ++i) {
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
            NvmV2Result result = nvm_v2_module_deserialize(bytes, (size_t)length, &module);
            if (claims) {
                uint32_t own = result == NVM_V2_OK ? module.ownership_size : 0;
                uint32_t pass = result == NVM_V2_OK ? module.passive_size : 0;
                printf("%u:%u:", (unsigned)result, own);
                for(uint32_t j=0;j<own;j++)printf("%02x",module.ownership_data[j]);
                printf(":%u:",pass);
                for(uint32_t j=0;j<pass;j++)printf("%02x",module.passive_data[j]);
                putchar('\n');
            } else if (passive) {
                uint32_t size = result == NVM_V2_OK ? module.passive_size : 0;
                printf("%u:%u:", (unsigned)result, size);
                for (uint32_t j = 0; j < size; ++j) printf("%02x", module.passive_data[j]);
                putchar('\n');
            } else printf("%u\n", (unsigned)result);
            nvm_v2_module_free(&module);
        }
        free(bytes);
    }
    return 0;
}
