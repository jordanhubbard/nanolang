/* I compare actual C-decoded/re-encoded fields with NanoLang's row views. */
#include "../src/nanoisa/nvm_v2_sections.h"
#include <stdio.h>
#include <stdlib.h>

#define CODEC(Type, prefix) do { \
    Type table = {0}; \
    status = prefix##_decode(bytes, (size_t)length, &table); \
    if (status == NVM_V2_OK) { \
        count = table.count; \
        encoded_size = prefix##_encoded_size(&table); \
        encoded = malloc(encoded_size ? encoded_size : 1); \
        if (!encoded) return 2; \
        status = prefix##_encode(&table, encoded, encoded_size); \
    } \
    prefix##_free(&table); \
} while (0)

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    int kind = atoi(argv[1]);
    for (int i = 2; i < argc; ++i) {
        FILE *input = fopen(argv[i], "rb");
        if (!input || fseek(input, 0, SEEK_END)) return 2;
        long length = ftell(input);
        if (length < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(length ? (size_t)length : 1);
        if (!bytes) return 2;
        if (fread(bytes, 1, (size_t)length, input) != (size_t)length) return 2;
        if (fclose(input)) return 2;
        NvmV2Result status = NVM_V2_ERR_SECTION_TYPE;
        uint32_t count = 0;
        size_t encoded_size = 0;
        unsigned char *encoded = NULL;
        switch (kind) {
        case 1: CODEC(NvmV2Metadata, nvm_v2_metadata); break;
        case 2: CODEC(NvmV2Constants, nvm_v2_constants); break;
        case 3: CODEC(NvmV2Signatures, nvm_v2_signatures); break;
        case 4: CODEC(NvmV2Layouts, nvm_v2_layouts); break;
        case 5: CODEC(NvmV2Functions, nvm_v2_functions); break;
        case 7: CODEC(NvmV2Globals, nvm_v2_globals); break;
        case 8: CODEC(NvmV2Imports, nvm_v2_imports); break;
        case 9: CODEC(NvmV2Links, nvm_v2_links); break;
        case 10: CODEC(NvmV2Debug, nvm_v2_debug); break;
        case 11: CODEC(NvmV2Callbacks, nvm_v2_callbacks); break;
        default: break;
        }
        printf("%u:%u:", (unsigned)status, (unsigned)count);
        if (status == NVM_V2_OK)
            for (size_t j = 0; j < encoded_size; ++j) printf("%02x", (unsigned)encoded[j]);
        putchar('\n');
        free(encoded);
        free(bytes);
    }
    return 0;
}
