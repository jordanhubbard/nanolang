/* I expose the existing whole-module decoder and primary scan as test oracles. */
#include "../src/nanoisa/nvm_v2_sections.h"
#include "../src/nanoisa/isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void hexadecimal(const void *bytes, size_t size) {
    const unsigned char *data = bytes;
    for (size_t i = 0; i < size; ++i) printf("%02x", data[i]);
}

/* I call the actual reverse converter and project its owned result directly. */
static void conversion_report(NvmV2Result status, const NvmV2Module *view) {
    NvmModule *model = NULL;
    if (status == NVM_V2_OK) status = nvm_v2_to_nvm_module(view, &model);
    printf("%u", (unsigned)status);
    if (status == NVM_V2_OK) {
        putchar(':'); hexadecimal(model->code, model->code_size);
        putchar(':'); hexadecimal(model->layout_data, model->layout_size);
        printf(":%u:%u:%u:%u", model->struct_count, model->enum_count,
               model->union_count, model->string_count);
        for (uint32_t i = 0; i < model->string_count; ++i) {
            putchar(':'); hexadecimal(model->strings[i], model->string_lengths[i]);
        }
        printf(":%u", model->function_count);
        for (uint32_t i = 0; i < model->function_count; ++i) {
            const NvmFunctionEntry *f = &model->functions[i];
            printf(":%u,%u,%u,%u,%u,%u,%u", f->code_offset, f->code_length,
                   f->arity, f->result_count, f->local_count, f->upvalue_count, f->result_tag);
            for (uint16_t j = 0; j < f->arity; ++j)
                printf(",%u", model->function_param_types[i][j]);
        }
    }
    putchar('\n');
    nvm_module_free(model);
}

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
    bool conversion = argc > 1 && !strcmp(argv[1], "--conversion");
    bool services = argc > 1 && !strcmp(argv[1], "--services");
    for (int i = scan || passive || claims || services || conversion ? 2 : 1; i < argc; ++i) {
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
            if (conversion) {
                conversion_report(result, &module);
            } else if (services) {
                uint32_t service = result == NVM_V2_OK ? module.service_size : 0;
                uint32_t own = result == NVM_V2_OK ? module.ownership_size : 0;
                uint32_t pass = result == NVM_V2_OK ? module.passive_size : 0;
                printf("%u:%u:", (unsigned)result, service);
                for(uint32_t j=0;j<service;j++)printf("%02x",module.service_data[j]);
                printf(":%u:",own);
                for(uint32_t j=0;j<own;j++)printf("%02x",module.ownership_data[j]);
                printf(":%u:",pass);
                for(uint32_t j=0;j<pass;j++)printf("%02x",module.passive_data[j]);
                putchar('\n');
            } else if (claims) {
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
