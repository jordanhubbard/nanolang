/* I query the actual public loader-depth/full verifier APIs, never a clone. */
#include "../src/nanoisa/nvm_v2_sections.h"
#include "../src/nanoisa/verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void refusal(const NvmModule *module, const uint16_t *declared,
                    bool declarations, NvmVerifyResult result) {
    const char *affine = "I refuse reference lifetime and ownership instruction dataflow: ";
    if (!strncmp(result.error_msg, affine, strlen(affine)) ||
        !strcmp(result.error_msg, "I require owned-transfer execution semantics before execution")) {
        puts("A");
        return;
    }
    /* I normalize only an actual public refusal equal to the shared decoder's
     * own error. Other policy/metadata/body messages remain exact. */
    for (uint32_t i = 0; i < module->function_count; ++i) {
        if (declarations && !declared[i]) continue;
        VmDecodedFunction decoded = {0};
        char error[VM_DECODE_ERROR_SIZE];
        bool valid = vm_decode_function(module, i, &decoded, error);
        vm_decoded_function_free(&decoded);
        if (!valid && !strcmp(error, result.error_msg)) {
            puts("D");
            return;
        }
    }
    printf("0:%s\n", result.error_msg);
}

int main(int argc, char **argv) {
    bool declarations_only = argc > 1 && !strcmp(argv[1], "--declared");
    for (int arg = declarations_only ? 2 : 1; arg < argc; ++arg) {
        FILE *input = fopen(argv[arg], "rb");
        if (!input || fseek(input, 0, SEEK_END)) return 2;
        long length = ftell(input);
        if (length < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(length ? (size_t)length : 1);
        if (!bytes || fread(bytes, 1, (size_t)length, input) != (size_t)length || fclose(input)) return 2;
        NvmV2Module wire = {0};
        NvmModule *module = NULL;
        NvmV2Result status = nvm_v2_module_deserialize(bytes, (size_t)length, &wire);
        if (status == NVM_V2_OK) status = nvm_v2_to_nvm_module(&wire, &module);
        if (status != NVM_V2_OK) printf("R:%u\n", (unsigned)status);
        else {
            uint32_t count = module->function_count;
            uint16_t *declared = calloc(count ? count : 1, sizeof *declared);
            uint16_t *depths = calloc(count ? count : 1, sizeof *depths);
            if (!declared || !depths || wire.functions.count != count) return 2;
            bool selected = false;
            for (uint32_t i = 0; i < count; ++i) {
                declared[i] = wire.functions.items[i].max_stack;
                selected |= declared[i] != 0;
            }
            NvmVerifyResult result = nvm_verify_declared_max_stacks(module, declared, count);
            bool declaration_failure = !result.ok;
            if (result.ok && !declarations_only) result = nvm_verify(module);
            if (!result.ok) {
                fprintf(stderr, "%s\t%s\t%s\n", argv[arg], declaration_failure ? "declared" : "full", result.error_msg);
                refusal(module, declared, declaration_failure, result);
            }
            else {
                for (uint32_t i = 0; i < count; ++i) {
                    if (declarations_only && (!selected || !declared[i])) continue;
                    depths[i] = 0xabcd;
                    NvmVerifyResult one = nvm_verify_function_max_stack(module, i, &depths[i]);
                    if (!one.ok) return 3;
                }
                printf("1:%u:", declarations_only ? 0u : 1u);
                for (uint32_t i = 0; i < count; ++i) printf("%s%u", i ? "," : "", depths[i]);
                putchar('\n');
            }
            free(depths);
            free(declared);
        }
        nvm_module_free(module);
        nvm_v2_module_free(&wire);
        free(bytes);
    }
    return 0;
}
