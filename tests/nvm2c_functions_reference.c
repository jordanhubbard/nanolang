/* I report the actual VM function decoder without verifier admission. */
#include "../src/nanovm/vm_decode.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t word(const unsigned char *p) {
    uint64_t result = 0;
    for (int i = 7; i >= 0; --i) result = (result << 8) | p[i];
    return result;
}

int main(int argc, char **argv) {
    for (int arg = 1; arg < argc; ++arg) {
        FILE *input = fopen(argv[arg], "rb");
        if (!input || fseek(input, 0, SEEK_END)) return 2;
        long size = ftell(input);
        if (size < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(size ? (size_t)size : 1);
        if (!bytes || fread(bytes, 1, (size_t)size, input) != (size_t)size || fclose(input)) return 2;
        VmDecodedFunction decoded = {0};
        bool valid = false;
        if (size >= 24 && word(bytes) <= UINT32_MAX && word(bytes + 8) <= UINT32_MAX &&
                word(bytes + 16) <= UINT32_MAX) {
            NvmFunctionEntry fn = {.code_offset = (uint32_t)word(bytes),
                                   .code_length = (uint32_t)word(bytes + 8)};
            NvmModule module = {.code = bytes + 24, .code_size = (uint64_t)size - 24,
                                .functions = &fn, .function_count = (uint32_t)word(bytes + 16)};
            char error[VM_DECODE_ERROR_SIZE];
            valid = vm_decode_function(&module, 0, &decoded, error);
        }
        if (!valid) puts("0");
        else {
            printf("1:%u:%u", decoded.code_size, decoded.instruction_count);
            for (uint32_t i = 0; i < decoded.instruction_count; i++) {
                const VmDecodedInstruction *item = &decoded.instructions[i];
                const InstructionInfo *info = isa_get_info(item->instruction.opcode);
                printf(":%u:%u:%u:%d:%d:", item->byte_offset, item->next_byte_offset,
                       item->resolved_target, info->pop_count, info->push_count);
                unsigned char encoded[64];
                uint32_t length = isa_encode(&item->instruction, encoded, sizeof(encoded));
                if (length != item->next_byte_offset - item->byte_offset) return 3;
                for (uint32_t j = 0; j < length; j++) printf("%02x", encoded[j]);
            }
            putchar(':');
            for (uint64_t i = 0; i <= decoded.code_size; i++)
                putchar(vm_decoded_function_has_boundary(&decoded, (uint32_t)i) ? '1' : '0');
            putchar(':');
            for (uint64_t i = 0; i <= decoded.code_size; i++)
                printf("%s%u", i ? "," : "", decoded.instruction_indices[i]);
            putchar('\n');
        }
        vm_decoded_function_free(&decoded);
        free(bytes);
    }
    return 0;
}
