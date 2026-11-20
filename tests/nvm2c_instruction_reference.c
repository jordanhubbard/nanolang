/* I expose actual primary C decoding, stack metadata and re-encoding. */
#include "../src/nanoisa/isa.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t word(const unsigned char *p) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
    return value;
}
int main(int argc, char **argv) {
    for (int arg = 1; arg < argc; ++arg) {
        FILE *input = fopen(argv[arg], "rb");
        if (!input || fseek(input, 0, SEEK_END)) return 2;
        long size = ftell(input);
        if (size < 0 || fseek(input, 0, SEEK_SET)) return 2;
        unsigned char *bytes = malloc(size ? (size_t)size : 1);
        if (!bytes || fread(bytes, 1, (size_t)size, input) != (size_t)size) return 2;
        if (fclose(input)) return 2;
        DecodedInstruction instruction = {0};
        uint32_t length = 0;
        if (size >= 16) {
            uint64_t offset = word(bytes), available = word(bytes + 8);
            uint64_t body = (uint64_t)size - 16;
            if (offset <= body && available <= body - offset)
                length = isa_decode(bytes + 16 + (size_t)offset, (size_t)available, &instruction);
        }
        if (!length) puts("0:-1:-1:-1:0:");
        else {
            const InstructionInfo *info = isa_get_info(instruction.opcode);
            printf("%u:%u:%d:%d:%u", length, instruction.opcode, info->pop_count,
                   info->push_count, instruction.operand_count);
            for (unsigned i = 0; i < instruction.operand_count; ++i) {
                uint64_t bits = 0;
                int64_t integer = 0;
                switch (instruction.operand_types[i]) {
                    case OPERAND_U8: bits = instruction.operands[i].u8; integer = (int64_t)bits; break;
                    case OPERAND_U16: bits = instruction.operands[i].u16; integer = (int64_t)bits; break;
                    case OPERAND_U32: bits = instruction.operands[i].u32; integer = (int64_t)bits; break;
                    case OPERAND_I32: integer = instruction.operands[i].i32; bits = (uint32_t)integer; break;
                    case OPERAND_I64: integer = instruction.operands[i].i64; bits = (uint64_t)integer; break;
                    case OPERAND_F64: memcpy(&bits, &instruction.operands[i].f64, sizeof bits); break;
                    case OPERAND_NONE: break;
                }
                printf(":%d:%" PRIu32 ":%" PRIu32 ":%" PRId64, (int)instruction.operand_types[i],
                       (uint32_t)bits, (uint32_t)(bits >> 32), integer);
            }
            unsigned char encoded[64];
            uint32_t encoded_size = isa_encode(&instruction, encoded, sizeof encoded);
            if (encoded_size != length) return 3;
            putchar(':');
            for (uint32_t i = 0; i < encoded_size; ++i) printf("%02x", encoded[i]);
            putchar('\n');
        }
        free(bytes);
    }
    return 0;
}
