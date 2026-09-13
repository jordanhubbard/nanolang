/* I expose the production decoder to the independent Sail comparison. */
#include "isa.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int hex_digit(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    return -1;
}

int main(void) {
    char line[258];
    while (fgets(line, sizeof(line), stdin)) {
        size_t length = strcspn(line, "\n");
        if (line[length] != '\n' || length % 2 || length > 256) return 2;
        uint8_t bytes[128];
        for (size_t i = 0; i < length / 2; i++) {
            int hi = hex_digit(line[2 * i]), lo = hex_digit(line[2 * i + 1]);
            if (hi < 0 || lo < 0) return 2;
            bytes[i] = (uint8_t)(hi * 16 + lo);
        }
        DecodedInstruction result;
        uint32_t consumed = isa_decode(bytes, length / 2, &result);
        if (!consumed) { puts("0"); continue; }
        uint64_t payload = 0;
        if (result.opcode == OP_PUSH_I64)
            memcpy(&payload, &result.operands[0].i64, sizeof(payload));
        printf("%" PRIu32 " %u %016" PRIx64 "\n", consumed, result.opcode, payload);
    }
    return ferror(stdin) ? 2 : 0;
}
