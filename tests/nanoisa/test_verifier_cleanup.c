/* I count verifier-owned allocations across rejection and allocation failure. */
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../src/nanoisa/isa.h"

static void *owned[16];
static int outstanding, allocation, fail_at;
static int unknown_effect;
static const InstructionInfo *checked_info(uint8_t opcode) {
    const InstructionInfo *info = isa_get_info(opcode);
    if (!unknown_effect) return info;
    static InstructionInfo injected;
    injected = *info;
    if (unknown_effect == 1) injected.pop_count = -1;
    else injected.push_count = -1;
    return &injected;
}
static void *checked_malloc(size_t size) {
    if (++allocation == fail_at) return NULL;
    void *pointer = malloc(size);
    if (pointer) {
        assert(outstanding < 16);
        owned[outstanding++] = pointer;
    }
    return pointer;
}
static void checked_free(void *pointer) {
    if (!pointer) return;
    int index = 0;
    while (index < outstanding && owned[index] != pointer) index++;
    assert(index < outstanding);
    owned[index] = owned[--outstanding];
    free(pointer);
}

#define malloc checked_malloc
#define free checked_free
#define isa_get_info checked_info
#include "../../src/nanoisa/verifier.c"
#undef isa_get_info
#undef malloc
#undef free

int main(void) {
    NvmFunctionEntry function = {.result_count = 1};
    NvmModule module = {.functions = &function, .function_count = 1};
    VmDecodedInstruction instruction = {0};
    instruction.instruction.opcode = OP_NOP;
    instruction.next_byte_offset = 1;
    VmDecodedFunction decoded = {.instructions = &instruction,
        .instruction_count = 1, .code_size = 1};
    for (int failure = 0; failure <= 3; failure++) {
        fail_at = failure;
        allocation = 0;
        uint16_t depth = 77;
        NvmVerifyResult result = verify_stack_heights(&module, &decoded, 0, &depth);
        assert(!result.ok);
        assert(strstr(result.error_msg, failure ? "allocate" : "reaches its end"));
        assert(depth == 77);
        assert(outstanding == 0);
    }
    function.result_count = 0;
    fail_at = allocation = 0;
    uint16_t depth = 77;
    assert(verify_stack_heights(&module, &decoded, 0, &depth).ok);
    assert(depth == 0 && outstanding == 0);
    for (unknown_effect = 1; unknown_effect <= 2; unknown_effect++) {
        allocation = 0;
        depth = 77;
        NvmVerifyResult result = verify_stack_heights(&module, &decoded, 0, &depth);
        assert(!result.ok && strstr(result.error_msg, "no known stack effect"));
        assert(depth == 77 && outstanding == 0);
    }
    unknown_effect = 0;
    allocation = 0;
    VmDecodedInstruction ownership_path[4] = {0};
    const uint8_t opcodes[] = {OP_PUSH_I64, OP_PUSH_I64, OP_GC_RETAIN, OP_GC_RELEASE};
    const uint32_t offsets[] = {0, 9, 18, 19, 20};
    for (int i = 0; i < 4; i++) {
        ownership_path[i].instruction.opcode = opcodes[i];
        ownership_path[i].byte_offset = offsets[i];
        ownership_path[i].next_byte_offset = offsets[i + 1];
    }
    decoded.instructions = ownership_path;
    decoded.instruction_count = 4;
    decoded.code_size = 20;
    depth = 77;
    NvmVerifyResult result = verify_stack_heights(&module, &decoded, 0, &depth);
    assert(!result.ok && strstr(result.error_msg, "reaches its end"));
    assert(depth == 77 && outstanding == 0);
    puts("I released verifier allocations across rejection, allocation failure and success.");
    return 0;
}
