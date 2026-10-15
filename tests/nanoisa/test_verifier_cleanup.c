/* I count verifier-owned allocations across rejection and allocation failure. */
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../src/nanoisa/isa.h"
#include "../../src/nanoisa/ownership_contracts.h"

static size_t structural_checks;
static NvmV2Result counted_contracts(const NvmModule *mod, bool *needs) {
    structural_checks++;
    return nvm_ownership_contracts_validate(mod, needs);
}

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

#define nvm_ownership_contracts_validate counted_contracts
#define malloc checked_malloc
#define free checked_free
#define isa_get_info checked_info
#include "../../src/nanoisa/verifier.c"
#undef nvm_ownership_contracts_validate
#undef isa_get_info
#undef malloc
#undef free

static void check_module_validation_scaling(void) {
    size_t baseline[3] = {0};
    for (unsigned count = 16; count <= 128; count *= 8) {
        NvmModule *mod = nvm_module_new();
        assert(mod);
        uint32_t name = nvm_add_string(mod, "entry", 5);
        uint8_t code[] = {OP_PUSH_I64, 1, 0, 0, 0, 0, 0, 0, 0, OP_RET};
        uint16_t depths[128];
        for (unsigned i = 0; i < count; ++i) {
            uint32_t offset = nvm_append_code(mod, code, sizeof code);
            NvmFunctionEntry fn = {.name_idx = name, .code_offset = offset,
                .code_length = sizeof code, .result_count = 1, .result_tag = TAG_INT};
            assert(nvm_add_function(mod, &fn) == i);
            depths[i] = 1;
        }
        mod->header.flags = NVM_FLAG_HAS_MAIN;
        for (unsigned route = 0; route < 3; ++route) {
            structural_checks = 0;
            NvmVerifyResult r = route == 0 ? nvm_verify(mod) : route == 1 ?
                nvm_verify_linked(mod, NULL, 0) : nvm_verify_declared_max_stacks(mod, depths, count);
            assert(r.ok && outstanding == 0);
            assert(structural_checks > 0 && structural_checks < 8);
            if (count == 16) baseline[route] = structural_checks;
            else assert(structural_checks == baseline[route]);
        }
        nvm_module_free(mod);
    }
}

int main(void) {
    check_module_validation_scaling();
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
