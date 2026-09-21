/* I check actual payload decoding, exact refusal and allocation rollback. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../src/nanoisa/capture_bindings.h"

static unsigned checks, allocation_calls, live;
static unsigned fail_allocation;
#define CHECK(x) do { ++checks; if (!(x)) { \
    fprintf(stderr, "I failed line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void *checked_calloc(size_t count, size_t width) {
    ++allocation_calls;
    if (fail_allocation && allocation_calls == fail_allocation) return NULL;
    void *result = calloc(count, width);
    if (result) ++live;
    return result;
}
static void *checked_malloc(size_t size) {
    ++allocation_calls;
    if (fail_allocation && allocation_calls == fail_allocation) return NULL;
    void *result = malloc(size);
    if (result) ++live;
    return result;
}
static void checked_free(void *pointer) {
    if (pointer) { CHECK(live > 0); --live; }
    free(pointer);
}
#define calloc checked_calloc
#define malloc checked_malloc
#define free checked_free
#include "../../src/nanoisa/capture_bindings.c"
#undef calloc
#undef malloc
#undef free

typedef struct {
    uint8_t bytes[256];
    size_t used, functions[3], sites[3];
} Fixture;
static void byte(Fixture *f, uint8_t value) {
    CHECK(f->used < sizeof(f->bytes));
    f->bytes[f->used++] = value;
}
static void word(Fixture *f, uint16_t value) {
    byte(f, (uint8_t)value); byte(f, (uint8_t)(value >> 8));
}
static void wide(Fixture *f, uint32_t value) {
    word(f, (uint16_t)value); word(f, (uint16_t)(value >> 16));
}
static Fixture fixture(void) {
    Fixture f = {0};
    word(&f, 1); word(&f, 0); wide(&f, 3); wide(&f, 3);
    f.functions[0] = f.used;
    wide(&f, 0); word(&f, 2); word(&f, 0); byte(&f, 1); byte(&f, 0);
    f.functions[1] = f.used;
    wide(&f, 1); word(&f, 1); word(&f, 3);
    byte(&f, 0); byte(&f, 1); byte(&f, 0); byte(&f, 1);
    f.functions[2] = f.used;
    wide(&f, 2); word(&f, 0); word(&f, 1); byte(&f, 1);
    for (unsigned i = 0; i < 2; ++i) {
        f.sites[i] = f.used;
        wide(&f, 0); wide(&f, i * 5); wide(&f, 1); word(&f, 3); word(&f, 0);
        /* I retain two edges to the same mutable local around a copied value. */
        byte(&f, 0); byte(&f, 1); word(&f, 0);
        byte(&f, 0); byte(&f, 0); word(&f, 1);
        byte(&f, 0); byte(&f, 1); word(&f, 0);
    }
    f.sites[2] = f.used;
    wide(&f, 1); wide(&f, 0); wide(&f, 2); word(&f, 1); word(&f, 0);
    byte(&f, 1); byte(&f, 1); word(&f, 0);
    return f;
}

static NvmModule module(NvmFunctionEntry functions[3]) {
    functions[0] = (NvmFunctionEntry){.arity=1, .local_count=2, .code_length=10};
    functions[1] = (NvmFunctionEntry){.arity=1, .local_count=1, .upvalue_count=3,
        .code_offset=10, .code_length=5};
    functions[2] = (NvmFunctionEntry){.upvalue_count=1, .code_offset=15, .code_length=1};
    return (NvmModule){.functions=functions, .function_count=3, .code_size=16};
}

static void refuse(const uint8_t *bytes, size_t size, const NvmModule *m,
    size_t limit, NvmCaptureResult expected) {
    NvmCaptureBindings output, before;
    memset(&output, 0xa5, sizeof(output));
    memcpy(&before, &output, sizeof(output));
    CHECK(nvm_capture_bindings_decode(bytes, size, m, limit, &output) == expected);
    CHECK(memcmp(&output, &before, sizeof(output)) == 0);
    CHECK(live == 0);
}

static void refuse_encoding(const NvmCaptureBindings *bindings, const NvmModule *m,
    size_t limit, NvmCaptureResult expected) {
    uint8_t sentinel = 0, *output = &sentinel;
    size_t size = 999;
    unsigned before = live;
    CHECK(nvm_capture_bindings_encode(bindings, m, limit, &output, &size) == expected);
    CHECK(output == &sentinel && size == 999 && sentinel == 0);
    CHECK(live == before);
}

static void encode_tests(const Fixture *fixture, const NvmModule *m, size_t table_bytes) {
    NvmCaptureBindings decoded = {0};
    CHECK(nvm_capture_bindings_decode(fixture->bytes, fixture->used, m, table_bytes, &decoded) == NVM_CAPTURE_OK);
    size_t budget = table_bytes + fixture->used;
    uint8_t *output = NULL;
    size_t size = 0;
    CHECK(nvm_capture_bindings_encode(&decoded, m, budget, &output, &size) == NVM_CAPTURE_OK);
    CHECK(size == fixture->used && memcmp(output, fixture->bytes, size) == 0);
    CHECK(live == 3);
    checked_free(output);
    refuse_encoding(&decoded, m, budget - 1, NVM_CAPTURE_LIMIT);
    refuse_encoding(&decoded, m, 0, NVM_CAPTURE_LIMIT);
    refuse_encoding(NULL, m, budget, NVM_CAPTURE_INVALID);
    NvmCaptureBindings missing = decoded;
    missing.functions = NULL;
    refuse_encoding(&missing, m, budget, NVM_CAPTURE_INVALID);
    missing = decoded; missing.sites = NULL;
    refuse_encoding(&missing, m, budget, NVM_CAPTURE_INVALID);
    missing = decoded; missing.function_count--;
    refuse_encoding(&missing, m, budget, NVM_CAPTURE_INVALID);
    const uint8_t *saved = decoded.functions[0].local_modes;
    decoded.functions[0].local_modes = NULL;
    refuse_encoding(&decoded, m, budget, NVM_CAPTURE_INVALID);
    uint8_t wrong_modes[2] = {2, 0};
    decoded.functions[0].local_modes = wrong_modes;
    refuse_encoding(&decoded, m, budget, NVM_CAPTURE_INVALID);
    decoded.functions[0].local_modes = saved;
    NvmCaptureSite site = decoded.sites[0];
    decoded.sites[0].sources = NULL;
    refuse_encoding(&decoded, m, budget, NVM_CAPTURE_INVALID);
    decoded.sites[0] = site;
    decoded.sites[0].target = UINT32_MAX;
    refuse_encoding(&decoded, m, budget, NVM_CAPTURE_INVALID);
    decoded.sites[0] = site;
    decoded.sites[1].instruction_offset = 0;
    refuse_encoding(&decoded, m, budget, NVM_CAPTURE_INVALID);
    decoded.sites[1].instruction_offset = 5;
    for (unsigned failure = 1; failure <= 3; ++failure) {
        allocation_calls = 0; fail_allocation = failure;
        refuse_encoding(&decoded, m, budget, NVM_CAPTURE_MEMORY);
        fail_allocation = 0;
        CHECK(nvm_capture_bindings_encode(&decoded, m, budget, &output, &size) == NVM_CAPTURE_OK);
        CHECK(size == fixture->used && memcmp(output, fixture->bytes, size) == 0);
        checked_free(output);
        CHECK(live == 2);
    }
    nvm_capture_bindings_free(&decoded);
    CHECK(live == 0);
    NvmModule none = {0};
    uint8_t empty[12] = {1};
    CHECK(nvm_capture_bindings_encode(&decoded, &none, 12, &output, &size) == NVM_CAPTURE_OK);
    CHECK(size == sizeof(empty) && memcmp(output, empty, size) == 0);
    checked_free(output);
    refuse_encoding(&decoded, &none, 11, NVM_CAPTURE_LIMIT);
}

static void code_tests(void) {
    Fixture f = fixture();
    NvmFunctionEntry functions[3];
    NvmModule m = module(functions);
    uint8_t code[] = {
        OP_CLOSURE_BIND,0,0,0,0, OP_CLOSURE_BIND,1,0,0,0,
        OP_PUSH_VOID, OP_BIND_INIT_LOCAL,1,0, OP_LOAD_LOCAL,1,0,
        OP_STORE_LOCAL,0,0, OP_BIND_CLEAR_LOCAL,1,0,
        OP_CLOSURE_BIND,2,0,0,0, OP_LOAD_UPVALUE,0,0,0,0,
        OP_STORE_UPVALUE,0,0,0,0, OP_RET
    };
    functions[0].code_length = 23;
    functions[1].code_offset = 23; functions[1].code_length = 15;
    functions[2].code_offset = 38; functions[2].code_length = 1;
    m.code = code; m.code_size = sizeof(code);
    CHECK(m.code_size == 39);
    NvmCaptureBindings decoded = {0};
    CHECK(nvm_capture_bindings_decode(f.bytes, f.used, &m, 4096, &decoded) == NVM_CAPTURE_OK);
    size_t work = sizeof(code) + 3;
    unsigned before = allocation_calls;
    CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work) == NVM_CAPTURE_OK);
    CHECK(allocation_calls == before);
    CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work - 1) == NVM_CAPTURE_LIMIT);
    CHECK(nvm_capture_bindings_verify_code(&decoded, &m, 0) == NVM_CAPTURE_LIMIT);
    const struct {size_t offset; uint8_t value;} invalid[] = {
        {1, 1}, {6, 0}, {24, 1}, /* duplicate/out-of-order/wrong-owner sites */
        {0, OP_PUSH_I64}, /* site hidden inside another instruction */
        {0, OP_NOP}, /* descriptor no longer names a closure */
        {0, OP_CLOSURE_NEW}, /* legacy environment refused */
        {12, 2}, {15, 2}, {18, 1}, {21, 2}, /* local range and immutable store */
        {29, 1}, {31, 3}, {34, 1}, {36, 1}, /* upvalue depth/range/immutable */
        {38, 0xff}
    };
    for (size_t i = 0; i < sizeof(invalid)/sizeof(invalid[0]); ++i) {
        uint8_t saved = code[invalid[i].offset];
        code[invalid[i].offset] = invalid[i].value;
        CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work) == NVM_CAPTURE_INVALID);
        code[invalid[i].offset] = saved;
    }
    uint32_t saved_count = decoded.site_count;
    decoded.site_count = 2;
    CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work) == NVM_CAPTURE_INVALID);
    decoded.site_count = saved_count;
    /* I retain a valid code stream while leaving an unclaimed descriptor. */
    memset(code + 23, OP_NOP, 5);
    CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work) == NVM_CAPTURE_INVALID);
    code[23] = OP_CLOSURE_BIND; code[24] = 2;
    functions[0].code_offset = UINT32_MAX;
    CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work) == NVM_CAPTURE_INVALID);
    functions[0].code_offset = 0;
    for (unsigned length = 1; length < 5; ++length) {
        functions[0].code_length = length;
        CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work) == NVM_CAPTURE_INVALID);
    }
    functions[0].code_length = 23;
    CHECK(nvm_capture_bindings_verify_code(&decoded, &m, work) == NVM_CAPTURE_OK);
    nvm_capture_bindings_free(&decoded);
    CHECK(live == 0);
    const uint8_t opcodes[] = {OP_BIND_INIT_LOCAL, OP_BIND_CLEAR_LOCAL, OP_CLOSURE_BIND};
    for (size_t i = 0; i < sizeof(opcodes); ++i) {
        DecodedInstruction in = {0}, out = {0};
        in.opcode = opcodes[i];
        if (i == 2) in.operands[0].u32 = UINT32_C(0x12345678);
        else in.operands[0].u16 = UINT16_C(0x5678);
        uint8_t bytes[8] = {0};
        uint32_t length = isa_encode(&in, bytes, sizeof(bytes));
        CHECK(length == (i == 2 ? 5u : 3u));
        CHECK(bytes[0] == opcodes[i] && bytes[1] == 0x78 && bytes[2] == 0x56);
        CHECK(isa_decode(bytes, length, &out) == length && out.opcode == in.opcode);
        CHECK(i == 2 ? out.operands[0].u32 == in.operands[0].u32 : out.operands[0].u16 == in.operands[0].u16);
        for (size_t short_length = 0; short_length < length; ++short_length)
            CHECK(isa_decode(bytes, short_length, &out) == 0);
    }
}

int main(void) {
    Fixture f = fixture();
    NvmFunctionEntry functions[3];
    NvmModule m = module(functions);
    const size_t exact = 3 * sizeof(NvmCaptureFunction) + 3 * sizeof(NvmCaptureSite);
    NvmCaptureBindings decoded = {0};
    CHECK(nvm_capture_bindings_decode(f.bytes, f.used, &m, exact, &decoded) == NVM_CAPTURE_OK);
    CHECK(decoded.function_count == 3 && decoded.site_count == 3);
    CHECK(decoded.allocation_bytes == exact && live == 2);
    CHECK(decoded.functions[0].local_modes[0] == NVM_CAPTURE_SHARED);
    CHECK(decoded.functions[0].local_modes[1] == NVM_CAPTURE_VALUE);
    CHECK(decoded.sites[1].owner == 0 && decoded.sites[1].instruction_offset == 5);
    uint8_t kind = 77, mode = 88;
    uint16_t slot = 99;
    CHECK(nvm_capture_source(&decoded.sites[0], 0, &kind, &mode, &slot));
    CHECK(kind == NVM_CAPTURE_LOCAL && mode == NVM_CAPTURE_SHARED && slot == 0);
    CHECK(nvm_capture_source(&decoded.sites[0], 2, &kind, &mode, &slot));
    CHECK(kind == NVM_CAPTURE_LOCAL && mode == NVM_CAPTURE_SHARED && slot == 0);
    CHECK(nvm_capture_source(&decoded.sites[2], 0, &kind, &mode, &slot));
    CHECK(kind == NVM_CAPTURE_UPVALUE && mode == NVM_CAPTURE_SHARED && slot == 0);
    CHECK(!nvm_capture_source(&decoded.sites[2], 1, &kind, &mode, &slot));
    CHECK(kind == NVM_CAPTURE_UPVALUE && mode == NVM_CAPTURE_SHARED && slot == 0);
    nvm_capture_bindings_free(&decoded);
    CHECK(!live && !decoded.functions && !decoded.sites && !decoded.allocation_bytes);
    nvm_capture_bindings_free(&decoded);

    for (size_t size = 0; size < f.used; ++size)
        refuse(f.bytes, size, &m, exact, NVM_CAPTURE_INVALID);
    refuse(f.bytes, f.used, &m, exact - 1, NVM_CAPTURE_LIMIT);
    refuse(f.bytes, f.used, &m, 0, NVM_CAPTURE_LIMIT);
    f.bytes[f.used] = 0;
    refuse(f.bytes, f.used + 1, &m, exact, NVM_CAPTURE_INVALID);

    const struct { size_t offset; uint8_t value; } invalid[] = {
        {0, 2}, {2, 1}, {4, 2}, {8, 255},
        {f.functions[0], 1}, {f.functions[0] + 4, 1},
        {f.functions[0] + 6, 1}, {f.functions[0] + 8, 2},
        {f.functions[1] + 9, 2},
        {f.sites[0], 3}, {f.sites[0] + 4, 6},
        {f.sites[0] + 8, 3}, {f.sites[0] + 12, 2},
        {f.sites[0] + 14, 1},
        {f.sites[0] + 16, 2}, {f.sites[0] + 17, 2},
        {f.sites[0] + 17, 0}, {f.sites[0] + 18, 2},
        {f.sites[0] + 18, 1}, {f.sites[1] + 4, 0},
        {f.sites[2], 0}, {f.sites[2] + 16, 0}, {f.sites[2] + 18, 3}
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        uint8_t saved = f.bytes[invalid[i].offset];
        f.bytes[invalid[i].offset] = invalid[i].value;
        refuse(f.bytes, f.used, &m, exact, NVM_CAPTURE_INVALID);
        f.bytes[invalid[i].offset] = saved;
    }
    functions[0].arity = 3;
    refuse(f.bytes, f.used, &m, exact, NVM_CAPTURE_INVALID);
    m = module(functions);
    functions[0].code_offset = UINT32_MAX;
    refuse(f.bytes, f.used, &m, exact, NVM_CAPTURE_INVALID);
    m = module(functions);
    functions[0].code_length = UINT32_MAX;
    refuse(f.bytes, f.used, &m, exact, NVM_CAPTURE_INVALID);
    m = module(functions);
    for (unsigned failure = 1; failure <= 2; ++failure) {
        allocation_calls = 0; fail_allocation = failure;
        refuse(f.bytes, f.used, &m, exact, NVM_CAPTURE_MEMORY);
        fail_allocation = 0;
        CHECK(nvm_capture_bindings_decode(f.bytes, f.used, &m, exact, &decoded) == NVM_CAPTURE_OK);
        nvm_capture_bindings_free(&decoded);
        CHECK(live == 0);
    }
    uint8_t empty[12] = {1};
    NvmModule none = {0};
    CHECK(nvm_capture_bindings_decode(empty, sizeof(empty), &none, 0, &decoded) == NVM_CAPTURE_OK);
    CHECK(!decoded.functions && !decoded.sites && !decoded.allocation_bytes && !live);
    nvm_capture_bindings_free(&decoded);
    encode_tests(&f, &m, exact);
    code_tests();
    printf("I passed %u capture payload checks; reader and writer allocation failures recover without live storage.\n", checks);
    return 0;
}
