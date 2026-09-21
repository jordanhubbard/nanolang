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
    printf("I passed %u capture payload checks; reader and writer allocation failures recover without live storage.\n", checks);
    return 0;
}
