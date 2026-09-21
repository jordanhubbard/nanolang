/* I qualify transport and refusal, never execution of capture-bearing code. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "assembler.h"
#include "capture_bindings.h"
#include "disassembler.h"
#include "isa.h"
#include "nvm2c.h"
#include "nvm_v2_sections.h"
#include "verifier.h"

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
    fprintf(stderr, "I failed line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static NvmModule *fixture(bool sites, bool captures) {
    static const uint8_t ordinary[] = {OP_RET, OP_RET, OP_RET, OP_RET};
    static const uint8_t closures[] = {
        OP_CLOSURE_BIND,0,0,0,0, OP_CLOSURE_BIND,1,0,0,0,
        OP_CLOSURE_BIND,2,0,0,0, OP_RET,
        OP_CLOSURE_BIND,3,0,0,0, OP_RET, OP_RET, OP_RET
    };
    static const uint8_t local0[] = {NVM_CAPTURE_SHARED, NVM_CAPTURE_VALUE};
    static const uint8_t local1[] = {NVM_CAPTURE_VALUE};
    static const uint8_t upvalues1[] = {
        NVM_CAPTURE_SHARED, NVM_CAPTURE_VALUE, NVM_CAPTURE_SHARED
    };
    static const uint8_t upvalues2[] = {NVM_CAPTURE_SHARED};
    static const uint8_t repeated[] = {
        NVM_CAPTURE_LOCAL,NVM_CAPTURE_SHARED,0,0,
        NVM_CAPTURE_LOCAL,NVM_CAPTURE_VALUE,1,0,
        NVM_CAPTURE_LOCAL,NVM_CAPTURE_SHARED,0,0
    };
    static const uint8_t forwarded[] = {NVM_CAPTURE_UPVALUE,NVM_CAPTURE_SHARED,0,0};
    const char *names[] = {"main", "nested", "forwarded", "empty"};
    const uint32_t offsets[] = {0,16,22,23}, lengths[] = {16,6,1,1};
    NvmModule *m = nvm_module_new();
    CHECK(m != NULL);
    nvm_append_code(m, sites ? closures : ordinary,
                    sites ? sizeof closures : sizeof ordinary);
    CHECK(m->code_size == (sites ? sizeof closures : sizeof ordinary));
    NvmCaptureFunction functions[4] = {
        {2,0,local0,NULL}, {1,sites ? 3 : 0,local1,sites ? upvalues1 : NULL},
        {0,sites ? 1 : 0,NULL,sites ? upvalues2 : NULL}, {0,0,NULL,NULL}
    };
    for (uint32_t i = 0; i < 4; ++i) {
        NvmFunctionEntry fn = {0};
        fn.name_idx = nvm_add_string(m, names[i], (uint32_t)strlen(names[i]));
        CHECK(fn.name_idx != UINT32_MAX);
        fn.local_count = functions[i].local_count;
        fn.upvalue_count = functions[i].upvalue_count;
        fn.code_offset = sites ? offsets[i] : i;
        fn.code_length = sites ? lengths[i] : 1;
        fn.result_tag = TAG_VOID;
        CHECK(nvm_add_function(m, &fn) == i);
    }
    m->header.flags |= NVM_FLAG_HAS_MAIN;
    m->header.entry_point = 0;
    if (captures) {
        NvmCaptureSite rows[] = {
            {0,0,1,3,repeated}, {0,5,1,3,repeated},
            {0,10,3,0,NULL}, {1,0,2,1,forwarded}
        };
        NvmCaptureBindings binding = {functions,sites ? rows : NULL,4,sites ? 4 : 0,0};
        size_t size = 0;
        CHECK(nvm_capture_bindings_encode(&binding, m, NVM_CAPTURE_TRANSPORT_BYTES,
                                          &m->capture_data, &size) == NVM_CAPTURE_OK);
        CHECK(size > 0 && size <= UINT32_MAX);
        m->capture_size = (uint32_t)size;
        CHECK(nvm_capture_bindings_validate_module(m) == NVM_CAPTURE_OK);
    }
    return m;
}

static uint8_t *wire(const NvmModule *m, size_t *size) {
    NvmV2Module view = {0};
    CHECK(nvm_v2_from_nvm_module(m, &view) == NVM_V2_OK);
    CHECK(view.capture_data == m->capture_data && view.capture_size == m->capture_size);
    CHECK(nvm_v2_module_serialize(&view, NULL, 0, size) == NVM_V2_OK);
    uint8_t *bytes = malloc(*size);
    CHECK(bytes != NULL);
    CHECK(nvm_v2_module_serialize(&view, bytes, *size, size) == NVM_V2_OK);
    nvm_v2_module_free(&view);
    CHECK(nvm_capture_bindings_validate_module(m) == NVM_CAPTURE_OK);
    return bytes;
}

static void refusal(const NvmModule *m) {
    CHECK(!nvm_verify(m).ok);
    CHECK(!nvm_verify_function(m, 0).ok);
    CHECK(!nvm_verify_affine_function(m, 0).ok);
    CHECK(!nvm_verify_owned_module(m).ok);
    uint32_t legacy_size = 999;
    CHECK(nvm_serialize(m, &legacy_size) == NULL && legacy_size == 0);
    char error[256] = {0};
    CHECK(nvm2c_emit(m, error, sizeof error) == NULL);
    CHECK(strstr(error, "capture") != NULL);
    NvmModule *plain = fixture(false, false);
    CHECK(nvm_verify(plain).ok);
    const NvmModule *links[] = {m};
    CHECK(!nvm_verify_linked(plain, links, 1).ok);
    nvm_module_free(plain);
}

static void roundtrip(bool sites) {
    NvmModule *source = fixture(sites, true);
    refusal(source);
    size_t size = 0;
    uint8_t *bytes = wire(source, &size);
    uint8_t *saved_wire = malloc(size);
    uint8_t *saved_payload = malloc(source->capture_size);
    CHECK(saved_wire && saved_payload);
    memcpy(saved_wire, bytes, size);
    const uint32_t payload_size = source->capture_size;
    memcpy(saved_payload, source->capture_data, payload_size);
    NvmV2Header header;
    CHECK(nvm_v2_read_header(bytes, size, &header) == NVM_V2_OK);
    CHECK((header.feature_bits & NVM_V2_FEATURE_CAPTURE_BINDINGS) != 0);
    NvmV2Module decoded = {0};
    CHECK(nvm_v2_module_deserialize(bytes, size, &decoded) == NVM_V2_OK);
    CHECK(decoded.capture_size == payload_size);
    CHECK(memcmp(decoded.capture_data, saved_payload, payload_size) == 0);
    NvmModule *copy = NULL;
    CHECK(nvm_v2_to_nvm_module(&decoded, &copy) == NVM_V2_OK && copy);
    CHECK(copy->capture_data != decoded.capture_data);
    CHECK(copy->capture_data != source->capture_data);
    nvm_v2_module_free(&decoded);
    free(bytes);
    nvm_module_free(source);
    /* I exercise the independent owner after both borrowed sources die. */
    CHECK(copy->capture_size == payload_size);
    CHECK(memcmp(copy->capture_data, saved_payload, payload_size) == 0);
    CHECK(nvm_capture_bindings_validate_module(copy) == NVM_CAPTURE_OK);
    refusal(copy);
    size_t copied_size = 0;
    uint8_t *copied_wire = wire(copy, &copied_size);
    CHECK(copied_size == size && memcmp(copied_wire, saved_wire, size) == 0);
    free(copied_wire);
    char *text = disasm_module_styled(copy, DISASM_STYLE_CANONICAL);
    CHECK(text && strstr(text, ".capture_bindings \""));
    AsmResult result;
    NvmModule *assembled = asm_assemble_unverified(text, &result);
    CHECK(assembled && result.error == ASM_OK);
    CHECK(assembled->capture_size == payload_size);
    CHECK(memcmp(assembled->capture_data, saved_payload, payload_size) == 0);
    CHECK(assembled->code_size == copy->code_size);
    CHECK(memcmp(assembled->code, copy->code, copy->code_size) == 0);
    uint8_t *text_wire = wire(assembled, &copied_size);
    CHECK(copied_size == size && memcmp(text_wire, saved_wire, size) == 0);
    free(text_wire);
    CHECK(asm_assemble(text, &result) == NULL && result.error == ASM_ERR_VERIFY);
    free(text);
    nvm_module_free(assembled);
    nvm_module_free(copy);
    free(saved_payload);
    free(saved_wire);
}

int main(void) {
    roundtrip(false);
    roundtrip(true);
    printf("I passed %u capture transport checks.\n", checks);
    return 0;
}
