#include "nvm_v2_sections.h"
#include "verifier.h"
#include "isa.h"
#include "assembler.h"
#include "disassembler.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static NvmModule *fixture(void) {
    NvmModule *m = nvm_module_new();
    assert(m);
    uint32_t library = nvm_add_string(m, "fixture", 7);
    uint32_t submit = nvm_add_string(m, "submit", 6);
    uint32_t adapter = nvm_add_string(m, "retained_submit", 15);
    uint32_t wait = nvm_add_string(m, "wait", 4);
    uint32_t wait_adapter = nvm_add_string(m, "retained_wait", 13);
    uint8_t params[] = {TAG_OPAQUE, TAG_FUNCTION, TAG_FUNCTION};
    assert(nvm_add_import(m, library, submit, 3, TAG_VOID, params) == 0);
    assert(nvm_add_import(m, library, wait, 1, TAG_VOID, params) == 1);
    NvmCallbackContract c = {.import_idx = 0, .adapter_name_idx = adapter,
        .parameter_idx = 1, .abi_version = NVM_CALLBACK_ABI_RETAINED_V1,
        .execution = NVM_FOREIGN_WORKER_THREAD, .param_count = 2,
        .return_tag = TAG_BOOL, .param_tags = {TAG_INT, TAG_FLOAT}};
    assert(nvm_add_callback_contract(m, &c));
    c.parameter_idx = 2; c.param_count = 1;
    c.param_tags[0] = TAG_U8; c.return_tag = TAG_VOID;
    assert(nvm_add_callback_contract(m, &c));
    c.import_idx = 1; c.parameter_idx = NVM_CALLBACK_NO_PARAMETER;
    c.param_count = 0; c.adapter_name_idx = wait_adapter;
    assert(nvm_add_callback_contract(m, &c));
    uint8_t code[] = {OP_RET};
    NvmFunctionEntry fn = {.name_idx = nvm_add_string(m, "main", 4),
                           .code_length = 1, .result_tag = TAG_VOID};
    nvm_append_code(m, code, sizeof(code));
    assert(nvm_add_function(m, &fn) == 0);
    m->header.flags |= NVM_FLAG_HAS_MAIN;
    assert(nvm_callback_contracts_valid(m));
    assert(nvm_verify(m).ok);
    return m;
}

static uint8_t *encode(NvmV2Module *m, size_t *size) {
    assert(nvm_v2_module_serialize(m, NULL, 0, size) == NVM_V2_OK);
    uint8_t *bytes = malloc(*size);
    assert(bytes);
    assert(nvm_v2_module_serialize(m, bytes, *size, size) == NVM_V2_OK);
    return bytes;
}

static void reject(NvmV2Module *m) {
    size_t size;
    uint8_t *bytes = encode(m, &size);
    NvmV2Module decoded;
    assert(nvm_v2_module_deserialize(bytes, size, &decoded) != NVM_V2_OK);
    free(bytes);
}

static void test_roundtrip(void) {
    NvmModule *m = fixture();
    uint32_t legacy_size = 99;
    assert(!nvm_serialize(m, &legacy_size) && legacy_size == 0);
    NvmV2Module wire;
    assert(nvm_v2_from_nvm_module(m, &wire) == NVM_V2_OK);
    size_t size;
    uint8_t *bytes = encode(&wire, &size);
    NvmV2Header header;
    assert(nvm_v2_read_header(bytes, size, &header) == NVM_V2_OK);
    assert(header.feature_bits & NVM_V2_FEATURE_CALLBACKS);
    NvmV2Module decoded;
    assert(nvm_v2_module_deserialize(bytes, size, &decoded) == NVM_V2_OK);
    NvmModule *back = NULL;
    assert(nvm_v2_to_nvm_module(&decoded, &back) == NVM_V2_OK && back);
    assert(nvm_callback_contracts_valid(back) && nvm_verify(back).ok);
    assert(back->callback_contract_count == 3);
    const NvmCallbackContract *c = back->callback_contracts;
    assert(c[0].param_count == 2 && c[0].return_tag == TAG_BOOL &&
           c[0].param_tags[0] == TAG_INT && c[0].param_tags[1] == TAG_FLOAT);
    assert(c[1].param_count == 1 && c[1].param_tags[0] == TAG_U8 && c[1].return_tag == TAG_VOID);
    assert(c[2].parameter_idx == NVM_CALLBACK_NO_PARAMETER && c[2].param_count == 0);
    assert(strcmp(nvm_get_string(back, c[0].adapter_name_idx), "retained_submit") == 0);
    nvm_module_free(back);
    nvm_v2_module_free(&decoded);
    /* I test every truncated file prefix, not just the missing final byte. */
    for (size_t n = 0; n < size; n++)
        assert(nvm_v2_module_deserialize(bytes, n, &decoded) != NVM_V2_OK);
    bytes[8] &= (uint8_t)~NVM_V2_FEATURE_CALLBACKS;
    assert(nvm_v2_module_deserialize(bytes, size, &decoded) == NVM_V2_ERR_FEATURE_MISMATCH);
    free(bytes);
    nvm_v2_module_free(&wire);
    nvm_module_free(m);
}

static void test_invalid_contracts(void) {
    NvmModule *m = fixture();
    NvmV2Module wire;
    assert(nvm_v2_from_nvm_module(m, &wire) == NVM_V2_OK);
    NvmV2Callback *c = &wire.callbacks.items[0], saved = *c;
    c->import_idx = UINT32_MAX; reject(&wire); *c = saved;
    c->parameter_idx = 0; reject(&wire); *c = saved;
    c->parameter_idx = 2; reject(&wire); *c = saved;
    c->parameter_idx = NVM_CALLBACK_NO_PARAMETER; reject(&wire); *c = saved;
    c->abi_version = 0; reject(&wire); *c = saved;
    c->abi_version = 2; reject(&wire); *c = saved;
    c->execution = 2; reject(&wire); *c = saved;
    c->execution = NVM_FOREIGN_OWNER_THREAD; reject(&wire); *c = saved;
    c->signature_idx = NVM_V2_NO_INDEX; reject(&wire); *c = saved;
    c->adapter_name_idx = UINT32_MAX; reject(&wire); *c = saved;
    uint32_t adapter = c->adapter_name_idx;
    NvmV2Constant old_name = wire.constants.items[adapter];
    wire.constants.items[adapter].length = 0; reject(&wire);
    wire.constants.items[adapter] = old_name;
    const uint8_t nul_name[] = {'a', 0, 'b'};
    wire.constants.items[adapter].payload = nul_name;
    wire.constants.items[adapter].length = sizeof(nul_name); reject(&wire);
    wire.constants.items[adapter] = old_name;
    NvmV2Signature *shape = &wire.signatures.items[c->signature_idx], old_shape = *shape;
    const uint8_t bad[] = {TAG_VOID, TAG_STRING, TAG_ARRAY, TAG_FUNCTION, TAG_CLOSURE};
    shape->param_count = 1;
    for (size_t i = 0; i < sizeof(bad); i++) { shape->param_tags = &bad[i]; reject(&wire); }
    *shape = old_shape;
    uint8_t too_many[17] = {0};
    memset(too_many, TAG_INT, sizeof(too_many));
    shape->param_count = 17; shape->param_tags = too_many; reject(&wire); *shape = old_shape;
    shape->result_count = 2; shape->result_tags = too_many; reject(&wire); *shape = old_shape;
    shape->result_tags = bad; reject(&wire); *shape = old_shape;
    NvmV2Callback policy = wire.callbacks.items[2];
    wire.callbacks.items[2].signature_idx = c->signature_idx; reject(&wire);
    wire.callbacks.items[2] = policy;
    /* Removing one callback argument contract must not authorize the others. */
    wire.callbacks.items[1] = policy;
    wire.callbacks.count = 2; reject(&wire);
    wire.callbacks.count = 0;
    wire.extra_features |= NVM_V2_FEATURE_CALLBACKS; reject(&wire);
    nvm_v2_module_free(&wire);
    m->callback_contracts[0].param_tags[0] = TAG_VOID;
    assert(!nvm_callback_contracts_valid(m) && !nvm_verify(m).ok);
    assert(nvm_v2_from_nvm_module(m, &wire) != NVM_V2_OK);
    nvm_module_free(m);
}

static void test_codec(void) {
    NvmModule *m = fixture();
    NvmV2Module wire;
    assert(nvm_v2_from_nvm_module(m, &wire) == NVM_V2_OK);
    size_t size = nvm_v2_callbacks_encoded_size(&wire.callbacks);
    uint8_t bytes[128];
    assert(size < sizeof(bytes));
    assert(nvm_v2_callbacks_encode(&wire.callbacks, bytes, size - 1) == NVM_V2_ERR_TRUNCATED);
    assert(nvm_v2_callbacks_encode(&wire.callbacks, bytes, size) == NVM_V2_OK);
    NvmV2Callbacks decoded;
    for (size_t n = 0; n < size; n++)
        assert(nvm_v2_callbacks_decode(bytes, n, &decoded) != NVM_V2_OK);
    assert(nvm_v2_callbacks_decode(bytes, size + 1, &decoded) != NVM_V2_OK);
    memset(bytes, 255, 4);
    assert(nvm_v2_callbacks_decode(bytes, size, &decoded) == NVM_V2_ERR_TRUNCATED);
    nvm_v2_module_free(&wire);
    nvm_module_free(m);
}

static void test_assembly(void) {
    NvmModule *m = fixture();
    const char *path = "/tmp/callback-fixture.so";
    m->imports[0].module_name_idx = nvm_add_string(m, path, strlen(path));
    m->imports[0].kind = NVM_IMPORT_ARTIFACT;
    m->imports[1].kind = NVM_IMPORT_COPROCESS;
    uint8_t code[] = {OP_LOAD_LOCAL, 0, 0, OP_RET};
    NvmFunctionEntry fn = {.name_idx = nvm_add_string(m, "identity", 8),
        .arity = 1, .local_count = 1, .result_count = 1,
        .result_tag = TAG_INT, .code_offset = m->code_size,
        .code_length = sizeof(code)};
    nvm_append_code(m, code, sizeof(code));
    assert(nvm_add_function(m, &fn) == 1);
    uint8_t tag = TAG_INT;
    assert(nvm_set_function_param_types(m, 1, &tag, 1));
    assert(nvm_verify(m).ok);
    char *source = disasm_module_styled(m, DISASM_STYLE_CANONICAL);
    assert(source);
    AsmResult result;
    NvmModule *back = asm_assemble(source, &result);
    if (!back) fprintf(stderr, "%s\nline %u: %s\n", source, result.line, result.message);
    assert(back && nvm_verify(back).ok);
    assert(back->imports[0].kind == NVM_IMPORT_ARTIFACT);
    assert(back->imports[1].kind == NVM_IMPORT_COPROCESS);
    assert(back->function_param_types[1][0] == TAG_INT);
    assert(back->callback_contract_count == m->callback_contract_count);
    for (uint32_t i = 0; i < m->callback_contract_count; i++) {
        const NvmCallbackContract *a = &m->callback_contracts[i];
        const NvmCallbackContract *b = &back->callback_contracts[i];
        assert(a->import_idx == b->import_idx && a->parameter_idx == b->parameter_idx);
        assert(a->abi_version == b->abi_version && a->execution == b->execution);
        assert(a->param_count == b->param_count && a->return_tag == b->return_tag);
        assert(!memcmp(a->param_tags, b->param_tags, a->param_count));
        assert(!strcmp(nvm_get_string(m, a->adapter_name_idx),
                       nvm_get_string(back, b->adapter_name_idx)));
    }
    free(source);
    nvm_module_free(back);
    nvm_module_free(m);
    const char *bad[] = {
        ".callback 0 0 \"adapter\" 2 worker void\n",
        ".callback 0 0 \"adapter\" 1 unknown void\n",
        ".parameters 0 int\n",
        ".function f 1 1 0 void 0\nRET\n.end\n.parameters 0\n",
        ".function f 1 1 0 void 0\nRET\n.end\n.parameters 0 int float\n",
        ".import_kind 0 artifact\n"
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
        assert(!asm_assemble(bad[i], &result));
}

int main(void) {
    test_roundtrip();
    test_invalid_contracts();
    test_codec();
    test_assembly();
    puts("I passed callback contract round-trip and malformed-input checks.");
    return 0;
}
