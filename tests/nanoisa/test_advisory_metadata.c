/* I preserve advisory facts without granting execution authority. */
#include "nvm_format.h"
#include "nvm_v2_sections.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "nvm2c.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
#define CHECK(x) do { ++checks; assert(x); } while (0)
static void word(uint8_t *p, uint32_t n) { for (unsigned i=0;i<4;i++) p[i]=(uint8_t)(n>>(8*i)); }
static void slot(uint8_t *p,uint8_t tag,uint8_t mode,uint32_t layout) {
    p[0]=tag; p[1]=mode; word(p+4,layout);
}
#include "multi_caller_fixture.h"
static uint8_t *wire(NvmModule *m, size_t *size) {
    NvmV2Module v;
    CHECK(nvm_v2_from_nvm_module(m, &v) == NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v, NULL, 0, size) == NVM_V2_OK);
    uint8_t *p = malloc(*size); CHECK(p);
    CHECK(nvm_v2_module_serialize(&v, p, *size, size) == NVM_V2_OK);
    nvm_v2_module_free(&v); return p;
}
static NvmModule *copy(NvmModule *m) {
    size_t size; uint8_t *p = wire(m, &size);
    NvmV2Module v; CHECK(nvm_v2_module_deserialize(p, size, &v) == NVM_V2_OK);
    NvmModule *out = NULL; CHECK(nvm_v2_to_nvm_module(&v, &out) == NVM_V2_OK);
    nvm_v2_module_free(&v); free(p); return out;
}
static NvmModule *ordinary(void) {
    AsmResult r;
    NvmModule *m = asm_assemble(".function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.entry main\n", &r);
    CHECK(m); return m;
}
static void compare_wire(NvmModule *a, NvmModule *b) {
    size_t na, nb; uint8_t *pa = wire(a, &na), *pb = wire(b, &nb);
    CHECK(na == nb && memcmp(pa, pb, na) == 0); free(pa); free(pb);
}
static void test_purity(void) {
    CHECK(nvm_function_purity(NULL, 0) == -1);
    CHECK(!nvm_add_function_purity(NULL, 0, true));
    NvmModule *m = ordinary();
    CHECK(nvm_function_purity(m, 0) == -1);
    CHECK(!nvm_add_function_purity(m, 1, true));
    CHECK(nvm_add_function_purity(m, 0, true));
    CHECK(nvm_function_purity(m, 0) == 1);
    CHECK(nvm_function_purity(m, 1) == -1);
    CHECK(!nvm_add_function_purity(m, 0, false));
    NvmModule *b = copy(m); compare_wire(m, b);
    CHECK(nvm_function_purity(b, 0) == 1);
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult result; NvmModule *c = asm_assemble(text, &result); CHECK(c);
    compare_wire(b, c); CHECK(nvm_function_purity(c, 0) == 1);
    free(text); nvm_module_free(b); nvm_module_free(c);
    nvm_strip_debug_info(m); CHECK(nvm_function_purity(m, 0) == 1);
    NvmMetadataEntry e = m->metadata[0];
    CHECK(nvm_add_metadata(m, e.key_idx, e.value_idx));
    CHECK(nvm_function_purity(m, 0) == -1); CHECK(nvm_verify(m).ok);
    b = copy(m); compare_wire(m, b); nvm_module_free(b); nvm_module_free(m);
    const unsigned char malformed[][8] = {
        {1,0,0,0,1,0,0,0}, {0,0,0,0,2,0,0,0},
        {0,0,0,0,1,1,0,0}, {0,0,0,0,1,0,1,0}, {0,0,0,0,1,0,0,1}
    };
    for (size_t i = 0; i < sizeof malformed / sizeof *malformed; ++i) {
        m = ordinary();
        uint32_t key = nvm_add_string(m, "nano.purity.v1", 14);
        uint32_t value = nvm_add_string(m, (const char *)malformed[i], 8);
        CHECK(nvm_add_metadata(m, key, value));
        CHECK(nvm_function_purity(m, 0) == -1);
        CHECK(!nvm_add_function_purity(m, 0, true));
        CHECK(nvm_verify(m).ok); b = copy(m); compare_wire(m, b);
        nvm_module_free(b); nvm_module_free(m);
    }
    m = ordinary();
    uint32_t key = nvm_add_string(m, "nano.purity.v1", 14);
    uint32_t value = nvm_add_string(m, "", 0);
    CHECK(nvm_add_metadata(m, key, value));
    CHECK(nvm_function_purity(m, 0) == -1); CHECK(nvm_verify(m).ok);
    nvm_module_free(m);
    m = ordinary(); key = nvm_add_string(m, "nano.purity.v2", 14);
    CHECK(nvm_add_metadata(m, key, key));
    CHECK(nvm_function_purity(m, 0) == -1);
    CHECK(nvm_add_function_purity(m, 0, false));
    CHECK(nvm_function_purity(m, 0) == 0); nvm_module_free(m);
    m = asm_assemble(".function main 0 0 0 int 1\n.function_purity 0\nPUSH_I64 42\nRET\n.end\n.entry main\n", &result);
    CHECK(m); CHECK(nvm_function_purity(m, 0) == 0); nvm_module_free(m);
    const char *invalid[] = {".function_purity 1\n", ".function main 0 0 0 void 0\n.function_purity 2\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.function_purity 1 junk\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.function_purity 1\n.function_purity 1\nRET\n.end\n"};
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i) {
        m = asm_assemble(invalid[i], &result); CHECK(!m);
    }
}
static void test_match_coverage(void) {
    NvmMatchCoverage view = {.line = 123};
    CHECK(!nvm_match_coverage_at(NULL, 0, &view) && view.line == 123);
    CHECK(!nvm_add_match_coverage(NULL, 0, 1, 1, 1, 1, "a.nano"));
    NvmModule *m = ordinary();
    CHECK(!nvm_add_match_coverage(m, 0, 0, 1, 1, 1, "a.nano"));
    CHECK(!nvm_add_match_coverage(m, 0, 1, 1, 3, 1, "a.nano"));
    CHECK(!nvm_add_match_coverage(m, 0, 1, 1, 1, 0, "a.nano"));
    CHECK(nvm_add_match_coverage(m, 0, 3, 7, 1, 1, "a.nano"));
    CHECK(nvm_match_coverage_at(m, 0, &view));
    CHECK(view.function == 0 && view.line == 3 && view.column == 7 && view.coverage == 1);
    CHECK(view.coordinates == 1 && view.source_path_length == 6 && !memcmp(view.source_path, "a.nano", 6));
    CHECK(!nvm_add_match_coverage(m, 0, 3, 7, 0, 1, "a.nano"));
    CHECK(nvm_add_match_coverage(m, 0, 3, 8, 0, 1, "a.nano"));
    CHECK(nvm_add_match_coverage(m, 0, 40, 7, 2, 2, ""));
    CHECK(nvm_match_coverage_at(m, 1, &view) && view.coverage == 0);
    CHECK(nvm_match_coverage_at(m, 2, &view) && view.coverage == 2);
    NvmModule *b = copy(m); compare_wire(m, b);
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult r; NvmModule *c = asm_assemble(text, &r); CHECK(c); compare_wire(b, c);
    free(text); nvm_module_free(b); nvm_module_free(c);
    CHECK(nvm_verify(m).ok);
    uint32_t value = m->metadata[0].value_idx;
    char *data = m->strings[value];
    const unsigned offsets[] = {0, 4, 8, 12, 13, 14, 15, 16, 20};
    for (unsigned i = 0; i < sizeof offsets / sizeof *offsets; ++i) {
        unsigned offset = offsets[i]; char old = data[offset];
        data[offset] = (offset == 4 || offset == 8 || offset == 13 || offset == 20) ? 0 : 127;
        view.line = 123;
        CHECK(!nvm_match_coverage_at(m, 0, &view) && view.line == 123);
        CHECK(nvm_verify(m).ok); /* I do not make advisory validity execution authority. */
        data[offset] = old;
    }
    uint32_t key = m->metadata[0].key_idx;
    char *key_text = m->strings[key];
    key_text[m->string_lengths[key]-1] = '2';
    CHECK(!nvm_match_coverage_at(m, 0, &view));
    CHECK(nvm_verify(m).ok);
    key_text[m->string_lengths[key]-1] = '1';
    uint32_t original_size = m->string_lengths[value];
    m->string_lengths[value] = 19;
    CHECK(!nvm_match_coverage_at(m, 0, &view));
    m->string_lengths[value] = original_size;
    CHECK(nvm_add_metadata(m, m->metadata[0].key_idx, value));
    CHECK(!nvm_match_coverage_at(m, 0, &view));
    nvm_module_free(m);
    m = asm_assemble(".function main 0 0 0 int 1\n.match_coverage 3 7 2 2 \"a.nano\"\nPUSH_I64 42\nRET\n.end\n.entry main\n", &r);
    CHECK(m && nvm_match_coverage_at(m, 0, &view) && view.coverage == 2);
    nvm_module_free(m);
    const char *bad[] = {"0 1 1 1 \"a\"", "1 1 3 1 \"a\"", "1 1 1 3 \"a\"", "1 1 1 1 \"a\" extra", "1 1 1 1 \"a\\0b\""};
    for (unsigned i = 0; i < sizeof bad / sizeof *bad; ++i) {
        char source[512]; snprintf(source, sizeof source, ".function main 0 0 0 int 1\n.match_coverage %s\nPUSH_I64 42\nRET\n.end\n", bad[i]);
        m = asm_assemble(source, &r); CHECK(!m);
    }
}

static void test_affine_use(void) {
    NvmAffineUse view = {.line = 123};
    CHECK(!nvm_affine_use_at(NULL, 0, &view) && view.line == 123);
    NvmModule *m = ordinary();
    CHECK(!nvm_add_affine_use(m, 0, 1, 1, 1, 1, "", ""));
    CHECK(!nvm_add_affine_use(m, 1, 1, 1, 1, 1, "", "x"));
    CHECK(!nvm_add_affine_use(m, 0, 0, 1, 1, 1, "", "x"));
    CHECK(!nvm_add_affine_use(m, 0, 1, 1, 3, 1, "", "x"));
    CHECK(!nvm_add_affine_use(m, 0, 1, 1, 1, 0, "", "x"));
    CHECK(nvm_add_affine_use(m, 0, 3, 7, 1, 1, "a.nano", "owner"));
    CHECK(nvm_affine_use_at(m, 0, &view));
    CHECK(view.function == 0 && view.line == 3 && view.column == 7 && view.state == 1);
    CHECK(view.coordinates == 1 && view.source_path_length == 6 && !memcmp(view.source_path, "a.nano", 6));
    CHECK(view.binding_length == 5 && !memcmp(view.binding, "owner", 5));
    /* I never use binding spelling as site identity. */
    CHECK(!nvm_add_affine_use(m, 0, 3, 7, 2, 1, "a.nano", "different"));
    CHECK(nvm_add_affine_use(m, 0, 3, 7, 2, 2, "a.nano", "owner"));
    CHECK(nvm_add_affine_use(m, 0, 3, 8, 0, 1, "", "x"));
    CHECK(nvm_affine_use_at(m, 1, &view) && view.state == 2);
    CHECK(nvm_affine_use_at(m, 2, &view) && view.state == 0);
    NvmModule *b = copy(m); compare_wire(m, b);
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult r; NvmModule *c = asm_assemble(text, &r); CHECK(c); compare_wire(b, c);
    free(text); nvm_module_free(b); nvm_module_free(c);
    char error[256]; char *native = nvm2c_emit(m, error, sizeof error); CHECK(native);
    uint32_t value = m->metadata[0].value_idx;
    char *data = m->strings[value];
    const unsigned offsets[] = {0, 4, 8, 12, 13, 14, 15, 16, 20, 24, 30};
    for (unsigned i = 0; i < sizeof offsets / sizeof *offsets; ++i) {
        unsigned offset = offsets[i]; char old = data[offset];
        data[offset] = (offset == 4 || offset == 8 || offset == 13 || offset >= 24) ? 0 : 127;
        view.line = 123;
        CHECK(!nvm_affine_use_at(m, 0, &view) && view.line == 123);
        /* I fail closed on another malformed entry, without execution authority. */
        CHECK(!nvm_affine_use_at(m, 1, &view) && view.line == 123);
        CHECK(nvm_verify(m).ok);
        char *again = nvm2c_emit(m, error, sizeof error); CHECK(again && !strcmp(native, again)); free(again);
        data[offset] = old;
    }
    free(native);
    uint32_t original_size = m->string_lengths[value];
    m->string_lengths[value] = 23;
    CHECK(!nvm_affine_use_at(m, 0, &view));
    m->string_lengths[value] = original_size;
    /* I forge a duplicate with a different spelling, bypassing the safe writer. */
    char duplicate[64]; CHECK(original_size < sizeof duplicate);
    memcpy(duplicate, data, original_size); memcpy(duplicate + 30, "other", 5);
    uint32_t other = nvm_add_string(m, duplicate, original_size);
    CHECK(nvm_add_metadata(m, m->metadata[0].key_idx, other));
    view.line = 123;
    CHECK(!nvm_affine_use_at(m, 0, &view) && view.line == 123);
    CHECK(nvm_verify(m).ok); nvm_module_free(m);
    const char *prefix = ".function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.entry main\n";
    char source[1024];
    snprintf(source, sizeof source, "%s.affine_use 0 3 7 2 2 \"a.nano\" \"owner\"\n", prefix);
    m = asm_assemble(source, &r); CHECK(m && nvm_affine_use_at(m, 0, &view) && view.state == 2); nvm_module_free(m);
    const char *bad[] = {"0 0 1 1 1 \"a\" \"x\"", "0 1 1 3 1 \"a\" \"x\"", "0 1 1 1 3 \"a\" \"x\"", "0 1 1 1 1 \"a\" \"\"", "0 1 1 1 1 \"a\" \"x\" extra", "0 1 1 1 1 \"a\\0b\" \"x\"", "0 1 1 1 1 \"a\" \"x\\0y\"", "1 1 1 1 1 \"a\" \"x\""};
    for (unsigned i = 0; i < sizeof bad / sizeof *bad; ++i) {
        snprintf(source, sizeof source, "%s.affine_use %s\n", prefix, bad[i]);
        CHECK(!asm_assemble(source, &r));
    }
    CHECK(!asm_assemble(".function main 0 0 0 int 1\n.affine_use 0 1 1 1 1 \"a\" \"x\"\nPUSH_I64 42\nRET\n.end\n", &r));
}

static void test_effects(void) {
    CHECK(nvm_function_effects(NULL, 0) == -1);
    CHECK(!nvm_add_function_effects(NULL, 0, 63));
    NvmModule *m = ordinary();
    CHECK(nvm_function_effects(m, 0) == -1);
    CHECK(!nvm_add_function_effects(m, 1, 63));
    CHECK(!nvm_add_function_effects(m, 0, 64));
    CHECK(nvm_add_function_effects(m, 0, 63));
    CHECK(nvm_function_effects(m, 0) == 63);
    CHECK(nvm_function_effects(m, 1) == -1);
    CHECK(!nvm_add_function_effects(m, 0, 0));
    NvmModule *b = copy(m); compare_wire(m, b);
    CHECK(nvm_function_effects(b, 0) == 63);
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult result; NvmModule *c = asm_assemble(text, &result); CHECK(c);
    compare_wire(b, c); CHECK(nvm_function_effects(c, 0) == 63);
    free(text); nvm_module_free(b); nvm_module_free(c);
    nvm_strip_debug_info(m); CHECK(nvm_function_effects(m, 0) == 63);
    NvmMetadataEntry e = m->metadata[0];
    CHECK(nvm_add_metadata(m, e.key_idx, e.value_idx));
    CHECK(nvm_function_effects(m, 0) == -1); CHECK(nvm_verify(m).ok);
    b = copy(m); compare_wire(m, b); nvm_module_free(b); nvm_module_free(m);
    const unsigned char malformed[][8] = {
        {1,0,0,0,1,0,0,0}, {0,0,0,0,64,0,0,0},
        {0,0,0,0,1,1,0,0}, {0,0,0,0,1,0,1,0}, {0,0,0,0,1,0,0,1}
    };
    for (size_t i = 0; i < sizeof malformed / sizeof *malformed; ++i) {
        m = ordinary();
        uint32_t key = nvm_add_string(m, "nano.effects.v1", 15);
        uint32_t value = nvm_add_string(m, (const char *)malformed[i], 8);
        CHECK(nvm_add_metadata(m, key, value));
        CHECK(nvm_function_effects(m, 0) == -1);
        CHECK(!nvm_add_function_effects(m, 0, 63));
        CHECK(nvm_verify(m).ok); b = copy(m); compare_wire(m, b);
        nvm_module_free(b); nvm_module_free(m);
    }
    m = ordinary();
    uint32_t key = nvm_add_string(m, "nano.effects.v1", 15);
    uint32_t value = nvm_add_string(m, "", 0);
    CHECK(nvm_add_metadata(m, key, value));
    CHECK(nvm_function_effects(m, 0) == -1); CHECK(nvm_verify(m).ok);
    nvm_module_free(m);
    m = ordinary(); key = nvm_add_string(m, "nano.effects.v2", 15);
    CHECK(nvm_add_metadata(m, key, key));
    CHECK(nvm_function_effects(m, 0) == -1);
    CHECK(nvm_add_function_effects(m, 0, 0));
    CHECK(nvm_function_effects(m, 0) == 0); nvm_module_free(m);
    m = asm_assemble(".function main 0 0 0 int 1\n.function_effects 0\nPUSH_I64 42\nRET\n.end\n.entry main\n", &result);
    CHECK(m); CHECK(nvm_function_effects(m, 0) == 0); nvm_module_free(m);
    const char *invalid[] = {".function_effects 1\n", ".function main 0 0 0 void 0\n.function_effects 64\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.function_effects 1 junk\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.function_effects 1\n.function_effects 1\nRET\n.end\n"};
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i) {
        m = asm_assemble(invalid[i], &result); CHECK(!m);
    }
}
static void test_affine(void) {
    CHECK(nvm_function_affine(NULL, 0) == -1);
    CHECK(!nvm_add_function_affine(NULL, 0, 1));
    NvmModule *m = ordinary();
    CHECK(nvm_function_affine(m, 0) == -1);
    CHECK(!nvm_add_function_affine(m, 1, 1));
    CHECK(!nvm_add_function_affine(m, 0, 2));
    CHECK(nvm_add_function_affine(m, 0, 1));
    CHECK(nvm_function_affine(m, 0) == 1);
    CHECK(nvm_function_affine(m, 1) == -1);
    CHECK(!nvm_add_function_affine(m, 0, 0));
    NvmModule *b = copy(m); compare_wire(m, b);
    CHECK(nvm_function_affine(b, 0) == 1);
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult result; NvmModule *c = asm_assemble(text, &result); CHECK(c);
    compare_wire(b, c); CHECK(nvm_function_affine(c, 0) == 1);
    free(text); nvm_module_free(b); nvm_module_free(c);
    nvm_strip_debug_info(m); CHECK(nvm_function_affine(m, 0) == 1);
    NvmMetadataEntry e = m->metadata[0];
    CHECK(nvm_add_metadata(m, e.key_idx, e.value_idx));
    CHECK(nvm_function_affine(m, 0) == -1); CHECK(nvm_verify(m).ok);
    b = copy(m); compare_wire(m, b); nvm_module_free(b); nvm_module_free(m);
    const unsigned char malformed[][8] = {
        {1,0,0,0,1,0,0,0}, {0,0,0,0,2,0,0,0},
        {0,0,0,0,1,1,0,0}, {0,0,0,0,1,0,1,0}, {0,0,0,0,1,0,0,1}
    };
    for (size_t i = 0; i < sizeof malformed / sizeof *malformed; ++i) {
        m = ordinary();
        uint32_t key = nvm_add_string(m, "nano.affine.v1", 14);
        uint32_t value = nvm_add_string(m, (const char *)malformed[i], 8);
        CHECK(nvm_add_metadata(m, key, value));
        CHECK(nvm_function_affine(m, 0) == -1);
        CHECK(!nvm_add_function_affine(m, 0, 1));
        CHECK(nvm_verify(m).ok); b = copy(m); compare_wire(m, b);
        nvm_module_free(b); nvm_module_free(m);
    }
    m = ordinary();
    uint32_t key = nvm_add_string(m, "nano.affine.v1", 14);
    uint32_t value = nvm_add_string(m, "", 0);
    CHECK(nvm_add_metadata(m, key, value));
    CHECK(nvm_function_affine(m, 0) == -1); CHECK(nvm_verify(m).ok);
    nvm_module_free(m);
    m = ordinary(); key = nvm_add_string(m, "nano.affine.v2", 14);
    CHECK(nvm_add_metadata(m, key, key));
    CHECK(nvm_function_affine(m, 0) == -1);
    CHECK(nvm_add_function_affine(m, 0, 0));
    CHECK(nvm_function_affine(m, 0) == 0); nvm_module_free(m);
    m = asm_assemble(".function main 0 0 0 int 1\n.function_affine 0\nPUSH_I64 42\nRET\n.end\n.entry main\n", &result);
    CHECK(m); CHECK(nvm_function_affine(m, 0) == 0); nvm_module_free(m);
    m = asm_assemble(".function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.affine_completion 0 1\n.entry main\n", &result);
    CHECK(m); CHECK(nvm_function_affine(m, 0) == 1); nvm_module_free(m);
    const char *invalid[] = {".affine_completion 0 1\n", ".function main 0 0 0 int 1\n.affine_completion 0 1\nPUSH_I64 42\nRET\n.end\n", ".function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.affine_completion 0 2\n", ".function_affine 1\n", ".function main 0 0 0 void 0\n.function_affine 2\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.function_affine 1 junk\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.function_affine 1\n.function_affine 1\nRET\n.end\n"};
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i) {
        m = asm_assemble(invalid[i], &result); CHECK(!m);
    }
}
static void test_generic_bindings(void) {
    NvmGenericBinding view = {.function = 777};
    CHECK(!nvm_generic_binding_at(NULL, 0, &view) && view.function == 777);
    CHECK(!nvm_add_generic_binding(NULL, 0, "id", "file", "T", "int"));
    NvmModule *m = ordinary();
    CHECK(!nvm_add_generic_binding(m, 1, "id", "file", "T", "int"));
    CHECK(!nvm_add_generic_binding(m, 0, "", "file", "T", "int"));
    CHECK(!nvm_add_generic_binding(m, 0, "id", "file", "", "int"));
    CHECK(!nvm_add_generic_binding(m, 0, "id", "file", "T", ""));
    CHECK(nvm_add_generic_binding(m, 0, "id", "file.nano", "T", "array<int>"));
    CHECK(nvm_add_generic_binding(m, 0, "id", "file.nano", "U", "fn(int)->(int,bool)"));
    CHECK(!nvm_add_generic_binding(m, 0, "id", "file.nano", "T", "bool"));
    CHECK(!nvm_add_generic_binding(m, 0, "other", "file.nano", "V", "bool"));
    CHECK(!nvm_add_generic_binding(m, 0, "id", "other.nano", "V", "bool"));
    CHECK(nvm_generic_binding_at(m, 0, &view));
    CHECK(view.function == 0 && view.variable_length == 1 && view.variable[0] == 'T');
    CHECK(view.source_name_length == 2 && !memcmp(view.source_name, "id", 2));
    CHECK(view.source_path_length == 9 && !memcmp(view.source_path, "file.nano", 9));
    CHECK(view.type_length == 10 && !memcmp(view.type, "array<int>", 10));
    NvmModule *b = copy(m); compare_wire(m, b);
    CHECK(nvm_generic_binding_at(b, 1, &view) && view.variable[0] == 'U');
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult result; NvmModule *c = asm_assemble(text, &result); CHECK(c);
    compare_wire(b, c); CHECK(nvm_generic_binding_at(c, 0, &view));
    free(text); nvm_module_free(b); nvm_module_free(c);
    nvm_strip_debug_info(m); CHECK(nvm_generic_binding_at(m, 0, &view));
    NvmMetadataEntry entry = m->metadata[0];
    CHECK(nvm_add_metadata(m, entry.key_idx, entry.value_idx));
    view.function = 777;
    CHECK(!nvm_generic_binding_at(m, 0, &view) && view.function == 777);
    CHECK(nvm_verify(m).ok); b = copy(m); compare_wire(m, b);
    nvm_module_free(b); nvm_module_free(m);
    /* I refuse every truncated header/payload and impossible count, without
     * rejecting optional metadata at the execution boundary. */
    unsigned char bytes[24] = {0};
    word(bytes + 4, 1); word(bytes + 8, 1); word(bytes + 12, 1); word(bytes + 16, 1);
    memcpy(bytes + 20, "ifTi", 4);
    for (unsigned length = 0; length < sizeof bytes; ++length) {
        m = ordinary();
        uint32_t key = nvm_add_string(m, "nano.generic.v1", 15);
        uint32_t value = nvm_add_string(m, (const char *)bytes, length);
        CHECK(nvm_add_metadata(m, key, value));
        view.function = 777;
        CHECK(!nvm_generic_binding_at(m, 0, &view) && view.function == 777);
        CHECK(!nvm_add_generic_binding(m, 0, "id", "", "U", "int"));
        CHECK(nvm_verify(m).ok); nvm_module_free(m);
    }
    const unsigned offsets[] = {0, 4, 8, 12, 16, 20, 23};
    for (unsigned i = 0; i < sizeof offsets / sizeof *offsets; ++i) {
        unsigned char malformed[24]; memcpy(malformed, bytes, sizeof bytes);
        if (offsets[i] < 20) word(malformed + offsets[i], UINT32_MAX);
        else malformed[offsets[i]] = 0;
        m = ordinary();
        uint32_t key = nvm_add_string(m, "nano.generic.v1", 15);
        uint32_t value = nvm_add_string(m, (const char *)malformed, sizeof malformed);
        CHECK(nvm_add_metadata(m, key, value)); CHECK(!nvm_generic_binding_at(m, 0, &view));
        CHECK(nvm_verify(m).ok); nvm_module_free(m);
    }
    m = ordinary();
    uint32_t key = nvm_add_string(m, "nano.generic.v2", 15);
    CHECK(nvm_add_metadata(m, key, key)); CHECK(!nvm_generic_binding_at(m, 0, &view));
    CHECK(nvm_add_generic_binding(m, 0, "id", "", "T", "int"));
    CHECK(nvm_generic_binding_at(m, 1, &view) && view.source_path_length == 0);
    nvm_module_free(m);
    m = asm_assemble(".function main 0 0 0 int 1\n.generic_binding \"id\" \"\" \"T\" \"int\"\nPUSH_I64 42\nRET\n.end\n.entry main\n", &result);
    CHECK(m); CHECK(nvm_generic_binding_at(m, 0, &view)); nvm_module_free(m);
    const char *bad[] = {".generic_binding \"id\" \"\" \"T\" \"int\"\n",
        ".function main 0 0 0 void 0\n.generic_binding \"id\" \"\" \"T\"\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.generic_binding \"id\" \"\" \"T\" \"int\" extra\nRET\n.end\n",
        ".function main 0 0 0 void 0\n.generic_binding \"id\" \"\" \"T\" \"int\"\n.generic_binding \"id\" \"\" \"T\" \"int\"\nRET\n.end\n"};
    for (unsigned i = 0; i < sizeof bad / sizeof *bad; ++i) {
        m = asm_assemble(bad[i], &result); CHECK(!m);
    }
}
static void test_borrow_places(void) {
    NvmModule *m=ordinary();
    NvmBorrowPlace fact={0,2,3,2,9,0,2,1,"module.nano","owner",".left",11,5,5},view={0};
    CHECK(nvm_add_borrow_place(m,&fact));CHECK(nvm_borrow_place_at(m,0,&view));
    CHECK(view.mode==2 && view.ordinal==0 && view.projection_length==5);
    NvmBorrowPlace alternate=fact;alternate.mode=1;alternate.root="other";
    CHECK(!nvm_add_borrow_place(m,&alternate)); /* descriptive changes do not create another site */
    NvmModule *c=copy(m);compare_wire(m,c);CHECK(nvm_borrow_place_at(c,0,&view));
    char *text=disasm_module_styled(c,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult assembled;NvmModule *round=asm_assemble(text,&assembled);CHECK(round);compare_wire(m,round);
    free(text);nvm_module_free(round);nvm_module_free(c);
    char error[256];char *native=nvm2c_emit(m,error,sizeof error);CHECK(native);
    uint32_t value=m->metadata[0].value_idx,size=m->string_lengths[value];
    uint8_t *bytes=(uint8_t *)m->strings[value];uint8_t saved[128];CHECK(size<=sizeof saved);memcpy(saved,bytes,size);
    const unsigned offsets[]={0,4,8,12,16,24,25,26,27,28,32,36,40};
    for(size_t i=0;i<sizeof offsets/sizeof *offsets;i++) {
        memcpy(bytes,saved,size);
        if(offsets[i]<=16)word(bytes+offsets[i],offsets[i]?0:UINT32_MAX);
        else if(offsets[i]>=28 && offsets[i]<=36)word(bytes+offsets[i],UINT32_MAX);
        else bytes[offsets[i]]=offsets[i]==40?0xff:0xff;
        view.call_line=777;CHECK(!nvm_borrow_place_at(m,0,&view) && view.call_line==777);
        CHECK(nvm_verify(m).ok);char *again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);
    }
    memcpy(bytes,saved,size);
    for(uint32_t n=0;n<size;n++) {m->string_lengths[value]=n;view.call_line=777;CHECK(!nvm_borrow_place_at(m,0,&view) && view.call_line==777);}
    m->string_lengths[value]=size;
    /* A forged duplicate with a changed mode is still ambiguous. */
    saved[24]=1;uint32_t duplicate=nvm_add_string(m,(const char *)saved,size);
    CHECK(duplicate!=UINT32_MAX && nvm_add_metadata(m,m->metadata[0].key_idx,duplicate));
    view.call_line=777;CHECK(!nvm_borrow_place_at(m,0,&view) && view.call_line==777);
    CHECK(!nvm_borrow_place_at(m,1,&view) && view.call_line==777);
    CHECK(nvm_verify(m).ok);char *again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);
    m->metadata_count=0;again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);free(native);
    nvm_module_free(m);
    m=ordinary();fact.projection="";fact.projection_length=0;CHECK(nvm_add_borrow_place(m,&fact));
    CHECK(nvm_borrow_place_at(m,0,&view) && !view.projection_length);
    uint32_t unknown=nvm_add_string(m,"nano.borrow-place.v9",20),junk=nvm_add_string(m,"opaque",6);
    CHECK(nvm_add_metadata(m,unknown,junk));CHECK(nvm_borrow_place_at(m,0,&view));nvm_module_free(m);
}

static void test_field_reads(void) {
    NvmModule *m=ordinary();
    NvmFieldRead fact={0,2,3,2,1,"module.nano","owner",".left",11,5,5},view={0};
    CHECK(nvm_add_field_read(m,&fact));CHECK(nvm_field_read_at(m,0,&view));
    CHECK(view.category==2 && view.projection_length==5);
    NvmFieldRead alternate=fact;alternate.category=1;alternate.root="other";
    CHECK(!nvm_add_field_read(m,&alternate)); /* descriptive changes do not create another site */
    NvmModule *c=copy(m);compare_wire(m,c);CHECK(nvm_field_read_at(c,0,&view));
    char *text=disasm_module_styled(c,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult assembled;NvmModule *round=asm_assemble(text,&assembled);CHECK(round);compare_wire(m,round);
    free(text);nvm_module_free(round);nvm_module_free(c);
    char error[256];char *native=nvm2c_emit(m,error,sizeof error);CHECK(native);
    uint32_t value=m->metadata[0].value_idx,size=m->string_lengths[value];
    uint8_t *bytes=(uint8_t *)m->strings[value];uint8_t saved[128];CHECK(size<=sizeof saved);memcpy(saved,bytes,size);
    const unsigned offsets[]={0,4,8,12,13,14,15,16,20,24,28};
    for(size_t i=0;i<sizeof offsets/sizeof *offsets;i++) {
        memcpy(bytes,saved,size);
        if(offsets[i]<=8)word(bytes+offsets[i],offsets[i]?0:UINT32_MAX);
        else if(offsets[i]>=16 && offsets[i]<=24)word(bytes+offsets[i],UINT32_MAX);
        else bytes[offsets[i]]=offsets[i]==28?0xff:0xff;
        view.line=777;CHECK(!nvm_field_read_at(m,0,&view) && view.line==777);
        CHECK(nvm_verify(m).ok);char *again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);
    }
    memcpy(bytes,saved,size);
    for(uint32_t n=0;n<size;n++) {m->string_lengths[value]=n;view.line=777;CHECK(!nvm_field_read_at(m,0,&view) && view.line==777);}
    m->string_lengths[value]=size;
    /* A forged duplicate with a changed category is still ambiguous. */
    saved[12]=1;uint32_t duplicate=nvm_add_string(m,(const char *)saved,size);
    CHECK(duplicate!=UINT32_MAX && nvm_add_metadata(m,m->metadata[0].key_idx,duplicate));
    view.line=777;CHECK(!nvm_field_read_at(m,0,&view) && view.line==777);
    CHECK(!nvm_field_read_at(m,1,&view) && view.line==777);
    CHECK(nvm_verify(m).ok);char *again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);
    m->metadata_count=0;again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);free(native);
    nvm_module_free(m);
    m=ordinary();fact.projection="";fact.projection_length=0;CHECK(!nvm_add_field_read(m,&fact));
    fact.projection=".left";fact.projection_length=5;CHECK(nvm_add_field_read(m,&fact));
    uint32_t unknown=nvm_add_string(m,"nano.field-read.v9",18),junk=nvm_add_string(m,"opaque",6);
    CHECK(nvm_add_metadata(m,unknown,junk));CHECK(nvm_field_read_at(m,0,&view));nvm_module_free(m);
}

static void test_field_writes(void) {
    NvmModule *m=ordinary();
    NvmFieldWrite fact={0,2,3,1,1,"module.nano","owner",".left",11,5,5},view={0};
    CHECK(nvm_add_field_write(m,&fact));CHECK(nvm_field_write_at(m,0,&view));
    CHECK(view.mode==1 && view.projection_length==5);
    NvmFieldWrite alternate=fact;alternate.mode=1;alternate.root="other";
    CHECK(!nvm_add_field_write(m,&alternate)); /* descriptive changes do not create another site */
    NvmModule *c=copy(m);compare_wire(m,c);CHECK(nvm_field_write_at(c,0,&view));
    char *text=disasm_module_styled(c,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult assembled;NvmModule *round=asm_assemble(text,&assembled);CHECK(round);compare_wire(m,round);
    free(text);nvm_module_free(round);nvm_module_free(c);
    char error[256];char *native=nvm2c_emit(m,error,sizeof error);CHECK(native);
    uint32_t value=m->metadata[0].value_idx,size=m->string_lengths[value];
    uint8_t *bytes=(uint8_t *)m->strings[value];uint8_t saved[128];CHECK(size<=sizeof saved);memcpy(saved,bytes,size);
    const unsigned offsets[]={0,4,8,12,13,14,15,16,20,24,28};
    for(size_t i=0;i<sizeof offsets/sizeof *offsets;i++) {
        memcpy(bytes,saved,size);
        if(offsets[i]<=8)word(bytes+offsets[i],offsets[i]?0:UINT32_MAX);
        else if(offsets[i]>=16 && offsets[i]<=24)word(bytes+offsets[i],UINT32_MAX);
        else bytes[offsets[i]]=offsets[i]==28?0xff:0xff;
        view.line=777;CHECK(!nvm_field_write_at(m,0,&view) && view.line==777);
        CHECK(nvm_verify(m).ok);char *again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);
    }
    memcpy(bytes,saved,size);
    for(uint32_t n=0;n<size;n++) {m->string_lengths[value]=n;view.line=777;CHECK(!nvm_field_write_at(m,0,&view) && view.line==777);}
    m->string_lengths[value]=size;
    /* A forged duplicate with a changed mode is still ambiguous. */
    saved[12]=1;uint32_t duplicate=nvm_add_string(m,(const char *)saved,size);
    CHECK(duplicate!=UINT32_MAX && nvm_add_metadata(m,m->metadata[0].key_idx,duplicate));
    view.line=777;CHECK(!nvm_field_write_at(m,0,&view) && view.line==777);
    CHECK(!nvm_field_write_at(m,1,&view) && view.line==777);
    CHECK(nvm_verify(m).ok);char *again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);
    m->metadata_count=0;again=nvm2c_emit(m,error,sizeof error);CHECK(again && !strcmp(native,again));free(again);free(native);
    nvm_module_free(m);
    m=ordinary();fact.projection="";fact.projection_length=0;CHECK(!nvm_add_field_write(m,&fact));
    fact.projection=".left";fact.projection_length=5;CHECK(nvm_add_field_write(m,&fact));
    uint32_t unknown=nvm_add_string(m,"nano.field-write.v9",19),junk=nvm_add_string(m,"opaque",6);
    CHECK(nvm_add_metadata(m,unknown,junk));CHECK(nvm_field_write_at(m,0,&view));nvm_module_free(m);
}

int main(int argc, char **argv) {
    test_purity();
    test_effects();
    test_affine();
    test_match_coverage(); test_affine_use(); test_borrow_places(); test_field_reads(); test_field_writes();
    test_generic_bindings();
    NvmModule *m = ordinary();
    uint32_t legacy_size; uint8_t *legacy = nvm_serialize(m, &legacy_size);
    CHECK(legacy && legacy_size); free(legacy);
    uint32_t k = nvm_add_string(m, "example.fact", 12);
    const char exact[] = {'a', '\0', '"', '\\', '\n', (char)0xff};
    uint32_t v = nvm_add_string(m, exact, sizeof exact);
    uint32_t empty = nvm_add_string(m, "", 0);
    uint32_t source = nvm_add_string(m, "nano.source_file", 16);
    uint32_t first = nvm_add_string(m, "first.nano", 10);
    uint32_t last = nvm_add_string(m, "last.nano", 9);
    CHECK(nvm_add_metadata(m, k, v)); CHECK(nvm_add_metadata(m, k, empty));
    CHECK(nvm_add_metadata(m, source, first)); CHECK(nvm_add_metadata(m, source, last));
    CHECK(m->source_file_idx == last && nvm_metadata_valid(m));
    CHECK(!nvm_add_metadata(m, UINT32_MAX, v));
    CHECK(!nvm_add_metadata(m, k, UINT32_MAX));
    legacy_size = 99; CHECK(!nvm_serialize(m, &legacy_size) && legacy_size == 0);
    NanoisaErr legacy_error;
    CHECK(!nanoisa_save_bytes_v1(m, &legacy_size, &legacy_error));
    CHECK(legacy_error.code == NANOISA_ERR_FORMAT && strstr(legacy_error.message, "v2"));
    CHECK(nvm_verify(m).ok);
    NvmModule *b = copy(m); compare_wire(m, b);
    CHECK(b->metadata_count == 4 && b->source_file_idx == last);
    CHECK(b->metadata[0].key_idx == k && b->metadata[1].key_idx == k);
    CHECK(b->string_lengths[v] == sizeof exact && memcmp(b->strings[v], exact, sizeof exact) == 0);
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult r; NvmModule *c = asm_assemble(text, &r); CHECK(c);
    compare_wire(b, c); CHECK(nvm_verify(c).ok); free(text);
    /* I free the source and decoded wire before using the independent module. */
    nvm_module_free(m); nvm_module_free(b);
    CHECK(c->metadata_count == 4 && c->strings[last][0] == 'l');
    c->source_file_idx = first;
    NvmV2Module bad; CHECK(nvm_v2_from_nvm_module(c, &bad) == NVM_V2_ERR_INDEX_RANGE);
    c->source_file_idx = last;
    nvm_strip_debug_info(c);
    CHECK(c->metadata_count == 2 && c->source_file_idx == 0);
    b = copy(c); compare_wire(c, b); nvm_module_free(b); nvm_module_free(c);

    /* I canonicalize a legacy source key once, without growing the pool again. */
    m = ordinary(); source = nvm_add_string(m, "nano.source_file", 16);
    m->source_file_idx = nvm_add_string(m, "legacy.nano", 11);
    b = copy(m); CHECK(b->metadata_count == 1 && b->metadata[0].key_idx == source);
    CHECK(b->string_count == m->string_count); c = copy(b); compare_wire(b, c);
    CHECK(c->string_count == b->string_count);
    nvm_module_free(m); nvm_module_free(b); nvm_module_free(c);

    m=ordinary(); m->source_file_idx=nvm_add_string(m,"new-key.nano",12);
    b=copy(m); CHECK(b->string_count==m->string_count+1 && b->metadata_count==1);
    c=copy(b); compare_wire(b,c); CHECK(c->string_count==b->string_count);
    nvm_module_free(m); nvm_module_free(b); nvm_module_free(c);

    /* I retain an explicit empty source value even at string index zero. */
    m = nvm_module_new(); CHECK(m);
    empty = nvm_add_string(m, "", 0); CHECK(empty == 0);
    source = nvm_add_string(m, "nano.source_file", 16);
    CHECK(nvm_add_metadata(m, source, empty));
    b = copy(m); CHECK(b->metadata_count == 1 && b->source_file_idx == 0);
    compare_wire(m, b); nvm_module_free(m); nvm_module_free(b);

    /* I carry advisory entries beside authoritative layouts/ownership unchanged. */
    uint8_t modes[2]={2,2};
    m=multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\n" TWO_SUM, WRITE_BOTH,2,modes);
    k=nvm_add_string(m,"example.advisory",16); v=nvm_add_string(m,"untrusted claim",15);
    CHECK(nvm_add_metadata(m,k,v)); b=copy(m); compare_wire(m,b);
    CHECK(nvm_verify(b).ok);
    CHECK(b->ownership_size==m->ownership_size && !memcmp(b->ownership_data,m->ownership_data,m->ownership_size));
    CHECK(b->layout_size==m->layout_size && !memcmp(b->layout_data,m->layout_data,m->layout_size));
    text=disasm_module_styled(b,DISASM_STYLE_CANONICAL); CHECK(text);
    c=asm_assemble(text,&r); CHECK(c); compare_wire(b,c); free(text);
    nvm_module_free(m); nvm_module_free(b); nvm_module_free(c);

    m=asm_assemble(".entry 0\n.function main 0 2 0 int 1\n"
        "PUSH_I64 17\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 25\nADD\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n",&r); CHECK(m);
    uint32_t fields[]={1,1,2,0,0,28,2,12,28,0,1,0,0,0,1,0,12,1,0,0,0,0};
    m->passive_size=sizeof fields; m->passive_data=malloc(sizeof fields); CHECK(m->passive_data);
    for(size_t i=0;i<sizeof fields/sizeof *fields;i++) word(m->passive_data+i*4,fields[i]);
    k=nvm_add_string(m,"example.purity",14); v=nvm_add_string(m,"unknown",7);
    CHECK(nvm_add_metadata(m,k,v)); CHECK(nvm_verify(m).ok); b=copy(m); compare_wire(m,b);
    CHECK(nvm_verify(b).ok && b->passive_size==m->passive_size && !memcmp(b->passive_data,m->passive_data,m->passive_size));
    nvm_module_free(m); nvm_module_free(b);

    /* I emit an ordinary artifact for actual CLI/native controls. */
    if(argc==2) {
        m=ordinary(); k=nvm_add_string(m,"example.fact",12); v=nvm_add_string(m,"advisory",8);
        CHECK(nvm_add_metadata(m,k,v));
        char path[4096],error[256]; size_t length; uint8_t *bytes=wire(m,&length);
        snprintf(path,sizeof path,"%s.nvm",argv[1]); FILE *f=fopen(path,"wb"); CHECK(f);
        CHECK(fwrite(bytes,1,length,f)==length); CHECK(!fclose(f)); free(bytes);
        char *native=nvm2c_emit(m,error,sizeof error); CHECK(native);
        snprintf(path,sizeof path,"%s.c",argv[1]); f=fopen(path,"w"); CHECK(f);
        CHECK(fwrite(native,1,strlen(native),f)==strlen(native)); CHECK(!fclose(f)); free(native);
        nvm_module_free(m);
    }

    const char *bad_text[] = {".metadata 0 0\n", ".string \"k\"\n.metadata -1 0\n",
        ".string \"k\"\n.metadata 0 1\n", ".string \"k\"\n.metadata 0 0 extra\n",
        ".string \"k\"\n.function f 0 0 0 void 0\n.metadata 0 0\nRET\n.end\n"};
    for (size_t i = 0; i < sizeof bad_text / sizeof *bad_text; ++i) {
        m = asm_assemble(bad_text[i], &r); CHECK(!m);
    }
    printf("I passed %u advisory metadata checks.\n", checks); return 0;
}
