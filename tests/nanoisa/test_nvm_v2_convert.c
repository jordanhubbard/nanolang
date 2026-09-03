/*
 * Unit tests for the NvmModule <-> v2 bridge.
 *
 * The bridge is what lets v2 be adopted without rewriting every producer at
 * once, so what matters is fidelity: a v1 module converted to v2, serialized,
 * read back, and converted to v1 again must be the same module. The two places
 * that can silently lose information are the string pool -- v1 strings carry
 * explicit lengths and may hold embedded zero bytes -- and the signature table,
 * where deduplication has to be exact or signature-index comparison stops being
 * a valid equality test.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nvm_v2_sections.h"
#include "nvm_format.h"
#include "isa.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, what) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s  (%s:%d)\n", (what), __FILE__, __LINE__); } \
} while (0)

#define CHECK_RESULT(actual, expected, what) do { \
    NvmV2Result _a = (actual), _e = (expected); \
    if (_a == _e) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s -- expected %s, got %s  (%s:%d)\n", \
        (what), nvm_v2_result_name(_e), nvm_v2_result_name(_a), __FILE__, __LINE__); } \
} while (0)

/* A v1 module with two identically-shaped functions, one differently-shaped,
 * an import, a module ref, a debug entry, and a string holding a zero byte. */
static NvmModule *build_v1(void) {
    NvmModule *m = nvm_module_new();
    if (!m) return NULL;

    uint32_t s_add  = nvm_add_string(m, "add", 3);
    uint32_t s_sub  = nvm_add_string(m, "sub", 3);
    uint32_t s_main = nvm_add_string(m, "main", 4);
    uint32_t s_bin  = nvm_add_string(m, "a\0b", 3);   /* embedded zero */
    uint32_t s_libc = nvm_add_string(m, "libc", 4);
    uint32_t s_puts = nvm_add_string(m, "puts", 4);
    (void)s_bin;

    static const uint8_t code[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    nvm_append_code(m, code, sizeof code);

    /* add and sub have the same shape; main differs. */
    NvmFunctionEntry f;
    memset(&f, 0, sizeof f);
    f.name_idx = s_add; f.arity = 2; f.code_offset = 0; f.code_length = 4;
    f.local_count = 2; f.upvalue_count = 0;
    f.result_tag = TAG_INT; f.result_count = 1;
    nvm_add_function(m, &f);

    f.name_idx = s_sub; f.code_offset = 4; f.code_length = 4;
    nvm_add_function(m, &f);

    memset(&f, 0, sizeof f);
    f.name_idx = s_main; f.arity = 0; f.code_offset = 8; f.code_length = 4;
    f.local_count = 1; f.upvalue_count = 0;
    f.result_tag = TAG_VOID; f.result_count = 0;
    nvm_add_function(m, &f);

    const uint8_t ptypes[1] = { TAG_STRING };
    nvm_add_import(m, s_libc, s_puts, 1, TAG_INT, ptypes);

    nvm_add_module_ref(m, s_libc);
    nvm_add_debug_entry(m, 0, 17, 3);
    m->header.entry_point = 2;   /* main */
    return m;
}

static void test_round_trip_through_v2(void) {
    NvmModule *v1 = build_v1();
    CHECK(v1 != NULL, "the v1 fixture builds");
    if (!v1) return;

    NvmV2Module v2;
    CHECK_RESULT(nvm_v2_from_nvm_module(v1, &v2), NVM_V2_OK, "converts to v2");

    size_t need = 0;
    CHECK_RESULT(nvm_v2_module_serialize(&v2, NULL, 0, &need), NVM_V2_OK, "sizes");
    uint8_t *buf = malloc(need);
    CHECK(buf != NULL, "allocates");
    if (!buf) { nvm_v2_module_free(&v2); nvm_module_free(v1); return; }

    size_t n = 0;
    CHECK_RESULT(nvm_v2_module_serialize(&v2, buf, need, &n), NVM_V2_OK, "serializes");

    /* Everything the bridge produced must satisfy the same cross-section rules
     * a hand-built module does -- otherwise the bridge is a way to smuggle an
     * invalid module past the validator. */
    NvmV2Module back;
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &back), NVM_V2_OK,
                 "the converted module passes cross-section validation");

    NvmModule *v1b = NULL;
    CHECK_RESULT(nvm_v2_to_nvm_module(&back, &v1b), NVM_V2_OK, "converts back to v1");
    CHECK(v1b != NULL, "produces a v1 module");

    if (v1b) {
        CHECK(v1b->function_count == 3, "three functions survive");
        CHECK(v1b->import_count == 1, "one import survives");
        CHECK(v1b->module_ref_count == 1, "one module ref survives");
        CHECK(v1b->code_size == 12, "the code section survives");
        CHECK(v1b->debug_count == 1, "the debug entry survives");
        CHECK(v1b->header.entry_point == 2, "the entry point survives");

        if (v1b->function_count == 3) {
            CHECK(v1b->functions[0].arity == 2, "arity survives");
            CHECK(v1b->functions[0].result_tag == TAG_INT, "result tag survives");
            CHECK(v1b->functions[0].result_count == 1, "result count survives");
            CHECK(v1b->functions[2].result_count == 0, "a void function stays void");
            CHECK(v1b->functions[1].code_offset == 4, "code offsets survive");
            CHECK(v1b->functions[0].local_count == 2, "local count survives");
        } else { g_fail += 6; printf("  FAIL: function table did not survive\n"); }

        /* The embedded zero is the whole reason the string pool carries
         * explicit lengths; strlen would have truncated this to one byte. */
        bool found_binary = false;
        for (uint32_t i = 0; i < v1b->string_count; i++) {
            if (v1b->string_lengths[i] == 3 &&
                memcmp(v1b->strings[i], "a\0b", 3) == 0) { found_binary = true; break; }
        }
        CHECK(found_binary, "a string with an embedded zero survives the round trip");

        if (v1b->import_count == 1) {
            CHECK(v1b->imports[0].param_count == 1, "import arity survives");
            CHECK(v1b->imports[0].return_type == TAG_INT, "import return type survives");
            CHECK(v1b->import_param_types && v1b->import_param_types[0] &&
                  v1b->import_param_types[0][0] == TAG_STRING,
                  "import parameter tags survive");
        } else { g_fail += 3; printf("  FAIL: import did not survive\n"); }

        nvm_module_free(v1b);
    }

    nvm_v2_module_free(&back);
    free(buf);
    nvm_v2_module_free(&v2);
    nvm_module_free(v1);
}

static void test_identical_shapes_share_a_signature(void) {
    NvmModule *v1 = build_v1();
    if (!v1) { g_fail++; printf("  FAIL: fixture\n"); return; }

    NvmV2Module v2;
    CHECK_RESULT(nvm_v2_from_nvm_module(v1, &v2), NVM_V2_OK, "converts");

    CHECK(v2.functions.count == 3, "three functions");
    if (v2.functions.count == 3) {
        CHECK(v2.functions.items[0].signature_idx == v2.functions.items[1].signature_idx,
              "two identically-shaped functions share one signature entry");
        CHECK(v2.functions.items[0].signature_idx != v2.functions.items[2].signature_idx,
              "a differently-shaped function gets its own");
    } else { g_fail += 2; printf("  FAIL: functions missing\n"); }

    /* add/sub, main, and the import's (string)->int shape: three distinct
     * signatures, not five. If dedup were incomplete the count would rise and
     * comparing signature indices would stop meaning "same type". */
    CHECK(v2.signatures.count == 3, "exactly three distinct signatures are emitted");

    nvm_v2_module_free(&v2);
    nvm_module_free(v1);
}

static void test_empty_module_converts(void) {
    NvmModule *v1 = nvm_module_new();
    if (!v1) { g_fail++; printf("  FAIL: fixture\n"); return; }
    NvmV2Module v2;
    CHECK_RESULT(nvm_v2_from_nvm_module(v1, &v2), NVM_V2_OK, "an empty module converts");
    CHECK(v2.functions.count == 0, "no functions");
    CHECK(v2.entry_point == NVM_V2_NO_ENTRY_POINT,
          "an empty module has no entry point rather than function 0");

    uint8_t buf[512]; size_t n = 0;
    CHECK_RESULT(nvm_v2_module_serialize(&v2, buf, sizeof buf, &n), NVM_V2_OK,
                 "and serializes");
    NvmV2Module back;
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &back), NVM_V2_OK,
                 "and reads back");
    nvm_v2_module_free(&back);
    nvm_v2_module_free(&v2);
    nvm_module_free(v1);
}

static void test_max_stack_is_zero_until_the_verifier_fills_it(void) {
    /* A v1 module does not record max_stack: nothing computes it at this
     * layer. The bridge emits 0 rather than a guess, and Task 13 populates it
     * from the verifier. Asserting it here keeps that gap visible. */
    NvmModule *v1 = build_v1();
    if (!v1) { g_fail++; printf("  FAIL: fixture\n"); return; }
    NvmV2Module v2;
    nvm_v2_from_nvm_module(v1, &v2);
    bool all_zero = true;
    for (uint32_t i = 0; i < v2.functions.count; i++)
        if (v2.functions.items[i].max_stack != 0) all_zero = false;
    CHECK(all_zero, "max_stack is 0 from a v1 source, not a guess");
    nvm_v2_module_free(&v2);
    nvm_module_free(v1);
}

int main(void) {
    printf("\n[nvm_v2_convert] NvmModule <-> v2 bridge tests...\n\n");
    test_round_trip_through_v2();
    test_identical_shapes_share_a_signature();
    test_empty_module_converts();
    test_max_stack_is_zero_until_the_verifier_fills_it();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
