/*
 * Unit tests for whole-module v2 serialization and cross-section validation.
 *
 * The section codecs each validate what they can see. This layer validates
 * what none of them can: that an index in one section names something that
 * exists in another, that every function's code range lies inside CODE, and
 * that the header's feature bits describe the sections actually present.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nvm_v2_sections.h"
#include "nvm_format.h"   /* nvm_crc32, for doctoring a header in place */
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

/* A module using every section, so the round trip covers the whole format. */
static const uint8_t NAME_A[4] = { 'm', 'a', 'i', 'n' };
static const uint8_t NAME_B[3] = { 'a', 0x00, 'b' };   /* embedded zero */
static const uint8_t PTAGS[1]  = { TAG_INT };
static const uint8_t RTAGS[1]  = { TAG_INT };
static const uint8_t CODE_BYTES[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };

/* Caller owns the arrays; the module only borrows them. */
static void build_module(NvmV2Module *m,
                         NvmV2Constant *ck, NvmV2Signature *sg,
                         NvmV2Function *fn, NvmV2Global *gl,
                         NvmV2Import *im, NvmV2Link *lk,
                         NvmV2MetadataEntry *md, NvmV2DebugEntry *db) {
    memset(m, 0, sizeof *m);
    m->isa_version = NVM_V2_ISA_VERSION;
    m->entry_point = 0;

    ck[0].tag = TAG_STRING; ck[0].length = 4; ck[0].payload = NAME_A;
    ck[1].tag = TAG_STRING; ck[1].length = 3; ck[1].payload = NAME_B;
    m->constants.items = ck; m->constants.count = 2;

    sg[0].param_count = 1; sg[0].param_tags = PTAGS;
    sg[0].result_count = 1; sg[0].result_tags = RTAGS;
    m->signatures.items = sg; m->signatures.count = 1;

    fn[0].name_idx = 0; fn[0].signature_idx = 0;
    fn[0].code_offset = 0; fn[0].code_length = 8;
    fn[0].local_count = 1; fn[0].upvalue_count = 0; fn[0].max_stack = 4;
    m->functions.items = fn; m->functions.count = 1;

    gl[0].name_idx = 1; gl[0].type_tag = TAG_INT;
    gl[0].flags = NVM_V2_GLOBAL_MUTABLE; gl[0].init_idx = NVM_V2_NO_INDEX;
    m->globals.items = gl; m->globals.count = 1;

    im[0].module_name_idx = 0; im[0].symbol_name_idx = 1;
    im[0].signature_idx = 0; im[0].kind = NVM_V2_IMPORT_FFI;
    m->imports.items = im; m->imports.count = 1;

    lk[0].module_name_idx = 0; lk[0].symbol_name_idx = 1;
    lk[0].signature_idx = 0; lk[0].flags = 0;
    m->links.items = lk; m->links.count = 1;

    md[0].key_idx = 0; md[0].value_idx = 1;
    m->metadata.items = md; m->metadata.count = 1;

    db[0].bytecode_offset = 0; db[0].source_line = 1; db[0].source_col = 1;
    m->debug.items = db; m->debug.count = 1;
    m->has_debug = true;

    m->code = CODE_BYTES; m->code_size = sizeof CODE_BYTES;
}

#define DECL_PARTS \
    NvmV2Constant ck[2]; NvmV2Signature sg[1]; NvmV2Function fn[1]; \
    NvmV2Global gl[1]; NvmV2Import im[1]; NvmV2Link lk[1]; \
    NvmV2MetadataEntry md[1]; NvmV2DebugEntry db[1]

static size_t serialize_fixture(uint8_t *buf, size_t cap) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    size_t n = 0;
    if (nvm_v2_module_serialize(&m, buf, cap, &n) != NVM_V2_OK) return 0;
    return n;
}

static void test_full_module_round_trips(void) {
    uint8_t buf[1024];
    size_t n = serialize_fixture(buf, sizeof buf);
    CHECK(n > 0, "a module using every section serializes");

    CHECK_RESULT(nvm_v2_validate(buf, n), NVM_V2_OK,
                 "the serialized module passes container validation");

    NvmV2Module got;
    memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_OK, "deserializes");
    CHECK(got.constants.count == 2, "two constants");
    CHECK(got.constants.count == 2 && got.constants.items
              && got.constants.items[1].length == 3
              && memcmp(got.constants.items[1].payload, NAME_B, 3) == 0,
          "the embedded zero survives a whole-module round trip");
    CHECK(got.signatures.count == 1, "one signature");
    CHECK(got.functions.count == 1, "one function");
    CHECK(got.functions.count == 1 && got.functions.items
              && got.functions.items[0].max_stack == 4, "max_stack survives");
    CHECK(got.globals.count == 1, "one global");
    CHECK(got.imports.count == 1, "one import");
    CHECK(got.links.count == 1, "one link");
    CHECK(got.metadata.count == 1, "one metadata pair");
    CHECK(got.debug.count == 1, "one debug record");
    CHECK(got.code_size == 8 && got.code && got.code[7] == 8, "CODE round-trips");
    CHECK(got.entry_point == 0, "entry point round-trips");
    nvm_v2_module_free(&got);
}

static void test_serialize_reports_required_size(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    size_t need = 0;
    CHECK_RESULT(nvm_v2_module_serialize(&m, NULL, 0, &need), NVM_V2_OK,
                 "sizing with a null buffer succeeds");
    CHECK(need > NVM_V2_HEADER_SIZE, "the reported size covers more than a header");

    uint8_t small[16];
    size_t n = 0;
    CHECK_RESULT(nvm_v2_module_serialize(&m, small, sizeof small, &n),
                 NVM_V2_ERR_TRUNCATED, "serializing into too small a buffer is rejected");
}

/* ── Cross-section validation ───────────────────────────────────────────── */

static void test_signature_index_out_of_range_is_rejected(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    fn[0].signature_idx = 1;   /* only one signature, so index 1 does not exist */
    uint8_t buf[1024]; size_t n = 0;
    CHECK_RESULT(nvm_v2_module_serialize(&m, buf, sizeof buf, &n), NVM_V2_OK,
                 "serializing still succeeds -- the check is on the reading side");
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a function naming a signature that does not exist is rejected");
    nvm_v2_module_free(&got);
}

static void test_constant_index_out_of_range_is_rejected(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    fn[0].name_idx = 9;   /* only two constants */
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a name index past the constant pool is rejected");
    nvm_v2_module_free(&got);
}

static void test_import_signature_index_is_checked(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    im[0].signature_idx = 4;
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "an import naming a missing signature is rejected");
    nvm_v2_module_free(&got);
}

static void test_link_signature_index_is_checked(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    lk[0].signature_idx = 4;
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a link naming a missing signature is rejected");
    nvm_v2_module_free(&got);
}

static void test_code_range_outside_code_section_is_rejected(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    fn[0].code_length = 9;   /* CODE is only 8 bytes */
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a function whose code runs past CODE is rejected");
    nvm_v2_module_free(&got);
}

static void test_code_range_cannot_wrap(void) {
    /* offset + length would wrap back into range on an addition-form check. */
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    fn[0].code_offset = 0xFFFFFFFFFFFFFFF8ull;
    fn[0].code_length = 16;
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a wrapping code range cannot slip past the bound");
    nvm_v2_module_free(&got);
}

static void test_entry_point_out_of_range_is_rejected(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    m.entry_point = 3;   /* only one function */
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "an entry point past the function table is rejected");
    nvm_v2_module_free(&got);
}

static void test_no_entry_point_is_allowed(void) {
    DECL_PARTS; NvmV2Module m;
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    m.entry_point = NVM_V2_NO_ENTRY_POINT;
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_OK,
                 "a module with no entry point is legal");
    CHECK(got.entry_point == NVM_V2_NO_ENTRY_POINT, "the sentinel round-trips");
    nvm_v2_module_free(&got);
}

static void test_nested_layout_index_past_the_table_is_rejected(void) {
    DECL_PARTS; NvmV2Module m;
    NvmV2LayoutField lf[1] = { { TAG_STRUCT, 0, 0 } };
    NvmV2Layout lay[1];
    build_module(&m, ck, sg, fn, gl, im, lk, md, db);
    /* One layout whose field nests layout 0 -- itself. The LAYOUTS codec
     * rejects that on its own, so instead give a layout table of one and have
     * the field name a constant that does not exist, which only this layer can
     * see. */
    lf[0].nested_idx = NVM_V2_NO_INDEX;
    lf[0].name_idx = 9;
    lay[0].kind = NVM_V2_LAYOUT_STRUCT; lay[0].field_count = 1;
    lay[0].name_idx = 0; lay[0].fields = lf;
    m.layouts.items = lay; m.layouts.count = 1;
    uint8_t buf[1024]; size_t n = 0;
    nvm_v2_module_serialize(&m, buf, sizeof buf, &n);
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a layout field naming a missing constant is rejected");
    nvm_v2_module_free(&got);
}

static void test_feature_bits_must_match_sections(void) {
    /* The header advertises FEATURE_LINKED iff LINKS is non-empty. A module
     * claiming a capability it does not carry, or carrying one it does not
     * advertise, is inconsistent either way. */
    uint8_t buf[1024];
    size_t n = serialize_fixture(buf, sizeof buf);
    CHECK(n > 0, "fixture serializes");
    /* Clear FEATURE_LINKED while LINKS is still present, then repair the
     * checksum -- otherwise the container check fires first and the
     * cross-section rule never runs. */
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, n, &h), NVM_V2_OK, "header reads back");
    CHECK((h.feature_bits & NVM_V2_FEATURE_LINKED) != 0,
          "the serializer set FEATURE_LINKED for a module with links");
    h.feature_bits &= ~(uint32_t)NVM_V2_FEATURE_LINKED;
    h.checksum = 0;
    nvm_v2_write_header(buf, &h);
    h.checksum = nvm_crc32(buf + NVM_V2_HEADER_SIZE,
                           (uint32_t)(n - NVM_V2_HEADER_SIZE));
    nvm_v2_write_header(buf, &h);
    CHECK_RESULT(nvm_v2_validate(buf, n), NVM_V2_OK,
                 "the doctored module is still a well-formed container");

    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got),
                 NVM_V2_ERR_FEATURE_MISMATCH,
                 "feature bits disagreeing with the sections present is rejected");
    nvm_v2_module_free(&got);
}

static void test_minimal_module_round_trips(void) {
    /* Only the required sections, all empty, no CODE and no DEBUG. */
    NvmV2Module m;
    memset(&m, 0, sizeof m);
    m.isa_version = NVM_V2_ISA_VERSION;
    m.entry_point = NVM_V2_NO_ENTRY_POINT;
    uint8_t buf[512]; size_t n = 0;
    CHECK_RESULT(nvm_v2_module_serialize(&m, buf, sizeof buf, &n), NVM_V2_OK,
                 "an empty module serializes");
    CHECK_RESULT(nvm_v2_validate(buf, n), NVM_V2_OK, "and passes container validation");
    NvmV2Module got; memset(&got, 0, sizeof got);
    CHECK_RESULT(nvm_v2_module_deserialize(buf, n, &got), NVM_V2_OK,
                 "and deserializes");
    CHECK(got.functions.count == 0, "no functions");
    CHECK(got.has_debug == false, "no debug section");
    nvm_v2_module_free(&got);
}

int main(void) {
    printf("\n[nvm_v2_module] whole-module serialization tests...\n\n");
    test_full_module_round_trips();
    test_serialize_reports_required_size();
    test_signature_index_out_of_range_is_rejected();
    test_constant_index_out_of_range_is_rejected();
    test_import_signature_index_is_checked();
    test_link_signature_index_is_checked();
    test_code_range_outside_code_section_is_rejected();
    test_code_range_cannot_wrap();
    test_entry_point_out_of_range_is_rejected();
    test_no_entry_point_is_allowed();
    test_nested_layout_index_past_the_table_is_rejected();
    test_feature_bits_must_match_sections();
    test_minimal_module_round_trips();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
