/*
 * Unit tests for the v2 LAYOUTS section.
 *
 * The property this section exists to guarantee is that the table is acyclic:
 * every nested layout index points at a lower-numbered entry. Anything walking
 * the table trusts that, so a forward or self reference has to be rejected at
 * decode rather than discovered by recursing forever.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nvm_v2_sections.h"
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

/* Layout 0: struct { int; float }  -- scalars only, no nesting.
 * Layout 1: struct { layout-0; bool } -- nests layout 0, a lower index. */
#define FIELD_BYTES 12
#define ENTRY_BYTES 8

static size_t build(uint8_t *buf, size_t cap) {
    NvmV2LayoutField f0[2] = {
        { TAG_INT,   NVM_V2_NO_INDEX, 10 },
        { TAG_FLOAT, NVM_V2_NO_INDEX, 11 },
    };
    NvmV2LayoutField f1[2] = {
        { TAG_STRUCT, 0,               12 },   /* nests layout 0 */
        { TAG_BOOL,   NVM_V2_NO_INDEX, 13 },
    };
    NvmV2Layout items[2];
    items[0].kind = NVM_V2_LAYOUT_STRUCT; items[0].field_count = 2;
    items[0].name_idx = 1; items[0].fields = f0;
    items[1].kind = NVM_V2_LAYOUT_STRUCT; items[1].field_count = 2;
    items[1].name_idx = 2; items[1].fields = f1;
    NvmV2Layouts l = { items, 2 };
    size_t n = nvm_v2_layouts_encoded_size(&l);
    if (n == 0 || n > cap) return 0;
    nvm_v2_layouts_encode(&l, buf, n);
    return n;
}

/* Byte offset of layout 1's first field's nested_idx. */
#define L1_FIELD0_NESTED (4 + ENTRY_BYTES + 2*FIELD_BYTES + ENTRY_BYTES + 4)

static void test_round_trips_a_nested_layout(void) {
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    CHECK(n > 0, "fixture encodes");

    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_OK, "decodes");
    CHECK(got.count == 2, "two layouts");
    if (got.count == 2 && got.items && got.items[0].fields && got.items[1].fields) {
        CHECK(got.items[0].kind == NVM_V2_LAYOUT_STRUCT, "layout 0 is a struct");
        CHECK(got.items[0].field_count == 2, "layout 0 has two fields");
        CHECK(got.items[0].name_idx == 1, "layout 0 name index round-trips");
        CHECK(got.items[0].fields[0].type_tag == TAG_INT, "field 0 tag round-trips");
        CHECK(got.items[0].fields[0].nested_idx == NVM_V2_NO_INDEX,
              "a scalar field has no nested layout");
        CHECK(got.items[0].fields[1].name_idx == 11, "field name index round-trips");
        CHECK(got.items[1].fields[0].nested_idx == 0,
              "the nested field points at layout 0");
        CHECK(got.items[1].fields[1].type_tag == TAG_BOOL, "later field round-trips");
    } else {
        g_fail += 8;
        printf("  FAIL: decode did not produce two usable layouts\n");
    }
    nvm_v2_layouts_free(&got);
}

static void test_forward_nested_reference_is_rejected(void) {
    /* Layout 0 pointing at layout 1 would make the table cyclic-capable. */
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    /* layout 0's field 0 nested_idx sits at 4 + ENTRY_BYTES + 4 */
    size_t off = 4 + ENTRY_BYTES + 4;
    buf[off] = 1; buf[off+1] = 0; buf[off+2] = 0; buf[off+3] = 0;
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a nested index pointing forward is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_self_nested_reference_is_rejected(void) {
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    /* layout 1's field 0 already nests 0; point it at itself instead. */
    size_t off = L1_FIELD0_NESTED;
    buf[off] = 1; buf[off+1] = 0; buf[off+2] = 0; buf[off+3] = 0;
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_ERR_INDEX_RANGE,
                 "a layout nesting itself is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_invalid_kind_is_rejected(void) {
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    buf[4] = NVM_V2_LAYOUT_KIND_MAX + 1;
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_ERR_SECTION_TYPE,
                 "an unknown layout kind is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_invalid_field_tag_is_rejected(void) {
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    buf[4 + ENTRY_BYTES] = TAG_COUNT;   /* layout 0, field 0, type_tag */
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_ERR_SECTION_TYPE,
                 "a field type tag at or above TAG_COUNT is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_nonzero_entry_padding_is_rejected(void) {
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    buf[5] = 0x01;   /* layout 0's reserved byte */
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "nonzero reserved padding in a layout entry is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_nonzero_field_padding_is_rejected(void) {
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    buf[4 + ENTRY_BYTES + 1] = 0x01;   /* layout 0, field 0, _pad[0] */
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "nonzero reserved padding in a field is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_field_count_past_the_end_is_rejected(void) {
    uint8_t buf[256];
    size_t n = build(buf, sizeof buf);
    buf[6] = 0xFF; buf[7] = 0xFF;   /* layout 0's field_count */
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_ERR_TRUNCATED,
                 "a field count past the end is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_empty_section_decodes(void) {
    NvmV2Layouts l = { NULL, 0 };
    uint8_t buf[8];
    size_t n = nvm_v2_layouts_encoded_size(&l);
    CHECK(n == 4, "an empty table is just the count");
    nvm_v2_layouts_encode(&l, buf, sizeof buf);
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, n, &got), NVM_V2_OK, "empty decodes");
    CHECK(got.count == 0, "no layouts");
    nvm_v2_layouts_free(&got);
}

static void test_absurd_count_is_rejected_without_allocating(void) {
    uint8_t buf[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    NvmV2Layouts got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_layouts_decode(buf, sizeof buf, &got), NVM_V2_ERR_TRUNCATED,
                 "a count larger than the section could hold is rejected");
    nvm_v2_layouts_free(&got);
}

static void test_encode_into_a_short_buffer_is_rejected(void) {
    NvmV2LayoutField f[1] = { { TAG_INT, NVM_V2_NO_INDEX, 0 } };
    NvmV2Layout items[1];
    items[0].kind = NVM_V2_LAYOUT_TUPLE; items[0].field_count = 1;
    items[0].name_idx = NVM_V2_NO_INDEX; items[0].fields = f;
    NvmV2Layouts l = { items, 1 };
    uint8_t buf[8];
    CHECK_RESULT(nvm_v2_layouts_encode(&l, buf, sizeof buf), NVM_V2_ERR_TRUNCATED,
                 "encoding into too small a buffer is rejected");
}

int main(void) {
    printf("\n[nvm_v2_layouts] LAYOUTS section tests...\n\n");
    test_round_trips_a_nested_layout();
    test_forward_nested_reference_is_rejected();
    test_self_nested_reference_is_rejected();
    test_invalid_kind_is_rejected();
    test_invalid_field_tag_is_rejected();
    test_nonzero_entry_padding_is_rejected();
    test_nonzero_field_padding_is_rejected();
    test_field_count_past_the_end_is_rejected();
    test_empty_section_decodes();
    test_absurd_count_is_rejected_without_allocating();
    test_encode_into_a_short_buffer_is_rejected();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
