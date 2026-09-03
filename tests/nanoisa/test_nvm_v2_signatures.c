/*
 * Unit tests for the v2 SIGNATURES section.
 *
 * Signature indices are compared for equality all over verification, so the
 * two properties that matter most here are that a shape round-trips exactly
 * and that nvm_v2_signature_equal distinguishes shapes that differ only
 * slightly -- a comparison that returns true too easily would make every
 * signature check vacuous.
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

/* (int, float) -> int  and  () -> void : one entry with both arrays non-empty
 * and lengths that are not multiples of four, one entry with both empty. */
static const uint8_t P0[2] = { TAG_INT, TAG_FLOAT };
static const uint8_t R0[1] = { TAG_INT };

static size_t build(uint8_t *buf, size_t cap) {
    NvmV2Signature items[2];
    items[0].param_count = 2; items[0].param_tags = P0;
    items[0].result_count = 1; items[0].result_tags = R0;
    items[1].param_count = 0; items[1].param_tags = NULL;
    items[1].result_count = 0; items[1].result_tags = NULL;
    NvmV2Signatures s = { items, 2 };
    size_t n = nvm_v2_signatures_encoded_size(&s);
    if (n == 0 || n > cap) return 0;
    nvm_v2_signatures_encode(&s, buf, n);
    return n;
}

static void test_round_trips_a_mixed_signature(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    CHECK(n > 0, "fixture encodes");

    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, n, &got), NVM_V2_OK, "decodes");
    CHECK(got.count == 2, "two signatures");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].param_count == 2, "first takes two parameters");
        CHECK(got.items[0].result_count == 1, "first returns one result");
        CHECK(got.items[0].param_tags
                  && got.items[0].param_tags[0] == TAG_INT
                  && got.items[0].param_tags[1] == TAG_FLOAT,
              "parameter tags round-trip in order");
        CHECK(got.items[0].result_tags && got.items[0].result_tags[0] == TAG_INT,
              "result tag round-trips");
        CHECK(got.items[1].param_count == 0 && got.items[1].result_count == 0,
              "the empty signature round-trips");
    } else {
        g_fail += 5;
        printf("  FAIL: decode did not produce two usable signatures\n");
    }
    nvm_v2_signatures_free(&got);
}

static void test_encoded_size_matches_what_encode_writes(void) {
    NvmV2Signature items[1];
    items[0].param_count = 2; items[0].param_tags = P0;
    items[0].result_count = 1; items[0].result_tags = R0;
    NvmV2Signatures s = { items, 1 };
    size_t n = nvm_v2_signatures_encoded_size(&s);
    /* 4 count + 4 (two u16) + pad4(2) + pad4(1) = 4 + 4 + 4 + 4 */
    CHECK(n == 16, "sizing accounts for both padded tag arrays");

    uint8_t buf[64];
    memset(buf, 0xCD, sizeof buf);
    CHECK_RESULT(nvm_v2_signatures_encode(&s, buf, sizeof buf), NVM_V2_OK, "encodes");
    CHECK(buf[n] == 0xCD, "encode wrote exactly encoded_size bytes and no more");
}

static void test_equal_distinguishes_shapes(void) {
    /* If this returns true too easily, every signature-index comparison in
     * verification silently passes. */
    static const uint8_t p_if[2] = { TAG_INT, TAG_FLOAT };
    static const uint8_t p_fi[2] = { TAG_FLOAT, TAG_INT };
    static const uint8_t r_i[1]  = { TAG_INT };
    static const uint8_t r_f[1]  = { TAG_FLOAT };

    NvmV2Signature a = { 2, 1, p_if, r_i };
    NvmV2Signature same = { 2, 1, p_if, r_i };
    NvmV2Signature reordered = { 2, 1, p_fi, r_i };   /* same tags, other order */
    NvmV2Signature other_result = { 2, 1, p_if, r_f };
    NvmV2Signature fewer_params = { 1, 1, p_if, r_i };

    CHECK(nvm_v2_signature_equal(&a, &same), "identical shapes are equal");
    CHECK(!nvm_v2_signature_equal(&a, &reordered),
          "parameter order is part of the shape");
    CHECK(!nvm_v2_signature_equal(&a, &other_result),
          "result tag is part of the shape");
    CHECK(!nvm_v2_signature_equal(&a, &fewer_params),
          "parameter count is part of the shape");
}

static void test_empty_section_decodes(void) {
    NvmV2Signatures s = { NULL, 0 };
    uint8_t buf[8];
    size_t n = nvm_v2_signatures_encoded_size(&s);
    CHECK(n == 4, "an empty table is just the count");
    nvm_v2_signatures_encode(&s, buf, sizeof buf);
    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, n, &got), NVM_V2_OK, "empty decodes");
    CHECK(got.count == 0, "no signatures");
    nvm_v2_signatures_free(&got);
}

static void test_truncated_count_is_rejected(void) {
    uint8_t buf[2] = { 0, 0 };
    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, 2, &got), NVM_V2_ERR_TRUNCATED,
                 "a section too short to hold the count is rejected");
}

static void test_param_count_past_the_end_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    buf[4] = 0xFF; buf[5] = 0xFF;   /* entry 0's param_count */
    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, n, &got), NVM_V2_ERR_TRUNCATED,
                 "a parameter count past the end is rejected");
}

static void test_invalid_param_tag_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    buf[8] = TAG_COUNT;   /* entry 0's first param tag */
    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, n, &got), NVM_V2_ERR_SECTION_TYPE,
                 "a parameter tag at or above TAG_COUNT is rejected");
}

static void test_invalid_result_tag_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    /* 4 count + 4 counts + pad4(2 param tags) = 12, so result tags start at 12 */
    buf[12] = TAG_COUNT;
    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, n, &got), NVM_V2_ERR_SECTION_TYPE,
                 "a result tag at or above TAG_COUNT is rejected");
}

static void test_nonzero_tag_padding_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    /* param tags occupy 8..9, so 10 and 11 are padding */
    buf[10] = 0x01;
    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, n, &got), NVM_V2_ERR_SECTION_RANGE,
                 "nonzero padding after the tag array is rejected");
}

static void test_absurd_count_is_rejected_without_allocating(void) {
    uint8_t buf[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    NvmV2Signatures got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_signatures_decode(buf, sizeof buf, &got), NVM_V2_ERR_TRUNCATED,
                 "a count larger than the section could hold is rejected");
}

static void test_encode_into_a_short_buffer_is_rejected(void) {
    NvmV2Signature items[1] = { { 2, 1, P0, R0 } };
    NvmV2Signatures s = { items, 1 };
    uint8_t buf[8];
    CHECK_RESULT(nvm_v2_signatures_encode(&s, buf, sizeof buf), NVM_V2_ERR_TRUNCATED,
                 "encoding into too small a buffer is rejected");
}

int main(void) {
    printf("\n[nvm_v2_signatures] SIGNATURES section tests...\n\n");
    test_round_trips_a_mixed_signature();
    test_encoded_size_matches_what_encode_writes();
    test_equal_distinguishes_shapes();
    test_empty_section_decodes();
    test_truncated_count_is_rejected();
    test_param_count_past_the_end_is_rejected();
    test_invalid_param_tag_is_rejected();
    test_invalid_result_tag_is_rejected();
    test_nonzero_tag_padding_is_rejected();
    test_absurd_count_is_rejected_without_allocating();
    test_encode_into_a_short_buffer_is_rejected();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
