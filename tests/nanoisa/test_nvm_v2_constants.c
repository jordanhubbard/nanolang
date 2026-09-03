/*
 * Unit tests for the v2 CONSTANTS section.
 *
 * The behaviour that matters most is that a string's bytes survive verbatim,
 * including an embedded zero -- that is the whole reason the entry carries an
 * explicit length rather than relying on termination.
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

/* Three bytes with a zero in the middle: the case a strlen-based pool
 * silently truncates to one byte. */
static const uint8_t EMBEDDED[3] = { 'a', 0x00, 'b' };
static const uint8_t SEVEN[1]    = { 7 };

/* Two entries whose payload lengths are not multiples of four, so the padding
 * path is exercised rather than skipped. */
static size_t build(uint8_t *buf, size_t cap) {
    NvmV2Constant items[2];
    items[0].tag = TAG_STRING; items[0].length = 3; items[0].payload = EMBEDDED;
    items[1].tag = TAG_INT;    items[1].length = 1; items[1].payload = SEVEN;
    NvmV2Constants c = { items, 2 };
    size_t n = nvm_v2_constants_encoded_size(&c);
    if (n > cap) return 0;
    nvm_v2_constants_encode(&c, buf, n);
    return n;
}

static void test_round_trips_an_embedded_zero(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    CHECK(n > 0, "fixture encodes");

    NvmV2Constants got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_constants_decode(buf, n, &got), NVM_V2_OK, "decodes");
    CHECK(got.count == 2, "two constants");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].tag == TAG_STRING, "first entry is a string");
        CHECK(got.items[0].length == 3, "length is 3, not strlen's 1");
        CHECK(got.items[0].payload && memcmp(got.items[0].payload, EMBEDDED, 3) == 0,
              "all three bytes survive, zero included");
        CHECK(got.items[1].tag == TAG_INT, "second entry is an int");
        CHECK(got.items[1].length == 1 && got.items[1].payload
                  && got.items[1].payload[0] == 7,
              "second payload round-trips");
    } else {
        g_fail += 5;
        printf("  FAIL: decode did not produce two usable entries\n");
    }
    nvm_v2_constants_free(&got);
}

static void test_encoded_size_matches_what_encode_writes(void) {
    /* If these disagree the whole-module serializer lays sections out wrongly,
     * and the directory offsets stop matching the payloads. */
    NvmV2Constant items[1];
    items[0].tag = TAG_STRING; items[0].length = 3; items[0].payload = EMBEDDED;
    NvmV2Constants c = { items, 1 };
    size_t n = nvm_v2_constants_encoded_size(&c);
    CHECK(n == 4 + 8 + 4, "one 3-byte entry is count + header + padded payload");

    uint8_t buf[64];
    memset(buf, 0xCD, sizeof buf);
    CHECK_RESULT(nvm_v2_constants_encode(&c, buf, sizeof buf), NVM_V2_OK, "encodes");
    CHECK(buf[n] == 0xCD, "encode wrote exactly encoded_size bytes and no more");
}

static void test_padding_is_written_as_zero(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    (void)n;
    /* entry 0: 4 count + 4 (tag+pad) + 4 length + 3 payload, so byte 15 is pad */
    CHECK(buf[15] == 0, "trailing payload padding is zero");
}

static void test_empty_section_decodes(void) {
    NvmV2Constants c = { NULL, 0 };
    uint8_t buf[8];
    size_t n = nvm_v2_constants_encoded_size(&c);
    CHECK(n == 4, "an empty pool is just the count");
    nvm_v2_constants_encode(&c, buf, sizeof buf);
    NvmV2Constants got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_constants_decode(buf, n, &got), NVM_V2_OK, "empty decodes");
    CHECK(got.count == 0, "no constants");
    nvm_v2_constants_free(&got);
}

static void test_truncated_count_is_rejected(void) {
    uint8_t buf[2] = { 0, 0 };
    NvmV2Constants got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_constants_decode(buf, 2, &got), NVM_V2_ERR_TRUNCATED,
                 "a section too short to hold the count is rejected");
}

static void test_payload_length_past_the_end_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    buf[8] = 0xFF;   /* low byte of entry 0's length */
    NvmV2Constants got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_constants_decode(buf, n, &got), NVM_V2_ERR_TRUNCATED,
                 "a payload length past the end is rejected");
}

static void test_invalid_value_tag_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    buf[4] = TAG_COUNT;   /* entry 0's tag */
    NvmV2Constants got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_constants_decode(buf, n, &got), NVM_V2_ERR_SECTION_TYPE,
                 "a value tag at or above TAG_COUNT is rejected");
}

static void test_nonzero_entry_padding_is_rejected(void) {
    uint8_t buf[64];
    size_t n = build(buf, sizeof buf);
    buf[5] = 0x01;   /* entry 0's _pad[0] */
    NvmV2Constants got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_constants_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "nonzero reserved padding in an entry is rejected");
}

static void test_absurd_count_is_rejected_without_allocating(void) {
    /* A four-byte section claiming four billion entries must be refused on the
     * arithmetic, not after trying to allocate for them. */
    uint8_t buf[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    NvmV2Constants got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_constants_decode(buf, sizeof buf, &got), NVM_V2_ERR_TRUNCATED,
                 "a count larger than the section could hold is rejected");
}

static void test_encode_into_a_short_buffer_is_rejected(void) {
    NvmV2Constant items[1];
    items[0].tag = TAG_STRING; items[0].length = 3; items[0].payload = EMBEDDED;
    NvmV2Constants c = { items, 1 };
    uint8_t buf[8];
    CHECK_RESULT(nvm_v2_constants_encode(&c, buf, sizeof buf), NVM_V2_ERR_TRUNCATED,
                 "encoding into too small a buffer is rejected");
}

int main(void) {
    printf("\n[nvm_v2_constants] CONSTANTS section tests...\n\n");
    test_round_trips_an_embedded_zero();
    test_encoded_size_matches_what_encode_writes();
    test_padding_is_written_as_zero();
    test_empty_section_decodes();
    test_truncated_count_is_rejected();
    test_payload_length_past_the_end_is_rejected();
    test_invalid_value_tag_is_rejected();
    test_nonzero_entry_padding_is_rejected();
    test_absurd_count_is_rejected_without_allocating();
    test_encode_into_a_short_buffer_is_rejected();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
