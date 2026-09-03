/*
 * Unit tests for the v2 section-decoding cursor.
 *
 * The cursor exists so that eight section decoders do not each write their own
 * bounds check, so the tests here are mostly about what it refuses.
 */

#include <stdio.h>
#include <string.h>
#include "nvm_v2_sections.h"

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

static void test_reads_are_little_endian_and_sequential(void) {
    uint8_t buf[8] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);

    uint8_t a; uint16_t b; uint32_t d;
    CHECK_RESULT(nvm_v2_u8(&c, &a), NVM_V2_OK, "u8 reads");
    CHECK(a == 0x01, "u8 returns the first byte");
    CHECK_RESULT(nvm_v2_u16(&c, &b), NVM_V2_OK, "u16 reads");
    CHECK(b == 0x0302, "u16 is little-endian");
    CHECK_RESULT(nvm_v2_u32(&c, &d), NVM_V2_OK, "u32 reads");
    CHECK(d == 0x07060504, "u32 is little-endian");
    CHECK(nvm_v2_cursor_exhausted(&c) == false, "one byte still remains");
    CHECK_RESULT(nvm_v2_u8(&c, &a), NVM_V2_OK, "the last byte reads");
    CHECK(nvm_v2_cursor_exhausted(&c) == true, "now exhausted");
}

static void test_u64_is_little_endian(void) {
    uint8_t buf[8] = { 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80 };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    uint64_t v;
    CHECK_RESULT(nvm_v2_u64(&c, &v), NVM_V2_OK, "u64 reads");
    CHECK(v == 0x8000000000000001ull, "u64 is little-endian across all eight bytes");
}

static void test_read_past_the_end_is_rejected(void) {
    uint8_t buf[2] = { 0xAA, 0xBB };
    NvmV2Cursor c;
    uint32_t d;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    CHECK_RESULT(nvm_v2_u32(&c, &d), NVM_V2_ERR_TRUNCATED,
                 "a u32 that does not fit is rejected");
}

static void test_a_failed_read_does_not_advance(void) {
    /* A decoder that ignores a failure must not silently resynchronise at a
     * different offset, so a rejected read leaves the cursor where it was. */
    uint8_t buf[2] = { 0xAA, 0xBB };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    uint32_t d; uint8_t a;
    nvm_v2_u32(&c, &d);
    CHECK_RESULT(nvm_v2_u8(&c, &a), NVM_V2_OK, "a byte is still readable");
    CHECK(a == 0xAA, "the cursor did not move on the failed read");
}

static void test_take_of_zero_bytes_succeeds(void) {
    /* Zero-length payloads are legal -- an empty string constant, say -- so
     * taking nothing must not be confused with running out. */
    uint8_t buf[1] = { 0x42 };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    const uint8_t *p = NULL;
    CHECK_RESULT(nvm_v2_take(&c, 0, &p), NVM_V2_OK, "a zero-length take succeeds");
    CHECK(p == buf, "it yields the current position");
}

static void test_take_cannot_be_wrapped_past_the_end(void) {
    /* SIZE_MAX would pass `pos + n > size` on a wrapping addition. The bound is
     * written by subtraction so it cannot. */
    uint8_t buf[4] = { 1, 2, 3, 4 };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    const uint8_t *p = NULL;
    CHECK_RESULT(nvm_v2_take(&c, (size_t)-1, &p), NVM_V2_ERR_TRUNCATED,
                 "a length that would wrap is rejected");
}

static void test_align4_accepts_zero_padding(void) {
    uint8_t buf[4] = { 0xAA, 0x00, 0x00, 0x00 };
    NvmV2Cursor c;
    uint8_t v;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    nvm_v2_u8(&c, &v);
    CHECK_RESULT(nvm_v2_align4(&c), NVM_V2_OK, "zero padding is accepted");
    CHECK(nvm_v2_cursor_exhausted(&c), "and consumes to the boundary");
}

static void test_align4_rejects_nonzero_padding(void) {
    uint8_t buf[4] = { 0xAA, 0x00, 0x99, 0x00 };
    NvmV2Cursor c;
    uint8_t v;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    nvm_v2_u8(&c, &v);
    CHECK_RESULT(nvm_v2_align4(&c), NVM_V2_ERR_SECTION_RANGE,
                 "nonzero padding is rejected");
}

static void test_align4_at_a_boundary_is_a_noop(void) {
    uint8_t buf[4] = { 1, 2, 3, 4 };
    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    CHECK_RESULT(nvm_v2_align4(&c), NVM_V2_OK, "already aligned succeeds");
    uint32_t d;
    CHECK_RESULT(nvm_v2_u32(&c, &d), NVM_V2_OK, "and consumed nothing");
}

static void test_align4_rejects_truncated_padding(void) {
    /* Aligned-to-4 is a property of the section, so a section that ends
     * mid-padding is malformed rather than merely finished. */
    uint8_t buf[2] = { 0xAA, 0x00 };
    NvmV2Cursor c;
    uint8_t v;
    nvm_v2_cursor_init(&c, buf, sizeof buf);
    nvm_v2_u8(&c, &v);
    CHECK_RESULT(nvm_v2_align4(&c), NVM_V2_ERR_TRUNCATED,
                 "padding running past the end is rejected");
}

int main(void) {
    printf("\n[nvm_v2_cursor] section-decoding cursor tests...\n\n");
    test_reads_are_little_endian_and_sequential();
    test_u64_is_little_endian();
    test_read_past_the_end_is_rejected();
    test_a_failed_read_does_not_advance();
    test_take_of_zero_bytes_succeeds();
    test_take_cannot_be_wrapped_past_the_end();
    test_align4_accepts_zero_padding();
    test_align4_rejects_nonzero_padding();
    test_align4_at_a_boundary_is_a_noop();
    test_align4_rejects_truncated_padding();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
