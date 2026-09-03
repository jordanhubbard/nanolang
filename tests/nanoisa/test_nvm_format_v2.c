/*
 * Unit tests for the NanoISA v2 container: header and section directory.
 *
 * The container's whole job is to refuse malformed input before anything
 * downstream trusts an offset, so most of these are rejection tests.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nvm_format_v2.h"
#include "nvm_format.h"   /* nvm_crc32 */

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

/* ── Fixture ─────────────────────────────────────────────────────────────
 * A minimal well-formed module: header, a two-entry directory, and payloads.
 * Tests mutate a copy to exercise each rejection path, so the fixture must
 * start valid or the tests prove nothing.
 */

#define FIX_SECTIONS 2
#define FIX_DIR_OFF  NVM_V2_HEADER_SIZE
#define FIX_DIR_LEN  (FIX_SECTIONS * NVM_V2_SECTION_ENTRY_SIZE)
#define FIX_PAY_OFF  (FIX_DIR_OFF + FIX_DIR_LEN)
#define FIX_PAY_A    16
#define FIX_PAY_B    8
#define FIX_TOTAL    (FIX_PAY_OFF + FIX_PAY_A + FIX_PAY_B)

static size_t build_fixture(uint8_t *buf) {
    memset(buf, 0, FIX_TOTAL);

    NvmV2SectionEntry a = { NVM_V2_SECTION_CODE, 0, FIX_PAY_OFF, FIX_PAY_A };
    NvmV2SectionEntry b = { NVM_V2_SECTION_CONSTANTS, 0,
                            FIX_PAY_OFF + FIX_PAY_A, FIX_PAY_B };
    nvm_v2_write_section(buf + FIX_DIR_OFF, &a);
    nvm_v2_write_section(buf + FIX_DIR_OFF + NVM_V2_SECTION_ENTRY_SIZE, &b);

    for (size_t i = 0; i < FIX_PAY_A + FIX_PAY_B; i++)
        buf[FIX_PAY_OFF + i] = (uint8_t)(i * 7 + 1);

    NvmV2Header h;
    memset(&h, 0, sizeof(h));
    h.magic[0] = NVM_V2_MAGIC_0; h.magic[1] = NVM_V2_MAGIC_1;
    h.magic[2] = NVM_V2_MAGIC_2; h.magic[3] = NVM_V2_MAGIC_3;
    h.format_version = NVM_V2_FORMAT_VERSION;
    h.isa_version    = NVM_V2_ISA_VERSION;
    h.feature_bits   = 0;
    h.total_size     = FIX_TOTAL;
    h.header_size    = NVM_V2_HEADER_SIZE;
    h.section_count  = FIX_SECTIONS;
    h.entry_point    = NVM_V2_NO_ENTRY_POINT;
    h.flags          = 0;
    h.checksum       = 0;
    nvm_v2_write_header(buf, &h);

    /* Checksum covers everything after the header, so it must be written last. */
    h.checksum = nvm_crc32(buf + NVM_V2_HEADER_SIZE, FIX_TOTAL - NVM_V2_HEADER_SIZE);
    nvm_v2_write_header(buf, &h);
    return FIX_TOTAL;
}

/* ── Header ─────────────────────────────────────────────────────────────── */

static void test_valid_fixture_is_accepted(void) {
    uint8_t buf[FIX_TOTAL];
    size_t n = build_fixture(buf);
    CHECK_RESULT(nvm_v2_validate(buf, n), NVM_V2_OK, "the fixture is valid");
}

static void test_header_round_trips(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, FIX_TOTAL, &h), NVM_V2_OK, "header reads");
    CHECK(h.magic[3] == NVM_V2_MAGIC_3, "magic[3] is 0x02");
    CHECK(h.format_version == NVM_V2_FORMAT_VERSION, "format_version round-trips");
    CHECK(h.isa_version == NVM_V2_ISA_VERSION, "isa_version round-trips");
    CHECK(h.total_size == FIX_TOTAL, "total_size round-trips");
    CHECK(h.header_size == NVM_V2_HEADER_SIZE, "header_size round-trips");
    CHECK(h.section_count == FIX_SECTIONS, "section_count round-trips");
    CHECK(h.entry_point == NVM_V2_NO_ENTRY_POINT, "entry_point round-trips");
}

static void test_v1_magic_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[3] = 0x01;  /* a v1 module handed to the v2 reader */
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, FIX_TOTAL, &h), NVM_V2_ERR_MAGIC,
                 "v1 magic is rejected rather than misread");
}

static void test_short_buffer_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, NVM_V2_HEADER_SIZE - 1, &h),
                 NVM_V2_ERR_TRUNCATED, "a buffer shorter than the header is rejected");
}

static void test_future_format_version_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[4] = NVM_V2_FORMAT_VERSION + 1;
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, FIX_TOTAL, &h), NVM_V2_ERR_FORMAT_VERSION,
                 "a newer layout version is rejected");
}

static void test_unknown_feature_bit_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[8] |= 0x80;  /* a bit outside NVM_V2_FEATURE_KNOWN_MASK */
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, FIX_TOTAL, &h), NVM_V2_ERR_UNKNOWN_FEATURE,
                 "an unknown feature bit fails closed");
}

static void test_total_size_mismatch_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[12] = (uint8_t)(FIX_TOTAL + 1);  /* low byte of total_size */
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, FIX_TOTAL, &h), NVM_V2_ERR_TOTAL_SIZE,
                 "total_size must equal the real byte length");
}

static void test_wrong_header_size_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[20] = NVM_V2_HEADER_SIZE + 8;
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, FIX_TOTAL, &h), NVM_V2_ERR_HEADER_SIZE,
                 "an unexpected header_size is rejected");
}

static void test_reserved_flags_must_be_zero(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[32] = 0x01;
    NvmV2Header h;
    CHECK_RESULT(nvm_v2_read_header(buf, FIX_TOTAL, &h), NVM_V2_ERR_RESERVED_FLAGS,
                 "reserved header flags must be zero");
}

/* ── Section directory ──────────────────────────────────────────────────── */

static void test_sections_round_trip(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    NvmV2Header h;
    nvm_v2_read_header(buf, FIX_TOTAL, &h);
    NvmV2SectionEntry e;
    CHECK_RESULT(nvm_v2_read_section(buf, FIX_TOTAL, &h, 0, &e), NVM_V2_OK,
                 "section 0 reads");
    CHECK(e.type == NVM_V2_SECTION_CODE, "section 0 is CODE");
    CHECK(e.offset == FIX_PAY_OFF, "section 0 offset round-trips");
    CHECK(e.size == FIX_PAY_A, "section 0 size round-trips");
    CHECK_RESULT(nvm_v2_read_section(buf, FIX_TOTAL, &h, 1, &e), NVM_V2_OK,
                 "section 1 reads");
    CHECK(e.type == NVM_V2_SECTION_CONSTANTS, "section 1 is CONSTANTS");
}

static void test_unknown_section_type_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[FIX_DIR_OFF] = 0x7F;  /* type field, low byte */
    CHECK_RESULT(nvm_v2_validate(buf, FIX_TOTAL), NVM_V2_ERR_SECTION_TYPE,
                 "an unknown section type is rejected");
}

static void test_section_escaping_the_file_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    /* size field of entry 0 lives 16 bytes into the entry */
    buf[FIX_DIR_OFF + 16] = 0xFF;
    CHECK_RESULT(nvm_v2_validate(buf, FIX_TOTAL), NVM_V2_ERR_SECTION_RANGE,
                 "a section running past the end of the file is rejected");
}

static void test_section_offset_overflow_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    /* offset = 2^64-8 and a size that would wrap offset+size back into range
     * if the check used addition rather than subtraction. */
    for (int i = 0; i < 8; i++) buf[FIX_DIR_OFF + 8 + i] = 0xFF;
    buf[FIX_DIR_OFF + 8] = 0xF8;
    CHECK_RESULT(nvm_v2_validate(buf, FIX_TOTAL), NVM_V2_ERR_SECTION_RANGE,
                 "a wrapping offset+size cannot slip past the range check");
}

static void test_duplicate_section_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    /* Make entry 1 a second CODE section. */
    buf[FIX_DIR_OFF + NVM_V2_SECTION_ENTRY_SIZE] = NVM_V2_SECTION_CODE;
    CHECK_RESULT(nvm_v2_validate(buf, FIX_TOTAL), NVM_V2_ERR_SECTION_DUPLICATE,
                 "a duplicate section type is rejected");
}

static void test_overlapping_sections_are_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    /* Point entry 1 at entry 0's payload. */
    NvmV2SectionEntry b = { NVM_V2_SECTION_CONSTANTS, 0, FIX_PAY_OFF, FIX_PAY_B };
    nvm_v2_write_section(buf + FIX_DIR_OFF + NVM_V2_SECTION_ENTRY_SIZE, &b);
    CHECK_RESULT(nvm_v2_validate(buf, FIX_TOTAL), NVM_V2_ERR_SECTION_OVERLAP,
                 "overlapping sections are rejected");
}

static void test_section_overlapping_the_directory_is_rejected(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    /* A payload that starts inside the directory it is described by. */
    NvmV2SectionEntry b = { NVM_V2_SECTION_CONSTANTS, 0, FIX_DIR_OFF, FIX_PAY_B };
    nvm_v2_write_section(buf + FIX_DIR_OFF + NVM_V2_SECTION_ENTRY_SIZE, &b);
    CHECK_RESULT(nvm_v2_validate(buf, FIX_TOTAL), NVM_V2_ERR_SECTION_OVERLAP,
                 "a section overlapping the header or directory is rejected");
}

/* The checksum is verified last, after the structural checks. A corrupt
 * directory should report what is structurally wrong with it, not merely that
 * some byte changed -- the structural error is the actionable one. */
static void test_corrupt_payload_fails_checksum(void) {
    uint8_t buf[FIX_TOTAL];
    build_fixture(buf);
    buf[FIX_PAY_OFF] ^= 0xFF;   /* payload only: structurally still valid */
    CHECK_RESULT(nvm_v2_validate(buf, FIX_TOTAL), NVM_V2_ERR_CHECKSUM,
                 "a corrupted payload fails the checksum");
}

int main(void) {
    printf("\n[nvm_format_v2] container tests...\n\n");
    test_valid_fixture_is_accepted();
    test_header_round_trips();
    test_v1_magic_is_rejected();
    test_short_buffer_is_rejected();
    test_future_format_version_is_rejected();
    test_unknown_feature_bit_is_rejected();
    test_total_size_mismatch_is_rejected();
    test_wrong_header_size_is_rejected();
    test_reserved_flags_must_be_zero();
    test_sections_round_trip();
    test_unknown_section_type_is_rejected();
    test_section_escaping_the_file_is_rejected();
    test_section_offset_overflow_is_rejected();
    test_duplicate_section_is_rejected();
    test_overlapping_sections_are_rejected();
    test_section_overlapping_the_directory_is_rejected();
    test_corrupt_payload_fails_checksum();

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
