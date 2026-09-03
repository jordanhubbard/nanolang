/*
 * Unit tests for the v2 FUNCTIONS and GLOBALS sections.
 *
 * Both are fixed-width record tables, so the interesting cases are the fields
 * that changed meaning from v1: FUNCTIONS no longer carries arity or result
 * shape (those moved to SIGNATURES) and gained max_stack; GLOBALS exists at
 * all, which is what lets a VM size its globals from declarations.
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

/* ── FUNCTIONS ──────────────────────────────────────────────────────────── */

#define FN_ENTRY_BYTES 32

static size_t build_fns(uint8_t *buf, size_t cap) {
    NvmV2Function items[2];
    /* A 64-bit code_offset above 2^32 so the widening from v1's u32 is
     * actually exercised rather than assumed. */
    items[0].name_idx = 7; items[0].signature_idx = 1;
    items[0].code_offset = 0x100000010ull; items[0].code_length = 64;
    items[0].local_count = 3; items[0].upvalue_count = 1; items[0].max_stack = 9;
    items[1].name_idx = 8; items[1].signature_idx = 0;
    items[1].code_offset = 0; items[1].code_length = 0;
    items[1].local_count = 0; items[1].upvalue_count = 0; items[1].max_stack = 0;
    NvmV2Functions f = { items, 2 };
    size_t n = nvm_v2_functions_encoded_size(&f);
    if (n == 0 || n > cap) return 0;
    nvm_v2_functions_encode(&f, buf, n);
    return n;
}

static void test_functions_round_trip(void) {
    uint8_t buf[128];
    size_t n = build_fns(buf, sizeof buf);
    CHECK(n > 0, "functions fixture encodes");
    CHECK(n == 4 + 2 * FN_ENTRY_BYTES, "two entries are 32 bytes each");

    NvmV2Functions got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_functions_decode(buf, n, &got), NVM_V2_OK, "decodes");
    CHECK(got.count == 2, "two functions");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].name_idx == 7, "name index round-trips");
        CHECK(got.items[0].signature_idx == 1, "signature index round-trips");
        CHECK(got.items[0].code_offset == 0x100000010ull,
              "a code offset above 2^32 survives the 64-bit field");
        CHECK(got.items[0].code_length == 64, "code length round-trips");
        CHECK(got.items[0].local_count == 3, "local count round-trips");
        CHECK(got.items[0].upvalue_count == 1, "upvalue count round-trips");
        CHECK(got.items[0].max_stack == 9, "max_stack round-trips");
        CHECK(got.items[1].max_stack == 0, "a function with no body has max_stack 0");
    } else {
        g_fail += 8;
        printf("  FAIL: decode did not produce two usable functions\n");
    }
    nvm_v2_functions_free(&got);
}

static void test_functions_reserved_flags_must_be_zero(void) {
    uint8_t buf[128];
    size_t n = build_fns(buf, sizeof buf);
    buf[4 + 30] = 0x01;   /* entry 0's flags, at offset 30 within the entry */
    NvmV2Functions got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_functions_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "nonzero reserved flags in a function entry are rejected");
    nvm_v2_functions_free(&got);
}

static void test_functions_truncated_entry_is_rejected(void) {
    uint8_t buf[128];
    size_t n = build_fns(buf, sizeof buf);
    NvmV2Functions got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_functions_decode(buf, n - 1, &got), NVM_V2_ERR_TRUNCATED,
                 "a section ending mid-entry is rejected");
    nvm_v2_functions_free(&got);
}

static void test_functions_absurd_count_is_rejected(void) {
    uint8_t buf[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    NvmV2Functions got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_functions_decode(buf, sizeof buf, &got), NVM_V2_ERR_TRUNCATED,
                 "a count larger than the section could hold is rejected");
    nvm_v2_functions_free(&got);
}

static void test_functions_empty_and_short_buffer(void) {
    NvmV2Functions f = { NULL, 0 };
    uint8_t buf[8];
    CHECK(nvm_v2_functions_encoded_size(&f) == 4, "an empty table is just the count");
    nvm_v2_functions_encode(&f, buf, sizeof buf);
    NvmV2Functions got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_functions_decode(buf, 4, &got), NVM_V2_OK, "empty decodes");
    CHECK(got.count == 0, "no functions");
    nvm_v2_functions_free(&got);

    NvmV2Function one[1];
    memset(one, 0, sizeof one);
    NvmV2Functions g = { one, 1 };
    uint8_t small[8];
    CHECK_RESULT(nvm_v2_functions_encode(&g, small, sizeof small),
                 NVM_V2_ERR_TRUNCATED, "encoding into too small a buffer is rejected");
}

/* ── GLOBALS ────────────────────────────────────────────────────────────── */

#define GL_ENTRY_BYTES 12

static size_t build_globals(uint8_t *buf, size_t cap) {
    NvmV2Global items[2];
    items[0].name_idx = 3; items[0].type_tag = TAG_INT;
    items[0].flags = NVM_V2_GLOBAL_MUTABLE; items[0].init_idx = 5;
    items[1].name_idx = 4; items[1].type_tag = TAG_STRING;
    items[1].flags = 0; items[1].init_idx = NVM_V2_NO_INDEX;
    NvmV2Globals g = { items, 2 };
    size_t n = nvm_v2_globals_encoded_size(&g);
    if (n == 0 || n > cap) return 0;
    nvm_v2_globals_encode(&g, buf, n);
    return n;
}

static void test_globals_round_trip(void) {
    uint8_t buf[128];
    size_t n = build_globals(buf, sizeof buf);
    CHECK(n > 0, "globals fixture encodes");
    CHECK(n == 4 + 2 * GL_ENTRY_BYTES, "two entries are 12 bytes each");

    NvmV2Globals got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_globals_decode(buf, n, &got), NVM_V2_OK, "decodes");
    CHECK(got.count == 2, "two globals");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].name_idx == 3, "name index round-trips");
        CHECK(got.items[0].type_tag == TAG_INT, "type tag round-trips");
        CHECK((got.items[0].flags & NVM_V2_GLOBAL_MUTABLE) != 0, "mutable flag survives");
        CHECK(got.items[0].init_idx == 5, "initializer index round-trips");
        CHECK(got.items[1].flags == 0, "an immutable global has no flags set");
        CHECK(got.items[1].init_idx == NVM_V2_NO_INDEX,
              "a zero-initialized global carries the sentinel, not index 0");
    } else {
        g_fail += 6;
        printf("  FAIL: decode did not produce two usable globals\n");
    }
    nvm_v2_globals_free(&got);
}

static void test_globals_invalid_type_tag_is_rejected(void) {
    uint8_t buf[128];
    size_t n = build_globals(buf, sizeof buf);
    buf[4 + 4] = TAG_COUNT;   /* entry 0's type_tag */
    NvmV2Globals got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_globals_decode(buf, n, &got), NVM_V2_ERR_SECTION_TYPE,
                 "a type tag at or above TAG_COUNT is rejected");
    nvm_v2_globals_free(&got);
}

static void test_globals_unknown_flag_bit_is_rejected(void) {
    uint8_t buf[128];
    size_t n = build_globals(buf, sizeof buf);
    buf[4 + 5] = 0x80;   /* entry 0's flags, a bit outside the known mask */
    NvmV2Globals got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_globals_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "an unknown global flag bit is rejected");
    nvm_v2_globals_free(&got);
}

static void test_globals_nonzero_padding_is_rejected(void) {
    uint8_t buf[128];
    size_t n = build_globals(buf, sizeof buf);
    buf[4 + 6] = 0x01;   /* entry 0's _pad */
    NvmV2Globals got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_globals_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "nonzero reserved padding in a global entry is rejected");
    nvm_v2_globals_free(&got);
}

static void test_globals_truncated_and_absurd(void) {
    uint8_t buf[128];
    size_t n = build_globals(buf, sizeof buf);
    NvmV2Globals got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_globals_decode(buf, n - 1, &got), NVM_V2_ERR_TRUNCATED,
                 "a section ending mid-entry is rejected");
    nvm_v2_globals_free(&got);

    uint8_t absurd[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    NvmV2Globals g2 = { NULL, 0 };
    CHECK_RESULT(nvm_v2_globals_decode(absurd, sizeof absurd, &g2),
                 NVM_V2_ERR_TRUNCATED, "an absurd count is rejected");
    nvm_v2_globals_free(&g2);
}

int main(void) {
    printf("\n[nvm_v2_functions] FUNCTIONS and GLOBALS section tests...\n\n");
    test_functions_round_trip();
    test_functions_reserved_flags_must_be_zero();
    test_functions_truncated_entry_is_rejected();
    test_functions_absurd_count_is_rejected();
    test_functions_empty_and_short_buffer();
    test_globals_round_trip();
    test_globals_invalid_type_tag_is_rejected();
    test_globals_unknown_flag_bit_is_rejected();
    test_globals_nonzero_padding_is_rejected();
    test_globals_truncated_and_absurd();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
