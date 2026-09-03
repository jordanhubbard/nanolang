/*
 * Unit tests for the v2 IMPORTS, LINKS, METADATA and DEBUG sections.
 *
 * All four are fixed-width record tables. What is worth testing is the
 * discrimination each one performs: an import kind outside the known set, a
 * link flag bit this reader does not implement, and a debug offset wide enough
 * to need the u64 that v1 did not have.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
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

/* ── IMPORTS ────────────────────────────────────────────────────────────── */

static size_t build_imports(uint8_t *buf, size_t cap) {
    NvmV2Import items[2] = {
        { 1, 2, 3, NVM_V2_IMPORT_FFI },
        { 4, 5, 6, NVM_V2_IMPORT_COPROCESS },
    };
    NvmV2Imports i = { items, 2 };
    size_t n = nvm_v2_imports_encoded_size(&i);
    if (n == 0 || n > cap) return 0;
    nvm_v2_imports_encode(&i, buf, n);
    return n;
}

static void test_imports_round_trip(void) {
    uint8_t buf[128];
    size_t n = build_imports(buf, sizeof buf);
    CHECK(n == 4 + 2 * 16, "two import entries are 16 bytes each");
    NvmV2Imports got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_imports_decode(buf, n, &got), NVM_V2_OK, "decodes");
    CHECK(got.count == 2, "two imports");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].module_name_idx == 1, "module name index round-trips");
        CHECK(got.items[0].symbol_name_idx == 2, "symbol name index round-trips");
        CHECK(got.items[0].signature_idx == 3, "signature index round-trips");
        CHECK(got.items[0].kind == NVM_V2_IMPORT_FFI, "an FFI import round-trips");
        CHECK(got.items[1].kind == NVM_V2_IMPORT_COPROCESS,
              "a co-process import round-trips");
    } else { g_fail += 5; printf("  FAIL: imports did not decode usably\n"); }
    nvm_v2_imports_free(&got);
}

static void test_imports_unknown_kind_is_rejected(void) {
    uint8_t buf[128];
    size_t n = build_imports(buf, sizeof buf);
    buf[4 + 12] = NVM_V2_IMPORT_KIND_MAX + 1;
    NvmV2Imports got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_imports_decode(buf, n, &got), NVM_V2_ERR_SECTION_TYPE,
                 "an unknown import kind is rejected");
    nvm_v2_imports_free(&got);
}

static void test_imports_nonzero_padding_is_rejected(void) {
    uint8_t buf[128];
    size_t n = build_imports(buf, sizeof buf);
    buf[4 + 13] = 0x01;
    NvmV2Imports got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_imports_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "nonzero reserved padding in an import entry is rejected");
    nvm_v2_imports_free(&got);
}

static void test_imports_truncated_and_absurd(void) {
    uint8_t buf[128];
    size_t n = build_imports(buf, sizeof buf);
    NvmV2Imports got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_imports_decode(buf, n - 1, &got), NVM_V2_ERR_TRUNCATED,
                 "a section ending mid-entry is rejected");
    nvm_v2_imports_free(&got);
    uint8_t absurd[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    NvmV2Imports g2 = { NULL, 0 };
    CHECK_RESULT(nvm_v2_imports_decode(absurd, 4, &g2), NVM_V2_ERR_TRUNCATED,
                 "an absurd import count is rejected");
    nvm_v2_imports_free(&g2);
}

/* ── LINKS ──────────────────────────────────────────────────────────────── */

static size_t build_links(uint8_t *buf, size_t cap) {
    NvmV2Link items[2] = { { 1, 2, 3, 0 }, { 4, 5, 6, NVM_V2_LINK_WEAK } };
    NvmV2Links l = { items, 2 };
    size_t n = nvm_v2_links_encoded_size(&l);
    if (n == 0 || n > cap) return 0;
    nvm_v2_links_encode(&l, buf, n);
    return n;
}

static void test_links_round_trip(void) {
    uint8_t buf[128];
    size_t n = build_links(buf, sizeof buf);
    CHECK(n == 4 + 2 * 16, "two link entries are 16 bytes each");
    NvmV2Links got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_links_decode(buf, n, &got), NVM_V2_OK, "decodes");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].flags == 0, "a strong link has no flags");
        CHECK((got.items[1].flags & NVM_V2_LINK_WEAK) != 0, "the weak flag survives");
        CHECK(got.items[1].signature_idx == 6, "signature index round-trips");
    } else { g_fail += 3; printf("  FAIL: links did not decode usably\n"); }
    nvm_v2_links_free(&got);
}

static void test_links_unknown_flag_is_rejected(void) {
    uint8_t buf[128];
    size_t n = build_links(buf, sizeof buf);
    buf[4 + 12] = 0x80;   /* entry 0's flags, a bit outside the known mask */
    NvmV2Links got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_links_decode(buf, n, &got), NVM_V2_ERR_RESERVED_FLAGS,
                 "an unknown link flag bit is rejected");
    nvm_v2_links_free(&got);
}

/* ── METADATA ───────────────────────────────────────────────────────────── */

static void test_metadata_round_trip(void) {
    NvmV2MetadataEntry items[2] = { { 1, 2 }, { 3, 4 } };
    NvmV2Metadata m = { items, 2 };
    uint8_t buf[64];
    size_t n = nvm_v2_metadata_encoded_size(&m);
    CHECK(n == 4 + 2 * 8, "two metadata entries are 8 bytes each");
    CHECK_RESULT(nvm_v2_metadata_encode(&m, buf, sizeof buf), NVM_V2_OK, "encodes");
    NvmV2Metadata got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_metadata_decode(buf, n, &got), NVM_V2_OK, "decodes");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].key_idx == 1 && got.items[0].value_idx == 2,
              "first pair round-trips");
        CHECK(got.items[1].key_idx == 3 && got.items[1].value_idx == 4,
              "second pair round-trips");
    } else { g_fail += 2; printf("  FAIL: metadata did not decode usably\n"); }
    nvm_v2_metadata_free(&got);

    uint8_t absurd[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    NvmV2Metadata g2 = { NULL, 0 };
    CHECK_RESULT(nvm_v2_metadata_decode(absurd, 4, &g2), NVM_V2_ERR_TRUNCATED,
                 "an absurd metadata count is rejected");
    nvm_v2_metadata_free(&g2);
}

/* ── DEBUG ──────────────────────────────────────────────────────────────── */

static void test_debug_round_trip(void) {
    /* An offset above 2^32 so the widening from v1's u32 is exercised. */
    NvmV2DebugEntry items[2] = { { 0x100000020ull, 42, 7 }, { 0, 1, 0 } };
    NvmV2Debug d = { items, 2 };
    uint8_t buf[64];
    size_t n = nvm_v2_debug_encoded_size(&d);
    CHECK(n == 4 + 2 * 16, "two debug entries are 16 bytes each");
    CHECK_RESULT(nvm_v2_debug_encode(&d, buf, sizeof buf), NVM_V2_OK, "encodes");
    NvmV2Debug got = { NULL, 0 };
    CHECK_RESULT(nvm_v2_debug_decode(buf, n, &got), NVM_V2_OK, "decodes");
    if (got.count == 2 && got.items) {
        CHECK(got.items[0].bytecode_offset == 0x100000020ull,
              "a bytecode offset above 2^32 survives the 64-bit field");
        CHECK(got.items[0].source_line == 42, "source line round-trips");
        CHECK(got.items[0].source_col == 7, "source column round-trips");
        CHECK(got.items[1].source_col == 0, "an unknown column stays 0");
    } else { g_fail += 4; printf("  FAIL: debug did not decode usably\n"); }
    nvm_v2_debug_free(&got);

    NvmV2Debug g2 = { NULL, 0 };
    CHECK_RESULT(nvm_v2_debug_decode(buf, n - 1, &g2), NVM_V2_ERR_TRUNCATED,
                 "a debug section ending mid-entry is rejected");
    nvm_v2_debug_free(&g2);
}

static void test_all_empty_sections_decode(void) {
    uint8_t buf[8];
    NvmV2Imports i = { NULL, 0 };  NvmV2Links l = { NULL, 0 };
    NvmV2Metadata m = { NULL, 0 }; NvmV2Debug d = { NULL, 0 };
    CHECK(nvm_v2_imports_encoded_size(&i) == 4, "empty imports is just the count");
    CHECK(nvm_v2_links_encoded_size(&l) == 4, "empty links is just the count");
    CHECK(nvm_v2_metadata_encoded_size(&m) == 4, "empty metadata is just the count");
    CHECK(nvm_v2_debug_encoded_size(&d) == 4, "empty debug is just the count");

    memset(buf, 0, sizeof buf);
    NvmV2Imports gi = { NULL, 0 };
    CHECK_RESULT(nvm_v2_imports_decode(buf, 4, &gi), NVM_V2_OK, "empty imports decode");
    CHECK(gi.count == 0, "no imports");
    nvm_v2_imports_free(&gi);
}

int main(void) {
    printf("\n[nvm_v2_imports] IMPORTS, LINKS, METADATA and DEBUG tests...\n\n");
    test_imports_round_trip();
    test_imports_unknown_kind_is_rejected();
    test_imports_nonzero_padding_is_rejected();
    test_imports_truncated_and_absurd();
    test_links_round_trip();
    test_links_unknown_flag_is_rejected();
    test_metadata_round_trip();
    test_debug_round_trip();
    test_all_empty_sections_decode();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
