#include "../../src/nanoisa/managed_strings.h"
#ifndef __wasm32__
#include <stdio.h>
#endif
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char bytes[] = {'a',0,'z'};
static const unsigned char empty[] = {0};
static const NmsView literals[] = {{bytes,3},{empty,0}};
static unsigned char large_bytes[100000];
static int same(const unsigned char *a, const unsigned char *b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

int nms_core_tests(void) {
    NmsRuntime runtime;
    nms_init(&runtime, literals, 2);
    NmsView view;
    CHECK(nms_view(&runtime, 1, &view) == NMS_OK && view.length == 3 && same(view.data,bytes,3));
    CHECK(nms_retain(&runtime, 2) == NMS_OK && nms_release(&runtime, 2) == NMS_OK);
    CHECK(nms_view(&runtime, 2, &view) == NMS_OK && view.data && !view.length);
    NmsHandle first, zero;
    CHECK(nms_create(&runtime, bytes, 3, &first) == NMS_OK);
    CHECK(first & NMS_DYNAMIC);
    CHECK(nms_create(&runtime, NULL, 0, &zero) == NMS_OK);
    CHECK(nms_view(&runtime, zero, &view) == NMS_OK && view.data && !view.length && !view.data[0]);
    CHECK(nms_retain(&runtime, first) == NMS_OK);
    CHECK(nms_release(&runtime, first) == NMS_OK);
    CHECK(nms_view(&runtime, first, &view) == NMS_OK && same(view.data,bytes,3));
    CHECK((uintptr_t)view.data % 16 == 0);
    const unsigned char *borrowed = view.data;
    NmsHandle handles[48];
    for (unsigned i = 0; i < 48; i++)
        CHECK(nms_create(&runtime, borrowed, 3, &handles[i]) == NMS_OK);
    CHECK(runtime.capacity >= 50 && runtime.live_objects == 50);
    CHECK(nms_view(&runtime, first, &view) == NMS_OK && view.data == borrowed && same(view.data,bytes,3));
    for (unsigned i = 0; i < 48; i += 2) CHECK(nms_release(&runtime, handles[i]) == NMS_OK);
    for (unsigned i = 1; i < 48; i += 2) CHECK(nms_release(&runtime, handles[i]) == NMS_OK);
    CHECK(nms_release(&runtime, first) == NMS_OK);
    CHECK(nms_release(&runtime, zero) == NMS_OK);
    CHECK(runtime.live_objects == 0 && runtime.live_bytes == 0);
    CHECK(nms_test_live_allocations() == 1); /* Only the reusable descriptor table. */
    CHECK(nms_dispose(&runtime) == NMS_OK && nms_test_live_allocations() == 0);
    CHECK(nms_dispose(&runtime) == NMS_OK);
    CHECK(nms_begin(&runtime) == NMS_DISPOSED);
    CHECK(nms_view(&runtime, first, &view) == NMS_DISPOSED);
    CHECK(nms_reserved_entry("nano_try_entry") && nms_reserved_entry("nano_dispose"));
    CHECK(nms_reserved_entry("nano_runtime_internal") && nms_reserved_entry("nms_create"));
    CHECK(!nms_reserved_entry("nano_entry") && !nms_reserved_entry("nano_try_entry_other"));
    return 0;
}
int nms_format_tests(void) {
    const struct { uint64_t bits; uint32_t tag, length; const char *expected; } cases[] = {
        {0,1,1,"0"}, {UINT64_MAX,1,2,"-1"},
        {INT64_MAX,1,19,"9223372036854775807"},
        {UINT64_C(1)<<63,1,20,"-9223372036854775808"},
        {0,2,1,"0"}, {255,2,3,"255"}, {0,4,5,"false"}, {1,4,4,"true"},
        {0,0,0,""}, {17,9,0,""}
    };
    NmsRuntime runtime;
    nms_init(&runtime, NULL, 0);
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        NmsHandle out = 123;
        CHECK(nms_format_scalar(&runtime, cases[i].bits, cases[i].tag, &out) == NMS_OK);
        NmsView view;
        CHECK(nms_view(&runtime, out, &view) == NMS_OK && view.length == cases[i].length);
        CHECK(same(view.data, (const unsigned char *)cases[i].expected, view.length));
        CHECK(nms_release(&runtime, out) == NMS_OK && runtime.live_objects == 0);
    }
#ifdef NMS_TESTING
    nms_test_fail_after(&runtime, 0);
    NmsHandle out = 123;
    CHECK(nms_format_scalar(&runtime, UINT64_C(1)<<63, 1, &out) == NMS_MEMORY);
    CHECK(out == 123 && runtime.live_objects == 0);
    nms_test_fail_after(&runtime, UINT64_MAX);
    CHECK(nms_format_scalar(&runtime, 7, 1, &out) == NMS_OK);
    CHECK(nms_release(&runtime, out) == NMS_OK);
#endif
    CHECK(nms_dispose(&runtime) == NMS_OK);
    return 0;
}
int nms_decimal_tests(void) {
    const struct { const char *text; uint32_t length; int64_t expected; } cases[] = {
        {"",0,0}, {"+",1,0}, {"-",1,0}, {"  -42tail",9,-42},
        {"\t\n\r\v\f +17",9,17}, {"12\0" "99",5,12}, {"0x20",4,0},
        {"9223372036854775807",19,INT64_MAX},
        {"9223372036854775808",19,INT64_MAX},
        {"-9223372036854775808",20,INT64_MIN},
        {"-9223372036854775809",20,INT64_MIN},
        {"999999999999999999999999",24,INT64_MAX},
        {"-999999999999999999999999",25,INT64_MIN}
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        NmsView literal = {(const unsigned char *)cases[i].text, cases[i].length};
        NmsRuntime runtime;
        nms_init(&runtime, &literal, 1);
        int64_t parsed = 123;
        CHECK(nms_parse_i64(&runtime, 1, &parsed) == NMS_OK && parsed == cases[i].expected);
        NmsHandle source;
        CHECK(nms_create(&runtime, literal.data, literal.length, &source) == NMS_OK);
        CHECK(nms_retain(&runtime, source) == NMS_OK);
#ifdef NMS_TESTING
        nms_test_fail_after(&runtime, 0);
#endif
        CHECK(nms_parse_i64(&runtime, source, &parsed) == NMS_OK && parsed == cases[i].expected);
        CHECK(runtime.live_objects == 1 && runtime.live_bytes == literal.length);
        CHECK(runtime.slots[source & ~NMS_DYNAMIC].references == 2);
        CHECK(nms_release(&runtime, source) == NMS_OK);
        CHECK(nms_release(&runtime, source) == NMS_OK);
        CHECK(nms_dispose(&runtime) == NMS_OK);
    }
    return 0;
}
int nms_substr_tests(void) {
    NmsRuntime runtime;
    nms_init(&runtime, literals, 2);
    NmsHandle source, out = 123;
    NmsView view;
    CHECK(nms_create(&runtime, bytes, 3, &source) == NMS_OK);
    CHECK(nms_retain(&runtime, source) == NMS_OK);
    CHECK(nms_substr_owned(&runtime, source, 1, 20, &out) == NMS_OK);
    CHECK(nms_view(&runtime, out, &view) == NMS_OK && view.length == 2);
    CHECK(view.data[0] == bytes[1] && view.data[1] == bytes[2]);
    CHECK(nms_release(&runtime, out) == NMS_OK);
    CHECK(nms_view(&runtime, source, &view) == NMS_OK && view.length == 3);
#ifdef NMS_TESTING
    nms_test_fail_after(&runtime, 0);
    out = 123;
    CHECK(nms_substr_owned(&runtime, source, 0, 1, &out) == NMS_MEMORY);
    CHECK(out == 123 && runtime.live_objects == 0);
    nms_test_fail_after(&runtime, UINT64_MAX);
#else
    CHECK(nms_release(&runtime, source) == NMS_OK);
#endif
    CHECK(nms_substr_owned(&runtime, 1, 3, 20, &out) == NMS_OK);
    CHECK(nms_view(&runtime, out, &view) == NMS_OK && view.length == 0);
    CHECK(nms_release(&runtime, out) == NMS_OK);
    CHECK(nms_dispose(&runtime) == NMS_OK);
#ifdef NMS_TESTING
    nms_init(&runtime, literals, 2);
    NmsHandle full[8];
    for (unsigned i = 0; i < 8; i++) CHECK(nms_create(&runtime, bytes, 3, &full[i]) == NMS_OK);
    NmsSlot *old = runtime.slots;
    for (unsigned fail = 0; fail < 2; fail++) {
        CHECK(nms_retain(&runtime, full[0]) == NMS_OK);
        nms_test_fail_after(&runtime, fail);
        out = 123;
        CHECK(nms_substr_owned(&runtime, full[0], 0, 2, &out) == NMS_MEMORY);
        CHECK(out == 123 && runtime.slots == old && runtime.capacity == 8);
        CHECK(runtime.live_objects == 8 && runtime.live_bytes == 24);
        CHECK(nms_view(&runtime, full[0], &view) == NMS_OK && view.length == 3);
        CHECK(runtime.slots[full[0] & ~NMS_DYNAMIC].references == 1);
    }
    nms_test_fail_after(&runtime, UINT64_MAX);
    CHECK(nms_retain(&runtime, full[0]) == NMS_OK);
    CHECK(nms_substr_owned(&runtime, full[0], 0, 2, &out) == NMS_OK);
    CHECK(runtime.capacity == 16 && runtime.live_objects == 9);
    CHECK(nms_release(&runtime, out) == NMS_OK);
    for (unsigned i = 0; i < 8; i++) CHECK(nms_release(&runtime, full[i]) == NMS_OK);
    CHECK(nms_dispose(&runtime) == NMS_OK && nms_test_live_allocations() == 0);
#endif
    return 0;
}
int nms_concat_tests(void) {
    NmsRuntime runtime;
    nms_init(&runtime, literals, 2);
    NmsHandle a, result;
    NmsView view;
    CHECK(nms_create(&runtime, bytes, 3, &a) == NMS_OK);
    CHECK(nms_retain(&runtime, a) == NMS_OK);
    CHECK(nms_retain(&runtime, a) == NMS_OK); /* Two inputs and an external alias. */
    CHECK(nms_concat_owned(&runtime, a, a, &result) == NMS_OK);
    CHECK(nms_view(&runtime, result, &view) == NMS_OK && view.length == 6);
    CHECK(same(view.data, bytes, 3) && same(view.data + 3, bytes, 3) && !view.data[6]);
    CHECK(nms_view(&runtime, a, &view) == NMS_OK && view.length == 3);
    CHECK(runtime.live_objects == 2);
    CHECK(nms_release(&runtime, a) == NMS_OK);
    CHECK(nms_concat_owned(&runtime, result, 2, &a) == NMS_OK);
    CHECK(runtime.live_objects == 1 && runtime.live_bytes == 6);
    CHECK(nms_release(&runtime, a) == NMS_OK);
    CHECK(nms_concat_owned(&runtime, 2, 2, &result) == NMS_OK);
    CHECK(nms_view(&runtime, result, &view) == NMS_OK && !view.length && view.data);
    CHECK(nms_release(&runtime, result) == NMS_OK);
    NmsHandle full[8];
    for (unsigned i = 0; i < 8; i++)
        CHECK(nms_create(&runtime, bytes, 3, &full[i]) == NMS_OK);
    NmsSlot *old = runtime.slots;
    for (unsigned fail = 0; fail < 2; fail++) {
        CHECK(nms_retain(&runtime, full[0]) == NMS_OK);
        CHECK(nms_retain(&runtime, full[1]) == NMS_OK);
        nms_test_fail_after(&runtime, fail);
        result = 123;
        CHECK(nms_concat_owned(&runtime, full[0], full[1], &result) == NMS_MEMORY);
        CHECK(result == 123 && runtime.slots == old && runtime.capacity == 8);
        CHECK(runtime.live_objects == 8 && runtime.live_bytes == 24);
        for (unsigned i = 0; i < 8; i++) {
            CHECK(nms_view(&runtime, full[i], &view) == NMS_OK && same(view.data, bytes, 3));
            CHECK(runtime.slots[full[i] & ~NMS_DYNAMIC].references == 1);
        }
    }
    nms_test_fail_after(&runtime, UINT64_MAX);
    CHECK(nms_retain(&runtime, full[0]) == NMS_OK);
    CHECK(nms_concat_owned(&runtime, full[0], 1, &result) == NMS_OK);
    CHECK(runtime.capacity == 16 && runtime.live_objects == 9);
    CHECK(nms_view(&runtime, result, &view) == NMS_OK && view.length == 6);
    CHECK(nms_release(&runtime, result) == NMS_OK);
    for (unsigned i = 0; i < 8; i++) CHECK(nms_release(&runtime, full[i]) == NMS_OK);
    CHECK(nms_create(&runtime, bytes, 3, &a) == NMS_OK);
    nms_test_fail_after(&runtime, 0);
    result = 123;
    CHECK(nms_concat_owned(&runtime, a, 1, &result) == NMS_MEMORY);
    CHECK(result == 123 && !runtime.live_objects); /* Failure consumed the only owner. */
    nms_test_fail_after(&runtime, UINT64_MAX);
    CHECK(nms_concat_owned(&runtime, 1, 2, &a) == NMS_OK);
    uint64_t pages = nms_test_memory_pages();
    for (unsigned i = 0; i < 2000; i++) {
        CHECK(nms_concat_owned(&runtime, a, 2, &a) == NMS_OK);
        CHECK(runtime.live_objects == 1 && runtime.live_bytes == 3);
    }
    CHECK(nms_test_memory_pages() == pages);
    CHECK(nms_release(&runtime, a) == NMS_OK);
    CHECK(nms_dispose(&runtime) == NMS_OK && !nms_test_live_allocations());
    return 0;
}
int nms_failure_tests(void) {
    NmsRuntime runtime;
    nms_init(&runtime, literals, 2);
    NmsHandle handles[8], out = 123;
    for (unsigned i = 0; i < 8; i++) CHECK(nms_create(&runtime, bytes, 3, &handles[i]) == NMS_OK);
    NmsSlot *old = runtime.slots;
    CHECK(runtime.capacity == 8 && !runtime.free_head);
    for (unsigned fail = 0; fail < 2; fail++) {
        nms_test_fail_after(&runtime, fail);
        CHECK(nms_create(&runtime, bytes, 3, &out) == NMS_MEMORY);
        CHECK(out == 123 && runtime.slots == old && runtime.capacity == 8);
        CHECK(runtime.live_objects == 8 && runtime.live_bytes == 24);
        CHECK(nms_test_live_allocations() == 9);
        for (unsigned i = 0; i < 8; i++) {
            NmsView view;
            CHECK(nms_view(&runtime, handles[i], &view) == NMS_OK && same(view.data,bytes,3));
        }
    }
    nms_test_fail_after(&runtime, UINT64_MAX);
    CHECK(nms_create(&runtime, bytes, UINT64_MAX, &out) == NMS_MEMORY && out == 123);
    CHECK(nms_create(&runtime, bytes, 3, &out) == NMS_OK && runtime.capacity == 16);
    uint32_t index = (uint32_t)(out & ~NMS_DYNAMIC);
    runtime.slots[index].references = UINT64_MAX;
    CHECK(nms_retain(&runtime, out) == NMS_MEMORY && runtime.slots[index].references == UINT64_MAX);
    runtime.slots[index].references = 1;
    CHECK(nms_begin(&runtime) == NMS_OK);
    CHECK(nms_begin(&runtime) == NMS_BUSY && nms_dispose(&runtime) == NMS_BUSY);
    CHECK(runtime.live_objects == 9);
    CHECK(nms_finish(&runtime, NMS_ASSERT, 999) == (UINT64_C(2) << 32));
    CHECK(nms_begin(&runtime) == NMS_OK);
    CHECK(nms_finish(&runtime, NMS_OK, -7) == UINT32_C(0xfffffff9));
    CHECK(nms_finish(&runtime, NMS_OK, 0) == (UINT64_C(6) << 32));
    CHECK(nms_dispose(&runtime) == NMS_OK && nms_test_live_allocations() == 0);
    return 0;
}
int nms_reuse_tests(void) {
    NmsRuntime runtime, second;
    nms_init(&runtime, literals, 2);
    nms_init(&second, literals, 2);
    NmsHandle a,b,c,d, other;
    for (unsigned i = 0; i < sizeof large_bytes; i++) large_bytes[i] = (unsigned char)(i * 17 + 3);
    /* The descriptor table is already allocated before the adjacent byte
     * allocations, so freeing a/b exercises coalescing around a live c. */
    CHECK(nms_create(&runtime, bytes, 3, &d) == NMS_OK);
    CHECK(nms_release(&runtime, d) == NMS_OK);
    CHECK(nms_create(&runtime, large_bytes, 24000, &a) == NMS_OK);
    CHECK(nms_create(&runtime, large_bytes, 24000, &b) == NMS_OK);
    CHECK(nms_create(&runtime, large_bytes, 24000, &c) == NMS_OK);
    uint64_t pages = nms_test_memory_pages();
    NmsView allocation;
    CHECK(nms_view(&runtime, a, &allocation) == NMS_OK);
#ifdef __wasm32__
    uintptr_t first_address = (uintptr_t)allocation.data;
#endif
    CHECK(nms_release(&runtime, a) == NMS_OK && nms_release(&runtime, b) == NMS_OK);
    CHECK(nms_create(&runtime, large_bytes + 123, 40000, &d) == NMS_OK);
    CHECK(nms_view(&runtime, d, &allocation) == NMS_OK && same(allocation.data,large_bytes+123,40000));
#ifdef __wasm32__
    /* First-fit can reuse this address for 40,000 bytes only if the two adjacent
     * freed 24,000-byte blocks coalesced; unused tail storage cannot fake this. */
    CHECK((uintptr_t)allocation.data == first_address);
#endif
    CHECK(nms_view(&runtime, c, &allocation) == NMS_OK && same(allocation.data,large_bytes,24000));
    CHECK(nms_test_memory_pages() == pages);
    CHECK(nms_release(&runtime, c) == NMS_OK && nms_release(&runtime, d) == NMS_OK);
    for (unsigned i = 0; i < 2000; i++) {
        CHECK(nms_create(&runtime, large_bytes, 60000, &a) == NMS_OK);
        CHECK(nms_release(&runtime, a) == NMS_OK);
    }
    CHECK(nms_test_memory_pages() == pages && runtime.live_objects == 0);
    CHECK(nms_create(&second, bytes, 3, &other) == NMS_OK);
    CHECK(nms_dispose(&runtime) == NMS_OK);
    NmsView view;
    CHECK(nms_view(&second, other, &view) == NMS_OK && same(view.data,bytes,3));
    CHECK(nms_dispose(&second) == NMS_OK && nms_test_live_allocations() == 0);
    return 0;
}
int nms_pressure_tests(void) {
#ifdef __wasm32__
    NmsRuntime runtime;
    nms_init(&runtime, literals, 2);
    NmsHandle handles[64];
    unsigned count = 0;
    while (count < 64) {
        NmsStatus status = nms_create(&runtime, large_bytes, 60000, &handles[count]);
        if (status == NMS_MEMORY) break;
        CHECK(status == NMS_OK);
        count++;
    }
    CHECK(count > 0 && count < 64);
    for (unsigned i = 0; i < count; i++) {
        NmsView view;
        CHECK(nms_view(&runtime, handles[i], &view) == NMS_OK && view.length == 60000);
        CHECK(nms_release(&runtime, handles[i]) == NMS_OK);
    }
    CHECK(runtime.live_objects == 0);
    NmsHandle again;
    CHECK(nms_create(&runtime, large_bytes, 60000, &again) == NMS_OK);
    CHECK(nms_release(&runtime, again) == NMS_OK);
    CHECK(nms_dispose(&runtime) == NMS_OK && nms_test_live_allocations() == 0);
#endif
    return 0;
}
#ifndef __wasm32__
int main(void) {
    int result = nms_core_tests();
    if (!result) result = nms_concat_tests();
    if (!result) result = nms_substr_tests();
    if (!result) result = nms_decimal_tests();
    if (!result) result = nms_format_tests();
    if (!result) result = nms_failure_tests();
    if (!result) result = nms_reuse_tests();
    if (result) { fprintf(stderr,"I failed managed-string check at line %d\n",result); return 1; }
    puts("I passed managed-string core, failure and reclamation checks.");
    return 0;
}
#endif
