#include "../../src/nanoisa/managed_strings.h"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char bytes[] = {'a', 0, 255};
static const NmsView literals[] = {{bytes, 3}};
int nms_array_tests(void) {
    NmsRuntime r, other;
    nms_init(&r, literals, 1); nms_init(&other, literals, 1);
    NmsHandle array = 999, child, held = 888, aliases[20];
    uint32_t length = 777;
    NmsView view = {bytes, 123};
    CHECK(nms_string_array_create(&r, &array) == NMS_OK);
    CHECK(nms_retain(&r, array) == NMS_OK);
    CHECK(nms_create(&r, bytes, 3, &child) == NMS_OK);
    CHECK(nms_string_array_append(&r, array, child) == NMS_OK);
    CHECK(nms_string_array_append(&r, array, child) == NMS_OK);
    CHECK(nms_release(&r, child) == NMS_OK);
    CHECK(nms_view(&r, array, &view) == NMS_TYPE && view.length == 123);
    CHECK(nms_string_array_append(&r, array, array) == NMS_TYPE);
    CHECK(nms_string_array_append(&r, child, child) == NMS_TYPE);
    CHECK(nms_string_array_length(&r, child, &length) == NMS_TYPE && length == 777);
    CHECK(nms_string_array_get(&r, child, 0, &held) == NMS_TYPE && held == 888);
    CHECK(nms_string_array_length(&r, array, &length) == NMS_OK && length == 2);
    /* Table relocation preserves array storage and child owners. */
    for (unsigned i = 0; i < 20; i++) CHECK(nms_create(&r, bytes, 3, &aliases[i]) == NMS_OK);
    for (unsigned i = 0; i < 100; i++) CHECK(nms_string_array_append(&r, array, i & 1 ? 1 : child) == NMS_OK);
    CHECK(nms_string_array_get(&r, array, UINT64_MAX, &held) == NMS_OK && held == 0);
    CHECK(nms_string_array_get(&r, array, 0, &held) == NMS_OK && held == child);
    CHECK(nms_release(&r, array) == NMS_OK);
    CHECK(nms_string_array_length(&r, array, &length) == NMS_OK && length == 102);
    CHECK(nms_release(&r, array) == NMS_OK);
    CHECK(nms_view(&r, held, &view) == NMS_OK && view.length == 3 && view.data[1] == 0 && view.data[2] == 255);
    CHECK(nms_release(&r, held) == NMS_OK);
    for (unsigned i = 0; i < 20; i++) CHECK(nms_release(&r, aliases[i]) == NMS_OK);
    CHECK(!r.live_objects && !r.live_bytes);
    /* Independent instances and terminal disposal with unreleased child edges. */
    CHECK(nms_string_array_create(&other, &array) == NMS_OK);
    CHECK(nms_create(&other, bytes, 3, &child) == NMS_OK);
    CHECK(nms_string_array_append(&other, array, child) == NMS_OK);
    CHECK(nms_string_array_append(&other, array, child) == NMS_OK);
    CHECK(nms_dispose(&other) == NMS_OK && !other.live_objects && !other.live_bytes);
    CHECK(nms_string_array_create(&other, &held) == NMS_DISPOSED);
    CHECK(nms_dispose(&r) == NMS_OK);
    return 0;
}
#ifdef NMS_TESTING
int nms_array_failures(void) {
    NmsRuntime r; nms_init(&r, literals, 1);
    NmsHandle array = 999, child, out = 888, fill[7];
    nms_test_fail_after(&r, 0);
    CHECK(nms_string_array_create(&r, &array) == NMS_MEMORY && array == 999);
    CHECK(!r.capacity && !r.live_objects && !nms_test_live_allocations());
    nms_test_fail_after(&r, UINT64_MAX);
    CHECK(nms_string_array_create(&r, &array) == NMS_OK);
    CHECK(nms_create(&r, bytes, 3, &child) == NMS_OK);
    uint64_t allocations = nms_test_live_allocations();
    nms_test_fail_after(&r, 0);
    CHECK(nms_string_array_append(&r, array, child) == NMS_MEMORY);
    CHECK(r.slots[(uint32_t)child].references == 1 && !r.slots[(uint32_t)array].length);
    CHECK(nms_test_live_allocations() == allocations);
    nms_test_fail_after(&r, UINT64_MAX);
    for (unsigned i = 0; i < 4; i++) CHECK(nms_string_array_append(&r, array, child) == NMS_OK);
    allocations = nms_test_live_allocations();
    unsigned char *buffer = r.slots[(uint32_t)array].data;
    nms_test_fail_after(&r, 0);
    CHECK(nms_string_array_append(&r, array, child) == NMS_MEMORY);
    CHECK(r.slots[(uint32_t)array].data == buffer && r.slots[(uint32_t)array].length == 4);
    CHECK(r.slots[(uint32_t)child].references == 5 && nms_test_live_allocations() == allocations);
    nms_test_fail_after(&r, UINT64_MAX);
    for (unsigned i = 0; i < 6; i++) CHECK(nms_string_array_create(&r, &fill[i]) == NMS_OK);
    nms_test_fail_after(&r, 0);
    CHECK(nms_string_array_create(&r, &out) == NMS_MEMORY && out == 888 && r.capacity == 8);
    CHECK(nms_dispose(&r) == NMS_OK && !nms_test_live_allocations());
    return 0;
}
int nms_array_pressure(void) {
#ifdef __wasm32__
    NmsRuntime r; nms_init(&r, literals, 1);
    NmsHandle array, child;
    CHECK(nms_string_array_create(&r, &array) == NMS_OK);
    CHECK(nms_create(&r, bytes, 3, &child) == NMS_OK);
    uint32_t count = 0;
    NmsStatus status = NMS_OK;
    while (count < 131072) {
        status = nms_string_array_append(&r, array, child);
        if (status != NMS_OK) break;
        count++;
    }
    CHECK(status == NMS_MEMORY && count > 1024);
    CHECK(r.slots[(uint32_t)array].length == count);
    CHECK(r.slots[(uint32_t)child].references == (uint64_t)count + 1);
    CHECK(nms_release(&r, array) == NMS_OK);
    CHECK(r.slots[(uint32_t)child].references == 1);
    CHECK(nms_release(&r, child) == NMS_OK);
    CHECK(nms_dispose(&r) == NMS_OK && !nms_test_live_allocations());
#endif
    return 0;
}
int nms_array_reuse(void) {
    uint64_t pages = 0;
    for (unsigned round = 0; round < 100; round++) {
        NmsRuntime r; nms_init(&r, literals, 1);
        NmsHandle array, child;
        CHECK(nms_string_array_create(&r, &array) == NMS_OK);
        CHECK(nms_create(&r, bytes, 3, &child) == NMS_OK);
        for (unsigned i = 0; i < 1024; i++) CHECK(nms_string_array_append(&r, array, child) == NMS_OK);
        CHECK(nms_release(&r, child) == NMS_OK);
        CHECK(nms_release(&r, array) == NMS_OK && !r.live_bytes && !r.live_objects);
        CHECK(nms_dispose(&r) == NMS_OK && !nms_test_live_allocations());
        if (!round) pages = nms_test_memory_pages();
        CHECK(pages == nms_test_memory_pages());
    }
    return 0;
}
#endif
#ifndef __wasm32__
int main(void) {
    int result = nms_array_tests();
#ifdef NMS_TESTING
    if (!result) result = nms_array_failures();
    if (!result) result = nms_array_reuse();
    if (!result) result = nms_array_pressure();
#endif
    return result;
}
#endif
