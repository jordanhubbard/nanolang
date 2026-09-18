/* I exercise private module ABI helpers, not emitted frame cleanup. */
#include "../../src/nanoisa/managed_module.c"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
int nms_module_tests(void) {
    static const unsigned char bytes[] = {'a', 0, 'z'};
    static const unsigned char empty[] = {0};
    static const NmsView literals[] = {{bytes, 3}, {empty, 0}};
    CHECK(nms_module_begin(literals, 2) == NMS_OK);
    uint64_t handle = nms_module_concat(1, 1);
    CHECK(handle && nms_module_status() == NMS_OK && nms_module_length(handle) == 6);
    CHECK(nms_module_retain(handle, 5) == NMS_OK);
    CHECK(nms_module_order(handle, 1) > 0);
    nms_module_release(handle, 5);
    nms_module_fail(NMS_ASSERT);
    nms_module_fail(NMS_TYPE);
    CHECK(nms_module_status() == NMS_ASSERT);
    CHECK(nms_module_begin(literals, 2) == NMS_BUSY);
    CHECK(nms_module_status() == NMS_ASSERT);
    CHECK(nms_module_dispose() == NMS_BUSY);
    CHECK(nms_module_finish(42) == ((uint64_t)NMS_ASSERT << 32));
    /* A retained root survives an invocation error, just as globals must. */
    CHECK(nms_module_begin(literals, 2) == NMS_OK && !nms_module_status());
    CHECK(nms_module_length(handle) == 6);
    uint64_t another = nms_module_concat(handle, 2);
    CHECK(another && nms_module_length(another) == 6);
    nms_module_release(another, 5);
    nms_module_release(0, 0);
    CHECK(!nms_module_status());
    CHECK(nms_module_finish(-3) == (uint32_t)-3);
    CHECK(nms_module_dispose() == NMS_OK && nms_module_dispose() == NMS_OK);
    CHECK(nms_module_begin(literals, 2) == NMS_DISPOSED);
    return 0;
}
int nms_module_dispose_first(void) {
    CHECK(nms_module_dispose() == NMS_OK);
    CHECK(nms_module_dispose() == NMS_OK);
    CHECK(nms_module_begin(NULL, 0) == NMS_DISPOSED);
    return 0;
}
#ifndef __wasm32__
int main(int argc, char **argv) {
    (void)argv;
    return argc > 1 ? nms_module_dispose_first() : nms_module_tests();
}
#endif
