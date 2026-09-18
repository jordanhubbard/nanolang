/* I compile this private target runtime as one module for LLVM packaging.
 * It is not linked into the VM or the public compiler host modules. */
#include "managed_strings.c"

/* My emitted constant descriptor adapter uses {pointer, uint32}. */
_Static_assert(offsetof(NmsView, data) == 0, "I require pointer-first string views");
_Static_assert(offsetof(NmsView, length) == sizeof(void *), "I require the declared view ABI");
_Static_assert(sizeof(NmsView) == 2 * sizeof(void *), "I require the declared view stride");

static NmsRuntime nms_module_instance;
static NmsStatus nms_module_error;
static unsigned nms_module_ready;

void nms_module_fail(uint32_t status) {
    if (!nms_module_error && status)
        nms_module_error = status <= NMS_STATE ? (NmsStatus)status : NMS_STATE;
}
uint32_t nms_module_status(void) { return nms_module_error; }
uint32_t nms_module_begin(const NmsView *literals, uint32_t count) {
    if (!nms_module_ready) {
        nms_init(&nms_module_instance, literals, count);
        nms_module_ready = 1;
    }
    NmsStatus status = nms_begin(&nms_module_instance);
    /* A refused nested entry must not erase its suspended caller's error. */
    if (status == NMS_OK) nms_module_error = NMS_OK;
    return status;
}
uint64_t nms_module_finish(int32_t result) {
    return nms_finish(&nms_module_instance, nms_module_error, result);
}
uint32_t nms_module_active(void) { return nms_module_instance.active; }
uint32_t nms_module_dispose(void) {
    if (!nms_module_ready) {
        nms_init(&nms_module_instance, NULL, 0);
        nms_module_ready = 1;
    }
    return nms_dispose(&nms_module_instance);
}
uint32_t nms_module_retain(uint64_t payload, uint32_t tag) {
    NmsStatus status = tag == 5 ? nms_retain(&nms_module_instance, payload) : NMS_OK;
    nms_module_fail(status);
    return status;
}
void nms_module_release(uint64_t payload, uint32_t tag) {
    if (tag == 5) nms_module_fail(nms_release(&nms_module_instance, payload));
}
uint64_t nms_module_format_scalar(uint64_t bits, uint32_t tag) {
    NmsHandle result = 0;
    nms_module_fail(nms_format_scalar(&nms_module_instance, bits, tag, &result));
    return result;
}
uint32_t nms_module_predicate(uint64_t source, uint64_t affix, uint32_t operation) {
    uint32_t answer = 0;
    nms_module_fail(nms_predicate(&nms_module_instance, source, affix, operation, &answer));
    return answer;
}
uint64_t nms_module_parse_f64(uint64_t source) {
    uint64_t result = 0;
    nms_module_fail(nms_parse_f64(&nms_module_instance, source, &result));
    return result;
}
int64_t nms_module_parse_i64(uint64_t source) {
    int64_t result = 0;
    nms_module_fail(nms_parse_i64(&nms_module_instance, source, &result));
    return result;
}
uint64_t nms_module_substr(uint64_t source, uint32_t start, uint32_t length) {
    NmsHandle result = 0;
    nms_module_fail(nms_substr_owned(&nms_module_instance, source, start, length, &result));
    return result;
}
uint64_t nms_module_concat(uint64_t left, uint64_t right) {
    NmsHandle result = 0;
    nms_module_fail(nms_concat_owned(&nms_module_instance, left, right, &result));
    return result;
}
uint64_t nms_module_length(uint64_t handle) {
    NmsView view;
    NmsStatus status = nms_view(&nms_module_instance, handle, &view);
    nms_module_fail(status);
    return status == NMS_OK ? view.length : 0;
}
int64_t nms_module_order(uint64_t left, uint64_t right) {
    NmsView a, b;
    NmsStatus status = nms_view(&nms_module_instance, left, &a);
    if (status == NMS_OK) status = nms_view(&nms_module_instance, right, &b);
    nms_module_fail(status);
    if (status != NMS_OK) return 0;
    uint32_t count = a.length < b.length ? a.length : b.length;
    for (uint32_t i = 0; i < count; i++)
        if (a.data[i] != b.data[i]) return (int64_t)a.data[i] - b.data[i];
    return (int64_t)a.length - b.length;
}
/* I expose read-only accounting to runtime conformance harnesses. Normal Wasm
 * publication exports only entry/status/disposal, not these helper symbols. */
uint64_t nms_module_live_objects(void) { return nms_module_instance.live_objects; }
uint64_t nms_module_live_bytes(void) { return nms_module_instance.live_bytes; }
