/* I compile this private target runtime as one module for LLVM packaging.
 * It is not linked into the VM or the public compiler host modules. */
#include "managed_strings.c"

/* My emitted constant descriptor adapter uses {pointer, uint32}. */
_Static_assert(offsetof(NmsView, data) == 0, "I require pointer-first string views");
_Static_assert(offsetof(NmsView, length) == sizeof(void *), "I require the declared view ABI");
_Static_assert(sizeof(NmsView) == 2 * sizeof(void *), "I require the declared view stride");
_Static_assert(offsetof(NmsRecordDescriptor, global_layout_index) == 0 &&
               offsetof(NmsRecordDescriptor, field_count) == 4 &&
               sizeof(NmsRecordDescriptor) == 8,
               "I require two exact u32 record descriptor fields");
_Static_assert(NMS_RECORD_TAG == 8, "I require the record handle tag ABI");
#define NMS_MODULE_RECORD_FIELDS 256u


static NmsRuntime nms_module_instance;
static NmsStatus nms_module_error;
static unsigned nms_module_ready;

void nms_module_fail(uint32_t status) {
    if (!nms_module_error && status)
        nms_module_error = status <= NMS_BOUNDS ? (NmsStatus)status : NMS_STATE;
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
/* Private graph adapters: I keep the ordinary leaf entry ABI unchanged. */
uint32_t nms_module_graph_begin(const NmsView *literals, uint32_t count) {
    NmsStatus status = (NmsStatus)nms_module_begin(literals, count);
    if (status != NMS_OK) return status;
    status = nms_prepare_collection(&nms_module_instance);
    nms_module_fail(status);
    return status;
}
/* I return status in the low word and acquisition in the high word. This
 * differs deliberately from the public try-entry result/status packing.
 * Descriptor binding is inactive and once-only; preparation follows begin. */
uint64_t nms_module_record_begin(const NmsView *literals, uint32_t literal_count,
                                  const NmsRecordDescriptor *records, uint32_t record_count) {
    if (!nms_module_ready) {
        nms_init(&nms_module_instance, literals, literal_count);
        nms_module_ready = 1;
    }
    if (nms_module_instance.disposed) return NMS_DISPOSED;
    if (nms_module_instance.active) return NMS_BUSY;
    if (nms_module_instance.literals != literals ||
        nms_module_instance.literal_count != literal_count) return NMS_STATE;
    NmsStatus status;
    if (!nms_module_instance.records_bound) {
        status = nms_bind_records(&nms_module_instance, records, record_count);
        if (status != NMS_OK) return status;
    } else if (nms_module_instance.record_descriptors != records ||
               nms_module_instance.record_count != record_count) return NMS_STATE;
    status = nms_begin(&nms_module_instance);
    if (status != NMS_OK) return status;
    nms_module_error = NMS_OK;
    status = nms_prepare_collection(&nms_module_instance);
    nms_module_fail(status);
    return (UINT64_C(1) << 32) | (uint32_t)status;
}
uint32_t nms_module_graph_collect(void) {
    if (!nms_module_ready || !nms_module_instance.active) return NMS_STATE;
    nms_module_fail(nms_collect_prepared(&nms_module_instance));
    return nms_module_error;
}
uint64_t nms_module_graph_finish(int32_t result) {
    if (!nms_module_ready || !nms_module_instance.active) return (uint64_t)NMS_STATE << 32;
    /* Preparation may have failed; there can be no graph allocation in that
     * failed entry. Preserve its error while ending the active entry. */
    if (nms_module_instance.collection_prepared)
        nms_module_fail(nms_collect_prepared(&nms_module_instance));
    else nms_module_fail(NMS_STATE);
    return nms_module_finish(result);
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
    NmsStatus status = (tag == 5 || tag == NMS_ARRAY_TAG || tag == NMS_RECORD_TAG) ? nms_retain(&nms_module_instance, payload) : NMS_OK;
    nms_module_fail(status);
    return status;
}
void nms_module_release(uint64_t payload, uint32_t tag) {
    if (tag == 5 || tag == NMS_ARRAY_TAG || tag == NMS_RECORD_TAG) nms_module_fail(nms_release(&nms_module_instance, payload));
}
/* I borrow counted constructor roots. My private vector does not alias a
 * relocating slot table and has the same explicit bound as the future emitter. */
uint64_t nms_module_record_literal(uint32_t ordinal, uint32_t count,
                                   const uint64_t *payloads, const uint32_t *tags) {
    if (count > NMS_MODULE_RECORD_FIELDS) { nms_module_fail(NMS_TYPE); return 0; }
    if (count && (!payloads || !tags)) { nms_module_fail(NMS_STATE); return 0; }
    NmsValue values[NMS_MODULE_RECORD_FIELDS];
    for (uint32_t i = 0; i < count; i++) values[i] = (NmsValue){payloads[i], tags[i]};
    NmsHandle result = 0;
    nms_module_fail(nms_record_create(&nms_module_instance, ordinal, values, count, &result));
    return result;
}
/* STRUCT reports TYPE for wrong receivers; AGG reports unavailable/BOUNDS.
 * Both accessors borrow their inputs. GET publishes one retained owner only
 * on success; the future emitter consumes originals exactly once. */
static NmsStatus nms_module_record_receiver(uint64_t record, uint32_t tag, uint32_t aggregate) {
    if (aggregate > 1) return NMS_STATE;
    if (tag != NMS_RECORD_TAG) return aggregate ? NMS_BOUNDS : NMS_TYPE;
    uint32_t ordinal, layout;
    NmsStatus status = nms_record_identity(&nms_module_instance, record, &ordinal, &layout);
    return aggregate && status == NMS_TYPE ? NMS_BOUNDS : status;
}
uint32_t nms_module_record_get_value(uint64_t record, uint32_t receiver_tag,
                                     uint32_t field, uint32_t aggregate,
                                     uint64_t *bits, uint32_t *tag) {
    NmsValue result = {0, 0};
    NmsStatus status = bits && tag ? nms_module_record_receiver(record, receiver_tag, aggregate) : NMS_STATE;
    if (status == NMS_OK)
        status = nms_record_get(&nms_module_instance, record, field, &result);
    if (status == NMS_OK) { *bits = result.payload; *tag = result.tag; }
    nms_module_fail(status);
    return status;
}
uint32_t nms_module_record_set_value(uint64_t record, uint32_t receiver_tag,
                                     uint32_t field, uint32_t aggregate,
                                     uint64_t bits, uint32_t tag) {
    NmsStatus status = nms_module_record_receiver(record, receiver_tag, aggregate);
    if (status == NMS_OK)
        status = nms_record_set(&nms_module_instance, record, field, (NmsValue){bits, tag});
    nms_module_fail(status);
    return status;
}
uint64_t nms_module_format_scalar(uint64_t bits, uint32_t tag) {
    NmsHandle result = 0;
    nms_module_fail(nms_format_scalar(&nms_module_instance, bits, tag, &result));
    return result;
}
int64_t nms_module_char_at(uint64_t source, uint64_t index_bits, uint32_t is_integer) {
    int64_t result = 0;
    nms_module_fail(nms_char_at(&nms_module_instance, source, index_bits, is_integer, &result));
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
uint64_t nms_module_split(uint64_t source, uint64_t delimiter) {
    NmsHandle result = 0;
    nms_module_fail(nms_split_owned(&nms_module_instance, source, delimiter, &result));
    return result;
}
/* My value accessors borrow arguments; split consumes its two string owners. */
/* I borrow scalar input arrays and source owners; emitted transfer comes later. */
uint64_t nms_module_array_literal(uint32_t tag, uint32_t count,
                                  const uint64_t *payloads, const uint32_t *tags) {
    NmsHandle result = 0;
    nms_module_fail(nms_vm_array_literal(&nms_module_instance, tag, payloads, tags, count, &result));
    return result;
}
uint64_t nms_module_array_slice(uint64_t source, uint64_t start_bits, uint32_t start_tag,
                                uint64_t end_bits, uint32_t end_tag) {
    uint32_t length = 0;
    NmsStatus status = nms_value_array_length(&nms_module_instance, source, &length);
    NmsHandle result = 0;
    if (status == NMS_OK) {
        uint32_t start = start_tag == 1 ? (uint32_t)start_bits : 0;
        uint32_t end = end_tag == 1 ? (uint32_t)end_bits : length;
        status = nms_vm_array_slice(&nms_module_instance, source, start, end, &result);
    }
    nms_module_fail(status);
    return result;
}
uint64_t nms_module_array_create(uint32_t tag) {
    NmsHandle result = 0;
    nms_module_fail(nms_vm_array_create(&nms_module_instance, tag, &result));
    return result;
}
uint64_t nms_module_split_values(uint64_t source, uint64_t delimiter) {
    NmsHandle result = 0;
    nms_module_fail(nms_split_values_owned(&nms_module_instance, source, delimiter, &result));
    return result;
}
uint32_t nms_module_array_append_value(uint64_t array, uint64_t bits, uint32_t tag) {
    NmsStatus status = nms_value_array_append(&nms_module_instance, array, (NmsValue){bits, tag});
    nms_module_fail(status);
    return status;
}
uint32_t nms_module_array_set_value(uint64_t array, uint64_t index, uint64_t bits, uint32_t tag) {
    uint32_t length = 0;
    NmsStatus status = nms_value_array_length(&nms_module_instance, array, &length);
    if (status == NMS_OK && index >= length) status = NMS_BOUNDS;
    if (status == NMS_OK)
        status = nms_value_array_set(&nms_module_instance, array, index, (NmsValue){bits, tag});
    nms_module_fail(status);
    return status;
}
uint32_t nms_module_array_get_value(uint64_t array, uint64_t index, uint64_t *bits, uint32_t *tag) {
    NmsValue result = {0, 0};
    NmsStatus status = bits && tag ? nms_value_array_get(&nms_module_instance, array, index, &result) : NMS_STATE;
    if (status == NMS_OK) { *bits = result.payload; *tag = result.tag; }
    nms_module_fail(status);
    return status;
}
uint32_t nms_module_array_pop_value(uint64_t array, uint64_t *bits, uint32_t *tag) {
    NmsValue result = {0, 0};
    NmsStatus status = bits && tag ? nms_value_array_pop(&nms_module_instance, array, &result) : NMS_STATE;
    if (status == NMS_OK) { *bits = result.payload; *tag = result.tag; }
    nms_module_fail(status);
    return status;
}
uint64_t nms_module_array_get(uint64_t array, uint64_t index) {
    NmsHandle result = 0;
    nms_module_fail(nms_string_array_get(&nms_module_instance, array, index, &result));
    return result;
}
uint32_t nms_module_array_length(uint64_t array) {
    uint32_t result = 0;
    nms_module_fail(nms_string_array_length(&nms_module_instance, array, &result));
    return result;
}
uint32_t nms_module_array_value_length(uint64_t array) {
    uint32_t result = 0;
    nms_module_fail(nms_value_array_length(&nms_module_instance, array, &result));
    return result;
}
uint64_t nms_module_replace(uint64_t source, uint64_t needle, uint64_t replacement) {
    NmsHandle result = 0;
    nms_module_fail(nms_replace_owned(&nms_module_instance, source, needle, replacement, &result));
    return result;
}
uint64_t nms_module_case(uint64_t source, uint32_t upper) {
    NmsHandle result = 0;
    nms_module_fail(nms_case_owned(&nms_module_instance, source, upper, &result));
    return result;
}
uint64_t nms_module_trim(uint64_t source) {
    NmsHandle result = 0;
    nms_module_fail(nms_trim_owned(&nms_module_instance, source, &result));
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
