#include "../../src/nanovm/vm_ffi_arrays.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static int reject_native, reject_vm_at, vm_allocations, reject_string, reject_snapshot;
static DynArray *native_new(ElementType type, int64_t count) {
    return reject_native ? NULL : dyn_array_new_with_capacity(type, count);
}
static VmArray *vm_new(VmHeap *heap, uint8_t tag, uint32_t count) {
    return (++vm_allocations == reject_vm_at) ? NULL : vm_array_new(heap, tag, count);
}
static VmString *string_new(VmHeap *heap, const char *text, uint32_t length) {
    return reject_string ? NULL : vm_string_new(heap, text, length);
}
static void *snapshot_new(size_t bytes) { return reject_snapshot ? NULL : malloc(bytes); }
#define dyn_array_new_with_capacity native_new
#define vm_array_new vm_new
#define vm_string_new string_new
#define malloc snapshot_new
#include "../../src/nanovm/vm_ffi_arrays.c"
#undef dyn_array_new_with_capacity
#undef vm_array_new
#undef vm_string_new
#undef malloc

static void integer_aliases(VmHeap *heap) {
    VmArray *array = vm_array_new(heap, TAG_INT, 8);
    assert(array && vm_array_push(heap, array, val_int(7)));
    for (int repeat = 0; repeat < 1000; ++repeat) {
        VmFfiArrayFrame frame = {.heap = heap};
        void *one, *two;
        char error[256];
        assert(vm_ffi_array_argument(&frame, val_array(array), &one, error, sizeof error));
        assert(vm_ffi_array_argument(&frame, val_array(array), &two, error, sizeof error));
        assert(one == two && frame.count == 1 && gc_get_stats().num_objects == 1);
        DynArray *native = one;
        if (!repeat) for (int i = 0; i < 20; ++i) dyn_array_push_int(native, i);
        ((int64_t *)native->data)[0] = repeat;
        assert(vm_ffi_arrays_commit(&frame, error, sizeof error));
        assert(array->length == 21 && vm_array_get(array, 0).as.i64 == repeat);
        NanoValue result;
        assert(vm_ffi_array_alias_result(&frame, one, &result));
        assert(result.as.array == array);
        vm_ffi_arrays_dispose(&frame);
        assert(gc_get_stats().num_objects == 0);
        vm_release(heap, result);
        assert(array->header.ref_count == 1);
    }
    vm_release(heap, val_array(array));
}

static void strings(VmHeap *heap) {
    VmArray *array = vm_array_new(heap, TAG_STRING, 8);
    VmString *original = vm_string_new(heap, "old", 3);
    assert(array && original && vm_array_push(heap, array, val_string(original)));
    VmFfiArrayFrame frame = {.heap = heap};
    void *pointer;
    char error[256];
    assert(vm_ffi_array_argument(&frame, val_array(array), &pointer, error, sizeof error));
    DynArray *native = pointer;
    char *snapshot = ((char **)native->data)[0];
    assert(snapshot != original->data);
    snapshot[0] = 'b';
    assert(!strcmp(original->data, "old"));
    dyn_array_push_string(native, "new");
    dyn_array_push_string(native, NULL);
    assert(vm_ffi_arrays_commit(&frame, error, sizeof error));
    vm_ffi_arrays_dispose(&frame);
    assert(array->length == 3);
    assert(!strcmp(vm_array_get(array, 0).as.string->data, "bld"));
    assert(!strcmp(vm_array_get(array, 1).as.string->data, "new"));
    assert(!strcmp(vm_array_get(array, 2).as.string->data, ""));
    assert(!strcmp(original->data, "old"));
    vm_release(heap, val_string(original));
    vm_release(heap, val_array(array));
}

static void scalar_types(VmHeap *heap) {
    const uint8_t tags[] = {TAG_FLOAT, TAG_BOOL, TAG_U8};
    NanoValue values[] = {val_float(2.5), val_bool(true), val_u8(255)};
    for (int i = 0; i < 3; ++i) {
        VmArray *array = vm_array_new(heap, tags[i], 8);
        assert(array && vm_array_push(heap, array, values[i]));
        VmFfiArrayFrame frame = {.heap = heap};
        void *pointer;
        char error[256];
        assert(vm_ffi_array_argument(&frame, val_array(array), &pointer, error, sizeof error));
        DynArray *native = pointer;
        assert(native->length == 1);
        if (i == 0) assert(dyn_array_get_float(native, 0) == 2.5);
        if (i == 1) assert(dyn_array_get_bool(native, 0));
        if (i == 2) assert(dyn_array_get_u8(native, 0) == 255);
        native->length = 0;
        assert(vm_ffi_arrays_commit(&frame, error, sizeof error));
        vm_ffi_arrays_dispose(&frame);
        assert(array->length == 0);
        vm_release(heap, val_array(array));
    }
}

static void failures(VmHeap *heap) {
    VmArray *one = vm_array_new(heap, TAG_INT, 8);
    VmArray *two = vm_array_new(heap, TAG_INT, 8);
    assert(vm_array_push(heap, one, val_int(1)) && vm_array_push(heap, two, val_int(2)));
    for (int failure = 0; failure < 3; ++failure) {
        VmFfiArrayFrame frame = {.heap = heap};
        void *a, *b;
        char error[256];
        assert(vm_ffi_array_argument(&frame, val_array(one), &a, error, sizeof error));
        assert(vm_ffi_array_argument(&frame, val_array(two), &b, error, sizeof error));
        ((int64_t *)((DynArray *)a)->data)[0] = 99;
        if (failure == 0) ((DynArray *)b)->length = -1;
        if (failure == 1) ((DynArray *)b)->elem_type = ELEM_FLOAT;
        if (failure == 2) { vm_allocations = 0; reject_vm_at = 2; }
        assert(!vm_ffi_arrays_commit(&frame, error, sizeof error));
        reject_vm_at = 0;
        assert(vm_array_get(one, 0).as.i64 == 1 && vm_array_get(two, 0).as.i64 == 2);
        vm_ffi_arrays_dispose(&frame);
        vm_gc_collect_cycles(heap);
        assert(gc_get_stats().num_objects == 0 && heap->stats.num_objects == 2);
    }
    VmFfiArrayFrame frame = {.heap = heap};
    void *out;
    char error[256];
    reject_native = 1;
    assert(!vm_ffi_array_argument(&frame, val_array(one), &out, error, sizeof error));
    reject_native = 0;
    vm_ffi_arrays_dispose(&frame);
    assert(one->header.ref_count == 1);
    VmArray *nested = vm_array_new(heap, TAG_ARRAY, 8);
    assert(!vm_ffi_array_argument(&frame, val_array(nested), &out, error, sizeof error));
    assert(!vm_ffi_array_argument(&frame, val_int(1), &out, error, sizeof error));
    vm_ffi_arrays_dispose(&frame);
    vm_release(heap, val_array(nested));
    vm_release(heap, val_array(one));
    vm_release(heap, val_array(two));

    VmArray *texts = vm_array_new(heap, TAG_STRING, 8);
    VmString *text = vm_string_new(heap, "ok", 2);
    assert(vm_array_push(heap, texts, val_string(text)));
    vm_release(heap, val_string(text));
    reject_snapshot = 1;
    assert(!vm_ffi_array_argument(&frame, val_array(texts), &out, error, sizeof error));
    reject_snapshot = 0;
    vm_ffi_arrays_dispose(&frame);
    assert(vm_ffi_array_argument(&frame, val_array(texts), &out, error, sizeof error));
    reject_string = 1;
    assert(!vm_ffi_arrays_commit(&frame, error, sizeof error));
    reject_string = 0;
    vm_ffi_arrays_dispose(&frame);
    assert(texts->header.ref_count == 1 && gc_get_stats().num_objects == 0);
    vm_release(heap, val_array(texts));
}

static void invalid_strings(VmHeap *heap) {
    const char invalid[][4] = {{'a', 0, 'b', 0}, {(char)0xff, 0, 0, 0}};
    const uint32_t lengths[] = {3, 1};
    char error[256];
    for (int i = 0; i < 2; ++i) {
        VmArray *array = vm_array_new(heap, TAG_STRING, 1);
        VmString *text = vm_string_new(heap, invalid[i], lengths[i]);
        assert(array && text && vm_array_push(heap, array, val_string(text)));
        vm_release(heap, val_string(text));
        VmFfiArrayFrame frame = {.heap = heap};
        void *out;
        assert(!vm_ffi_array_argument(&frame, val_array(array), &out, error, sizeof error));
        vm_ffi_arrays_dispose(&frame);
        assert(array->header.ref_count == 1);
        vm_release(heap, val_array(array));
    }
    DynArray *native = dyn_array_new(ELEM_STRING);
    dyn_array_push_string(native, invalid[1]);
    assert(!vm_ffi_array_import(heap, native, error, sizeof error));
    gc_release(native);
    DynArray bad = {.length = 1, .capacity = 1, .elem_type = ELEM_INT,
                    .elem_size = sizeof(int64_t), .data = NULL};
    assert(!vm_ffi_array_import(heap, &bad, error, sizeof error));
    int64_t storage = 1;
    bad.data = &storage;
    bad.capacity = 0;
    assert(!vm_ffi_array_import(heap, &bad, error, sizeof error));
    bad.capacity = 1;
    bad.elem_size = 1;
    assert(!vm_ffi_array_import(heap, &bad, error, sizeof error));
}

int main(void) {
    VmHeap heap;
    vm_heap_init(&heap);
    gc_init();
    integer_aliases(&heap);
    strings(&heap);
    scalar_types(&heap);
    failures(&heap);
    invalid_strings(&heap);
    vm_gc_collect_cycles(&heap);
    assert(gc_get_stats().num_objects == 0 && heap.stats.num_objects == 0);
    gc_shutdown();
    vm_heap_destroy(&heap);
    return 0;
}
