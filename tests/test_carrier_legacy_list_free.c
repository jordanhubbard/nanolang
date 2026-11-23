/* I retire explicit list payloads while tracing only live record storage. */
#include "eval/carrier_values.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define NANO_LIST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(NANO_LIST_ASAN)
#include <sanitizer/asan_interface.h>
#endif
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "I fail %s at line %d.\n", #c, __LINE__); exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static List_int *recycle;
static bool model;
static size_t frees;
void nano_test_list_int_free(List_int *list) {
    if (!list) { list_int_free(list); return; }
    frees++;
    if (!model) { list_int_free(list); return; }
    /* My separate lifetime model keeps only its arena header, not payloads.
     * My ordinary cases above/below this model call the real native free. */
    CHECK(!recycle);
    free(list->data); list->data = NULL; list->length = list->capacity = 0;
    recycle = list;
#if defined(__SANITIZE_ADDRESS__) || defined(NANO_LIST_ASAN)
    __asan_poison_memory_region(list, sizeof *list);
#endif
}
static List_int *new_lifetime(void) {
    CHECK(recycle);
    List_int *list = recycle; recycle = NULL;
#if defined(__SANITIZE_ADDRESS__) || defined(NANO_LIST_ASAN)
    __asan_unpoison_memory_region(list, sizeof *list);
#endif
    memset(list, 0, sizeof *list);
    return list;
}
static Value record(void) {
    Value text = create_string("copied"); char *name = "text";
    Value result = create_struct("ConsumedItem", &name, &text, 1);
    gc_release(text.as.string_val); return result;
}
static void free_record(Value value) {
    StructValue *r = value.as.struct_val;
    for (int i = 0; i < r->field_count; ++i)
        if (r->field_values[i].type == VAL_STRING) gc_release(r->field_values[i].as.string_val);
    release_struct_carrier_names(r); free(r->field_values); free(r);
}
static NlCarrierRef logical(List_int *list) {
    NlCarrierRef result = nl_carrier_legacy_record_list(list);
    result.record_list = NL_CARRIER_RECORD_LIST; return result;
}
static void graph_and_failure(void) {
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value child = record(); CHECK(nl_carrier_values_adopt(heap, nl_carrier_value(child)));
    List_int *list = list_int_new(); list_int_push(list, (intptr_t)child.as.struct_val);
    NlCarrierRef root = nl_carrier_legacy_record_list(list);
    CHECK(nl_carrier_values_adopt(heap, root));
    char *name = "items"; Value address = create_int((intptr_t)list);
    Value parent = create_struct("ConsumedBox", &name, &address, 1);
    NlCarrierRef parent_root = nl_carrier_value(parent), edge = logical(list);
    CHECK(nl_carrier_values_adopt(heap, parent_root));
    CHECK(nl_carrier_values_edges(heap, parent_root, &edge, 1));
    CHECK(nl_carrier_values_collect(heap, &parent_root, 1));
    size_t before = frees;
    CHECK(nl_carrier_values_consume_legacy_list(heap, list));
    CHECK(frees == before + 1);
    CHECK(nl_carrier_values_stats(heap).destroyed == 1 && nl_carrier_values_stats(heap).live == 2);
    CHECK(!nl_carrier_values_consume_legacy_list(heap, list));
    CHECK(!nl_carrier_values_contains(heap, root));
    NlCarrierRef roots[] = {parent_root, root, edge, nl_carrier_value(child)};
    CHECK(nl_carrier_values_collect(heap, roots, 4));
    CHECK(nl_carrier_values_consumed_count(heap) == 1);
    CHECK(!strcmp(child.as.struct_val->field_values[0].as.string_val, "copied"));
    CHECK(!nl_carrier_values_transfer(heap, &parent_root, 1));
    CHECK(nl_carrier_values_stats(heap).transferred == 0);
    NlCarrierRef malformed = root; malformed.record_list = 255;
    CHECK(!nl_carrier_values_collect(heap, &malformed, 1));
    CHECK(nl_carrier_values_consumed_count(heap) == 1);
    CHECK(nl_carrier_values_collect(heap, roots, 4));
    CHECK(nl_carrier_values_stats(heap).live == 2);
    CHECK(nl_carrier_values_collect(heap, roots + 3, 1));
    CHECK(nl_carrier_values_consumed_count(heap) == 0);
    CHECK(nl_carrier_values_stats(heap).live == 1);
    CHECK(nl_carrier_values_free(heap));
}
static void borrowed_and_teardown(bool heap_first) {
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value caller = record(); List_int *list = list_int_new();
    list_int_push(list, (intptr_t)caller.as.struct_val);
    CHECK(nl_carrier_values_begin_list_borrow(heap, nl_carrier_legacy_record_list(list)));
    NlCarrierRef root = nl_carrier_legacy_record_list(list);
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    CHECK(nl_carrier_values_stats(heap).registered == 0);
    CHECK(nl_carrier_values_consume_legacy_list(heap, list));
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    CHECK(nl_carrier_values_stats(heap).destroyed == 0);
    CHECK(!strcmp(caller.as.struct_val->field_values[0].as.string_val, "copied"));
    if (heap_first) { CHECK(nl_carrier_values_free(heap)); free_record(caller); }
    else { free_record(caller); CHECK(nl_carrier_values_free(heap)); }
}
static void exact_address_lifetimes(void) {
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    List_int *list = calloc(1, sizeof(DynArray)); CHECK(list);
    NlCarrierRef root = nl_carrier_legacy_record_list(list);
    CHECK(nl_carrier_values_adopt(heap, root));
    model = true;
    CHECK(nl_carrier_values_consume_legacy_list(heap, list));
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    list = new_lifetime(); Value caller = record();
    list_int_push(list, (intptr_t)caller.as.struct_val);
    CHECK(nl_carrier_values_begin_list_borrow(heap, nl_carrier_legacy_record_list(list)));
    CHECK(nl_carrier_values_consumed_count(heap) == 0);
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    CHECK(nl_carrier_values_stats(heap).registered == 1);
    CHECK(nl_carrier_values_consume_legacy_list(heap, list));
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    list = new_lifetime();
    CHECK(nl_carrier_values_adopt(heap, root));
    CHECK(!nl_carrier_values_begin_list_borrow(heap, nl_carrier_legacy_record_list(list)));
    CHECK(nl_carrier_values_consumed_count(heap) == 0);
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    CHECK(nl_carrier_values_stats(heap).live == 1);
    CHECK(nl_carrier_values_consume_legacy_list(heap, list));
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    DynArray *canonical = (DynArray *)new_lifetime();
    int64_t element = (intptr_t)caller.as.struct_val;
    canonical->length = canonical->capacity = 1;
    canonical->elem_type = ELEM_INT; canonical->elem_size = sizeof element;
    canonical->data = &element;
    CHECK(!nl_carrier_values_begin_list_borrow(heap, nl_carrier_value(create_int((intptr_t)canonical))));
    CHECK(nl_carrier_values_consumed_count(heap) == 1);
    root = nl_carrier_record_list(canonical);
    CHECK(nl_carrier_values_begin_list_borrow(heap, root));
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    CHECK(nl_carrier_values_stats(heap).registered == 2 && nl_carrier_values_stats(heap).destroyed == 2);
    CHECK(nl_carrier_values_consumed_count(heap) == 0);
    model = false;
    CHECK(nl_carrier_values_free(heap));
    CHECK(!strcmp(caller.as.struct_val->field_values[0].as.string_val, "copied"));
    free(canonical); free_record(caller);
}
static void bounded_history(void) {
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    CHECK(nl_carrier_values_consume_legacy_list(heap, NULL));
    for (size_t i = 0; i < 512; ++i) {
        List_int *list = list_int_new();
        NlCarrierRef root = nl_carrier_legacy_record_list(list);
        CHECK(nl_carrier_values_adopt(heap, root));
        CHECK(nl_carrier_values_consume_legacy_list(heap, list));
        CHECK(nl_carrier_values_collect(heap, &root, 1));
        CHECK(nl_carrier_values_consumed_count(heap) == 1);
        NlCarrierRef bits = nl_carrier_value(root.value);
        CHECK(nl_carrier_values_collect(heap, &bits, 1));
        CHECK(nl_carrier_values_consumed_count(heap) == 0);
        CHECK(nl_carrier_values_consumed_capacity(heap) <= 16);
        CHECK(nl_carrier_values_stats(heap).live == 0);
    }
    CHECK(nl_carrier_values_stats(heap).registered == 512);
    CHECK(nl_carrier_values_stats(heap).destroyed == 512);
    CHECK(nl_carrier_values_free(heap));
}
static void refusal_boundaries(void) {
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    List_int caller = {0};
    CHECK(!nl_carrier_values_begin_list_borrow(heap, nl_carrier_value(create_int((intptr_t)&caller))));
    caller.length = 1;
    CHECK(!nl_carrier_values_begin_list_borrow(heap, nl_carrier_legacy_record_list(&caller)));
    caller.length = 0;
    CHECK(nl_carrier_values_begin_list_borrow(heap, nl_carrier_legacy_record_list(&caller)));
    DynArray *managed = dyn_array_new(ELEM_INT); CHECK(managed);
    CHECK(!nl_carrier_values_begin_list_borrow(heap, nl_carrier_legacy_record_list((List_int *)managed)));
    CHECK(!nl_carrier_values_consume_legacy_list(heap, (List_int *)managed));
    CHECK(nl_carrier_values_begin_list_borrow(heap, nl_carrier_record_list(managed)));
    gc_release(managed);
    DynArray *opaque = gc_alloc_opaque(sizeof *opaque, NULL); CHECK(opaque);
    CHECK(!nl_carrier_values_begin_list_borrow(heap, nl_carrier_record_list(opaque)));
    gc_release(opaque);
    Value owned = record(); CHECK(nl_carrier_values_adopt(heap, nl_carrier_value(owned)));
    CHECK(!nl_carrier_values_begin_list_borrow(heap, nl_carrier_legacy_record_list((List_int *)owned.as.struct_val)));
    CHECK(!nl_carrier_values_consume_legacy_list(heap, (List_int *)owned.as.struct_val));
    CHECK(nl_carrier_values_stats(heap).live == 1 && nl_carrier_values_consumed_count(heap) == 0);
    CHECK(nl_carrier_values_free(heap));
    puts("I refuse scalar bits, malformed headers, wrong managed layouts and owned nonlist storage without consuming or adopting caller objects.");
}
int main(void) {
    refusal_boundaries();
    graph_and_failure(); borrowed_and_teardown(false); borrowed_and_teardown(true);
    exact_address_lifetimes(); bounded_history();
    CHECK(!recycle && !model);
    CHECK(gc_get_stats().num_objects == 0 && gc_get_stats().current_usage == 0);
    puts("I preserve immediate payload destruction, copied records, alias/field roots, failure atomicity, borrowed teardown, explicit new lifetimes and 512 bounded retirement rounds.");
    return 0;
}
