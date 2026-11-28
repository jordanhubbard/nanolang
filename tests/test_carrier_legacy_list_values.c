/* I distinguish malloc record lists from managed DynArray roots explicitly. */
#include "eval/carrier_values.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "I fail %s at line %d.\n", #c, __LINE__); exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static void free_record(Value record) {
    StructValue *r = record.as.struct_val;
    for (int i = 0; i < r->field_count; ++i) {
        Value value = r->field_values[i];
        if (value.type == VAL_STRING) gc_release(value.as.string_val);
        if (value.type == VAL_FUNCTION) {
            free(value.as.function_val.function_name);
            free_function_signature(value.as.function_val.signature);
        }
    }
    release_struct_carrier_names(r); free(r->field_values); free(r);
}
static void owned_or_transferred(bool transfer) {
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Type argument = TYPE_INT;
    Value input[] = {create_string("kept"), create_function("legacy_callback",
        create_function_signature(&argument, 1, TYPE_INT))};
    char *names[] = {"text", "action"};
    Value record = create_struct("LegacyRecord", names, input, 2);
    gc_release(input[0].as.string_val);
    free(input[1].as.function_val.function_name);
    free_function_signature(input[1].as.function_val.signature);
    CHECK(nl_carrier_values_adopt(heap, nl_carrier_value(record)));
    List_int *list = list_int_with_capacity(0);
    for (int i = 0; i < 32; ++i) list_int_push(list, (intptr_t)record.as.struct_val);
    NlCarrierRef root = nl_carrier_legacy_record_list(list);
    CHECK(nl_carrier_values_adopt(heap, root));
    CHECK(!nl_carrier_values_adopt(heap, root));
    NlCarrierRef logical = root; logical.record_list = true;
    Value encoded = create_int((intptr_t)list);
    char *parent_names[] = {"items"};
    Value parent = create_struct("LegacyEnvelope", parent_names, &encoded, 1);
    CHECK(nl_carrier_values_adopt(heap, nl_carrier_value(parent)));
    CHECK(nl_carrier_values_edges(heap, nl_carrier_value(parent), &logical, 1));
    NlCarrierRef parent_root = nl_carrier_value(parent);
    CHECK(nl_carrier_values_collect(heap, &parent_root, 1));
    int length = list->length; list->length = list->capacity + 1;
    CHECK(!nl_carrier_values_collect(heap, &parent_root, 1));
    CHECK(nl_carrier_values_stats(heap).live == 3);
    list->length = length;
    CHECK(nl_carrier_values_collect(heap, &parent_root, 1));
    CHECK(nl_carrier_values_stats(heap).live == 3);
    CHECK((StructValue *)(intptr_t)list_int_get(list, 0) == record.as.struct_val);
    CHECK(!strcmp(record.as.struct_val->field_values[0].as.string_val, "kept"));
    CHECK(!strcmp(record.as.struct_val->field_values[1].as.function_val.function_name, "legacy_callback"));
    if (transfer) {
        CHECK(nl_carrier_values_transfer(heap, &parent_root, 1));
        CHECK(nl_carrier_values_stats(heap).transferred == 3);
        NlCarrierRef borrowed[] = {root, logical};
        CHECK(nl_carrier_values_collect(heap, borrowed, 2));
        CHECK(nl_carrier_values_stats(heap).registered == 3);
        CHECK(nl_carrier_values_free(heap));
        CHECK(!strcmp(record.as.struct_val->field_values[0].as.string_val, "kept"));
        free_record(parent); free_record(record); list_int_free(list);
    } else {
        NlCarrierRef integer = nl_carrier_value(create_int((intptr_t)list));
        CHECK(nl_carrier_values_collect(heap, &integer, 1));
        CHECK(nl_carrier_values_stats(heap).live == 0);
        CHECK(nl_carrier_values_stats(heap).destroyed == 3);
        CHECK(nl_carrier_values_free(heap));
    }
}
static void borrowed_layout_reuse(void) {
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    union { List_int legacy; DynArray dynamic; } storage = {0};
    NlCarrierRef root = nl_carrier_legacy_record_list(&storage.legacy);
    NlCarrierRef logical = root; logical.record_list = true;
    NlCarrierRef roots[] = {root, logical};
    CHECK(nl_carrier_values_collect(heap, roots, 2));
    memset(&storage, 0, sizeof storage);
    StructValue caller = {0}; caller.struct_name = "caller";
    int64_t data = (intptr_t)&caller;
    storage.dynamic.length = storage.dynamic.capacity = 1;
    storage.dynamic.elem_type = ELEM_INT; storage.dynamic.elem_size = sizeof data;
    storage.dynamic.data = &data;
    root = nl_carrier_record_list(&storage.dynamic);
    CHECK(nl_carrier_values_collect(heap, &root, 1));
    CHECK(nl_carrier_values_stats(heap).registered == 0);
    root.record_list = 255;
    CHECK(!nl_carrier_values_collect(heap, &root, 1));
    root.value = create_int(0);
    CHECK(!nl_carrier_values_collect(heap, &root, 1));
    CHECK(nl_carrier_values_free(heap));
    CHECK(!strcmp(caller.struct_name, "caller") && data == (intptr_t)&caller);
}
int main(void) {
    typedef struct { Value value; bool record_list; } PreviousRef;
    CHECK(sizeof(NlCarrierRef) == sizeof(PreviousRef));
    CHECK(offsetof(NlCarrierRef, record_list) == offsetof(PreviousRef, record_list));
    CHECK(sizeof(((NlCarrierRef *)0)->record_list) == 1);
    owned_or_transferred(false);
    owned_or_transferred(true);
    borrowed_layout_reuse();
    CHECK(gc_get_stats().num_objects == 0 && gc_get_stats().current_usage == 0);
    puts("I preserve legacy list growth, graph transfer, caller lifetime, root-layout reuse, malformed refusal and ordinary-INT nonownership.");
    return 0;
}
