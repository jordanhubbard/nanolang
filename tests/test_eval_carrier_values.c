/* I qualify actual payload ownership before enabling source collection. */
#include "eval/carrier_values.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "I fail carrier value check %s at line %d.\n", #c, __LINE__); exit(1); } } while (0)

int g_argc = 0;
char **g_argv = NULL;

static void *metadata[4096];
static size_t metadata_count, budget = SIZE_MAX, failures;
static bool admit_allocation(void) {
    if (!budget) { failures++; return false; }
    if (budget != SIZE_MAX) budget--;
    return true;
}
static void remember(void *p) {
    if (p) { CHECK(metadata_count < sizeof metadata / sizeof *metadata); metadata[metadata_count++] = p; }
}
void *nano_carrier_test_malloc(size_t size) {
    if (!admit_allocation()) return NULL;
    void *p = malloc(size); remember(p); return p;
}
void *nano_carrier_test_calloc(size_t count, size_t size) {
    if (!admit_allocation()) return NULL;
    void *p = calloc(count, size); remember(p); return p;
}
void *nano_carrier_test_realloc(void *pointer, size_t size) {
    if (!admit_allocation()) return NULL;
    CHECK(size > 0);
    size_t index = 0; while (index < metadata_count && metadata[index] != pointer) index++;
    void *p = realloc(pointer, size);
    if (p) { if (index < metadata_count) metadata[index] = p; else remember(p); }
    return p;
}
void nano_carrier_test_free(void *pointer) {
    for (size_t i = 0; i < metadata_count; ++i) if (metadata[i] == pointer) {
        metadata[i] = metadata[--metadata_count]; break;
    }
    free(pointer);
}
static NlCarrierRef ref(Value v) { return nl_carrier_value(v); }
static Value callback(void) {
    Type argument = TYPE_INT;
    FunctionSignature *signature = create_function_signature(&argument, 1, TYPE_FUNCTION);
    signature->return_fn_sig = create_function_signature(&argument, 1, TYPE_INT);
    return create_function("carrier_callback", signature);
}
static void release_scalar(Value v) {
    if (v.type == VAL_STRING) { if (gc_is_managed(v.as.string_val)) gc_release(v.as.string_val); else free(v.as.string_val); }
    if (v.type == VAL_FUNCTION) { free(v.as.function_val.function_name); free_function_signature(v.as.function_val.signature); }
}
static void release_record(Value v) {
    StructValue *r = v.as.struct_val;
    for (int i = 0; i < r->field_count; ++i) {
        if (r->field_values[i].type == VAL_STRUCT) release_record(r->field_values[i]);
        else release_scalar(r->field_values[i]);
    }
    release_struct_carrier_names(r); free(r->field_values); free(r);
}
static void adopt_record_tree(NlCarrierValues *heap, Value v) {
    CHECK(v.type == VAL_STRUCT); CHECK(nl_carrier_values_adopt(heap, ref(v)));
    for (int i = 0; i < v.as.struct_val->field_count; ++i)
        if (v.as.struct_val->field_values[i].type == VAL_STRUCT)
            adopt_record_tree(heap, v.as.struct_val->field_values[i]);
}
static void stats(NlCarrierValues *heap, size_t registered, size_t destroyed, size_t transferred) {
    NlCarrierHeapStats s = nl_carrier_values_stats(heap);
    CHECK(s.registered == registered && s.destroyed == destroyed && s.transferred == transferred);
    CHECK(s.live == registered - destroyed - transferred);
}
static Value text_record(const char *text) {
    char *names[] = {"text"}; Value item = create_string(text);
    Value record = create_struct("Text", names, &item, 1); release_scalar(item); return record;
}
static void finish(NlCarrierValues *heap, size_t objects) {
    CHECK(nl_carrier_values_free(heap)); CHECK(metadata_count == 0);
    CHECK(gc_get_stats().num_objects == objects);
}
static void creation_and_adoption_failures(void) {
    size_t objects = gc_get_stats().num_objects;
    for (size_t stop = 0; stop < 3; ++stop) {
        size_t before = failures; budget = stop;
        CHECK(!nl_carrier_values_new()); CHECK(failures == before + 1);
        budget = SIZE_MAX; CHECK(metadata_count == 0);
    }
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value caller = text_record("caller retains failed adoption"); size_t baseline = metadata_count;
    for (size_t stop = 0; stop < 2; ++stop) {
        size_t before = failures; budget = stop;
        CHECK(!nl_carrier_values_adopt(heap, ref(caller))); CHECK(failures == before + 1);
        budget = SIZE_MAX;
        CHECK(metadata_count == baseline); stats(heap, 0, 0, 0);
        CHECK(strcmp(caller.as.struct_val->field_values[0].as.string_val, "caller retains failed adoption") == 0);
    }
    CHECK(nl_carrier_values_adopt(heap, ref(caller)));
    CHECK(!nl_carrier_values_adopt(heap, ref(caller)));
    CHECK(nl_carrier_values_collect(heap, NULL, 0)); stats(heap, 1, 1, 0); finish(heap, objects);
    puts("I leave payload ownership with the caller through five actual creation/adoption allocation failures.");
}

static void mixed_constructors(void) {
    size_t objects = gc_get_stats().num_objects;
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    char *leaf_names[] = {"λ", ""}; Value inputs[] = {create_string("caller"), callback()};
    Value leaf = create_struct("Payload", leaf_names, inputs, 2);
    Value array = create_array(VAL_INT, 1, 1); ((long long *)array.as.array_val->data)[0] = 73;
    char *parent_names[] = {"payload", "array"}; Value members[] = {leaf, array};
    Value parent = create_struct("Envelope", parent_names, members, 2);
    Value tuple_items[] = {parent, inputs[0], inputs[1]}; Value tuple = create_tuple(tuple_items, 3);
    char *union_names[] = {"value", "text", "action"};
    Value tagged = create_union("Result", 0, "Ok", union_names, tuple_items, 3);
    adopt_record_tree(heap, parent); CHECK(nl_carrier_values_adopt(heap, ref(array)));
    CHECK(nl_carrier_values_adopt(heap, ref(tuple))); CHECK(nl_carrier_values_adopt(heap, ref(tagged)));
    adopt_record_tree(heap, tagged.as.union_val->field_values[0]);
    stats(heap, 7, 0, 0);
    CHECK(parent.as.struct_val->field_values[0].as.struct_val != leaf.as.struct_val);
    release_record(leaf); release_scalar(inputs[0]); release_scalar(inputs[1]);
    NlCarrierRef root = ref(tuple); CHECK(nl_carrier_values_collect(heap, &root, 1)); stats(heap, 7, 3, 0);
    CHECK(tuple.as.tuple_val->elements[0].as.struct_val == parent.as.struct_val);
    StructValue *payload = parent.as.struct_val->field_values[0].as.struct_val;
    CHECK(strcmp(payload->field_values[0].as.string_val, "caller") == 0);
    CHECK(strcmp(payload->field_names[0], "λ") == 0 && strcmp(payload->field_names[1], "") == 0);
    CHECK(payload->field_values[1].as.function_val.signature->return_fn_sig->param_types[0] == TYPE_INT);
    CHECK(parent.as.struct_val->field_values[1].as.array_val == array.as.array_val);
    CHECK(((long long *)array.as.array_val->data)[0] == 73);
    Value fields[2] = {tuple, create_void()}; StructValue legacy = {0};
    legacy.struct_name = "borrowed"; legacy.field_values = fields; legacy.field_count = 2;
    fields[1].type = VAL_STRUCT; fields[1].as.struct_val = &legacy;
    Value borrowed = create_void(); borrowed.type = VAL_STRUCT; borrowed.as.struct_val = &legacy;
    root = ref(borrowed); CHECK(nl_carrier_values_collect(heap, &root, 1)); stats(heap, 7, 3, 0);
    CHECK(tuple.as.tuple_val->elements[0].as.struct_val == parent.as.struct_val);
    CHECK(nl_carrier_values_collect(heap, NULL, 0)); stats(heap, 7, 7, 0);
    CHECK(strcmp(legacy.struct_name, "borrowed") == 0 && legacy.field_count == 2);
    finish(heap, objects);
    puts("I retire actual nested record/union/tuple copies while preserving shared arrays and borrowed cyclic containers.");
}
static void typed_lists_and_integer_refusal(void) {
    size_t objects = gc_get_stats().num_objects;
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value before = text_record("before"), after = text_record("after");
    DynArray *first = dyn_array_new(ELEM_INT), *second = dyn_array_new(ELEM_INT); CHECK(first && second);
    dyn_array_push_int(first, (int64_t)(intptr_t)before.as.struct_val);
    dyn_array_push_int(first, (int64_t)(intptr_t)before.as.struct_val);
    dyn_array_push_int(second, (int64_t)(intptr_t)before.as.struct_val);
    Value encoded[] = {create_int((long long)(intptr_t)first), create_int((long long)(intptr_t)second)};
    char *name[] = {"first", "second"};
    Value holder = create_struct("Holder", name, encoded, 2);
    CHECK(nl_carrier_values_adopt(heap, ref(before))); CHECK(nl_carrier_values_adopt(heap, ref(after)));
    NlCarrierRef lists[] = {nl_carrier_record_list(first), nl_carrier_record_list(second)};
    CHECK(nl_carrier_values_adopt(heap, lists[0])); CHECK(nl_carrier_values_adopt(heap, lists[1]));
    CHECK(nl_carrier_values_adopt(heap, ref(holder))); CHECK(nl_carrier_values_edges(heap, ref(holder), lists, 2));
    NlCarrierRef roots[] = {ref(holder), ref(after)};
    CHECK(nl_carrier_values_collect(heap, roots, 2)); stats(heap, 5, 0, 0);
    CHECK((StructValue *)(intptr_t)dyn_array_get_int(first, 0) == before.as.struct_val);
    CHECK(dyn_array_get_int(first, 0) == dyn_array_get_int(first, 1));
    CHECK(strcmp(before.as.struct_val->field_values[0].as.string_val, "before") == 0);
    budget = 0; size_t failed = failures;
    CHECK(!nl_carrier_values_edges(heap, ref(holder), lists + 1, 1)); CHECK(failures == failed + 1); budget = SIZE_MAX;
    CHECK(nl_carrier_values_collect(heap, roots, 2)); stats(heap, 5, 0, 0);
    CHECK(nl_carrier_values_edges(heap, ref(holder), lists + 1, 1));
    holder.as.struct_val->field_values[0] = create_int(0);
    CHECK(nl_carrier_values_collect(heap, roots, 2)); stats(heap, 5, 1, 0);
    CHECK(strcmp(before.as.struct_val->field_values[0].as.string_val, "before") == 0);
    CHECK(nl_carrier_values_collect(heap, roots + 1, 1)); stats(heap, 5, 4, 0);
    CHECK(strcmp(after.as.struct_val->field_values[0].as.string_val, "after") == 0);
    finish(heap, objects);

    heap = nl_carrier_values_new(); CHECK(heap);
    before = text_record("integer is not a root"); first = dyn_array_new(ELEM_INT); CHECK(first);
    dyn_array_push_int(first, (int64_t)(intptr_t)before.as.struct_val);
    CHECK(nl_carrier_values_adopt(heap, ref(before)));
    CHECK(nl_carrier_values_adopt(heap, nl_carrier_record_list(first)));
    NlCarrierRef ordinary = ref(create_int((long long)(intptr_t)first));
    CHECK(nl_carrier_values_collect(heap, &ordinary, 1)); stats(heap, 2, 2, 0); finish(heap, objects);

    heap = nl_carrier_values_new(); CHECK(heap); first = dyn_array_new(ELEM_INT); CHECK(first);
    before = text_record("borrowed list"); CHECK(nl_carrier_values_adopt(heap, ref(before)));
    dyn_array_push_int(first, (int64_t)(intptr_t)before.as.struct_val);
    NlCarrierRef borrowed = nl_carrier_record_list(first);
    CHECK(nl_carrier_values_collect(heap, &borrowed, 1)); stats(heap, 1, 0, 0);
    bool popped = false; CHECK(dyn_array_pop_int(first, &popped) == (int64_t)(intptr_t)before.as.struct_val); CHECK(popped);
    CHECK(nl_carrier_values_collect(heap, &borrowed, 1)); stats(heap, 1, 1, 0);
    CHECK(gc_is_managed(first)); gc_release(first); finish(heap, objects);
    puts("I preserve duplicate/multiple typed-list roots and explicit field facts, refuse integer guessing and never adopt a borrowed list.");
}
static void nested_typed_list_payload(void) {
    size_t objects = gc_get_stats().num_objects;
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value caller = text_record("nested list payload"); char *inner_name[] = {"inner"};
    Value tagged = create_union("Payload", 0, "Some", inner_name, &caller, 1);
    Value array = create_array(VAL_INT, 1, 1); ((long long *)array.as.array_val->data)[0] = 91;
    Value action = callback(), label = create_string("independent");
    char *names[] = {"wrapped", "rows", "action", "label"}; Value fields[] = {tagged, array, action, label};
    Value record = create_struct("Complete", names, fields, 4);
    DynArray *list = dyn_array_new(ELEM_INT); CHECK(list);
    dyn_array_push_int(list, (int64_t)(intptr_t)record.as.struct_val);
    CHECK(nl_carrier_values_adopt(heap, ref(record)));
    CHECK(nl_carrier_values_adopt(heap, ref(tagged)));
    adopt_record_tree(heap, tagged.as.union_val->field_values[0]);
    CHECK(nl_carrier_values_adopt(heap, ref(array)));
    NlCarrierRef root = nl_carrier_record_list(list); CHECK(nl_carrier_values_adopt(heap, root));
    release_record(caller); release_scalar(action); release_scalar(label);
    CHECK(nl_carrier_values_collect(heap, &root, 1)); stats(heap, 5, 0, 0);
    CHECK((StructValue *)(intptr_t)dyn_array_get_int(list, 0) == record.as.struct_val);
    CHECK(record.as.struct_val->field_values[0].as.union_val == tagged.as.union_val);
    CHECK(strcmp(tagged.as.union_val->field_values[0].as.struct_val->field_values[0].as.string_val, "nested list payload") == 0);
    CHECK(((long long *)record.as.struct_val->field_values[1].as.array_val->data)[0] == 91);
    CHECK(record.as.struct_val->field_values[2].as.function_val.signature->return_fn_sig->param_types[0] == TYPE_INT);
    CHECK(strcmp(record.as.struct_val->field_values[3].as.string_val, "independent") == 0);
    CHECK(nl_carrier_values_collect(heap, NULL, 0)); stats(heap, 5, 5, 0); finish(heap, objects);
    puts("I preserve nested union/record/STRING/callable/array payloads through an actual typed-list root.");
}
static void managed_tuple_storage(void) {
    size_t objects = gc_get_stats().num_objects;
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value text = create_string("owned managed tuple"); Value tuple = create_tuple(&text, 1);
    free(tuple.as.tuple_val->elements[0].as.string_val);
    tuple.as.tuple_val->elements[0] = text; /* I transfer this caller-owned managed scalar. */
    CHECK(nl_carrier_values_adopt(heap, ref(tuple)));
    CHECK(nl_carrier_values_collect(heap, NULL, 0)); stats(heap, 1, 1, 0); finish(heap, objects);
    puts("I retire caller-created tuple storage containing an owned managed STRING.");
}

static void transfer_and_array_payloads(void) {
    size_t objects = gc_get_stats().num_objects;
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value array = create_array(VAL_INT, 1, 1); ((long long *)array.as.array_val->data)[0] = 39;
    char *names[] = {"items"}; Value tagged = create_union("Items", 0, "Some", names, &array, 1);
    Value record = create_struct("Envelope", names, &tagged, 1);
    CHECK(nl_carrier_values_adopt(heap, ref(array))); CHECK(nl_carrier_values_adopt(heap, ref(tagged)));
    CHECK(nl_carrier_values_adopt(heap, ref(record)));
    NlCarrierRef root = ref(record); CHECK(nl_carrier_values_transfer(heap, &root, 1)); stats(heap, 3, 0, 3);
    CHECK(nl_carrier_values_collect(heap, NULL, 0)); CHECK(nl_carrier_values_free(heap)); CHECK(metadata_count == 0);
    CHECK(record.as.struct_val->field_values[0].as.union_val == tagged.as.union_val);
    CHECK(tagged.as.union_val->field_values[0].as.array_val == array.as.array_val);
    CHECK(((long long *)array.as.array_val->data)[0] == 39);
    release_record(record);
    free(tagged.as.union_val->field_names[0]); free(tagged.as.union_val->field_names);
    free(tagged.as.union_val->field_values); free(tagged.as.union_val->union_name); free(tagged.as.union_val->variant_name); free(tagged.as.union_val);
    free(array.as.array_val->data); free(array.as.array_val);

    heap = nl_carrier_values_new(); CHECK(heap);
    Value strings = create_array(VAL_STRING, 2, 2);
    ((char **)strings.as.array_val->data)[0] = strdup("raw");
    ((char **)strings.as.array_val->data)[1] = create_string("managed").as.string_val;
    Value actions = create_array(VAL_FUNCTION, 2, 2);
    ((Value *)actions.as.array_val->data)[0] = callback(); ((Value *)actions.as.array_val->data)[1] = callback();
    Value cycle = create_array(VAL_ARRAY, 1, 1); ((Value *)cycle.as.array_val->data)[0] = cycle;
    CHECK(nl_carrier_values_adopt(heap, ref(strings))); CHECK(nl_carrier_values_adopt(heap, ref(actions)));
    CHECK(nl_carrier_values_adopt(heap, ref(cycle)));
    root = ref(cycle); CHECK(nl_carrier_values_collect(heap, &root, 1)); stats(heap, 3, 2, 0);
    CHECK(((Value *)cycle.as.array_val->data)[0].as.array_val == cycle.as.array_val);
    CHECK(nl_carrier_values_collect(heap, NULL, 0)); stats(heap, 3, 3, 0); finish(heap, objects);
    puts("I transfer complete actual graphs before caller teardown and retire STRING/FUNCTION arrays and a self-referential static array.");
}
static void legacy_names_and_borrowed_elements(void) {
    size_t objects = gc_get_stats().num_objects;
    NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
    Value empty = create_struct("", NULL, NULL, 0); CHECK(nl_carrier_values_adopt(heap, ref(empty)));
    Value empty_tuple = create_tuple(NULL, 0), empty_union = create_union("Empty", 0, "None", NULL, NULL, 0);
    CHECK(nl_carrier_values_adopt(heap, ref(empty_tuple))); CHECK(nl_carrier_values_adopt(heap, ref(empty_union)));
    DynArray *opaque = gc_alloc_opaque(sizeof(DynArray), NULL); CHECK(opaque);
    memset(opaque, 0, sizeof *opaque); opaque->elem_type = ELEM_INT; opaque->elem_size = sizeof(int64_t);
    CHECK(!nl_carrier_values_adopt(heap, nl_carrier_record_list(opaque))); gc_release(opaque);
    StructValue *legacy = calloc(1, sizeof *legacy); CHECK(legacy);
    legacy->struct_name = strdup("Legacy"); legacy->field_count = 2;
    legacy->field_names = malloc(2 * sizeof(char *)); legacy->field_values = malloc(2 * sizeof(Value));
    CHECK(legacy->field_names && legacy->field_values);
    legacy->field_names[0] = strdup("λ"); legacy->field_names[1] = strdup("");
    legacy->field_values[0] = create_string("owned"); legacy->field_values[1] = callback();
    Value raw = create_void(); raw.type = VAL_STRUCT; raw.as.struct_val = legacy;
    CHECK(nl_carrier_values_adopt(heap, ref(raw))); CHECK(!nl_carrier_values_adopt(heap, ref(raw)));
    Value borrowed = text_record("caller element"), array = create_array(VAL_STRUCT, 1, 1);
    ((StructValue **)array.as.array_val->data)[0] = borrowed.as.struct_val;
    CHECK(nl_carrier_values_adopt(heap, ref(array)));
    CHECK(!nl_carrier_values_adopt(heap, ref(create_int(73))));
    NlCarrierRef invalid = ref(create_bool(true)); invalid.record_list = true;
    CHECK(!nl_carrier_values_collect(heap, &invalid, 1)); CHECK(!nl_carrier_values_transfer(heap, &invalid, 1));
    stats(heap, 5, 0, 0);
    CHECK(nl_carrier_values_collect(heap, NULL, 0)); stats(heap, 5, 5, 0);
    CHECK(strcmp(borrowed.as.struct_val->field_values[0].as.string_val, "caller element") == 0);
    release_record(borrowed); finish(heap, objects);
    puts("I retire packed/legacy/empty names and scalar metadata without owning referenced caller record elements.");
}
static void allocation_rollback(void) {
    size_t objects = gc_get_stats().num_objects; size_t injected = 0;
    for (int transfer = 0; transfer < 2; ++transfer) for (size_t stop = 0; stop < 9; ++stop) {
        NlCarrierValues *heap = nl_carrier_values_new(); CHECK(heap);
        Value fields[128];
        for (size_t i = 0; i < 128; ++i) { fields[i] = text_record("failure survivor"); CHECK(nl_carrier_values_adopt(heap, ref(fields[i]))); }
        StructValue legacy = {0}; legacy.struct_name = "borrowed"; legacy.field_values = fields; legacy.field_count = 128;
        Value borrowed = create_void(); borrowed.type = VAL_STRUCT; borrowed.as.struct_val = &legacy;
        Value tuple = create_tuple(&borrowed, 1); CHECK(nl_carrier_values_adopt(heap, ref(tuple)));
        NlCarrierRef root = ref(tuple); size_t before = failures; budget = stop;
        bool ok = transfer ? nl_carrier_values_transfer(heap, &root, 1) : nl_carrier_values_collect(heap, &root, 1);
        CHECK(!ok && failures == before + 1); injected++; budget = SIZE_MAX;
        stats(heap, 129, 0, 0);
        CHECK(strcmp(fields[0].as.struct_val->field_values[0].as.string_val, "failure survivor") == 0);
        CHECK(strcmp(fields[127].as.struct_val->field_values[0].as.string_val, "failure survivor") == 0);
        CHECK(nl_carrier_values_collect(heap, &root, 1)); stats(heap, 129, 0, 0);
        finish(heap, objects);
    }
    CHECK(injected == 18);
    puts("I preserve all 129 actual carriers across 18 injected graph-walk failures, then recover and release all metadata.");
}
int main(int argc, char **argv) {
    CHECK(setvbuf(stdout, NULL, _IONBF, 0) == 0);
    if (argc == 2 && strcmp(argv[1], "--managed-tuple-only") == 0) { managed_tuple_storage(); return 0; }
    CHECK(argc == 1);
    creation_and_adoption_failures(); mixed_constructors(); typed_lists_and_integer_refusal(); nested_typed_list_payload();
    managed_tuple_storage(); transfer_and_array_payloads();
    legacy_names_and_borrowed_elements(); allocation_rollback();
    CHECK(metadata_count == 0);
    puts("I complete actual carrier adapter controls; interpreter source activation remains pending.");
    return 0;
}
