/* I execute my actual private lookup/context helpers with real STRING/GC values. */
#include "nanolang.h"
#include "runtime/gc.h"
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t comparisons;
static int counted_name(const char *left, const char *right) {
    ++comparisons;
    return strcmp(left, right);
}
#define strcmp counted_name
#include "eval/record_fields.inc"
#undef strcmp

static int first_name(char **names, int count, const char *query) {
    for (int i = 0; i < count; ++i)
        if (strcmp(names[i], query) == 0) return i;
    return -1;
}

static void ordered_fields(void) {
    char *original[] = {"beta", "alpha", "alphabet", "alpha", "", "",
                        "\xc3\xa9", ("\xc3\xa9" "clair"), "\xc3\xb1", "\x80"};
    Type original_types[] = {TYPE_INT, TYPE_INT, TYPE_U8, TYPE_U8, TYPE_INT,
                             TYPE_U8, TYPE_INT, TYPE_U8, TYPE_INT, TYPE_U8};
    enum { count = sizeof original / sizeof *original };
    const int64_t inputs[] = {INT64_MIN, -1, 0, 255, 256, 300, INT64_MAX};
    for (int rotation = 0; rotation < count; ++rotation) {
        char *names[count];
        Type types[count];
        Value values[count];
        for (int i = 0; i < count; ++i) {
            int at = (i + rotation) % count;
            names[i] = original[at];
            types[i] = original_types[at];
            values[i] = create_int(100 + at);
        }
        StructDef definition = {0};
        definition.field_names = names;
        definition.field_types = types;
        definition.field_count = count;
        Value carrier = create_struct("Reordered", names, values, count);
        assert(carrier.type == VAL_STRUCT && carrier.as.struct_val->packed_names);
        for (int query = 0; query < count; ++query) {
            int expected = first_name(names, count, original[query]);
            assert(expected >= 0);
            Value selected = eval_record_field_value(carrier.as.struct_val, original[query]);
            assert(selected.type == VAL_INT && selected.as.int_val == values[expected].as.int_val);
            for (size_t sample = 0; sample < sizeof inputs / sizeof *inputs; ++sample) {
                Value value = create_int(inputs[sample]);
                assert(eval_record_field_context(&definition, original[query], &value));
                int64_t wanted = types[expected] == TYPE_U8 ?
                    (int64_t)((uint64_t)inputs[sample] & UINT64_C(255)) : inputs[sample];
                assert(value.type == VAL_INT && value.as.int_val == wanted);
            }
        }
        release_struct_carrier_names(carrier.as.struct_val);
        free(carrier.as.struct_val->field_values);
        free(carrier.as.struct_val);
    }
}

static void context_refusals(void) {
    char *names[] = {"alpha", "alphabet", ""};
    Type types[] = {TYPE_INT, TYPE_U8, TYPE_U8};
    StructDef definition = {0};
    definition.field_names = names;
    definition.field_types = types;
    definition.field_count = 3;
    Value unchanged = create_bool(true);
    assert(eval_record_field_context(NULL, "alphabet", &unchanged));
    assert(eval_record_field_context(&definition, "missing", &unchanged));
    assert(eval_record_field_context(&definition, "alpha", &unchanged));
    assert(unchanged.type == VAL_BOOL && unchanged.as.bool_val);
    assert(!eval_record_field_context(&definition, "alphabet", &unchanged));
    assert(unchanged.type == VAL_BOOL && unchanged.as.bool_val);
    Value fractional = create_float(1.5);
    assert(!eval_record_field_context(&definition, "", &fractional));
    assert(fractional.type == VAL_FLOAT && fractional.as.float_val == 1.5);
    definition.field_count = 0;
    definition.field_names = NULL;
    definition.field_types = NULL;
    assert(eval_record_field_context(&definition, "", &unchanged));
    Value absent = create_int(99);
    assert(eval_record_field_context(&definition, "missing", &absent));
    assert(absent.as.int_val == 99);
}

static void owned_string_and_identity(void) {
    char *names[] = {"string", "scalar", "array"};
    Array array = {0};
    Value values[] = {create_string("I retain the read after its source is released."),
                      create_int(71), create_void()};
    values[1].is_break = true;
    values[2].type = VAL_ARRAY;
    values[2].as.array_val = &array;
    StructValue record = {0};
    record.struct_name = "Identity";
    record.field_count = 3;
    record.field_names = names;
    record.field_values = values;
    Value copy = eval_record_field_value(&record, "string");
    assert(copy.type == VAL_STRING && gc_is_managed(copy.as.string_val));
    assert(copy.as.string_val != values[0].as.string_val);
    assert(strcmp(copy.as.string_val, values[0].as.string_val) == 0);
    gc_release(values[0].as.string_val);
    values[0] = create_void();
    assert(strcmp(copy.as.string_val, "I retain the read after its source is released.") == 0);
    gc_release(copy.as.string_val);
    Value scalar = eval_record_field_value(&record, "scalar");
    assert(scalar.type == VAL_INT && scalar.as.int_val == 71 && scalar.is_break);
    Value selected = eval_record_field_value(&record, "array");
    assert(selected.type == VAL_ARRAY && selected.as.array_val == &array);
    assert(eval_record_field_value(&record, "absent").type == VAL_VOID);
    record.field_count = 0;
    record.field_names = NULL;
    record.field_values = NULL;
    assert(eval_record_field_value(&record, "").type == VAL_VOID);
}

static void comparison_observation(void) {
    enum { count = 64, repeats = 1000 };
    char storage[count][24];
    char *names[count];
    Type types[count];
    Value values[count];
    for (int i = 0; i < count; ++i) {
        snprintf(storage[i], sizeof storage[i], "alpha_%d", i);
        names[i] = storage[i];
        types[i] = TYPE_INT;
        values[i] = create_int(i);
    }
    names[count-1] = "zebra";
    StructDef definition = {0};
    definition.field_names = names;
    definition.field_types = types;
    definition.field_count = count;
    StructValue record = {0};
    record.struct_name = "Wide";
    record.field_names = names;
    record.field_values = values;
    record.field_count = count;
    comparisons = 0;
    for (int i = 0; i < repeats; ++i) {
        Value selected = eval_record_field_value(&record, "zebra");
        assert(selected.type == VAL_INT && selected.as.int_val == count-1);
        Value input = create_int(300);
        assert(eval_record_field_context(&definition, "zebra", &input));
        assert(input.as.int_val == 300);
    }
    printf("I perform 2000 actual record lookups with %zu exact-name comparisons.\n", comparisons);
}

int main(void) {
    gc_init();
    ordered_fields();
    context_refusals();
    owned_string_and_identity();
    comparison_observation();
    assert(gc_get_stats().num_objects == 0);
    gc_shutdown();
    puts("I retain ordered record lookup, byte context/refusals and owned STRING reads.");
    return 0;
}
