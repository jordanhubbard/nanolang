/* I require a callable record field to survive its defining source call. */
#define main eval_original_suite_main
#include "test_eval.c"
#undef main
static void release_field(Value value) {
    if (value.type == VAL_FUNCTION) {
        nano_test_metadata_free(value.as.function_val.function_name);
        free_function_signature(value.as.function_val.signature);
    } else if (value.type == VAL_STRUCT) {
        StructValue *record = value.as.struct_val;
        for (int i = 0; i < record->field_count; ++i) release_field(record->field_values[i]);
        release_struct_carrier_names(record);
        free(record->field_values);
        free(record);
    } else if (value.type == VAL_UNION) {
        UnionValue *tagged = value.as.union_val;
        for (int i = 0; i < tagged->field_count; ++i) {
            release_field(tagged->field_values[i]);
            free(tagged->field_names[i]);
        }
        free(tagged->field_names);
        free(tagged->field_values);
        free(tagged->union_name);
        free(tagged->variant_name);
        free(tagged);
    }
}

static void independent_copy(Value actual, Value source) {
    ASSERT_EQ(actual.type, VAL_FUNCTION);
    ASSERT(actual.as.function_val.function_name != source.as.function_val.function_name);
    ASSERT(strcmp(actual.as.function_val.function_name, source.as.function_val.function_name) == 0);
    FunctionSignature *copy = actual.as.function_val.signature;
    FunctionSignature *original = source.as.function_val.signature;
    ASSERT(copy && copy != original);
    ASSERT(function_signatures_equal(copy, original));
    ASSERT(copy->param_types != original->param_types);
    ASSERT(copy->return_fn_sig && copy->return_fn_sig != original->return_fn_sig);
    ASSERT(copy->return_fn_sig->param_types != original->return_fn_sig->param_types);
}

/* I retire borrowed sources before reading each independently copied field.
 * Explicit test cleanup does not claim production carrier reclamation. */
static void constructor_controls(void) {
    Type argument = TYPE_INT;
    FunctionSignature *signature = create_function_signature(&argument, 1, TYPE_FUNCTION);
    signature->return_fn_sig = create_function_signature(&argument, 1, TYPE_INT);
    s_metadata_allocated = s_metadata_released = 0;
    s_metadata_target = "lifetime_target";
    s_metadata_tracking = true;
    Value original = create_function("lifetime_target", signature);
    char *names[] = {"action"};
    Value record = create_struct("Holder", names, &original, 1);
    Value tagged = create_union("Callback", 0, "Some", names, &original, 1);
    independent_copy(record.as.struct_val->field_values[0], original);
    independent_copy(tagged.as.union_val->field_values[0], original);
    Value nested = create_struct("Envelope", names, &record, 1);
    Value copied = create_struct("Holder", names, record.as.struct_val->field_values, 1);
    Value tuple = create_tuple(&original, 1);
    independent_copy(copied.as.struct_val->field_values[0], original);
    independent_copy(nested.as.struct_val->field_values[0].as.struct_val->field_values[0], original);
    independent_copy(tuple.as.tuple_val->elements[0], original);
    release_field(original);
    release_field(record);
    Value survivors[] = {tagged.as.union_val->field_values[0], copied.as.struct_val->field_values[0],
        nested.as.struct_val->field_values[0].as.struct_val->field_values[0], tuple.as.tuple_val->elements[0]};
    for (size_t i = 0; i < sizeof survivors / sizeof *survivors; ++i) {
        ASSERT(strcmp(survivors[i].as.function_val.function_name, "lifetime_target") == 0);
        ASSERT_EQ(survivors[i].as.function_val.signature->return_fn_sig->param_types[0], TYPE_INT);
    }
    release_field(tagged);
    release_field(copied);
    release_field(nested);
    free_tuple(tuple.as.tuple_val);
    s_metadata_tracking = false;
    ASSERT_EQ(s_metadata_allocated, 6);
    ASSERT_EQ(s_metadata_allocated, s_metadata_released);
    puts("I preserve six independent nested-signature constructor values through source retirement.");
}

int main(int argc, char **argv) {
    ASSERT_EQ(setvbuf(stdout,NULL,_IONBF,0),0);
    constructor_controls();
    if (argc == 2 && strcmp(argv[1], "--constructors-only") == 0) return 0;
    const char *source =
        "struct Holder { action: fn(int) -> int }\n"
        "union Callback { Some { action: fn(int) -> int } }\n"
        "fn lifetime_target(x: int) -> int { return x }\n"
        "shadow lifetime_target { assert (== (lifetime_target 7) 7) }\n"
        "fn record_factory(action: fn(int) -> int) -> Holder { let local: fn(int) -> int = action return Holder { action: local } }\n"
        "shadow record_factory { let holder_value: Holder = (record_factory lifetime_target) let picked: fn(int) -> int = holder_value.action assert (== (picked 7) 7) }\n"
        "fn record_case() -> int { let holder_value: Holder = (record_factory lifetime_target) let picked: fn(int) -> int = holder_value.action return (picked 7) }\n"
        "shadow record_case { assert (== (record_case) 7) }\n"
        "fn union_factory(action: fn(int) -> int) -> Callback { let local: fn(int) -> int = action return Callback.Some { action: local } }\n"
        "shadow union_factory { let tagged: Callback = (union_factory lifetime_target) assert (== match tagged { Some(payload) => { let picked: fn(int) -> int = payload.action (picked 7) } } 7) }\n"
        "fn union_case() -> int { let tagged: Callback = (union_factory lifetime_target) return match tagged { Some(payload) => { let picked: fn(int) -> int = payload.action (picked 7) } } }\n"
        "shadow union_case { assert (== (union_case) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n";
    RunCtx ctx;ASSERT(run_ctx_init(&ctx,source));
    Value result=call_function("record_case",NULL,0,ctx.env);
    printf("returned type=%d\n",result.type);
    ASSERT_EQ(result.type,VAL_INT);ASSERT_EQ(result.as.int_val,7);
    result = call_function("union_case", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_INT); ASSERT_EQ(result.as.int_val, 7);
    ASSERT(run_shadow_tests(ctx.program,ctx.env,false));
    puts("I preserve record and union callbacks after their defining calls and run every fixture shadow.");
    run_ctx_free(&ctx);return 0;
}
