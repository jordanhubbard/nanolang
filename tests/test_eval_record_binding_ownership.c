/* I distinguish copied binding literals from caller and raw-carrier inputs. */
#define main original_eval_suite_main
#include "test_eval.c"
#undef main

static void release_caller_record(Value record) {
    for (int i = 0; i < record.as.struct_val->field_count; ++i) {
        Value value = record.as.struct_val->field_values[i];
        if (value.type == VAL_FUNCTION) {
            nano_test_metadata_free(value.as.function_val.function_name);
            free_function_signature(value.as.function_val.signature);
        } else if (value.type == VAL_STRUCT) release_caller_record(value);
        else if (value.type == VAL_STRING) gc_release(value.as.string_val);
    }
    release_struct_carrier_names(record.as.struct_val);
    free(record.as.struct_val->field_values);
    free(record.as.struct_val);
}

static void source_globals(void) {
    const char *source =
        "struct Holder { action: fn(int) -> int, text: string, items: array<int> }\n"
        "fn lifetime_target(x: int) -> int { return x }\n"
        "shadow lifetime_target { assert (== (lifetime_target 7) 7) }\n"
        "let shared: array<int> = [71]\n"
        "let mut holder: Holder = Holder { action: lifetime_target, text: \"first\", items: shared }\n"
        "let snapshot: Holder = holder\n"
        "fn replace() -> int { set holder Holder { action: lifetime_target, text: \"second\", items: shared } set holder holder let action: fn(int) -> int = holder.action return (+ (action 7) (at holder.items 0)) }\n"
        "shadow replace { assert (== (replace) 78) assert (== snapshot.text \"first\") }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n";
    s_metadata_allocated = s_metadata_released = 0;
    s_metadata_target = "lifetime_target";
    s_metadata_tracking = true;
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx, source));
    for (int round = 0; round < 3; ++round) {
        ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
        for (int i = 0; i < 64; ++i) {
            Value result = call_function("replace", NULL, 0, ctx.env);
            ASSERT_EQ(result.type, VAL_INT);
            ASSERT_EQ(result.as.int_val, 78);
        }
        Symbol *snapshot = env_get_var(ctx.env, "snapshot");
        Symbol *holder = env_get_var(ctx.env, "holder");
        ASSERT(snapshot && holder);
        ASSERT(snapshot->value.as.struct_val != holder->value.as.struct_val);
        ASSERT(strcmp(eval_test_record_field(snapshot->value, "text").as.string_val, "first") == 0);
        ASSERT(strcmp(eval_test_record_field(holder->value, "text").as.string_val, "second") == 0);
        ASSERT(eval_test_record_field(snapshot->value, "items").as.array_val == eval_test_record_field(holder->value, "items").as.array_val);
    }
    run_ctx_free(&ctx);
    s_metadata_tracking = false;
    ASSERT(s_metadata_allocated > 0);
    printf("I balance global literal/replacement copies: %zu allocations, %zu releases.\n", s_metadata_allocated, s_metadata_released);
    ASSERT_EQ(s_metadata_allocated, s_metadata_released);
}

static void caller_and_legacy_inputs(void) {
    Type argument = TYPE_INT;
    Value callback = create_function("caller_callback", create_function_signature(&argument, 1, TYPE_INT));
    StructValue *legacy = calloc(1, sizeof *legacy);
    legacy->struct_name = strdup("Legacy");
    legacy->field_count = 1;
    legacy->field_names = malloc(sizeof(char *));
    legacy->field_names[0] = strdup("action");
    legacy->field_values = malloc(sizeof(Value));
    legacy->field_values[0] = callback; /* I deliberately borrow this field. */
    Value raw = create_void(); raw.type = VAL_STRUCT; raw.as.struct_val = legacy;
    Environment *env = create_environment();
    env_define_var(env, "owned", TYPE_STRUCT, true, raw);
    Symbol *owned = env_get_var(env, "owned");
    ASSERT(owned->value.as.struct_val != legacy);
    ASSERT(owned->value.as.struct_val->field_values[0].as.function_val.function_name != callback.as.function_val.function_name);
    env_set_var(env, "owned", owned->value);
    env_set_var(env, "owned", raw);
    env_define_var(env, "borrowed", TYPE_BORROW_SHARED, false, raw);
    free_environment(env);
    ASSERT(strcmp(callback.as.function_val.function_name, "caller_callback") == 0);
    ASSERT_EQ(callback.as.function_val.signature->param_types[0], TYPE_INT);
    /* I preserve the raw-array cleanup policy: the carrier owns no callback. */
    Value array = create_array(VAL_STRUCT, 1, 1);
    ((StructValue **)array.as.array_val->data)[0] = legacy;
    env = create_environment();
    env_define_var(env, "raw_array", TYPE_ARRAY, false, array);
    free_environment(env);
    ASSERT(strcmp(callback.as.function_val.function_name, "caller_callback") == 0);
    nano_test_metadata_free(callback.as.function_val.function_name);
    free_function_signature(callback.as.function_val.signature);
    puts("I preserve caller-owned callable metadata through legacy and raw-array cleanup.");
}

static void nested_copy_aliases(void) {
    Type argument = TYPE_INT;
    Value callback = create_function("nested_callback", create_function_signature(&argument, 1, TYPE_INT));
    Value shared = create_array(VAL_INT, 1, 1);
    ((long long *)shared.as.array_val->data)[0] = 71;
    char *inner_names[] = {"action", "items"}; Value inner_values[] = {callback, shared};
    Value inner = create_struct("Inner", inner_names, inner_values, 2);
    char *outer_names[] = {"child"};
    Value outer = create_struct("Outer", outer_names, &inner, 1);
    Environment *env = create_environment();
    env_define_var(env, "outer", TYPE_STRUCT, false, outer);
    Value binding = env_get_var(env, "outer")->value;
    Value child = binding.as.struct_val->field_values[0];
    ASSERT(child.as.struct_val != outer.as.struct_val->field_values[0].as.struct_val);
    ASSERT(child.as.struct_val->field_values[1].as.array_val == shared.as.array_val);
    /* I keep the existing nested carrier lifetime; teardown reclaims shared arrays. */
    free_environment(env);
    ASSERT(strcmp(child.as.struct_val->field_values[0].as.function_val.function_name, "nested_callback") == 0);
    ASSERT(strcmp(inner.as.struct_val->field_values[0].as.function_val.function_name, "nested_callback") == 0);
    release_caller_record(child);
    release_caller_record(outer);
    release_caller_record(inner);
    nano_test_metadata_free(callback.as.function_val.function_name);
    free_function_signature(callback.as.function_val.signature);
    puts("I preserve nested record lifetime and shared array identity without broad reclamation.");
}

int main(void) {
    ASSERT_EQ(setvbuf(stdout, NULL, _IONBF, 0), 0);
    source_globals();
    caller_and_legacy_inputs();
    nested_copy_aliases();
    return 0;
}
