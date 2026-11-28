/* I exercise non-owning C/API views without granting source write permission. */
#define main original_eval_suite_main
#include "test_eval.c"
#undef main

static void retire_function(Value value) {
    nano_test_metadata_free(value.as.function_val.function_name);
    free_function_signature(value.as.function_val.signature);
}

static void caller_callables(Type view_type) {
    Environment *env = create_environment();
    Type argument = TYPE_INT;
    Value first = create_function("borrow_first", create_function_signature(&argument, 1, TYPE_INT));
    Value second = create_function("borrow_second", create_function_signature(&argument, 1, TYPE_INT));
    const char *first_name = first.as.function_val.function_name;
    FunctionSignature *first_signature = first.as.function_val.signature;
    env_define_var(env, "view", view_type, true, first);
    ASSERT(env_get_var(env, "view")->value.as.function_val.function_name == first_name);
    env_set_var(env, "view", first);
    env_set_var(env, "view", second);
    ASSERT(env_get_var(env, "view")->value.as.function_val.function_name == second.as.function_val.function_name);
    ASSERT(first.as.function_val.function_name == first_name);
    ASSERT(first.as.function_val.signature == first_signature);
    ASSERT(strcmp(first_name, "borrow_first") == 0);
    ASSERT_EQ(first_signature->param_types[0], TYPE_INT);
    free_environment(env);
    ASSERT(strcmp(first_name, "borrow_first") == 0);
    ASSERT(strcmp(second.as.function_val.function_name, "borrow_second") == 0);
    ASSERT_EQ(second.as.function_val.signature->return_type, TYPE_INT);
    retire_function(first);
    retire_function(second);
}

static void caller_strings(Type view_type) {
    Environment *env = create_environment();
    Value first = create_string("I keep the first borrowed string.");
    Value second = create_string("I keep the second borrowed string.");
    env_define_var(env, "view", view_type, true, first);
    env_set_var(env, "view", first);
    env_set_var(env, "view", second);
    ASSERT(env_get_var(env, "view")->value.as.string_val == second.as.string_val);
    ASSERT(strcmp(first.as.string_val, "I keep the first borrowed string.") == 0);
    free_environment(env);
    ASSERT(strcmp(second.as.string_val, "I keep the second borrowed string.") == 0);
    gc_release(first.as.string_val);
    gc_release(second.as.string_val);
}

static void caller_aggregate_graph(Type view_type) {
    Environment *env = create_environment();
    Value array = create_array(VAL_INT, 2, 2);
    ((long long *)array.as.array_val->data)[0] = 71;
    ((long long *)array.as.array_val->data)[1] = -19;
    char *fields[] = {"items"};
    Value record = create_struct("CallerGraph", fields, &array, 1);
    Value tuple = create_tuple(&record, 1);
    size_t before = env_static_array_reclaim_count();
    env_define_var(env, "view", view_type, true, array);
    env_set_var(env, "view", array);
    env_define_var(env, "record_view", view_type, false, record);
    env_define_var(env, "tuple_view", view_type, false, tuple);
    env_set_var(env, "view", record);
    ASSERT(env_get_var(env, "view")->value.as.struct_val == record.as.struct_val);
    free_environment(env);
    ASSERT_EQ(env_static_array_reclaim_count(), before);
    ASSERT_EQ(record.as.struct_val->field_values[0].type, VAL_ARRAY);
    ASSERT(tuple.as.tuple_val->elements[0].as.struct_val == record.as.struct_val);
    ASSERT_EQ(((long long *)array.as.array_val->data)[0], 71);
    ASSERT_EQ(((long long *)array.as.array_val->data)[1], -19);
    free_tuple(tuple.as.tuple_val);
    release_struct_carrier_names(record.as.struct_val);
    free(record.as.struct_val->field_values);
    free(record.as.struct_val);
    free(array.as.array_val->data);
    free(array.as.array_val);
}

static void owned_roots_still_retire(void) {
    Environment *env = create_environment();
    Value array = create_array(VAL_INT, 1, 1);
    ((long long *)array.as.array_val->data)[0] = 47;
    size_t before = env_static_array_reclaim_count();
    env_define_var(env, "owner", TYPE_ARRAY, false, array);
    env_define_var(env, "shared", TYPE_BORROW_SHARED, false, array);
    env_define_var(env, "mutable", TYPE_BORROW_MUT, true, array);
    free_environment(env);
    ASSERT_EQ(env_static_array_reclaim_count(), before + 1);

    env = create_environment();
    Value text = create_string("I retain one owner, not an extra reference for each view.");
    size_t objects = gc_get_stats().num_objects;
    env_define_var(env, "owner", TYPE_STRING, false, text);
    env_define_var(env, "shared", TYPE_BORROW_SHARED, false, text);
    env_define_var(env, "mutable", TYPE_BORROW_MUT, true, text);
    free_environment(env);
    ASSERT_EQ(gc_get_stats().num_objects + 1, objects);
}

int main(void) {
    ASSERT_EQ(setvbuf(stdout, NULL, _IONBF, 0), 0);
    gc_init();
    size_t objects = gc_get_stats().num_objects;
    const Type views[] = {TYPE_BORROW_SHARED, TYPE_BORROW_MUT};
    for (size_t i = 0; i < sizeof views / sizeof *views; ++i) {
        caller_callables(views[i]);
        caller_strings(views[i]);
        caller_aggregate_graph(views[i]);
    }
    owned_roots_still_retire();
    ASSERT_EQ(gc_get_stats().num_objects, objects);
    gc_shutdown();
    puts("I preserve shared/mutable caller values and exactly-once owned reclamation.");
    return 0;
}
