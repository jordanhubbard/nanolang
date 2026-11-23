/* I retire scalar snapshots without retiring their caller-owned containers. */
#define main original_eval_suite_main
#include "test_eval.c"
#undef main

static void release_scalar(Value value) {
    if (value.type == VAL_STRING) gc_release(value.as.string_val);
    else if (value.type == VAL_FUNCTION) {
        nano_test_metadata_free(value.as.function_val.function_name);
        free_function_signature(value.as.function_val.signature);
    }
}

static Value scalar(bool callable, const char *text) {
    if (!callable) return create_string(text);
    Type parameter = TYPE_INT;
    FunctionSignature *signature = create_function_signature(&parameter, 1, TYPE_FUNCTION);
    signature->return_fn_sig = create_function_signature(&parameter, 1, TYPE_INT);
    return create_function("projection_target", signature);
}

static void independent(Value result, Value source) {
    ASSERT_EQ(result.type, source.type);
    if (source.type == VAL_STRING) {
        ASSERT(result.as.string_val != source.as.string_val);
        ASSERT(strcmp(result.as.string_val, source.as.string_val) == 0);
    } else {
        ASSERT(result.as.function_val.function_name != source.as.function_val.function_name);
        ASSERT(result.as.function_val.signature != source.as.function_val.signature);
        ASSERT(result.as.function_val.signature->return_fn_sig != source.as.function_val.signature->return_fn_sig);
        ASSERT_EQ(result.as.function_val.signature->return_fn_sig->param_types[0], TYPE_INT);
    }
}

static void release_union(Value value) {
    UnionValue *u = value.as.union_val;
    for (int i = 0; i < u->field_count; ++i) {
        if (u->field_values[i].type == VAL_STRING || u->field_values[i].type == VAL_FUNCTION)
            release_scalar(u->field_values[i]);
        free(u->field_names[i]);
    }
    free(u->field_names); free(u->field_values);
    free(u->union_name); free(u->variant_name); free(u);
}

static Value read_projection(Environment *env, const char *route, bool literal) {
    ASTNode name = {.type = AST_IDENTIFIER}; name.as.identifier = "payload_box";
    ASTNode fallback = {.type = AST_IDENTIFIER}; fallback.as.identifier = "fallback";
    if (literal) { fallback.type = AST_STRING; fallback.as.string_val = "literal-default"; }
    if (strcmp(route, "tuple") == 0) {
        ASTNode index = {.type = AST_TUPLE_INDEX};
        index.as.tuple_index.tuple = &name; index.as.tuple_index.index = 0;
        return repl_eval_node(&index, env);
    }
    if (strcmp(route, "try") == 0) {
        ASTNode attempt = {.type = AST_TRY_OP}; attempt.as.try_op.operand = &name;
        return repl_eval_node(&attempt, env);
    }
    ASTNode *arguments[] = {&name, &fallback};
    ASTNode call = {.type = AST_CALL}; call.as.call.name = (char *)route;
    call.as.call.args = arguments;
    call.as.call.arg_count = strcmp(route, "result_unwrap_or") == 0 ? 2 : 1;
    return repl_eval_node(&call, env);
}

static void projection_case(const char *route, bool callable, bool error, bool literal) {
    Value input = scalar(callable, "payload"), fallback = scalar(callable, "fallback");
    char *fields[] = {error ? "error" : "value"};
    bool tuple = strcmp(route, "tuple") == 0;
    Value box = tuple ? create_tuple(&input, 1) : create_union("Result", error ? 1 : 0,
        error ? "Err" : "Ok", fields, &input, 1);
    Value payload = tuple ? box.as.tuple_val->elements[0] : box.as.union_val->field_values[0];
    Environment *env = create_environment();
    /* These are existing C/API views; I grant no new source borrow admission. */
    env_define_var(env, "payload_box", TYPE_BORROW_SHARED, false, box);
    env_define_var(env, "fallback", TYPE_BORROW_SHARED, false, fallback);
    size_t objects = gc_get_stats().num_objects;
    Value result = read_projection(env, route, literal);
    if (literal) {
        ASSERT_EQ(result.type, VAL_STRING);
        ASSERT(strcmp(result.as.string_val, "literal-default") == 0);
    } else independent(result, error && strcmp(route, "result_unwrap_or") == 0 ? fallback : payload);
    if (!callable) ASSERT_EQ(gc_get_stats().num_objects, objects + 1);
    release_scalar(result);
    ASSERT_EQ(gc_get_stats().num_objects, objects);
    /* Low-level Result helper APIs continue to borrow their input slots. */
    if (!tuple) {
        Value args[] = {box, fallback};
        Value borrowed = error ? builtin_result_unwrap_err(args) : builtin_result_unwrap(args);
        if (callable) { ASSERT(borrowed.as.function_val.function_name == payload.as.function_val.function_name); }
        else { ASSERT(borrowed.as.string_val == payload.as.string_val); }
    }
    if (callable) { ASSERT(strcmp(payload.as.function_val.function_name, "projection_target") == 0); }
    else { ASSERT(strcmp(payload.as.string_val, "payload") == 0); }
    free_environment(env);
    if (tuple) free_tuple(box.as.tuple_val); else release_union(box);
    release_scalar(input); release_scalar(fallback);
    printf("I preserve %s %s error=%d literal=%d snapshots and caller ownership.\n",
           route, callable ? "callable" : "string", error, literal);
}

static void aggregate_identity(void) {
    Value array = create_array(VAL_INT, 1, 1);
    ((long long *)array.as.array_val->data)[0] = 7;
    char *fields[] = {"value"};
    Value tuple = create_tuple(&array, 1), box = create_union("Result", 0, "Ok", fields, &array, 1);
    Environment *env = create_environment();
    env_define_var(env, "payload_box", TYPE_BORROW_SHARED, false, tuple);
    Value result = read_projection(env, "tuple", false);
    ASSERT(result.as.array_val == array.as.array_val);
    env_set_var(env, "payload_box", box);
    const char *routes[] = {"try", "result_unwrap", "result_unwrap_or"};
    env_define_var(env, "fallback", TYPE_BORROW_SHARED, false, array);
    for (size_t i = 0; i < sizeof routes / sizeof *routes; ++i) {
        result = read_projection(env, routes[i], false);
        ASSERT_EQ(result.type, VAL_ARRAY); ASSERT(result.as.array_val == array.as.array_val);
        ((long long *)result.as.array_val->data)[0] += 1;
    }
    ASSERT_EQ(((long long *)array.as.array_val->data)[0], 10);
    free_environment(env); free_tuple(tuple.as.tuple_val); release_union(box);
    free(array.as.array_val->data); free(array.as.array_val);
    puts("I preserve shared mutable non-scalar projection identity.");
}

static Value cached_result(void *data, int id) { (void)id; return *(Value *)data; }

static void cached_snapshots(bool callable) {
    Value input = scalar(callable, "cached");
    nano_scheduler_init();
    int id = nano_coro_spawn(cached_result, &input); ASSERT(id >= 0);
    nano_scheduler_run_until_done();
    Environment *env = create_environment();
    Value handle = create_void(); handle.type = VAL_COROUTINE; handle.as.int_val = id;
    env_define_var(env, "job", TYPE_BORROW_SHARED, false, handle);
    ASTNode identifier = {.type = AST_IDENTIFIER}; identifier.as.identifier = "job";
    ASTNode *arguments[] = {&identifier};
    ASTNode call = {.type = AST_CALL}; call.as.call.name = "coro_result";
    call.as.call.args = arguments; call.as.call.arg_count = 1;
    ASTNode await = {.type = AST_AWAIT}; await.as.await_expr.expr = &identifier;
    Value first = repl_eval_node(&call, env), second = repl_eval_node(&await, env);
    independent(first, input); independent(second, input); independent(first, second);
    release_scalar(first); release_scalar(second);
    ASSERT(nano_coro_release(id));
    if (callable) { ASSERT(strcmp(input.as.function_val.function_name, "projection_target") == 0); }
    else { ASSERT(strcmp(input.as.string_val, "cached") == 0); }
    free_environment(env); release_scalar(input);
    puts("I preserve independently owned cached scalar snapshots and caller values.");
}

int main(void) {
    ASSERT_EQ(setvbuf(stdout, NULL, _IONBF, 0), 0);
    size_t objects = gc_get_stats().num_objects;
    s_metadata_allocated = s_metadata_released = 0;
    s_metadata_target = "projection_target"; s_metadata_tracking = true;
    for (int callable = 0; callable < 2; ++callable) {
        projection_case("tuple", callable, false, false);
        projection_case("try", callable, false, false);
        projection_case("result_unwrap", callable, false, false);
        projection_case("result_unwrap_err", callable, true, false);
        projection_case("result_unwrap_or", callable, false, false);
        projection_case("result_unwrap_or", callable, true, false);
        cached_snapshots(callable);
    }
    projection_case("result_unwrap_or", false, true, true);
    aggregate_identity();
    s_metadata_tracking = false;
    ASSERT_EQ(s_metadata_allocated, s_metadata_released);
    ASSERT_EQ(gc_get_stats().num_objects, objects);
    printf("I balance scalar snapshot metadata: %zu allocations, %zu releases.\n", s_metadata_allocated, s_metadata_released);
    return 0;
}
