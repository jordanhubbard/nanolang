/* I reuse the actual parsed interpreter fixture and allocation forwarding. */
#define main eval_original_suite_main
#include "test_eval.c"
#undef main

static void release_callable(Value value) {
    ASSERT_EQ(value.type, VAL_FUNCTION);
    nano_test_metadata_free(value.as.function_val.function_name);
    free_function_signature(value.as.function_val.signature);
}

int main(void) {
    /* I preserve the final cohort even when LeakSanitizer terminates at exit. */
    ASSERT_EQ(setvbuf(stdout, NULL, _IONBF, 0), 0);
    const char *source =
        "fn lifetime_target(x: int) -> int { return x }\n"
        "shadow lifetime_target { assert (== (lifetime_target 7) 7) }\n"
        "fn direct() -> int { return (lifetime_target 7) }\n"
        "shadow direct { assert (== (direct) 7) }\n"
        "fn bound() -> int { let action: fn(int) -> int = lifetime_target return (action 7) }\n"
        "shadow bound { assert (== (bound) 7) }\n"
        "fn declaration_return() -> fn(int) -> int { return lifetime_target }\n"
        "shadow declaration_return { assert (== ((declaration_return) 7) 7) }\n"
        "fn binding_return() -> fn(int) -> int { let action: fn(int) -> int = lifetime_target return action }\n"
        "shadow binding_return { assert (== ((binding_return) 7) 7) }\n"
        "fn recursive_scalar(depth: int) -> int { if (> depth 0) { return (recursive_scalar (- depth 1)) } return (bound) }\n"
        "shadow recursive_scalar { assert (== (recursive_scalar 3) 7) }\n"
        "fn recursive_return(depth: int) -> fn(int) -> int { if (> depth 0) { return (recursive_return (- depth 1)) } return lifetime_target }\n"
        "shadow recursive_return { assert (== ((recursive_return 3) 7) 7) }\n"
        "fn invoke(action: fn(int) -> int) -> int { return (action 7) }\n"
        "shadow invoke { assert (== (invoke lifetime_target) 7) }\n"
        "fn forward(action: fn(int) -> int) -> fn(int) -> int { return action }\n"
        "shadow forward { assert (== ((forward lifetime_target) 7) 7) }\n"
        "fn computed() -> int { return ((declaration_return) 7) }\n"
        "shadow computed { assert (== (computed) 7) }\n"
        "fn main() -> int { return 0 }\n";
    struct { const char *name; bool depth, parameter, returned; } routes[] = {
        {"direct", false, false, false}, {"bound", false, false, false},
        {"declaration_return", false, false, true},
        {"binding_return", false, false, true},
        {"recursive_scalar", true, false, false},
        {"recursive_return", true, false, true},
        {"invoke", false, true, false}, {"forward", false, true, true},
        {"computed", false, false, false},
    };
    size_t outstanding = 0;
    for (size_t route = 0; route < sizeof routes / sizeof *routes; ++route) {
        RunCtx ctx;
        ASSERT(run_ctx_init(&ctx, source));
        ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
        /* I keep this caller-owned value outside the observed cohort. */
        Type argument_type = TYPE_INT;
        Value caller = create_function("lifetime_target",
            create_function_signature(&argument_type, 1, TYPE_INT));
        const char *caller_name = caller.as.function_val.function_name;
        FunctionSignature *caller_signature = caller.as.function_val.signature;
        int first = ctx.env->symbol_count;
        s_metadata_allocated = s_metadata_released = 0;
        s_metadata_target = "lifetime_target";
        s_metadata_tracking = true;
        for (int iteration = 0; iteration < 64; ++iteration) {
            Value depth = create_int(3);
            Value *argument = routes[route].parameter ? &caller : &depth;
            int count = routes[route].parameter || routes[route].depth ? 1 : 0;
            Value value = call_function(routes[route].name, argument, count, ctx.env);
            if (routes[route].returned) {
                ASSERT_EQ(value.type, VAL_FUNCTION);
                ASSERT(strcmp(value.as.function_val.function_name, "lifetime_target") == 0);
                ASSERT(value.as.function_val.function_name != caller_name);
                ASSERT(value.as.function_val.signature != caller_signature);
                ASSERT_EQ(value.as.function_val.signature->param_count, 1);
                ASSERT_EQ(value.as.function_val.signature->param_types[0], TYPE_INT);
                ASSERT_EQ(value.as.function_val.signature->return_type, TYPE_INT);
                Value seven = create_int(7);
                Value actual = call_function(value.as.function_val.function_name, &seven, 1, ctx.env);
                ASSERT_EQ(actual.type, VAL_INT);
                ASSERT_EQ(actual.as.int_val, 7);
                release_callable(value);
            } else {
                ASSERT_EQ(value.type, VAL_INT);
                ASSERT_EQ(value.as.int_val, 7);
            }
            ASSERT_EQ(ctx.env->symbol_count, first);
            ASSERT(caller.as.function_val.function_name == caller_name);
            ASSERT(caller.as.function_val.signature == caller_signature);
            ASSERT(strcmp(caller_name, "lifetime_target") == 0);
            ASSERT_EQ(caller_signature->param_types[0], TYPE_INT);
        }
        size_t before_control = s_metadata_allocated;
        /* I prove that actual environment teardown reaches the tracked free. */
        Value control = create_function("lifetime_target",
            create_function_signature(&argument_type, 1, TYPE_INT));
        env_define_var(ctx.env, "teardown_control", TYPE_FUNCTION, false, control);
        ASSERT_EQ(s_metadata_allocated, before_control + 1);
        ASSERT(s_metadata_allocations[before_control].pointer == control.as.function_val.function_name);
        run_ctx_free(&ctx);
        ASSERT(s_metadata_allocations[before_control].released);
        s_metadata_tracking = false;
        release_callable(caller);
        size_t retained = s_metadata_allocated - s_metadata_released;
        printf("I observe %s: %zu names, %zu releases, %zu outstanding after teardown.\n",
               routes[route].name, s_metadata_allocated, s_metadata_released, retained);
        outstanding += retained;
    }
    printf("I retain %zu callable names across nine 64-call routes.\n", outstanding);
    /* I leave this acceptance assertion strict while ownership remains open. */
    return outstanding ? EXIT_FAILURE : EXIT_SUCCESS;
}
