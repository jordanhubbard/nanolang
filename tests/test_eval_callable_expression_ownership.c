/* I observe actual expression allocations without changing interpreter cleanup. */
#define main eval_original_suite_main
#include "test_eval.c"
#undef main

static void expression_reads(void) {
    ASSERT_EQ(setvbuf(stdout,NULL,_IONBF,0),0);
    const char *source =
        "struct Holder { action: fn(int) -> int }\n"
        "fn lifetime_target(x: int) -> int { return x }\n"
        "shadow lifetime_target { assert (== (lifetime_target 7) 7) }\n"
        "let bound: fn(int) -> int = lifetime_target\n"
        "let holder: Holder = Holder { action: bound }\n"
        "let stored: array<fn(int) -> int> = [bound]\n"
        "fn probe_decl() -> fn(int) -> int { return lifetime_target }\n"
        "shadow probe_decl { assert (== ((probe_decl) 7) 7) }\n"
        "fn probe_bound() -> fn(int) -> int { return bound }\n"
        "shadow probe_bound { assert (== ((probe_bound) 7) 7) }\n"
        "fn probe_field() -> fn(int) -> int { return holder.action }\n"
        "shadow probe_field { assert (== ((probe_field) 7) 7) }\n"
        "fn probe_array() -> fn(int) -> int { return (at stored 0) }\n"
        "shadow probe_array { assert (== ((probe_array) 7) 7) }\n"
        "fn probe_call() -> fn(int) -> int { return (probe_decl) }\n"
        "shadow probe_call { assert (== ((probe_call) 7) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n";
    RunCtx ctx;ASSERT(run_ctx_init(&ctx,source));
    ASSERT(run_shadow_tests(ctx.program,ctx.env,false));
    Symbol *binding=env_get_var(ctx.env,"bound"),*holder=env_get_var(ctx.env,"holder");
    ASSERT(binding && holder);ASSERT_EQ(binding->value.type,VAL_FUNCTION);ASSERT_EQ(holder->value.type,VAL_STRUCT);
    const char *caller_name=binding->value.as.function_val.function_name;
    FunctionSignature *caller_signature=binding->value.as.function_val.signature;
    const char *field_name=holder->value.as.struct_val->field_values[0].as.function_val.function_name;
    const char *names[]={"probe_decl","probe_bound","probe_field","probe_array","probe_call"};
    s_metadata_allocated=s_metadata_released=0;s_metadata_target="lifetime_target";s_metadata_tracking=true;
    for(size_t n=0;n<sizeof names/sizeof *names;n++) {
        Function *function=env_get_function(ctx.env,names[n]);ASSERT(function && function->body);
        ASTNode *statement=function->body->as.block.statements[0];ASSERT_EQ(statement->type,AST_RETURN);
        size_t before=s_metadata_allocated;
        Value value=repl_eval_node(statement->as.return_stmt.value,ctx.env);ASSERT_EQ(value.type,VAL_FUNCTION);
        ASSERT(value.as.function_val.signature);ASSERT_EQ(value.as.function_val.signature->param_count,1);
        ASSERT_EQ(value.as.function_val.signature->param_types[0],TYPE_INT);
        ASSERT_EQ(s_metadata_allocated - before, 1);
        ASSERT(s_metadata_allocations[before].pointer == value.as.function_val.function_name);
        ASSERT(!s_metadata_allocations[before].released);
        ASSERT(value.as.function_val.function_name != caller_name);
        ASSERT(value.as.function_val.function_name != field_name);
        ASSERT(value.as.function_val.signature != caller_signature);
        Value seven=create_int(7),actual=call_function(value.as.function_val.function_name,&seven,1,ctx.env);
        ASSERT_EQ(actual.type,VAL_INT);ASSERT_EQ(actual.as.int_val,7);
        printf("I own an independent callable result from %s.\n", names[n]);
        nano_test_metadata_free(value.as.function_val.function_name);
        free_function_signature(value.as.function_val.signature);
        ASSERT(binding->value.as.function_val.function_name==caller_name);
        ASSERT(binding->value.as.function_val.signature==caller_signature);
        ASSERT(strcmp(caller_name,"lifetime_target")==0);ASSERT_EQ(caller_signature->param_types[0],TYPE_INT);
    }
    ASSERT_EQ(s_metadata_allocated, 5);
    ASSERT_EQ(s_metadata_allocated, s_metadata_released);
    run_ctx_free(&ctx);s_metadata_tracking=false;
}


/* I count source-owned temporaries without counting a preexisting caller. */
static void source_temporaries(void) {
    const char *source =
        "effect Invoke { apply : fn(int) -> int -> int }\n"
        "fn lifetime_target(x: int) -> int { return x }\n"
        "shadow lifetime_target { assert (== (lifetime_target 7) 7) }\n"
        "fn produce() -> fn(int) -> int { return lifetime_target }\n"
        "shadow produce { let action: fn(int) -> int = (produce) assert (== (action 7) 7) }\n"
        "fn discarded() -> int { (produce)\n (produce)\n let mut index: int = 0 while (< index 3) { set index (+ index 1) (produce) } return 7 }\n"
        "shadow discarded { assert (== (discarded) 7) }\n"
        "fn replacement() -> int { let mut action: fn(int) -> int = lifetime_target set action action set action lifetime_target return (action 7) }\n"
        "shadow replacement { assert (== (replacement) 7) }\n"
        "fn handled() -> int { return handle { perform Invoke.apply(lifetime_target) } with { apply action -> { (action 7) } } }\n"
        "shadow handled { assert (== (handled) 7) }\n"
        "async fn async_forward(action: fn(int) -> int) -> fn(int) -> int { return action }\n"
        "shadow async_forward { let action: fn(int) -> int = (async_forward lifetime_target) assert (== (action 7) 7) }\n"
        "fn async_case() -> int { let action: fn(int) -> int = (async_forward lifetime_target) return (action 7) }\n"
        "shadow async_case { assert (== (async_case) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n";
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx, source));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    const char *routes[] = {"discarded", "replacement", "handled", "async_case"};
    for (size_t route = 0; route < sizeof routes / sizeof *routes; ++route) {
        s_metadata_allocated = s_metadata_released = 0;
        s_metadata_target = "lifetime_target";
        s_metadata_tracking = true;
        int first = ctx.env->symbol_count;
        for (int iteration = 0; iteration < 64; ++iteration) {
            Value result = call_function(routes[route], NULL, 0, ctx.env);
            ASSERT_EQ(result.type, VAL_INT);
            ASSERT_EQ(result.as.int_val, 7);
            ASSERT_EQ(ctx.env->symbol_count, first);
            ASSERT(nl_effect_find_handler("Invoke", "apply", NULL) == NULL);
        }
        s_metadata_tracking = false;
        printf("I observe %s: %zu allocations and %zu releases.\n", routes[route], s_metadata_allocated, s_metadata_released);
        ASSERT(s_metadata_allocated > 0);
        ASSERT_EQ(s_metadata_allocated, s_metadata_released);
    }
    run_ctx_free(&ctx);
}

int main(void) {
    ASSERT_EQ(setvbuf(stdout, NULL, _IONBF, 0), 0);
    expression_reads();
    source_temporaries();
    return 0;
}
