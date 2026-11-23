/* I run every valid source shadow and preserve the existing static refusal. */
#define main original_eval_suite_main
#include "test_eval.c"
#undef main

static char *fixture(const char *path) {
    FILE *file = fopen(path, "rb"); ASSERT(file);
    ASSERT_EQ(fseek(file, 0, SEEK_END), 0);
    long size = ftell(file); ASSERT(size >= 0); rewind(file);
    char *source = malloc((size_t)size + 1); ASSERT(source);
    ASSERT_EQ(fread(source, 1, (size_t)size, file), (size_t)size);
    source[size] = '\0'; fclose(file); return source;
}

static void retain_refusal(void) {
    char *source = fixture("tests/fixtures/scalar_projection_invalid_heap_borrow.nano");
    RunCtx ctx = {0}; ctx.tokens = tokenize(source, &ctx.token_count); ASSERT(ctx.tokens);
    ctx.program = parse_program(ctx.tokens, ctx.token_count); ASSERT(ctx.program);
    clear_module_cache(); ctx.env = create_environment(); typecheck_set_current_file("<test>");
    bool accepted = type_check(ctx.program, ctx.env); ASSERT(!accepted);
    run_ctx_free(&ctx); free(source);
    puts("I preserve the parsed heap-bearing mutable borrow refusal.");
}

static void retire_returned_union(Value value) {
    UnionValue *u = value.as.union_val;
    for (int i = 0; i < u->field_count; ++i) {
        if (u->field_values[i].type == VAL_STRING) gc_release(u->field_values[i].as.string_val);
        free(u->field_names[i]);
    }
    free(u->field_names); free(u->field_values);
    free(u->union_name); free(u->variant_name); free(u);
}

int main(void) {
    ASSERT_EQ(setvbuf(stdout, NULL, _IONBF, 0), 0);
    retain_refusal();
    char *source = fixture("tests/fixtures/scalar_projection_ownership.nano");
    RunCtx ctx; ASSERT(run_ctx_init(&ctx, source));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    const char *functions[] = {"tuple_case", "unwrap_case", "unwrap_error_case", "fallback_case"};
    const long long expected[] = {6, 6, 5, 19};
    for (size_t i = 0; i < sizeof functions / sizeof *functions; ++i) {
        Value result = call_function(functions[i], NULL, 0, ctx.env);
        ASSERT_EQ(result.type, VAL_INT); ASSERT_EQ(result.as.int_val, expected[i]);
        printf("I preserve %s after nested lexical cleanup.\n", functions[i]);
    }
    Value result = call_function("try_case", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_UNION); ASSERT(strcmp(result.as.union_val->variant_name, "Ok") == 0);
    ASSERT(strcmp(result.as.union_val->field_values[0].as.string_val, "before") == 0);
    ASSERT(!result.is_return); retire_returned_union(result);
    result = call_function("try_error_case", NULL, 0, ctx.env);
    ASSERT_EQ(result.type, VAL_UNION); ASSERT(strcmp(result.as.union_val->variant_name, "Err") == 0);
    ASSERT_EQ(result.as.union_val->field_values[0].as.int_val, 17);
    ASSERT(!result.is_return); retire_returned_union(result);
    run_ctx_free(&ctx); free(source);
    puts("I execute all scalar projection source shadows and caller-return checks.");
    return 0;
}
