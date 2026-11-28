/* I observe actual live values before my production tracer formats them. */
#define main original_eval_suite_main
#include "test_eval.c"
#undef main
#include "../src/tracing.h"

static size_t observed_assignments;
static bool observe_values;

void nano_test_trace_var_set(const char *name, Value old_value, Value new_value,
                             int line, int column, const char *scope) {
    if (observe_values) {
        ASSERT_EQ(old_value.type, new_value.type);
        if (old_value.type == VAL_STRING) {
            ASSERT(strcmp(old_value.as.string_val, "before") == 0 || strcmp(old_value.as.string_val, "after") == 0);
            ASSERT(strcmp(new_value.as.string_val, "after") == 0);
        } else if (old_value.type == VAL_FUNCTION) {
            ASSERT(strcmp(old_value.as.function_val.function_name, "first") == 0 || strcmp(old_value.as.function_val.function_name, "second") == 0);
            ASSERT(strcmp(new_value.as.function_val.function_name, "second") == 0);
            ASSERT_EQ(old_value.as.function_val.signature->param_types[0], TYPE_INT);
            ASSERT_EQ(new_value.as.function_val.signature->return_type, TYPE_INT);
        } else if (old_value.type == VAL_STRUCT) {
            Value old_text = eval_test_record_field(old_value, "text");
            Value new_text = eval_test_record_field(new_value, "text");
            Value old_action = eval_test_record_field(old_value, "action");
            ASSERT(strcmp(old_text.as.string_val, "old-record") == 0 || strcmp(old_text.as.string_val, "new-record") == 0);
            ASSERT(strcmp(new_text.as.string_val, "new-record") == 0);
            ASSERT(strcmp(old_action.as.function_val.function_name, "first") == 0 || strcmp(old_action.as.function_val.function_name, "second") == 0);
            ASSERT_EQ(old_action.as.function_val.signature->param_types[0], TYPE_INT);
        } else {
            ASSERT_EQ(old_value.type, VAL_INT);
            ASSERT_EQ(old_value.as.int_val, 1);
            ASSERT_EQ(new_value.as.int_val, 2);
        }
        observed_assignments++;
    }
    trace_var_set(name, old_value, new_value, line, column, scope);
}

static char *capture_changes(RunCtx *ctx) {
    FILE *capture = tmpfile(); ASSERT(capture);
    ASSERT_EQ(fflush(stdout), 0);
    int original = dup(STDOUT_FILENO); ASSERT(original >= 0);
    ASSERT_EQ(dup2(fileno(capture), STDOUT_FILENO), STDOUT_FILENO);
    Value result = call_function("change", NULL, 0, ctx->env);
    ASSERT_EQ(result.type, VAL_INT); ASSERT_EQ(result.as.int_val, 9);
    ASSERT_EQ(fflush(stdout), 0);
    ASSERT_EQ(dup2(original, STDOUT_FILENO), STDOUT_FILENO); close(original);
    ASSERT_EQ(fseek(capture, 0, SEEK_END), 0);
    long size = ftell(capture); ASSERT(size >= 0);
    rewind(capture);
    char *text = malloc((size_t)size + 1); ASSERT(text);
    ASSERT_EQ(fread(text, 1, (size_t)size, capture), (size_t)size);
    text[size] = '\0'; fclose(capture);
    return text;
}

static void run_case(bool enabled, bool filtered) {
    const char *source =
        "struct Payload { text: string, action: fn(int) -> int }\n"
        "fn first(x: int) -> int { return x }\n"
        "shadow first { assert (== (first 3) 3) }\n"
        "fn second(x: int) -> int { return (+ x 1) }\n"
        "shadow second { assert (== (second 3) 4) }\n"
        "let mut text: string = \"before\"\n"
        "let mut action: fn(int) -> int = first\n"
        "let mut payload: Payload = Payload { text: \"old-record\", action: first }\n"
        "let mut number: int = 1\n"
        "fn change() -> int { set text \"after\" set action second set payload Payload { text: \"new-record\", action: second } set number 2 set text text set action action set payload payload return 9 }\n"
        "shadow change { assert (== (change) 9) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n";
    RunCtx ctx; ASSERT(run_ctx_init(&ctx, source));
    ASSERT(run_shadow_tests(ctx.program, ctx.env, false));
    run_ctx_free(&ctx);
    ASSERT(run_ctx_init(&ctx, source));
    tracing_init();
    g_tracing_config.enabled = enabled;
    g_tracing_config.trace_variables = true;
    if (filtered) {
        g_tracing_config.variable_filter_count = 1;
        g_tracing_config.variable_filters = malloc(sizeof(char *));
        g_tracing_config.variable_filters[0] = strdup("number");
    }
    observe_values = true; observed_assignments = 0;
    char *output = capture_changes(&ctx);
    observe_values = false;
    ASSERT_EQ(observed_assignments, 7);
    size_t sets = 0;
    for (const char *p = output; (p = strstr(p, " VAR_SET ")) != NULL; p++) sets++;
    ASSERT_EQ(sets, enabled ? (filtered ? 1 : 7) : 0);
    if (enabled) {
        ASSERT(strstr(output, "  old_value: 1\n  new_value: 2\n"));
        if (!filtered) {
            ASSERT(strstr(output, "  old_value: \"before\"\n  new_value: \"after\"\n"));
            ASSERT(strstr(output, "  old_value: \"after\"\n  new_value: \"after\"\n"));
            ASSERT(strstr(output, "VAR_SET action:"));
            ASSERT(strstr(output, "VAR_SET payload:"));
        } else {
            ASSERT(!strstr(output, "VAR_SET text:"));
            ASSERT(!strstr(output, "VAR_SET action:"));
            ASSERT(!strstr(output, "VAR_SET payload:"));
        }
    } else ASSERT_EQ(strlen(output), 0);
    fputs(output, stdout); free(output);
    g_tracing_config.enabled = false;
    tracing_cleanup(); memset(&g_tracing_config, 0, sizeof g_tracing_config);
    run_ctx_free(&ctx);
    printf("I preserve live assignment values with tracing enabled=%d filtered=%d.\n", enabled, filtered);
}

static void runtime_type_labels(void) {
    const ValueType kinds[] = {VAL_INT, VAL_FLOAT, VAL_BOOL, VAL_STRING, VAL_ARRAY,
        VAL_DYN_ARRAY, VAL_STRUCT, VAL_GC_STRUCT, VAL_UNION, VAL_FUNCTION,
        VAL_TUPLE, VAL_COROUTINE, VAL_VOID, VAL_BSTRING};
    const char *labels[] = {"int", "float", "bool", "string", "array", "dyn_array",
        "struct", "gc_struct", "union", "function", "tuple", "coroutine", "void", "bstring"};
    tracing_init();
    g_tracing_config.enabled = g_tracing_config.trace_functions = true;
    FILE *capture = tmpfile(); ASSERT(capture);
    ASSERT_EQ(fflush(stdout), 0);
    int original = dup(STDOUT_FILENO); ASSERT(original >= 0);
    ASSERT_EQ(dup2(fileno(capture), STDOUT_FILENO), STDOUT_FILENO);
    const char *names[] = {"input"};
    for (size_t i = 0; i < sizeof kinds / sizeof *kinds; ++i) {
        Value value = create_void(); value.type = kinds[i];
        memset(&value.as, 0, sizeof value.as);
        trace_function_call("runtime_kind", &value, 1, names, 1, 1);
    }
    ASSERT_EQ(fflush(stdout), 0);
    ASSERT_EQ(dup2(original, STDOUT_FILENO), STDOUT_FILENO); close(original);
    ASSERT_EQ(fseek(capture, 0, SEEK_END), 0);
    long size = ftell(capture); ASSERT(size >= 0); rewind(capture);
    char *output = malloc((size_t)size + 1); ASSERT(output);
    ASSERT_EQ(fread(output, 1, (size_t)size, capture), (size_t)size);
    output[size] = '\0'; fclose(capture);
    for (size_t i = 0; i < sizeof kinds / sizeof *kinds; ++i) {
        char expected[128];
        snprintf(expected, sizeof expected, "name: \"input\", type: \"%s\",", labels[i]);
        ASSERT(strstr(output, expected));
    }
    fputs(output, stdout); free(output);
    tracing_cleanup(); memset(&g_tracing_config, 0, sizeof g_tracing_config);
    puts("I label all fourteen runtime kinds independently of declaration ordinals.");
}

int main(void) {
    ASSERT_EQ(setvbuf(stdout, NULL, _IONBF, 0), 0);
    run_case(true, false);
    run_case(true, true);
    run_case(false, false);
    runtime_type_labels();
    return 0;
}
