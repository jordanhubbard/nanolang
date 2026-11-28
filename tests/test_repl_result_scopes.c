/* I measure actual REPL result storage before the GC registry can hide it. */
#include "../src/repl.c"
#include "eval/carrier_runtime.h"
#include "runtime/gc.h"

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "I failed %s at line %d.\n", #condition, __LINE__); exit(1); \
} } while (0)

static void submit(Environment *env, const char *source) {
    REQUIRE(recorded_source(REPL_INPUT, "<repl>", strdup(source), env, false));
}

static void require_live(GCStats expected) {
    GCStats actual = gc_get_stats();
    if (actual.current_usage != expected.current_usage || actual.num_objects != expected.num_objects) {
        fprintf(stderr, "I retain %zu bytes/%zu objects; I require %zu/%zu.\n",
                actual.current_usage, actual.num_objects,
                expected.current_usage, expected.num_objects);
        exit(1);
    }
}

int main(void) {
    gc_init();
    GCStats empty = gc_get_stats();
    char *text = malloc(2 * 1024 * 1024 + 1);
    REQUIRE(text);
    memset(text, 'p', 2 * 1024 * 1024);
    text[2 * 1024 * 1024] = '\0';
    Value control = create_string(text);
    GCStats positive = gc_get_stats();
    REQUIRE(positive.current_usage >= empty.current_usage + 2 * 1024 * 1024);
    REQUIRE(positive.num_objects == empty.num_objects + 1);
    gc_release(control.as.string_val);
    free(text);
    require_live(empty);
    printf("I observe and retire my 2MiB positive control: %zu bytes.\n",
           positive.current_usage - empty.current_usage);
    fflush(stdout);

    Environment *env = create_environment();
    REQUIRE(env_carrier_enable(env));
    REQUIRE(env_carrier_enable_collection(env, NULL, 0));
    FILE *sink = tmpfile();
    REQUIRE(sink);
    int original_stdout = dup(STDOUT_FILENO);
    REQUIRE(original_stdout >= 0 && dup2(fileno(sink), STDOUT_FILENO) >= 0);
    submit(env, "let anchor: string = \"retained\"");
    Symbol *binding = env_get_var(env, "anchor");
    REQUIRE(binding && binding->value.type == VAL_STRING);
    char *anchor = binding->value.as.string_val;
    GCStats bound = gc_get_stats();
    REQUIRE(bound.num_objects == empty.num_objects + 1);
    submit(env, "fn display_target() -> string { return \"returned\" }");
    submit(env, "shadow display_target { assert (== (display_target) \"returned\") }");
    for (int round = 0; round < 128; ++round) {
        submit(env, "\"fresh-display\"");
        require_live(bound);
        submit(env, "(+ \"left\" \"right\")");
        require_live(bound);
        submit(env, "(display_target)");
        require_live(bound);
        submit(env, "anchor");
        REQUIRE(env_get_var(env, "anchor")->value.as.string_val == anchor);
        REQUIRE(!strcmp(anchor, "retained"));
        require_live(bound);
    }
    env_clear_variables(env);
    REQUIRE(env_carrier_collect(env));
    require_live(empty);
    free_environment(env);
    repl_free_all();
    repl_history_free();
    clear_module_cache();
    require_live(empty);
    fflush(stdout);
    REQUIRE(dup2(original_stdout, STDOUT_FILENO) >= 0);
    close(original_stdout);
    REQUIRE(fseek(sink, 0, SEEK_SET) == 0);
    char line[256];
    unsigned fresh = 0, joined = 0, returned = 0, borrowed = 0;
    while (fgets(line, sizeof line, sink)) {
        if (!strcmp(line, "=> fresh-display\n")) ++fresh;
        if (!strcmp(line, "=> leftright\n")) ++joined;
        if (!strcmp(line, "=> returned\n")) ++returned;
        if (!strcmp(line, "=> retained\n")) ++borrowed;
    }
    REQUIRE(!ferror(sink));
    REQUIRE(fresh == 128 && joined == 128 && returned == 128 && borrowed == 128);
    REQUIRE(fclose(sink) == 0);
    gc_shutdown();
    puts("I retire all 512 displayed STRING results and preserve my borrowed binding.");
    return 0;
}
