/* I exercise the actual evaluator TU or an actual emitted split helper. */
#define _POSIX_C_SOURCE 200809L
#include "nanolang.h"
#include "runtime/gc.h"
#include "runtime/dyn_array.h"
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

static int split_active, split_calls, split_fail;
static char *split_string(size_t size);
static DynArray *split_array(ElementType type);
#define gc_alloc_string split_string
#define dyn_array_new split_array
#ifdef SPLIT_NATIVE_HEADER
#include SPLIT_NATIVE_HEADER
#else
#include "../src/eval.c"
#endif
#undef gc_alloc_string
#undef dyn_array_new
int g_argc;
char **g_argv;

static int split_refuse(const char *site) {
    if (!split_active) return 0;
    ++split_calls;
    if (split_fail != split_calls) return 0;
    fprintf(stderr, "SPLIT_FAULT %d %s\n", split_calls, site);
    return 1;
}
static char *split_string(size_t size) {
    return split_refuse("segment") ? NULL : gc_alloc_string(size);
}
static DynArray *split_array(ElementType type) {
    return split_refuse("array") ? NULL : dyn_array_new(type);
}
static DynArray *split_actual(char *source, char *delimiter) {
#ifdef SPLIT_NATIVE_HEADER
    return nl_str_split(source, delimiter);
#else
    Environment *environment = create_environment();
    assert(environment);
    ASTNode first = {0}, second = {0}, call = {0};
    Value source_value = {0}, delimiter_value = {0};
    source_value.type = VAL_STRING; source_value.as.string_val = source;
    delimiter_value.type = VAL_STRING; delimiter_value.as.string_val = delimiter;
    env_define_var(environment, "split_input", TYPE_STRING, false, source_value);
    env_define_var(environment, "split_delimiter", TYPE_STRING, false, delimiter_value);
    assert(env_get_var(environment, "split_input")->value.as.string_val == source);
    assert(env_get_var(environment, "split_delimiter")->value.as.string_val == delimiter);
    first.type = AST_IDENTIFIER; first.as.identifier = "split_input";
    second.type = AST_IDENTIFIER; second.as.identifier = "split_delimiter";
    ASTNode *arguments[] = {&first, &second};
    call.type = AST_CALL; call.as.call.name = "str_split";
    call.as.call.arg_count = 2; call.as.call.args = arguments;
    Value result = eval_call_impl(&call, environment, NULL);
    assert(result.type == VAL_DYN_ARRAY);
    /* I keep fixture ownership of these two exact borrowed input buffers. */
    env_get_var(environment, "split_input")->value = create_void();
    env_get_var(environment, "split_delimiter")->value = create_void();
    free_environment(environment);
    return result.as.dyn_array_val;
#endif
}
struct SplitCase { const char *input, *delimiter; int count; const char *parts[5]; };
static const struct SplitCase split_cases[] = {
    {"a,b,c", ",", 3, {"a", "b", "c"}},
    {"", "\n", 1, {""}},
    {"\né\n\nend\n", "\n", 5, {"", "é", "", "end", ""}},
    {"a\r\nb", "\n", 2, {"a\r", "b"}},
    {"abc", "", 3, {"a", "b", "c"}},
    {"", "", 0, {NULL}},
    {"same", "unmatched", 1, {"same"}},
    {"a::b::::", "::", 4, {"a", "b", "", ""}},
    {"a\nb", "\n", 2, {"a", "b"}}
};
int main(int argc, char **argv) {
    struct rlimit no_core = {0, 0};
    assert(setrlimit(RLIMIT_CORE, &no_core) == 0);
    assert(argc == 3);
    int index = atoi(argv[1]); split_fail = atoi(argv[2]);
    assert(index >= 0 && (size_t)index < sizeof split_cases / sizeof *split_cases);
    const struct SplitCase *test = &split_cases[index];
    gc_init();
    char *source = strdup(test->input), *delimiter = strdup(test->delimiter);
    assert(source && delimiter);
    if (index == 8) {
        static const char counted[] = "a\nb\0\nlost";
        free(source);
        source = malloc(sizeof counted);
        assert(source);
        memcpy(source, counted, sizeof counted);
    }
    split_active = 1;
    DynArray *result = split_actual(source, delimiter);
    split_active = 0;
    assert(split_fail == 0 || split_fail > split_calls);
    assert(result && result->elem_type == ELEM_STRING);
    assert(result->elem_size == sizeof(char *) && result->length == test->count);
    memset(source, 'x', strlen(source)); free(source);
    memset(delimiter, 'y', strlen(delimiter)); free(delimiter);
    /* I keep every segment alive after its input and unrelated allocations. */
    for (int i = 0; i < 128; ++i) { char *p = gc_alloc_string(32); assert(p); memset(p, 'z', 32); }
    for (int i = 0; i < test->count; ++i) {
        const char *part = dyn_array_get_string(result, i);
        assert(part && gc_is_managed((void *)part));
        assert(strcmp(part, test->parts[i]) == 0);
        for (int j = 0; j < i; ++j) assert(part != dyn_array_get_string(result, j));
    }
    assert(split_calls == test->count + 1);
    printf("{\"case\":%d,\"segments\":%d,\"allocations\":%d}\n", index, test->count, split_calls);
    gc_shutdown();
    return 0;
}
