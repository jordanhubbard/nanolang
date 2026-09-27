/* I check indexed lookup against authoritative slots, including allocation failure. */
#include "nanolang.h"
#include "runtime/gc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_index_allocation, index_allocation_number;
static size_t name_comparisons;
static void *index_calloc(size_t n, size_t size) {
    if (fail_index_allocation && ++index_allocation_number == fail_index_allocation) return NULL;
    return calloc(n, size);
}
static void *index_realloc(void *p, size_t size) {
    if (fail_index_allocation && ++index_allocation_number == fail_index_allocation) return NULL;
    return realloc(p, size);
}
static int compare_name(const char *a, const char *b) {
    ++name_comparisons;
    return strcmp(a, b);
}

/* I inject failures only in this test translation unit, with no runtime mode. */
#define calloc index_calloc
#define realloc index_realloc
#define safe_strcmp compare_name
#include "../src/env.c"
#undef calloc
#undef realloc
#undef safe_strcmp

int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static Symbol *linear(Environment *env, const char *name) {
    for (int i = env->symbol_count - 1; i >= 0; --i)
        if (env->symbols[i].name && strcmp(env->symbols[i].name, name) == 0)
            return &env->symbols[i];
    return NULL;
}

static void pop_to(Environment *env, int count) {
    for (int i = count; i < env->symbol_count; ++i) {
        free(env->symbols[i].name);
        free(env->symbols[i].struct_type_name);
    }
    env->symbol_count = count;
}

int main(void) {
    gc_init();
    Environment *env = create_environment();
    assert(!env_get_var(env, "absent"));
    for (int i = 0; i < 4096; ++i) {
        char name[32];
        snprintf(name, sizeof name, "name_%d", i);
        env_define_var(env, name, TYPE_INT, true, create_int(i));
    }
    name_comparisons = 0;
    for (int i = 0; i < 1000; ++i) {
        assert(!env_get_var(env, "ordinary_function_without_a_variable"));
        assert(env_get_var(env, "name_0")->value.as.int_val == 0);
    }
    printf("I perform 2000 lookups with %zu name comparisons.\n", name_comparisons);
    assert(name_comparisons < 64000);

    int outer = env->symbol_count;
    char first_file[] = "a.nano", equal_file[] = "a.nano";
    env_set_current_file(env, first_file);
    env_define_var(env, "record", TYPE_STRUCT, false, create_void());
    env_get_var(env, "record")->struct_type_name = strdup("OwnerA");
    env_set_current_file(env, "b.nano");
    env_define_var(env, "record", TYPE_STRUCT, false, create_void());
    assert(!env_get_var(env, "record")->struct_type_name);
    env_get_var(env, "record")->struct_type_name = strdup("OwnerB");
    env_set_current_file(env, equal_file);
    env_define_var(env, "record", TYPE_STRUCT, false, create_void());
    assert(!env_get_var(env, "record")->struct_type_name);
    assert(strcmp(env->symbols[outer].struct_type_name, "OwnerA") == 0);
    assert(strcmp(env->symbols[outer + 1].struct_type_name, "OwnerB") == 0);
    pop_to(env, outer);
    assert(!env_get_var(env, "record"));

    /* The same count after pop/reinsert must not preserve the old name index. */
    env_define_var(env, "retired_name", TYPE_INT, false, create_int(1));
    assert(env_get_var(env, "retired_name"));
    pop_to(env, outer);
    env_define_var(env, "replacement_name", TYPE_INT, false, create_int(2));
    assert(!env_get_var(env, "retired_name"));
    assert(env_get_var(env, "replacement_name") == linear(env, "replacement_name"));
    assert(env_get_var(env, "replacement_name")->value.as.int_val == 2);
    pop_to(env, outer);

    /* I repeatedly reuse freed slots and compare every name with reverse scan. */
    for (int round = 0; round < 80; ++round) {
        int mark = env->symbol_count;
        for (int i = 0; i < 37; ++i) {
            char name[32];
            snprintf(name, sizeof name, "name_%d", (round * 19 + i) % 4096);
            env_define_var(env, name, TYPE_INT, true, create_int(round + i));
            assert(env_get_var(env, name) == linear(env, name));
        }
        pop_to(env, mark);
        /* I insert before querying, so synchronization must precede slot reuse. */
        env_define_var(env, "reused", TYPE_INT, true, create_int(round));
        assert(env_get_var(env, "reused")->value.as.int_val == round);
        for (int i = 0; i < 80; ++i) {
            char name[32];
            snprintf(name, sizeof name, "name_%d", (round * 19 + i) % 4096);
            assert(env_get_var(env, name) == linear(env, name));
        }
        pop_to(env, mark);
    }

    /* Imported constants explicitly invalidate before an external slot write. */
    env_define_var(env, "old_slot", TYPE_INT, false, create_int(1));
    assert(env_get_var(env, "old_slot"));
    pop_to(env, outer);
    env_symbol_index_invalidate(env);
    Symbol imported = {0};
    imported.name = strdup("imported");
    imported.value = create_int(42);
    env->symbols[env->symbol_count++] = imported;
    assert(!env_get_var(env, "old_slot"));
    assert(env_get_var(env, "imported")->value.as.int_val == 42);

    env_symbol_index_invalidate(env);
    env->symbols[env->symbol_count++] = (Symbol){0};
    assert(env_get_var(env, "imported")->value.as.int_val == 42);
    for (int failure = 1; failure <= 3; ++failure) {
        env_symbol_index_invalidate(env);
        index_allocation_number = 0;
        fail_index_allocation = failure;
        assert(env_get_var(env, "imported") == linear(env, "imported"));
        assert(index_allocation_number == failure);
        assert(env_get_var(env, "name_0") == linear(env, "name_0"));
        assert(!env_get_var(env, "missing"));
        fail_index_allocation = 0;
    }
    assert(env_get_var(env, "name_0") == linear(env, "name_0"));
    /* Graph visibility does not rewrite source diagnostics or escape lexical/file bounds. */
    env_set_current_file(env, "flow.nano");
    env_define_var(env, "graph_scalar", TYPE_INT, false, create_int(7));
    Symbol *graph = env_get_var(env, "graph_scalar");
    graph->def_line = 20; graph->def_column = 9;
    graph->flow_start_line = 10; graph->flow_start_column = 5;
    graph->scope_end_line = 30; graph->scope_end_column = 1;
    assert(!env_get_var_visible_at(env, "graph_scalar", 9, 1));
    assert(!env_get_var_visible_at(env, "graph_scalar", 10, 4));
    assert(env_get_var_visible_at(env, "graph_scalar", 15, 1) == graph);
    assert(graph->def_line == 20 && graph->def_column == 9);
    assert(!env_get_var_visible_at(env, "graph_scalar", 30, 1));
    env_set_current_file(env, "other.nano");
    assert(!env_get_var_visible_at(env, "graph_scalar", 15, 1));
    env_set_current_file(env, "flow.nano");
    env_define_var(env, "ordinary_scalar", TYPE_INT, false, create_int(8));
    Symbol *ordinary = env_get_var(env, "ordinary_scalar");
    ordinary->def_line = 20; ordinary->def_column = 9;
    assert(!env_get_var_visible_at(env, "ordinary_scalar", 15, 1));
    assert(env_get_var_visible_at(env, "ordinary_scalar", 20, 9) == ordinary);

    pop_to(env, 0);
    assert(!env_get_var(env, "imported"));
    env_define_var(env, "after_reset", TYPE_INT, false, create_int(9));
    assert(env_get_var(env, "after_reset")->value.as.int_val == 9);
    free_environment(env);
    gc_shutdown();
    puts("I preserve indexed symbol scope, metadata, reset and fallback semantics.");
    return 0;
}
