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

/* I compare selection with an independent exhaustive scan, including the
 * priority of located locals, imports and unlocated runtime bindings. */
static Symbol *linear_visible(Environment *env, const char *name, int line, int column) {
    if (line <= 0) return linear(env, name);
    Symbol *located = NULL, *unknown = NULL;
    for (int i = 0; i < env->symbol_count; ++i) {
        Symbol *sym = &env->symbols[i];
        if (!sym->name || strcmp(sym->name, name)) continue;
        if (sym->scope_end_line > 0 && (line > sym->scope_end_line ||
            (line == sym->scope_end_line && column >= sym->scope_end_column))) continue;
        if (sym->def_line <= 0) unknown = sym;
        int start = sym->flow_start_line > 0 ? sym->flow_start_line : sym->def_line;
        int col = sym->flow_start_line > 0 ? sym->flow_start_column : sym->def_column;
        if (start <= 0 || start > line || (start == line && column > 0 && col > column)) continue;
        if (sym->def_file && env->current_file && strcmp(sym->def_file, env->current_file)) continue;
        located = sym;
    }
    if (located) return located;
    Symbol *imported = env_global_import_symbol(env, name);
    return imported ? imported : unknown;
}

static void check_visible_index(Environment *env) {
    int mark = env->symbol_count;
    ASTNode declaration = {0};
    env_set_current_file(env, "import-owner.nano");
    env_define_var(env, "owner_value", TYPE_INT, false, create_int(91));
    env_get_var(env, "owner_value")->global_declaration = &declaration;
    assert(env_import_global(env, "a.nano", "selected", &declaration));
    for (int round = 0; round < 12; ++round) {
        int inner = env->symbol_count;
        for (int i = 0; i < 24; ++i) {
            env_set_current_file(env, i % 2 ? "a.nano" : "b.nano");
            env_define_var(env, "selected", TYPE_INT, false, create_int(i));
            Symbol *sym = env_get_var(env, "selected");
            sym->def_line = i % 4 ? i + 3 : 0;
            sym->def_column = 4;
            sym->flow_start_line = i % 3 ? 0 : i + 1;
            sym->flow_start_column = 2;
            sym->scope_end_line = i % 5 ? i + 8 : 0;
            sym->scope_end_column = 6;
        }
        const char *files[] = {"a.nano", "b.nano", "unrelated.nano", NULL};
        for (size_t file = 0; file < sizeof files / sizeof *files; ++file) {
            env_set_current_file(env, files[file]);
            for (int line = 0; line < 36; ++line)
                for (int col = 0; col < 8; ++col)
                    assert(env_get_var_visible_at(env, "selected", line, col) ==
                           linear_visible(env, "selected", line, col));
        }
        pop_to(env, inner);
        env_set_current_file(env, "a.nano");
        env_define_var(env, "reused_visible_slot", TYPE_INT, false, create_int(round));
        assert(env_get_var_visible_at(env, "selected", 40, 1)->value.as.int_val == 91);
        assert(env_get_var_visible_at(env, "reused_visible_slot", 40, 1)->value.as.int_val == round);
        pop_to(env, inner);
    }
    for (int failure = 1; failure <= 3; ++failure) {
        env_symbol_index_invalidate(env);
        index_allocation_number = 0;
        fail_index_allocation = failure;
        assert(env_get_var_visible_at(env, "name_0", 1, 1) == linear(env, "name_0"));
        assert(index_allocation_number == failure);
        fail_index_allocation = 0;
    }
    name_comparisons = 0;
    for (int i = 0; i < 1000; ++i) {
        assert(env_get_var_visible_at(env, "name_0", 1, 1) == linear(env, "name_0"));
        assert(!env_get_var_visible_at(env, "missing_visible", 1, 1));
    }
    printf("I perform 2000 source-position lookups with %zu name comparisons.\n", name_comparisons);
    assert(name_comparisons < 64000);
    pop_to(env, mark);
}

/* I compare indexed function candidates with declaration-order selection. */
static Function *linear_function(Environment *env, const char *name, const char *owner,
                                  bool owned, bool body) {
    for (int i = 0; i < env->function_count; ++i) {
        Function *f = &env->functions[i];
        if (!f->name || strcmp(f->name, name)) continue;
        if (body && (f->is_extern || !f->body)) continue;
        if (owned && ((!owner != !f->module_name) ||
            (owner && strcmp(owner, f->module_name)))) continue;
        return f;
    }
    return NULL;
}

static void check_function_index(void) {
    Environment *env = create_environment();
    char names[513][32];
    char *owners[] = {NULL, "alpha", "beta"};
    ASTNode body = {0};
    for (int i = 0; i < 513; ++i) {
        snprintf(names[i], sizeof names[i], "indexed_fn_%d", i % 97);
        Function f = {0};
        f.name = names[i]; f.module_name = owners[i % 3];
        f.is_extern = i % 5 == 0; f.body = i % 7 ? &body : NULL;
        env_define_function(env, f);
        /* I force lookup across every append and function-table reallocation. */
        assert(function_named(env, names[i], NULL, false, false) ==
               linear_function(env, names[i], NULL, false, false));
    }
    env_define_function(env, (Function){0});
    for (int owner = 0; owner < 3; ++owner) {
        env->current_module = owners[owner];
        for (int i = 0; i < 513; ++i) {
            for (int owned = 0; owned < 2; ++owned)
                for (int body_only = 0; body_only < 2; ++body_only)
                    assert(function_named(env, names[i], owners[owner], owned, body_only) ==
                           linear_function(env, names[i], owners[owner], owned, body_only));
            Function *local = linear_function(env, names[i], owners[owner], true, false);
            assert(env_get_function(env, names[i]) == (local ? local :
                   linear_function(env, names[i], NULL, false, false)));
        }
        char **exports = malloc(sizeof *exports); exports[0] = strdup(names[0]);
        env_register_namespace(env, "Selected", "beta", exports, 1, NULL, 0, NULL, 0, NULL, 0);
        assert(env_get_function(env, "Selected.indexed_fn_0") ==
               linear_function(env, names[0], "beta", true, false));
        assert(!env_get_function(env, "Selected.missing"));
    }
    env->current_module = "unregistered_owner";
    assert(!env_get_function(env, "Selected.indexed_fn_0"));
    assert(!env_get_function(env, "absent_function"));
    Function *builtin = env_get_function(env, "abs"); assert(builtin);
    env_define_function(env, (Function){.name="abs", .body=&body});
    assert(env_get_function(env, "abs") == builtin);
    env->current_module = "alpha";
    env_define_function(env, (Function){.name="array_push", .module_name="alpha", .body=&body});
    assert(env_get_function(env, "array_push") == &env->functions[env->function_count-1]);
    env->current_module = "beta";
    assert(env_get_function(env, "array_push") != &env->functions[env->function_count-1]);
    /* I read changed ownership/body data directly rather than caching selection. */
    env->functions[0].module_name = "beta"; env->functions[0].body = &body;
    assert(function_named(env, names[0], "beta", true, false) == &env->functions[0]);
    for (int failure = 1; failure <= 3; ++failure) {
        function_index_free(env);
        fail_index_allocation = failure; index_allocation_number = 0;
        assert(function_named(env, names[0], "beta", true, false) == &env->functions[0]);
        assert(!env->function_index);
        fail_index_allocation = 0;
    }
    assert(!function_named(env, "absent_function", NULL, false, false));
    free_environment(env);
}

int main(void) {
    gc_init();
    check_function_index();
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

    check_visible_index(env);

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
    assert(strcmp(env_get_var(env, "record")->struct_type_name, "OwnerA") == 0);
    pop_to(env, outer);
    assert(!env_get_var(env, "record"));

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
