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
#include "../src/eval/record_carrier_cleanup.inc"

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
        assert(function_named(env, names[i], NULL, false, false, 0, 0) ==
               linear_function(env, names[i], NULL, false, false));
    }
    env_define_function(env, (Function){0});
    for (int owner = 0; owner < 3; ++owner) {
        env->current_module = owners[owner];
        for (int i = 0; i < 513; ++i) {
            for (int owned = 0; owned < 2; ++owned)
                for (int body_only = 0; body_only < 2; ++body_only)
                    assert(function_named(env, names[i], owners[owner], owned, body_only, 0, 0) ==
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
    assert(function_named(env, names[0], "beta", true, false, 0, 0) == &env->functions[0]);
    for (int failure = 1; failure <= 3; ++failure) {
        function_index_free(env);
        fail_index_allocation = failure; index_allocation_number = 0;
        assert(function_named(env, names[0], "beta", true, false, 0, 0) == &env->functions[0]);
        assert(!env->function_index);
        fail_index_allocation = 0;
    }
    assert(!function_named(env, "absent_function", NULL, false, false, 0, 0));
    free_environment(env);
}

/* I retain every immutable builtin signature and refuse near-miss names. */
static void check_builtin_registry_selection(void) {
    Environment *env = create_environment();
    assert(!env_get_function(env, NULL));
    assert(!env_get_function(env, ""));
    for (int i = 0; i < builtin_registry_count; ++i) {
        const BuiltinEntry *entry = &builtin_registry[i];
        if (!(entry->flags & BUILTIN_LANG)) continue;
        Function *function = env_get_function(env, entry->name);
        assert(function && !strcmp(function->name, entry->name));
        assert(function->param_count == entry->arity);
        assert(function->return_type == entry->return_type);
        char missing[256];
        assert(snprintf(missing, sizeof missing, "%s_missing", entry->name) < (int)sizeof missing);
        assert(!env_get_function(env, missing));
    }
    name_comparisons = 0;
    for (int i = 0; i < 1000; ++i)
        assert(!env_get_function(env, "nisa_ordinary_source_function"));
    size_t matching_first_bytes = 0;
    for (int i = 0; i < builtin_registry_count; ++i)
        if ((builtin_registry[i].flags & BUILTIN_LANG) && builtin_registry[i].name[0] == 'n')
            ++matching_first_bytes;
    assert(name_comparisons == 1000 * matching_first_bytes);
    free_environment(env);
}

/* I keep metadata detached and retain both carrier storage/cleanup forms. */
static void check_record_carriers(void) {
    GCStats before = gc_get_stats();
    char name[] = "é";
    char field[] = "茶";
    char *fields[] = {field};
    Value text = create_string("detached");
    Value record = create_struct(name, fields, &text, 1);
    name[0] = 'x'; field[0] = 'x';
    gc_release(text.as.string_val);
    assert(!strcmp(record.as.struct_val->struct_name, "é"));
    assert(!strcmp(record.as.struct_val->field_names[0], "茶"));
    assert(!strcmp(record.as.struct_val->field_values[0].as.string_val, "detached"));
    env_free_value(record);
    env_free_value(create_struct("", NULL, NULL, 0));

    char *nested_names[] = {"inner"};
    Value source = create_string("nested");
    Value inner = create_struct("Inner", nested_names, &source, 1);
    Value outer = create_struct("Outer", nested_names, &inner, 1);
    Value held = outer.as.struct_val->field_values[0];
    assert(held.as.struct_val != inner.as.struct_val);
    gc_release(source.as.string_val);
    env_free_value(inner);
    assert(!strcmp(held.as.struct_val->field_values[0].as.string_val, "nested"));
    /* I retain the existing explicit nested-carrier cleanup boundary. */
    env_free_value(held);
    env_free_value(outer);

    Value array = create_array(VAL_STRUCT, 2, 2);
    StructValue **items = array.as.array_val->data;
    Value value = create_string("packed");
    items[0] = create_struct("Packed", nested_names, &value, 1).as.struct_val;
    gc_release(value.as.string_val);
    /* I also accept caller-created, separately allocated metadata. */
    items[1] = calloc(1, sizeof *items[1]);
    items[1]->struct_name = strdup("Legacy");
    items[1]->field_count = 1;
    items[1]->field_names = malloc(sizeof(char *));
    items[1]->field_names[0] = strdup("held");
    items[1]->field_values = malloc(sizeof(Value));
    items[1]->field_values[0] = create_string("legacy");
    EnvArrayReclaim reclaim = {0};
    env_release_array(&reclaim, array.as.array_val);
    free(reclaim.freed);
    GCStats after = gc_get_stats();
    assert(after.current_usage == before.current_usage && after.num_objects == before.num_objects);
}

/* I reclaim owned nested literals while preserving their borrowed arrays. */
static void check_record_literal_cleanup(void) {
    GCStats before = gc_get_stats();
    Value shared = create_array(VAL_INT, 1, 1);
    ((long long *)shared.as.array_val->data)[0] = 42;
    Value original = create_string("held");
    char *inner_names[] = {"text", "shared"};
    Value inner_values[] = {original, shared};
    Value inner = create_struct("Inner", inner_names, inner_values, 2);
    char *outer_names[] = {"nested"};
    Value outer = create_struct("Outer", outer_names, &inner, 1);
    discard_literal_record(inner.as.struct_val);
    assert(!strcmp(outer.as.struct_val->field_values[0].as.struct_val->field_values[0].as.string_val, "held"));
    discard_literal_record(outer.as.struct_val);
    assert(!strcmp(original.as.string_val, "held"));
    assert(shared.as.array_val->length == 1);
    assert(((long long *)shared.as.array_val->data)[0] == 42);
    gc_release(original.as.string_val);
    free(shared.as.array_val->data); free(shared.as.array_val);
    StructValue *legacy = calloc(1, sizeof *legacy);
    legacy->struct_name = strdup("Legacy"); legacy->field_count = 1;
    legacy->field_names = malloc(sizeof(char *)); legacy->field_names[0] = strdup("plain");
    legacy->field_values = malloc(sizeof(Value));
    legacy->field_values[0] = create_void();
    legacy->field_values[0].type = VAL_STRING;
    legacy->field_values[0].as.string_val = strdup("unmanaged");
    discard_literal_record(legacy);
    Value empty = create_struct("", NULL, NULL, 0);
    discard_literal_record(empty.as.struct_val);
    GCStats after = gc_get_stats();
    assert(after.current_usage == before.current_usage && after.num_objects == before.num_objects);
}

/* I preserve borrowed aliases and metadata beside constructor-only elision. */
static void check_evaluated_string_bindings(void) {
    Environment *env = create_environment();
    TypeInfo info = {.base_type = TYPE_STRING};
    ASTNode literal = {.type = AST_STRING};
    ASTNode identifier = {.type = AST_IDENTIFIER};
    ASTNode call = {.type = AST_CALL};
    call.as.call.name = "str_concat";
    ASTNode addition = {.type = AST_PREFIX_OP};
    addition.as.prefix_op.op = TOKEN_PLUS;
    addition.as.prefix_op.arg_count = 2;
    Value original = create_string("held");
    char *pointer = original.as.string_val;
    env_define_evaluated_var(env, "source", TYPE_STRING, TYPE_UNKNOWN, &info, true, original, &literal);
    Symbol *source = env_get_var(env, "source");
    assert(source->type == TYPE_STRING && source->element_type == TYPE_UNKNOWN);
    assert(source->type_info == &info && source->is_mut);
    assert(source->value.as.string_val == pointer && gc_get_header(pointer)->ref_count == 1);
    env_define_evaluated_var(env, "alias", TYPE_STRING, TYPE_UNKNOWN, &info, false, original, &identifier);
    env_define_evaluated_var(env, "call_alias", TYPE_STRING, TYPE_UNKNOWN, &info, false, original, &call);
    assert(gc_get_header(pointer)->ref_count == 3);
    Value target = create_string("replaced");
    char *old_target = target.as.string_val;
    env_define_evaluated_var(env, "target", TYPE_STRING, TYPE_UNKNOWN, &info, true, target, &literal);
    env_set_evaluated_var(env, "target", original, &call);
    assert(!gc_is_managed(old_target) && gc_get_header(pointer)->ref_count == 4);
    env_set_evaluated_var(env, "source", create_string("replacement"), &addition);
    assert(gc_get_header(pointer)->ref_count == 3);
    assert(!strcmp(env_get_var(env, "source")->value.as.string_val, "replacement"));
    assert(!strcmp(env_get_var(env, "alias")->value.as.string_val, "held"));
    char boundary;
    Value returned = original;
    returned.is_return = true; returned.return_target = &boundary;
    env_define_evaluated_var(env, "signal", TYPE_STRING, TYPE_UNKNOWN, &info, false, returned, &addition);
    assert(gc_get_header(pointer)->ref_count == 4);
    assert(env_get_var(env, "signal")->value.return_target == &boundary);
    addition.as.prefix_op.arg_count = 1;
    env_define_evaluated_var(env, "unary", TYPE_STRING, TYPE_UNKNOWN, &info, false, original, &addition);
    assert(gc_get_header(pointer)->ref_count == 5);
    env_set_evaluated_var(env, "target", returned, &identifier);
    assert(gc_get_header(pointer)->ref_count == 5);
    assert(env_get_var(env, "target")->value.is_return);
    assert(env_get_var(env, "target")->value.return_target == &boundary);
    free_environment(env);
    assert(!gc_is_managed(pointer));
}

/* I transfer completed local literals and preserve every borrowed/copy route. */
static void check_evaluated_record_bindings(void) {
    GCStats before = gc_get_stats();
    Environment *env = create_environment();
    ASTNode literal = {.type = AST_STRUCT_LITERAL};
    ASTNode identifier = {.type = AST_IDENTIFIER};
    ASTNode call = {.type = AST_CALL};
    TypeInfo info = {.base_type = TYPE_STRUCT};
    char owner[] = "owner.nano";
    env_set_current_file(env, owner);
    Value array = create_array(VAL_INT, 1, 1);
    ((long long *)array.as.array_val->data)[0] = 42;
    Value fields[] = {create_string("owned"), array};
    char *names[] = {"茶", "array"};
    Value inner = create_struct("Inner", names, fields, 2);
    gc_release(fields[0].as.string_val);
    char *outer_names[] = {"inner"};
    Value fresh = create_struct("é", outer_names, &inner, 1);
    discard_literal_record(inner.as.struct_val);
    env_define_evaluated_var(env, "fresh", TYPE_STRUCT, TYPE_UNKNOWN, &info,
                             true, fresh, &literal);
    Symbol *bound = env_get_var(env, "fresh");
    assert(bound->value.as.struct_val == fresh.as.struct_val);
    assert(bound->type_info == &info && bound->is_mut && bound->def_file == owner);
    assert(!strcmp(bound->value.as.struct_val->struct_name, "é"));
    env_define_evaluated_var(env, "copy", TYPE_STRUCT, TYPE_UNKNOWN, &info,
                             false, fresh, &identifier);
    StructValue *copy = env_get_var(env, "copy")->value.as.struct_val;
    assert(copy != fresh.as.struct_val);
    StructValue *copied_inner = copy->field_values[0].as.struct_val;
    StructValue *original_inner = fresh.as.struct_val->field_values[0].as.struct_val;
    assert(copied_inner != original_inner);
    assert(copied_inner->field_values[0].as.string_val != original_inner->field_values[0].as.string_val);
    assert(copied_inner->field_values[1].as.array_val == array.as.array_val);
    ((long long *)array.as.array_val->data)[0] = 71;
    assert(((long long *)copied_inner->field_values[1].as.array_val->data)[0] == 71);
    bound = env_get_var(env, "fresh");
    bound->value = create_void();
    discard_literal_record(fresh.as.struct_val);
    assert(!strcmp(copied_inner->field_values[0].as.string_val, "owned"));
    discard_literal_record(copy);
    env_get_var(env, "copy")->value = create_void();
    for (int route = 0; route < 7; ++route) {
        fresh = create_struct("", NULL, NULL, 0);
        ASTNode *expression = route == 0 ? &call : route == 1 ? NULL : &literal;
        fresh.is_return = route == 2;
        fresh.is_break = route == 3;
        fresh.is_continue = route == 4;
        if (route == 5) {
            env_define_var(env, "route", TYPE_STRUCT, false, fresh);
        } else if (route == 6) {
            env_define_var_with_type_info(env, "route", TYPE_STRUCT, TYPE_UNKNOWN,
                                          &info, false, fresh);
        } else {
            env_define_evaluated_var(env, "route", TYPE_STRUCT, TYPE_UNKNOWN,
                                     &info, false, fresh, expression);
        }
        Value held = env_get_var(env, "route")->value;
        assert(held.type == VAL_STRUCT && held.as.struct_val != fresh.as.struct_val);
        discard_literal_record(fresh.as.struct_val);
        discard_literal_record(held.as.struct_val);
        env_get_var(env, "route")->value = create_void();
    }
    fresh = create_struct("Borrowed", NULL, NULL, 0);
    for (int mut = 0; mut < 2; ++mut) {
        env_define_evaluated_var(env, "borrow", mut ? TYPE_BORROW_MUT : TYPE_BORROW_SHARED,
                                 TYPE_UNKNOWN, NULL, false, fresh, &literal);
        assert(env_get_var(env, "borrow")->value.as.struct_val == fresh.as.struct_val);
        env_get_var(env, "borrow")->value = create_void();
    }
    discard_literal_record(fresh.as.struct_val);
    fresh = create_struct("", NULL, NULL, 0);
    env_define_evaluated_var(env, "empty", TYPE_STRUCT, TYPE_UNKNOWN, NULL, false,
                             fresh, &literal);
    assert(env_get_var(env, "empty")->value.as.struct_val == fresh.as.struct_val);
    free_environment(env);
    free(array.as.array_val->data);
    free(array.as.array_val);
    GCStats after = gc_get_stats();
    assert(after.current_usage == before.current_usage && after.num_objects == before.num_objects);
}

static void check_fresh_string_copy(void) {
    Environment *env = create_environment();
    Value caller = create_string("caller storage");
    char *caller_pointer = caller.as.string_val;
    assert(gc_get_header(caller_pointer)->ref_count == 1);
    env_define_string_copy(env, "copied", TYPE_STRING, false, caller_pointer);
    Symbol *copied = env_get_var(env, "copied");
    char *copy_pointer = copied->value.as.string_val;
    assert(copied->type == TYPE_STRING && !copied->is_mut);
    assert(copy_pointer != caller_pointer && !strcmp(copy_pointer, caller_pointer));
    assert(gc_get_header(caller_pointer)->ref_count == 1);
    assert(gc_get_header(copy_pointer)->ref_count == 1);
    gc_release(caller_pointer);
    assert(!gc_is_managed(caller_pointer));
    assert(gc_is_managed(copy_pointer) && !strcmp(copy_pointer, "caller storage"));
    env_define_var(env, "borrowed", TYPE_STRING, true, copied->value);
    assert(gc_get_header(copy_pointer)->ref_count == 2);
    env_set_var(env, "borrowed", create_string("replacement"));
    assert(gc_get_header(copy_pointer)->ref_count == 1);
    env_define_string_copy(env, "empty", TYPE_STRING, true, "");
    assert(!strcmp(env_get_var(env, "empty")->value.as.string_val, ""));
    free_environment(env);
    assert(!gc_is_managed(copy_pointer));
}

static void check_string_self_assignment(void) {
    Environment *env = create_environment();
    env_define_var(env, "text", TYPE_STRING, true, create_string("retained"));
    Value original = env_get_var(env, "text")->value;
    char *pointer = original.as.string_val;
    assert(gc_is_managed(pointer));
    assert(gc_get_header(pointer)->ref_count == 1);
    env_set_var(env, "text", original);
    assert(gc_is_managed(pointer));
    assert(gc_get_header(pointer)->ref_count == 1);
    assert(!strcmp(env_get_var(env, "text")->value.as.string_val, "retained"));

    char target;
    Value flagged = original;
    flagged.is_return = true;
    flagged.return_target = &target;
    env_set_var(env, "text", flagged);
    assert(env_get_var(env, "text")->value.is_return);
    assert(env_get_var(env, "text")->value.return_target == &target);
    assert(gc_get_header(pointer)->ref_count == 1);
    env_set_var(env, "text", original);
    env_define_var(env, "alias", TYPE_STRING, true, original);
    assert(gc_get_header(pointer)->ref_count == 2);
    env_set_var(env, "text", original);
    assert(gc_get_header(pointer)->ref_count == 2);
    env_set_var(env, "alias", create_string("new alias"));
    assert(gc_is_managed(pointer) && gc_get_header(pointer)->ref_count == 1);
    env_set_var(env, "text", create_string("replacement"));
    assert(!gc_is_managed(pointer));

    Value plain = create_void();
    plain.type = VAL_STRING;
    plain.as.string_val = strdup("ordinary allocation");
    assert(plain.as.string_val);
    env_define_var(env, "plain", TYPE_STRING, true, plain);
    env_set_var(env, "plain", plain);
    assert(!strcmp(env_get_var(env, "plain")->value.as.string_val, "ordinary allocation"));
    free_environment(env);
}

int main(void) {
    gc_init();
    check_function_index();
    check_evaluated_string_bindings();
    check_evaluated_record_bindings();
    check_record_carriers();
    check_record_literal_cleanup();
    check_builtin_registry_selection();
    check_string_self_assignment();
    check_fresh_string_copy();
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

    /* I select lexical functions by use position without exporting them. */
    ASTNode outer_scope = {.type=AST_BLOCK, .scope_end_line=30, .scope_end_column=1};
    ASTNode inner_scope = {.type=AST_BLOCK, .scope_end_line=25, .scope_end_column=1};
    ASTNode outer_body = {.type=AST_BLOCK, .line=12, .column=8};
    ASTNode inner_body = {.type=AST_BLOCK, .line=20, .column=8};
    int first_function = env->function_count;
    env_define_function(env, (Function){.name="lexical_probe", .return_type=TYPE_INT});
    env_define_function(env, (Function){.name="lexical_probe", .return_type=TYPE_INT,
        .body=&outer_body, .lexical_scope=&outer_scope, .source_file="lexical.nano"});
    env_define_function(env, (Function){.name="lexical_probe", .return_type=TYPE_INT,
        .body=&inner_body, .lexical_scope=&inner_scope, .source_file="lexical.nano"});
    env_set_current_file(env, "lexical.nano");
    assert(env_get_function_at(env,"lexical_probe",11,1)==&env->functions[first_function]);
    assert(env_get_function_at(env,"lexical_probe",12,7)==&env->functions[first_function]);
    assert(env_get_function_at(env,"lexical_probe",12,8)==&env->functions[first_function+1]);
    assert(env_get_function_at(env,"lexical_probe",21,1)==&env->functions[first_function+2]);
    assert(env_get_function_at(env,"lexical_probe",25,1)==&env->functions[first_function+1]);
    assert(env_get_function_at(env,"lexical_probe",30,1)==&env->functions[first_function]);
    env_set_current_file(env, "other.nano");
    assert(env_get_function_at(env,"lexical_probe",21,1)==&env->functions[first_function]);
    assert(env_get_function(env,"lexical_probe")==&env->functions[first_function]);

    pop_to(env, 0);
    assert(!env_get_var(env, "imported"));
    env_define_var(env, "after_reset", TYPE_INT, false, create_int(9));
    assert(env_get_var(env, "after_reset")->value.as.int_val == 9);
    free_environment(env);
    gc_shutdown();
    puts("I preserve indexed symbol scope, metadata, reset and fallback semantics.");
    return 0;
}
