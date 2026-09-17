/*
 * Symbol lookup must not compare source positions across files.
 *
 * Visibility is decided by "the most recent definition at or before this
 * line". A line number only means something inside the file it came from, so
 * comparing a position in one file against a definition in another returns
 * whichever unrelated symbol happens to sit at a lower line there.
 *
 * That is not hypothetical: it silently retyped an imported function's
 * parameters, and operator lowering picks the integer or the float opcode from
 * that type, so a float `lerp` compiled to I64_SUB and I64_ADD (issue #223).
 *
 * The end-to-end regression for that lives in tests/test_symbol_scoping.nano,
 * but it passes for a second reason too -- the parameter is the most recent
 * definition, so a backward scan finds it first regardless of file. These
 * tests pin the invariant itself, so it survives a change in the order
 * definitions happen to be added.
 */

#include "nanolang.h"
#include "runtime/gc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by the runtime this links against. */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, what) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s  (%s:%d)\n", (what), __FILE__, __LINE__); } \
} while (0)

static Symbol *define_at(Environment *env, const char *file, const char *name,
                         Type type, int line) {
    env_set_current_file(env, file);
    Value unset = create_void();
    env_define_var(env, name, type, false, unset);
    Symbol *sym = env_get_var(env, name);
    if (sym) {
        sym->def_line = line;
        sym->def_column = 1;
    }
    return sym;
}

/* The case that caused issue #223: a symbol defined earlier in another file
 * must not answer a lookup made while compiling this one. */
static void test_lookup_ignores_other_files(void) {
    Environment *env = create_environment();

    define_at(env, "main.nano", "a", TYPE_INT, 3);
    define_at(env, "module.nano", "a", TYPE_FLOAT, 40);

    /* Asking from module.nano at line 41: the float is the only candidate in
     * this file, and the int in main.nano sits at a lower line, which is
     * exactly the shape that used to win. */
    env_set_current_file(env, "module.nano");
    Symbol *found = env_get_var_visible_at(env, "a", 41, 1);
    CHECK(found != NULL, "a symbol is found");
    CHECK(found && found->type == TYPE_FLOAT,
          "the definition from this file wins, not the lower-numbered one elsewhere");

    /* And symmetrically from the other file. */
    env_set_current_file(env, "main.nano");
    found = env_get_var_visible_at(env, "a", 4, 1);
    CHECK(found != NULL, "a symbol is found from the other side");
    CHECK(found && found->type == TYPE_INT,
          "each file sees its own definition");

    free_environment(env);
}

/* A symbol defined in another file must not answer even when it is the only
 * located candidate: it is not visible here at all. */
static void test_other_file_is_not_visible(void) {
    Environment *env = create_environment();
    define_at(env, "other.nano", "only_there", TYPE_STRING, 2);

    env_set_current_file(env, "here.nano");
    Symbol *found = env_get_var_visible_at(env, "only_there", 99, 1);
    CHECK(found == NULL, "a located symbol from another file is not visible");

    free_environment(env);
}

/* Symbols registered without a file -- builtins, and anything added before a
 * file is in scope -- must stay reachable everywhere, or the change would
 * break every builtin lookup. */
static void test_fileless_symbols_stay_visible(void) {
    Environment *env = create_environment();

    env_set_current_file(env, NULL);
    Value unset = create_void();
    env_define_var(env, "builtin_thing", TYPE_INT, false, unset);

    env_set_current_file(env, "anywhere.nano");
    Symbol *found = env_get_var_visible_at(env, "builtin_thing", 100, 1);
    CHECK(found != NULL, "a symbol with no file is visible from any file");
    CHECK(found && found->type == TYPE_INT, "and keeps its type");

    free_environment(env);
}

/* Redefining a name must not inherit metadata from a same-named symbol in a
 * different file. That path is how a struct-typed parameter picked up the
 * wrong struct name, which is the field-access half of the same bug. */
static void test_redefinition_does_not_inherit_across_files(void) {
    Environment *env = create_environment();

    env_set_current_file(env, "a.nano");
    Value unset = create_void();
    env_define_var(env, "p", TYPE_STRUCT, false, unset);
    Symbol *first = env_get_var(env, "p");
    CHECK(first != NULL, "the first definition exists");
    if (first) first->struct_type_name = strdup("TypeFromA");

    env_set_current_file(env, "b.nano");
    env_define_var(env, "p", TYPE_STRUCT, false, unset);
    Symbol *second = env_get_var(env, "p");
    CHECK(second != NULL, "the second definition exists");
    CHECK(second && second != first, "it is a separate symbol, not the first updated");
    CHECK(second && second->struct_type_name == NULL,
          "it does not inherit the struct type recorded in another file");

    free_environment(env);
}

/* Two files may each have a symbol named the same thing without either
 * seeing the other -- which is the property that matters once modules start
 * competing for one namespace. */
static void test_same_name_in_many_files(void) {
    Environment *env = create_environment();
    define_at(env, "one.nano", "shared", TYPE_INT, 10);
    define_at(env, "two.nano", "shared", TYPE_FLOAT, 10);
    define_at(env, "three.nano", "shared", TYPE_BOOL, 10);

    const char *files[] = { "one.nano", "two.nano", "three.nano" };
    Type expected[] = { TYPE_INT, TYPE_FLOAT, TYPE_BOOL };
    for (int i = 0; i < 3; i++) {
        env_set_current_file(env, files[i]);
        Symbol *s = env_get_var_visible_at(env, "shared", 20, 1);
        CHECK(s && s->type == expected[i],
              "each file resolves `shared` to its own definition");
    }

    free_environment(env);
}

static char **one_export(const char *name) {
    char **names = malloc(sizeof(*names));
    names[0] = strdup(name);
    return names;
}

static void test_import_alias_owners(void) {
    Environment *env = create_environment();
    Function left = {0}, right = {0};
    left.name = right.name = "answer";
    left.module_name = "LeftValue";
    right.module_name = "RightValue";
    env_define_function(env, left);
    env_define_function(env, right);
    StructDef record = {0};
    EnumDef enumeration = {0};
    UnionDef choice = {0};
    record.name = strdup("Record");
    enumeration.name = strdup("Enumeration");
    choice.name = strdup("Choice");
    record.module_name = enumeration.module_name = choice.module_name = "LeftValue";
    env_define_struct(env, record);
    env_define_enum(env, enumeration);
    env_define_union(env, choice);

    env->current_module = "LeftWrapper";
    env_register_namespace(env, "lib", "LeftValue", one_export("answer"), 1,
                           one_export("Record"), 1, one_export("Enumeration"), 1,
                           one_export("Choice"), 1);
    env->current_module = "RightWrapper";
    env_register_namespace(env, "lib", "RightValue", one_export("answer"), 1,
                           NULL, 0, NULL, 0, NULL, 0);
    CHECK(env->namespace_count == 2, "I retain the same alias in distinct importers");
    Function *found = env_get_function(env, "lib.answer");
    CHECK(found && strcmp(found->module_name, "RightValue") == 0,
          "I resolve the right importer's alias");
    CHECK(env_get_struct(env, "lib.Record") == NULL, "I do not borrow another importer's struct alias");
    CHECK(env_get_enum(env, "lib.Enumeration") == NULL, "I do not borrow another importer's enum alias");
    CHECK(env_get_union(env, "lib.Choice") == NULL, "I do not borrow another importer's union alias");
    env->current_module = "LeftWrapper";
    found = env_get_function(env, "lib.answer");
    CHECK(found && strcmp(found->module_name, "LeftValue") == 0,
          "I resolve the left importer's alias");
    CHECK(env_get_struct(env, "lib.Record") != NULL, "I resolve an owned struct alias");
    CHECK(env_get_enum(env, "lib.Enumeration") != NULL, "I resolve an owned enum alias");
    CHECK(env_get_union(env, "lib.Choice") != NULL, "I resolve an owned union alias");
    env->current_module = NULL;
    CHECK(env_get_function(env, "lib.answer") == NULL,
          "I do not expose a dependency's alias at the root");
    env_register_namespace(env, "lib", "LeftValue", one_export("answer"), 1,
                           NULL, 0, NULL, 0, NULL, 0);
    CHECK(env_get_function(env, "lib.answer") != NULL, "I resolve a root-owned alias");
    env->current_module = "Unrelated";
    CHECK(env_get_function(env, "lib.answer") == NULL,
          "I do not inherit a root alias into an unrelated module");
    env->current_module = NULL;
    free_environment(env);
}

static void test_import_owner_restoration(void) {
    const char *sources[] = {
        "module Declared\n",
        "module Declared\nmodule \"/__nano_missing_owner_test__/absent.nano\" as lib\n"
    };
    for (int i = 0; i < 2; i++) {
        Environment *env = create_environment();
        char *owner = "Caller";
        env->current_module = owner;
        int count = 0;
        Token *tokens = tokenize(sources[i], &count);
        ASTNode *program = parse_program(tokens, count);
        CHECK(program != NULL, "I parse the import-context fixture");
        bool ok = process_imports(program, env, NULL, "owner.nano");
        CHECK(ok == (i == 0), "I distinguish successful and failed import processing");
        CHECK(env->current_module == owner, "I restore the caller after either outcome");
        env->current_module = NULL;
        free_environment(env);
        free_ast(program);
        free_tokens(tokens, count);
    }
}

static void test_retained_block_bounds(void) {
    Environment *env = create_environment();
    define_at(env, "scope.nano", "value", TYPE_FLOAT, 1);
    Symbol *inner = define_at(env, "scope.nano", "value", TYPE_STRING, 2);
    inner->scope_end_line = 3;
    inner->scope_end_column = 20;
    Symbol *found = env_get_var_visible_at(env, "value", 3, 19);
    CHECK(found && found->type == TYPE_STRING, "I see the inner binding before its closing brace");
    found = env_get_var_visible_at(env, "value", 3, 20);
    CHECK(found && found->type == TYPE_FLOAT, "I restore the outer binding at the scope boundary");
    found = env_get_var_visible_at(env, "value", 4, 1);
    CHECK(found && found->type == TYPE_FLOAT, "I restore the outer binding after the block");
    CHECK(env->symbol_count == 2, "I retain both symbols for later emission");
    free_environment(env);
}

static void test_union_owns_string_payload(void) {
    gc_init();
    Value local = create_string("retained text");
    char *names[] = {"value"};
    Value result = create_union("Result", 0, "Ok", names, &local, 1);
    Value alias = result;
    CHECK(result.as.union_val->field_values[0].as.string_val != local.as.string_val,
          "I copy a union string independently of its constructing local");
    gc_release(local.as.string_val);
    CHECK(strcmp(alias.as.union_val->field_values[0].as.string_val, "retained text") == 0,
          "I retain a returned union string after its local is released");
    CHECK(alias.as.union_val == result.as.union_val,
          "A Value copy preserves the existing shared union identity");
    gc_release(result.as.union_val->field_values[0].as.string_val);
    free(result.as.union_val->field_names[0]);
    free(result.as.union_val->field_names);
    free(result.as.union_val->field_values);
    free(result.as.union_val->union_name);
    free(result.as.union_val->variant_name);
    free(result.as.union_val);
    gc_shutdown();
}

static void test_borrowed_record_identity(void) {
    Environment *owner_env = create_environment();
    char *names[] = {"fd"};
    Value fields[] = {create_int(7)};
    StructValue initial = {.struct_name = "Handle", .field_names = names,
                           .field_values = fields, .field_count = 1};
    Value value = {.type = VAL_STRUCT, .as.struct_val = &initial};
    env_define_var(owner_env, "owner", TYPE_STRUCT, true, value);
    Value owner = env_get_var(owner_env, "owner")->value;
    Environment *borrow_env = create_environment();
    env_define_var(borrow_env, "first", TYPE_BORROW_SHARED, false, owner);
    env_define_var(borrow_env, "second", TYPE_BORROW_SHARED, false, owner);
    CHECK(env_get_var(borrow_env, "first")->value.as.struct_val == owner.as.struct_val,
          "I retain the owner's record identity for a shared parameter");
    CHECK(env_get_var(borrow_env, "second")->value.as.struct_val == owner.as.struct_val,
          "I retain the same identity for repeated shared aliases");
    free_environment(borrow_env);
    CHECK(owner.as.struct_val->field_values[0].as.int_val == 7,
          "I leave the owner alive after borrowed bindings are destroyed");
    free_environment(owner_env);
}

int main(void) {
    printf("\n[env_scoping] symbol visibility is confined to one file...\n\n");
    test_lookup_ignores_other_files();
    test_other_file_is_not_visible();
    test_fileless_symbols_stay_visible();
    test_redefinition_does_not_inherit_across_files();
    test_same_name_in_many_files();
    test_import_alias_owners();
    test_import_owner_restoration();
    test_retained_block_bounds();
    test_union_owns_string_payload();
    test_borrowed_record_identity();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
