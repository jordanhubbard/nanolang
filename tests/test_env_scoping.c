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

int main(void) {
    printf("\n[env_scoping] symbol visibility is confined to one file...\n\n");
    test_lookup_ignores_other_files();
    test_other_file_is_not_visible();
    test_fileless_symbols_stay_visible();
    test_redefinition_does_not_inherit_across_files();
    test_same_name_in_many_files();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
