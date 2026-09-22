/* I inspect actual checked declaration rows; I do not invoke a foreign ABI. */
#include "nanolang.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int g_argc;
char **g_argv;
static int label_call, label_failure;
static char *checked_label_copy(const char *text) {
    if (label_failure && ++label_call == label_failure) return NULL;
    return strdup(text);
}
#define strdup checked_label_copy
#include "../src/typechecker.c"
#undef strdup

typedef struct { Token *tokens; int count; ASTNode *program; } Parsed;
static Parsed parsed(const char *source) {
    Parsed p = {0}; p.tokens = tokenize(source, &p.count); assert(p.tokens);
    p.program = parse_program(p.tokens, p.count); assert(p.program);
    return p;
}
static ASTNode *declaration(Parsed *p) {
    assert(p->program->as.program.count == 1);
    ASTNode *node = p->program->as.program.items[0];
    assert(node->type == AST_FUNCTION && node->as.function.is_extern);
    return node;
}
static void dispose(Parsed p) { free_ast(p.program); free_tokens(p.tokens, p.count); }
static void check_order(bool root_first) {
    Parsed p = parsed("extern fn owner_probe(value: int) -> int");
    Parsed root = parsed("extern fn owner_probe(value: int) -> int fn main() -> int { return 0 } shadow main { assert true }");
    Environment *env = create_environment(); assert(env);
    const char *owners[2] = {root_first ? NULL : "foreign", root_first ? "foreign" : NULL};
    for (int i = 0; i < 2; ++i) {
        env->current_module = (char *)owners[i];
        assert(owners[i] ? type_check_module(p.program, env) : type_check(root.program, env));
        Function *function = env_get_function(env, "owner_probe"); assert(function);
        assert((!owners[i] && !function->module_name) ||
               (owners[i] && function->module_name && !strcmp(owners[i], function->module_name)));
        assert(!function->is_pub && function->is_extern);
        assert(extern_declaration_state(env, declaration(&p)) == 1);
    }
    int count = env->function_count;
    assert(type_check_module(p.program, env)); assert(env->function_count == count);
    for (int i = 0; i < 2; ++i) {
        env->current_module = (char *)owners[i];
        Function *function = env_get_function(env, "owner_probe"); assert(function);
        assert((!owners[i] && !function->module_name) ||
               (owners[i] && function->module_name && !strcmp(owners[i], function->module_name)));
    }
    env->current_module = NULL; free_environment(env); dispose(p); dispose(root);
}
static void check_signature(const char *left, const char *right, bool expected) {
    Parsed a = parsed(left), b = parsed(right);
    Environment *env = create_environment(); assert(env);
    Function av = extern_declaration_view(env, declaration(&a));
    Function bv = extern_declaration_view(env, declaration(&b));
    assert(functions_match(env, &av, &bv) == expected);
    assert(functions_match(env, &bv, &av) == expected);
    free_environment(env); dispose(a); dispose(b);
}
static void check_publication(void) {
    Parsed p = parsed("extern fn allocation_probe(value: array<string>) -> array<string>");
    Environment *env = create_environment(); assert(env); env->current_module = "foreign";
    int before = env->function_count;
    for (int fail = 1; fail <= 2; ++fail) {
        label_call = 0; label_failure = fail;
        assert(!register_owned_extern_declaration(env, declaration(&p)));
        label_failure = 0;
        assert(env->function_count == before);
        assert(!env_get_function(env, "allocation_probe"));
    }
    assert(register_owned_extern_declaration(env, declaration(&p)));
    assert(env->function_count == before + 1);
    assert(extern_declaration_state(env, declaration(&p)) == 1);
    declaration(&p)->as.function.is_pub = true;
    assert(extern_declaration_state(env, declaration(&p)) == -1);
    assert(env->function_count == before + 1);
    env->current_module = NULL; free_environment(env); dispose(p);
}
static void check_alias_conflict(void) {
    Parsed p = parsed("extern fn owner_probe(value: int) -> int");
    Environment *env = create_environment(); assert(env); env->current_module = "foreign";
    assert(register_owned_extern_declaration(env, declaration(&p)));
    Function *alias = &env->functions[env->function_count - 1];
    alias->alias_of = alias->name;
    alias->name = env_own_checker_allocation(env, strdup("renamed_probe"));
    assert(alias->name); env_function_index_invalidate(env);
    env->current_module = NULL;
    assert(extern_declaration_state(env, declaration(&p)) == 0);
    alias->return_type = TYPE_BOOL;
    assert(extern_declaration_state(env, declaration(&p)) == -1);
    assert(env->function_count == 1);
    free_environment(env); dispose(p);
}
static void check_nominal_owners(void) {
    Environment *env = create_environment(); assert(env);
    StructDef left = {.name = strdup("LeftKey"), .original_name = strdup("Same"),
        .module_name = env_own_checker_allocation(env, strdup("left"))};
    StructDef right = {.name = strdup("RightKey"), .original_name = strdup("Same"),
        .module_name = env_own_checker_allocation(env, strdup("right"))};
    assert(left.name && left.original_name && left.module_name);
    assert(right.name && right.original_name && right.module_name);
    env_define_struct(env, left); env_define_struct(env, right);
    Parameter parameter = {.type = TYPE_STRUCT, .struct_type_name = "Same"};
    Function a = {.params = &parameter, .param_count = 1, .return_type = TYPE_INT, .module_name = "left"};
    Function b = a; b.module_name = "right";
    assert(functions_match(env, &a, &a));
    assert(!functions_match(env, &a, &b));
    TypeInfo member = {.base_type = TYPE_STRUCT, .generic_name = "Same"};
    TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = &member};
    parameter.type = TYPE_ARRAY; parameter.struct_type_name = NULL;
    parameter.element_type = TYPE_STRUCT; parameter.type_info = &array;
    assert(functions_match(env, &a, &a)); assert(!functions_match(env, &a, &b));
    env->current_module = NULL; free_environment(env);
}
int main(int argc, char **argv) {
    g_argc = argc; g_argv = argv;
    check_order(false); check_order(true);
    check_signature("extern fn p(a: array<int>) -> int", "extern fn p(b: array<int>) -> int", true);
    check_signature("extern fn p(a: array<int>) -> int", "extern fn p(a: array<string>) -> int", false);
    check_signature("extern fn p() -> array<int>", "extern fn p() -> array<string>", false);
    check_signature("extern fn p(a: fn(int) -> int) -> int", "extern fn p(a: fn(string) -> int) -> int", false);
    check_signature("extern fn p(a: (int, string)) -> int", "extern fn p(a: (int, int)) -> int", false);
    check_publication(); check_alias_conflict(); check_nominal_owners();
    puts("I retained exact extern declaration owners, complete signatures and unpublished failed labels.");
    return 0;
}
