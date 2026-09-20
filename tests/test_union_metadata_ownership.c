/* I retain independent union metadata after either original owner is gone. */
#define _POSIX_C_SOURCE 200809L
#include "../src/nanolang.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;
char **g_argv;

static void check_union(const UnionDef *u) {
    assert(!strcmp(u->name, "Box"));
    assert(u->generic_param_count == 1 && !strcmp(u->generic_params[0], "T"));
    assert(!strcmp(u->module_name, "Owner"));
    assert(u->variant_count == 2 && u->variant_field_counts[0] == 2);
    assert(!strcmp(u->variant_field_type_names[0][0], "T"));
    assert(u->variant_field_type_info[0][1]->base_type == TYPE_ARRAY);
    assert(u->variant_field_type_info[0][1]->element_type->base_type == TYPE_ARRAY);
    assert(!strcmp(u->variant_field_type_info[0][1]->element_type->element_type->generic_name, "T"));
}

static void lifetime(int metadata_first) {
    int count = 0;
    Token *tokens = tokenize("union Box<T> { Some { value: T, nested: array<array<T>> }, None {} }", &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    assert(program);
    Environment *env = create_environment();
    env->current_module = "Owner";
    typecheck_set_current_file("<union-ownership>");
    assert(type_check_module(program, env));
    UnionDef *original = env_get_union(env, "Box");
    assert(original);
    ModuleMetadata *first = extract_module_metadata(env, "first");
    ModuleMetadata *second = extract_module_metadata(env, "second");
    assert(first && second && first->union_count == 1 && second->union_count == 1);
    UnionDef *a = &first->unions[0], *b = &second->unions[0];
    assert(a->generic_params != original->generic_params && a->generic_params != b->generic_params);
    assert(a->generic_params[0] != original->generic_params[0] && a->generic_params[0] != b->generic_params[0]);
    assert(a->module_name != original->module_name && a->module_name != b->module_name);
    assert(a->variant_field_type_names != original->variant_field_type_names);
    assert(a->variant_field_type_names[0] != original->variant_field_type_names[0]);
    assert(a->variant_field_type_names[0][0] != original->variant_field_type_names[0][0]);
    assert(a->variant_field_type_names[0][0] != b->variant_field_type_names[0][0]);
    free_ast(program);
    free_tokens(tokens, count);
    check_union(original); check_union(a); check_union(b);
    if (metadata_first) {
        free_module_metadata(first); check_union(original); check_union(b);
        free_environment(env); check_union(b);
    } else {
        free_environment(env); check_union(a); check_union(b);
        free_module_metadata(first); check_union(b);
    }
    free_module_metadata(second);
}

static void empty_declaration(void) {
    Environment *env = create_environment();
    UnionDef empty = {0};
    empty.name = strdup("Empty");
    /* I also cover allocated zero-length arrays, not only null pointers. */
    empty.variant_names = calloc(0, sizeof(char *));
    empty.variant_field_counts = calloc(0, sizeof(int));
    empty.variant_field_names = calloc(0, sizeof(char **));
    empty.variant_field_types = calloc(0, sizeof(Type *));
    empty.variant_field_type_names = calloc(0, sizeof(char **));
    env_define_union(env, empty);
    ModuleMetadata *meta = extract_module_metadata(env, "empty");
    assert(meta && meta->union_count == 1);
    free_environment(env);
    assert(!strcmp(meta->unions[0].name, "Empty"));
    assert(!meta->unions[0].variant_count && !meta->unions[0].generic_param_count);
    assert(!meta->unions[0].module_name && !meta->unions[0].generic_params);
    free_module_metadata(meta);
}

int main(void) {
    lifetime(0); lifetime(1); empty_declaration();
    Environment *empty = create_environment();
    ModuleMetadata *meta = extract_module_metadata(empty, "none");
    assert(meta && !meta->union_count);
    free_environment(empty); free_module_metadata(meta);
    puts("I passed independent union metadata lifetime checks.");
    return 0;
}
