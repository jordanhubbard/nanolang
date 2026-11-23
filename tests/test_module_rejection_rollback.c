/* I withdraw rejected module publications and permit a real corrected retry. */
#include "../src/nanolang.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int g_argc;
char **g_argv;

static void put(const char *path, const char *source) {
    FILE *file = fopen(path, "w"); assert(file);
    assert(fputs(source, file) >= 0); assert(fclose(file) == 0);
}
static ASTNode *parse(const char *source, bool repl) {
    int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
    ASTNode *program = repl ? parse_repl_input(tokens, count) : parse_program(tokens, count);
    free_tokens(tokens, count); assert(program); return program;
}
static int64_t call(Environment *env, const char *owner, const char *source) {
    char *saved = env->current_module; env->current_module = (char *)owner;
    ASTNode *program = parse(source, true);
    assert(program->as.program.count == 1);
    Value value = repl_eval_node(program->as.program.items[0], env);
    assert(value.type == VAL_INT); int64_t result = value.as.int_val;
    free_ast(program); env->current_module = saved; return result;
}
static bool namespace_present(Environment *env, const char *owner, const char *alias) {
    for (int i = 0; i < env->namespace_count; ++i) {
        ModuleNamespace *space = &env->namespaces[i];
        if (space->owner_module && !strcmp(space->owner_module, owner) && !strcmp(space->alias, alias)) return true;
    }
    return false;
}
static char *path(const char *directory, const char *name) {
    char *result = NULL; assert(asprintf(&result, "%s/%s", directory, name) >= 0); return result;
}

int main(void) {
    Environment *shape_env = create_environment(); assert(shape_env);
    ASTNode *shape = parse("struct RejectedRecord { value: int } "
                           "struct RejectedHolder { values: List<RejectedRecord> }", false);
    assert(type_check_module(shape, shape_env)); assert(shape_env->generic_instance_count > 0);
    GenericInstantiation *instance = &shape_env->generic_instances[shape_env->generic_instance_count - 1];
    char *helper = NULL; assert(asprintf(&helper, "%s_new", instance->concrete_name) >= 0);
    assert(env_get_function(shape_env, helper)); free(helper);
    free_environment(shape_env); free_ast(shape);
    puts("I exercise actual record-list generic/helper registration before checking failed-module withdrawal.");
    char directory[] = "/tmp/nano-module-rollback-XXXXXX"; assert(mkdtemp(directory));
    char *bad = path(directory, "bad.nano"), *child_a = path(directory, "child_a.nano");
    char *child_b = path(directory, "child_b.nano"), *parent = path(directory, "parent.nano");
    char *cycle_a = path(directory, "cycle_a.nano"), *cycle_b = path(directory, "cycle_b.nano");
    char *absent = path(directory, "absent.nano");
    clear_module_cache(); Environment *env = create_environment(); assert(env);
    env_set_current_file(env, "retained-root.nano"); typecheck_set_current_file("retained-root.nano");
    ASTNode *root = parse("fn anchor() -> int { return 7 } shadow anchor { assert (== (anchor) 7) }", false);
    assert(type_check_module(root, env));
    env_define_var(env, "live_text", TYPE_STRING, false, create_string("retained"));
    Value array = create_array(VAL_INT, 1, 1); ((int64_t *)array.as.array_val->data)[0] = 42;
    env_define_var_with_element_type(env, "live_array", TYPE_ARRAY, TYPE_INT, false, array);
    ASTNode *shadow = env_get_function_owned(env, "anchor", NULL)->shadow_test;
    int functions = env->function_count, symbols = env->symbol_count;
    int records = env->struct_count, enums = env->enum_count, unions = env->union_count;
    int opaque = env->opaque_type_count, effects = env->effect_count;
    int generic = env->generic_instance_count, generic_functions = env->generic_func_instance_count;
    int modules = env->module_count;
    const char *failed =
        "module BrokenOwner\nstruct RejectedRecord { value: int }\n"
        "struct RejectedHolder { values: List<RejectedRecord> }\n"
        "enum RejectedEnum { One }\nunion RejectedUnion<T> { Item { value: T } }\n"
        "opaque type RejectedOpaque\neffect RejectedEffect { now : void -> void }\n"
        "let rejected_global: int = 5\n"
        "pub fn broken() -> int { return missing_value }\n"
        "shadow broken { assert (== (broken) 9) }\n"
        "shadow anchor { assert false }\n";
    put(bad, failed);
    for (int iteration = 0; iteration < 32; ++iteration) {
        assert(!load_module(bad, env)); assert(!get_cached_module_ast(bad));
        assert(!env_get_function_owned(env, "broken", "BrokenOwner"));
        assert(!env_get_var(env, "rejected_global"));
        assert(!env_get_struct(env, "RejectedRecord")); assert(!env_get_struct(env, "RejectedHolder"));
        assert(!env_get_enum(env, "RejectedEnum"));
        assert(!env_get_union(env, "RejectedUnion")); assert(!env_get_opaque_type(env, "RejectedOpaque"));
        assert(!env_get_effect_owned(env, "RejectedEffect", "BrokenOwner"));
        assert(env->function_count == functions && env->symbol_count == symbols);
        assert(env->struct_count == records && env->enum_count == enums && env->union_count == unions);
        assert(env->opaque_type_count == opaque && env->effect_count == effects);
        assert(env->generic_instance_count == generic && env->generic_func_instance_count == generic_functions);
        assert(env->module_count == modules && !env->current_module);
        assert(!strcmp(env_current_file(env), "retained-root.nano"));
        assert(env_get_function_owned(env, "anchor", NULL)->shadow_test == shadow);
        assert(!strcmp(env_get_var(env, "live_text")->value.as.string_val, "retained"));
        assert(env_get_var(env, "live_array")->value.as.array_val == array.as.array_val);
        assert(((int64_t *)array.as.array_val->data)[0] == 42 && call(env, NULL, "(anchor)") == 7);
    }
    put(bad, "module BrokenOwner\npub fn broken() -> int { return 9 }\nshadow broken { assert (== (broken) 9) }\n");
    ASTNode *corrected = load_module(bad, env); assert(corrected);
    assert(get_cached_module_ast(bad) == corrected && load_module(bad, env) == corrected);
    assert(call(env, "BrokenOwner", "(broken)") == 9);
    puts("I withdraw all failed declaration kinds across 32 attempts, restore prior shadows/live owners and admit a corrected same-path load.");

    put(child_a, "module ChildA\npub fn helper() -> int { return 11 }\nshadow helper { assert (== (helper) 11) }\n"
                 "pub enum FlagA { One = -3, Next, High = 17 }\npub effect SignalA { now : void -> void }\npub let number_a: int = 11\n");
    put(child_b, "module ChildB\npub fn helper() -> int { return 22 }\nshadow helper { assert (== (helper) 22) }\n"
                 "pub enum FlagB { One, Two }\npub effect SignalB { now : void -> void }\npub let number_b: int = 22\n");
    assert(load_module(child_a, env));
    EnumDef *flag = env_get_enum(env, "FlagA"); assert(flag && flag->variant_count == 3);
    assert(flag->variant_values[0] == -3 && flag->variant_values[1] == -2 && flag->variant_values[2] == 17);
    assert(!strcmp(flag->module_name, "ChildA"));
    assert(env_import_function(env, "prior_helper", env_get_function_owned(env, "helper", "ChildA")));
    put(parent, "module ParentOwner\nmodule \"child_a.nano\" as dep\n"
                "from \"child_a.nano\" import helper as chosen, FlagA as SelectedFlag, SignalA as SelectedSignal, number_a as selected_number\n"
                "pub fn broken_parent() -> int { return missing_value }\n"
                "shadow broken_parent { assert (== (broken_parent) 11) }\n");
    assert(!load_module(parent, env)); assert(get_cached_module_ast(child_a));
    assert(!namespace_present(env, "ParentOwner", "dep"));
    assert(!env_get_function_owned(env, "chosen", "ParentOwner"));
    assert(!env_lookup_global_import_at(env, parent, "selected_number"));
    env->current_module = "ParentOwner";
    assert(!env_get_enum(env, "SelectedFlag")); assert(!env_get_effect(env, "SelectedSignal"));
    env->current_module = NULL;
    assert(call(env, NULL, "(prior_helper)") == 11);
    put(parent, "module ParentOwner\nmodule \"child_b.nano\" as dep\n"
                "from \"child_b.nano\" import helper as chosen, FlagB as SelectedFlag, SignalB as SelectedSignal, number_b as selected_number\n"
                "pub fn parent_value() -> int { return (+ (dep.helper) (chosen)) }\n"
                "shadow parent_value { assert (== (parent_value) 44) }\n");
    assert(load_module(parent, env)); assert(call(env, "ParentOwner", "(parent_value)") == 44);
    flag = env_get_enum(env, "FlagB"); assert(flag && flag->variant_count == 2);
    assert(flag->variant_values[0] == 0 && flag->variant_values[1] == 1 && !strcmp(flag->module_name, "ChildB"));
    assert(call(env, NULL, "(prior_helper)") == 11);
    puts("I withdraw failed namespace/function/enum/effect/global aliases, preserve dependencies and prior slot aliases, and bind changed imports on retry.");

    put(cycle_a, "module CycleA\nmodule \"cycle_b.nano\" as nested\n");
    put(cycle_b, "module CycleB\nmodule \"cycle_a.nano\" as nested\n");
    assert(!load_module(cycle_a, env)); assert(!get_cached_module_ast(cycle_a) && !get_cached_module_ast(cycle_b));
    put(cycle_b, "module CycleB\npub fn settled() -> int { return 33 }\nshadow settled { assert (== (settled) 33) }\n");
    assert(load_module(cycle_a, env)); assert(call(env, "CycleB", "(settled)") == 33);
    puts("I preserve circular-import refusal and remove only markers owned by terminal failed loads before corrected recovery.");

    assert(!load_module(absent, env));
    put(absent, "fn malformed(\n"); assert(!load_module(absent, env));
    put(absent, "\"unfinished\\"); assert(!load_module(absent, env));
    put(absent, "pub fn recovered() -> int { return 55 }\nshadow recovered { assert (== (recovered) 55) }\n");
    assert(load_module(absent, env)); assert(call(env, "absent", "(recovered)") == 55);
    puts("I recover the same source path after missing-file, parser and lexer failures without a stale circular-import marker.");

    free_environment(env); clear_module_cache(); free_ast(root);
    char *paths[] = {bad, child_a, child_b, parent, cycle_a, cycle_b, absent};
    for (size_t i = 0; i < sizeof paths / sizeof *paths; ++i) { assert(unlink(paths[i]) == 0); free(paths[i]); }
    assert(rmdir(directory) == 0);
    puts("I retire rejected module metadata and all retained successful owners at explicit teardown.");
    return 0;
}
