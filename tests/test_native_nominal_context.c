/* I check emitter context restoration across successive and enclosing calls. */
#include "../src/transpiler.c"
#include <assert.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static void emit_checked(const char *source, const char *expected_declaration) {
    int count = 0;
    Token *tokens = tokenize(source, &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    assert(program);
    Environment *env = create_environment();
    assert(env);
    typecheck_set_current_file("<native-nominal-context>");
    assert(type_check(program, env));
    char *text = transpile_to_c(program, env, "<native-nominal-context>");
    assert(text);
    if (expected_declaration) {
        assert(strstr(text, expected_declaration));
        assert(!strstr(text, "typedef struct void*"));
    } else {
        assert(!strstr(text, "nl_T"));
    }
    free(text);
    free_environment(env);
    free_ast(program);
    free_tokens(tokens, count);
}

static void fresh_binding_metadata(void) {
    Environment *env = create_environment();
    assert(env);
    env_define_var(env, "value", TYPE_STRUCT, false, create_void());
    Symbol *old = &env->symbols[0];
    old->struct_type_name = strdup("EarlierRecord");
    assert(old->struct_type_name);
    TypeInfo integer = {.base_type = TYPE_INT};
    TypeInfo *children[] = {&integer, &integer};
    TypeInfo tuple = {.base_type = TYPE_TUPLE, .tuple_element_count = 2,
                      .type_params = children, .type_param_count = 2};
    env_define_var_with_type_info(env, "value", TYPE_TUPLE, TYPE_UNKNOWN,
                                 &tuple, false, create_void());
    assert(env->symbol_count == 2);
    assert(!env->symbols[1].struct_type_name);
    assert(env->symbols[1].type_info == &tuple);
    assert(!strcmp(env->symbols[0].struct_type_name, "EarlierRecord"));
    env_define_var(env, "value", TYPE_INT, true, create_int(7));
    assert(!env->symbols[2].struct_type_name);
    env_set_var(env, "value", create_int(9));
    assert(env->symbols[2].value.as.int_val == 9);

    char *vars[] = {"T"};
    Type types[] = {TYPE_STRUCT};
    char *names[] = {"CurrentRecord"};
    GenericFuncInstance inst = {.var_names = vars, .bound_types = types,
        .bound_type_names = names, .binding_count = 1};
    TypeInfo generic = {.base_type = TYPE_STRUCT, .generic_name = "T"};
    Parameter param = {.name = "value", .type = TYPE_STRUCT,
        .struct_type_name = "T", .type_info = &generic};
    bind_native_generic_parameter(env, &param, &inst);
    Symbol *bound = &env->symbols[3];
    assert(!strcmp(bound->struct_type_name, "CurrentRecord"));
    assert(bound->type_info != &generic && bound->type_info->base_type == TYPE_STRUCT);
    assert(!strcmp(bound->type_info->generic_name, "CurrentRecord"));
    assert(!strcmp(generic.generic_name, "T"));
    types[0] = TYPE_INT; names[0] = NULL;
    bind_native_generic_parameter(env, &param, &inst);
    assert(env->symbols[4].type == TYPE_INT && !env->symbols[4].struct_type_name);
    assert(env->symbols[4].type_info->base_type == TYPE_INT);
    assert(!env->symbols[4].type_info->generic_name);

    TypeInfo string = {.base_type = TYPE_STRING};
    TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = &string};
    param = (Parameter){.name = "value", .type = TYPE_ARRAY,
        .element_type = TYPE_STRING, .type_info = &array};
    bind_native_generic_parameter(env, &param, &inst);
    assert(env->symbols[5].type_info == &array);
    assert(env->symbols[5].element_type == TYPE_STRING);
    assert(!env->symbols[5].struct_type_name);
    param = (Parameter){.name = "value", .type = TYPE_TUPLE, .type_info = &tuple};
    bind_native_generic_parameter(env, &param, &inst);
    assert(env->symbols[6].type_info == &tuple && !env->symbols[6].struct_type_name);
    Type arguments[] = {TYPE_INT};
    FunctionSignature signature = {.param_types = arguments, .param_count = 1,
                                   .return_type = TYPE_INT};
    param = (Parameter){.name = "value", .type = TYPE_FUNCTION, .fn_sig = &signature};
    bind_native_generic_parameter(env, &param, &inst);
    assert(env->symbols[7].type_info->fn_sig != &signature);
    assert(env->symbols[7].type_info->fn_sig->param_types[0] == TYPE_INT);
    assert(!env->symbols[7].struct_type_name);
    free_environment(env);
}

int main(void) {
    fresh_binding_metadata();
    const char *declared = "union T { Some { value: int } } fn main() -> int { return 0 } shadow main { assert (== (main) 0) }";
    const char *generic = "fn identity(value: T) -> T { return value } shadow identity { assert (== (identity 7) 7) } fn main() -> int { return (- (identity 7) 7) } shadow main { assert (== (main) 0) }";
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    emit_checked(declared, "typedef struct nl_T");
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    emit_checked(generic, NULL);
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));

    emit_checked("enum T { First, Second } fn main() -> int { return 0 } shadow main { assert (== (main) 0) }", "} nl_T;");
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    emit_checked(generic, NULL);

    /* An enclosing emission's snapshot survives an inner success or refusal. */
    native_declared_letters = UINT32_C(1) << ('U' - 'A');
    emit_checked(declared, "typedef struct nl_T");
    assert(!strcmp(get_prefixed_type_name("U"), "nl_U"));
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    assert(!transpile_to_c(NULL, NULL, "<refused>"));
    assert(!strcmp(get_prefixed_type_name("U"), "nl_U"));
    native_declared_letters = 0;
    puts("I passed native nominal context restoration checks.");
    return 0;
}
