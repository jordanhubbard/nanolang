/* I exercise the actual checker predicates, without foreign calls or evaluation. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/typechecker.c"
#include <assert.h>
#ifdef NDEBUG
#error I require active fixture assertions.
#endif

int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static void identity_record(Environment *env, const char *name, const char *owner) {
    StructDef record = {0};
    record.name = strdup(name); record.module_name = (char *)owner;
    assert(record.name && record.module_name);
    env_define_struct(env, record);
}
static void intrinsic_identity(void) {
    Environment *env = create_environment(); assert(env);
    const char *names[] = {"at", "array_get", "array_push", "array_slice", "array_new", "map", "filter"};
    for (size_t i = 0; i < sizeof names / sizeof *names; ++i) {
        Function *builtin = env_get_function(env, names[i]);
        assert(builtin && env_function_is_builtin(builtin));
        Function copy = *builtin;
        assert(!env_function_is_builtin(&copy));
        Function foreign = {0};
        foreign.name = (char *)names[i]; foreign.is_extern = true; foreign.module_name = "Foreign"; foreign.is_pub = true;
        foreign.param_count = builtin->param_count; foreign.return_type = TYPE_INT;
        env_define_function(env, foreign);
        assert(!env_function_is_builtin(&env->functions[env->function_count - 1]));
        /* Reserved unqualified lookup still selects the real builtin cache. */
        assert(env_get_function(env, names[i]) == builtin);
        ASTNode call = {0}; call.type = AST_CALL; call.as.call.name = (char *)names[i];
        call.as.call.arg_count = builtin->param_count;
        assert(nominal_array_builtin(&call, env, names[i], builtin->param_count));
        env_define_var(env, names[i], TYPE_INT, false, create_int(1));
        assert(!nominal_array_builtin(&call, env, names[i], builtin->param_count));
    }
    assert(!env_array_push_is_builtin(env, 0, 0));
    char **exports = calloc(1, sizeof *exports); assert(exports);
    exports[0] = strdup("array_push"); assert(exports[0]);
    env_register_namespace(env, "foreign", "Foreign", exports, 1, NULL, 0, NULL, 0, NULL, 0);
    Function *selected = env_get_function(env, "foreign.array_push");
    assert(selected && selected->is_extern && !env_function_is_builtin(selected));
    ASTNode qualified = {0}; qualified.type = AST_MODULE_QUALIFIED_CALL;
    qualified.as.module_qualified_call.module_alias = "foreign";
    qualified.as.module_qualified_call.function_name = "array_push";
    qualified.as.module_qualified_call.arg_count = 2;
    assert(!nominal_array_builtin(&qualified, env, "array_push", 2));
    NominalView view = {0};
    assert(nominal_value_view(&qualified, env, 0, &view));
    assert(view.info->base_type == TYPE_INT && !strcmp(view.owner, "Foreign"));
    nominal_view_discard(&view);
    assert(!env_function_is_builtin(NULL));
    free_environment(env);

    env = create_environment(); assert(env);
    ASTNode body = {0}; body.type = AST_BLOCK;
    Function local = {0}; local.name = "array_push"; local.body = &body;
    env_define_function(env, local);
    assert(env_get_function(env, "array_push") == &env->functions[0]);
    assert(!env_array_push_is_builtin(env, 0, 0));
    free_environment(env);
}
static void parsed_extern_policy(void) {
    const char *names[] = {"at", "array_push"};
    for (size_t i = 0; i < sizeof names / sizeof *names; ++i) {
        char source[512];
        int length = snprintf(source, sizeof source,
            "extern fn %s(values: array<int>, index: int) -> int\n"
            "fn main() -> int { return 0 }\nshadow main { assert true }\n", names[i]);
        assert(length > 0 && (size_t)length < sizeof source);
        for (int module = 0; module < 2; ++module) {
            int count = 0;
            Token *tokens = tokenize(source, &count); assert(tokens);
            ASTNode *program = parse_program(tokens, count); assert(program);
            Environment *env = create_environment(); assert(env);
            bool checked = module ? type_check_module(program, env) : type_check(program, env);
            assert(checked == !module);
            Function *selected = env_get_function(env, names[i]);
            assert(env_function_is_builtin(selected));
            bool foreign_registered = false;
            for (int j = 0; j < env->function_count; ++j)
                if (!strcmp(env->functions[j].name, names[i]) && env->functions[j].is_extern) {
                    foreign_registered = true;
                    assert(!env_function_is_builtin(&env->functions[j]));
                }
            assert(foreign_registered == !module);
            free_environment(env); free_ast(program); free_tokens(tokens, count);
        }
    }
}
static void declaration_identity(void) {
    Environment *env = create_environment(); assert(env);
    env->current_module = "Records";
    identity_record(env, "Shared", "Records");
    identity_record(env, "Different", "Records");
    identity_record(env, "Q", "Records");
    UnionDef other = {0}; other.name = strdup("Shared"); other.module_name = strdup("Other");
    assert(other.name && other.module_name); env_define_union(env, other);
    env_define_opaque_type(env, "Shared");
    env_define_opaque_type(env, "Q");
    TypeInfo shared = {.base_type = TYPE_STRUCT, .generic_name = "Shared"};
    TypeInfo wrong = {.base_type = TYPE_STRUCT, .generic_name = "Different"};
    TypeInfo letter = {.base_type = TYPE_STRUCT, .generic_name = "Q"};
    TypeInfo missing = {.base_type = TYPE_STRUCT, .generic_name = "Z"};
    assert(nominal_array_requires_identity(env, &shared, "Records", 0));
    assert(!nominal_array_requires_identity(env, &shared, "Other", 0));
    assert(nominal_array_requires_identity(env, &shared, "Unrelated", 0));
    assert(nominal_array_requires_identity(env, &letter, "Records", 0));
    assert(nominal_array_requires_identity(env, &missing, "Records", 0));
    TypeInfo expected = {.base_type = TYPE_ARRAY, .element_type = &shared};
    TypeInfo different = {.base_type = TYPE_ARRAY, .element_type = &wrong};
    env_define_var_with_type_info(env, "good", TYPE_ARRAY, TYPE_STRUCT, &expected, false, create_void());
    env_define_var_with_type_info(env, "bad", TYPE_ARRAY, TYPE_STRUCT, &different, false, create_void());
    ASTNode value = {0}; value.type = AST_IDENTIFIER; value.as.identifier = "good";
    assert(check_nominal_array_contract(env, &expected, "Records", &value));
    value.as.identifier = "bad";
    assert(!check_nominal_array_contract(env, &expected, "Records", &value));
    TypeInfo unresolved = {.base_type = TYPE_ARRAY, .element_type = &missing};
    ASTNode empty = {0}; empty.type = AST_ARRAY_LITERAL;
    assert(!check_nominal_array_contract(env, &unresolved, "Records", &empty));
    char *formals[] = {"T"};
    UnionDef declaration = {0}; declaration.generic_param_count = 1; declaration.generic_params = formals;
    TypeInfo *arguments[] = {&shared};
    TypeInfo instance = {.type_param_count = 1, .type_params = arguments};
    NominalSubstitution context = {&declaration, &instance, "Records", NULL};
    TypeInfo formal = {.base_type = TYPE_STRUCT, .generic_name = "T"};
    assert(nominal_array_requires_context(env, &formal, "Definitions", &context, 0));
    assert(checked_annotations_equal_context(env, &formal, "Definitions", &shared, "Records", 0, &context, NULL));
    assert(!checked_annotations_equal_context(env, &formal, "Definitions", &wrong, "Records", 0, &context, NULL));
    free_environment(env);
}
static void mixed_substitution_identity(void) {
    Environment *env = create_environment(); assert(env);
    identity_record(env, "Item", "Definitions");
    identity_record(env, "Item", "Caller");
    TypeInfo fixed = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
    TypeInfo formal = {.base_type = TYPE_STRUCT, .generic_name = "T"};
    TypeInfo inner_formal = {.base_type = TYPE_STRUCT, .generic_name = "U"};
    TypeInfo fixed_array = {.base_type = TYPE_ARRAY, .element_type = &fixed};
    TypeInfo formal_array = {.base_type = TYPE_ARRAY, .element_type = &formal};
    TypeInfo nested_array = {.base_type = TYPE_ARRAY, .element_type = &inner_formal};
    char *outer_names[] = {"T"}, *inner_names[] = {"U"};
    UnionDef outer = {.generic_param_count = 1, .generic_params = outer_names};
    UnionDef inner = {.generic_param_count = 1, .generic_params = inner_names};
    TypeInfo *outer_args[] = {&fixed}, *inner_args[] = {&formal};
    TypeInfo outer_instance = {.type_param_count = 1, .type_params = outer_args};
    TypeInfo inner_instance = {.type_param_count = 1, .type_params = inner_args};
    NominalSubstitution outer_context = {&outer, &outer_instance, "Caller", NULL};
    NominalSubstitution inner_context = {&inner, &inner_instance, "Definitions", &outer_context};
    assert(checked_annotations_equal_context(env, &fixed_array, "Definitions", &fixed_array, "Definitions", 0, &outer_context, NULL));
    assert(!checked_annotations_equal_context(env, &fixed_array, "Definitions", &fixed_array, "Caller", 0, &outer_context, NULL));
    assert(checked_annotations_equal_context(env, &formal_array, "Definitions", &fixed_array, "Caller", 0, &outer_context, NULL));
    assert(!checked_annotations_equal_context(env, &formal_array, "Definitions", &fixed_array, "Definitions", 0, &outer_context, NULL));
    assert(checked_annotations_equal_context(env, &nested_array, "Definitions", &fixed_array, "Caller", 0, &inner_context, NULL));
    assert(!checked_annotations_equal_context(env, &nested_array, "Definitions", &fixed_array, "Definitions", 0, &inner_context, NULL));
    /* I retain both owners inside one full callback annotation. */
    Type tags[] = {TYPE_ARRAY, TYPE_ARRAY};
    TypeInfo *expected[] = {&fixed_array, &formal_array};
    TypeInfo *actual[] = {&fixed_array, &fixed_array};
    FunctionSignature wanted = {.param_count = 2, .param_types = tags, .param_type_info = expected, .return_type = TYPE_VOID};
    FunctionSignature wrong = {.param_count = 2, .param_types = tags, .param_type_info = actual, .return_type = TYPE_VOID};
    assert(!checked_signature_equal_context(env, &wanted, "Definitions", &wrong, "Definitions", 0, &outer_context, NULL));
    assert(!checked_signature_equal_context(env, &wanted, "Definitions", &wrong, "Caller", 0, &outer_context, NULL));
    assert(checked_signature_equal_context(env, &wanted, "Definitions", &wanted, "Definitions", 0, &outer_context, &outer_context));
    free_environment(env);
}
static void identity_union(Environment *env, const char *name, const char *formal,
                           const char **fields, TypeInfo **annotations, int count) {
    UnionDef definition = {0};
    definition.name = strdup(name); definition.module_name = strdup("Definitions");
    definition.generic_param_count = 1; definition.generic_params = calloc(1, sizeof(char *));
    definition.variant_count = 1; definition.variant_names = calloc(1, sizeof(char *));
    definition.variant_field_counts = calloc(1, sizeof(int));
    definition.variant_field_names = calloc(1, sizeof(char **));
    definition.variant_field_types = calloc(1, sizeof(Type *));
    definition.variant_field_type_info = calloc(1, sizeof(TypeInfo **));
    assert(definition.name && definition.module_name && definition.generic_params && definition.variant_names &&
           definition.variant_field_counts && definition.variant_field_names && definition.variant_field_types && definition.variant_field_type_info);
    definition.generic_params[0] = strdup(formal); definition.variant_names[0] = strdup("Payload");
    definition.variant_field_counts[0] = count;
    definition.variant_field_names[0] = calloc((size_t)count, sizeof(char *));
    definition.variant_field_types[0] = calloc((size_t)count, sizeof(Type));
    definition.variant_field_type_info[0] = calloc((size_t)count, sizeof(TypeInfo *));
    assert(definition.generic_params[0] && definition.variant_names[0] && definition.variant_field_names[0] &&
           definition.variant_field_types[0] && definition.variant_field_type_info[0]);
    for (int i = 0; i < count; ++i) {
        definition.variant_field_names[0][i] = strdup(fields[i]);
        definition.variant_field_types[0][i] = annotations[i]->base_type;
        assert(definition.variant_field_names[0][i] &&
               copy_payload_type_info_checked(annotations[i], &definition.variant_field_type_info[0][i]));
    }
    env_define_union(env, definition);
    assert(env_register_nominal_import(env, "Caller", name,
        env_nominal_identity(env, name, "Definitions", TYPE_UNION)));
}
static void nested_payload_views(void) {
    Environment *env = create_environment(); assert(env);
    identity_record(env, "Item", "Definitions"); identity_record(env, "Item", "Caller");
    TypeInfo item = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
    TypeInfo t = {.base_type = TYPE_STRUCT, .generic_name = "T"};
    TypeInfo fixed = {.base_type = TYPE_LIST_GENERIC, .generic_name = "Item"};
    TypeInfo supplied = {.base_type = TYPE_LIST_GENERIC, .generic_name = "U"};
    TypeInfo *inner_args[] = {&t};
    TypeInfo inner = {.base_type = TYPE_UNION, .generic_name = "Inner", .type_param_count = 1, .type_params = inner_args};
    const char *inner_fields[] = {"fixed", "supplied"}, *outer_fields[] = {"inner"};
    TypeInfo *inner_annotations[] = {&fixed, &supplied}, *outer_annotations[] = {&inner};
    identity_union(env, "Inner", "U", inner_fields, inner_annotations, 2);
    identity_union(env, "Outer", "T", outer_fields, outer_annotations, 1);
    TypeInfo *arguments[] = {&item};
    TypeInfo root = {.base_type = TYPE_UNION, .generic_name = "Outer", .type_param_count = 1, .type_params = arguments};
    env->current_module = "Caller";
    env_define_var_with_type_info(env, "value", TYPE_UNION, TYPE_UNKNOWN, &root, false, create_void());
    env_define_var(env, "outer", TYPE_STRUCT, false, create_void());
    ASTNode value = {.type = AST_IDENTIFIER}; value.as.identifier = "value";
    assert(retain_union_binding_context(env, env_get_var(env, "outer"), &value, "Payload"));
    ASTNode outer = {.type = AST_IDENTIFIER}; outer.as.identifier = "outer";
    ASTNode field = {.type = AST_FIELD_ACCESS}; field.as.field_access.object = &outer; field.as.field_access.field_name = "inner";
    NominalView alias = {0}; assert(nominal_value_view(&field, env, 0, &alias));
    assert(alias.owned_context && alias.info->base_type == TYPE_UNION && !alias.payload);
    env_define_var(env, "alias", TYPE_UNION, false, create_void());
    assert(nominal_view_retain(env, env_get_var(env, "alias"), &alias));
    assert(!alias.info && !alias.owner && !alias.owned_context);
    /* My retained chain no longer borrows the original argument tree. */
    item.generic_name = "Missing";
    ASTNode inner_value = {.type = AST_IDENTIFIER}; inner_value.as.identifier = "alias";
    env_define_var(env, "payload", TYPE_STRUCT, false, create_void());
    assert(retain_union_binding_context(env, env_get_var(env, "payload"), &inner_value, "Payload"));
    assert(!retain_union_binding_context(env, env_get_var(env, "payload"), &inner_value, "Absent"));
    ASTNode payload = {.type = AST_IDENTIFIER}; payload.as.identifier = "payload";
    field.as.field_access.object = &payload;
    for (int i = 0; i < 2; ++i) {
        field.as.field_access.field_name = (char *)inner_fields[i];
        NominalView projected = {0}; assert(nominal_value_view(&field, env, 0, &projected));
        NominalIdentity expected = env_nominal_identity(env, "Item", i ? "Caller" : "Definitions", TYPE_STRUCT);
        assert(nominal_equal(nominal_view_identity(env, &projected, TYPE_LIST_GENERIC), expected));
        TypeInfo actual = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
        TypeInfo *parameter[] = {&actual};
        TypeInfo explicit_list = {.base_type = TYPE_LIST_GENERIC, .generic_name = "List", .type_param_count = 1, .type_params = parameter};
        assert(checked_annotations_equal_context(env, projected.info, projected.owner, &explicit_list,
            i ? "Caller" : "Definitions", 0, nominal_view_context(&projected), NULL));
        assert(!checked_annotations_equal_context(env, projected.info, projected.owner, &explicit_list,
            i ? "Definitions" : "Caller", 0, nominal_view_context(&projected), NULL));
        TypeInfo *concrete = NULL;
        assert(nominal_materialize(env, projected.info, projected.owner, nominal_view_context(&projected), 0, &concrete));
        assert(concrete->base_type == TYPE_LIST_GENERIC && concrete->type_param_count == 1 &&
               !strcmp(concrete->type_params[0]->generic_name, "Item"));
        char *compact_key = typeinfo_to_generic_arg_name(concrete);
        char *explicit_key = typeinfo_to_generic_arg_name(&explicit_list);
        assert(compact_key && explicit_key && !strcmp(compact_key, explicit_key));
        free(compact_key); free(explicit_key);
        free_payload_type_info(concrete); nominal_view_discard(&projected);
    }
    /* A payload alias keeps its variant, not only its union arguments. */
    assert(nominal_value_view(&payload, env, 0, &alias) && alias.payload);
    env_define_var(env, "payload_alias", TYPE_STRUCT, false, create_void());
    assert(nominal_view_retain(env, env_get_var(env, "payload_alias"), &alias));
    payload.as.identifier = "payload_alias";
    assert(nominal_expression(&field, env, TYPE_LIST_GENERIC, 0).ordinal ==
           env_nominal_identity(env, "Item", "Caller", TYPE_STRUCT).ordinal);
    NominalView untouched = {.info = &root, .owner = "sentinel"};
    assert(!nominal_view_copy_context(env, &root, "Caller", NULL, 0, &untouched));
    assert(untouched.info == &root && !strcmp(untouched.owner, "sentinel"));
    assert(!nominal_view_copy_context(env, &fixed, "Definitions", NULL, 129, &untouched));
    assert(untouched.info == &root && !strcmp(untouched.owner, "sentinel"));
    free_environment(env);
}
static void retained_callable_consumers(void) {
    Environment *env = create_environment(); assert(env);
    identity_record(env, "Item", "Definitions"); identity_record(env, "Item", "Caller");
    assert(env_register_nominal_import(env, "Caller", "FixedItem",
        env_nominal_identity(env, "Item", "Definitions", TYPE_STRUCT)));
    TypeInfo fixed = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
    TypeInfo formal = {.base_type = TYPE_STRUCT, .generic_name = "T"};
    Type tags[] = {TYPE_STRUCT, TYPE_STRUCT};
    TypeInfo *parameters[] = {&fixed, &formal};
    char *names[] = {"Item", "T"};
    FunctionSignature signature = {.param_count = 2, .param_types = tags, .param_type_info = parameters,
        .param_struct_names = names, .return_type = TYPE_STRUCT, .return_type_info = &formal, .return_struct_name = "T"};
    TypeInfo callable = {.base_type = TYPE_FUNCTION, .fn_sig = &signature};
    FunctionSignature factory_signature = {.return_type = TYPE_FUNCTION, .return_type_info = &callable, .return_fn_sig = &signature};
    TypeInfo factory = {.base_type = TYPE_FUNCTION, .fn_sig = &factory_signature};
    Type unary_tags[] = {TYPE_STRUCT}; TypeInfo *unary_parameters[] = {&fixed}; char *unary_names[] = {"Item"};
    FunctionSignature mapper_signature = {.param_count = 1, .param_types = unary_tags,
        .param_type_info = unary_parameters, .param_struct_names = unary_names,
        .return_type = TYPE_STRUCT, .return_type_info = &formal, .return_struct_name = "T"};
    FunctionSignature predicate_signature = mapper_signature;
    predicate_signature.return_type = TYPE_BOOL; predicate_signature.return_type_info = NULL;
    predicate_signature.return_struct_name = NULL;
    TypeInfo mapper = {.base_type = TYPE_FUNCTION, .fn_sig = &mapper_signature};
    TypeInfo predicate = {.base_type = TYPE_FUNCTION, .fn_sig = &predicate_signature};
    const char *fields[] = {"callback", "factory", "mapper", "predicate"};
    TypeInfo *annotations[] = {&callable, &factory, &mapper, &predicate};
    identity_union(env, "Callbacks", "T", fields, annotations, 4);
    TypeInfo *arguments[] = {&fixed};
    TypeInfo instance = {.base_type = TYPE_UNION, .generic_name = "Callbacks", .type_param_count = 1, .type_params = arguments};
    env->current_module = "Caller";
    ASTNode constructor = {.type = AST_UNION_CONSTRUCT};
    constructor.as.union_construct.union_name = "Callbacks";
    constructor.as.union_construct.variant_name = "Payload";
    assert(nominal_constructor_retain(env, &constructor, &instance, "Caller", NULL, 0));
    const NominalView *origin = nominal_constructor_view(env, &constructor);
    assert(origin && nominal_constructor_retain(env, &constructor, &instance, "Caller", NULL, 0));
    assert(origin == nominal_constructor_view(env, &constructor));
    assert(!nominal_constructor_retain(env, &constructor, &instance, "Definitions", NULL, 0));
    env->current_module = "Definitions";
    NominalView constructor_view = {0};
    assert(nominal_value_view(&constructor, env, 0, &constructor_view));
    assert(!strcmp(constructor_view.owner, "Caller")); nominal_view_discard(&constructor_view);
    env->current_module = "Caller";
    env_define_var_with_type_info(env, "source", TYPE_UNION, TYPE_UNKNOWN, &instance, false, create_void());
    env_define_var(env, "payload", TYPE_STRUCT, false, create_void());
    ASTNode source = {.type = AST_IDENTIFIER}; source.as.identifier = "source";
    assert(retain_union_binding_context(env, env_get_var(env, "payload"), &source, "Payload"));
    env_get_var(env, "payload")->struct_type_name = strdup("Callbacks.Payload");
    assert(env_get_var(env, "payload")->struct_type_name);
    ASTNode payload = {.type = AST_IDENTIFIER}; payload.as.identifier = "payload";
    ASTNode field = {.type = AST_FIELD_ACCESS}; field.as.field_access.object = &payload; field.as.field_access.field_name = "callback";
    NominalView projected = {0}; assert(nominal_callable_view(&field, env, 0, &projected));
    env_define_var(env, "alias", TYPE_FUNCTION, false, create_void());
    assert(nominal_view_retain(env, env_get_var(env, "alias"), &projected));
    ASTNode alias = {.type = AST_IDENTIFIER}; alias.as.identifier = "alias";
    TypeInfo fixed_alias = {.base_type = TYPE_STRUCT, .generic_name = "FixedItem"};
    TypeInfo *expected_parameters[] = {&fixed_alias, &fixed}; char *expected_names[] = {"FixedItem", "Item"};
    FunctionSignature expected = {.param_count = 2, .param_types = tags, .param_type_info = expected_parameters,
        .param_struct_names = expected_names, .return_type = TYPE_STRUCT, .return_type_info = &fixed, .return_struct_name = "Item"};
    assert(check_callable_contract(env, &expected, "Caller", &field, 0));
    assert(check_callable_contract(env, &expected, "Caller", &alias, 0));
    TypeInfo *swapped_parameters[] = {&fixed, &fixed_alias}; char *swapped_names[] = {"Item", "FixedItem"};
    FunctionSignature swapped = expected; swapped.param_type_info = swapped_parameters; swapped.param_struct_names = swapped_names;
    assert(!check_callable_contract(env, &swapped, "Caller", &alias, 0));
    ASTNode branch = {.type = AST_IF}; branch.as.if_stmt.then_branch = &field; branch.as.if_stmt.else_branch = &alias;
    assert(nominal_callable_view(&branch, env, 0, &projected)); nominal_view_discard(&projected);
    ASTNode terminal = {.type = AST_RETURN}; terminal.as.return_stmt.value = &alias;
    branch.as.if_stmt.then_branch = &terminal;
    assert(nominal_callable_view(&branch, env, 0, &projected)); nominal_view_discard(&projected);
    branch.as.if_stmt.else_branch = &terminal;
    assert(!nominal_callable_view(&branch, env, 0, &projected) && !projected.info);
    branch.as.if_stmt.then_branch = &field;
    TypeInfo wrong_callable = {.base_type = TYPE_FUNCTION, .fn_sig = &swapped};
    env_define_var_with_type_info(env, "wrong", TYPE_FUNCTION, TYPE_UNKNOWN, &wrong_callable, false, create_void());
    ASTNode wrong = {.type = AST_IDENTIFIER}; wrong.as.identifier = "wrong";
    branch.as.if_stmt.else_branch = &wrong;
    assert(!nominal_callable_view(&branch, env, 0, &projected) && !projected.info);
    env_define_var_with_type_info(env, "fixed", TYPE_STRUCT, TYPE_UNKNOWN, &fixed_alias, false, create_void());
    env_define_var_with_type_info(env, "actual", TYPE_STRUCT, TYPE_UNKNOWN, &fixed, false, create_void());
    ASTNode first = {.type = AST_IDENTIFIER}, second = {.type = AST_IDENTIFIER};
    first.as.identifier = "fixed"; second.as.identifier = "actual";
    ASTNode *call_args[] = {&first, &second};
    ASTNode call = {.type = AST_CALL}; call.as.call.name = "alias"; call.as.call.arg_count = 2; call.as.call.args = call_args;
    assert(check_indirect_call(&call, env, NULL) == TYPE_STRUCT);
    assert(nominal_expression(&call, env, TYPE_STRUCT, 0).ordinal == env_nominal_identity(env, "Item", "Caller", TYPE_STRUCT).ordinal);
    assert(!contextual_argument_matches(&second, env, &fixed, "Definitions", NULL, 0));
    assert(!contextual_argument_matches(&first, env, &formal, "Definitions",
        nominal_view_context(env_get_var(env, "alias")->checker_nominal_view), 0));
    /* I consume retained selected-field aliases through both array callbacks. */
    for (int i = 0; i < 2; ++i) {
        const char *name = i ? "predicate" : "mapper";
        field.as.field_access.field_name = (char *)name;
        assert(nominal_callable_view(&field, env, 0, &projected));
        env_define_var(env, name, TYPE_FUNCTION, false, create_void());
        assert(nominal_view_retain(env, env_get_var(env, name), &projected));
        ASTNode callback_alias = {.type = AST_IDENTIFIER}; callback_alias.as.identifier = (char *)name;
        ASTNode *elements[] = {&first}; ASTNode array = {.type = AST_ARRAY_LITERAL};
        array.as.array_literal.elements = elements; array.as.array_literal.element_count = 1;
        ASTNode *operands[] = {&array, &callback_alias}; ASTNode operation = {.type = AST_CALL};
        operation.as.call.name = i ? "filter" : "map";
        operation.as.call.args = operands; operation.as.call.arg_count = 2;
        assert(nominal_value_view(&operation, env, 0, &projected));
        assert(nominal_view_element(env, &projected, 0));
        NominalIdentity expected_owner = env_nominal_identity(env, "Item", i ? "Definitions" : "Caller", TYPE_STRUCT);
        assert(nominal_equal(nominal_view_identity(env, &projected, TYPE_STRUCT), expected_owner));
        nominal_view_discard(&projected);
        elements[0] = &second;
        assert(!nominal_value_view(&operation, env, 0, &projected) && !projected.info);
    }
    field.as.field_access.field_name = "factory";
    ASTNode factory_call = {.type = AST_CALL}; factory_call.as.call.func_expr = &field;
    assert(check_indirect_call(&factory_call, env, NULL) == TYPE_FUNCTION);
    assert(nominal_callable_view(&factory_call, env, 0, &projected));
    assert(projected.owned_context); nominal_view_discard(&projected);
    call.as.call.name = NULL; call.as.call.func_expr = &factory_call;
    assert(check_indirect_call(&call, env, NULL) == TYPE_STRUCT);
    free_function_signature(call.as.call.checked_signature); free(call.as.call.return_struct_type_name);
    free_function_signature(factory_call.as.call.checked_signature); free(factory_call.as.call.return_struct_type_name);
    free_environment(env);
}
static void constructor_failure_rollback(void) {
    for (int invalid = 0; invalid < 2; ++invalid) {
        char source[512];
        int length = snprintf(source, sizeof source,
            "struct Item { value:int }\n"
            "union Box<T> { Value { item:T } }\n"
            "fn sample()->int { let box:Box<Item> =Box<Item>.Value{item:Item{value:3}} return %s }\n"
            "shadow sample { assert true }\n", invalid ? "true" : "0");
        assert(length > 0 && (size_t)length < sizeof source);
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        Environment *env = create_environment(); assert(env);
        bool ok = type_check_module(program, env);
        assert(ok == !invalid);
        assert((env->checker_nominal_expressions != NULL) == !invalid);
        /* I mirror fresh failed-loader/root teardown: destructors never read keys. */
        free_ast(program); free_tokens(tokens, count); free_environment(env);
    }
}
int main(void) {
    intrinsic_identity(); parsed_extern_policy(); declaration_identity(); mixed_substitution_identity(); nested_payload_views(); retained_callable_consumers(); constructor_failure_rollback();
    puts("I checked actual builtin objects and owner-bound array declaration obligations.");
    return 0;
}
