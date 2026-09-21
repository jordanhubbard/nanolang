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
static void assignment_branch_growth(Environment *env, const NominalView *proof,
                                     TypeInfo *explicit_type, ASTNode *value, bool valid) {
    env_define_var_with_type_info(env, "assignment_target", TYPE_FUNCTION, TYPE_UNKNOWN,
                                 explicit_type, true, create_void());
    int destination_index = env->symbol_count - 1;
    if (proof) {
        NominalView copy = {0}; assert(nominal_view_clone(env, proof, 0, &copy));
        assert(nominal_view_retain(env, &env->symbols[destination_index], &copy));
    }
    const void *retained = env->symbols[destination_index].checker_nominal_view;
    int capacity = env->symbol_capacity;
    assert(capacity > 0 && capacity < 4096);
    int count = capacity + 1;
    ASTNode *bindings = calloc((size_t)count, sizeof *bindings);
    ASTNode **statements = calloc((size_t)count + 1, sizeof *statements);
    char (*names)[40] = calloc((size_t)count, sizeof *names);
    assert(bindings && statements && names);
    ASTNode number = {.type = AST_NUMBER}, boolean = {.type = AST_BOOL};
    boolean.as.bool_val = true;
    for (int i = 0; i < count; ++i) {
        snprintf(names[i], sizeof names[i], "assignment_local_%d", i);
        bindings[i].type = AST_LET;
        bindings[i].as.let.name = i == count - 1 ? "assignment_target" : names[i];
        bindings[i].as.let.var_type = i == count - 1 ? TYPE_BOOL : TYPE_INT;
        bindings[i].as.let.value = i == count - 1 ? &boolean : &number;
        statements[i] = &bindings[i];
    }
    statements[count] = value;
    ASTNode block = {.type = AST_BLOCK}; block.as.block.statements = statements; block.as.block.count = count + 1;
    ASTNode *conditions[] = {&boolean}, *values[] = {&block};
    ASTNode branch = {.type = AST_COND};
    branch.as.cond_expr.clause_count = 1; branch.as.cond_expr.conditions = conditions;
    branch.as.cond_expr.values = values; branch.as.cond_expr.else_value = value;
    ASTNode assignment = {.type = AST_SET};
    assignment.as.set.name = "assignment_target"; assignment.as.set.value = &branch;
    TypeChecker checker = {.env = env};
    int errors = g_typecheck_error_count;
    check_statement(&checker, &assignment);
    assert(checker.has_error == !valid);
    assert(env->symbol_capacity > capacity);
    assert(env->symbols[destination_index].type == TYPE_FUNCTION);
    assert(env->symbols[destination_index].checker_nominal_view == retained);
    /* Name-only reacquisition would select this branch-local Boolean instead. */
    assert(env_get_var(env, "assignment_target")->type == TYPE_BOOL);
    g_typecheck_error_count = errors;
    for (int i = 0; i < count; ++i) assert(!bindings[i].as.let.type_info);
    free(names); free(statements); free(bindings);
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
    Type payload_tags[] = {TYPE_STRUCT, TYPE_STRUCT, TYPE_FUNCTION};
    char *payload_names[] = {"Item", "T", NULL};
    TypeInfo *payload_children[] = {&fixed, &formal, &callable};
    TypeInfo payload_tuple = {.base_type = TYPE_TUPLE, .tuple_element_count = 3,
        .tuple_types = payload_tags, .tuple_type_names = payload_names,
        .type_param_count = 3, .type_params = payload_children};
    const char *fields[] = {"callback", "factory", "mapper", "predicate", "tuple"};
    TypeInfo *annotations[] = {&callable, &factory, &mapper, &predicate, &payload_tuple};
    identity_union(env, "Callbacks", "T", fields, annotations, 5);
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
    field.as.field_access.field_name = "callback";
    /* I compose independently owned children, including a retained mixed-owner callback. */
    ASTNode *tuple_items[] = {&first, &second, &alias};
    ASTNode tuple = {.type = AST_TUPLE_LITERAL};
    tuple.as.tuple_literal.elements = tuple_items; tuple.as.tuple_literal.element_count = 3;
    NominalView composed = {0}, copied = {0};
    assert(nominal_value_view(&tuple, env, 0, &composed));
    assert(composed.children && composed.child_count == 3 && nominal_view_composed_valid(&composed));
    assert(nominal_view_clone(env, &composed, 0, &copied));
    assert(copied.children != composed.children && nominal_view_equal(env, &composed, &copied, 0));
    assert(nominal_view_child(env, &copied, 0, 0, &projected));
    assert(nominal_equal(nominal_view_identity(env, &projected, TYPE_STRUCT),
        env_nominal_identity(env, "Item", "Definitions", TYPE_STRUCT)));
    nominal_view_discard(&projected);
    assert(nominal_view_child(env, &copied, 1, 0, &projected));
    assert(nominal_equal(nominal_view_identity(env, &projected, TYPE_STRUCT),
        env_nominal_identity(env, "Item", "Caller", TYPE_STRUCT)));
    nominal_view_discard(&projected);
    env_define_var(env, "packed", TYPE_TUPLE, true, create_void());
    assert(nominal_view_retain(env, env_get_var(env, "packed"), &composed));
    nominal_view_discard(&copied);
    ASTNode packed = {.type = AST_IDENTIFIER}; packed.as.identifier = "packed";
    ASTNode index = {.type = AST_TUPLE_INDEX}; index.as.tuple_index.tuple = &packed; index.as.tuple_index.index = 2;
    assert(check_expression(&index, env) == TYPE_FUNCTION);
    assert(check_callable_contract(env, &expected, "Caller", &index, 0));
    assert(!check_callable_contract(env, &swapped, "Caller", &index, 0));
    ASTNode packed_call = {.type = AST_CALL}; packed_call.as.call.func_expr = &index;
    packed_call.as.call.arg_count = 2; packed_call.as.call.args = call_args;
    assert(check_indirect_call(&packed_call, env, NULL) == TYPE_STRUCT);
    free_function_signature(packed_call.as.call.checked_signature); free(packed_call.as.call.return_struct_type_name);
    const NominalView *packed_proof = env_get_var(env, "packed")->checker_nominal_view;
    assert(nominal_view_matches_value(env, packed_proof, &tuple, 0));
    ASTNode *swapped_items[] = {&second, &first, &alias}; ASTNode swapped_tuple = tuple;
    swapped_tuple.as.tuple_literal.elements = swapped_items;
    assert(!nominal_view_matches_value(env, packed_proof, &swapped_tuple, 0));
    ASTNode *nested_items[] = {&packed, &alias}; ASTNode nested_tuple = {.type = AST_TUPLE_LITERAL};
    nested_tuple.as.tuple_literal.elements = nested_items; nested_tuple.as.tuple_literal.element_count = 2;
    ASTNode inner_index = {.type = AST_TUPLE_INDEX}; inner_index.as.tuple_index.tuple = &nested_tuple;
    ASTNode outer_index = index; outer_index.as.tuple_index.tuple = &inner_index;
    assert(check_callable_contract(env, &expected, "Caller", &outer_index, 0));
    ASTNode *array_items[] = {&packed, &tuple}; ASTNode tuples = {.type = AST_ARRAY_LITERAL};
    tuples.as.array_literal.elements = array_items; tuples.as.array_literal.element_count = 2;
    assert(nominal_value_view(&tuples, env, 0, &composed));
    assert(composed.children && nominal_view_element(env, &composed, 0));
    assert(nominal_view_equal(env, &composed, packed_proof, 0)); nominal_view_discard(&composed);
    array_items[1] = &swapped_tuple;
    assert(!nominal_value_view(&tuples, env, 0, &composed) && !composed.info);
    /* I also project through a real selected payload's declaration context. */
    ASTNode payload_field = {.type = AST_FIELD_ACCESS};
    payload_field.as.field_access.object = &payload;
    payload_field.as.field_access.field_name = "tuple";
    ASTNode payload_index = {.type = AST_TUPLE_INDEX};
    payload_index.as.tuple_index.tuple = &payload_field; payload_index.as.tuple_index.index = 2;
    assert(check_callable_contract(env, &expected, "Caller", &payload_index, 0));
    assert(!check_callable_contract(env, &swapped, "Caller", &payload_index, 0));
    assert(nominal_value_view(&payload_field, env, 0, &projected));
    assert(nominal_view_equal(env, packed_proof, &projected, 0)); nominal_view_discard(&projected);
    branch.as.if_stmt.then_branch = &payload_index; branch.as.if_stmt.else_branch = &index;
    assert(nominal_callable_view(&branch, env, 0, &projected)); nominal_view_discard(&projected);
    assert(nominal_callable_view(&payload_index, env, 0, &projected));
    env_define_var(env, "projected_callback", TYPE_FUNCTION, false, create_void());
    assert(nominal_view_retain(env, env_get_var(env, "projected_callback"), &projected));
    ASTNode projected_alias = {.type = AST_IDENTIFIER}; projected_alias.as.identifier = "projected_callback";
    assert(check_callable_contract(env, &expected, "Caller", &projected_alias, 0));
    ASTNode projected_call = {.type = AST_CALL}; projected_call.as.call.func_expr = &projected_alias;
    projected_call.as.call.arg_count = 2; projected_call.as.call.args = call_args;
    assert(check_indirect_call(&projected_call, env, NULL) == TYPE_STRUCT);
    free_function_signature(projected_call.as.call.checked_signature);
    free(projected_call.as.call.return_struct_type_name);
    index.as.tuple_index.index = 3;
    assert(!nominal_value_view(&index, env, 0, &projected) && !projected.info);
    const NominalView *assignment_proof = env_get_var(env, "alias")->checker_nominal_view;
    assert(assignment_proof);
    assignment_branch_growth(env, assignment_proof, NULL, &field, true);
    assignment_branch_growth(env, assignment_proof, NULL, &wrong, false);
    TypeInfo explicit_destination = {.base_type = TYPE_FUNCTION, .fn_sig = &expected};
    assignment_branch_growth(env, NULL, &explicit_destination, &field, true);
    assignment_branch_growth(env, NULL, &explicit_destination, &wrong, false);
    free_payload_type_info(field.as.field_access.resolved_type_info);
    field.as.field_access.resolved_type_info = NULL;
    free_payload_type_info(payload_field.as.field_access.resolved_type_info);
    payload_field.as.field_access.resolved_type_info = NULL;
    free_environment(env);
}
static void complete_tuple_annotations(void) {
    const char *source =
        "struct Item { value:int }\n"
        "fn sample(values:(fn(Item)->Item,array<Item>,(Item,fn()->Item)))->int { return 0 }\n"
        "shadow sample { assert true }\n";
    int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
    ASTNode *program = parse_program(tokens, count); assert(program && program->as.program.count == 3);
    TypeInfo *tuple = program->as.program.items[1]->as.function.params[0].type_info;
    assert(tuple && type_info_tuple_valid(tuple) && tuple->type_param_count == 3);
    TypeInfo flat;
    const TypeInfo *callback = type_info_tuple_element(tuple, 0, &flat);
    assert(callback && callback->fn_sig && callback->fn_sig->param_count == 1);
    const TypeInfo *array = type_info_tuple_element(tuple, 1, &flat);
    assert(array && array->element_type && !strcmp(array->element_type->generic_name, "Item"));
    const TypeInfo *nested = type_info_tuple_element(tuple, 2, &flat);
    assert(nested && type_info_tuple_valid(nested));
    const TypeInfo *factory = type_info_tuple_element(nested, 1, &flat);
    assert(factory && factory->fn_sig && factory->fn_sig->return_type == TYPE_STRUCT);
    TypeInfo *copy = NULL;
    assert(copy_payload_type_info_checked(tuple, &copy));
    assert(copy->type_params[0] != callback && type_infos_equal(tuple, copy));
    copy->tuple_types[0] = TYPE_INT;
    assert(!type_info_tuple_valid(copy) && !type_infos_equal(tuple, copy));
    TypeInfo sentinel = {.base_type = TYPE_BOOL}, *output = &sentinel;
    assert(!copy_payload_type_info_checked(copy, &output) && output == &sentinel);
    free_payload_type_info(copy);
    Environment *env = create_environment(); assert(env);
    assert(type_check_module(program, env));
    free_environment(env); free_ast(program); free_tokens(tokens, count);
    /* I refuse missing metadata before the old flat fallback can publish it. */
    env = create_environment(); assert(env);
    ASTNode number = {.type = AST_NUMBER}, literal = {.type = AST_TUPLE_LITERAL};
    ASTNode *elements[] = {&number};
    literal.as.tuple_literal.elements = elements; literal.as.tuple_literal.element_count = 1;
    ASTNode binding = {.type = AST_LET};
    binding.as.let.name = "missing_tuple_annotation"; binding.as.let.var_type = TYPE_TUPLE;
    binding.as.let.value = &literal;
    TypeChecker checker = {.env = env}; int errors = g_typecheck_error_count;
    check_statement(&checker, &binding);
    assert(checker.has_error && !env_get_var(env, binding.as.let.name) && !binding.as.let.type_info);
    g_typecheck_error_count = errors;
    free(literal.as.tuple_literal.element_types);
    free_environment(env);
}
static void constructor_annotation_parsing(void) {
    const char *source =
        "fn sample()->int { let box =Box<Item,fn(Item)->Item,array<Item>,List<Item>,Box<Item>,fn()->fn(Item)->Item,fn()->fn(T)->Item>.Value{} return 0 }\n"
        "shadow sample { assert true }\n";
    int count = 0;
    Token *tokens = tokenize(source, &count); assert(tokens);
    ASTNode *program = parse_program(tokens, count); assert(program);
    assert(program->as.program.count == 2);
    ASTNode *body = program->as.program.items[0]->as.function.body;
    assert(body && body->type == AST_BLOCK && body->as.block.count == 2);
    ASTNode *constructor = body->as.block.statements[0]->as.let.value;
    assert(constructor && constructor->type == AST_UNION_CONSTRUCT);
    TypeInfo *info = constructor->as.union_construct.type_info;
    assert(info && info->type_param_count == 7 && !strcmp(info->generic_name, "Box"));
    TypeInfo **args = info->type_params;
    assert(args[0]->base_type == TYPE_STRUCT && !strcmp(args[0]->generic_name, "Item"));
    assert(args[1]->base_type == TYPE_FUNCTION && args[1]->fn_sig);
    FunctionSignature *signature = args[1]->fn_sig;
    assert(signature->param_count == 1 && signature->param_types[0] == TYPE_STRUCT);
    assert(!strcmp(signature->param_struct_names[0], "Item"));
    assert(signature->return_type == TYPE_STRUCT && !strcmp(signature->return_struct_name, "Item"));
    assert(args[2]->base_type == TYPE_ARRAY && args[2]->element_type);
    assert(args[2]->element_type->base_type == TYPE_STRUCT && !strcmp(args[2]->element_type->generic_name, "Item"));
    assert(args[3]->base_type == TYPE_LIST_GENERIC && !strcmp(args[3]->generic_name, "Item"));
    assert(args[4]->base_type == TYPE_UNION && args[4]->type_param_count == 1);
    assert(!strcmp(args[4]->generic_name, "Box") && !strcmp(args[4]->type_params[0]->generic_name, "Item"));
    assert(args[5]->base_type == TYPE_FUNCTION && args[5]->fn_sig);
    FunctionSignature *factory = args[5]->fn_sig;
    assert(factory->return_type_info && factory->return_type_info->base_type == TYPE_FUNCTION);
    assert(!factory->return_type_info->fn_sig && factory->return_fn_sig);
    Environment *env = create_environment(); assert(env);
    identity_record(env, "Item", "Parsed");
    NominalView callee = {0}, result = {0}; TypeInfo *concrete = NULL;
    assert(nominal_view_copy_context(env, args[5], "Parsed", NULL, 0, &callee));
    assert(nominal_callable_result(env, &callee, 0, &result));
    assert(result.info->fn_sig && result.info->fn_sig != factory->return_fn_sig);
    assert(checked_signature_equal(env, result.info->fn_sig, "Parsed", factory->return_fn_sig, "Parsed", 0));
    assert(nominal_materialize(env, args[5], "Parsed", NULL, 0, &concrete));
    assert(!concrete->fn_sig->return_type_info->fn_sig && concrete->fn_sig->return_fn_sig);
    assert(concrete->fn_sig->return_fn_sig != factory->return_fn_sig);
    free_payload_type_info(concrete); concrete = NULL;
    char *formals[] = {"T"}; UnionDef declaration = {.generic_param_count = 1, .generic_params = formals};
    TypeInfo integer = {.base_type = TYPE_INT}; TypeInfo *parameters[] = {&integer};
    TypeInfo instance = {.type_param_count = 1, .type_params = parameters};
    NominalSubstitution context = {&declaration, &instance, "Parsed", NULL};
    assert(args[6]->base_type == TYPE_FUNCTION && args[6]->fn_sig);
    assert(nominal_materialize(env, args[6], "Parsed", &context, 0, &concrete));
    assert(concrete->fn_sig->return_type_info->fn_sig && concrete->fn_sig->return_fn_sig);
    assert(concrete->fn_sig->return_fn_sig->param_types[0] == TYPE_INT);
    assert(args[6]->fn_sig->return_fn_sig->param_types[0] == TYPE_STRUCT);
    assert(!strcmp(args[6]->fn_sig->return_fn_sig->param_struct_names[0], "T"));
    assert(!args[6]->fn_sig->return_type_info->fn_sig);
    assert(concrete->fn_sig->return_type_info->fn_sig != concrete->fn_sig->return_fn_sig);
    assert(checked_signature_equal(env, concrete->fn_sig->return_type_info->fn_sig, "Parsed",
        concrete->fn_sig->return_fn_sig, "Parsed", 0));
    assert(!factory->return_type_info->fn_sig); /* My borrowed parser source is unchanged. */
    FunctionSignature conflict = *factory->return_fn_sig;
    conflict.return_type = TYPE_BOOL; conflict.return_type_info = NULL; conflict.return_struct_name = NULL;
    TypeInfo conflicting_return = *factory->return_type_info; conflicting_return.fn_sig = &conflict;
    FunctionSignature conflicting_factory = *factory; conflicting_factory.return_type_info = &conflicting_return;
    TypeInfo conflicting = {.base_type = TYPE_FUNCTION, .fn_sig = &conflicting_factory};
    NominalView untouched = {.variant = 17};
    assert(!nominal_view_copy_context(env, &conflicting, "Parsed", NULL, 0, &untouched));
    assert(untouched.variant == 17 && !untouched.info && !untouched.owner);
    free_payload_type_info(concrete); nominal_view_discard(&result); nominal_view_discard(&callee);
    free_environment(env);
    free_ast(program); free_tokens(tokens, count);
    const char *malformed[] = {
        "fn sample()->int { let box =Box<Item,fn(Item)-> >.Value{} return 0 }",
        "fn sample()->int { let box =Box<Item,array<Item>.Value{} return 0 }"
    };
    for (size_t i = 0; i < sizeof malformed / sizeof *malformed; ++i) {
        tokens = tokenize(malformed[i], &count); assert(tokens);
        program = parse_program(tokens, count); assert(!program);
        free_tokens(tokens, count);
    }
}
static void dotted_constructor_checking(void) {
    const char *declaration = "struct Item { value:int } union Box<T> { Value { item:T } }\n";
    for (int forward = 0; forward < 2; ++forward)
    for (int explicit_type = 0; explicit_type < 2; ++explicit_type)
    for (int invalid = 0; invalid < 3; ++invalid) {
        char source[1024];
        int length = snprintf(source, sizeof source,
            "%sfn sample()->int { let box:Box<Item> =%s.%s{item:%s} return 0 }\n"
            "shadow sample { assert true }\n%s",
            forward ? "" : declaration, explicit_type ? "Box<Item>" : "Box",
            invalid == 1 ? "Missing" : "Value", invalid == 2 ? "true" : "Item{value:3}",
            forward ? declaration : "");
        assert(length > 0 && (size_t)length < sizeof source);
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        ASTNode *function = NULL;
        for (int i = 0; i < program->as.program.count; ++i)
            if (program->as.program.items[i]->type == AST_FUNCTION) function = program->as.program.items[i];
        assert(function && function->as.function.body->type == AST_BLOCK);
        ASTNode *node = function->as.function.body->as.block.statements[0]->as.let.value;
        assert(node && (node->type == AST_STRUCT_LITERAL || node->type == AST_UNION_CONSTRUCT));
        Environment *env = create_environment(); assert(env);
        bool ok = type_check_module(program, env);
        assert(ok == (invalid == 0));
        if (ok) {
            assert(node->type == AST_UNION_CONSTRUCT);
            assert(!strcmp(node->as.union_construct.variant_name, "Value"));
            assert(node->as.union_construct.field_count == 1);
        }
        free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
}
static void constructor_payload_destinations(void) {
    const char *forms[] = {
        "fn sample()->int { let box:Box<Item> =Box.Value{item:%s,fixed:Item{value:1}} return 0 }",
        "fn sample()->int { let box =Box<Item>.Value{item:%s,fixed:Item{value:1}} return 0 }",
        "fn sample()->int { let mut box:Box<Item> =Box.Value{item:Item{value:0},fixed:Item{value:1}} set box Box.Value{item:%s,fixed:Item{value:1}} return 0 }",
        "fn sample()->int { return (consume Box.Value{item:%s,fixed:Item{value:1}}) }",
        "fn sample()->Box<Item> { return Box.Value{item:%s,fixed:Item{value:1}} }",
        "fn sample()->int { let holder:Holder =Holder{box:Box.Value{item:%s,fixed:Item{value:1}}} return 0 }"
    };
    for (size_t route = 0; route < sizeof forms / sizeof *forms; ++route)
    for (int wrong = 0; wrong < 3; ++wrong) {
        char function[768], source[1400];
        int n = snprintf(function, sizeof function, forms[route],
            wrong == 1 ? "true" : wrong == 2 ? "Other{value:2}" : "Item{value:2}");
        assert(n > 0 && (size_t)n < sizeof function);
        n = snprintf(source, sizeof source,
            "struct Item{value:int} struct Other{value:int} "
            "union Box<T>{Value{item:T,fixed:Item}} struct Holder{box:Box<Item>} "
            "fn consume(box:Box<Item>)->int{return 0} shadow consume{assert true} "
            "%s shadow sample{assert true}", function);
        assert(n > 0 && (size_t)n < sizeof source);
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        Environment *env = create_environment(); assert(env);
        assert(type_check_module(program, env) == (wrong == 0));
        free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
    const char *fields[] = {"item:Item{value:1},item:Item{value:2}",
        "item:Item{value:1}", "item:Item{value:1},unknown:Item{value:2}",
        "item:Item{value:1},fixed:Other{value:2}"};
    for (size_t i = 0; i < sizeof fields / sizeof *fields; ++i) {
        char source[768];
        int n = snprintf(source, sizeof source,
            "struct Item{value:int} struct Other{value:int} union Box<T>{Value{item:T,fixed:Item}} "
            "fn sample()->int{let box:Box<Item> =Box.Value{%s} return 0} shadow sample{assert true}", fields[i]);
        assert(n > 0 && (size_t)n < sizeof source);
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        Environment *env = create_environment(); assert(env);
        assert(!type_check_module(program, env));
        free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
}
static void union_scalar_policy(void) {
    Type tags[] = {TYPE_INT, TYPE_U8, TYPE_ENUM, TYPE_UNKNOWN, TYPE_BOOL, TYPE_FLOAT, TYPE_STRING, TYPE_FUNCTION};
    for (int metadata = 0; metadata < 2; ++metadata) {
        Environment *env = create_environment(); assert(env);
        for (size_t a = 0; a < sizeof tags / sizeof *tags; ++a) {
            char name[32]; snprintf(name, sizeof name, "scalar_%zu", a);
            TypeInfo info = {.base_type = tags[a]};
            env_define_var_with_type_info(env, name, tags[a], TYPE_UNKNOWN,
                metadata ? &info : NULL, false, create_void());
            ASTNode value = {.type = AST_IDENTIFIER}; value.as.identifier = name;
            for (size_t e = 0; e + 1 < sizeof tags / sizeof *tags; ++e) {
                bool numeric = (tags[a] == TYPE_INT || tags[a] == TYPE_U8 || tags[a] == TYPE_ENUM) &&
                    (tags[e] == TYPE_INT || tags[e] == TYPE_U8 || tags[e] == TYPE_ENUM);
                bool wanted = tags[a] != TYPE_UNKNOWN && tags[e] != TYPE_UNKNOWN && (numeric || tags[a] == tags[e]);
                assert(union_scalar_payload_matches(&value, env, tags[e]) == wanted);
            }
        }
        ASTNode literal = {.type = AST_NUMBER};
        int64_t values[] = {-1, 0, 255, 256};
        for (size_t i = 0; i < sizeof values / sizeof *values; ++i) {
            literal.as.number = values[i];
            assert(union_scalar_payload_matches(&literal, env, TYPE_U8) == (values[i] >= 0 && values[i] <= 255));
            assert(union_scalar_payload_matches(&literal, env, TYPE_INT));
            assert(union_scalar_payload_matches(&literal, env, TYPE_ENUM));
        }
        free_environment(env);
    }
    {
        Environment *env = create_environment(); assert(env);
        EnumDef enumeration = {0}; enumeration.name = strdup("Tag"); enumeration.module_name = "Enums";
        assert(enumeration.name); env_define_enum(env, enumeration);
        identity_record(env, "Tag", "Records");
        for (int owner = 0; owner < 3; ++owner) {
            char name[32]; snprintf(name, sizeof name, "named_scalar_%d", owner);
            env_define_var(env, name, TYPE_STRUCT, false, create_void());
            Symbol *symbol = env_get_var(env, name); assert(symbol);
            symbol->struct_type_name = strdup("Tag"); assert(symbol->struct_type_name);
            symbol->nominal_owner = owner == 0 ? "Enums" : owner == 1 ? "Records" : "Missing";
            ASTNode value = {.type = AST_IDENTIFIER}; value.as.identifier = name;
            assert(union_scalar_payload_matches(&value, env, TYPE_INT) == (owner == 0));
            assert(union_scalar_payload_matches(&value, env, TYPE_U8) == (owner == 0));
            assert(union_scalar_payload_matches(&value, env, TYPE_ENUM) == (owner == 0));
        }
        free_environment(env);
    }
    const char *types[] = {"int", "u8", "Tag", "bool", "float", "string"};
    for (int generic = 0; generic < 2; ++generic)
    for (size_t expected = 0; expected < sizeof types / sizeof *types; ++expected)
    for (size_t actual = 0; actual < sizeof types / sizeof *types; ++actual) {
        char source[768], declaration[128], constructor[64];
        if (generic) {
            snprintf(declaration, sizeof declaration, "union Box<T>{Value{value:T}}");
            snprintf(constructor, sizeof constructor, "Box<%s>", types[expected]);
        } else {
            snprintf(declaration, sizeof declaration, "union Box{Value{value:%s}}", types[expected]);
            snprintf(constructor, sizeof constructor, "Box");
        }
        int n = snprintf(source, sizeof source,
            "enum Tag{One,Two} %s fn sample(value:%s)->int{let box =%s.Value{value:value} return 0} shadow sample{assert true}",
            declaration, types[actual], constructor);
        assert(n > 0 && (size_t)n < sizeof source);
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        Environment *env = create_environment(); assert(env);
        bool wanted = expected == actual || (expected < 3 && actual < 3);
        assert(type_check_module(program, env) == wanted);
        if (!generic) {
            /* I also enter the legacy checker route without the full binder. */
            Type actual_tag = actual == 0 ? TYPE_INT : actual == 1 ? TYPE_U8 : actual == 2 ? TYPE_ENUM
                : actual == 3 ? TYPE_BOOL : actual == 4 ? TYPE_FLOAT : TYPE_STRING;
            env_define_var(env, "payload", actual_tag, false, create_void());
            ASTNode value = {.type = AST_IDENTIFIER}; value.as.identifier = "payload";
            char *names[] = {"value"}; ASTNode *values[] = {&value};
            ASTNode constructor_node = {.type = AST_STRUCT_LITERAL};
            constructor_node.as.struct_literal.struct_name = "Box.Value";
            constructor_node.as.struct_literal.field_count = 1;
            constructor_node.as.struct_literal.field_names = names;
            constructor_node.as.struct_literal.field_values = values;
            g_typecheck_error_count = 0;
            Type checked = check_expression(&constructor_node, env);
            assert((checked == TYPE_UNION && g_typecheck_error_count == 0) == wanted);
        }
        free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
}
static void generic_byte_payload_context(void) {
    const char *values[] = {"(+ value 256)", "255", "256", "true"};
    for (size_t i = 0; i < sizeof values / sizeof *values; ++i) {
        char source[512];
        int n = snprintf(source, sizeof source,
            "union Box<T>{Value{value:T}} "
            "fn sample(value:int)->int{let box =Box<u8>.Value{value:%s} return 0} shadow sample{assert true}", values[i]);
        assert(n > 0 && (size_t)n < sizeof source);
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        Environment *env = create_environment(); assert(env);
        assert(type_check_module(program, env) == (i < 2));
        free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
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
static void emission_entry_rollback(void) {
    for (int entry = 0; entry < 5; ++entry) for (int invalid = 1; invalid >= 0; --invalid) {
        char source[512];
        const char *format = entry < 2
            ? "fn main()->int { let pair:(int,int) =(1,2) return %s } shadow main { assert true }"
            : entry < 4
            ? "fn main()->int { return 0 } shadow main { let pair:(int,int) =(1,2) assert %s }"
            : "fn main()->int { let pair:(int,int) =(1,2) %s }";
        const char *tail = entry < 2 ? (invalid ? "true" : "0")
            : entry < 4 ? (invalid ? "1" : "true") : (invalid ? "missing" : "0");
        int length = snprintf(source, sizeof source, format, tail);
        assert(length > 0 && (size_t)length < sizeof source);
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        Environment *env = create_environment(); assert(env);
        TypeInfo scalar = {.base_type = TYPE_INT}, array = {.base_type = TYPE_ARRAY, .element_type = &scalar};
        Type tags[] = {TYPE_INT}; TypeInfo *children[] = {&scalar};
        TypeInfo tuple = {.base_type = TYPE_TUPLE, .tuple_element_count = 1,
            .tuple_types = tags, .type_param_count = 1, .type_params = children};
        ASTNode prior = {.type = AST_TUPLE_LITERAL}, prior_array = {.type = AST_ARRAY_LITERAL};
        prior.as.tuple_literal.element_count = 1;
        assert(env_bind_tuple_literal(env, &prior, &tuple));
        assert(env_bind_array_expression(env, &prior_array, &array));
        const TypeInfo *saved_tuple = env_tuple_literal_info(env, &prior);
        const TypeInfo *saved_array = env_array_expression_info(env, &prior_array);
        if (entry == 2 || entry == 3) assert(type_check_module(program, env));
        NativeContextMark before = native_context_mark(env);
        bool ok;
        if (entry == 0) ok = type_check(program, env);
        else if (entry == 1) ok = type_check_module(program, env);
        else if (entry == 2) ok = type_check_root_shadows(program, env);
        else if (entry == 3) ok = type_check_shadow_scope(program, env, NULL, "native-cache.nano", false);
        else {
            g_typecheck_error_count = 0;
            assert(program->as.program.count == 1 && program->as.program.items[0]->type == AST_FUNCTION);
            Type result = check_expression(program->as.program.items[0]->as.function.body, env);
            assert(invalid ? g_typecheck_error_count > 0 : g_typecheck_error_count == 0);
            ok = result != TYPE_UNKNOWN && g_typecheck_error_count == 0;
        }
        assert(ok == !invalid);
        assert(env_tuple_literal_info(env, &prior) == saved_tuple && type_infos_equal(saved_tuple, &tuple));
        assert(env_array_expression_info(env, &prior_array) == saved_array && type_infos_equal(saved_array, &array));
        if (invalid) {
            assert(env->tuple_literal_binding_count == before.tuples);
            assert(env->array_expression_binding_count == before.arrays);
        } else assert(env->tuple_literal_binding_count > before.tuples);
        free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
}
extern void test_nominal_constructor_allocations(void);
int main(void) {
    test_nominal_constructor_allocations();
    intrinsic_identity(); parsed_extern_policy(); declaration_identity(); mixed_substitution_identity(); nested_payload_views(); retained_callable_consumers(); complete_tuple_annotations(); constructor_annotation_parsing(); dotted_constructor_checking(); constructor_payload_destinations(); union_scalar_policy(); generic_byte_payload_context(); constructor_failure_rollback(); emission_entry_rollback();
    puts("I checked actual builtin objects and owner-bound array declaration obligations.");
    return 0;
}
