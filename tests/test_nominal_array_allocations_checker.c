/* I compile the actual checker under a separate allocation domain. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/nanolang.h"
#include <stdlib.h>
#include <string.h>
void *array_alloc_malloc(size_t);
void *array_alloc_calloc(size_t, size_t);
void *array_alloc_realloc(void *, size_t);
char *array_alloc_strdup(const char *);
void array_alloc_free(void *);
#define malloc array_alloc_malloc
#define calloc array_alloc_calloc
#define realloc array_alloc_realloc
#define strdup array_alloc_strdup
#define free array_alloc_free
#include "../src/typechecker.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

bool array_test_view(Environment *env, ASTNode *expression, unsigned depth,
                     TypeInfo **info, const char **owner) {
    NominalView view = {.info = *info, .owner = *owner};
    if (!nominal_value_view(expression, env, depth, &view)) {
        /* I check the actual private output, not only my wrapper's outputs. */
        if (view.info != *info || view.owner != *owner) abort();
        return false;
    }
    /* This wrapper's original vectors have only root-module scalar leaves. */
    if (view.owner || view.owned_context || view.payload) abort();
    *info = view.info; *owner = NULL;
    view.info = NULL;
    nominal_view_discard(&view);
    return true;
}

/* I exercise owned context copying, materialization and both registrations. */
bool array_test_owned_context(Environment *env, Symbol *output) {
    char formal_name[] = "T", record_name[] = "Item", owner[] = "Caller";
    char *formals[] = {formal_name};
    UnionDef declaration = {.generic_param_count = 1, .generic_params = formals};
    TypeInfo leaf = {.base_type = TYPE_STRUCT, .generic_name = record_name};
    TypeInfo *arguments[] = {&leaf};
    TypeInfo instance = {.type_param_count = 1, .type_params = arguments};
    NominalSubstitution context = {&declaration, &instance, owner, NULL};
    TypeInfo compact = {.base_type = TYPE_LIST_GENERIC, .generic_name = formal_name};
    TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = &compact};
    NominalView view = {0};
    if (!nominal_view_copy_context(env, &array, "Definitions", &context, 0, &view)) return false;
    formal_name[0] = 'X'; record_name[0] = 'X'; owner[0] = 'X';
    bool ok = nominal_view_retain(env, output, &view);
    nominal_view_discard(&view);
    if (ok) {
        const NominalView *retained = output->checker_nominal_view;
        if (!retained || !retained->owned_context || strcmp(retained->owner, "Definitions") ||
            strcmp(retained->owned_context->context.argument_owner, "Caller") ||
            strcmp(output->type_info->element_type->type_params[0]->generic_name, "Item")) abort();
    }
    return ok;
}

bool array_test_prepare_callable(Environment *env, bool legacy_return) {
    TypeInfo fixed = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
    TypeInfo formal = {.base_type = TYPE_STRUCT, .generic_name = "T"};
    TypeInfo list = {.base_type = TYPE_LIST_GENERIC, .generic_name = "T"};
    Type tuple_types[] = {TYPE_STRUCT, TYPE_LIST_GENERIC}; char *tuple_names[] = {"Item", "T"};
    TypeInfo tuple = {.base_type = TYPE_TUPLE, .tuple_element_count = 2,
        .tuple_types = tuple_types, .tuple_type_names = tuple_names};
    Type nested_tags[] = {TYPE_STRUCT}; TypeInfo *nested_params[] = {&formal}; char *nested_names[] = {"T"};
    FunctionSignature nested = {.param_count = 1, .param_types = nested_tags,
        .param_type_info = nested_params, .param_struct_names = nested_names,
        .return_type = TYPE_STRUCT, .return_type_info = &fixed, .return_struct_name = "Item"};
    TypeInfo callback = {.base_type = TYPE_FUNCTION, .fn_sig = &nested};
    Type tags[] = {TYPE_TUPLE, TYPE_LIST_GENERIC, TYPE_FUNCTION};
    TypeInfo *parameters[] = {&tuple, &list, &callback}; char *names[] = {NULL, "T", NULL};
    TypeInfo legacy = {.base_type = TYPE_FUNCTION};
    FunctionSignature signature = {.param_count = 3, .param_types = tags, .param_type_info = parameters,
        .param_struct_names = names, .return_type = TYPE_FUNCTION,
        .return_type_info = legacy_return ? &legacy : &callback, .return_fn_sig = &nested};
    TypeInfo callable = {.base_type = TYPE_FUNCTION, .fn_sig = &signature};
    char *formals[] = {"T"}; UnionDef declaration = {.generic_param_count = 1, .generic_params = formals};
    TypeInfo *arguments[] = {&fixed}; TypeInfo instance = {.type_param_count = 1, .type_params = arguments};
    NominalSubstitution context = {&declaration, &instance, "Caller", NULL};
    NominalView view = {0};
    if (!nominal_view_copy_context(env, &callable, "Definitions", &context, 0, &view)) return false;
    Symbol *binding = env_get_var(env, "callback");
    bool ok = nominal_view_retain(env, binding, &view);
    nominal_view_discard(&view);
    return ok;
}
bool array_test_callable_consumer(Environment *env, TypeInfo **output) {
    ASTNode identifier = {.type = AST_IDENTIFIER}; identifier.as.identifier = "callback";
    ASTNode call = {.type = AST_CALL}; call.as.call.name = "callback";
    NominalView value = {0}, callee = {0}, result = {0};
    TypeInfo *concrete = NULL;
    bool ok = nominal_callable_view(&identifier, env, 0, &value) &&
        nominal_callee_view(&call, env, 0, &callee) && nominal_callable_result(env, &callee, 0, &result) &&
        checked_annotations_equal_context(env, value.info, value.owner, callee.info, callee.owner,
            0, nominal_view_context(&value), nominal_view_context(&callee)) &&
        nominal_materialize(env, value.info, value.owner, nominal_view_context(&value), 0, &concrete);
    if (ok) {
        if (!result.owned_context || result.info->base_type != TYPE_FUNCTION ||
            strcmp(concrete->fn_sig->param_type_info[0]->tuple_type_names[1], "Item") ||
            concrete->fn_sig->param_type_info[1]->base_type != TYPE_LIST_GENERIC ||
            strcmp(concrete->fn_sig->param_type_info[1]->type_params[0]->generic_name, "Item") ||
            strcmp(concrete->fn_sig->return_fn_sig->param_type_info[0]->generic_name, "Item")) abort();
        *output = concrete;
    } else free_payload_type_info(concrete);
    nominal_view_discard(&value); nominal_view_discard(&callee); nominal_view_discard(&result);
    return ok;
}

/* I build a tuple through actual expression consumers, not a fabricated proof. */
bool array_test_prepare_tuple_leaves(Environment *env) {
    TypeInfo item = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
    const char *names[] = {"fixed", "actual"};
    const char *owners[] = {"Definitions", "Caller"};
    for (size_t i = 0; i < 2; ++i) {
        env_define_var(env, names[i], TYPE_STRUCT, false, create_void());
        NominalView leaf = {0};
        if (!nominal_view_copy_context(env, &item, owners[i], NULL, 0, &leaf)) return false;
        bool ok = nominal_view_retain(env, env_get_var(env, names[i]), &leaf);
        nominal_view_discard(&leaf);
        if (!ok) return false;
    }
    return true;
}
bool array_test_tuple_consumer(Environment *env, Symbol *publication, TypeInfo **output) {
    ASTNode fixed = {.type = AST_IDENTIFIER}, actual = {.type = AST_IDENTIFIER};
    ASTNode callback = {.type = AST_IDENTIFIER}, tuple = {.type = AST_TUPLE_LITERAL};
    fixed.as.identifier = "fixed"; actual.as.identifier = "actual"; callback.as.identifier = "callback";
    ASTNode *elements[] = {&fixed, &actual, &callback};
    tuple.as.tuple_literal.elements = elements; tuple.as.tuple_literal.element_count = 3;
    NominalView source = {0}, copy = {0}, projected = {0};
    TypeInfo *concrete = NULL;
    bool ok = nominal_value_view(&tuple, env, 0, &source);
    if (ok && publication) ok = nominal_view_retain(env, publication, &source);
    else if (ok) {
        ok = nominal_view_clone(env, &source, 0, &copy) && nominal_view_wrap_array(&copy) &&
            nominal_view_element(env, &copy, 0) && nominal_view_child(env, &copy, 2, 0, &projected) &&
            nominal_view_equal(env, &source, &copy, 0) && nominal_view_materialize(env, &copy, 0, &concrete);
        if (ok) {
            TypeInfo flat;
            const TypeInfo *child = type_info_tuple_element(concrete, 2, &flat);
            if (!projected.owned_context || !child || child->base_type != TYPE_FUNCTION ||
                strcmp(child->fn_sig->param_type_info[1]->type_params[0]->generic_name, "Item") ||
                !nominal_equal(nominal_view_identity(env, &copy.children[0], TYPE_STRUCT),
                    env_nominal_identity(env, "Item", "Definitions", TYPE_STRUCT)) ||
                !nominal_equal(nominal_view_identity(env, &copy.children[1], TYPE_STRUCT),
                    env_nominal_identity(env, "Item", "Caller", TYPE_STRUCT))) abort();
            *output = concrete;
        }
    }
    if (!ok) free_payload_type_info(concrete);
    nominal_view_discard(&projected); nominal_view_discard(&copy); nominal_view_discard(&source);
    return ok;
}

/* The AST key remains borrowed; failed publication must leave the list intact. */
bool array_test_constructor_registry(Environment *env, ASTNode *expression) {
    TypeInfo argument = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
    TypeInfo *arguments[] = {&argument};
    TypeInfo instance = {.base_type = TYPE_UNION, .generic_name = "Box",
        .type_param_count = 1, .type_params = arguments};
    return nominal_constructor_retain(env, expression, &instance, "Caller", NULL, 0);
}
