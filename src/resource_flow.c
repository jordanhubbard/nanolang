#include "resource_tracking.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* I keep lexical ownership separate from symbols retained for C emission. */
typedef struct {
    const char *name;
    const char *nominal;
    const TypeInfo *type_info; /* Borrowed from the function AST or retained environment. */
    bool resource;
    bool moved;
    FunctionSignature *signature;
} OwnBinding;

typedef struct OwnLoop {
    OwnBinding *entry;
    size_t count;
    struct OwnLoop *parent;
} OwnLoop;

typedef struct {
    Environment *env;
    OwnBinding *bindings;
    size_t count, capacity;
    size_t exit_first;
    OwnLoop *loop;
    unsigned restricted;
    bool *error;
} OwnFlow;

enum { OWN_NEXT = 1, OWN_RETURN = 2, OWN_BREAK = 4, OWN_CONTINUE = 8 };
static unsigned own_node(OwnFlow *, ASTNode *, bool);
static FunctionSignature *own_signature(OwnFlow *, ASTNode *);
static const TypeInfo *own_expr_info(OwnFlow *, ASTNode *);

static void own_error(OwnFlow *flow, ASTNode *node, const char *message, const char *name) {
    fprintf(stderr, "I cannot verify affine ownership at line %d, column %d: %s%s%s\n",
            node ? node->line : 0, node ? node->column : 0,
            message, name ? " — " : "", name ? name : "");
    *flow->error = true;
}

/* I resolve a selected nongeneric arm independently of its sibling variants. */
static UnionDef *own_variant(Environment *env, const char *name, int *variant) {
    if (!name) return NULL;
    for (int i = 0; i < env->union_count; ++i) {
        UnionDef *def = &env->unions[i];
        size_t length = strlen(def->name);
        if (def->generic_param_count || strncmp(name, def->name, length) || name[length] != '.') continue;
        for (int v = 0; v < def->variant_count; ++v)
            if (!strcmp(name + length + 1, def->variant_names[v])) { *variant = v; return def; }
    }
    return NULL;
}

static bool own_resource(OwnFlow *flow, const char *name) {
    int variant = -1;
    UnionDef *def = own_variant(flow->env, name, &variant);
    if (!def) return is_resource_type(flow->env, name);
    for (int f = 0; f < def->variant_field_counts[variant]; ++f)
        if (def->variant_field_type_names && def->variant_field_type_names[variant] &&
            is_resource_type(flow->env, def->variant_field_type_names[variant][f])) return true;
    return false;
}

/* Complete copying does not establish substitution inside every metadata shape. */
static bool own_unresolved_payload_shape(const TypeInfo *info, unsigned depth) {
    if (!info) return false;
    if (depth > 512) return true;
    if (info->tuple_element_count || info->row_field_count || info->row_var_name || info->fn_sig) return true;
    if (own_unresolved_payload_shape(info->element_type, depth + 1)) return true;
    for (int i = 0; info->type_params && i < info->type_param_count; ++i)
        if (own_unresolved_payload_shape(info->type_params[i], depth + 1)) return true;
    return false;
}

static bool own_info_resource(OwnFlow *flow, ASTNode *at, const TypeInfo *info, unsigned depth) {
    if (!info) return false;
    if (depth > 512) {
        own_error(flow, at, "ownership type metadata exceeds my checked depth", NULL);
        return true;
    }
    UnionDef *def = info->generic_name ? env_get_union(flow->env, info->generic_name) : NULL;
    if (def && def->generic_param_count && info->type_param_count) {
        if (def->generic_param_count != info->type_param_count) {
            own_error(flow, at, "my generic ownership arguments do not match the declaration", info->generic_name);
            return true;
        }
        for (int arm = 0; arm < def->variant_count; ++arm) {
            for (int field = 0; field < def->variant_field_counts[arm]; ++field) {
                const TypeInfo *declared = def->variant_field_type_info && def->variant_field_type_info[arm]
                    ? def->variant_field_type_info[arm][field] : NULL;
                if (own_unresolved_payload_shape(declared, 0)) {
                    /* I preserve the prior resource-argument guard until these
                     * payload shapes have complete formal substitution. */
                    for (int arg = 0; info->type_params && arg < info->type_param_count; ++arg)
                        if (own_info_resource(flow, at, info->type_params[arg], depth + 1)) return true;
                }
                TypeInfo *concrete = resolve_union_payload_type_info(def, arm, field, info);
                if (!concrete) {
                    own_error(flow, at, "I require complete generic payload ownership metadata", info->generic_name);
                    return true;
                }
                bool contained = own_info_resource(flow, at, concrete, depth + 1);
                free_payload_type_info(concrete);
                if (contained) return true;
            }
        }
        return false;
    }
    if (own_resource(flow, info->generic_name)) return true;
    if (own_info_resource(flow, at, info->element_type, depth + 1)) return true;
    for (int i = 0; i < info->type_param_count; ++i)
        if (info->type_params && own_info_resource(flow, at, info->type_params[i], depth + 1)) return true;
    for (int i = 0; i < info->tuple_element_count; ++i)
        if (info->tuple_type_names && own_resource(flow, info->tuple_type_names[i])) return true;
    return false;
}

static void own_metadata(OwnFlow *flow, ASTNode *at, Type type, const char *name, const TypeInfo *info) {
    bool nominal = own_resource(flow, name);
    bool contained = own_info_resource(flow, at, info, 0);
    if (has_resource_collection_payload(flow->env, name)) {
        own_error(flow, at, "resource-bearing union collection payloads are not supported", name);
        return;
    }
    if ((type == TYPE_ARRAY || type == TYPE_LIST_GENERIC || type == TYPE_HASHMAP) && (nominal || contained))
        own_error(flow, at, "I reject resource-bearing collection elements", name);
    else if (contained && !nominal)
        own_error(flow, at, "generic or tuple resource ownership needs lowering", name);
}

static OwnBinding *own_find(OwnFlow *flow, const char *name) {
    if (!name) return NULL;
    for (size_t i = flow->count; i > 0; --i)
        if (!strcmp(flow->bindings[i - 1].name, name)) return &flow->bindings[i - 1];
    return NULL;
}

static bool own_add(OwnFlow *flow, ASTNode *node, const char *name, const char *nominal, const TypeInfo *info) {
    if (!name) return true;
    if (flow->count == flow->capacity) {
        size_t capacity = flow->capacity ? flow->capacity * 2 : 16;
        if (capacity < flow->capacity || capacity > SIZE_MAX / sizeof(*flow->bindings)) {
            own_error(flow, node, "my ownership table is too large", name);
            return false;
        }
        OwnBinding *bindings = realloc(flow->bindings, capacity * sizeof(*bindings));
        if (!bindings) {
            own_error(flow, node, "I cannot allocate ownership state", name);
            return false;
        }
        flow->bindings = bindings;
        flow->capacity = capacity;
    }
    bool resource = own_resource(flow, nominal) || own_info_resource(flow, node, info, 0);
    flow->bindings[flow->count++] = (OwnBinding){.name = name, .nominal = nominal, .type_info = info, .resource = resource};
    if (resource && flow->restricted)
        own_error(flow, node, "this control-flow boundary needs ownership lowering", name);
    return true;
}

static bool own_clone(OwnFlow *copy, OwnFlow *source, ASTNode *at) {
    *copy = *source;
    copy->bindings = NULL;
    copy->capacity = source->count;
    if (!source->count) return true;
    copy->bindings = malloc(source->count * sizeof(*copy->bindings));
    if (!copy->bindings) {
        copy->count = copy->capacity = 0;
        own_error(source, at, "I cannot allocate branch ownership state", NULL);
        return false;
    }
    memcpy(copy->bindings, source->bindings, source->count * sizeof(*copy->bindings));
    return true;
}

static Function *own_function(OwnFlow *flow, ASTNode *node) {
    if (node->type == AST_CALL && node->as.call.name && !node->as.call.func_expr &&
        !own_find(flow, node->as.call.name))
        return env_get_function(flow->env, node->as.call.name);
    if (node->type == AST_MODULE_QUALIFIED_CALL) {
        const char *alias = node->as.module_qualified_call.module_alias;
        const char *name = node->as.module_qualified_call.function_name;
        size_t size = strlen(alias) + strlen(name) + 2;
        char *qualified = malloc(size);
        if (!qualified) { own_error(flow, node, "I cannot resolve call ownership", name); return NULL; }
        snprintf(qualified, size, "%s.%s", alias, name);
        Function *function = env_get_function(flow->env, qualified);
        free(qualified);
        return function;
    }
    return NULL;
}

static FunctionSignature *own_signature(OwnFlow *flow, ASTNode *node) {
    if (!node) return NULL;
    if (node->type == AST_IDENTIFIER) {
        OwnBinding *binding = own_find(flow, node->as.identifier);
        return binding ? binding->signature : NULL;
    }
    Function *function = own_function(flow, node);
    if (function) return function->return_fn_sig;
    if (node->type == AST_CALL) {
        OwnBinding *binding = own_find(flow, node->as.call.name);
        FunctionSignature *callee = node->as.call.func_expr ? own_signature(flow, node->as.call.func_expr) :
            (binding ? binding->signature : NULL);
        return callee ? callee->return_fn_sig : NULL;
    }
    return NULL;
}

static const char *own_type(OwnFlow *flow, ASTNode *node) {
    if (!node) return NULL;
    switch (node->type) {
        case AST_IDENTIFIER: {
            OwnBinding *binding = own_find(flow, node->as.identifier);
            return binding ? binding->nominal : NULL;
        }
        case AST_STRUCT_LITERAL: {
            const char *name = node->as.struct_literal.struct_name;
            /* I retain the union owner of the parser's Choice.Some literal. */
            for (int i = 0; name && i < flow->env->union_count; ++i) {
                const char *owner = flow->env->unions[i].name;
                size_t length = strlen(owner);
                if (!strncmp(name, owner, length) && name[length] == '.') return owner;
            }
            return name;
        }
        case AST_UNION_CONSTRUCT: return node->as.union_construct.union_name;
        case AST_IF: {
            const char *type = own_type(flow, node->as.if_stmt.then_branch);
            return type ? type : own_type(flow, node->as.if_stmt.else_branch);
        }
        case AST_COND:
            return node->as.cond_expr.clause_count ? own_type(flow, node->as.cond_expr.values[0]) : own_type(flow, node->as.cond_expr.else_value);
        case AST_MATCH:
            return node->as.match_expr.arm_count ? own_type(flow, node->as.match_expr.arm_bodies[0]) : NULL;
        case AST_BLOCK:
        case AST_UNSAFE_BLOCK: {
            ASTNode **items = node->type == AST_BLOCK ? node->as.block.statements : node->as.unsafe_block.statements;
            int count = node->type == AST_BLOCK ? node->as.block.count : node->as.unsafe_block.count;
            OwnFlow scope;
            if (!own_clone(&scope, flow, node)) return NULL;
            for (int i = 0; i + 1 < count; ++i) if (items[i]->type == AST_LET) {
                const char *type = items[i]->as.let.type_name ? items[i]->as.let.type_name : own_type(&scope, items[i]->as.let.value);
                if (own_add(&scope, items[i], items[i]->as.let.name, type, items[i]->as.let.type_info ? items[i]->as.let.type_info : own_expr_info(&scope, items[i]->as.let.value)))
                    scope.bindings[scope.count - 1].signature = items[i]->as.let.fn_sig ? items[i]->as.let.fn_sig : own_signature(&scope, items[i]->as.let.value);
            }
            const char *type = count ? own_type(&scope, items[count - 1]) : NULL;
            free(scope.bindings);
            return type;
        }
        case AST_CALL:
        case AST_MODULE_QUALIFIED_CALL: {
            Function *function = own_function(flow, node);
            if (function) return function->return_struct_type_name;
            if (node->type == AST_CALL) {
                OwnBinding *binding = own_find(flow, node->as.call.name);
                FunctionSignature *signature = node->as.call.func_expr ? own_signature(flow, node->as.call.func_expr) :
                    (binding ? binding->signature : NULL);
                if (signature) return signature->return_struct_name;
            }
            return node->type == AST_CALL ? node->as.call.return_struct_type_name :
                node->as.module_qualified_call.return_struct_type_name;
        }
        case AST_FIELD_ACCESS: {
            const char *name = own_type(flow, node->as.field_access.object);
            StructDef *record = name ? env_get_struct(flow->env, name) : NULL;
            for (int i = 0; record && i < record->field_count; ++i)
                if (!strcmp(record->field_names[i], node->as.field_access.field_name))
                    return record->field_type_names ? record->field_type_names[i] : NULL;
            int variant = -1;
            UnionDef *def = own_variant(flow->env, name, &variant);
            for (int i = 0; def && i < def->variant_field_counts[variant]; ++i)
                if (!strcmp(def->variant_field_names[variant][i], node->as.field_access.field_name))
                    return def->variant_field_type_names && def->variant_field_type_names[variant] ?
                        def->variant_field_type_names[variant][i] : NULL;
            return NULL;
        }
        default: return NULL;
    }
}

/* I borrow complete immutable trees whose owners outlive every branch clone.
 * Substituted temporary payload trees are consumed and freed only in classification. */
static const TypeInfo *own_expr_info(OwnFlow *flow, ASTNode *node) {
    if (!node) return NULL;
    switch (node->type) {
        case AST_IDENTIFIER: {
            OwnBinding *binding = own_find(flow, node->as.identifier);
            return binding ? binding->type_info : NULL;
        }
        case AST_LET: return node->as.let.type_info;
        case AST_UNION_CONSTRUCT: return node->as.union_construct.type_info;
        case AST_FIELD_ACCESS: return node->as.field_access.resolved_type_info;
        case AST_CALL:
        case AST_MODULE_QUALIFIED_CALL: {
            Function *function = own_function(flow, node);
            return function ? function->return_type_info : NULL;
        }
        case AST_IF: {
            const TypeInfo *info = own_expr_info(flow, node->as.if_stmt.then_branch);
            return info ? info : own_expr_info(flow, node->as.if_stmt.else_branch);
        }
        case AST_COND:
            return node->as.cond_expr.clause_count ? own_expr_info(flow, node->as.cond_expr.values[0]) : own_expr_info(flow, node->as.cond_expr.else_value);
        case AST_MATCH:
            return node->as.match_expr.arm_count ? own_expr_info(flow, node->as.match_expr.arm_bodies[0]) : NULL;
        case AST_BLOCK:
        case AST_UNSAFE_BLOCK: {
            ASTNode **items = node->type == AST_BLOCK ? node->as.block.statements : node->as.unsafe_block.statements;
            int count = node->type == AST_BLOCK ? node->as.block.count : node->as.unsafe_block.count;
            OwnFlow scope;
            if (!own_clone(&scope, flow, node)) return NULL;
            for (int i = 0; i + 1 < count; ++i) if (items[i]->type == AST_LET) {
                const TypeInfo *info = items[i]->as.let.type_info ? items[i]->as.let.type_info : own_expr_info(&scope, items[i]->as.let.value);
                own_add(&scope, items[i], items[i]->as.let.name, items[i]->as.let.type_name, info);
            }
            const TypeInfo *info = count ? own_expr_info(&scope, items[count - 1]) : NULL;
            free(scope.bindings);
            return info;
        }
        default: return NULL;
    }
}

static void own_leaks(OwnFlow *flow, size_t first, ASTNode *at) {
    for (size_t i = first; i < flow->count; ++i)
        if (flow->bindings[i].resource && !flow->bindings[i].moved)
            own_error(flow, at, "resource remains live at scope exit", flow->bindings[i].name);
}

static void own_loop_edge(OwnFlow *flow, ASTNode *at) {
    if (!flow->loop) return;
    own_leaks(flow, flow->loop->count, at);
    for (size_t i = 0; i < flow->loop->count; ++i)
        if (flow->bindings[i].resource && flow->bindings[i].moved != flow->loop->entry[i].moved)
            own_error(flow, at, "loop changes ownership of an outer resource", flow->bindings[i].name);
}

static unsigned own_sequence(OwnFlow *flow, ASTNode **items, int count, ASTNode *at, bool value) {
    size_t first = flow->count;
    unsigned result = OWN_NEXT;
    for (int i = 0; i < count && (result & OWN_NEXT); ++i)
        result = (result & ~OWN_NEXT) | own_node(flow, items[i], value && i == count - 1);
    if (result & OWN_NEXT) own_leaks(flow, first, at);
    flow->count = first;
    return result;
}

static unsigned own_branches(OwnFlow *flow, ASTNode *left, ASTNode *right, ASTNode *at, bool value) {
    OwnFlow a, b;
    if (!own_clone(&a, flow, at)) return OWN_NEXT;
    if (!own_clone(&b, flow, at)) { free(a.bindings); return OWN_NEXT; }
    unsigned ar = own_node(&a, left, value), br = own_node(&b, right, value);
    for (size_t i = 0; i < flow->count; ++i) {
        if ((ar & OWN_NEXT) && (br & OWN_NEXT) && a.bindings[i].moved != b.bindings[i].moved)
            own_error(flow, at, "branches disagree on resource ownership", flow->bindings[i].name);
        if (ar & OWN_NEXT) flow->bindings[i].moved = a.bindings[i].moved;
        else if (br & OWN_NEXT) flow->bindings[i].moved = b.bindings[i].moved;
    }
    free(a.bindings);
    free(b.bindings);
    return ar | br;
}

static void own_nested_function(OwnFlow *flow, ASTNode *function) {
    if (!function || function->type != AST_FUNCTION || function->as.function.is_extern) return;
    OwnFlow nested;
    if (!own_clone(&nested, flow, function)) return;
    nested.exit_first = nested.count;
    nested.loop = NULL;
    for (int i = 0; i < function->as.function.param_count; ++i) {
        Parameter *parameter = &function->as.function.params[i];
        if (own_add(&nested, function, parameter->name, parameter->struct_type_name, parameter->type_info))
            nested.bindings[nested.count - 1].signature = parameter->fn_sig;
    }
    unsigned result = own_node(&nested, function->as.function.body, false);
    if (result & OWN_NEXT) own_leaks(&nested, nested.exit_first, function);
    free(nested.bindings);
}

static unsigned own_node(OwnFlow *flow, ASTNode *node, bool move) {
    if (!node) return OWN_NEXT;
    if (node->lambda_definition) own_nested_function(flow, node->lambda_definition);
    const char *nominal = own_type(flow, node);
    bool resource = own_resource(flow, nominal) || own_info_resource(flow, node, own_expr_info(flow, node), 0);
    switch (node->type) {
        case AST_IDENTIFIER: {
            OwnBinding *binding = own_find(flow, node->as.identifier);
            if (binding && binding->resource) {
                if ((size_t)(binding - flow->bindings) < flow->exit_first)
                    own_error(flow, node, "resource captures need ownership lowering", binding->name);
                if (flow->restricted) own_error(flow, node, "this boundary needs ownership lowering", binding->name);
                if (binding->moved) own_error(flow, node, "I cannot use a moved value", binding->name);
                else if (move) binding->moved = true;
            }
            return OWN_NEXT;
        }
        case AST_FIELD_ACCESS: {
            unsigned result = own_node(flow, node->as.field_access.object, false);
            if (move && resource) own_error(flow, node, "I cannot partially move a resource field", node->as.field_access.field_name);
            return result;
        }
        case AST_LET: {
            const char *type = node->as.let.type_name ? node->as.let.type_name : own_type(flow, node->as.let.value);
            const TypeInfo *info = node->as.let.type_info ? node->as.let.type_info : own_expr_info(flow, node->as.let.value);
            own_metadata(flow, node, node->as.let.var_type, type, info);
            unsigned result = node->as.let.is_destructure_projection ? OWN_NEXT : own_node(flow, node->as.let.value, true);
            FunctionSignature *signature = node->as.let.fn_sig ? node->as.let.fn_sig : own_signature(flow, node->as.let.value);
            if ((result & OWN_NEXT) && own_add(flow, node, node->as.let.name, type, info)) {
                flow->bindings[flow->count - 1].signature = signature;
                if (node->as.let.is_destructure) flow->bindings[flow->count - 1].moved = true;
            }
            return result;
        }
        case AST_SET: {
            OwnBinding *binding = own_find(flow, node->as.set.name);
            size_t index = binding ? (size_t)(binding - flow->bindings) : SIZE_MAX;
            if (binding && binding->resource && !binding->moved)
                own_error(flow, node, "I cannot overwrite a live resource", binding->name);
            if ((!binding || !binding->resource) && own_resource(flow, own_type(flow, node->as.set.value)))
                own_error(flow, node, "resource assignment needs an owned destination", node->as.set.name);
            unsigned result = own_node(flow, node->as.set.value, true);
            if (index != SIZE_MAX && (result & OWN_NEXT)) flow->bindings[index].moved = false;
            return result;
        }
        case AST_RETURN: {
            unsigned result = own_node(flow, node->as.return_stmt.value, true);
            if (result & OWN_NEXT) own_leaks(flow, flow->exit_first, node);
            return (result & ~OWN_NEXT) | OWN_RETURN;
        }
        case AST_BLOCK:
            return own_sequence(flow, node->as.block.statements, node->as.block.count, node, move);
        case AST_UNSAFE_BLOCK:
            return own_sequence(flow, node->as.unsafe_block.statements, node->as.unsafe_block.count, node, move);
        case AST_IF: {
            unsigned result = own_node(flow, node->as.if_stmt.condition, false);
            if (!(result & OWN_NEXT)) return result;
            return (result & ~OWN_NEXT) | own_branches(flow, node->as.if_stmt.then_branch, node->as.if_stmt.else_branch, node, move);
        }
        case AST_WHILE:
        case AST_FOR: {
            ASTNode *condition = node->type == AST_WHILE ? node->as.while_stmt.condition : node->as.for_stmt.range_expr;
            ASTNode *body = node->type == AST_WHILE ? node->as.while_stmt.body : node->as.for_stmt.body;
            OwnFlow entry;
            if (!own_clone(&entry, flow, node)) return OWN_NEXT;
            OwnLoop loop = {entry.bindings, entry.count, flow->loop};
            flow->loop = &loop;
            unsigned result = own_node(flow, condition, false);
            if (!(result & OWN_NEXT)) {
                flow->loop = loop.parent;
                free(entry.bindings);
                return result;
            }
            own_loop_edge(flow, node);
            size_t first = flow->count;
            if (node->type == AST_FOR) own_add(flow, node, node->as.for_stmt.var_name, NULL, NULL);
            if (result & OWN_NEXT) result = (result & ~OWN_NEXT) | own_node(flow, body, false);
            if (result & OWN_NEXT) own_loop_edge(flow, node);
            flow->count = first;
            for (size_t i = 0; i < entry.count; ++i) flow->bindings[i].moved = entry.bindings[i].moved;
            flow->loop = loop.parent;
            free(entry.bindings);
            return OWN_NEXT | (result & OWN_RETURN);
        }
        case AST_BREAK:
        case AST_CONTINUE:
            own_loop_edge(flow, node);
            return node->type == AST_BREAK ? OWN_BREAK : OWN_CONTINUE;
        case AST_CALL:
        case AST_MODULE_QUALIFIED_CALL: {
            ASTNode **args = node->type == AST_CALL ? node->as.call.args : node->as.module_qualified_call.args;
            int count = node->type == AST_CALL ? node->as.call.arg_count : node->as.module_qualified_call.arg_count;
            unsigned result = node->type == AST_CALL ? own_node(flow, node->as.call.func_expr, false) : OWN_NEXT;
            for (int i = 0; i < count && (result & OWN_NEXT); ++i)
                result = (result & ~OWN_NEXT) | own_node(flow, args[i], true);
            if ((result & OWN_NEXT) && resource && (!move || flow->restricted))
                own_error(flow, node, "resource result has no resolved owner", nominal);
            return result;
        }
        case AST_STRUCT_LITERAL:
        case AST_UNION_CONSTRUCT: {
            ASTNode **values = node->type == AST_STRUCT_LITERAL ? node->as.struct_literal.field_values : node->as.union_construct.field_values;
            int count = node->type == AST_STRUCT_LITERAL ? node->as.struct_literal.field_count : node->as.union_construct.field_count;
            unsigned result = OWN_NEXT;
            if (node->type == AST_STRUCT_LITERAL) result = own_node(flow, node->as.struct_literal.spread_source, true);
            for (int i = 0; i < count && (result & OWN_NEXT); ++i)
                result = (result & ~OWN_NEXT) | own_node(flow, values[i], true);
            if ((result & OWN_NEXT) && resource && (!move || flow->restricted))
                own_error(flow, node, "resource construction has no resolved owner", nominal);
            return result;
        }
        case AST_ARRAY_LITERAL:
        case AST_TUPLE_LITERAL: {
            ASTNode **items = node->type == AST_ARRAY_LITERAL ? node->as.array_literal.elements : node->as.tuple_literal.elements;
            int count = node->type == AST_ARRAY_LITERAL ? node->as.array_literal.element_count : node->as.tuple_literal.element_count;
            unsigned result = OWN_NEXT;
            for (int i = 0; i < count && (result & OWN_NEXT); ++i) {
                if (own_resource(flow, own_type(flow, items[i])))
                    own_error(flow, items[i], node->type == AST_ARRAY_LITERAL ? "I reject resource array elements" : "resource tuple ownership needs lowering", NULL);
                result = (result & ~OWN_NEXT) | own_node(flow, items[i], true);
            }
            return result;
        }
        case AST_PREFIX_OP: {
            unsigned result = OWN_NEXT;
            if ((node->as.prefix_op.op == TOKEN_AND || node->as.prefix_op.op == TOKEN_OR) &&
                node->as.prefix_op.arg_count == 2) {
                result = own_node(flow, node->as.prefix_op.args[0], false);
                if (!(result & OWN_NEXT)) return result;
                return (result & ~OWN_NEXT) | own_branches(flow, node->as.prefix_op.args[1], NULL, node, false);
            }
            for (int i = 0; i < node->as.prefix_op.arg_count && (result & OWN_NEXT); ++i)
                result = (result & ~OWN_NEXT) | own_node(flow, node->as.prefix_op.args[i], false);
            return result;
        }
        case AST_PRINT: return own_node(flow, node->as.print.expr, false);
        case AST_ASSERT: return own_node(flow, node->as.assert.condition, false);
        case AST_TUPLE_INDEX: return own_node(flow, node->as.tuple_index.tuple, false);
        case AST_AWAIT: return own_node(flow, node->as.await_expr.expr, move);
        case AST_TRY_OP:
            own_leaks(flow, 0, node);
            return own_node(flow, node->as.try_op.operand, move);
        case AST_PAR_BLOCK: {
            unsigned result = OWN_NEXT;
            for (int i = 0; i < node->as.par_block.count && (result & OWN_NEXT); ++i)
                result = (result & ~OWN_NEXT) | own_node(flow, node->as.par_block.bindings[i], false);
            return result;
        }
        case AST_PAR_LET: {
            size_t first = flow->count;
            unsigned result = OWN_NEXT;
            for (int i = 0; i < node->as.par_let.count && (result & OWN_NEXT); ++i) {
                const char *type = own_type(flow, node->as.par_let.values[i]);
                result = (result & ~OWN_NEXT) | own_node(flow, node->as.par_let.values[i], true);
                if (result & OWN_NEXT) own_add(flow, node, node->as.par_let.names[i], type, own_expr_info(flow, node->as.par_let.values[i]));
            }
            if (result & OWN_NEXT) result = (result & ~OWN_NEXT) | own_node(flow, node->as.par_let.body, move);
            if (result & OWN_NEXT) own_leaks(flow, first, node);
            flow->count = first;
            return result;
        }
        case AST_COND: {
            if (!node->as.cond_expr.clause_count) return own_node(flow, node->as.cond_expr.else_value, move);
            ASTNode rest = *node;
            rest.as.cond_expr.conditions++;
            rest.as.cond_expr.values++;
            rest.as.cond_expr.clause_count--;
            unsigned result = own_node(flow, node->as.cond_expr.conditions[0], false);
            if (!(result & OWN_NEXT)) return result;
            return (result & ~OWN_NEXT) | own_branches(flow, node->as.cond_expr.values[0], &rest, node, move);
        }
        case AST_MATCH: {
            const char *input = own_type(flow, node->as.match_expr.expr);
            bool owned_match = own_resource(flow, input) || own_info_resource(flow, node, own_expr_info(flow, node->as.match_expr.expr), 0);
            UnionDef *def = input ? env_get_union(flow->env, input) : NULL;
            bool supported = def && !def->generic_param_count &&
                !has_resource_collection_payload(flow->env, input) &&
                node->as.match_expr.arm_count == def->variant_count;
            for (int arm = 0; supported && arm < node->as.match_expr.arm_count; ++arm) {
                const char *variant = node->as.match_expr.pattern_variants[arm];
                bool found = false;
                for (int v = 0; v < def->variant_count; ++v)
                    if (!strcmp(variant, def->variant_names[v])) found = true;
                for (int prior = 0; prior < arm; ++prior)
                    if (!strcmp(variant, node->as.match_expr.pattern_variants[prior])) found = false;
                if (!found || (node->as.match_expr.guard_exprs && node->as.match_expr.guard_exprs[arm])) supported = false;
            }
            if (owned_match && !supported)
                own_error(flow, node, "I require an exhaustive unguarded nongeneric owned match", input);
            unsigned result = own_node(flow, node->as.match_expr.expr, owned_match && supported);
            if (!(result & OWN_NEXT)) return result;
            OwnFlow joined;
            if (!own_clone(&joined, flow, node)) return result;
            bool has_next = false;
            unsigned exits = result & ~OWN_NEXT;
            for (int arm = 0; arm < node->as.match_expr.arm_count; ++arm) {
                OwnFlow branch;
                if (!own_clone(&branch, flow, node)) break;
                size_t first = branch.count;
                char *payload_type = NULL;
                if (supported) {
                    const char *variant = node->as.match_expr.pattern_variants[arm];
                    size_t size = strlen(def->name) + strlen(variant) + 2;
                    payload_type = malloc(size);
                    if (!payload_type) own_error(flow, node, "I cannot allocate selected ownership identity", input);
                    else snprintf(payload_type, size, "%s.%s", def->name, variant);
                }
                const char *binding = node->as.match_expr.pattern_bindings[arm];
                if (payload_type && own_resource(&branch, payload_type) &&
                    (!binding || !strcmp(binding, "_")))
                    own_error(flow, node, "I require a binding for the selected resource payload", payload_type);
                own_add(&branch, node, binding, payload_type, NULL);
                if (owned_match && !supported) branch.restricted++;
                ASTNode *guard = node->as.match_expr.guard_exprs ? node->as.match_expr.guard_exprs[arm] : NULL;
                unsigned branch_result = own_node(&branch, guard, false);
                /* A false guard continues with the same scrutinee and owners. */
                if (guard) for (size_t i = 0; i < first; ++i)
                    if (branch.bindings[i].resource && branch.bindings[i].moved != flow->bindings[i].moved)
                        own_error(flow, guard, "a match guard changes ownership", branch.bindings[i].name);
                if (branch_result & OWN_NEXT)
                    branch_result = (branch_result & ~OWN_NEXT) | own_node(&branch, node->as.match_expr.arm_bodies[arm], move);
                exits |= branch_result & ~OWN_NEXT;
                if (branch_result & OWN_NEXT) {
                    own_leaks(&branch, first, node);
                    for (size_t i = 0; i < first; ++i) {
                        if (has_next && branch.bindings[i].moved != joined.bindings[i].moved)
                            own_error(flow, node, "match arms disagree on ownership", branch.bindings[i].name);
                        joined.bindings[i].moved = branch.bindings[i].moved;
                    }
                    has_next = true;
                }
                free(branch.bindings);
                free(payload_type);
            }
            if (has_next) for (size_t i = 0; i < flow->count; ++i) flow->bindings[i].moved = joined.bindings[i].moved;
            free(joined.bindings);
            return exits | (has_next ? OWN_NEXT : 0);
        }
        case AST_HANDLE_EXPR:
        case AST_EFFECT_HANDLER: {
            /* I visit each region, but reject ownership across nonlocal control. */
            own_leaks(flow, 0, node);
            ASTNode *body = node->type == AST_HANDLE_EXPR ? node->as.handle_expr.body : node->as.effect_handler.body;
            ASTNode **handlers = node->type == AST_HANDLE_EXPR ? node->as.handle_expr.handler_bodies : node->as.effect_handler.handler_bodies;
            int count = node->type == AST_HANDLE_EXPR ? node->as.handle_expr.handler_count : node->as.effect_handler.handler_count;
            for (int i = -1; i < count; ++i) {
                OwnFlow branch;
                if (!own_clone(&branch, flow, node)) break;
                branch.restricted++;
                own_node(&branch, i < 0 ? body : handlers[i], move);
                free(branch.bindings);
            }
            return OWN_NEXT;
        }
        case AST_EFFECT_OP: {
            own_leaks(flow, 0, node);
            flow->restricted++;
            for (int i = 0; i < node->as.effect_op.arg_count; ++i) own_node(flow, node->as.effect_op.args[i], true);
            flow->restricted--;
            return OWN_NEXT;
        }
        case AST_NUMBER: case AST_FLOAT: case AST_STRING: case AST_BOOL:
        case AST_QUALIFIED_NAME: case AST_STRUCT_DEF: case AST_UNION_DEF:
        case AST_ENUM_DEF: case AST_IMPORT: case AST_MODULE_DECL:
        case AST_OPAQUE_TYPE: case AST_EFFECT_DECL:
            return OWN_NEXT;
        case AST_FUNCTION:
            own_nested_function(flow, node);
            return OWN_NEXT;
        case AST_ASYNC_FN:
            own_nested_function(flow, node->as.async_fn.function);
            return OWN_NEXT;
        case AST_SHADOW: case AST_PROGRAM:
            return OWN_NEXT;
    }
    own_error(flow, node, "I do not recognize this ownership expression", NULL);
    return OWN_NEXT;
}

void check_function_ownership(Environment *env, ASTNode *function, bool *has_error) {
    if (!function || function->type != AST_FUNCTION) return;
    bool any_resource = false;
    for (int i = 0; i < env->struct_count; ++i) any_resource |= env->structs[i].is_resource;
    if (!any_resource) return;
    OwnFlow flow = {.env = env, .error = has_error};
    own_metadata(&flow, function, function->as.function.return_type,
                 function->as.function.return_struct_type_name, function->as.function.return_type_info);
    for (int i = 0; i < function->as.function.param_count; ++i) {
        Parameter *parameter = &function->as.function.params[i];
        own_metadata(&flow, function, parameter->type, parameter->struct_type_name, parameter->type_info);
        if (!function->as.function.is_extern && own_add(&flow, function, parameter->name, parameter->struct_type_name, parameter->type_info))
            flow.bindings[flow.count - 1].signature = parameter->fn_sig;
    }
    if (function->as.function.is_extern) return;
    unsigned result = own_node(&flow, function->as.function.body, false);
    if (result & OWN_NEXT) own_leaks(&flow, 0, function);
    free(flow.bindings);
}
