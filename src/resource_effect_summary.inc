/* I compute escaping operations to a fixed point over direct calls. A local
 * handler contains only the operations it actually handles. Unknown callable
 * targets remain potentially effectful while an owner is suspended. */
typedef struct OwnEffectScope {
    const struct OwnEffectScope *parent;
    ASTNode **items;
    int count;
    Parameter *params;
    int param_count;
    const char *binding;
    const OwnBinding *bindings;
    size_t binding_count;
    char **names;
    int name_count;
} OwnEffectScope;

static bool own_effect_bound(const OwnEffectScope *scope, const char *name) {
    if (!name) return true;
    for (; scope; scope = scope->parent) {
        if (scope->binding && !strcmp(scope->binding, name)) return true;
        for (size_t i = 0; i < scope->binding_count; ++i)
            if (!strcmp(scope->bindings[i].name, name)) return true;
        for (int i = 0; i < scope->name_count; ++i)
            if (!strcmp(scope->names[i], name)) return true;
        for (int i = 0; i < scope->param_count; ++i)
            if (!strcmp(scope->params[i].name, name)) return true;
        for (int i = 0; i < scope->count; ++i) {
            ASTNode *item = scope->items[i];
            if (item->type == AST_LET && !strcmp(item->as.let.name, name)) return true;
            if (item->type == AST_FUNCTION && !strcmp(item->as.function.name, name)) return true;
        }
    }
    return false;
}

static EffectDef *own_effect_handler_definition(Environment *env, ASTNode *node) {
    if (node->as.handle_expr.effect_name)
        return env_get_effect(env, node->as.handle_expr.effect_name);
    /* I can inspect a later function before its handler has been annotated. */
    EffectDef *selected = NULL;
    for (int i = 0; i < env->effect_count; ++i) {
        EffectDef *candidate = &env->effects[i];
        if (!env_effect_visible(env, candidate)) continue;
        bool matches = true;
        for (int arm = 0; arm < node->as.handle_expr.handler_count; ++arm) {
            EffectOp *op = effect_get_op(candidate, node->as.handle_expr.handler_op_names[arm]);
            if (!op || op->param_count != node->as.handle_expr.handler_param_counts[arm]) matches = false;
        }
        if (matches) {
            if (selected) return NULL;
            selected = candidate;
        }
    }
    return selected;
}

static bool own_effect_escape_node(Environment *env, ASTNode *node, const char *effect,
                                   const char *operation, const bool *facts,
                                   const OwnEffectScope *scope, unsigned depth) {
    if (!node) return false;
    if (depth > 512) return true;
#define ESC(child) own_effect_escape_node(env, (child), effect, operation, facts, scope, depth + 1)
#define ESC_LIST(items, count) do { for (int e = 0; e < (count); ++e) if (ESC((items)[e])) return true; } while (0)
    switch (node->type) {
    case AST_EFFECT_OP: {
        EffectDef *definition = env_get_effect(env, node->as.effect_op.effect_name);
        if (!definition) return true;
        if (!strcmp(definition->name, effect) && !strcmp(node->as.effect_op.op_name, operation)) return true;
        ESC_LIST(node->as.effect_op.args, node->as.effect_op.arg_count);
        return false;
    }
    case AST_HANDLE_EXPR: {
        EffectDef *definition = own_effect_handler_definition(env, node);
        if (definition && !strcmp(definition->name, effect))
            for (int i = 0; i < node->as.handle_expr.handler_count; ++i)
                if (!strcmp(node->as.handle_expr.handler_op_names[i], operation)) return false;
        if (ESC(node->as.handle_expr.body)) return true;
        for (int arm = 0; arm < node->as.handle_expr.handler_count; ++arm) {
            OwnEffectScope local = {.parent = scope, .names = node->as.handle_expr.handler_param_names[arm],
                                    .name_count = node->as.handle_expr.handler_param_counts[arm]};
            if (own_effect_escape_node(env, node->as.handle_expr.handler_bodies[arm], effect, operation, facts, &local, depth + 1)) return true;
        }
        return false;
    }
    case AST_BLOCK: case AST_UNSAFE_BLOCK: {
        ASTNode **items = node->type == AST_BLOCK ? node->as.block.statements : node->as.unsafe_block.statements;
        int count = node->type == AST_BLOCK ? node->as.block.count : node->as.unsafe_block.count;
        OwnEffectScope local = {.parent = scope, .items = items};
        for (int i = 0; i < count; ++i) {
            local.count = i;
            if (own_effect_escape_node(env, items[i], effect, operation, facts, &local, depth + 1)) return true;
        }
        return false;
    }
    case AST_CALL: case AST_MODULE_QUALIFIED_CALL: {
        bool plain = node->type == AST_CALL;
        ASTNode **args = plain ? node->as.call.args : node->as.module_qualified_call.args;
        int count = plain ? node->as.call.arg_count : node->as.module_qualified_call.arg_count;
        ESC_LIST(args, count);
        if (plain && node->as.call.borrow_mode) return false;
        if (plain && (node->as.call.func_expr || own_effect_bound(scope, node->as.call.name))) return true;
        bool lookup_error = false;
        OwnFlow lookup = {.env = env, .error = &lookup_error};
        Function *callee = own_function(&lookup, node);
        if (lookup_error) return true;
        if (!callee) return !(plain && is_builtin_function(node->as.call.name));
        if (plain && !callee->body && is_builtin_function(node->as.call.name)) {
            const char *name = node->as.call.name;
            int callback = (!strcmp(name, "map") || !strcmp(name, "filter") ||
                            !strcmp(name, "array_map") || !strcmp(name, "array_filter")) ? 1 :
                           (!strcmp(name, "reduce") || !strcmp(name, "array_fold")) ? 2 : -1;
            if (callback < 0) return false;
            if (callback >= count || args[callback]->type != AST_IDENTIFIER) return true;
            ASTNode invoke = {.type = AST_CALL};
            invoke.as.call.name = args[callback]->as.identifier;
            return own_effect_escape_node(env, &invoke, effect, operation, facts, scope, depth + 1);
        }
        if (callee->is_extern) {
            for (int i = 0; i < callee->param_count; ++i) {
                if (callee->params[i].type != TYPE_FUNCTION) continue;
                if (i >= count || args[i]->type != AST_IDENTIFIER) return true;
                ASTNode callback = {.type = AST_CALL};
                callback.as.call.name = args[i]->as.identifier;
                if (own_effect_escape_node(env, &callback, effect, operation, facts, scope, depth + 1)) return true;
            }
            return callee->effect_count != 0;
        }
        for (int i = 0; i < env->function_count; ++i)
            if (&env->functions[i] == callee) return facts[i];
        return true;
    }
    case AST_LET: return ESC(node->as.let.value);
    case AST_SET: return ESC(node->as.set.value);
    case AST_RETURN: return ESC(node->as.return_stmt.value);
    case AST_PRINT: return ESC(node->as.print.expr);
    case AST_ASSERT: return ESC(node->as.assert.condition);
    case AST_IF: return ESC(node->as.if_stmt.condition) || ESC(node->as.if_stmt.then_branch) || ESC(node->as.if_stmt.else_branch);
    case AST_WHILE: return ESC(node->as.while_stmt.condition) || ESC(node->as.while_stmt.body);
    case AST_FOR: {
        if (ESC(node->as.for_stmt.range_expr)) return true;
        OwnEffectScope local = {.parent = scope, .binding = node->as.for_stmt.var_name};
        return own_effect_escape_node(env, node->as.for_stmt.body, effect, operation, facts, &local, depth + 1);
    }
    case AST_COND:
        ESC_LIST(node->as.cond_expr.conditions, node->as.cond_expr.clause_count);
        ESC_LIST(node->as.cond_expr.values, node->as.cond_expr.clause_count);
        return ESC(node->as.cond_expr.else_value);
    case AST_MATCH:
        if (ESC(node->as.match_expr.expr)) return true;
        for (int i = 0; i < node->as.match_expr.arm_count; ++i) {
            OwnEffectScope local = {.parent = scope, .binding = node->as.match_expr.pattern_bindings[i]};
            if (node->as.match_expr.guard_exprs && own_effect_escape_node(env, node->as.match_expr.guard_exprs[i], effect, operation, facts, &local, depth + 1)) return true;
            if (own_effect_escape_node(env, node->as.match_expr.arm_bodies[i], effect, operation, facts, &local, depth + 1)) return true;
        }
        return false;
    case AST_PREFIX_OP: ESC_LIST(node->as.prefix_op.args, node->as.prefix_op.arg_count); return false;
    case AST_STRUCT_LITERAL:
        if (ESC(node->as.struct_literal.spread_source)) return true;
        ESC_LIST(node->as.struct_literal.field_values, node->as.struct_literal.field_count); return false;
    case AST_UNION_CONSTRUCT: ESC_LIST(node->as.union_construct.field_values, node->as.union_construct.field_count); return false;
    case AST_ARRAY_LITERAL: ESC_LIST(node->as.array_literal.elements, node->as.array_literal.element_count); return false;
    case AST_TUPLE_LITERAL: ESC_LIST(node->as.tuple_literal.elements, node->as.tuple_literal.element_count); return false;
    case AST_FIELD_ACCESS: return ESC(node->as.field_access.object);
    case AST_TUPLE_INDEX: return ESC(node->as.tuple_index.tuple);
    case AST_FUNCTION: case AST_SHADOW: case AST_NUMBER: case AST_FLOAT: case AST_STRING: case AST_BOOL:
    case AST_IDENTIFIER: case AST_BREAK: case AST_CONTINUE: case AST_STRUCT_DEF: case AST_UNION_DEF:
    case AST_ENUM_DEF: case AST_EFFECT_DECL: case AST_OPAQUE_TYPE: case AST_IMPORT: case AST_MODULE_DECL:
    case AST_QUALIFIED_NAME: return false;
    default: return true;
    }
#undef ESC_LIST
#undef ESC
}

static bool own_call_effect_escape(OwnFlow *flow, ASTNode *node) {
    Environment *env = flow->env;
    if (!env->effect_count) return false;
    bool *facts = calloc(env->function_count ? (size_t)env->function_count : 1, sizeof(*facts));
    if (!facts) { own_error(flow, node, "I cannot allocate effect-call ownership facts", NULL); return true; }
    bool escapes = false;
    char *saved_module = env->current_module;
    for (int e = 0; e < env->effect_count && !escapes; ++e) {
        EffectDef *effect = &env->effects[e];
        for (int op = 0; op < effect->op_count && !escapes; ++op) {
            memset(facts, 0, (size_t)env->function_count * sizeof(*facts));
            bool changed;
            do {
                changed = false;
                for (int f = 0; f < env->function_count; ++f) {
                    Function *function = &env->functions[f];
                    if (facts[f] || function->is_extern) continue;
                    OwnEffectScope scope = {.params = function->params, .param_count = function->param_count};
                    env->current_module = function->module_name;
                    if (own_effect_escape_node(env, function->body, effect->name, effect->ops[op].name, facts, &scope, 0)) {
                        facts[f] = true; changed = true;
                    }
                }
            } while (changed);
            env->current_module = saved_module;
            OwnEffectScope caller = {.bindings = flow->bindings, .binding_count = flow->count};
            escapes = own_effect_escape_node(env, node, effect->name, effect->ops[op].name, facts, &caller, 0);
        }
    }
    env->current_module = saved_module;
    free(facts);
    return escapes;
}
