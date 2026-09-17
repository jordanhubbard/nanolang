/* I distinguish mutation of scalar frame locals from shared effects. This
 * bounded par classifier does not relax the separate pure-fn contract. */
static bool par_value_type(Type type) {
    return type == TYPE_INT || type == TYPE_FLOAT || type == TYPE_BOOL || type == TYPE_STRING;
}
static bool par_closed_node(PurityWalk *, ASTNode *, unsigned);
static bool par_closed_function(PurityWalk *parent, Function *fn, unsigned depth) {
    if (!fn || !fn->body || fn->is_extern || depth >= 64 ||
        (!par_value_type(fn->return_type) && fn->return_type != TYPE_VOID)) return false;
    int index = (int)(fn - parent->env->functions);
    if (index < 0 || index >= parent->env->function_count || parent->seen[index]) return false;
    parent->seen[index] = true;
    PurityWalk walk = {0};
    walk.env = parent->env; walk.seen = parent->seen; walk.source_file = fn->source_file;
    char *saved = walk.env->current_module;
    walk.env->current_module = fn->module_name;
    bool ok = true;
    for (int i = 0; ok && i < fn->param_count; ++i)
        ok = par_value_type(fn->params[i].type) && purity_add(&walk, fn->params[i].name, false);
    if (ok) ok = par_closed_node(&walk, fn->body, depth + 1);
    walk.env->current_module = saved;
    free(walk.locals); free(walk.mutable);
    parent->seen[index] = false;
    return ok;
}
static bool par_closed_target(PurityWalk *walk, const char *name, int arity, unsigned depth) {
    if (!name || purity_local(walk, name) >= 0) return false;
    Function *fn = purity_function_decl(walk, name);
    if (fn && (fn->body || fn->is_extern))
        return fn->param_count == arity && par_closed_function(walk, fn, depth);
    /* The language not operator lowers to a scalar instruction, not a host ABI call. */
    return arity == 1 && !strcmp(name, "not");
}
static bool par_closed_node_target(PurityWalk *walk, ASTNode *node, unsigned depth) {
    if (node->type == AST_CALL)
        return !node->as.call.func_expr && par_closed_target(walk, node->as.call.name, node->as.call.arg_count, depth);
    if (node->type != AST_MODULE_QUALIFIED_CALL) return false;
    const char *module = node->as.module_qualified_call.module_alias;
    const char *function = node->as.module_qualified_call.function_name;
    size_t length = strlen(module) + strlen(function) + 2;
    char *name = malloc(length);
    if (!name) return false;
    snprintf(name, length, "%s.%s", module, function);
    bool ok = par_closed_target(walk, name, node->as.module_qualified_call.arg_count, depth);
    free(name);
    return ok;
}
static bool par_closed_node(PurityWalk *walk, ASTNode *node, unsigned depth) {
    if (!node) return true;
    switch (node->type) {
    case AST_NUMBER: case AST_FLOAT: case AST_BOOL: case AST_STRING:
    case AST_BREAK: case AST_CONTINUE: return true;
    case AST_IDENTIFIER: return purity_local(walk, node->as.identifier) >= 0;
    case AST_PREFIX_OP:
        for (int i = 0; i < node->as.prefix_op.arg_count; ++i)
            if (!par_closed_node(walk, node->as.prefix_op.args[i], depth)) return false;
        return true;
    case AST_CALL:
        if (node->as.call.func_expr || !par_closed_target(walk, node->as.call.name, node->as.call.arg_count, depth)) return false;
        for (int i = 0; i < node->as.call.arg_count; ++i)
            if (!par_closed_node(walk, node->as.call.args[i], depth)) return false;
        return true;
    case AST_MODULE_QUALIFIED_CALL:
        if (!par_closed_node_target(walk, node, depth)) return false;
        for (int i = 0; i < node->as.module_qualified_call.arg_count; ++i)
            if (!par_closed_node(walk, node->as.module_qualified_call.args[i], depth)) return false;
        return true;
    case AST_LET:
        if ((node->as.let.var_type != TYPE_UNKNOWN && !par_value_type(node->as.let.var_type)) ||
            !par_closed_node(walk, node->as.let.value, depth)) return false;
        return purity_add(walk, node->as.let.name, node->as.let.is_mut);
    case AST_SET:
        return (!node->as.set.field_name || !node->as.set.field_name[0]) &&
               purity_local(walk, node->as.set.name) >= 0 &&
               par_closed_node(walk, node->as.set.value, depth);
    case AST_BLOCK: {
        int saved = walk->count;
        bool ok = true;
        for (int i = 0; ok && i < node->as.block.count; ++i)
            ok = par_closed_node(walk, node->as.block.statements[i], depth);
        walk->count = saved;
        return ok;
    }
    case AST_IF:
        return par_closed_node(walk, node->as.if_stmt.condition, depth) &&
               par_closed_node(walk, node->as.if_stmt.then_branch, depth) &&
               par_closed_node(walk, node->as.if_stmt.else_branch, depth);
    case AST_WHILE:
        return par_closed_node(walk, node->as.while_stmt.condition, depth) &&
               par_closed_node(walk, node->as.while_stmt.body, depth);
    case AST_RETURN: return par_closed_node(walk, node->as.return_stmt.value, depth);
    case AST_ASSERT: return par_closed_node(walk, node->as.assert.condition, depth);
    default: return false;
    }
}
static bool par_closed_call(ASTNode *node, Environment *env) {
    if (!node || (node->type != AST_CALL && node->type != AST_MODULE_QUALIFIED_CALL)) return false;
    if (node->type == AST_CALL && (node->as.call.func_expr ||
        env_get_var_visible_at(env, node->as.call.name, node->line, node->column))) return false;
    PurityWalk walk = {0}; walk.env = env;
    walk.seen = calloc(env->function_count ? (size_t)env->function_count : 1, sizeof(bool));
    if (!walk.seen) return false;
    bool ok = par_closed_node_target(&walk, node, 0);
    free(walk.seen);
    return ok;
}
