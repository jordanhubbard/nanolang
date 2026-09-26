typedef struct { size_t tuples, arrays; } NativeContextMark;
static NativeContextMark native_context_mark(const Environment *env) {
    return (NativeContextMark){env ? env->tuple_literal_binding_count : 0,
                               env ? env->array_expression_binding_count : 0};
}
/* Cache rows append independent owners. Rollback never allocates or replaces
 * a prior snapshot; no caller borrows the private backing row vector. */
static void native_context_restore(Environment *env, NativeContextMark mark) {
    if (!env) return;
    while (env->tuple_literal_binding_count > mark.tuples) {
        TupleLiteralBinding *row = &env->tuple_literal_bindings[--env->tuple_literal_binding_count];
        free_payload_type_info(row->type_info); memset(row, 0, sizeof *row);
    }
    while (env->array_expression_binding_count > mark.arrays) {
        TupleLiteralBinding *row = &env->array_expression_bindings[--env->array_expression_binding_count];
        free_payload_type_info(row->type_info); memset(row, 0, sizeof *row);
    }
    if (!env->tuple_literal_binding_count) {
        free(env->tuple_literal_bindings); env->tuple_literal_bindings = NULL;
    }
    if (!env->array_expression_binding_count) {
        free(env->array_expression_bindings); env->array_expression_bindings = NULL;
    }
}

/* I publish independent native metadata only after source compatibility has
 * selected a complete view. These copies never authorize nominal identity. */
static bool native_bind_checked_view_impl(Environment *env, const NominalView *view,
                                     ASTNode *value, unsigned depth) {
    if (!env || !view || !view->info || !value || depth > 128) return false;
    if (value->type == AST_BLOCK && value->as.block.count)
        return native_bind_checked_view_impl(env, view,
            value->as.block.statements[value->as.block.count - 1], depth + 1);
    if (value->type == AST_RETURN)
        return native_bind_checked_view_impl(env, view, value->as.return_stmt.value, depth + 1);
    if (value->type == AST_IF || value->type == AST_COND || value->type == AST_MATCH) {
        int count = value->type == AST_IF ? 2 : value->type == AST_MATCH
            ? value->as.match_expr.arm_count : value->as.cond_expr.clause_count + 1;
        for (int i = 0; i < count; ++i) {
            ASTNode *body = value->type == AST_IF ? (i ? value->as.if_stmt.else_branch : value->as.if_stmt.then_branch)
                : value->type == AST_MATCH ? value->as.match_expr.arm_bodies[i]
                : i < value->as.cond_expr.clause_count ? value->as.cond_expr.values[i] : value->as.cond_expr.else_value;
            if (ast_always_returns(body)) continue;
            if (!native_bind_checked_view_impl(env, view, body, depth + 1)) return false;
        }
        return true;
    }
    bool array = view->info->base_type == TYPE_ARRAY;
    bool tuple = view->info->base_type == TYPE_TUPLE && value->type == AST_TUPLE_LITERAL;
    if (!array && !tuple) return true;
    TypeInfo *concrete = NULL;
    if (!nominal_view_materialize(env, view, depth + 1, &concrete)) return false;
    const TypeInfo *retained = array ? env_array_expression_info(env, value) : NULL;
    /* A prior parser pass can retain a redundant flattened record spelling on
     * an array. I compare its complete element tree before reusing that row. */
    bool ok = retained ? reduce_types_exact(retained, concrete, env, depth + 1)
                       : array ? env_bind_array_expression(env, value, concrete)
                               : env_bind_tuple_literal(env, value, concrete);
    free_payload_type_info(concrete);
    if (!ok) return false;
    bool literal = array ? value->type == AST_ARRAY_LITERAL : tuple;
    if (!literal) return true;
    int count = array ? value->as.array_literal.element_count : value->as.tuple_literal.element_count;
    if (count < 0) return false;
    for (int i = 0; i < count; ++i) {
        NominalView child = {0};
        ASTNode *item = array ? value->as.array_literal.elements[i] : value->as.tuple_literal.elements[i];
        ok = nominal_view_child(env, view, array ? 0 : (size_t)i, depth + 1, &child) &&
             native_bind_checked_view_impl(env, &child, item, depth + 1);
        nominal_view_discard(&child);
        if (!ok) return false;
    }
    return true;
}
static bool native_bind_checked_view(Environment *env, const NominalView *view,
                                     ASTNode *value, unsigned depth) {
    NativeContextMark mark = native_context_mark(env);
    bool ok = native_bind_checked_view_impl(env, view, value, depth);
    if (!ok) native_context_restore(env, mark);
    return ok;
}
static bool native_bind_checked_annotation(Environment *env, const TypeInfo *info,
    const char *owner, const NominalSubstitution *context, ASTNode *value, unsigned depth) {
    if (!info) return true;
    if (!nominal_substitute_annotation(&info, &owner, &context)) return false;
    if (info->base_type != TYPE_ARRAY && info->base_type != TYPE_TUPLE) return true;
    NominalView view = {0};
    bool ok = nominal_view_copy_context(env, info, owner, context, depth + 1, &view) &&
              native_bind_checked_view(env, &view, value, depth + 1);
    nominal_view_discard(&view);
    if (!ok) {
        env->opaque_resolution_failed = true;
        emit_context_error("E001 TYPE MISMATCH", value ? value->line : 0, value ? value->column : 0, 1,
            "I cannot retain the checked native container annotation.",
            "Preserve its complete child facts and allocation boundary.");
    }
    return ok;
}

/* I return Environment-owned emission storage. My callers may not use this
 * flattened snapshot in place of the original owner/context proof. */
const TypeInfo *checked_expression_type_info(ASTNode *expr, Environment *env) {
    if (!expr || !env) return NULL;
    const TypeInfo *old = env_array_expression_info(env, expr);
    if (!old && expr->type == AST_TUPLE_LITERAL) old = env_tuple_literal_info(env, expr);
    if (old) return old;
    NativeContextMark mark = native_context_mark(env);
    NominalView view = {0};
    TypeInfo *copy = NULL;
    if (!nominal_value_view(expr, env, 0, &view) || view.payload ||
        !nominal_view_materialize(env, &view, 0, &copy)) {
        nominal_view_discard(&view); free_payload_type_info(copy);
        native_context_restore(env, mark); return NULL;
    }
    bool stored = native_bind_checked_view(env, &view, expr, 0);
    nominal_view_discard(&view);
    if (!stored || !env_own_checker_type_info(env, copy)) {
        free_payload_type_info(copy); native_context_restore(env, mark);
        env->opaque_resolution_failed = true; return NULL;
    }
    return copy;
}
FunctionSignature *checked_callable_signature_copy(ASTNode *expr, Environment *env) {
    NativeContextMark mark = native_context_mark(env);
    NominalView view = {0};
    TypeInfo *copy = NULL;
    FunctionSignature *result = NULL;
    bool ok = expr && env && nominal_callable_view(expr, env, 0, &view) &&
        nominal_view_materialize(env, &view, 0, &copy) && copy &&
        copy->base_type == TYPE_FUNCTION && copy->fn_sig &&
        copy_function_signature_checked(copy->fn_sig, &result);
    nominal_view_discard(&view); free_payload_type_info(copy);
    if (!ok) native_context_restore(env, mark);
    return ok ? result : NULL;
}

/* I project emission storage only from the original declaration proof. */
bool checked_union_projection_copy(ASTNode *expr, Environment *env,
                                   TypeInfo **annotation, char **variant) {
    if (!expr || !env || !annotation || !variant) return false;
    NominalView view = {0};
    TypeInfo *copy = NULL;
    char *selected = NULL, *canonical = NULL;
    if (!nominal_value_view(expr, env, 0, &view) || !view.info || view.children) goto fail;
    const TypeInfo *source = view.info;
    const char *owner = view.owner;
    const NominalSubstitution *context = nominal_view_context(&view);
    if (!nominal_substitute_annotation(&source, &owner, &context) ||
        source->base_type != TYPE_UNION ||
        !checked_annotations_equal_context(env, source, owner, source, owner, 0, context, context)) goto fail;
    NominalIdentity identity = env_nominal_identity(env, source->generic_name, owner, TYPE_UNION);
    if (!identity.ordinal) goto fail;
    UnionDef *definition = &env->unions[identity.ordinal - 1];
    if (view.payload) {
        if (view.variant < 0 || view.variant >= definition->variant_count ||
            !definition->variant_names || !definition->variant_names[view.variant]) goto fail;
        selected = strdup(definition->variant_names[view.variant]);
        if (!selected) goto fail;
    }
    const char *name = env_nominal_name(env, identity);
    if (!name || !(canonical = strdup(name)) ||
        !nominal_view_materialize(env, &view, 0, &copy) || !copy || copy->base_type != TYPE_UNION) goto fail;
    free(copy->generic_name);
    copy->generic_name = canonical;
    canonical = NULL;
    nominal_view_discard(&view);
    *annotation = copy;
    *variant = selected;
    return true;
fail:
    nominal_view_discard(&view);
    free_payload_type_info(copy);
    free(selected); free(canonical);
    return false;
}
