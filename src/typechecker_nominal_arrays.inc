/* I own each annotation tree and pair it with its declaring Environment owner.
 * A NULL owner denotes the root module, not an unsuccessful lookup. */
static bool nominal_view_copy(Environment *env, const TypeInfo *info, Type type, Type element,
                              const char *name, const char *owner, NominalView *out) {
    TypeInfo scalar = {0}, container = {0};
    scalar.base_type = type == TYPE_ARRAY ? element : type;
    scalar.generic_name = (char *)name;
    container.base_type = TYPE_ARRAY;
    container.element_type = &scalar;
    const TypeInfo *source = info ? info : type == TYPE_ARRAY ? &container : &scalar;
    if (source->base_type == TYPE_UNKNOWN ||
        !checked_signature_annotation(env, type, type == TYPE_ARRAY ? NULL : name, source, owner)) return false;
    return nominal_view_copy_context(env, source, owner, NULL, 0, out);
}
static bool nominal_view_wrap_array(NominalView *view) {
    if (view->payload) return false;
    TypeInfo *array = calloc(1, sizeof *array);
    if (!array) return false;
    array->base_type = TYPE_ARRAY;
    array->element_type = view->info;
    view->info = array;
    return true;
}
static bool nominal_view_element(Environment *env, NominalView *view, unsigned depth) {
    if (!view->info || view->info->base_type != TYPE_ARRAY || !view->info->element_type || view->payload)
        return false;
    NominalView element = {0};
    if (!nominal_view_copy_context(env, view->info->element_type, view->owner,
            nominal_view_context(view), depth + 1, &element)) return false;
    nominal_view_discard(view);
    *view = element;
    return true;
}
static bool nominal_field_view(ASTNode *expr, Environment *env, unsigned depth, NominalView *out) {
    NominalView object = {0};
    if (depth > 128 || !nominal_value_view(expr->as.field_access.object, env, depth + 1, &object)) return false;
    bool ok = false;
    if (object.payload) {
        NominalIdentity id = env_nominal_identity(env, object.info->generic_name, object.owner, TYPE_UNION);
        if (!id.ordinal) goto done;
        UnionDef *definition = &env->unions[id.ordinal - 1];
        int arm = object.variant;
        if (arm < 0 || arm >= definition->variant_count) goto done;
        for (int field = 0; field < definition->variant_field_counts[arm]; ++field) {
            if (strcmp(definition->variant_field_names[arm][field], expr->as.field_access.field_name)) continue;
            const TypeInfo *info = definition->variant_field_type_info && definition->variant_field_type_info[arm]
                ? definition->variant_field_type_info[arm][field] : NULL;
            NominalSubstitution context = {definition, object.info, object.owner, nominal_view_context(&object)};
            TypeInfo flat = {.base_type = definition->variant_field_types[arm][field],
                .generic_name = definition->variant_field_type_names && definition->variant_field_type_names[arm]
                    ? definition->variant_field_type_names[arm][field] : NULL};
            ok = nominal_view_copy_context(env, info ? info : &flat, definition->module_name, &context, depth + 1, out);
            break;
        }
    } else {
        NominalIdentity id = nominal_view_identity(env, &object, TYPE_STRUCT);
        if (!id.ordinal) goto done;
        StructDef *record = &env->structs[id.ordinal - 1];
        for (int field = 0; field < record->field_count; ++field) {
            if (strcmp(record->field_names[field], expr->as.field_access.field_name)) continue;
            ok = nominal_view_copy(env, record->field_type_info ? record->field_type_info[field] : NULL,
                record->field_types[field], record->field_element_types ? record->field_element_types[field] : TYPE_UNKNOWN,
                record->field_type_names ? record->field_type_names[field] : NULL, record->module_name, out);
            break;
        }
    }
done:
    nominal_view_discard(&object);
    return ok;
}
static bool nominal_array_builtin(ASTNode *expr, Environment *env, const char *name, int arity) {
    if (expr->type != AST_CALL || expr->as.call.func_expr || !expr->as.call.name ||
        strcmp(expr->as.call.name, name) || expr->as.call.arg_count != arity ||
        env_get_var_visible_at(env, name, expr->line, expr->column)) return false;
    Function *function = env_get_function(env, name);
    return env_function_is_builtin(function);
}
static bool nominal_value_view(ASTNode *expr, Environment *env, unsigned depth, NominalView *out) {
    if (!expr || depth > 128 || !out) return false;
    NominalView view = {0};
    if (expr->type == AST_BLOCK && expr->as.block.count)
        return nominal_value_view(expr->as.block.statements[expr->as.block.count - 1], env, depth + 1, out);
    if (expr->type == AST_RETURN)
        return nominal_value_view(expr->as.return_stmt.value, env, depth + 1, out);
    if (expr->type == AST_IF || expr->type == AST_COND || expr->type == AST_MATCH) {
        int count = expr->type == AST_IF ? 2 : expr->type == AST_MATCH
            ? expr->as.match_expr.arm_count : expr->as.cond_expr.clause_count + 1;
        if (count <= 0) return false;
        for (int i = 0; i < count; ++i) {
            ASTNode *body = expr->type == AST_IF ? (i ? expr->as.if_stmt.else_branch : expr->as.if_stmt.then_branch)
                : expr->type == AST_MATCH ? expr->as.match_expr.arm_bodies[i]
                : i < expr->as.cond_expr.clause_count ? expr->as.cond_expr.values[i] : expr->as.cond_expr.else_value;
            NominalView arm = {0};
            if (!nominal_value_view(body, env, depth + 1, &arm)) {
                if (body && body->type == AST_ARRAY_LITERAL) continue;
                goto fail;
            }
            if (!view.info) view = arm;
            else {
                bool same = checked_annotations_equal_context(env, view.info, view.owner, arm.info, arm.owner, depth + 1, nominal_view_context(&view), nominal_view_context(&arm)) &&
                    view.payload == arm.payload && (!view.payload || view.variant == arm.variant);
                nominal_view_discard(&arm);
                if (!same) goto fail;
            }
        }
        if (!view.info) goto fail;
        for (int i = 0; i < count; ++i) {
            ASTNode *body = expr->type == AST_IF ? (i ? expr->as.if_stmt.else_branch : expr->as.if_stmt.then_branch)
                : expr->type == AST_MATCH ? expr->as.match_expr.arm_bodies[i]
                : i < expr->as.cond_expr.clause_count ? expr->as.cond_expr.values[i] : expr->as.cond_expr.else_value;
            if (body && body->type == AST_ARRAY_LITERAL &&
                !nominal_array_matches_context(env, view.info, view.owner, body, depth + 1, nominal_view_context(&view))) goto fail;
        }
        *out = view; return true;
    }
    if (expr->type == AST_IDENTIFIER) {
        Symbol *symbol = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
        if (symbol && symbol->checker_nominal_view)
            return nominal_view_clone(env, symbol->checker_nominal_view, depth + 1, out);
        if (symbol) return nominal_view_copy(env, symbol->type_info, symbol->type, symbol->element_type,
            symbol->struct_type_name, symbol->type == TYPE_FUNCTION ? symbol->callable_owner : symbol->nominal_owner, out);
        FunctionSignature *signature = NULL;
        const char *owner = NULL;
        if (nominal_callable_view(expr, env, depth + 1, &signature, &owner)) {
            TypeInfo function = {.base_type = TYPE_FUNCTION, .fn_sig = signature};
            bool ok = nominal_view_copy(env, &function, TYPE_FUNCTION, TYPE_UNKNOWN, NULL, owner, out);
            free_function_signature(signature);
            return ok;
        }
    }
    if (expr->type == AST_FIELD_ACCESS) {
        ASTNode *object = expr->as.field_access.object;
        if (object && object->type == AST_IDENTIFIER &&
            !env_get_var_visible_at(env, object->as.identifier, object->line, object->column)) {
            EnumDef *definition = env_get_enum(env, object->as.identifier);
            if (definition && check_expression(expr, env) == TYPE_ENUM)
                return nominal_view_copy(env, NULL, TYPE_ENUM, TYPE_UNKNOWN,
                    definition->name, definition->module_name, out);
        }
        return nominal_field_view(expr, env, depth + 1, out);
    }
    if (expr->type == AST_ARRAY_LITERAL) {
        int count = expr->as.array_literal.element_count;
        if (count <= 0) return false; /* Only a destination can type an empty array. */
        for (int i = 0; i < count; ++i) {
            NominalView element = {0};
            if (!nominal_value_view(expr->as.array_literal.elements[i], env, depth + 1, &element)) {
                if (expr->as.array_literal.elements[i]->type == AST_ARRAY_LITERAL) continue;
                goto fail;
            }
            if (!view.info) view = element;
            else {
                bool same = checked_annotations_equal_context(env, view.info, view.owner, element.info, element.owner, depth + 1, nominal_view_context(&view), nominal_view_context(&element)) &&
                    view.payload == element.payload && (!view.payload || view.variant == element.variant);
                nominal_view_discard(&element);
                if (!same) goto fail;
            }
        }
        if (!view.info) goto fail;
        for (int i = 0; i < count; ++i)
            if (expr->as.array_literal.elements[i]->type == AST_ARRAY_LITERAL &&
                !nominal_array_matches_context(env, view.info, view.owner, expr->as.array_literal.elements[i], depth + 1, nominal_view_context(&view))) goto fail;
        if (!nominal_view_wrap_array(&view)) goto fail;
        *out = view; return true;
    }
    if (nominal_array_builtin(expr, env, "at", 2) || nominal_array_builtin(expr, env, "array_get", 2)) {
        if (!nominal_value_view(expr->as.call.args[0], env, depth + 1, &view) || !nominal_view_element(env, &view, depth + 1)) goto fail;
        *out = view; return true;
    }
    if (nominal_array_builtin(expr, env, "array_push", 2)) {
        ASTNode *receiver = expr->as.call.args[0];
        NominalView element = {0};
        if (!nominal_value_view(expr->as.call.args[1], env, depth + 1, &element)) return false;
        if (receiver->type == AST_ARRAY_LITERAL && receiver->as.array_literal.element_count == 0) {
            view = element;
            if (!nominal_view_wrap_array(&view)) goto fail;
        } else {
            bool ok = nominal_value_view(receiver, env, depth + 1, &view) &&
                view.info->base_type == TYPE_ARRAY && view.info->element_type &&
                (!nominal_array_requires_context(env, view.info->element_type, view.owner, nominal_view_context(&view), depth + 1) ||
                 checked_annotations_equal_context(env, view.info->element_type, view.owner,
                    element.info, element.owner, depth + 1, nominal_view_context(&view), nominal_view_context(&element)));
            nominal_view_discard(&element);
            if (!ok) goto fail;
        }
        *out = view; return true;
    }
    if (nominal_array_builtin(expr, env, "array_slice", 3))
        return nominal_value_view(expr->as.call.args[0], env, depth + 1, out);
    if (nominal_array_builtin(expr, env, "range", 2))
        return nominal_view_copy(env, NULL, TYPE_ARRAY, TYPE_INT, NULL, NULL, out);
    if (nominal_array_builtin(expr, env, "array_new", 2)) {
        if (!nominal_value_view(expr->as.call.args[1], env, depth + 1, &view) || !nominal_view_wrap_array(&view)) goto fail;
        *out = view; return true;
    }
    if (nominal_array_builtin(expr, env, "map", 2) || nominal_array_builtin(expr, env, "filter", 2)) {
        bool filter = !strcmp(expr->as.call.name, "filter");
        FunctionSignature *callback = NULL;
        const char *owner = NULL;
        if (!nominal_callable_view(expr->as.call.args[1], env, depth + 1, &callback, &owner)) return false;
        bool ok = callback->param_count == 1 && callback->param_types &&
            nominal_value_view(expr->as.call.args[0], env, depth + 1, &view) &&
            view.info->base_type == TYPE_ARRAY && view.info->element_type;
        NominalView parameter = {0};
        if (ok && nominal_array_requires_context(env, view.info->element_type, view.owner, nominal_view_context(&view), depth + 1)) {
            ok = nominal_view_copy(env, callback->param_type_info ? callback->param_type_info[0] : NULL,
                callback->param_types[0], TYPE_UNKNOWN,
                callback->param_struct_names ? callback->param_struct_names[0] : NULL, owner, &parameter) &&
                checked_annotations_equal_context(env, view.info->element_type, view.owner, parameter.info, owner, depth + 1, nominal_view_context(&view), nominal_view_context(&parameter));
        }
        nominal_view_discard(&parameter);
        if (ok && filter) ok = callback->return_type == TYPE_BOOL;
        if (ok && !filter) {
            nominal_view_discard(&view);
            ok = nominal_view_copy(env, callback->return_type_info, callback->return_type, TYPE_UNKNOWN,
                callback->return_struct_name, owner, &view) && nominal_view_wrap_array(&view);
        }
        free_function_signature(callback);
        if (!ok) goto fail;
        *out = view; return true;
    }
    if (expr->type == AST_CALL || expr->type == AST_MODULE_QUALIFIED_CALL) {
        Function *function = nominal_direct_function(expr, env);
        if (function) return nominal_view_copy(env, function->return_type_info, function->return_type,
            function->return_element_type, function->return_struct_type_name,
            env_function_signature_owner(env, function), out);
        if (expr->type == AST_CALL && expr->as.call.checked_signature) {
            FunctionSignature *signature = expr->as.call.checked_signature;
            const char *owner = NULL;
            if (!nominal_callable_owner(expr, env, depth + 1, &owner)) return false;
            return nominal_view_copy(env, signature->return_type_info, signature->return_type, TYPE_UNKNOWN,
                signature->return_struct_name, owner, out);
        }
    }
    if (expr->type == AST_UNION_CONSTRUCT) {
        TypeInfo simple = {.base_type = TYPE_UNION, .generic_name = expr->as.union_construct.union_name};
        const TypeInfo *info = expr->as.union_construct.type_info ? expr->as.union_construct.type_info : &simple;
        return nominal_view_copy_context(env, info, env->current_module, NULL, depth + 1, out);
    }
    if (expr->type == AST_STRUCT_LITERAL && expr->as.struct_literal.struct_name) {
        const char *name = expr->as.struct_literal.struct_name;
        const char *dot = strrchr(name, '.');
        if (dot) {
            char *prefix = strndup(name, (size_t)(dot - name));
            if (!prefix) return false;
            NominalIdentity id = env_nominal_identity(env, prefix, env->current_module, TYPE_UNION);
            if (id.ordinal) {
                UnionDef *definition = &env->unions[id.ordinal - 1];
                bool selected = false;
                for (int i = 0; i < definition->variant_count; ++i)
                    if (!strcmp(definition->variant_names[i], dot + 1)) selected = true;
                TypeInfo info = {.base_type = TYPE_UNION, .generic_name = prefix};
                bool ok = selected && nominal_view_copy_context(env, &info, env->current_module, NULL, depth + 1, out);
                free(prefix);
                return ok;
            }
            free(prefix);
        }
    }
    /* Nominal scalar constructors and list intrinsics use the exact resolver;
     * these are not indexing calls, so this fallback cannot recurse into itself. */
    if (expr->type == AST_STRUCT_LITERAL || expr->type == AST_CALL) {
        NominalIdentity id = nominal_expression(expr, env, TYPE_STRUCT, depth + 1);
        if (id.ordinal) return nominal_view_copy(env, NULL, TYPE_STRUCT, TYPE_UNKNOWN,
            env_nominal_name(env, id), env_nominal_owner(env, id), out);
    }
    Type scalar = expr->type == AST_NUMBER ? TYPE_INT : expr->type == AST_FLOAT ? TYPE_FLOAT
        : expr->type == AST_STRING ? TYPE_STRING : expr->type == AST_BOOL ? TYPE_BOOL : TYPE_UNKNOWN;
    if (scalar == TYPE_UNKNOWN && expr->type == AST_PREFIX_OP) {
        Type checked = check_expression(expr, env);
        if (checked == TYPE_INT || checked == TYPE_U8 || checked == TYPE_FLOAT ||
            checked == TYPE_BOOL || checked == TYPE_STRING) scalar = checked;
    }
    if (scalar != TYPE_UNKNOWN) return nominal_view_copy(env, NULL, scalar, TYPE_UNKNOWN, NULL, NULL, out);
    return false;
fail:
    nominal_view_discard(&view);
    return false;
}
