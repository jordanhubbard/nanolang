/* I own every link needed after the projection stack unwinds. */
typedef struct OwnedNominalContext {
    NominalSubstitution context;
    UnionDef declaration;
    struct OwnedNominalContext *parent;
} OwnedNominalContext;

static const NominalSubstitution *nominal_view_context(const NominalView *view) {
    return view->owned_context ? &view->owned_context->context : NULL;
}
static void nominal_context_free(OwnedNominalContext *owned) {
    if (!owned) return;
    nominal_context_free(owned->parent);
    for (int i = 0; owned->declaration.generic_params && i < owned->declaration.generic_param_count; ++i)
        free(owned->declaration.generic_params[i]);
    free(owned->declaration.generic_params);
    free_payload_type_info((TypeInfo *)owned->context.arguments);
    free((char *)owned->context.argument_owner);
    free(owned);
}
static bool nominal_context_copy(const NominalSubstitution *source, unsigned depth,
                                  OwnedNominalContext **out) {
    if (!source) { *out = NULL; return true; }
    if (depth > 128 || !source->declaration || source->declaration->generic_param_count < 0 ||
        (source->declaration->generic_param_count && !source->declaration->generic_params)) return false;
    OwnedNominalContext *copy = calloc(1, sizeof *copy);
    if (!copy) return false;
    copy->context.declaration = &copy->declaration;
    copy->declaration.generic_param_count = source->declaration->generic_param_count;
    if (copy->declaration.generic_param_count) {
        copy->declaration.generic_params = calloc((size_t)copy->declaration.generic_param_count, sizeof(char *));
        if (!copy->declaration.generic_params) goto fail;
        for (int i = 0; i < copy->declaration.generic_param_count; ++i) {
            if (!source->declaration->generic_params[i]) goto fail;
            copy->declaration.generic_params[i] = strdup(source->declaration->generic_params[i]);
            if (!copy->declaration.generic_params[i]) goto fail;
        }
    }
    TypeInfo *arguments = NULL;
    if (!source->arguments || !copy_payload_type_info_checked(source->arguments, &arguments)) goto fail;
    copy->context.arguments = arguments;
    if (source->argument_owner) {
        copy->context.argument_owner = strdup(source->argument_owner);
        if (!copy->context.argument_owner) goto fail;
    }
    if (!nominal_context_copy(source->argument_context, depth + 1, &copy->parent)) goto fail;
    copy->context.argument_context = copy->parent ? &copy->parent->context : NULL;
    *out = copy;
    return true;
fail:
    nominal_context_free(copy);
    return false;
}
static void nominal_view_discard(NominalView *view) {
    for (size_t i = 0; view->children && i < view->child_count; ++i) nominal_view_discard(&view->children[i]);
    free(view->children);
    free_payload_type_info(view->info);
    free((char *)view->owner);
    nominal_context_free(view->owned_context);
    memset(view, 0, sizeof *view);
}
static void nominal_view_destroy(void *object) {
    nominal_view_discard(object);
    free(object);
}
static bool nominal_view_copy_context(Environment *env, const TypeInfo *source,
    const char *owner, const NominalSubstitution *context, unsigned depth, NominalView *out) {
    if (!source || depth > 128 || !nominal_substitute_annotation(&source, &owner, &context) ||
        !checked_annotations_equal_context(env, source, owner, source, owner, depth + 1, context, context))
        return false;
    NominalView copy = {0};
    if (!copy_payload_type_info_checked(source, &copy.info)) return false;
    Type kind; const char *name;
    if (!checked_annotation_kind(env, source, owner, &kind, &name)) goto fail;
    copy.info->base_type = kind;
    if (owner) {
        copy.owner = strdup(owner);
        if (!copy.owner) goto fail;
    }
    if (!nominal_context_copy(context, depth + 1, &copy.owned_context)) goto fail;
    *out = copy;
    return true;
fail:
    nominal_view_discard(&copy);
    return false;
}
/* Composed outer metadata duplicates storage; each child remains authority. */
static bool nominal_view_composed_valid(const NominalView *view) {
    if (!view || !view->children || !view->info || view->owner || view->owned_context || view->payload ||
        view->info->generic_name || view->info->opaque_type_name || view->info->fn_sig ||
        view->info->row_field_count || view->info->type_var_count || view->info->row_var_name || view->info->is_open_row)
        return false;
    const TypeInfo *info = view->info;
    if (info->base_type == TYPE_ARRAY) {
        return view->child_count == 1 && info->element_type && !info->type_param_count &&
            !info->tuple_element_count && type_infos_equal(info->element_type, view->children[0].info);
    }
    if (info->base_type != TYPE_TUPLE || info->element_type || !type_info_tuple_valid(info) ||
        info->tuple_element_count < 0 || (size_t)info->tuple_element_count != view->child_count) return false;
    for (size_t i = 0; i < view->child_count; ++i) {
        TypeInfo flat;
        const TypeInfo *child = type_info_tuple_element(info, (int)i, &flat);
        if (!child || !type_infos_equal(child, view->children[i].info)) return false;
    }
    return true;
}
static bool nominal_view_clone(Environment *env, const NominalView *source, unsigned depth, NominalView *out) {
    if (!source || depth > 128) return false;
    if (!source->children) {
        if (source->child_count || !nominal_view_copy_context(env, source->info, source->owner,
                nominal_view_context(source), depth + 1, out)) return false;
        out->payload = source->payload;
        out->variant = source->variant;
        return true;
    }
    if (!nominal_view_composed_valid(source) || source->child_count > SIZE_MAX / sizeof(NominalView)) return false;
    NominalView copy = {0};
    if (!copy_payload_type_info_checked(source->info, &copy.info)) return false;
    copy.children = calloc(source->child_count ? source->child_count : 1, sizeof *copy.children);
    if (!copy.children) goto fail;
    copy.child_count = source->child_count;
    for (size_t i = 0; i < copy.child_count; ++i)
        if (!nominal_view_clone(env, &source->children[i], depth + 1, &copy.children[i])) goto fail;
    *out = copy;
    return true;
fail:
    nominal_view_discard(&copy);
    return false;
}
static bool nominal_view_child(Environment *env, const NominalView *source, size_t index,
                                unsigned depth, NominalView *out) {
    if (!source || !source->info || source->payload || depth > 128) return false;
    if (source->children) {
        return nominal_view_composed_valid(source) && index < source->child_count &&
            nominal_view_clone(env, &source->children[index], depth + 1, out);
    }
    TypeInfo flat;
    const TypeInfo *child = NULL;
    if (source->info->base_type == TYPE_ARRAY && index == 0) child = source->info->element_type;
    if (source->info->base_type == TYPE_TUPLE && index <= INT_MAX)
        child = type_info_tuple_element(source->info, (int)index, &flat);
    return child && nominal_view_copy_context(env, child, source->owner, nominal_view_context(source), depth + 1, out);
}
static bool nominal_view_equal(Environment *env, const NominalView *a, const NominalView *b, unsigned depth) {
    if (!a || !b || !a->info || !b->info || depth > 128 || a->payload != b->payload ||
        (a->payload && a->variant != b->variant)) return false;
    if (!a->children && !b->children)
        return checked_annotations_equal_context(env, a->info, a->owner, b->info, b->owner,
            depth + 1, nominal_view_context(a), nominal_view_context(b));
    if ((a->children && !nominal_view_composed_valid(a)) || (b->children && !nominal_view_composed_valid(b)) ||
        a->info->base_type != b->info->base_type || a->payload) return false;
    const TypeInfo *outer[] = {a->info, b->info};
    for (size_t i = 0; i < 2; ++i)
        if (outer[i]->generic_name || outer[i]->opaque_type_name || outer[i]->fn_sig ||
            outer[i]->row_field_count || outer[i]->type_var_count || outer[i]->row_var_name || outer[i]->is_open_row ||
            (outer[i]->base_type == TYPE_ARRAY && (outer[i]->type_param_count || outer[i]->tuple_element_count)) ||
            (outer[i]->base_type == TYPE_TUPLE && outer[i]->element_type)) return false;
    size_t count = a->info->base_type == TYPE_ARRAY ? 1 : (size_t)a->info->tuple_element_count;
    if (a->info->base_type != TYPE_ARRAY && (a->info->base_type != TYPE_TUPLE ||
        !type_info_tuple_valid(a->info) || !type_info_tuple_valid(b->info) ||
        a->info->tuple_element_count != b->info->tuple_element_count)) return false;
    for (size_t i = 0; i < count; ++i) {
        NominalView left = {0}, right = {0};
        bool ok = nominal_view_child(env, a, i, depth + 1, &left) &&
            nominal_view_child(env, b, i, depth + 1, &right) && nominal_view_equal(env, &left, &right, depth + 1);
        nominal_view_discard(&left); nominal_view_discard(&right);
        if (!ok) return false;
    }
    return true;
}
static bool nominal_view_matches_annotation(Environment *env, const NominalView *view,
    const TypeInfo *info, const char *owner, const NominalSubstitution *context, unsigned depth) {
    if (!view || !view->info || depth > 128) return false;
    if (!view->children) return checked_annotations_equal_context(env, info, owner, view->info, view->owner,
        depth + 1, context, nominal_view_context(view));
    NominalView expected = {0};
    bool ok = nominal_view_copy_context(env, info, owner, context, depth + 1, &expected) &&
        nominal_view_equal(env, &expected, view, depth + 1);
    nominal_view_discard(&expected);
    return ok;
}

static bool nominal_view_matches_value(Environment *env, const NominalView *expected, ASTNode *value, unsigned depth) {
    if (!expected || !expected->info || !value || depth > 128) return false;
    if (expected->children && !nominal_view_composed_valid(expected)) return false;
    if (value->type == AST_ARRAY_LITERAL || value->type == AST_TUPLE_LITERAL) {
        const NominalView *retained = nominal_constructor_view(env, value);
        if (retained) return nominal_view_equal(env, expected, retained, depth + 1);
    }
    if (value->type == AST_BLOCK && value->as.block.count)
        return nominal_view_matches_value(env, expected, value->as.block.statements[value->as.block.count - 1], depth + 1);
    if (value->type == AST_RETURN) return nominal_view_matches_value(env, expected, value->as.return_stmt.value, depth + 1);
    if (value->type == AST_IF || value->type == AST_COND || value->type == AST_MATCH) {
        int count = value->type == AST_IF ? 2 : value->type == AST_MATCH
            ? value->as.match_expr.arm_count : value->as.cond_expr.clause_count + 1;
        bool surviving = false;
        for (int i = 0; i < count; ++i) {
            ASTNode *body = value->type == AST_IF ? (i ? value->as.if_stmt.else_branch : value->as.if_stmt.then_branch)
                : value->type == AST_MATCH ? value->as.match_expr.arm_bodies[i]
                : i < value->as.cond_expr.clause_count ? value->as.cond_expr.values[i] : value->as.cond_expr.else_value;
            if (ast_always_returns(body)) continue;
            surviving = true;
            if (!nominal_view_matches_value(env, expected, body, depth + 1)) return false;
        }
        return surviving;
    }
    bool array = value->type == AST_ARRAY_LITERAL && expected->info->base_type == TYPE_ARRAY;
    bool tuple = value->type == AST_TUPLE_LITERAL && expected->info->base_type == TYPE_TUPLE;
    if (!expected->payload && (array || tuple)) {
        int count = array ? value->as.array_literal.element_count : value->as.tuple_literal.element_count;
        if (count < 0 || (tuple && count != expected->info->tuple_element_count)) return false;
        for (int i = 0; i < count; ++i) {
            NominalView child = {0};
            ASTNode *element = array ? value->as.array_literal.elements[i] : value->as.tuple_literal.elements[i];
            bool ok = nominal_view_child(env, expected, array ? 0 : (size_t)i, depth + 1, &child) &&
                nominal_view_matches_value(env, &child, element, depth + 1);
            nominal_view_discard(&child);
            if (!ok) return false;
        }
        return true;
    }
    NominalView actual = {0};
    bool ok = nominal_value_view(value, env, depth + 1, &actual) && nominal_view_equal(env, expected, &actual, depth + 1);
    nominal_view_discard(&actual);
    return ok;
}

/* My emitted annotations are copies, not provenance authorities. */
static bool nominal_materialize(Environment *, const TypeInfo *, const char *,
    const NominalSubstitution *, unsigned, TypeInfo **);
static bool nominal_materialize_flat(Environment *env, Type original, const char *name,
    const char *owner, const NominalSubstitution *context, unsigned depth, Type *type, char **spelling) {
    TypeInfo source = {.base_type = original, .generic_name = (char *)name};
    TypeInfo *copy = NULL;
    if (!nominal_materialize(env, &source, owner, context, depth + 1, &copy)) return false;
    const char *list_element = copy->base_type == TYPE_LIST_GENERIC ? list_annotation_spelling(copy, NULL) : NULL;
    char *resolved = list_element ? strdup(list_element) : copy->generic_name ? typeinfo_to_generic_arg_name(copy) : NULL;
    if (copy->generic_name && !resolved) { free_payload_type_info(copy); return false; }
    *type = copy->base_type;
    free(*spelling);
    *spelling = resolved;
    free_payload_type_info(copy);
    return true;
}
/* I borrow a complete result view without transferring either source owner. */
static TypeInfo nominal_signature_result(const FunctionSignature *signature) {
    TypeInfo view = signature->return_type_info ? *signature->return_type_info
        : (TypeInfo){.base_type = signature->return_type, .generic_name = signature->return_struct_name};
    if (view.base_type == TYPE_FUNCTION && !view.fn_sig) view.fn_sig = signature->return_fn_sig;
    return view;
}
static bool nominal_materialize_signature(Environment *env, const FunctionSignature *source,
    FunctionSignature *copy, const char *owner, const NominalSubstitution *context, unsigned depth) {
    if (depth > 128 || !source || !copy) return false;
    for (int i = 0; i < source->param_count; ++i) {
        TypeInfo flat = {.base_type = source->param_types[i],
            .generic_name = source->param_struct_names ? source->param_struct_names[i] : NULL};
        const TypeInfo *annotation = source->param_type_info && source->param_type_info[i]
            ? source->param_type_info[i] : &flat;
        TypeInfo *concrete = NULL;
        if (!nominal_materialize(env, annotation, owner, context, depth + 1, &concrete)) return false;
        if (!copy->param_type_info) {
            copy->param_type_info = calloc((size_t)source->param_count, sizeof(TypeInfo *));
            if (!copy->param_type_info) { free_payload_type_info(concrete); return false; }
        }
        free_payload_type_info(copy->param_type_info[i]);
        copy->param_type_info[i] = concrete;
        copy->param_types[i] = concrete->base_type;
        const char *name = concrete->base_type == TYPE_LIST_GENERIC
            ? list_annotation_spelling(concrete, NULL) : concrete->generic_name;
        if (!copy->param_struct_names) {
            copy->param_struct_names = calloc((size_t)source->param_count, sizeof(char *));
            if (!copy->param_struct_names) return false;
        }
        free(copy->param_struct_names[i]);
        copy->param_struct_names[i] = name ? strdup(name) : NULL;
        if (name && !copy->param_struct_names[i]) return false;
    }
    TypeInfo result = nominal_signature_result(source);
    TypeInfo *concrete = NULL;
    if (!nominal_materialize(env, &result,
            owner, context, depth + 1, &concrete)) return false;
    free_payload_type_info(copy->return_type_info);
    copy->return_type_info = concrete;
    copy->return_type = concrete->base_type;
    const char *name = concrete->base_type == TYPE_LIST_GENERIC
        ? list_annotation_spelling(concrete, NULL) : concrete->generic_name;
    free(copy->return_struct_name);
    copy->return_struct_name = name ? strdup(name) : NULL;
    if (name && !copy->return_struct_name) return false;
    /* The complete result annotation is authoritative; the duplicate callable
     * ABI field must contain the same materialized signature when present. */
    if (concrete->base_type == TYPE_FUNCTION) {
        FunctionSignature *nested = NULL;
        if (!copy_function_signature_checked(concrete->fn_sig, &nested)) return false;
        free_function_signature(copy->return_fn_sig);
        copy->return_fn_sig = nested;
    }
    return true;
}
static bool nominal_materialize(Environment *env, const TypeInfo *source, const char *owner,
    const NominalSubstitution *context, unsigned depth, TypeInfo **out) {
    if (!source || depth > 128 || !nominal_substitute_annotation(&source, &owner, &context)) return false;
    if (!context) return copy_payload_type_info_checked(source, out);
    TypeInfo *copy = NULL;
    if (!copy_payload_type_info_checked(source, &copy)) return false;
    if (source->tuple_element_count && !copy->tuple_type_names) {
        copy->tuple_type_names = calloc((size_t)source->tuple_element_count, sizeof(char *));
        if (!copy->tuple_type_names) goto fail;
    }
    if (source->row_field_count && !copy->row_field_type_names) {
        copy->row_field_type_names = calloc((size_t)source->row_field_count, sizeof(char *));
        if (!copy->row_field_type_names) goto fail;
    }
    if (source->base_type == TYPE_LIST_GENERIC && !source->type_param_count) {
        TypeInfo element = {.base_type = TYPE_STRUCT, .generic_name = source->generic_name};
        copy->type_params = calloc(1, sizeof(TypeInfo *));
        if (!copy->type_params) goto fail;
        copy->type_param_count = 1;
        if (!nominal_materialize(env, &element, owner, context, depth + 1, &copy->type_params[0])) goto fail;
        free(copy->generic_name);
        copy->generic_name = strdup("List");
        if (!copy->generic_name) goto fail;
    }
    if (source->element_type) {
        TypeInfo *element = NULL;
        if (!nominal_materialize(env, source->element_type, owner, context, depth + 1, &element)) goto fail;
        free_payload_type_info(copy->element_type);
        copy->element_type = element;
    }
    for (int i = 0; i < source->type_param_count; ++i) {
        TypeInfo *parameter = NULL;
        if (!nominal_materialize(env, source->type_params[i], owner, context, depth + 1, &parameter)) goto fail;
        free_payload_type_info(copy->type_params[i]);
        copy->type_params[i] = parameter;
    }
    if (source->base_type == TYPE_TUPLE && source->type_param_count) {
        if (!type_info_tuple_valid(source) || !type_info_tuple_refresh(copy)) goto fail;
    } else for (int i = 0; i < source->tuple_element_count; ++i)
        if (!nominal_materialize_flat(env, source->tuple_types[i],
                source->tuple_type_names ? source->tuple_type_names[i] : NULL,
                owner, context, depth + 1, &copy->tuple_types[i], &copy->tuple_type_names[i])) goto fail;
    for (int i = 0; i < source->row_field_count; ++i)
        if (!nominal_materialize_flat(env, source->row_field_types[i],
                source->row_field_type_names ? source->row_field_type_names[i] : NULL,
                owner, context, depth + 1, &copy->row_field_types[i], &copy->row_field_type_names[i])) goto fail;
    if (source->fn_sig && !nominal_materialize_signature(env, source->fn_sig, copy->fn_sig,
                                                        owner, context, depth + 1)) goto fail;
    *out = copy;
    return true;
fail:
    free_payload_type_info(copy);
    return false;
}

static bool nominal_view_materialize(Environment *env, const NominalView *view, unsigned depth, TypeInfo **out) {
    if (!view || depth > 128) return false;
    if (!view->children) return nominal_materialize(env, view->info, view->owner, nominal_view_context(view), depth + 1, out);
    if (!nominal_view_composed_valid(view)) return false;
    TypeInfo *copy = NULL;
    if (!copy_payload_type_info_checked(view->info, &copy)) return false;
    for (size_t i = 0; i < view->child_count; ++i) {
        TypeInfo *child = NULL;
        if (!nominal_view_materialize(env, &view->children[i], depth + 1, &child)) goto fail;
        TypeInfo **slot = copy->base_type == TYPE_ARRAY ? &copy->element_type : &copy->type_params[i];
        free_payload_type_info(*slot); *slot = child;
    }
    if (copy->base_type == TYPE_TUPLE && !type_info_tuple_refresh(copy)) goto fail;
    *out = copy;
    return true;
fail:
    free_payload_type_info(copy);
    return false;
}

/* I publish the proof and emitted copy only after both have an owner. */
static bool nominal_view_retain(Environment *env, Symbol *binding, NominalView *view) {
    TypeInfo *concrete = NULL;
    if (!binding || !view->info || !nominal_view_materialize(env, view, 0, &concrete)) return false;
    NominalView *retained = calloc(1, sizeof *retained);
    if (!retained) { free_payload_type_info(concrete); return false; }
    if (!env_own_checker_type_info(env, concrete)) {
        free(retained); free_payload_type_info(concrete); return false;
    }
    if (!env_own_checker_object(env, retained, nominal_view_destroy)) {
        free(retained); return false; /* The concrete copy is registered, but unpublished. */
    }
    *retained = *view;
    memset(view, 0, sizeof *view);
    binding->type_info = concrete;
    binding->nominal_owner = retained->owner;
    binding->checker_nominal_view = retained;
    return true;
}

/* I retain constructor origins independently of mutable emitted annotations. */
typedef struct CheckerNominalExpression {
    const ASTNode *expression;
    NominalView view;
    struct CheckerNominalExpression *next;
} CheckerNominalExpression;
static const NominalView *nominal_constructor_view(Environment *env, const ASTNode *expression) {
    for (CheckerNominalExpression *entry = env->checker_nominal_expressions; entry; entry = entry->next)
        if (entry->expression == expression) return &entry->view;
    return NULL;
}
static void nominal_constructor_destroy(void *object) {
    CheckerNominalExpression *entry = object;
    nominal_view_discard(&entry->view);
    free(entry);
}
/* I retain checked literal conversions as complete proofs, independently of
 * flattened native emission rows. A different destination cannot replace one. */
static bool nominal_literal_retain(Environment *env, const ASTNode *expression,
                                  const NominalView *view, unsigned depth) {
    if (!view || !view->info || view->payload ||
        (view->info->base_type != TYPE_ARRAY && view->info->base_type != TYPE_TUPLE)) return false;
    const NominalView *prior = nominal_constructor_view(env, expression);
    if (prior) return nominal_view_equal(env, prior, view, depth + 1);
    CheckerNominalExpression *entry = calloc(1, sizeof *entry);
    if (!entry) return false;
    if (!nominal_view_clone(env, view, depth + 1, &entry->view) ||
        !env_own_checker_object(env, entry, nominal_constructor_destroy)) {
        nominal_constructor_destroy(entry);
        return false;
    }
    entry->expression = expression;
    entry->next = env->checker_nominal_expressions;
    env->checker_nominal_expressions = entry;
    return true;
}
static bool nominal_constructor_retain(Environment *env, const ASTNode *expression,
    const TypeInfo *info, const char *owner, const NominalSubstitution *context, unsigned depth) {
    const NominalView *prior = nominal_constructor_view(env, expression);
    if (prior) return checked_annotations_equal_context(env, info, owner, prior->info, prior->owner,
        depth + 1, context, nominal_view_context(prior));
    CheckerNominalExpression *entry = calloc(1, sizeof *entry);
    if (!entry) return false;
    if (!nominal_view_copy_context(env, info, owner, context, depth + 1, &entry->view) ||
        entry->view.info->base_type != TYPE_UNION || entry->view.payload ||
        !env_own_checker_object(env, entry, nominal_constructor_destroy)) {
        nominal_constructor_destroy(entry);
        return false;
    }
    entry->expression = expression;
    entry->next = env->checker_nominal_expressions;
    env->checker_nominal_expressions = entry;
    return true;
}
