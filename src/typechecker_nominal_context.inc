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
static bool nominal_view_clone(Environment *env, const NominalView *source, unsigned depth, NominalView *out) {
    if (!nominal_view_copy_context(env, source->info, source->owner, nominal_view_context(source), depth + 1, out))
        return false;
    out->payload = source->payload;
    out->variant = source->variant;
    return true;
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
    for (int i = 0; i < source->tuple_element_count; ++i)
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

/* I publish the proof and emitted copy only after both have an owner. */
static bool nominal_view_retain(Environment *env, Symbol *binding, NominalView *view) {
    TypeInfo *concrete = NULL;
    if (!binding || !view->info || !nominal_materialize(env, view->info, view->owner,
            nominal_view_context(view), 0, &concrete)) return false;
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
