/* Only evaluator constructors publish roots here. Aliases never transfer ownership. */
struct EnvUnionRoot {
    struct EnvUnionRoot *next;
    UnionValue value;
};

static void env_union_root_discard(struct EnvUnionRoot *root) {
    UnionValue *value = &root->value;
    for (int i = 0; i < value->field_count; ++i) {
        free(value->field_names[i]);
        env_discard_value_snapshot(value->field_values[i]);
    }
    free(value->field_names);
    free(value->field_values);
    free(value->union_name);
    free(value->variant_name);
    free(root);
}

bool env_union_result_borrowed(Environment *env, Value value) {
    if (!env || value.type != VAL_UNION || !value.as.union_val) return false;
    for (struct EnvUnionRoot *root = env->union_roots; root; root = root->next)
        if (&root->value == value.as.union_val) return true;
    return false;
}

bool env_create_union(Environment *env, const char *name, int variant,
                      const char *variant_name, char **names, Value *values,
                      int count, Value *out) {
    if (!env || !out || !name || !variant_name || count < 0 ||
        (count && (!names || !values)) ||
        (size_t)count > SIZE_MAX / sizeof(char *) ||
        (size_t)count > SIZE_MAX / sizeof(Value)) return false;
    struct EnvUnionRoot *root = calloc(1, sizeof(*root));
    if (!root) return false;
    UnionValue *value = &root->value;
    value->union_name = strdup(name);
    value->variant_name = strdup(variant_name);
    value->variant_index = variant;
    if (!value->union_name || !value->variant_name) goto fail;
    if (count) {
        value->field_names = calloc((size_t)count, sizeof(char *));
        value->field_values = calloc((size_t)count, sizeof(Value));
        if (!value->field_names || !value->field_values) goto fail;
    }
    for (int i = 0; i < count; ++i) {
        if (!names[i]) goto fail;
        value->field_names[i] = strdup(names[i]);
        if (!value->field_names[i]) goto fail;
        value->field_values[i] = create_void();
        value->field_count = i + 1;
        if (!env_clone_value_snapshot(values[i], &value->field_values[i])) goto fail;
    }
    Value result = create_void();
    result.type = VAL_UNION;
    result.as.union_val = value;
    root->next = env->union_roots;
    env->union_roots = root;
    *out = result;
    return true;
fail:
    env_union_root_discard(root);
    return false;
}

static void env_union_storage_free(Environment *env) {
    while (env->union_roots) {
        struct EnvUnionRoot *root = env->union_roots;
        env->union_roots = root->next;
        env_union_root_discard(root);
    }
}
