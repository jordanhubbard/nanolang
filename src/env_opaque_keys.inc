/* I frame complete opaque-bearing annotations independently of C spelling. */
typedef struct { char *data; size_t size, capacity; bool ok; } OpaqueKeyBuffer;
static void opaque_key_bytes(OpaqueKeyBuffer *b, const char *data, size_t size) {
    if (!b->ok || size > SIZE_MAX - b->size - 1) { b->ok = false; return; }
    size_t needed = b->size + size + 1;
    if (needed > b->capacity) {
        size_t capacity = b->capacity ? b->capacity : 128;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2) { capacity = needed; break; }
            capacity *= 2;
        }
        char *next = realloc(b->data, capacity);
        if (!next) { b->ok = false; return; }
        b->data = next; b->capacity = capacity;
    }
    memcpy(b->data + b->size, data, size); b->size += size; b->data[b->size] = 0;
}
static void opaque_key_number(OpaqueKeyBuffer *b, size_t value) {
    char number[3 * sizeof(size_t) + 3];
    int count = snprintf(number, sizeof number, "%zu:", value);
    if (count < 0 || (size_t)count >= sizeof number) { b->ok = false; return; }
    opaque_key_bytes(b, number, (size_t)count);
}
static void opaque_key_string(OpaqueKeyBuffer *b, const char *value) {
    if (!value) { opaque_key_bytes(b, "-", 1); return; }
    size_t size = strlen(value);
    opaque_key_bytes(b, "+", 1); opaque_key_number(b, size);
    opaque_key_bytes(b, value, size);
}
static bool opaque_key_leaf(const char *name) {
    return name && !strncmp(name, "#opaque:", 8);
}
static bool opaque_info_present_depth(const TypeInfo *, unsigned);
static bool opaque_signature_present_depth(const FunctionSignature *s, unsigned depth) {
    if (!s) return false;
    if (depth > 128) return true;
    if (opaque_key_leaf(s->return_struct_name) ||
        opaque_info_present_depth(s->return_type_info, depth + 1) ||
        opaque_signature_present_depth(s->return_fn_sig, depth + 1)) return true;
    for (int i = 0; i < s->param_count; ++i)
        if ((s->param_struct_names && opaque_key_leaf(s->param_struct_names[i])) ||
            (s->param_type_info && opaque_info_present_depth(s->param_type_info[i], depth + 1))) return true;
    return false;
}
static bool opaque_info_present_depth(const TypeInfo *t, unsigned depth) {
    if (!t) return false;
    if (depth > 128) return true;
    if (opaque_key_leaf(t->generic_name) || opaque_key_leaf(t->opaque_type_name) ||
        opaque_info_present_depth(t->element_type, depth + 1) ||
        opaque_signature_present_depth(t->fn_sig, depth + 1)) return true;
    for (int i = 0; t->type_params && i < t->type_param_count; ++i)
        if (opaque_info_present_depth(t->type_params[i], depth + 1)) return true;
    for (int i = 0; t->tuple_type_names && i < t->tuple_element_count; ++i)
        if (opaque_key_leaf(t->tuple_type_names[i])) return true;
    for (int i = 0; t->row_field_type_names && i < t->row_field_count; ++i)
        if (opaque_key_leaf(t->row_field_type_names[i])) return true;
    return false;
}
bool opaque_type_info_present(const TypeInfo *info) { return opaque_info_present_depth(info, 0); }
static void opaque_encode_info(OpaqueKeyBuffer *, const TypeInfo *, unsigned);
static void opaque_encode_signature(OpaqueKeyBuffer *b, const FunctionSignature *s, unsigned depth) {
    if (!s) { opaque_key_bytes(b, "-", 1); return; }
    if (depth > 128 || s->param_count < 0 || (s->param_count && !s->param_types)) { b->ok = false; return; }
    opaque_key_bytes(b, "F", 1); opaque_key_number(b, (size_t)s->param_count);
    for (int i = 0; i < s->param_count; ++i) {
        TypeInfo view = {.base_type = s->param_types[i],
            .generic_name = s->param_struct_names ? s->param_struct_names[i] : NULL};
        opaque_encode_info(b, s->param_type_info && s->param_type_info[i] ? s->param_type_info[i] : &view, depth + 1);
    }
    TypeInfo result = {.base_type = s->return_type, .generic_name = s->return_struct_name,
        .fn_sig = s->return_fn_sig};
    if (s->return_type_info) result = *s->return_type_info;
    if (!result.fn_sig) result.fn_sig = s->return_fn_sig;
    opaque_encode_info(b, &result, depth + 1);
}
static void opaque_encode_info(OpaqueKeyBuffer *b, const TypeInfo *t, unsigned depth) {
    if (!t) { opaque_key_bytes(b, "-", 1); return; }
    if (depth > 128 || t->base_type < 0 || t->type_param_count < 0 || t->tuple_element_count < 0 ||
        t->row_field_count < 0 || t->type_var_count < 0 ||
        (t->type_param_count && !t->type_params) || (t->tuple_element_count && !t->tuple_types) ||
        (t->row_field_count && (!t->row_field_types || !t->row_field_names)) ||
        (t->type_var_count && !t->type_var_names)) { b->ok = false; return; }
    if (t->base_type == TYPE_TUPLE && !type_info_tuple_valid(t)) { b->ok = false; return; }
    const char *opaque = opaque_key_leaf(t->opaque_type_name) ? t->opaque_type_name :
        (opaque_key_leaf(t->generic_name) ? t->generic_name : NULL);
    if (opaque) {
        if ((t->base_type != TYPE_STRUCT && t->base_type != TYPE_OPAQUE) ||
            t->type_param_count || t->element_type || t->fn_sig || t->tuple_element_count ||
            t->row_field_count || t->type_var_count || t->is_open_row) { b->ok = false; return; }
        opaque_key_bytes(b, "O", 1); opaque_key_string(b, opaque); return;
    }
    opaque_key_bytes(b, "T", 1); opaque_key_number(b, (size_t)t->base_type);
    opaque_key_string(b, t->generic_name); opaque_key_string(b, t->opaque_type_name);
    opaque_encode_info(b, t->element_type, depth + 1);
    opaque_key_number(b, t->base_type == TYPE_TUPLE ? 0 : (size_t)t->type_param_count);
    for (int i = 0; t->base_type != TYPE_TUPLE && i < t->type_param_count; ++i) {
        if (!t->type_params[i]) { b->ok = false; return; }
        opaque_encode_info(b, t->type_params[i], depth + 1);
    }
    opaque_key_number(b, (size_t)t->tuple_element_count);
    for (int i = 0; i < t->tuple_element_count; ++i) {
        TypeInfo item = {.base_type = t->tuple_types[i],
            .generic_name = t->tuple_type_names ? t->tuple_type_names[i] : NULL};
        const TypeInfo *child = t->base_type == TYPE_TUPLE ? type_info_tuple_element(t, i, &item) : &item;
        if (!child) { b->ok = false; return; }
        opaque_encode_info(b, child, depth + 1);
    }
    opaque_encode_signature(b, t->fn_sig, depth + 1);
    opaque_key_number(b, t->is_open_row ? 1 : 0); opaque_key_string(b, t->row_var_name);
    opaque_key_number(b, (size_t)t->row_field_count);
    for (int i = 0; i < t->row_field_count; ++i) {
        opaque_key_string(b, t->row_field_names[i]);
        TypeInfo field = {.base_type = t->row_field_types[i],
            .generic_name = t->row_field_type_names ? t->row_field_type_names[i] : NULL};
        opaque_encode_info(b, &field, depth + 1);
    }
    opaque_key_number(b, (size_t)t->type_var_count);
    for (int i = 0; i < t->type_var_count; ++i) opaque_key_string(b, t->type_var_names[i]);
}
char *opaque_type_info_key(const TypeInfo *info) {
    OpaqueKeyBuffer b = {.ok = true};
    opaque_key_bytes(&b, "#otype:", 7); opaque_encode_info(&b, info, 0);
    if (!b.ok) { free(b.data); return NULL; }
    return b.data;
}

/* I reserve only canonical decimal prefix indices, before native emission. */
bool env_reserve_opaque_symbol_prefix(Environment *env, const char *name) {
    static const char prefix[] = "__nano_opaque_";
    const char *start = name ? strstr(name, prefix) : NULL;
    if (!start) return true;
    const unsigned char *p = (const unsigned char *)start + sizeof prefix - 1;
    if (*p < '0' || *p > '9' || (*p == '0' && p[1] != '_')) return true;
    size_t number = 0;
    while (*p >= '0' && *p <= '9') {
        size_t digit = (size_t)(*p++ - '0');
        if (number > (SIZE_MAX - digit) / 10) return true;
        number = number * 10 + digit;
    }
    if (*p != '_') return true;
    for (size_t i = 0; i < env->opaque_reserved_count; ++i)
        if (env->opaque_reserved_indices[i] == number) return true;
    if (env->opaque_reserved_count >= SIZE_MAX / sizeof(size_t)) return false;
    size_t *rows = realloc(env->opaque_reserved_indices,
        (env->opaque_reserved_count + 1) * sizeof(size_t));
    if (!rows) return false;
    rows[env->opaque_reserved_count++] = number; env->opaque_reserved_indices = rows;
    return true;
}
size_t env_opaque_symbol_prefix(const Environment *env) {
    size_t candidate = 0;
    for (;;) {
        bool present = false;
        for (size_t i = 0; i < env->opaque_reserved_count; ++i)
            if (env->opaque_reserved_indices[i] == candidate) { present = true; break; }
        if (!present) return candidate;
        ++candidate; /* At most count distinct stored indices precede the gap. */
    }
}

static char *opaque_generic_argument_name(TypeInfo *);
char *typeinfo_to_generic_arg_name(TypeInfo *param) {
    if (!param) return strdup("unknown");
    if (opaque_type_info_present(param) && !((param->base_type == TYPE_STRUCT || param->base_type == TYPE_OPAQUE) && !param->type_param_count && !param->element_type && !param->fn_sig))
        return opaque_type_info_key(param);
    if (param->generic_name && param->type_param_count > 0)
        return opaque_generic_argument_name(param);

    switch (param->base_type) {
        case TYPE_INT:
            return strdup("int");
        case TYPE_U8:
            return strdup("u8");
        case TYPE_STRING:
            return strdup("string");
        case TYPE_BOOL:
            return strdup("bool");
        case TYPE_FLOAT:
            return strdup("float");
        case TYPE_OPAQUE:
        case TYPE_STRUCT:
        case TYPE_UNION:
        case TYPE_ENUM:
            if (param->generic_name) return strdup(param->generic_name);
            return strdup("unknown");
        case TYPE_ARRAY: {
            char *elem = typeinfo_to_generic_arg_name(param->element_type);
            if (!elem) return strdup("array_unknown");
            size_t n = strlen(elem) + 7;
            char *out = malloc(n);
            if (!out) {
                free(elem);
                return strdup("array_unknown");
            }
            snprintf(out, n, "array_%s", elem);
            free(elem);
            return out;
        }
        default:
            return strdup("unknown");
    }
}

static char *opaque_generic_argument_name(TypeInfo *info) {
    if (!info || !info->generic_name) return NULL;
    if (opaque_type_info_present(info)) return opaque_type_info_key(info);
    if (info->type_param_count <= 0) return strdup(info->generic_name);

    size_t cap = 256;
    char *out = malloc(cap);
    if (!out) return NULL;
    out[0] = '\0';

    snprintf(out, cap, "%s", info->generic_name);
    for (int i = 0; i < info->type_param_count; i++) {
        char *arg = typeinfo_to_generic_arg_name(info->type_params[i]);
        if (!arg) arg = strdup("unknown");
        size_t need = strlen(out) + 1 + strlen(arg) + 1;
        if (need > cap) {
            cap = need * 2;
            char *bigger = realloc(out, cap);
            if (!bigger) {
                free(arg);
                free(out);
                return NULL;
            }
            out = bigger;
        }
        strcat(out, "_");
        strcat(out, arg);
        free(arg);
    }

    return out;
}

/* I copy complete trees before publishing a new opaque-bearing instance.
 * My existing destructors accept each zero-initialized partial tree. */
static void *opaque_copy_array(const void *source, int count, size_t width, bool *ok) {
    if (!source || !*ok) return NULL;
    if (count < 0 || (size_t)count > SIZE_MAX / width) { *ok = false; return NULL; }
    void *copy = calloc(count ? (size_t)count : 1, width);
    if (!copy) { *ok = false; return NULL; }
    memcpy(copy, source, (size_t)count * width); return copy;
}
static char *opaque_copy_name(const char *source, bool *ok) {
    if (!source || !*ok) return NULL;
    char *copy = strdup(source);
    if (!copy) *ok = false;
    return copy;
}
static char **opaque_copy_names(char *const *source, int count, bool *ok) {
    if (!source || !*ok) return NULL;
    if (count < 0 || (size_t)count > SIZE_MAX / sizeof(char *)) { *ok = false; return NULL; }
    char **copy = calloc(count ? (size_t)count : 1, sizeof(char *));
    if (!copy) { *ok = false; return NULL; }
    for (int i = 0; *ok && i < count; ++i) copy[i] = opaque_copy_name(source[i], ok);
    return copy;
}
static TypeInfo *opaque_copy_info(const TypeInfo *, unsigned, bool *);
static FunctionSignature *opaque_copy_signature(const FunctionSignature *source, unsigned depth, bool *ok) {
    if (!source || !*ok) return NULL;
    if (depth > 128 || source->param_count < 0 ||
        (source->param_count && !source->param_types)) { *ok = false; return NULL; }
    FunctionSignature *copy = calloc(1, sizeof *copy);
    if (!copy) { *ok = false; return NULL; }
    copy->param_count = source->param_count; copy->return_type = source->return_type;
    copy->param_types = opaque_copy_array(source->param_types, source->param_count, sizeof(Type), ok);
    copy->param_struct_names = opaque_copy_names(source->param_struct_names, source->param_count, ok);
    copy->return_struct_name = opaque_copy_name(source->return_struct_name, ok);
    if (*ok && source->param_type_info) {
        if ((size_t)source->param_count > SIZE_MAX / sizeof(TypeInfo *)) *ok = false;
        else copy->param_type_info = calloc(source->param_count ? (size_t)source->param_count : 1, sizeof(TypeInfo *));
        if (!copy->param_type_info) *ok = false;
        for (int i = 0; *ok && i < source->param_count; ++i)
            copy->param_type_info[i] = opaque_copy_info(source->param_type_info[i], depth + 1, ok);
    }
    copy->return_type_info = opaque_copy_info(source->return_type_info, depth + 1, ok);
    copy->return_fn_sig = opaque_copy_signature(source->return_fn_sig, depth + 1, ok);
    if (!*ok) { free_function_signature(copy); return NULL; }
    return copy;
}
static TypeInfo *opaque_copy_info(const TypeInfo *source, unsigned depth, bool *ok) {
    if (!source || !*ok) return NULL;
    if (depth > 128 || source->type_param_count < 0 || source->tuple_element_count < 0 ||
        source->row_field_count < 0 || source->type_var_count < 0) { *ok = false; return NULL; }
    TypeInfo *copy = calloc(1, sizeof *copy);
    if (!copy) { *ok = false; return NULL; }
    copy->base_type = source->base_type; copy->is_open_row = source->is_open_row;
    copy->type_param_count = source->type_param_count;
    copy->tuple_element_count = source->tuple_element_count;
    copy->row_field_count = source->row_field_count; copy->type_var_count = source->type_var_count;
    copy->generic_name = opaque_copy_name(source->generic_name, ok);
    copy->opaque_type_name = opaque_copy_name(source->opaque_type_name, ok);
    copy->element_type = opaque_copy_info(source->element_type, depth + 1, ok);
    if (*ok && source->type_params) {
        if ((size_t)source->type_param_count > SIZE_MAX / sizeof(TypeInfo *)) *ok = false;
        else copy->type_params = calloc(source->type_param_count ? (size_t)source->type_param_count : 1, sizeof(TypeInfo *));
        if (!copy->type_params) *ok = false;
        for (int i = 0; *ok && i < source->type_param_count; ++i)
            copy->type_params[i] = opaque_copy_info(source->type_params[i], depth + 1, ok);
    }
    copy->tuple_types = opaque_copy_array(source->tuple_types, source->tuple_element_count, sizeof(Type), ok);
    copy->tuple_type_names = opaque_copy_names(source->tuple_type_names, source->tuple_element_count, ok);
    copy->fn_sig = opaque_copy_signature(source->fn_sig, depth + 1, ok);
    copy->row_var_name = opaque_copy_name(source->row_var_name, ok);
    copy->row_field_names = opaque_copy_names(source->row_field_names, source->row_field_count, ok);
    copy->row_field_types = opaque_copy_array(source->row_field_types, source->row_field_count, sizeof(Type), ok);
    copy->row_field_type_names = opaque_copy_names(source->row_field_type_names, source->row_field_count, ok);
    copy->type_var_names = opaque_copy_names(source->type_var_names, source->type_var_count, ok);
    if (!*ok) { free_payload_type_info(copy); return NULL; }
    return copy;
}
TypeInfo *copy_complete_type_info_checked(const TypeInfo *source) {
    if (!source) return NULL;
    char *validated = opaque_type_info_key(source);
    if (!validated) return NULL;
    free(validated);
    bool ok = true;
    return opaque_copy_info(source, 0, &ok);
}
static void opaque_free_instance(GenericInstantiation *inst) {
    free(inst->generic_name); free(inst->concrete_name); free(inst->type_args);
    for (int i = 0; inst->type_arg_names && i < inst->type_arg_count; ++i) free(inst->type_arg_names[i]);
    free(inst->type_arg_names); free_payload_type_info(inst->type_info);
}
bool env_register_opaque_union_context(Environment *env, const TypeInfo *info, bool *added) {
    if (added) *added = false;
    if (!env || !info || !info->generic_name || info->type_param_count <= 0 ||
        !opaque_type_info_present(info)) return false;
    char *key = opaque_type_info_key(info);
    if (!key) return false;
    for (int i = 0; i < env->generic_instance_count; ++i) {
        const GenericInstantiation *old = &env->generic_instances[i];
        if (old->concrete_name && !strcmp(old->concrete_name, key)) { free(key); return true; }
    }
    bool ok = true;
    GenericInstantiation staged = {.concrete_name = key, .type_arg_count = info->type_param_count};
    staged.generic_name = opaque_copy_name(info->generic_name, &ok);
    staged.type_info = opaque_copy_info(info, 0, &ok);
    if ((size_t)info->type_param_count > SIZE_MAX / sizeof(Type) ||
        (size_t)info->type_param_count > SIZE_MAX / sizeof(char *)) ok = false;
    if (ok) {
        staged.type_args = calloc((size_t)info->type_param_count, sizeof(Type));
        staged.type_arg_names = calloc((size_t)info->type_param_count, sizeof(char *));
        if (!staged.type_args || !staged.type_arg_names) ok = false;
    }
    for (int i = 0; ok && i < info->type_param_count; ++i) {
        staged.type_args[i] = info->type_params[i]->base_type;
        staged.type_arg_names[i] = typeinfo_to_generic_arg_name(info->type_params[i]);
        if (!staged.type_arg_names[i]) ok = false;
    }
    if (env->generic_instance_count < 0 || env->generic_instance_count == INT_MAX ||
        (size_t)env->generic_instance_count + 1 > SIZE_MAX / sizeof(GenericInstantiation)) ok = false;
    GenericInstantiation *rows = ok ? malloc(((size_t)env->generic_instance_count + 1) * sizeof *rows) : NULL;
    if (!rows) { opaque_free_instance(&staged); return false; }
    memcpy(rows, env->generic_instances, (size_t)env->generic_instance_count * sizeof *rows);
    rows[env->generic_instance_count] = staged;
    free(env->generic_instances); env->generic_instances = rows;
    ++env->generic_instance_count; env->generic_instance_capacity = env->generic_instance_count;
    if (added) *added = true;
    return true;
}

const TypeInfo *env_tuple_literal_info(const Environment *env, const ASTNode *literal) {
    if (!env || !literal) return NULL;
    for (size_t i = 0; i < env->tuple_literal_binding_count; ++i)
        if (env->tuple_literal_bindings[i].literal == literal)
            return env->tuple_literal_bindings[i].type_info;
    return NULL;
}
bool env_bind_tuple_literal(Environment *env, const ASTNode *literal, const TypeInfo *info) {
    if (!env || !literal || literal->type != AST_TUPLE_LITERAL || !type_info_tuple_valid(info) ||
        info->tuple_element_count != literal->as.tuple_literal.element_count) return false;
    const TypeInfo *old = env_tuple_literal_info(env, literal);
    if (old) return type_infos_equal(old, info);
    if (env->tuple_literal_binding_count >= SIZE_MAX / sizeof(TupleLiteralBinding)) return false;
    TypeInfo *copy = copy_complete_type_info_checked(info);
    if (!copy) return false;
    size_t count = env->tuple_literal_binding_count;
    TupleLiteralBinding *rows = malloc((count + 1) * sizeof *rows);
    if (!rows) { free_payload_type_info(copy); return false; }
    if (count) memcpy(rows, env->tuple_literal_bindings, count * sizeof *rows);
    rows[count] = (TupleLiteralBinding){literal, copy};
    free(env->tuple_literal_bindings); env->tuple_literal_bindings = rows;
    ++env->tuple_literal_binding_count;
    return true;
}

/* I retain array contexts separately from AST storage and borrowed symbol rows. */
const TypeInfo *env_array_expression_info(const Environment *env, const ASTNode *expression) {
    if (!env || !expression) return NULL;
    for (size_t i = 0; i < env->array_expression_binding_count; ++i)
        if (env->array_expression_bindings[i].literal == expression)
            return env->array_expression_bindings[i].type_info;
    return NULL;
}
bool env_bind_array_expression(Environment *env, const ASTNode *expression, const TypeInfo *info) {
    if (!env || !expression || !info || info->base_type != TYPE_ARRAY || !info->element_type) return false;
    const TypeInfo *old = env_array_expression_info(env, expression);
    if (old) return type_infos_equal(old, info);
    size_t count = env->array_expression_binding_count;
    if (count >= SIZE_MAX / sizeof(TupleLiteralBinding)) return false;
    TypeInfo *copy = copy_complete_type_info_checked(info);
    if (!copy) return false;
    TupleLiteralBinding *rows = malloc((count + 1) * sizeof *rows);
    if (!rows) { free_payload_type_info(copy); return false; }
    if (count) memcpy(rows, env->array_expression_bindings, count * sizeof *rows);
    rows[count] = (TupleLiteralBinding){expression, copy};
    free(env->array_expression_bindings); env->array_expression_bindings = rows;
    ++env->array_expression_binding_count;
    return true;
}

bool type_info_exact_array_element(const TypeInfo *info) {
    return info && info->base_type != TYPE_ARRAY &&
        (info->base_type == TYPE_TUPLE || info->base_type == TYPE_FUNCTION ||
         info->base_type == TYPE_UNION || opaque_type_info_present(info));
}
bool type_info_needs_array_context(const TypeInfo *info) {
    unsigned depth = 0;
    while (info && info->base_type == TYPE_ARRAY && depth++ <= 128) info = info->element_type;
    return type_info_exact_array_element(info);
}
