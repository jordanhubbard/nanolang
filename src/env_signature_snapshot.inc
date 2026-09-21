#ifdef NANO_EVALUATOR_LIFETIME_TIMING
#if defined(calloc) || defined(strdup)
#error I require unwrapped annotation allocators for this timing domain.
#endif
static void *lifetime_annotation_calloc(size_t count, size_t size) {
    LIFE_COUNT(LIFE_CHECKED_METADATA_ALLOC); return calloc(count, size);
}
static char *lifetime_annotation_strdup(const char *text) {
    LIFE_COUNT(LIFE_CHECKED_METADATA_ALLOC); return strdup(text);
}
#define calloc lifetime_annotation_calloc
#define strdup lifetime_annotation_strdup
#endif
/* I copy task callable metadata with checked rollback. My legacy compiler
 * metadata copier keeps its separate fatal allocation policy. */
static void *snapshot_array(size_t count, size_t size) {
    if (count && size > SIZE_MAX / count) return NULL;
    return calloc(count ? count : 1, size);
}
static bool snapshot_name(const char *source, char **out) {
    if (!source) { *out = NULL; return true; }
    char *copy = strdup(source);
    if (!copy) return false;
    *out = copy;
    return true;
}
static bool snapshot_names(char *const *source, int count, char ***out) {
    if (count < 0) return false;
    if (!source) { *out = NULL; return true; }
    char **copy = snapshot_array((size_t)count, sizeof(*copy));
    if (!copy) return false;
    for (int i = 0; i < count; ++i) {
        if (!snapshot_name(source[i], &copy[i])) {
            payload_free_names(copy, count);
            return false;
        }
    }
    *out = copy;
    return true;
}
static bool snapshot_type(const TypeInfo *, TypeInfo **, unsigned);
static bool snapshot_signature(const FunctionSignature *source,
                               FunctionSignature **out, unsigned depth) {
    if (!source) { *out = NULL; return true; }
    if (depth >= 128 || source->param_count < 0) return false;
    FunctionSignature *copy = calloc(1, sizeof(*copy));
    if (!copy) return false;
    copy->param_count = source->param_count;
    copy->return_type = source->return_type;
    if (source->param_types) {
        copy->param_types = snapshot_array((size_t)source->param_count, sizeof(Type));
        if (!copy->param_types) goto fail;
        memcpy(copy->param_types, source->param_types, (size_t)source->param_count * sizeof(Type));
    }
    if (!snapshot_names(source->param_struct_names, source->param_count, &copy->param_struct_names) ||
        !snapshot_name(source->return_struct_name, &copy->return_struct_name) ||
        !snapshot_signature(source->return_fn_sig, &copy->return_fn_sig, depth + 1)) goto fail;
    if (source->param_type_info) {
        copy->param_type_info = snapshot_array((size_t)source->param_count, sizeof(TypeInfo *));
        if (!copy->param_type_info) goto fail;
        for (int i = 0; i < source->param_count; ++i)
            if (!snapshot_type(source->param_type_info[i], &copy->param_type_info[i], depth + 1)) goto fail;
    }
    if (!snapshot_type(source->return_type_info, &copy->return_type_info, depth + 1)) goto fail;
    *out = copy;
    return true;
fail:
    free_function_signature(copy);
    return false;
}
static bool snapshot_type(const TypeInfo *source, TypeInfo **out, unsigned depth) {
    if (!source) { *out = NULL; return true; }
    if (depth >= 128 || source->type_param_count < 0 || source->tuple_element_count < 0 ||
        source->row_field_count < 0 || source->type_var_count < 0) return false;
    TypeInfo *copy = calloc(1, sizeof(*copy));
    if (!copy) return false;
    copy->base_type = source->base_type;
    copy->type_param_count = source->type_param_count;
    copy->tuple_element_count = source->tuple_element_count;
    copy->is_open_row = source->is_open_row;
    copy->row_field_count = source->row_field_count;
    copy->type_var_count = source->type_var_count;
    if (!snapshot_name(source->generic_name, &copy->generic_name)) goto fail;
    if (!snapshot_name(source->opaque_type_name, &copy->opaque_type_name)) goto fail;
    if (!snapshot_name(source->row_var_name, &copy->row_var_name)) goto fail;
    if (!snapshot_type(source->element_type, &copy->element_type, depth + 1)) goto fail;
    if (source->type_params) {
        copy->type_params = snapshot_array((size_t)source->type_param_count, sizeof(TypeInfo *));
        if (!copy->type_params) goto fail;
        for (int i = 0; i < source->type_param_count; ++i)
            if (!snapshot_type(source->type_params[i], &copy->type_params[i], depth + 1)) goto fail;
    }
    if (source->tuple_types) {
        copy->tuple_types = snapshot_array((size_t)source->tuple_element_count, sizeof(Type));
        if (!copy->tuple_types) goto fail;
        memcpy(copy->tuple_types, source->tuple_types, (size_t)source->tuple_element_count * sizeof(Type));
    }
    if (source->row_field_types) {
        copy->row_field_types = snapshot_array((size_t)source->row_field_count, sizeof(Type));
        if (!copy->row_field_types) goto fail;
        memcpy(copy->row_field_types, source->row_field_types, (size_t)source->row_field_count * sizeof(Type));
    }
    if (!snapshot_names(source->tuple_type_names, source->tuple_element_count, &copy->tuple_type_names)) goto fail;
    if (!snapshot_names(source->row_field_names, source->row_field_count, &copy->row_field_names)) goto fail;
    if (!snapshot_names(source->row_field_type_names, source->row_field_count, &copy->row_field_type_names)) goto fail;
    if (!snapshot_names(source->type_var_names, source->type_var_count, &copy->type_var_names)) goto fail;
    if (!snapshot_signature(source->fn_sig, &copy->fn_sig, depth + 1)) goto fail;
    *out = copy;
    return true;
fail:
    free_payload_type_info(copy);
    return false;
}
bool copy_function_signature_checked(const FunctionSignature *source, FunctionSignature **out) {
    if (!out) return false;
    return snapshot_signature(source, out, 0);
}

bool copy_payload_type_info_checked(const TypeInfo *source, TypeInfo **out) {
    if (!out) return false;
    return snapshot_type(source, out, 0);
}

#ifdef NANO_EVALUATOR_LIFETIME_TIMING
#undef calloc
#undef strdup
#endif
