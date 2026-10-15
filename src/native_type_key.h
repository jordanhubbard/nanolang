#ifndef NANOLANG_NATIVE_TYPE_KEY_H
#define NANOLANG_NATIVE_TYPE_KEY_H

/* I encode complete tuple/callable arguments with counted fields and explicit
 * type boundaries. Canonical nominal names remain distinct from this prefix. */
typedef struct { char *text; size_t length, capacity; } NativeTypeKey;

static bool native_key_append(NativeTypeKey *key, const char *text) {
    size_t length = strlen(text);
    if (length > SIZE_MAX - key->length - 1) return false;
    size_t need = key->length + length + 1;
    if (need > key->capacity) {
        size_t capacity = need <= SIZE_MAX / 2 ? need * 2 : need;
        char *grown = realloc(key->text, capacity);
        if (!grown) return false;
        key->text = grown; key->capacity = capacity;
    }
    memcpy(key->text + key->length, text, length + 1);
    key->length += length;
    return true;
}

static bool native_key_number(NativeTypeKey *key, char tag, size_t number) {
    char text[64];
    snprintf(text, sizeof text, "%c%zu_", tag, number);
    return native_key_append(key, text);
}

static bool native_key_name(NativeTypeKey *key, const char *name) {
    if (!name) return native_key_append(key, "n_");
    if (!native_key_number(key, 'n', strlen(name))) return false;
    static const char hex[] = "0123456789abcdef";
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        char byte[3] = {hex[*p >> 4], hex[*p & 15], 0};
        if (!native_key_append(key, byte)) return false;
    }
    return native_key_append(key, "_");
}

static bool native_key_type(NativeTypeKey *key, const TypeInfo *info, unsigned depth) {
    if (!info) return native_key_append(key, "x");
    if (depth > 512 || !native_key_number(key, 't', info->base_type)) return false;
    if (info->generic_name && !native_key_name(key, info->generic_name)) return false;
    if (info->opaque_type_name &&
        (!native_key_append(key, "o") || !native_key_name(key, info->opaque_type_name))) return false;
    if (info->service_declaration &&
        (!native_key_number(key, 'd', info->service_declaration) ||
         !native_key_number(key, 'm', info->service_module) ||
         !native_key_number(key, 'i', info->service_ordinal) ||
         !native_key_number(key, 'c', info->service_category))) return false;
    if (info->element_type &&
        (!native_key_append(key, "e") || !native_key_type(key, info->element_type, depth + 1))) return false;
    if (info->type_param_count) {
        if (info->type_param_count < 0 || !info->type_params ||
            !native_key_number(key, 'g', info->type_param_count)) return false;
        for (int i = 0; i < info->type_param_count; ++i)
            if (!info->type_params[i] || !native_key_type(key, info->type_params[i], depth + 1)) return false;
    }
    if (info->base_type == TYPE_TUPLE) {
        if (info->tuple_element_count < 2 || !native_key_number(key, 'u', info->tuple_element_count)) return false;
        for (int i = 0; i < info->tuple_element_count; ++i) {
            TypeInfo fallback = {0};
            const TypeInfo *element = tuple_element_info(info, i, &fallback);
            if (!element || !native_key_type(key, element, depth + 1)) return false;
        }
    }
    if (info->base_type == TYPE_FUNCTION) {
        const FunctionSignature *sig = info->fn_sig;
        if (!sig || sig->param_count < 0 || !native_key_number(key, 'f', sig->param_count)) return false;
        for (int i = 0; i < sig->param_count; ++i) {
            if (!sig->param_types) return false;
            TypeInfo fallback = {.base_type = sig->param_types[i],
                .generic_name = sig->param_struct_names ? sig->param_struct_names[i] : NULL};
            const TypeInfo *parameter = sig->param_type_info && sig->param_type_info[i]
                ? sig->param_type_info[i] : &fallback;
            if (!native_key_type(key, parameter, depth + 1)) return false;
        }
        TypeInfo fallback = {.base_type = sig->return_type, .generic_name = sig->return_struct_name,
                             .fn_sig = sig->return_fn_sig};
        if (!native_key_type(key, sig->return_type_info ? sig->return_type_info : &fallback, depth + 1)) return false;
    }
    if (info->row_field_count || info->is_open_row || info->row_var_name) {
        if (info->row_field_count < 0 || !native_key_number(key, 'r', info->row_field_count) ||
            !native_key_number(key, 'b', info->is_open_row) || !native_key_name(key, info->row_var_name)) return false;
        for (int i = 0; i < info->row_field_count; ++i) {
            TypeInfo fallback;
            if (!native_key_type(key, row_field_info(info, i, &fallback), depth + 1)) return false;
            if (!info->row_field_types || !info->row_field_names ||
                !native_key_name(key, info->row_field_names[i]) ||
                !native_key_number(key, 't', info->row_field_types[i]) ||
                !native_key_name(key, info->row_field_type_names ? info->row_field_type_names[i] : NULL)) return false;
        }
    }
    if (info->type_var_count) {
        if (info->type_var_count < 0 || !info->type_var_names ||
            !native_key_number(key, 'v', info->type_var_count)) return false;
        for (int i = 0; i < info->type_var_count; ++i)
            if (!native_key_name(key, info->type_var_names[i])) return false;
    }
    return native_key_append(key, "z");
}

static char *native_structural_type_key(const TypeInfo *info) {
    NativeTypeKey key = {0};
    if (!native_key_append(&key, "__nano_shape_") || !native_key_type(&key, info, 0)) {
        free(key.text);
        fprintf(stderr, "I cannot encode a complete native generic argument within my annotation nesting limit.\n");
        exit(1);
    }
    return key.text;
}
#endif
