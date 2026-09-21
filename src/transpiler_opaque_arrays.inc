/* I use complete checked element annotations for exact native value carriers. */
static bool native_array_struct_value(const TypeInfo *element) {
    if (!element) return false;
    return element->base_type == TYPE_STRUCT || element->base_type == TYPE_OPAQUE ||
        element->base_type == TYPE_UNION || element->base_type == TYPE_TUPLE || element->base_type == TYPE_FUNCTION;
}
static char *native_array_c_type(const TypeInfo *element, Environment *env) {
    if (!element) native_opaque_name_failure();
    StringBuilder *text = sb_create();
    emit_native_type_info(env, text, (TypeInfo *)element);
    char *result = text->buffer; free(text); return result;
}
static const char *native_array_scalar_suffix(const TypeInfo *element) {
    switch (element->base_type) {
        case TYPE_FLOAT: return "float";
        case TYPE_STRING: return "string";
        case TYPE_BOOL: return "bool";
        case TYPE_U8: return "u8";
        case TYPE_ARRAY: return "array";
        default: return "int";
    }
}
static const char *native_array_tag(const TypeInfo *element) {
    if (native_array_struct_value(element)) return "ELEM_STRUCT";
    switch (element->base_type) {
        case TYPE_FLOAT: return "ELEM_FLOAT";
        case TYPE_STRING: return "ELEM_STRING";
        case TYPE_BOOL: return "ELEM_BOOL";
        case TYPE_U8: return "ELEM_U8";
        case TYPE_ARRAY: return "ELEM_ARRAY";
        default: return "ELEM_INT";
    }
}
static void native_array_shape(WorkList *list, const char *array, const char *ctype) {
    /* A fresh empty carrier has no allocation or established width. */
    emit_formatted(list, "assert(%s->elem_type == ELEM_STRUCT); "
        "assert(%s->elem_size == sizeof(%s) || (%s->elem_size == 0 && %s->length == 0 && %s->data == NULL)); ",
        array, array, ctype, array, array, array);
}
static void native_array_load(WorkList *list, const TypeInfo *element, const char *ctype,
                              const char *array, const char *index, const char *local) {
    if (native_array_struct_value(element)) {
        native_array_shape(list, array, ctype);
        emit_formatted(list, "%s %s; memcpy(&%s, dyn_array_get_struct(%s, %s), sizeof(%s)); ",
            ctype, local, local, array, index, local);
    } else emit_formatted(list, "%s %s = dyn_array_get_%s(%s, %s); ",
        ctype, local, native_array_scalar_suffix(element), array, index);
}
static void native_array_push_value(WorkList *list, const TypeInfo *element,
                                    const char *array, const char *value) {
    if (native_array_struct_value(element))
        emit_formatted(list, "dyn_array_push_struct(%s, &%s, sizeof(%s)); ", array, value, value);
    else emit_formatted(list, "dyn_array_push_%s(%s, %s); ", native_array_scalar_suffix(element), array, value);
}
static unsigned native_array_capture(WorkList *list, ASTNode **args, int count,
                                     int typed_index, const char *ctype, Environment *env) {
    unsigned id = next_ordered_call_id(env, count);
    for (int i = 0; i < count; ++i) {
        emit_formatted(list, "%s __nl_arg_%u_%d = ", i == typed_index ? ctype : "__auto_type", id, i);
        build_expr(list, args[i], env); emit_literal(list, "; ");
    }
    return id;
}
static bool native_opaque_array_literal(WorkList *list, ASTNode *expr, Environment *env) {
    const TypeInfo *array = env_array_expression_info(env, expr);
    const TypeInfo *element = array ? array->element_type : NULL;
    if (!element || !type_info_exact_array_element(element)) return false;
    char *ctype = native_array_c_type(element, env);
    emit_literal(list, "({ ");
    unsigned id = next_ordered_call_id(env, expr->as.array_literal.element_count);
    for (int i = 0; i < expr->as.array_literal.element_count; ++i) {
        emit_formatted(list, "%s __nl_arg_%u_%d = ", ctype, id, i);
        build_expr(list, expr->as.array_literal.elements[i], env); emit_literal(list, "; ");
    }
    emit_literal(list, "DynArray* _oa = dyn_array_new(ELEM_STRUCT); ");
    for (int i = 0; i < expr->as.array_literal.element_count; ++i)
        emit_formatted(list, "dyn_array_push_struct(_oa, &__nl_arg_%u_%d, sizeof(__nl_arg_%u_%d)); ", id, i, id, i);
    emit_literal(list, "_oa; })"); free(ctype); return true;
}
static bool native_opaque_array_call(WorkList *list, ASTNode *expr, Environment *env) {
    const char *name = expr->as.call.name;
    if (!name || expr->as.call.func_expr || expr->as.call.borrow_mode) return false;
    bool make = !strcmp(name, "array_new"), push = !strcmp(name, "array_push"), set = !strcmp(name, "array_set");
    bool get = !strcmp(name, "at") || !strcmp(name, "array_get"), pop = !strcmp(name, "array_pop");
    bool map = !strcmp(name, "map"), filter = !strcmp(name, "filter"), reduce = !strcmp(name, "reduce");
    if (!(make || push || set || get || pop || map || filter || reduce)) return false;
    if (push && !env_array_push_is_builtin(env, expr->line, expr->column)) return false;
    int count = set || reduce ? 3 : pop ? 1 : 2;
    if (expr->as.call.arg_count != count) return false;
    const TypeInfo *array = checked_expression_type_info(make ? expr : expr->as.call.args[0], env);
    if (!array || array->base_type != TYPE_ARRAY || !array->element_type) return false;
    const TypeInfo *element = array->element_type;
    FunctionSignature *callback = map || filter || reduce
        ? checked_callable_signature_copy(expr->as.call.args[reduce ? 2 : 1], env) : NULL;
    TypeInfo flat_result = {0};
    const TypeInfo *output = element;
    if (callback && (map || reduce)) {
        flat_result = (TypeInfo){.base_type = callback->return_type, .generic_name = callback->return_struct_name,
            .fn_sig = callback->return_fn_sig};
        if (callback->return_type_info) flat_result = *callback->return_type_info;
        if (!flat_result.fn_sig) flat_result.fn_sig = callback->return_fn_sig;
        output = &flat_result;
    }
    bool affected = type_info_exact_array_element(element) || type_info_exact_array_element(output);
    if (!affected || ((map || filter || reduce) && !callback)) { free_function_signature(callback); return false; }
    /* Nested array carriers retain their existing pointer representation. */
    if (!(map || filter || reduce) && element->base_type == TYPE_ARRAY) { free_function_signature(callback); return false; }
    char *ctype = native_array_c_type(element, env);
    char *outtype = native_array_c_type(output, env);
    emit_literal(list, "({ ");
    int typed = make || push || reduce ? 1 : set ? 2 : -1;
    unsigned id = native_array_capture(list, expr->as.call.args, count, typed, reduce ? outtype : ctype, env);
    if (make) {
        emit_formatted(list, "if (__nl_arg_%u_0 < 0) { fputs(\"I require a non-negative array count\\n\", stderr); abort(); } "
            "DynArray* _oa = dyn_array_new(ELEM_STRUCT); for (int64_t _oi = 0; _oi < __nl_arg_%u_0; ++_oi) { "
            "dyn_array_push_struct(_oa, &__nl_arg_%u_1, sizeof(__nl_arg_%u_1)); } _oa; })", id, id, id, id);
    } else {
        emit_formatted(list, "DynArray* _oa = __nl_arg_%u_0; ", id);
        if (native_array_struct_value(element)) native_array_shape(list, "_oa", ctype);
        if (push || set) {
            if (push) emit_formatted(list, "dyn_array_push_struct(_oa, &__nl_arg_%u_1, sizeof(__nl_arg_%u_1)); _oa; })", id, id);
            else emit_formatted(list, "dyn_array_set_struct(_oa, __nl_arg_%u_1, &__nl_arg_%u_2, sizeof(__nl_arg_%u_2)); })", id, id, id);
        } else if (get) {
            char index[64]; snprintf(index, sizeof index, "__nl_arg_%u_1", id);
            native_array_load(list, element, ctype, "_oa", index, "_ov"); emit_literal(list, "_ov; })");
        } else if (pop) {
            emit_formatted(list, "%s _ov = {0}; if (_oa->length != 0) { "
                "bool _ok; dyn_array_pop_struct(_oa, &_ov, sizeof(_ov), &_ok); assert(_ok); } _ov; })", ctype);
        } else {
            emit_formatted(list, "__auto_type _of = __nl_arg_%u_%d; int64_t _on = dyn_array_length(_oa); ", id, reduce ? 2 : 1);
            if (reduce) emit_formatted(list, "%s _oc = __nl_arg_%u_1; ", outtype, id);
            else emit_formatted(list, "DynArray* _out = dyn_array_new(%s); ", native_array_tag(output));
            emit_literal(list, "for (int64_t _oi = 0; _oi < _on; ++_oi) { ");
            native_array_load(list, element, ctype, "_oa", "_oi", "_ov");
            if (reduce) emit_literal(list, "_oc = _of(_oc, _ov); ");
            else if (filter) {
                emit_literal(list, "if (_of(_ov)) { "); native_array_push_value(list, element, "_out", "_ov"); emit_literal(list, "} ");
            } else {
                emit_formatted(list, "%s _mapped = _of(_ov); ", outtype);
                native_array_push_value(list, output, "_out", "_mapped");
            }
            emit_literal(list, reduce ? "} _oc; })" : "} _out; })");
        }
    }
    free(ctype); free(outtype); free_function_signature(callback); return true;
}
