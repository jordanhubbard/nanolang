/* I stage complete derived declarations before emitting any layout body. */
typedef enum { ND_RECORD, ND_UNION, ND_INSTANCE, ND_TUPLE, ND_FUNCTION } NativeDerivedKind;
typedef struct {
    NativeDerivedKind kind;
    int source_index;
    char *key, *name;
    TypeInfo *info; /* Owned only for tuple/callable rows. */
    unsigned state;
} NativeDerivedRow;
typedef struct {
    Environment *env;
    FunctionTypeRegistry *functions;
    TupleTypeRegistry *tuples;
    NativeDerivedRow *rows;
    size_t count;
} NativeDerivedGraph;
static _Thread_local NativeDerivedGraph *native_derived_graph;

static size_t native_derived_find(const char *key) {
    for (size_t i = 0; native_derived_graph && i < native_derived_graph->count; ++i)
        if (!strcmp(native_derived_graph->rows[i].key, key)) return i;
    return SIZE_MAX;
}
static size_t native_derived_add(NativeDerivedGraph *graph, NativeDerivedKind kind,
                                int index, const char *key, const char *name, const TypeInfo *info) {
    size_t old = native_derived_find(key);
    if (old != SIZE_MAX) return old;
    NativeDerivedRow row = {.kind = kind, .source_index = index};
    row.key = strdup(key); row.name = strdup(name);
    if (info) row.info = copy_complete_type_info_checked(info);
    NativeDerivedRow *rows = graph->count < SIZE_MAX / sizeof *rows
        ? malloc((graph->count + 1) * sizeof *rows) : NULL;
    if (!row.key || !row.name || (info && !row.info) || !rows) {
        free(row.key); free(row.name); free_payload_type_info(row.info); free(rows);
        native_opaque_name_failure();
    }
    if (graph->count) memcpy(rows, graph->rows, graph->count * sizeof *rows);
    rows[graph->count] = row; free(graph->rows); graph->rows = rows;
    return graph->count++;
}
static void native_derived_collect(NativeDerivedGraph *, const TypeInfo *, unsigned);
static void native_derived_signature_children(NativeDerivedGraph *graph, const FunctionSignature *sig, unsigned depth) {
    if (!sig) return;
    for (int i = 0; i < sig->param_count; ++i) {
        TypeInfo flat = {.base_type = sig->param_types[i],
            .generic_name = sig->param_struct_names ? sig->param_struct_names[i] : NULL};
        native_derived_collect(graph, sig->param_type_info && sig->param_type_info[i] ? sig->param_type_info[i] : &flat, depth + 1);
    }
    TypeInfo result = {.base_type = sig->return_type, .generic_name = sig->return_struct_name,
        .fn_sig = sig->return_fn_sig};
    if (sig->return_type_info) result = *sig->return_type_info;
    if (!result.fn_sig) result.fn_sig = sig->return_fn_sig;
    native_derived_collect(graph, &result, depth + 1);
}
static void native_derived_collect(NativeDerivedGraph *graph, const TypeInfo *info, unsigned depth) {
    if (!info) return;
    if (depth > 128) native_opaque_name_failure();
    if (info->base_type == TYPE_FUNCTION && !info->fn_sig) native_opaque_name_failure();
    if (info->base_type == TYPE_TUPLE || info->base_type == TYPE_FUNCTION) {
        char *key = opaque_type_info_key(info);
        if (!key) native_opaque_name_failure();
        if (native_derived_find(key) != SIZE_MAX) { free(key); return; }
        const char *name = info->base_type == TYPE_TUPLE
            ? register_tuple_type(graph->tuples, (TypeInfo *)info)
            : register_function_signature(graph->functions, info->fn_sig);
        size_t index = native_derived_add(graph, info->base_type == TYPE_TUPLE ? ND_TUPLE : ND_FUNCTION,
                                         -1, key, name, info);
        free(key);
        /* Registry rows must not borrow a substituted temporary. Replace only
         * exact borrowed pointers with this separately allocated owned snapshot. */
        TypeInfo *owned = graph->rows[index].info;
        if (info->base_type == TYPE_TUPLE) {
            for (int i = 0; i < graph->tuples->count; ++i)
                if (graph->tuples->tuples[i] == info && !graph->tuples->owned[i]) graph->tuples->tuples[i] = owned;
            for (int i = 0; i < owned->tuple_element_count; ++i) {
                TypeInfo flat;
                const TypeInfo *child = type_info_tuple_element(owned, i, &flat);
                if (!child) native_opaque_name_failure();
                native_derived_collect(graph, child, depth + 1);
            }
        } else {
            for (int i = 0; i < graph->functions->count; ++i)
                if (graph->functions->signatures[i] == info->fn_sig) graph->functions->signatures[i] = owned->fn_sig;
            native_derived_signature_children(graph, owned->fn_sig, depth + 1);
        }
        return;
    }
    native_derived_collect(graph, info->element_type, depth + 1);
    for (int i = 0; info->type_params && i < info->type_param_count; ++i)
        native_derived_collect(graph, info->type_params[i], depth + 1);
}
static const char *native_derived_type_name(const TypeInfo *info) {
    if (!native_derived_graph || !info) return NULL;
    char *key = opaque_type_info_key(info);
    if (!key) native_opaque_name_failure();
    size_t row = native_derived_find(key); free(key);
    return row == SIZE_MAX ? NULL : native_derived_graph->rows[row].name;
}
static bool native_derived_forwarded(const char *name) {
    for (size_t i = 0; native_derived_graph && i < native_derived_graph->count; ++i)
        if (native_derived_graph->rows[i].kind != ND_FUNCTION &&
            !strcmp(native_derived_graph->rows[i].name, name)) return true;
    return false;
}
static bool native_derived_emitted(const TypeInfo *info) {
    if (!native_derived_graph) return false;
    char *key = opaque_type_info_key(info);
    if (!key) native_opaque_name_failure();
    size_t row = native_derived_find(key); free(key);
    return row != SIZE_MAX && native_derived_graph->rows[row].state == 2;
}
static void native_derived_graph_free(NativeDerivedGraph *graph) {
    if (!graph) return;
    for (size_t i = 0; i < graph->count; ++i) {
        free(graph->rows[i].key); free(graph->rows[i].name);
        free_payload_type_info(graph->rows[i].info);
    }
    free(graph->rows); free(graph);
}
static bool native_derived_annotation_required(const TypeInfo *info, unsigned depth) {
    if (!info || depth > 128) return false;
    if ((info->base_type == TYPE_TUPLE || info->base_type == TYPE_FUNCTION) &&
        opaque_type_info_present(info)) return true;
    /* Ordinary derived values need this graph only at the new array boundary. */
    if (info->base_type == TYPE_ARRAY && info->element_type &&
        (info->element_type->base_type == TYPE_TUPLE || info->element_type->base_type == TYPE_FUNCTION)) return true;
    if (native_derived_annotation_required(info->element_type, depth + 1)) return true;
    for (int i = 0; info->type_params && i < info->type_param_count; ++i)
        if (native_derived_annotation_required(info->type_params[i], depth + 1)) return true;
    return false;
}
static NativeDerivedGraph *native_derived_prepare(Environment *env, FunctionTypeRegistry *functions,
                                                  TupleTypeRegistry *tuples) {
    bool needed = false;
    for (int i = 0; i < functions->count; ++i) {
        TypeInfo info = {.base_type = TYPE_FUNCTION, .fn_sig = functions->signatures[i]};
        if (opaque_type_info_present(&info)) needed = true;
    }
    for (int i = 0; i < tuples->count; ++i)
        if (opaque_type_info_present(tuples->tuples[i])) needed = true;
    for (int i = 0; i < env->generic_instance_count; ++i)
        if (opaque_type_info_present(env->generic_instances[i].type_info)) needed = true;
    for (int i = 0; i < env->struct_count; ++i)
        for (int j = 0; env->structs[i].field_type_info && j < env->structs[i].field_count; ++j) {
            const TypeInfo *info = env->structs[i].field_type_info[j];
            if (native_derived_annotation_required(info, 0)) needed = true;
        }
    for (int i = 0; i < env->union_count; ++i) {
        UnionDef *def = &env->unions[i];
        if (def->generic_param_count) continue;
        for (int a = 0; def->variant_field_type_info && a < def->variant_count; ++a)
            for (int f = 0; def->variant_field_type_info[a] && f < def->variant_field_counts[a]; ++f) {
                const TypeInfo *info = def->variant_field_type_info[a][f];
                if (native_derived_annotation_required(info, 0)) needed = true;
            }
    }
    for (size_t i = 0; i < env->array_expression_binding_count; ++i)
        if (native_derived_annotation_required(env->array_expression_bindings[i].type_info, 0)) needed = true;
    for (int i = 0; i < env->function_count; ++i) {
        Function *function = &env->functions[i];
        if (native_derived_annotation_required(function->return_type_info, 0)) needed = true;
        for (int p = 0; function->params && p < function->param_count; ++p)
            if (native_derived_annotation_required(function->params[p].type_info, 0)) needed = true;
    }
    if (!needed) return NULL;
    NativeDerivedGraph *graph = calloc(1, sizeof *graph);
    if (!graph) native_opaque_name_failure();
    graph->env = env; graph->functions = functions; graph->tuples = tuples;
    native_derived_graph = graph;
    for (int i = 0; i < env->struct_count; ++i) {
        StructDef *def = &env->structs[i];
        if (!def->name || def->is_extern) continue;
        TypeInfo info = {.base_type = TYPE_STRUCT, .generic_name = def->name};
        char *key = opaque_type_info_key(&info);
        if (!key) native_opaque_name_failure();
        native_derived_add(graph, ND_RECORD, i, key, get_prefixed_type_name(def->name), NULL); free(key);
    }
    for (int i = 0; i < env->union_count; ++i) {
        UnionDef *def = &env->unions[i];
        if (!def->name || def->is_extern || def->generic_param_count) continue;
        TypeInfo info = {.base_type = TYPE_UNION, .generic_name = def->name};
        char *key = opaque_type_info_key(&info);
        if (!key) native_opaque_name_failure();
        native_derived_add(graph, ND_UNION, i, key, get_prefixed_type_name(def->name), NULL); free(key);
    }
    for (int i = 0; i < env->generic_instance_count; ++i) {
        GenericInstantiation *instance = &env->generic_instances[i];
        UnionDef *def = env_get_union(env, instance->generic_name);
        if (!def || !def->generic_param_count || def->is_extern) continue;
        char name[256];
        if (!build_monomorphized_name(name, sizeof name, instance->generic_name,
                (const char **)instance->type_arg_names, instance->type_arg_count, instance->type_info)) native_opaque_name_failure();
        char *key = instance->type_info ? opaque_type_info_key(instance->type_info) : strdup(instance->concrete_name);
        if (!key) native_opaque_name_failure();
        native_derived_add(graph, ND_INSTANCE, i, key, get_prefixed_type_name(name), NULL); free(key);
    }
    for (size_t i = 0; i < env->array_expression_binding_count; ++i)
        native_derived_collect(graph, env->array_expression_bindings[i].type_info, 0);
    for (int i = 0; i < env->function_count; ++i) {
        Function *function = &env->functions[i];
        native_derived_collect(graph, function->return_type_info, 0);
        for (int p = 0; function->params && p < function->param_count; ++p)
            native_derived_collect(graph, function->params[p].type_info, 0);
    }
    for (int i = 0; i < functions->count; ++i) {
        TypeInfo info = {.base_type = TYPE_FUNCTION, .fn_sig = functions->signatures[i]};
        native_derived_collect(graph, &info, 0);
    }
    for (int i = 0; i < tuples->count; ++i) native_derived_collect(graph, tuples->tuples[i], 0);
    for (size_t i = 0; i < graph->count; ++i) {
        NativeDerivedKind kind = graph->rows[i].kind;
        int source = graph->rows[i].source_index;
        if (kind == ND_RECORD) {
            StructDef *def = &env->structs[source];
            for (int j = 0; def->field_type_info && j < def->field_count; ++j)
                native_derived_collect(graph, def->field_type_info[j], 0);
        } else if (kind == ND_UNION || kind == ND_INSTANCE) {
            GenericInstantiation *instance = kind == ND_INSTANCE ? &env->generic_instances[source] : NULL;
            UnionDef *def = instance ? env_get_union(env, instance->generic_name) : &env->unions[source];
            for (int a = 0; a < def->variant_count; ++a)
                for (int f = 0; f < def->variant_field_counts[a]; ++f) {
                    TypeInfo *payload = resolve_union_payload_type_info(def, a, f, instance ? instance->type_info : NULL);
                    native_derived_collect(graph, payload, 0); free_payload_type_info(payload);
                }
        }
    }
    return graph;
}
static void native_derived_emit_row(NativeDerivedGraph *, size_t, StringBuilder *, unsigned);
static void native_derived_dependency(NativeDerivedGraph *graph, const TypeInfo *info,
                                      bool declaration_only, StringBuilder *sb, unsigned depth) {
    if (!info) return;
    if (depth > 128) native_opaque_name_failure();
    if (info->base_type == TYPE_ARRAY || info->base_type == TYPE_OPAQUE ||
        (info->generic_name && env_get_opaque_type(graph->env, info->generic_name))) return;
    char *key = NULL;
    const char *name = NULL;
    if (info->base_type == TYPE_FUNCTION || info->base_type == TYPE_TUPLE) {
        name = native_derived_type_name(info);
        if (!name) native_opaque_name_failure();
        if (declaration_only && info->base_type == TYPE_TUPLE) return;
    } else if ((info->base_type == TYPE_STRUCT || info->base_type == TYPE_UNION) && info->generic_name) {
        if (declaration_only) return;
        key = typeinfo_to_generic_arg_name((TypeInfo *)info);
        if (!key) native_opaque_name_failure();
        name = get_prefixed_type_name(key);
    } else return;
    size_t row = SIZE_MAX;
    for (size_t i = 0; i < graph->count; ++i)
        if (!strcmp(graph->rows[i].name, name)) { row = i; break; }
    free(key);
    if (row != SIZE_MAX) native_derived_emit_row(graph, row, sb, depth + 1);
}
static void native_derived_emit_row(NativeDerivedGraph *graph, size_t index, StringBuilder *sb, unsigned depth) {
    if (depth > 128 || index >= graph->count) native_opaque_name_failure();
    NativeDerivedRow *row = &graph->rows[index];
    if (row->state == 2) return;
    if (row->state == 1) {
        fprintf(stderr, "I cannot emit a recursive by-value native layout\n"); exit(1);
    }
    row->state = 1;
    if (row->kind == ND_RECORD) {
        StructDef *def = &graph->env->structs[row->source_index];
        for (int i = 0; i < def->field_count; ++i) {
            TypeInfo flat = {.base_type = def->field_types[i],
                .generic_name = def->field_type_names ? def->field_type_names[i] : NULL};
            native_derived_dependency(graph, def->field_type_info && def->field_type_info[i] ? def->field_type_info[i] : &flat, false, sb, depth + 1);
        }
        emit_struct_definition_single(graph->env, sb, def);
    } else if (row->kind == ND_UNION || row->kind == ND_INSTANCE) {
        GenericInstantiation *instance = row->kind == ND_INSTANCE ? &graph->env->generic_instances[row->source_index] : NULL;
        UnionDef *def = instance ? env_get_union(graph->env, instance->generic_name) : &graph->env->unions[row->source_index];
        for (int a = 0; a < def->variant_count; ++a)
            for (int f = 0; f < def->variant_field_counts[a]; ++f) {
                TypeInfo *payload = resolve_union_payload_type_info(def, a, f, instance ? instance->type_info : NULL);
                TypeInfo flat = {.base_type = def->variant_field_types[a][f],
                    .generic_name = def->variant_field_type_names && def->variant_field_type_names[a] ? def->variant_field_type_names[a][f] : NULL};
                native_derived_dependency(graph, payload ? payload : &flat, false, sb, depth + 1);
                free_payload_type_info(payload);
            }
        if (!instance) emit_union_definition_single(graph->env, sb, def);
        else {
            char name[256];
            if (!build_monomorphized_name(name, sizeof name, instance->generic_name,
                    (const char **)instance->type_arg_names, instance->type_arg_count, instance->type_info)) native_opaque_name_failure();
            emit_generic_union_instantiation(graph->env, sb, def, instance, name);
        }
    } else if (row->kind == ND_TUPLE) {
        for (int i = 0; i < row->info->tuple_element_count; ++i) {
            TypeInfo flat;
            const TypeInfo *child = type_info_tuple_element(row->info, i, &flat);
            if (!child) native_opaque_name_failure();
            native_derived_dependency(graph, child, false, sb, depth + 1);
        }
        generate_tuple_typedef(sb, row->info, row->name, graph->env);
    } else {
        FunctionSignature *signature = row->info->fn_sig;
        for (int i = 0; i < signature->param_count; ++i) {
            TypeInfo flat = {.base_type = signature->param_types[i],
                .generic_name = signature->param_struct_names ? signature->param_struct_names[i] : NULL};
            native_derived_dependency(graph, signature->param_type_info && signature->param_type_info[i] ? signature->param_type_info[i] : &flat, true, sb, depth + 1);
        }
        TypeInfo result = {.base_type = signature->return_type, .generic_name = signature->return_struct_name, .fn_sig = signature->return_fn_sig};
        if (signature->return_type_info) result = *signature->return_type_info;
        if (!result.fn_sig) result.fn_sig = signature->return_fn_sig;
        native_derived_dependency(graph, &result, true, sb, depth + 1);
        generate_function_typedef(sb, signature, row->name, graph->env);
    }
    row->state = 2;
}
static void native_derived_emit(NativeDerivedGraph *graph, StringBuilder *sb) {
    for (size_t i = 0; i < graph->count; ++i)
        if (graph->rows[i].kind != ND_FUNCTION)
            sb_appendf(sb, "#ifndef DEFINED_%s\ntypedef struct %s %s;\n#endif\n", graph->rows[i].name, graph->rows[i].name, graph->rows[i].name);
    for (size_t i = 0; i < graph->count; ++i) native_derived_emit_row(graph, i, sb, 0);
}
