#ifndef NANOLANG_LAYOUT_VALIDATION_H
#define NANOLANG_LAYOUT_VALIDATION_H

/* I compare complete nominal applications, never their flattened C names. */
typedef struct { TypeInfo *info; bool active; } LayoutEntry;
typedef struct {
    Environment *env;
    char **formals;
    int formal_count;
    LayoutEntry *entries;
    int count;
} LayoutCheck;

static int layout_visit(LayoutCheck *check, const TypeInfo *info, int depth) {
    if (!info) return 0;
    if (info->base_type == TYPE_TUPLE) {
        for (int i = 0; i < info->tuple_element_count; ++i) {
            TypeInfo fallback = {0};
            int result = layout_visit(check, tuple_element_info(info, i, &fallback), depth);
            if (result) return result;
        }
        return 0;
    }
    if (info->base_type != TYPE_STRUCT && info->base_type != TYPE_UNION &&
        info->base_type != TYPE_GENERIC) return 0;
    if (!info->generic_name) return 0;
    if (!info->type_param_count) {
        for (int i = 0; i < check->formal_count; ++i)
            if (!strcmp(info->generic_name, check->formals[i])) return 0;
    }
    StructDef *record = env_get_struct(check->env, info->generic_name);
    UnionDef *sum = record ? NULL : env_get_union(check->env, info->generic_name);
    if ((!record && !sum) || (record && record->is_extern) || (sum && sum->is_extern)) return 0;
    TypeInfo canonical = *info;
    canonical.base_type = record ? TYPE_STRUCT : TYPE_UNION;
    canonical.generic_name = record ? record->name : sum->name;
    for (int i = 0; i < check->count; ++i)
        if (type_infos_equal(check->entries[i].info, &canonical))
            return check->entries[i].active ? 1 : 0;
    if (depth > 128) return 2;
    LayoutEntry *entries = realloc(check->entries, (check->count + 1) * sizeof *entries);
    if (!entries) return 3;
    check->entries = entries;
    int slot = check->count++;
    entries[slot] = (LayoutEntry){copy_payload_type_info(&canonical), true};
    char *saved_module = check->env->current_module;
    check->env->current_module = record ? record->module_name : sum->module_name;
    int result = 0;
    if (record) {
        for (int i = 0; i < record->field_count && !result; ++i) {
            TypeInfo fallback = {0};
            fallback.base_type = record->field_types[i];
            fallback.generic_name = record->field_type_names ? record->field_type_names[i] : NULL;
            result = layout_visit(check, record->field_type_info && record->field_type_info[i]
                                  ? record->field_type_info[i] : &fallback, depth);
        }
    } else {
        for (int arm = 0; arm < sum->variant_count && !result; ++arm) {
            for (int field = 0; field < sum->variant_field_counts[arm] && !result; ++field) {
                TypeInfo *payload = resolve_union_payload_type_info(sum, arm, field, &canonical);
                if (!payload) result = 3;
                else result = layout_visit(check, payload, depth + (sum->generic_param_count > 0));
                free_payload_type_info(payload);
            }
        }
    }
    check->env->current_module = saved_module;
    check->entries[slot].active = false;
    return result;
}

static bool validate_declaration_layouts(ASTNode *program, Environment *env) {
    for (int i = 0; i < program->as.program.count; ++i) {
        ASTNode *item = program->as.program.items[i];
        TypeInfo root = {0};
        LayoutCheck check = {.env = env};
        if (item->type == AST_STRUCT_DEF) {
            root.base_type = TYPE_STRUCT;
            root.generic_name = item->as.struct_def.name;
        } else if (item->type == AST_UNION_DEF) {
            root.base_type = TYPE_UNION;
            root.generic_name = item->as.union_def.name;
            UnionDef *sum = env_get_union(env, root.generic_name);
            if (!sum) continue;
            check.formals = sum->generic_params;
            check.formal_count = sum->generic_param_count;
            root.type_param_count = check.formal_count;
            if (root.type_param_count) {
                root.type_params = calloc(root.type_param_count, sizeof *root.type_params);
                if (!root.type_params) return false;
                for (int p = 0; p < root.type_param_count; ++p) {
                    TypeInfo formal = {.base_type = TYPE_STRUCT, .generic_name = check.formals[p]};
                    root.type_params[p] = copy_payload_type_info(&formal);
                }
            }
        } else continue;
        int result = layout_visit(&check, &root, 0);
        for (int p = 0; p < root.type_param_count; ++p) free_payload_type_info(root.type_params[p]);
        free(root.type_params);
        for (int p = 0; p < check.count; ++p) free_payload_type_info(check.entries[p].info);
        free(check.entries);
        if (result) {
            const char *message = result == 1 ? "I cannot admit cyclic by-value type layouts."
                : result == 2 ? "I cannot validate by-value layout expansion beyond 128 nested generic substitutions."
                : "I cannot validate incomplete by-value layout metadata.";
            emit_context_error("[E0038] TYPE LAYOUT", item->line, item->column, 1, message, NULL);
            return false;
        }
    }
    return true;
}
#endif
