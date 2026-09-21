/* I retain declaration-owned opaque identity independently of C spelling. */
static char *opaque_origin(const char *source) {
    if (!source) return strdup("R");
    char *canonical = realpath(source, NULL);
    if (!canonical) return NULL;
    size_t length = strlen(canonical);
    char *origin = length < SIZE_MAX - 2 ? malloc(length + 2) : NULL;
    if (origin) { origin[0] = 'F'; memcpy(origin + 1, canonical, length + 1); }
    free(canonical);
    return origin;
}
static void opaque_definition_clear(OpaqueTypeDef *def) {
    free(def->name); free(def->c_type_name); free(def->origin); free(def->identity);
    memset(def, 0, sizeof(*def));
}
static void opaque_binding_clear(OpaqueTypeBinding *binding) {
    free(binding->origin); free(binding->name); free(binding->identity);
    memset(binding, 0, sizeof(*binding));
}
void env_free_opaque_types(Environment *env) {
    for (int i = 0; i < env->opaque_type_count; ++i) opaque_definition_clear(&env->opaque_types[i]);
    for (size_t i = 0; i < env->opaque_binding_count; ++i) opaque_binding_clear(&env->opaque_bindings[i]);
    free(env->opaque_types); free(env->opaque_bindings);
    for (size_t i = 0; i < env->tuple_literal_binding_count; ++i)
        free_payload_type_info(env->tuple_literal_bindings[i].type_info);
    free(env->tuple_literal_bindings);
    for (size_t i = 0; i < env->array_expression_binding_count; ++i)
        free_payload_type_info(env->array_expression_bindings[i].type_info);
    free(env->array_expression_bindings);
    env->array_expression_bindings = NULL; env->array_expression_binding_count = 0;
    env->tuple_literal_bindings = NULL; env->tuple_literal_binding_count = 0;
    free(env->opaque_reserved_indices);
    env->opaque_reserved_indices = NULL; env->opaque_reserved_count = 0;
    env->opaque_types = NULL; env->opaque_bindings = NULL;
    env->opaque_type_count = env->opaque_type_capacity = 0; env->opaque_binding_count = 0;
}
static OpaqueTypeDef *opaque_by_identity(Environment *env, const char *identity) {
    for (int i = 0; i < env->opaque_type_count; ++i)
        if (!strcmp(env->opaque_types[i].identity, identity)) return &env->opaque_types[i];
    return NULL;
}
static OpaqueTypeDef *opaque_local(Environment *env, const char *origin, const char *name) {
    for (int i = 0; i < env->opaque_type_count; ++i)
        if (!strcmp(env->opaque_types[i].origin, origin) && !strcmp(env->opaque_types[i].name, name))
            return &env->opaque_types[i];
    return NULL;
}
static bool opaque_definition_make(OpaqueTypeDef *def, const char *origin, const char *name) {
    size_t a = strlen(origin), b = strlen(name);
    if (a > (SIZE_MAX - 96) / 2 || b > (SIZE_MAX - 96) / 2 - a) return false;
    size_t capacity = 2 * (a + b) + 96;
    def->name = strdup(name); def->origin = strdup(origin);
    def->c_type_name = b < SIZE_MAX - 2 ? malloc(b + 2) : NULL;
    def->identity = malloc(capacity);
    if (!def->name || !def->origin || !def->c_type_name || !def->identity) {
        opaque_definition_clear(def); return false;
    }
    memcpy(def->c_type_name, name, b); memcpy(def->c_type_name + b, "*", 2);
    int written = snprintf(def->identity, capacity, "#opaque:%zu:", a);
    if (written < 0 || (size_t)written >= capacity) {
        opaque_definition_clear(def); return false;
    }
    size_t used = (size_t)written;
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < a; ++i) {
        unsigned char byte = (unsigned char)origin[i];
        def->identity[used++] = hex[byte >> 4]; def->identity[used++] = hex[byte & 15];
    }
    written = snprintf(def->identity + used, capacity - used, ":%zu:", b);
    if (written < 0 || (size_t)written >= capacity - used) {
        opaque_definition_clear(def); return false;
    }
    used += (size_t)written;
    for (size_t i = 0; i < b; ++i) {
        unsigned char byte = (unsigned char)name[i];
        def->identity[used++] = hex[byte >> 4]; def->identity[used++] = hex[byte & 15];
    }
    def->identity[used] = 0;
    return true;
}
/* I publish the new array only after every owning row is complete. */
static bool opaque_publish(Environment *env, OpaqueTypeDef *staged, size_t count) {
    size_t old = (size_t)env->opaque_type_count;
    if (count > (size_t)INT_MAX - old || old + count > SIZE_MAX / sizeof(OpaqueTypeDef)) return false;
    if (!count) return true;
    OpaqueTypeDef *rows = malloc((old + count) * sizeof(*rows));
    if (!rows) return false;
    if (old) memcpy(rows, env->opaque_types, old * sizeof(*rows));
    memcpy(rows + old, staged, count * sizeof(*rows));
    free(env->opaque_types); env->opaque_types = rows;
    env->opaque_type_count = env->opaque_type_capacity = (int)(old + count);
    memset(staged, 0, count * sizeof(*staged));
    return true;
}
bool env_define_opaque_type(Environment *env, const char *name) {
    if (!env || !name || !*name) return false;
    char *origin = opaque_origin(env_current_file(env));
    if (!origin) { env->opaque_resolution_failed = true; return false; }
    OpaqueTypeDef staged = {0};
    bool ok = opaque_local(env, origin, name) != NULL ||
        (opaque_definition_make(&staged, origin, name) && opaque_publish(env, &staged, 1));
    opaque_definition_clear(&staged); free(origin);
    if (!ok) env->opaque_resolution_failed = true;
    return ok;
}
static const char *opaque_decl_name(const ASTNode *node) {
    if (node->type == AST_OPAQUE_TYPE) return node->as.opaque_type.name;
    if (node->type == AST_STRUCT_DEF) return node->as.struct_def.original_name ?
        node->as.struct_def.original_name : node->as.struct_def.name;
    if (node->type == AST_UNION_DEF) return node->as.union_def.name;
    if (node->type == AST_ENUM_DEF) return node->as.enum_def.name;
    return NULL;
}
bool env_prepare_opaque_types(Environment *env, ASTNode *program) {
    if (!env || !program || program->type != AST_PROGRAM || env->opaque_resolution_failed) return false;
    bool needed = env->opaque_binding_count != 0;
    for (int i = 0; i < program->as.program.count; ++i)
        if (program->as.program.items[i]->type == AST_OPAQUE_TYPE) needed = true;
    if (!needed) return true;
    char *origin = opaque_origin(env_current_file(env));
    if (!origin) { env->opaque_resolution_failed = true; return false; }
    size_t count = 0;
    for (int i = 0; i < program->as.program.count; ++i) {
        ASTNode *node = program->as.program.items[i];
        if (node->type != AST_OPAQUE_TYPE) continue;
        for (int j = 0; j < program->as.program.count; ++j) {
            const char *other = opaque_decl_name(program->as.program.items[j]);
            if (i != j && other && !strcmp(other, node->as.opaque_type.name)) {
                fprintf(stderr, "I cannot define an opaque type twice or with another kind: %s\n", other);
                free(origin); return false;
            }
        }
        if (!opaque_local(env, origin, node->as.opaque_type.name)) ++count;
    }
    if (count > SIZE_MAX / sizeof(OpaqueTypeDef)) { free(origin); return false; }
    OpaqueTypeDef *staged = count ? calloc(count, sizeof(*staged)) : NULL;
    if (count && !staged) { free(origin); return false; }
    size_t used = 0; bool ok = true;
    for (int i = 0; i < program->as.program.count && ok; ++i) {
        ASTNode *node = program->as.program.items[i];
        if (node->type == AST_OPAQUE_TYPE && !opaque_local(env, origin, node->as.opaque_type.name))
            ok = opaque_definition_make(&staged[used++], origin, node->as.opaque_type.name);
    }
    /* I reject ambiguous imports unless an actual local declaration wins. */
    for (size_t i = 0; i < env->opaque_binding_count && ok; ++i) {
        OpaqueTypeBinding *a = &env->opaque_bindings[i];
        if (strcmp(a->origin, origin)) continue;
        bool local = false;
        for (int k = 0; k < program->as.program.count; ++k) {
            const char *name = opaque_decl_name(program->as.program.items[k]);
            if (name && !strcmp(name, a->name)) local = true;
        }
        for (size_t j = 0; !local && j < i; ++j) {
            OpaqueTypeBinding *b = &env->opaque_bindings[j];
            if (!strcmp(a->origin, b->origin) && !strcmp(a->name, b->name) && strcmp(a->identity, b->identity)) {
                fprintf(stderr, "I cannot resolve an ambiguous imported opaque type: %s\n", a->name); ok = false;
            }
        }
    }
    if (ok) ok = opaque_publish(env, staged, used);
    if (ok) {
        for (size_t i = 0; i < env->opaque_binding_count; ++i) {
            OpaqueTypeBinding *binding = &env->opaque_bindings[i];
            if (strcmp(binding->origin, origin)) continue;
            for (int j = 0; j < program->as.program.count; ++j) {
                const ASTNode *local = program->as.program.items[j];
                const char *name = opaque_decl_name(local);
                if (name && !strcmp(name, binding->name)) binding->hidden_by_local_type = true;
            }
        }
    }
    for (size_t i = 0; i < used; ++i) opaque_definition_clear(&staged[i]);
    free(staged); free(origin);
    if (!ok) env->opaque_resolution_failed = true;
    return ok;
}
OpaqueTypeDef *env_get_opaque_type(Environment *env, const char *name) {
    if (!env || !name || !env->opaque_type_count || env->opaque_resolution_failed) return NULL;
    OpaqueTypeDef *direct = opaque_by_identity(env, name);
    if (direct) return direct;
    char *origin = opaque_origin(env_current_file(env));
    if (!origin) { env->opaque_resolution_failed = true; return NULL; }
    OpaqueTypeDef *result = opaque_local(env, origin, name);
    if (!result) {
        for (size_t i = 0; i < env->opaque_binding_count; ++i) {
            OpaqueTypeBinding *binding = &env->opaque_bindings[i];
            if (binding->hidden_by_local_type || strcmp(binding->origin, origin) || strcmp(binding->name, name)) continue;
            OpaqueTypeDef *found = opaque_by_identity(env, binding->identity);
            if (!found || (result && result != found)) { result = NULL; break; }
            result = found;
        }
    }
    free(origin); return result;
}
bool env_import_opaque_types(Environment *env, const ASTNode *import, const ASTNode *target_program,
                             const char *importer, const char *target) {
    if (!env || !import || import->type != AST_IMPORT || !target_program ||
        target_program->type != AST_PROGRAM) return false;
    char *from = opaque_origin(importer), *to = opaque_origin(target);
    if (!from || !to) { free(from); free(to); return false; }
    size_t multiplier = import->as.import_stmt.is_selective && !import->as.import_stmt.is_wildcard ?
        (size_t)import->as.import_stmt.import_symbol_count : 1;
    size_t count = (size_t)target_program->as.program.count;
    if (multiplier && count > SIZE_MAX / multiplier) { free(from); free(to); return false; }
    size_t capacity = count * multiplier;
    if (capacity > SIZE_MAX / sizeof(OpaqueTypeBinding)) { free(from); free(to); return false; }
    OpaqueTypeBinding *staged = capacity ? calloc(capacity, sizeof(*staged)) : NULL;
    if (capacity && !staged) { free(from); free(to); return false; }
    size_t used = 0; bool ok = true;
    for (int i = 0; i < target_program->as.program.count && ok; ++i) {
        const ASTNode *declaration = target_program->as.program.items[i];
        const char *original = opaque_decl_name(declaration);
        if (!original) continue;
        bool opaque = declaration->type == AST_OPAQUE_TYPE;
        bool visible = opaque ||
            (declaration->type == AST_STRUCT_DEF && declaration->as.struct_def.is_pub) ||
            (declaration->type == AST_UNION_DEF && declaration->as.union_def.is_pub) ||
            (declaration->type == AST_ENUM_DEF && declaration->as.enum_def.is_pub);
        if (!visible) continue;
        OpaqueTypeDef other_kind = {0};
        const OpaqueTypeDef *def = opaque ? opaque_local(env, to, original) : &other_kind;
        if (!opaque) {
            if (!opaque_definition_make(&other_kind, to, original)) { ok = false; break; }
            /* I retain other authorized type kinds for ambiguity detection;
             * this key can never resolve to an opaque declaration. */
            other_kind.identity[0] = '!';
        }
        if (!def) { ok = false; break; }
        for (size_t j = 0; j < multiplier && ok; ++j) {
            const char *visible = def->name, *qualifier = import->as.import_stmt.module_alias;
            if (import->as.import_stmt.is_selective && !import->as.import_stmt.is_wildcard) {
                if (strcmp(import->as.import_stmt.import_symbols[j], def->name)) continue;
                if (import->as.import_stmt.import_aliases && import->as.import_stmt.import_aliases[j])
                    visible = import->as.import_stmt.import_aliases[j];
                qualifier = NULL;
            }
            size_t a = qualifier ? strlen(qualifier) : 0, b = strlen(visible);
            if (a > SIZE_MAX - 2 || b > SIZE_MAX - a - 2) { ok = false; break; }
            OpaqueTypeBinding *row = &staged[used++];
            row->origin = strdup(from); row->identity = strdup(def->identity); row->name = malloc(a + b + 2);
            if (!row->origin || !row->identity || !row->name) { ok = false; break; }
            if (a) { memcpy(row->name, qualifier, a); row->name[a++] = '.'; }
            memcpy(row->name + a, visible, b + 1);
        }
        opaque_definition_clear(&other_kind);
    }
    if (ok && used) {
        size_t old = env->opaque_binding_count;
        if (used > SIZE_MAX - old || old + used > SIZE_MAX / sizeof(OpaqueTypeBinding)) ok = false;
        OpaqueTypeBinding *rows = ok ? malloc((old + used) * sizeof(*rows)) : NULL;
        if (!rows) ok = false;
        if (ok) {
            if (old) memcpy(rows, env->opaque_bindings, old * sizeof(*rows));
            memcpy(rows + old, staged, used * sizeof(*rows));
            free(env->opaque_bindings); env->opaque_bindings = rows; env->opaque_binding_count = old + used;
            memset(staged, 0, used * sizeof(*staged));
        }
    }
    for (size_t i = 0; i < used; ++i) opaque_binding_clear(&staged[i]);
    free(staged); free(from); free(to);
    return ok;
}
