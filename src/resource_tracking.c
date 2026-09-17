#include "resource_tracking.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* I resolve each field in its declaring module, not the querying module. */
static bool resource_field_bearing(Environment *env, Type type, const char *name,
                                   char *owner, const bool *bearing) {
    if (!name || (type != TYPE_STRUCT && type != TYPE_UNION)) return false;
    char *caller = env->current_module;
    env->current_module = owner;
    StructDef *record = type == TYPE_STRUCT ? env_get_struct(env, name) : NULL;
    UnionDef *variant = type == TYPE_UNION ? env_get_union(env, name) : NULL;
    env->current_module = caller;
    for (int i = 0; record && i < env->struct_count; i++)
        if (record == &env->structs[i]) return bearing[i];
    for (int i = 0; variant && i < env->union_count; i++)
        if (variant == &env->unions[i]) return bearing[(size_t)env->struct_count + i];
    return false;
}

typedef struct ResourceExpansion {
    const UnionDef *declaration;
    const char *identity;
    bool collections_only;
    const struct ResourceExpansion *previous;
} ResourceExpansion;

/* I expand concrete union fields against the current fixed-point facts. A
 * repeated instantiation is an edge, not evidence of an owned resource. */
static bool resource_payload_fact(Environment *env, const TypeInfo *info,
                                  char *module, const UnionDef *formals,
                                  const bool *bearing, const bool *collections,
                                  bool collections_only, unsigned depth,
                                  const ResourceExpansion *active) {
    if (!info) return false;
    if (depth > 128) return true;
    if (info->base_type == TYPE_ARRAY)
        return resource_payload_fact(env, info->element_type, module, formals,
            bearing, collections, false, depth + 1, active);
    if (info->base_type == TYPE_LIST_GENERIC || info->base_type == TYPE_HASHMAP) {
        for (int i = 0; i < info->type_param_count; ++i)
            if (resource_payload_fact(env, info->type_params[i], module, formals,
                    bearing, collections, false, depth + 1, active)) return true;
        return false;
    }
    const char *name = info->generic_name;
    if (!name) return false;
    for (int i = 0; formals && i < formals->generic_param_count; ++i)
        if (!strcmp(name, formals->generic_params[i])) return false;
    if (info->base_type != TYPE_STRUCT && info->base_type != TYPE_UNION) return false;
    char *caller = env->current_module;
    env->current_module = module;
    UnionDef *def = env_get_union(env, name);
    env->current_module = caller;
    if (def && info->type_param_count > 0) {
        if (def->generic_param_count != info->type_param_count) return true;
        char *identity = typeinfo_to_generic_arg_name((TypeInfo*)info);
        if (!identity) return true;
        for (const ResourceExpansion *seen = active; seen; seen = seen->previous) {
            if (seen->declaration == def && seen->collections_only == collections_only && !strcmp(seen->identity, identity)) {
                free(identity);
                return false;
            }
        }
        ResourceExpansion expansion = {def, identity, collections_only, active};
        bool result = false;
        for (int arm = 0; !result && arm < def->variant_count; ++arm) {
            for (int field = 0; !result && field < def->variant_field_counts[arm]; ++field) {
                TypeInfo *payload = resolve_union_payload_type_info(def, arm, field, info);
                if (!payload) result = true;
                else result = resource_payload_fact(env, payload, def->module_name, formals,
                    bearing, collections, collections_only, depth + 1, &expansion);
                free_payload_type_info(payload);
            }
        }
        free(identity);
        return result;
    }
    const bool *facts = collections_only ? collections : bearing;
    return resource_field_bearing(env, TYPE_STRUCT, name, module, facts) ||
           resource_field_bearing(env, TYPE_UNION, name, module, facts);
}

/* I propagate both obligations and unsupported collection boundaries to a least
 * fixed point. A recursive declaration alone creates neither fact. */
static bool resource_classify(Environment *env, const char *name, const TypeInfo *info, bool collections_only) {
    if (!name && !info) return false;
    StructDef *sdef = name ? env_get_struct(env, name) : NULL;
    UnionDef *udef = name && !sdef ? env_get_union(env, name) : NULL;
    if (!sdef && !udef && !info) return false;
    bool roots = false;
    for (int i = 0; i < env->struct_count; ++i) roots |= env->structs[i].is_resource;
    if (!roots) return false;
    size_t count = (size_t)env->struct_count + (size_t)env->union_count;
    bool *bearing = calloc(count, sizeof(*bearing));
    bool *collections = calloc(count, sizeof(*collections));
    if (!bearing || !collections) {
        free(bearing);
        free(collections);
        fprintf(stderr, "I cannot allocate aggregate ownership classification state\n");
        return true;
    }
    for (int i = 0; i < env->struct_count; ++i) bearing[i] = env->structs[i].is_resource;
    bool changed;
    do {
        changed = false;
        for (int i = 0; i < env->struct_count; ++i) {
            StructDef *record = &env->structs[i];
            for (int field = 0; field < record->field_count; ++field) {
                if (!record->field_types || !record->field_type_names) continue;
                const TypeInfo *info = record->field_type_info ? record->field_type_info[field] : NULL;
                if (!bearing[i] && (resource_payload_fact(env, info, record->module_name, NULL,
                        bearing, collections, false, 0, NULL) ||
                        resource_field_bearing(env, record->field_types[field],
                            record->field_type_names[field], record->module_name, bearing)))
                    bearing[i] = changed = true;
                if (!collections[i] && (resource_payload_fact(env, info, record->module_name, NULL,
                        bearing, collections, true, 0, NULL) ||
                        resource_field_bearing(env, record->field_types[field],
                            record->field_type_names[field], record->module_name, collections)))
                    collections[i] = changed = true;
            }
        }
        for (int i = 0; i < env->union_count; ++i) {
            UnionDef *variant = &env->unions[i];
            size_t index = (size_t)env->struct_count + (size_t)i;
            for (int arm = 0; arm < variant->variant_count; ++arm) {
                if (!variant->variant_field_counts || !variant->variant_field_types ||
                    !variant->variant_field_types[arm]) continue;
                for (int field = 0; field < variant->variant_field_counts[arm]; ++field) {
                    const char *field_name = variant->variant_field_type_names &&
                        variant->variant_field_type_names[arm] ? variant->variant_field_type_names[arm][field] : NULL;
                    bool formal = false;
                    for (int param = 0; field_name && param < variant->generic_param_count; ++param)
                        if (!strcmp(field_name, variant->generic_params[param])) formal = true;
                    const TypeInfo *info = variant->variant_field_type_info && variant->variant_field_type_info[arm]
                        ? variant->variant_field_type_info[arm][field] : NULL;
                    if (!bearing[index] && (resource_payload_fact(env, info, variant->module_name, variant, bearing, collections, false, 0, NULL) ||
                        (!formal && resource_field_bearing(env, variant->variant_field_types[arm][field],
                            field_name, variant->module_name, bearing)))) bearing[index] = changed = true;
                    /* An array of an owned aggregate is unsupported even when a
                     * function merely passes the enclosing union through. */
                    if (!collections[index] &&
                        (resource_payload_fact(env, info, variant->module_name, variant,
                            bearing, collections, true, 0, NULL) ||
                         (!formal && resource_field_bearing(env, variant->variant_field_types[arm][field],
                            field_name, variant->module_name, collections)))) collections[index] = changed = true;
                }
            }
        }
    } while (changed);
    bool result = false;
    const bool *facts = collections_only ? collections : bearing;
    for (int i = 0; i < env->struct_count; ++i)
        if (sdef == &env->structs[i]) result = facts[i];
    for (int i = 0; i < env->union_count; ++i)
        if (udef == &env->unions[i]) result = facts[(size_t)env->struct_count + (size_t)i];
    if (info) result = resource_payload_fact(env, info, env->current_module, NULL,
        bearing, collections, collections_only, 0, NULL);
    free(bearing);
    free(collections);
    return result;
}

bool is_resource_type(Environment *env, const char *name) {
    return resource_classify(env, name, NULL, false);
}

bool has_resource_collection_payload(Environment *env, const char *name) {
    return resource_classify(env, name, NULL, true);
}

bool is_resource_type_info(Environment *env, const TypeInfo *info) {
    return resource_classify(env, NULL, info, false);
}

bool has_resource_collection_type_info(Environment *env, const TypeInfo *info) {
    return resource_classify(env, NULL, info, true);
}

/* Mark a variable as a resource if its type is a resource struct */
void mark_variable_as_resource_if_needed(Environment *env, const char *var_name, const char *struct_type_name) {
    if (!var_name || !struct_type_name) return;
    
    Symbol *sym = env_get_var(env, var_name);
    if (!sym) return;
    
    if (is_resource_type(env, struct_type_name)) {
        sym->is_resource = true;
        sym->resource_state = RESOURCE_UNUSED;
    }
}

/* Check resource usage (read/borrow) and update state */
void check_resource_use(Environment *env, const char *var_name, int line, int column, bool *has_error) {
    Symbol *sym = env_get_var(env, var_name);
    if (!sym || !sym->is_resource) return;
    
    /* Check if resource was already consumed */
    if (sym->resource_state == RESOURCE_CONSUMED) {
        fprintf(stderr, "Error at line %d, column %d: Cannot use resource '%s' after it has been consumed\n",
                line, column, var_name);
        *has_error = true;
        return;
    }
    
    /* Mark as used (but not consumed) */
    if (sym->resource_state == RESOURCE_UNUSED) {
        sym->resource_state = RESOURCE_USED;
    }
}

/* Check resource consumption (ownership transfer) */
void check_resource_consume(Environment *env, const char *var_name, int line, int column, bool *has_error) {
    Symbol *sym = env_get_var(env, var_name);
    if (!sym || !sym->is_resource) return;
    
    /* Check if resource was already consumed */
    if (sym->resource_state == RESOURCE_CONSUMED) {
        fprintf(stderr, "Error at line %d, column %d: Cannot consume resource '%s' - already consumed\n",
                line, column, var_name);
        *has_error = true;
        return;
    }
    
    /* Mark as consumed */
    sym->resource_state = RESOURCE_CONSUMED;
}

/* Check for resource leaks at end of scope */
void check_resource_leaks(Environment *env, bool *has_error) {
    for (int i = 0; i < env->symbol_count; i++) {
        Symbol *sym = &env->symbols[i];
        
        if (sym->is_resource && sym->resource_state != RESOURCE_CONSUMED) {
            fprintf(stderr, "Error at line %d, column %d: Resource '%s' must be consumed before going out of scope (resource leak)\n",
                    sym->def_line, sym->def_column, sym->name);
            *has_error = true;
        }
    }
}
