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

/* I walk fixed array payload annotations without mistaking declaration formals
 * for unrelated resource records. Generic substitution remains a separate step. */
static bool resource_payload_bearing(Environment *env, const TypeInfo *info,
                                      const UnionDef *owner, const bool *bearing,
                                      unsigned depth) {
    if (!info) return false;
    if (depth > 512) return true;
    if (info->base_type == TYPE_ARRAY)
        return resource_payload_bearing(env, info->element_type, owner, bearing, depth + 1);
    const char *name = info->generic_name;
    if (!name) return false;
    for (int i = 0; i < owner->generic_param_count; ++i)
        if (!strcmp(name, owner->generic_params[i])) return false;
    if (info->base_type != TYPE_STRUCT && info->base_type != TYPE_UNION) return false;
    return resource_field_bearing(env, TYPE_STRUCT, name, owner->module_name, bearing) ||
           resource_field_bearing(env, TYPE_UNION, name, owner->module_name, bearing);
}

/* I propagate both obligations and unsupported collection boundaries to a least
 * fixed point. A recursive declaration alone creates neither fact. */
static bool resource_classify(Environment *env, const char *name, bool collections_only) {
    if (!name) return false;
    StructDef *sdef = env_get_struct(env, name);
    UnionDef *udef = sdef ? NULL : env_get_union(env, name);
    if (!sdef && !udef) return false;
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
                if (!bearing[i] && resource_field_bearing(env, record->field_types[field],
                        record->field_type_names[field], record->module_name, bearing))
                    bearing[i] = changed = true;
                if (!collections[i] && resource_field_bearing(env, record->field_types[field],
                        record->field_type_names[field], record->module_name, collections))
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
                    if (!bearing[index] && (resource_payload_bearing(env, info, variant, bearing, 0) ||
                        (!formal && resource_field_bearing(env, variant->variant_field_types[arm][field],
                            field_name, variant->module_name, bearing)))) bearing[index] = changed = true;
                    /* An array of an owned aggregate is unsupported even when a
                     * function merely passes the enclosing union through. */
                    if (!collections[index] &&
                        (resource_payload_bearing(env, info, variant,
                            info && info->base_type == TYPE_ARRAY ? bearing : collections, 0) ||
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
    free(bearing);
    free(collections);
    return result;
}

bool is_resource_type(Environment *env, const char *name) {
    return resource_classify(env, name, false);
}

bool has_resource_collection_payload(Environment *env, const char *name) {
    return resource_classify(env, name, true);
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
