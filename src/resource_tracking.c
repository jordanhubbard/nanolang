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

/* I classify registered named records and unions, not generic substitutions. */
bool is_resource_type(Environment *env, const char *struct_name) {
    if (!struct_name) return false;
    
    StructDef *sdef = env_get_struct(env, struct_name);
    UnionDef *udef = sdef ? NULL : env_get_union(env, struct_name);
    if (!sdef && !udef) return false;
    
    if (sdef && sdef->is_resource) return true;
    /* I compute the least fixed point, so a cycle alone is not a resource.
     * Allocation failure conservatively retains the obligation. */
    bool *bearing = calloc((size_t)env->struct_count + (size_t)env->union_count, sizeof(*bearing));
    if (!bearing) {
        fprintf(stderr, "I cannot allocate aggregate ownership classification state\n");
        return true;
    }
    for (int i = 0; i < env->struct_count; i++)
        bearing[i] = env->structs[i].is_resource;
    bool changed;
    do {
        changed = false;
        for (int i = 0; i < env->struct_count; i++) {
            StructDef *record = &env->structs[i];
            if (bearing[i]) continue;
            for (int field = 0; field < record->field_count && !bearing[i]; field++) {
                if (record->field_types && record->field_type_names &&
                    resource_field_bearing(env, record->field_types[field],
                        record->field_type_names[field], record->module_name, bearing))
                    bearing[i] = changed = true;
            }
        }
        for (int i = 0; i < env->union_count; i++) {
            UnionDef *variant = &env->unions[i];
            size_t index = (size_t)env->struct_count + i;
            if (bearing[index]) continue;
            for (int arm = 0; arm < variant->variant_count && !bearing[index]; arm++) {
                if (!variant->variant_field_counts || !variant->variant_field_types ||
                    !variant->variant_field_type_names || !variant->variant_field_types[arm] ||
                    !variant->variant_field_type_names[arm]) continue;
                for (int field = 0; field < variant->variant_field_counts[arm]; field++) {
                    if (resource_field_bearing(env, variant->variant_field_types[arm][field],
                            variant->variant_field_type_names[arm][field], variant->module_name, bearing)) {
                        bearing[index] = changed = true;
                        break;
                    }
                }
            }
        }
    } while (changed);
    bool result = false;
    for (int i = 0; i < env->struct_count; i++)
        if (sdef == &env->structs[i]) result = bearing[i];
    for (int i = 0; i < env->union_count; i++)
        if (udef == &env->unions[i]) result = bearing[(size_t)env->struct_count + i];
    free(bearing);
    return result;
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
