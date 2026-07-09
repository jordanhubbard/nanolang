#include "resource_tracking.h"
#include <stdio.h>
#include <string.h>

/* Check if a struct type is a resource type */
bool is_resource_type(Environment *env, const char *struct_name) {
    if (!struct_name) return false;
    
    StructDef *sdef = env_get_struct(env, struct_name);
    if (!sdef) return false;
    
    return sdef->is_resource;
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

