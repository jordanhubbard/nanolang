#ifndef NANO_CHECKED_LOOP_BINDING_H
#define NANO_CHECKED_LOOP_BINDING_H

#include "nanolang.h"

/* I restore the checker's declaration, not a newly inferred element type.
 * Call only after evaluating the iterable, before lowering the loop body. */
static bool reestablish_checked_loop_binding(Environment *env, ASTNode *loop) {
    Symbol checked;
    bool found = false;
    for (int i = env->symbol_count - 1; i >= 0; --i) {
        Symbol *symbol = &env->symbols[i];
        if (symbol->name && !strcmp(symbol->name, loop->as.for_stmt.var_name) &&
            symbol->def_line == loop->line && symbol->def_column == loop->column &&
            (!symbol->def_file || !env->current_file || !strcmp(symbol->def_file, env->current_file))) {
            checked = *symbol;
            found = true;
            break;
        }
    }
    if (!found) return false;
    char *nominal = checked.struct_type_name ? strdup(checked.struct_type_name) : NULL;
    if (checked.struct_type_name && !nominal) return false;
    env_define_var_with_type_info(env, loop->as.for_stmt.var_name, checked.type,
        checked.element_type, checked.type_info, checked.is_mut, create_void());
    Symbol *binding = &env->symbols[env->symbol_count - 1];
    binding->def_line = checked.def_line;
    binding->def_column = checked.def_column;
    binding->def_file = checked.def_file;
    free(binding->struct_type_name);
    binding->struct_type_name = nominal;
    binding->is_resource = checked.is_resource;
    binding->scope_end_line = loop->as.for_stmt.body->scope_end_line;
    binding->scope_end_column = loop->as.for_stmt.body->scope_end_column;
    return true;
}

#endif
