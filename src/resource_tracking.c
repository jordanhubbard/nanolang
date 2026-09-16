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

typedef struct {
    const char *name;
    bool moved;
    int line;
    int column;
} OwnershipPlace;

typedef struct {
    OwnershipPlace *places;
    size_t count;
    size_t capacity;
    size_t required_start;
    bool reachable;
    bool return_resource;
} OwnershipState;

static int ownership_find(const OwnershipState *state, const char *name) {
    for (size_t i = state->count; i > 0; i--)
        if (strcmp(state->places[i - 1].name, name) == 0) return (int)(i - 1);
    return -1;
}

static bool ownership_add(OwnershipState *state, const char *name, int line, int column) {
    if (state->count == state->capacity) {
        size_t capacity = state->capacity ? state->capacity * 2 : 16;
        OwnershipPlace *places = realloc(state->places, capacity * sizeof(*places));
        if (!places) return false;
        state->places = places;
        state->capacity = capacity;
    }
    state->places[state->count++] = (OwnershipPlace){name, false, line, column};
    return true;
}

static bool ownership_clone(OwnershipState *out, const OwnershipState *in) {
    memset(out, 0, sizeof(*out));
    out->reachable = in->reachable;
    out->required_start = in->required_start;
    out->return_resource = in->return_resource;
    if (!in->count) return true;
    out->places = malloc(in->count * sizeof(*out->places));
    if (!out->places) return false;
    memcpy(out->places, in->places, in->count * sizeof(*out->places));
    out->count = out->capacity = in->count;
    return true;
}

static void ownership_error(bool *has_error, int line, int column,
                            const char *kind, const char *name) {
    fprintf(stderr, "Error E0035 at line %d, column %d: I cannot %s resource `%s` after its ownership was moved.\n",
            line, column, kind, name);
    *has_error = true;
}

static void ownership_observe(ASTNode *expr, OwnershipState *state, bool *has_error);

static void ownership_move(ASTNode *expr, OwnershipState *state, bool *has_error) {
    if (!expr) return;
    if (expr->type == AST_IDENTIFIER) {
        int index = ownership_find(state, expr->as.identifier);
        if (index < 0) return;
        if (state->places[index].moved)
            ownership_error(has_error, expr->line, expr->column, "move", expr->as.identifier);
        else
            state->places[index].moved = true;
        return;
    }
    if (expr->type == AST_FIELD_ACCESS) {
        ASTNode *object = expr->as.field_access.object;
        if (object && object->type == AST_IDENTIFIER && ownership_find(state, object->as.identifier) >= 0) {
            fprintf(stderr, "Error E0037 at line %d, column %d: I cannot move a field from live resource `%s`; move or destructure the whole value.\n",
                    expr->line, expr->column, object->as.identifier);
            *has_error = true;
            return;
        }
    }
    ownership_observe(expr, state, has_error);
}

static void ownership_observe(ASTNode *expr, OwnershipState *state, bool *has_error) {
    if (!expr) return;
    switch (expr->type) {
        case AST_IDENTIFIER: {
            int index = ownership_find(state, expr->as.identifier);
            if (index >= 0 && state->places[index].moved)
                ownership_error(has_error, expr->line, expr->column, "use", expr->as.identifier);
            break;
        }
        case AST_CALL:
            if (expr->as.call.func_expr) ownership_observe(expr->as.call.func_expr, state, has_error);
            for (int i = 0; i < expr->as.call.arg_count; i++)
                ownership_move(expr->as.call.args[i], state, has_error);
            break;
        case AST_MODULE_QUALIFIED_CALL:
            for (int i = 0; i < expr->as.module_qualified_call.arg_count; i++)
                ownership_move(expr->as.module_qualified_call.args[i], state, has_error);
            break;
        case AST_FIELD_ACCESS:
            ownership_observe(expr->as.field_access.object, state, has_error);
            break;
        case AST_PREFIX_OP:
            for (int i = 0; i < expr->as.prefix_op.arg_count; i++)
                ownership_observe(expr->as.prefix_op.args[i], state, has_error);
            break;
        case AST_ARRAY_LITERAL:
            for (int i = 0; i < expr->as.array_literal.element_count; i++)
                ownership_observe(expr->as.array_literal.elements[i], state, has_error);
            break;
        case AST_STRUCT_LITERAL:
            if (expr->as.struct_literal.spread_source)
                ownership_move(expr->as.struct_literal.spread_source, state, has_error);
            for (int i = 0; i < expr->as.struct_literal.field_count; i++)
                ownership_move(expr->as.struct_literal.field_values[i], state, has_error);
            break;
        case AST_TUPLE_LITERAL:
            for (int i = 0; i < expr->as.tuple_literal.element_count; i++)
                ownership_move(expr->as.tuple_literal.elements[i], state, has_error);
            break;
        case AST_TUPLE_INDEX:
            ownership_observe(expr->as.tuple_index.tuple, state, has_error);
            break;
        case AST_UNION_CONSTRUCT:
            for (int i = 0; i < expr->as.union_construct.field_count; i++)
                ownership_move(expr->as.union_construct.field_values[i], state, has_error);
            break;
        case AST_TRY_OP:
            ownership_observe(expr->as.try_op.operand, state, has_error);
            break;
        default:
            break;
    }
}

static void ownership_unresolved(const OwnershipState *state, size_t start,
                                 int line, int column, bool *has_error) {
    if (!state->reachable) return;
    for (size_t i = start; i < state->count; i++) {
        if (!state->places[i].moved) {
            fprintf(stderr, "Error E0036 at line %d, column %d: I require resource `%s` to be moved or consumed before scope exit.\n",
                    line, column, state->places[i].name);
            *has_error = true;
        }
    }
}

static void ownership_statement(Environment *env, ASTNode *stmt,
                                OwnershipState *state, bool *has_error) {
    if (!stmt || !state->reachable) return;
    switch (stmt->type) {
        case AST_BLOCK:
            for (int i = 0; i < stmt->as.block.count && state->reachable; i++)
                ownership_statement(env, stmt->as.block.statements[i], state, has_error);
            break;
        case AST_LET:
            if ((stmt->as.let.var_type == TYPE_STRUCT || stmt->as.let.var_type == TYPE_UNION) &&
                is_resource_type(env, stmt->as.let.type_name)) {
                ownership_move(stmt->as.let.value, state, has_error);
                if (
                !ownership_add(state, stmt->as.let.name, stmt->line, stmt->column)) {
                    fprintf(stderr, "I cannot allocate ownership state for `%s`\n", stmt->as.let.name);
                    *has_error = true;
                }
            } else ownership_observe(stmt->as.let.value, state, has_error);
            break;
        case AST_SET: {
            int index = ownership_find(state, stmt->as.set.name);
            if (index >= 0 && !state->places[index].moved) {
                fprintf(stderr, "Error E0038 at line %d, column %d: I cannot overwrite live resource `%s`.\n",
                        stmt->line, stmt->column, stmt->as.set.name);
                *has_error = true;
            }
            ownership_move(stmt->as.set.value, state, has_error);
            if (index >= 0) state->places[index].moved = false;
            break;
        }
        case AST_RETURN:
            if (state->return_resource) ownership_move(stmt->as.return_stmt.value, state, has_error);
            else ownership_observe(stmt->as.return_stmt.value, state, has_error);
            ownership_unresolved(state, state->required_start, stmt->line, stmt->column, has_error);
            state->reachable = false;
            break;
        case AST_IF: {
            ownership_observe(stmt->as.if_stmt.condition, state, has_error);
            OwnershipState then_state, else_state;
            if (!ownership_clone(&then_state, state) || !ownership_clone(&else_state, state)) {
                fprintf(stderr, "I cannot allocate ownership branch state\n");
                *has_error = true;
                free(then_state.places); free(else_state.places);
                break;
            }
            ownership_statement(env, stmt->as.if_stmt.then_branch, &then_state, has_error);
            ownership_statement(env, stmt->as.if_stmt.else_branch, &else_state, has_error);
            if (!then_state.reachable) {
                free(state->places); *state = else_state; memset(&else_state, 0, sizeof(else_state));
            } else if (!else_state.reachable) {
                free(state->places); *state = then_state; memset(&then_state, 0, sizeof(then_state));
            } else {
                size_t common = then_state.count < else_state.count ? then_state.count : else_state.count;
                for (size_t i = 0; i < common; i++) {
                    if (then_state.places[i].moved != else_state.places[i].moved) {
                        fprintf(stderr, "Error E0039 at line %d, column %d: Resource `%s` has incompatible ownership states across branches.\n",
                                stmt->line, stmt->column, then_state.places[i].name);
                        *has_error = true;
                    }
                    state->places[i].moved = then_state.places[i].moved;
                }
            }
            free(then_state.places); free(else_state.places);
            break;
        }
        case AST_WHILE: {
            ownership_observe(stmt->as.while_stmt.condition, state, has_error);
            OwnershipState body;
            if (!ownership_clone(&body, state)) { *has_error = true; break; }
            ownership_statement(env, stmt->as.while_stmt.body, &body, has_error);
            if (body.reachable) {
                size_t common = body.count < state->count ? body.count : state->count;
                for (size_t i = 0; i < common; i++)
                    if (body.places[i].moved != state->places[i].moved) {
                        fprintf(stderr, "Error E0040 at line %d, column %d: Loop changes ownership of outer resource `%s`.\n",
                                stmt->line, stmt->column, state->places[i].name);
                        *has_error = true;
                    }
                ownership_unresolved(&body, state->count, stmt->line, stmt->column, has_error);
            }
            free(body.places);
            break;
        }
        default:
            ownership_observe(stmt, state, has_error);
            break;
    }
}

void check_resource_function(Environment *env, ASTNode *function, bool *has_error) {
    if (!function || function->type != AST_FUNCTION || function->as.function.is_extern ||
        !function->as.function.body) return;
    OwnershipState state = {.reachable = true};
    state.return_resource = (function->as.function.return_type == TYPE_STRUCT ||
                             function->as.function.return_type == TYPE_UNION) &&
                            is_resource_type(env, function->as.function.return_struct_type_name);
    for (int i = 0; i < function->as.function.param_count; i++) {
        Parameter *param = &function->as.function.params[i];
        if ((param->type == TYPE_STRUCT || param->type == TYPE_UNION) &&
            is_resource_type(env, param->struct_type_name) &&
            !ownership_add(&state, param->name, function->line, function->column)) {
            *has_error = true;
            free(state.places);
            return;
        }
    }
    /* By-value parameters arrived through a consuming call boundary. The
     * caller transfer is checked here; terminal foreign/resource APIs remain
     * the boundary that discharges the callee-side obligation. */
    state.required_start = state.count;
    ownership_statement(env, function->as.function.body, &state, has_error);
    ownership_unresolved(&state, state.required_start, function->line, function->column, has_error);
    free(state.places);
}
